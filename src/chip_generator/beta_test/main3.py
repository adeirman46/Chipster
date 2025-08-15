import streamlit as st
import os
import json
import pandas as pd
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Any, Optional
import shutil
from openlane.state import State
from openlane.steps import Step
from openlane.config import Config
from pathlib import Path
import re
import subprocess
import difflib
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Gemini LLM Initialization ---
# Make sure you have GOOGLE_API_KEY in your .env file
try:
    # Using a powerful model suitable for code generation and analysis
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
except Exception as e:
    st.error(f"Error initializing Gemini LLM: {e}. Make sure your GOOGLE_API_KEY is set in a .env file.")
    llm = None


# --- Agentic Workflow State ---
class AgentState(TypedDict):
    uploaded_files: List[Any]
    top_level_module: str
    design_name: str
    verilog_files: List[str]
    original_verilog_code: Dict[str, str]
    modified_verilog_code: Optional[str] # Now a single string
    decomposed_files: Dict[str, str]
    testbench_file: Optional[str]
    original_testbench_code: Optional[str]
    modified_testbench_code: Optional[str]
    config: Dict[str, Any]
    run_path: str
    update_attempt: int
    max_die_width_mm: float
    max_die_height_mm: float
    die_area_mm2: float
    die_width_mm: float
    die_height_mm: float
    simulation_passed: bool
    simulation_output: str
    feedback_log: List[str]
    synthesis_state_out: Optional[State]
    floorplan_state_out: Optional[State]
    tap_endcap_state_out: Optional[State]
    io_placement_state_out: Optional[State]
    pdn_state_out: Optional[State]
    global_placement_state_out: Optional[State]
    detailed_placement_state_out: Optional[State]
    cts_state_out: Optional[State]
    global_routing_state_out: Optional[State]
    detailed_routing_state_out: Optional[State]
    fill_insertion_state_out: Optional[State]
    rcx_state_out: Optional[State]
    sta_state_out: Optional[State]
    stream_out_state_out: Optional[State]
    drc_state_out: Optional[State]
    spice_extraction_state_out: Optional[State]
    lvs_state_out: Optional[State]
    worst_tns: Optional[float]


# --- Agent Definitions ---

def file_processing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📂 Agent 1: File Processing")
    st.info("This agent processes uploaded Verilog files, creates a dedicated run directory, and separates design files from the testbench.")
    uploaded_files = state["uploaded_files"]
    top_level_module = state["top_level_module"]
    design_name = top_level_module

    run_path = os.path.abspath(os.path.join("..", "..", "..", "examples", "generated_chips", f"generated_{design_name}"))
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path, exist_ok=True)

    src_dir = os.path.join(run_path, "src")
    os.makedirs(src_dir, exist_ok=True)

    verilog_files = []
    original_verilog_code = {}
    testbench_file = None
    original_testbench_code = None

    for file in uploaded_files:
        file_path = os.path.join(src_dir, file.name)
        file_content_buffer = file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(file_content_buffer)

        if file.name.endswith((".v", ".vh")):
            decoded_content = file_content_buffer.tobytes().decode('utf-8', errors='ignore')
            if "tb" in file.name.lower():
                 testbench_file = file_path
                 original_testbench_code = decoded_content
            else:
                 verilog_files.append(file_path)
                 original_verilog_code[file.name] = decoded_content

    st.write(f"✅ Top-level module '{top_level_module}' selected.")
    st.write(f"✅ Verilog files saved in: `{src_dir}`")
    if testbench_file:
        st.write(f"✅ Testbench file found: `{os.path.basename(testbench_file)}`")

    os.chdir(run_path)
    st.write(f"✅ Changed working directory to: `{os.getcwd()}`")

    return {
        "design_name": design_name,
        "verilog_files": [os.path.relpath(p, os.getcwd()) for p in verilog_files],
        "original_verilog_code": original_verilog_code,
        "decomposed_files": original_verilog_code,
        "testbench_file": os.path.relpath(testbench_file, os.getcwd()) if testbench_file else None,
        "original_testbench_code": original_testbench_code,
        "run_path": os.getcwd(),
        "feedback_log": ["Starting the design flow."],
        "update_attempt": 0,
    }

def verilog_corrector_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧠 Agent 2: Verilog Corrector (LLM)")
    st.info("This agent uses a Large Language Model (LLM) to analyze and rewrite the Verilog code based on feedback from failed simulation or layout stages.")

    if not llm:
        st.error("Gemini LLM not initialized. Skipping correction.")
        return {"modified_verilog_code": "\n".join(state["original_verilog_code"].values())}

    feedback = "\n".join(state['feedback_log'])
    st.write("#### Feedback for Correction:")
    st.code(feedback, language='text')

    prompt = f"""
    You are an expert Verilog designer. Your task is to optimize the given Verilog code based on the following feedback.
    The primary goal is to simplify the design to reduce its area or fix simulation errors. You may need to create new, simplified modules.

    Feedback:
    {feedback}

    Optimization Strategies:
    1.  **Constant Propagation:** If a complex function is used with constant inputs (e.g., `cos(pi/2)`), replace it with the calculated result (`0`).
    2.  **Module Simplification:** If a module is instantiated but its functionality is not fully required, simplify or replace it.
    3.  **Combine files:** Combine all Verilog modules into a single, monolithic block of code. This simplifies the next steps.
    4.  **Bit-width Reduction:** Carefully reduce the bit-width of registers and wires if the full range is not necessary. Be extremely careful to update all related calculations and instantiations to avoid functional errors.
    5.  **Analyze the code and apply any other relevant simplifications.**

    RULES:
    - You MUST generate pure, synthesizable Verilog-2001 compatible code. Pay close attention to module instantiation syntax.
    - DO NOT use any SystemVerilog features like `logic`, `always_ff`, `always_comb`, or tasks/functions with multiple statements without `begin`/`end` blocks.
    - Combine all Verilog modules into a single, monolithic block of code.
    - Do NOT include the testbench.
    - Your output MUST be only the Verilog code, enclosed in a single markdown block.

    Original Verilog Code:
    ---
    """
    code_to_correct = state.get("decomposed_files") or state["original_verilog_code"]
    for filename, code in code_to_correct.items():
        prompt += f"--- {filename} ---\n{code}\n"

    prompt += "---"

    st.write("🤖 Asking Gemini to optimize the Verilog code...")
    response = llm.invoke(prompt)

    st.write("#### Gemini's Response:")
    st.markdown(response.content)

    modified_code_match = re.search(r"```verilog\n(.*?)```", response.content, re.DOTALL)

    if not modified_code_match:
         st.warning("LLM did not provide a valid Verilog code block. No changes will be made.")
         return {"modified_verilog_code": "\n".join(code_to_correct.values())}

    modified_verilog_code = modified_code_match.group(1).strip()
    return {"modified_verilog_code": modified_verilog_code}

def code_decomposer_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧩 Agent 3: Code Decomposer (LLM-Powered)")
    st.info("After the LLM generates a single block of corrected Verilog, this agent intelligently splits it back into separate files, one for each module.")

    monolithic_code = state.get("modified_verilog_code")
    if not monolithic_code:
        st.error("No modified Verilog code found to decompose.")
        return {"decomposed_files": state.get("decomposed_files", state["original_verilog_code"])}

    st.write("Decomposing LLM-generated code into separate files using Gemini...")

    prompt = f"""
    You are an expert Verilog refactoring tool.
    Your task is to analyze the following monolithic Verilog code and decompose it into multiple files.

    RULES:
    1.  Separate each `module` into its own file. The filename should be the module name with a `.v` extension (e.g., `module_name.v`).
    2.  Return a single, valid JSON object where keys are the filenames and values are the complete code content for that file.
    3.  Your final output **MUST** be only the JSON object, enclosed in a markdown block.

    **MONOLITHIC VERILOG CODE:**
    ```verilog
    {monolithic_code}
    ```

    **RESPONSE (Valid JSON object only):**
    ```json
    """

    response = llm.invoke(prompt)
    st.write("#### Decomposer LLM Response:")
    st.markdown(response.content)

    try:
        # Improved regex to find JSON blocks, even with imperfections
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response.content, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{.*?\})", response.content, re.DOTALL) # Fallback
            if not json_match:
                raise json.JSONDecodeError("No valid JSON object found in the LLM response.", response.content, 0)

        json_str = json_match.group(1)
        decomposed_files = json.loads(json_str)

        if not isinstance(decomposed_files, dict) or not decomposed_files:
            raise ValueError("Parsed JSON is not a valid, non-empty dictionary.")

        st.write("✅ Decomposed code successfully:")
        for filename in decomposed_files.keys():
            st.write(f"  - Created `{filename}`")

    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Failed to parse valid JSON from decomposer. Error: {e}. Falling back to previous version.")
        return {"decomposed_files": state.get("decomposed_files", state["original_verilog_code"])}

    return {"decomposed_files": decomposed_files}


def testbench_corrector_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧠 Agent 4: Testbench Corrector (LLM)")
    st.info("This agent updates the testbench to match any changes made to the design modules (e.g., port name or bit-width changes) by the Verilog Corrector.")

    if not llm or not state.get("original_testbench_code"):
        st.warning("No LLM or original testbench found. Skipping testbench correction.")
        return {}

    tb_to_correct = state.get("modified_testbench_code") or state["original_testbench_code"]

    prompt = f"""
    You are an expert Verilog testbench writer. Your task is to ensure the given testbench is compatible with the provided design modules.
    The design modules might have been changed (e.g., module names, ports, bit widths). Update the testbench accordingly.

    RULES:
    - You MUST generate pure Verilog-2001 compatible code for the testbench.
    - DO NOT use any SystemVerilog features.

    Design Modules:
    ---
    """
    for filename, code in state['decomposed_files'].items():
        prompt += f"--- {filename} ---\n{code}\n"

    prompt += f"""
    ---
    Original Testbench Code (`{os.path.basename(state['testbench_file'])}`):
    ---
    {tb_to_correct}
    ---
    Provide the updated, complete, and corrected testbench code in a single Verilog code block.
    """

    st.write("🤖 Asking Gemini to update the testbench...")
    response = llm.invoke(prompt)

    st.write("#### Gemini's Response:")
    st.markdown(response.content)

    modified_code = re.search(r"```verilog\n(.*?)```", response.content, re.DOTALL)
    if not modified_code:
        st.error("Could not extract corrected testbench code from LLM response.")
        return {"modified_testbench_code": tb_to_correct}

    corrected_tb_code = modified_code.group(1).strip()

    st.write("#### Testbench Changes:")
    diff = difflib.unified_diff(
        tb_to_correct.splitlines(keepends=True),
        corrected_tb_code.splitlines(keepends=True),
        fromfile='original_tb', tofile='modified_tb',
    )
    st.code(''.join(diff), language='diff')

    return {"modified_testbench_code": corrected_tb_code}


def file_saver_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💾 Agent 5: File Saver")
    st.info("This agent saves the newly corrected and decomposed Verilog files to a versioned subdirectory, ensuring a clean state for the next simulation or layout attempt.")

    update_attempt = state.get("update_attempt", 0) + 1

    verilog_to_save = state["decomposed_files"]
    tb_to_save = state.get("modified_testbench_code") or state["original_testbench_code"]

    save_dir_name = f"updated_codes_{update_attempt}"
    save_path = os.path.join(state['run_path'], save_dir_name)
    os.makedirs(save_path, exist_ok=True)
    st.write(f"Saving updated files to: `{save_path}`")

    saved_verilog_files = []

    for filename, content in verilog_to_save.items():
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w') as f: f.write(content)
        saved_verilog_files.append(os.path.relpath(file_path, state['run_path']))
        st.write(f"  - Saved `{filename}`")

    if state.get("testbench_file") and tb_to_save:
        tb_filename = os.path.basename(state["testbench_file"])
        file_path = os.path.join(save_path, tb_filename)
        with open(file_path, 'w') as f: f.write(tb_to_save)
        # Add the testbench to the list of files for simulation
        saved_verilog_files.append(os.path.relpath(file_path, state['run_path']))
        st.write(f"  - Saved `{tb_filename}`")

    return {
        "verilog_files": saved_verilog_files,
        "update_attempt": update_attempt
    }


def icarus_simulation_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 6: Icarus Simulation")
    st.info("This agent compiles and runs a simulation using the Icarus Verilog simulator to functionally verify the design's behavior before attempting the expensive synthesis and layout process.")

    if not state.get('testbench_file'):
        st.warning("No testbench file found. Skipping simulation.")
        return {"simulation_passed": True, "simulation_output": "No testbench provided."}

    run_path = state['run_path']
    # Use the latest saved Verilog files for simulation
    verilog_files_to_sim = [os.path.join(run_path, f) for f in state['verilog_files']]

    # Ensure the source directory is correctly identified from the file paths
    src_dir = os.path.dirname(verilog_files_to_sim[0])
    output_vvp_file = os.path.join(run_path, "design.vvp")

    compile_command = ["iverilog", "-g2005-sv", "-o", output_vvp_file, "-I", src_dir] + verilog_files_to_sim

    try:
        st.write(f"Running compilation: `{' '.join(compile_command)}`")
        compile_process = subprocess.run(compile_command, capture_output=True, text=True, check=True, timeout=30)
        st.write("✅ Compilation successful.")

        sim_command = ["vvp", output_vvp_file]
        st.write(f"Running simulation: `{' '.join(sim_command)}`")
        sim_process = subprocess.run(sim_command, capture_output=True, text=True, check=True, timeout=60)

        st.success("✅ Simulation finished successfully.")
        st.text(sim_process.stdout)
        return {"simulation_passed": True, "simulation_output": sim_process.stdout}

    except subprocess.CalledProcessError as e:
        error_message = f"ERROR during {'compilation' if 'iverilog' in ' '.join(e.cmd) else 'simulation'}:\n{e.stderr or e.stdout}"
        st.error(error_message)
        return {"simulation_passed": False, "simulation_output": error_message}
    except subprocess.TimeoutExpired as e:
        error_message = f"ERROR: {'Compilation' if 'iverilog' in ' '.join(e.cmd) else 'Simulation'} timed out."
        st.error(error_message)
        return {"simulation_passed": False, "simulation_output": error_message}

def setup_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🛠️ Agent 7: OpenLane Setup")
    st.info("This agent initializes the OpenLane 2.0 configuration for the design, setting up the PDK, clock signal, and other core parameters for the physical design flow.")

    config_or_dict = state.get('config')

    if config_or_dict:
        st.write("♻️ Looping back: Using existing (potentially modified) configuration.")
        # On the correction loop from STA, this will be a dict. Otherwise, it's a Config object.
        if isinstance(config_or_dict, dict):
            config = Config(config_or_dict) # Create a new Config object from the modified dict
        else:
            config = config_or_dict # It's already a Config object

        # Clean up previous OpenLane run directories to avoid conflicts and ensure a fresh start
        for item in os.listdir(state['run_path']):
            if item.startswith('runs'):
                shutil.rmtree(os.path.join(state['run_path'], item))
                st.write(f"🧹 Removed old OpenLane run directory: {item}")
    else:
        st.write("🚀 Initial run: Creating new OpenLane configuration.")
        config = Config.interactive(
            state["design_name"], PDK="gf180mcuC",
            CLOCK_PORT="clk", CLOCK_NET="clk", CLOCK_PERIOD=10,
            PRIMARY_GDSII_STREAMOUT_TOOL="klayout",
        )
    st.write("✅ OpenLane configuration loaded.")
    st.info(f"**Clock Period set to: {config['CLOCK_PERIOD']} ns**")
    return {"config": config}


def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 8: Synthesis")
    st.info("This agent converts the high-level Verilog RTL (Register Transfer Level) code into a gate-level netlist, which is a low-level description of the circuit using standard logic gates.")

    # Filter out testbenches from synthesizable files
    synthesizable_files = [f for f in state["verilog_files"] if "_tb" not in f.lower() and "tb." not in f.lower()]
    st.write("Synthesizing the following files:")
    for f in synthesizable_files:
        st.write(f"- `{f}`")

    Synthesis = Step.factory.get("Yosys.Synthesis")
    synthesis_step = Synthesis(config=state["config"], state_in=State(), VERILOG_FILES=synthesizable_files)
    synthesis_step.start()
    report_path = os.path.join(synthesis_step.step_dir, "reports", "stat.json")
    with open(report_path) as f: metrics = json.load(f)
    st.write("#### Synthesis Metrics")
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=["Value"]).astype(str))
    return {"synthesis_state_out": synthesis_step.state_out}


def floorplan_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🏗️ Agent 9: Floorplanning")
    st.info("This agent defines the overall chip dimensions (die area), places the I/O pins on the boundary, and allocates space for the core logic.")
    Floorplan = Step.factory.get("OpenROAD.Floorplan")
    floorplan_step = Floorplan(config=state["config"], state_in=state["synthesis_state_out"])
    floorplan_step.start()
    metrics_path = os.path.join(floorplan_step.step_dir, "or_metrics_out.json")
    with open(metrics_path) as f: metrics = json.load(f)

    st.write("#### Floorplan Metrics")
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).astype(str))

    die_width_um, die_height_um = 0, 0
    bbox_str = metrics.get('design__die__bbox')
    if bbox_str:
        try:
            coords = [float(x) for x in bbox_str.split()]
            if len(coords) == 4:
                llx, lly, urx, ury = coords
                die_width_um, die_height_um = urx - llx, ury - lly
        except (ValueError, IndexError):
            st.warning("Could not parse 'design__die__bbox'.")

    die_width_mm, die_height_mm = die_width_um / 1000, die_height_um / 1000

    st.write("---")
    st.write(f"#### Die Area Analysis")
    st.write(f"Your design is **{die_width_mm:.3f} mm** x **{die_height_mm:.3f} mm**.")
    st.write(f"Maximum allowed size is **{state['max_die_width_mm']:.3f} mm** x **{state['max_die_height_mm']:.3f} mm**.")

    return {
        "floorplan_state_out": floorplan_step.state_out,
        "die_width_mm": die_width_mm, "die_height_mm": die_height_mm
    }

def tap_endcap_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💠 Agent 10: Tap/Endcap Insertion")
    st.info("This agent inserts special cells (tap cells and endcaps) into the floorplan to prevent latch-up issues and ensure proper row termination.")
    TapEndcap = Step.factory.get("OpenROAD.TapEndcapInsertion")
    tap_step = TapEndcap(config=state["config"], state_in=state["floorplan_state_out"])
    tap_step.start()
    return {"tap_endcap_state_out": tap_step.state_out}

def io_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📍 Agent 11: I/O Pin Placement")
    st.info("This agent performs the detailed placement of the input/output (I/O) pads around the periphery of the chip.")
    IOPlacement = Step.factory.get("OpenROAD.IOPlacement")
    ioplace_step = IOPlacement(config=state["config"], state_in=state["tap_endcap_state_out"])
    ioplace_step.start()
    return {"io_placement_state_out": ioplace_step.state_out}

def generate_pdn_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⚡ Agent 12: Power Distribution Network (PDN)")
    st.info("This agent generates the grid of power (Vdd) and ground (GND) stripes that supply electricity to all the cells in the design.")
    GeneratePDN = Step.factory.get("OpenROAD.GeneratePDN")
    pdn_step = GeneratePDN(config=state["config"], state_in=state["io_placement_state_out"])
    pdn_step.start()
    return {"pdn_state_out": pdn_step.state_out}

def global_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌍 Agent 13: Global Placement")
    st.info("This agent determines the approximate locations for all the standard cells in the core area, aiming to minimize wire length and congestion.")
    GlobalPlacement = Step.factory.get("OpenROAD.GlobalPlacement")
    gpl_step = GlobalPlacement(config=state["config"], state_in=state["pdn_state_out"])
    gpl_step.start()
    return {"global_placement_state_out": gpl_step.state_out}

def detailed_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📐 Agent 14: Detailed Placement")
    st.info("This agent refines the placement from the previous step, legalizing all cell positions to snap them onto the site grid and removing any overlaps.")
    DetailedPlacement = Step.factory.get("OpenROAD.DetailedPlacement")
    dpl_step = DetailedPlacement(config=state["config"], state_in=state["global_placement_state_out"])
    dpl_step.start()
    return {"detailed_placement_state_out": dpl_step.state_out}

def cts_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌳 Agent 15: Clock Tree Synthesis (CTS)")
    st.info("This agent builds the clock tree, a network of buffers that distributes the clock signal to all sequential elements (flip-flops) with minimal skew.")
    CTS = Step.factory.get("OpenROAD.CTS")
    cts_step = CTS(config=state["config"], state_in=state["detailed_placement_state_out"])
    cts_step.start()
    return {"cts_state_out": cts_step.state_out}

def global_routing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🗺️ Agent 16: Global Routing")
    st.info("This agent plans the general paths for all the signal nets on the routing grid, breaking down long wires and avoiding congested areas.")
    GlobalRouting = Step.factory.get("OpenROAD.GlobalRouting")
    grt_step = GlobalRouting(config=state["config"], state_in=state["cts_state_out"])
    grt_step.start()
    return {"global_routing_state_out": grt_step.state_out}

def detailed_routing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ✍️ Agent 17: Detailed Routing")
    st.info("This agent performs the final, exact routing of all wires, connecting the cell pins according to the netlist and global routing plan.")
    DetailedRouting = Step.factory.get("OpenROAD.DetailedRouting")
    drt_step = DetailedRouting(config=state["config"], state_in=state["global_routing_state_out"])
    drt_step.start()
    return {"detailed_routing_state_out": drt_step.state_out}

def fill_insertion_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧱 Agent 18: Fill Insertion")
    st.info("This agent adds non-functional 'filler' cells to empty spaces in the layout to ensure metal density uniformity, which is required for manufacturing.")
    FillInsertion = Step.factory.get("OpenROAD.FillInsertion")
    fill_step = FillInsertion(config=state["config"], state_in=state["detailed_routing_state_out"])
    fill_step.start()
    return {"fill_insertion_state_out": fill_step.state_out}

def rcx_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔌 Agent 19: Parasitics Extraction (RCX)")
    st.info("This agent extracts the parasitic resistance (R) and capacitance (C) of the routed wires. This information is crucial for accurate timing analysis.")
    RCX = Step.factory.get("OpenROAD.RCX")
    rcx_step = RCX(config=state["config"], state_in=state["fill_insertion_state_out"])
    rcx_step.start()
    return {"rcx_state_out": rcx_step.state_out}

def sta_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⏱️ Agent 20: Static Timing Analysis (STA)")
    st.info("This final analysis step uses the extracted parasitics to verify that the chip meets its timing constraints (i.e., that signals arrive on time).")
    STAPostPNR = Step.factory.get("OpenROAD.STAPostPNR")
    sta_step = STAPostPNR(config=state["config"], state_in=state["rcx_state_out"])
    sta_step.start()
    st.write("#### STA Timing Violation Summary")
    sta_results = []
    value_re = re.compile(r":\s*(-?[\d\.]+)")
    # --- FIX: Expanded the list to find all four requested reports ---
    reports_to_find = ["tns.max.rpt", "tns.min.rpt", "wns.max.rpt", "wns.min.rpt"]
    all_tns = []

    for root, _, files in os.walk(sta_step.step_dir):
        for file in files:
            if file in reports_to_find:
                corner = os.path.basename(root)
                metric = file.replace(".rpt", "").replace(".", " ").title()
                with open(os.path.join(root, file)) as f:
                    content = f.read()
                    match = value_re.search(content)
                    if match:
                        value = float(match.group(1))
                        sta_results.append([corner, metric, value])
                        if "Tns Max" in metric:
                           all_tns.append(value)

    worst_tns = min(all_tns) if all_tns else 0
    st.info(f"**Worst Total Negative Slack (TNS) across all corners: {worst_tns:.2f} ps**")

    if sta_results:
        df_sta = pd.DataFrame(sta_results, columns=["Corner", "Metric", "Value (ps)"])
        pivoted_df = df_sta.pivot(index='Metric', columns='Corner', values='Value (ps)').fillna(0)
        def style_violations(val):
             try:
                 color = 'green' if float(val) >= 0 else 'red'
                 return f'color: {color}'
             except (ValueError, TypeError): return ''
        styled_df = pivoted_df.style.applymap(style_violations).format("{:.2f}")
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("Could not parse key STA report files (TNS, WNS).")

    return {"sta_state_out": sta_step.state_out, "worst_tns": worst_tns / 1000.0}

def sta_correction_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🤖 Agent 21: STA Corrector")
    st.info("If timing violations are found, this agent attempts to fix them by increasing the clock period (i.e., slowing down the chip's target frequency).")
    st.error("❌ Timing violations detected! Attempting to fix by adjusting clock period.")

    current_config = state["config"]
    # Convert the immutable Config object to a mutable dictionary to allow modification.
    config_dict = dict(current_config)
    current_period = float(config_dict["CLOCK_PERIOD"])
    worst_tns_ns = state["worst_tns"]

    abs_tns = abs(worst_tns_ns)
    if abs_tns > 500:
        new_period = current_period * 10
        st.warning(f"CRITICAL violation (TNS = {worst_tns_ns:.2f} ns). Drastically increasing clock period 10x.")
    elif abs_tns > 50:
        new_period = current_period * 2
        st.warning(f"HIGH violation (TNS = {worst_tns_ns:.2f} ns). Increasing clock period 2x.")
    else:
        new_period = current_period * 1.5
        st.warning(f"Small violation (TNS = {worst_tns_ns:.2f} ns). Increasing clock period 1.5x.")

    st.write(f"Old Clock Period: {current_period:.2f} ns")
    st.success(f"**New Clock Period: {new_period:.2f} ns**")

    # Update the clock period in our mutable dictionary.
    config_dict["CLOCK_PERIOD"] = new_period
    
    # Create a new, updated Config object from the modified dictionary.
    new_config = Config(config_dict)

    feedback = state.get("feedback_log", []) + [f"STA failed with TNS={worst_tns_ns:.2f}ns. Increased clock period from {current_period}ns to {new_period}ns and re-running PnR."]
    
    # Pass the new Config object back to the workflow.
    return {"config": new_config, "feedback_log": feedback}


def stream_out_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💾 Agent 22: GDSII Stream Out")
    st.info("This agent generates the final GDSII file, a standard file format used by semiconductor foundries to manufacture the chip.")
    StreamOut = Step.factory.get("KLayout.StreamOut")
    gds_step = StreamOut(config=state["config"], state_in=state["sta_state_out"])
    gds_step.start()
    return {"stream_out_state_out": gds_step.state_out}

def drc_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ✅ Agent 23: Design Rule Check (DRC)")
    st.info("This agent checks if the final layout adheres to the geometric and electrical rules defined by the foundry (the 'PDK'). This is a critical manufacturing prerequisite.")
    DRC = Step.factory.get("Magic.DRC")
    drc_step = DRC(config=state["config"], state_in=state["stream_out_state_out"])
    drc_step.start()
    report_path = os.path.join(drc_step.step_dir, "reports", "drc_violations.magic.rpt")
    try:
        with open(report_path) as f:
            content = f.read()
            count_match = re.search(r"\[INFO\] COUNT: (\d+)", content)
            if count_match:
                count = int(count_match.group(1))
                if count == 0: st.success("✅ No DRC violations found.")
                else: st.error(f"❌ Found {count} DRC violations.")
                st.text(content)
    except FileNotFoundError: st.warning("DRC report file not found.")
    return {"drc_state_out": drc_step.state_out}

def spice_extraction_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⚡ Agent 24: SPICE Extraction")
    st.info("This agent extracts a detailed SPICE netlist from the final layout. This netlist includes all parasitic effects and represents the 'as-built' circuit.")
    SpiceExtraction = Step.factory.get("Magic.SpiceExtraction")
    spx_step = SpiceExtraction(config=state["config"], state_in=state["drc_state_out"])
    spx_step.start()
    return {"spice_extraction_state_out": spx_step.state_out}

def lvs_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ↔️ Agent 25: Layout vs. Schematic (LVS)")
    st.info("This final verification step compares the SPICE netlist extracted from the layout against the original gate-level netlist from synthesis to ensure they are electrically identical.")
    LVS = Step.factory.get("Netgen.LVS")
    lvs_step = LVS(config=state["config"], state_in=state["spice_extraction_state_out"])
    lvs_step.start()
    report_path = os.path.join(lvs_step.step_dir, "reports", "lvs.netgen.rpt")
    try:
        with open(report_path) as f:
            content = f.read()
            final_result_match = re.search(r"Final result:\s*(.*)", content)
            if final_result_match:
                result = final_result_match.group(1).strip()
                if "Circuits match uniquely" in result: st.success(f"✅ **LVS Passed:** {result}")
                else: st.error(f"❌ **LVS Failed:** {result}")
    except FileNotFoundError: st.warning("LVS report file not found.")
    return {"lvs_state_out": lvs_step.state_out}

def render_step_image(state: AgentState, state_key_in: str, caption: str):
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.write(f"#### 🖼️ Visualizing: {caption}")
        st.info("This image shows the physical layout of the design at the current stage.")
    with col2:
      Render = Step.factory.get("KLayout.Render")
      render_step = Render(config=state["config"], state_in=state[state_key_in])
      render_step.start()
      image_path = os.path.join(render_step.step_dir, "out.png")
      if os.path.exists(image_path):
          st.image(image_path, caption=caption, use_column_width='auto')
      else:
          st.warning("Could not render image.")
    return {}

# --- Conditional Logic ---

def check_simulation(state: AgentState) -> str:
    if state["simulation_passed"]:
        st.success("✅ Simulation Passed. Proceeding to OpenLane flow.")
        return "continue_to_synthesis"
    else:
        st.error("❌ Simulation Failed.")
        feedback = state.get("feedback_log", []) + [f"Icarus simulation failed. Please examine the following error and fix the Verilog code:\n{state['simulation_output']}"]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 2: # Limit attempts
            st.error("Simulation failed after multiple correction attempts. Halting.")
            return "end"
        st.warning("Looping back to Verilog Corrector for another attempt.")
        return "fix_verilog"

def check_floorplan(state: AgentState) -> str:
    width_ok = state['die_width_mm'] <= state['max_die_width_mm']
    height_ok = state['die_height_mm'] <= state['max_die_height_mm']

    if width_ok and height_ok:
        st.success("✅ Die size is within limits. Proceeding with Place and Route.")
        return "continue_to_pnr"
    else:
        st.error("❌ Die size exceeds maximum limits.")
        feedback = state.get("feedback_log", []) + [f"Floorplan failed. Die size {state['die_width_mm']:.3f}x{state['die_height_mm']:.3f}mm exceeds limit of {state['max_die_width_mm']:.3f}x{state['max_die_height_mm']:.3f}mm. Please simplify the design to reduce its area."]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 2: # Limit attempts
            st.error("Die size too large after multiple correction attempts. Halting.")
            return "end"
        return "fix_verilog"

def check_sta_violations(state: AgentState) -> str:
    worst_tns = state.get("worst_tns", 0.0)
    if worst_tns < 0:
        st.error(f"❌ STA VIOLATION DETECTED (TNS={worst_tns:.2f} ns).")
        if state.get("update_attempt", 0) > 5: # Limit STA loops
             st.error("Could not meet timing after multiple attempts. Halting.")
             return "end"
        return "fix_sta"
    else:
        st.success(f"✅ Timing constraints met (TNS={worst_tns:.2f} ns). Proceeding to final signoff.")
        return "continue_to_signoff"


# --- Build the graph ---
workflow = StateGraph(AgentState)

# Add Nodes
node_definitions = {
    "file_processing": file_processing_agent, "verilog_corrector": verilog_corrector_agent,
    "code_decomposer": code_decomposer_agent, "testbench_corrector": testbench_corrector_agent,
    "file_saver": file_saver_agent, "icarus_simulation": icarus_simulation_agent,
    "setup": setup_agent, "synthesis": synthesis_agent, "floorplan": floorplan_agent,
    "tap_endcap": tap_endcap_agent, "io_placement": io_placement_agent,
    "generate_pdn": generate_pdn_agent, "global_placement": global_placement_agent,
    "detailed_placement": detailed_placement_agent, "cts": cts_agent,
    "global_routing": global_routing_agent, "detailed_routing": detailed_routing_agent,
    "fill_insertion": fill_insertion_agent, "rcx": rcx_agent, "sta": sta_agent,
    "sta_correction": sta_correction_agent, "stream_out": stream_out_agent,
    "drc": drc_agent, "spice_extraction": spice_extraction_agent, "lvs": lvs_agent,
    "render_floorplan": lambda s: render_step_image(s, "floorplan_state_out", "Floorplan Layout"),
    "render_routing": lambda s: render_step_image(s, "detailed_routing_state_out", "Post-Routing Layout"),
    "render_gds": lambda s: render_step_image(s, "stream_out_state_out", "Final GDSII Layout")
}
for name, func in node_definitions.items():
    workflow.add_node(name, func)

# Define Edges
workflow.add_edge(START, "file_processing")
workflow.add_edge("file_processing", "icarus_simulation")

# Conditional Edge 1: Simulation Check
workflow.add_conditional_edges(
    "icarus_simulation", check_simulation,
    {"continue_to_synthesis": "setup", "fix_verilog": "verilog_corrector", "end": END}
)

# Correction Loop 1: Verilog/LLM fix
workflow.add_edge("verilog_corrector", "code_decomposer")
workflow.add_edge("code_decomposer", "testbench_corrector")
workflow.add_edge("testbench_corrector", "file_saver")
workflow.add_edge("file_saver", "icarus_simulation") # Loop back to re-verify

# Main Flow Path
workflow.add_edge("setup", "synthesis")
workflow.add_edge("synthesis", "floorplan")
workflow.add_edge("floorplan", "render_floorplan")

# Conditional Edge 2: Floorplan/Area Check
workflow.add_conditional_edges(
    "render_floorplan", check_floorplan,
    {"continue_to_pnr": "tap_endcap", "fix_verilog": "verilog_corrector", "end": END}
)

# PNR Chain
pnr_chain = [
    "tap_endcap", "io_placement", "generate_pdn", "global_placement", "detailed_placement",
    "cts", "global_routing", "detailed_routing", "render_routing"
]
for i in range(len(pnr_chain) - 1):
    workflow.add_edge(pnr_chain[i], pnr_chain[i+1])

# Post-PNR -> STA
workflow.add_edge("render_routing", "fill_insertion")
workflow.add_edge("fill_insertion", "rcx")
workflow.add_edge("rcx", "sta")

# Conditional Edge 3: STA/Timing Check
workflow.add_conditional_edges(
    "sta", check_sta_violations,
    {"continue_to_signoff": "stream_out", "fix_sta": "sta_correction", "end": END}
)

# Correction Loop 2: STA/Timing fix
workflow.add_edge("sta_correction", "setup") # Loop back to setup with new config

# Final Signoff Chain
signoff_chain = ["stream_out", "render_gds", "drc", "spice_extraction", "lvs"]
for i in range(len(signoff_chain) - 1):
    workflow.add_edge(signoff_chain[i], signoff_chain[i+1])

workflow.add_edge("lvs", END)

app = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 LLM for Chip Design Automation")
st.write("This application uses a multi-agent workflow to automate the digital chip design flow, from RTL to GDSII. It includes intelligent feedback loops to correct functional, area, and timing violations.")

st.sidebar.header("1. Upload Your Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload Verilog design (.v) and a single testbench (tb.v) file", accept_multiple_files=True
)

if uploaded_files:
    verilog_file_names = [f.name for f in uploaded_files if f.name.endswith((".v", ".vh")) and "tb" not in f.name.lower()]
    if not verilog_file_names:
        st.sidebar.warning("Please upload at least one Verilog design file (not a testbench).")
    else:
        top_level_module_options = [Path(name).stem for name in verilog_file_names]
        top_level_module = st.sidebar.selectbox("Select the top-level module", options=top_level_module_options)

        st.sidebar.header("2. Set Constraints")
        max_w = st.sidebar.number_input("Max Die Width (mm)", min_value=0.01, value=0.8, step=0.01, format="%.3f")
        max_h = st.sidebar.number_input("Max Die Height (mm)", min_value=0.01, value=0.8, step=0.01, format="%.3f")

        if st.sidebar.button("🚀 Run Agentic Flow"):
            if not llm:
                st.error("Cannot run flow: Gemini LLM is not initialized.")
            else:
                original_cwd = os.getcwd()
                try:
                    with st.spinner("🚀 Agents at work... This will take several minutes."):
                        initial_state = {
                            "uploaded_files": uploaded_files,
                            "top_level_module": top_level_module,
                            "max_die_width_mm": max_w,
                            "max_die_height_mm": max_h,
                        }
                        # The recursion limit is increased to handle the potential loops in the graph.
                        app.invoke(initial_state, {"recursion_limit": 150})
                    st.success("✅ Agentic flow completed!")
                except Exception as e:
                    st.error(f"An error occurred during the flow: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    os.chdir(original_cwd)
                    st.write(f"✅ Restored working directory to: `{os.getcwd()}`")

# --- Detailed Workflow Graph ---
st.write("### Agentic Workflow Graph")
st.graphviz_chart("""
digraph G {
    graph [fontname="sans-serif", label="Digital Design Flow with Intelligent Correction Loops", labelloc=t, fontsize=20, rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.8];
    node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, width=2.2, height=0.5];
    edge [fontname="sans-serif", fontsize=8];

    subgraph cluster_prep {
        label="1. Pre-Processing & Verification";
        style="rounded,filled";
        color="#e3f2fd";
        node[fillcolor="#bbdefb"];
        file_processing [label="1. File Processing"];
        icarus_simulation [label="6. Icarus Simulation", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }

    subgraph cluster_correction_verilog {
        label="A. Verilog Correction Loop";
        style="rounded,filled";
        color="#ffebee";
        node[fillcolor="#ffcdd2"];
        verilog_corrector [label="2. Verilog Corrector (LLM)"];
        code_decomposer [label="3. Code Decomposer (LLM)"];
        testbench_corrector [label="4. Testbench Corrector (LLM)"];
        file_saver [label="5. File Saver"];
    }

    subgraph cluster_pnr {
        label="2. Physical Design (PnR)";
        style="rounded,filled";
        color="#e8f5e9";
        node[fillcolor="#c8e6c9"];
        setup [label="7. OpenLane Setup"];
        synthesis [label="8. Synthesis"];
        floorplan [label="9. Floorplan", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
        render_floorplan [label="Render Floorplan", shape=note, fillcolor="#fafafa"];
        pnr_group [label="PnR Steps (10-19)", shape=box3d, style=filled, fillcolor="#a5d6a7"];
        sta [label="20. Static Timing Analysis", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }
    
    subgraph cluster_correction_sta {
        label="B. Timing Correction Loop";
        style="rounded,filled";
        color="#fff3e0";
        node[fillcolor="#ffe0b2"];
        sta_correction [label="21. STA Corrector"];
    }

    subgraph cluster_signoff {
        label="3. Final Signoff";
        style="rounded,filled";
        color="#f3e5f5";
        node[fillcolor="#e1bee7", width=1.8, height=0.4];
        stream_out [label="22. GDSII Stream Out"];
        render_gds [label="Render GDS", shape=note, fillcolor="#fafafa"];
        drc [label="23. DRC"];
        spice_extraction [label="24. SPICE Extraction"];
        lvs [label="25. LVS"];
    }
    
    end_node [label="Flow Complete", shape=ellipse, style=filled, fillcolor="#b2dfdb"];

    // Main Flow Edges
    file_processing -> icarus_simulation;
    icarus_simulation -> setup [label="Sim OK", color=darkgreen, fontcolor=darkgreen, style=bold];
    setup -> synthesis -> floorplan -> render_floorplan;
    render_floorplan -> pnr_group [label="Area OK", color=darkgreen, fontcolor=darkgreen, style=bold];
    pnr_group -> sta;
    sta -> stream_out [label="Timing OK", color=darkgreen, fontcolor=darkgreen, style=bold];
    stream_out -> render_gds -> drc -> spice_extraction -> lvs -> end_node;

    // Correction Loop 1: Verilog Fix (Red)
    icarus_simulation -> verilog_corrector [label=" Sim FAIL", style=dashed, color=red, constraint=false, fontcolor=red];
    render_floorplan -> verilog_corrector [label=" Area TOO BIG", style=dashed, color=red, constraint=false, fontcolor=red];
    verilog_corrector -> code_decomposer -> testbench_corrector -> file_saver -> icarus_simulation [style=dashed, color=red, arrowhead=normal, label="   Re-verify"];

    // Correction Loop 2: STA Fix (Blue)
    sta -> sta_correction [label=" TNS < 0", style=dashed, color=blue, constraint=false, fontcolor=blue];
    sta_correction -> setup [style=dashed, color=blue, arrowhead=normal, label=" Re-run PnR w/ new clock"];
}
""")
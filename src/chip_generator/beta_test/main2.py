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
    # Corrected the model name to a valid identifier
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
    worst_wns: Optional[float]


# --- Agent Definitions ---

def file_processing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📂 Agent 1: File Processing")
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
        "decomposed_files": original_verilog_code, # Initially, decomposed is the same as original
        "testbench_file": os.path.relpath(testbench_file, os.getcwd()) if testbench_file else None,
        "original_testbench_code": original_testbench_code,
        "run_path": os.getcwd(),
        "feedback_log": ["Starting the design flow."],
        "update_attempt": 0,
    }

def verilog_corrector_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧠 Agent 2: Verilog Corrector (Gemini)")

    if not llm:
        st.error("Gemini LLM not initialized. Skipping correction.")
        return {"modified_verilog_code": None}

    feedback = "\n".join(state['feedback_log'])
    st.write("#### Feedback for Correction:")
    st.code(feedback, language='text')

    prompt = f"""
    You are an expert Verilog designer. Your task is to optimize the given Verilog code based on the following feedback.
    The primary goal is to simplify the design to reduce its area. You may need to create new, simplified modules.

    Feedback:
    {feedback}

    Optimization Strategies:
    1.  **Constant Propagation:** If a complex function is used with constant inputs (e.g., `cos(pi/2)`), replace it with the calculated result (`0`).
    2.  **Module Simplification:** If a module is instantiated but its functionality is not fully required, simplify or replace it.
    3. **Combine files:** If multiple files contain similar or redundant code, combine them into a single file to reduce complexity.
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
    st.write("### 🧩 Agent: Code Decomposer (LLM-Powered)")

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
        json_match = re.search(r"```json\n(\{.*?\})\n```", response.content, re.DOTALL)
        if not json_match:
            # Fallback for cases where the LLM might not use the json markdown tag
            json_match = re.search(r"(\{.*?\})", response.content, re.DOTALL)
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
        # Fallback to the last known good decomposition
        return {"decomposed_files": state.get("decomposed_files", state["original_verilog_code"])}

    return {"decomposed_files": decomposed_files}


def testbench_corrector_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧠 Agent 3: Testbench Corrector (Gemini)")

    if not llm:
        st.error("Gemini LLM not initialized. Skipping correction.")
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
    st.write("### 💾 Agent: File Saver")

    update_attempt = state.get("update_attempt", 0) + 1

    verilog_to_save = state["decomposed_files"]
    tb_to_save = state.get("modified_testbench_code") or state["original_testbench_code"]

    save_dir_name = f"updated_codes_{update_attempt}"
    save_path = os.path.join(state['run_path'], save_dir_name)
    os.makedirs(save_path, exist_ok=True)
    st.write(f"Saving updated files to: `{save_path}`")

    saved_verilog_files = []

    # Save Verilog modules
    for filename, content in verilog_to_save.items():
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w') as f: f.write(content)
        saved_verilog_files.append(os.path.relpath(file_path, state['run_path']))
        st.write(f"  - Saved `{filename}`")

    # Save Testbench
    if state.get("testbench_file") and tb_to_save:
        tb_filename = os.path.basename(state["testbench_file"])
        file_path = os.path.join(save_path, tb_filename)
        with open(file_path, 'w') as f: f.write(tb_to_save)
        # Add testbench to the list of files for simulation
        saved_verilog_files.append(os.path.relpath(file_path, state['run_path']))
        st.write(f"  - Saved `{tb_filename}`")

    return {
        "verilog_files": saved_verilog_files,
        "update_attempt": update_attempt
    }


def icarus_simulation_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 4: Icarus Simulation")

    if not state.get('testbench_file'):
        st.warning("No testbench file found. Skipping simulation.")
        return {"simulation_passed": True}

    run_path = state['run_path']
    verilog_files_to_sim = [os.path.join(run_path, f) for f in state['verilog_files']]

    src_dir = os.path.dirname(verilog_files_to_sim[0])
    output_vvp_file = os.path.join(run_path, "design.vvp")

    # Added -g2005-sv flag to support some SystemVerilog features while discouraging their generation via prompts
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
    st.write("### 🛠️ Agent 5: OpenLane Setup")

    # If we are re-running setup, use the modified config, otherwise create a new one.
    if state.get('config') and state.get('update_attempt', 0) > 0:
        st.write("Re-running setup with modified configuration.")
        config = state['config']
    else:
        st.write("Creating initial OpenLane configuration.")
        for item in os.listdir(state['run_path']):
            if item.startswith('runs'):
                shutil.rmtree(os.path.join(state['run_path'], item))
                st.write(f"🧹 Removed old OpenLane run directory: {item}")

        config = Config.interactive(
            state["design_name"], PDK="gf180mcuC",
            CLOCK_PORT="clk", CLOCK_NET="clk", CLOCK_PERIOD=10, # Start with 10ns = 100MHz
            PRIMARY_GDSII_STREAMOUT_TOOL="klayout",
        )
    st.write("✅ OpenLane configuration created successfully.")
    st.write(f"**Clock Period set to: {config['CLOCK_PERIOD']} ns**")
    return {"config": config}


def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 6: Synthesis")
    st.write("""Converting high-level Verilog to a netlist of standard cells.""")

    # Filter out testbench files before synthesis
    synthesizable_files = [f for f in state["verilog_files"] if "_tb" not in f.lower()]
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
    st.write("### 🏗️ Agent 7: Floorplanning")
    st.write("""Determining the chip's dimensions and creating the cell placement grid.""")
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
    else:
        st.warning("'design__die__bbox' not found.")

    die_area_mm2 = metrics.get('design__die__area_um^2', 0) / 1_000_000
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
    st.write("### 💠 Agent 8: Tap/Endcap Insertion")
    st.write("""Placing tap and endcap cells for power stability.""")
    TapEndcap = Step.factory.get("OpenROAD.TapEndcapInsertion")
    tap_step = TapEndcap(config=state["config"], state_in=state["floorplan_state_out"])
    tap_step.start()
    return {"tap_endcap_state_out": tap_step.state_out}

def io_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📍 Agent 9: I/O Pin Placement")
    st.write("Placing I/O pins at the edges of the design.")
    IOPlacement = Step.factory.get("OpenROAD.IOPlacement")
    ioplace_step = IOPlacement(config=state["config"], state_in=state["tap_endcap_state_out"])
    ioplace_step.start()
    return {"io_placement_state_out": ioplace_step.state_out}

def generate_pdn_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⚡ Agent 10: Power Distribution Network (PDN) Generation")
    st.write("""Creating the metal grid for power and ground.""")
    GeneratePDN = Step.factory.get("OpenROAD.GeneratePDN")
    pdn_step = GeneratePDN(config=state["config"], state_in=state["io_placement_state_out"], FP_PDN_VWIDTH=2, FP_PDN_HWIDTH=2, FP_PDN_VPITCH=30, FP_PDN_HPITCH=30)
    pdn_step.start()
    return {"pdn_state_out": pdn_step.state_out}

def global_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌍 Agent 11: Global Placement")
    st.write("""Finding an approximate location for all standard cells.""")
    GlobalPlacement = Step.factory.get("OpenROAD.GlobalPlacement")
    gpl_step = GlobalPlacement(config=state["config"], state_in=state["pdn_state_out"])
    gpl_step.start()
    return {"global_placement_state_out": gpl_step.state_out}

def detailed_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📐 Agent 12: Detailed Placement")
    st.write("""Snapping cells to the legal manufacturing grid.""")
    DetailedPlacement = Step.factory.get("OpenROAD.DetailedPlacement")
    dpl_step = DetailedPlacement(config=state["config"], state_in=state["global_placement_state_out"])
    dpl_step.start()
    return {"detailed_placement_state_out": dpl_step.state_out}

def cts_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌳 Agent 13: Clock Tree Synthesis (CTS)")
    st.write("""Building the clock distribution network.""")
    CTS = Step.factory.get("OpenROAD.CTS")
    cts_step = CTS(config=state["config"], state_in=state["detailed_placement_state_out"])
    cts_step.start()
    return {"cts_state_out": cts_step.state_out}

def global_routing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🗺️ Agent 14: Global Routing")
    st.write("""Planning the paths for the interconnect wires.""")
    GlobalRouting = Step.factory.get("OpenROAD.GlobalRouting")
    grt_step = GlobalRouting(config=state["config"], state_in=state["cts_state_out"])
    grt_step.start()
    metrics_path = os.path.join(grt_step.step_dir, "or_metrics_out.json")
    with open(metrics_path) as f: metrics = json.load(f)
    st.write("#### Global Routing Metrics")
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).astype(str))
    return {"global_routing_state_out": grt_step.state_out}

def detailed_routing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ✍️ Agent 15: Detailed Routing")
    st.write("""Creating the final physical wires on the metal layers.""")
    DetailedRouting = Step.factory.get("OpenROAD.DetailedRouting")
    drt_step = DetailedRouting(config=state["config"], state_in=state["global_routing_state_out"])
    drt_step.start()
    return {"detailed_routing_state_out": drt_step.state_out}

def fill_insertion_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧱 Agent 16: Fill Insertion")
    st.write("""Filling empty gaps in the design with 'fill cells' for manufacturability.""")
    FillInsertion = Step.factory.get("OpenROAD.FillInsertion")
    fill_step = FillInsertion(config=state["config"], state_in=state["detailed_routing_state_out"])
    fill_step.start()
    return {"fill_insertion_state_out": fill_step.state_out}

def rcx_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔌 Agent 17: Parasitics Extraction (RCX)")
    st.write("""This step computes the parasitic resistance and capacitance of the wires, which affect timing.""")
    RCX = Step.factory.get("OpenROAD.RCX")
    rcx_step = RCX(config=state["config"], state_in=state["fill_insertion_state_out"])
    rcx_step.start()
    metrics_path = os.path.join(rcx_step.step_dir, "or_metrics_out.json")
    with open(metrics_path) as f: metrics = json.load(f)
    st.write("#### Parasitics Extraction Metrics")
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).astype(str))
    return {"rcx_state_out": rcx_step.state_out}

def sta_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⏱️ Agent 18: Static Timing Analysis (STA)")
    st.write("""This final analysis step verifies that the chip meets its timing constraints to run at the rated clock speed.""")
    STAPostPNR = Step.factory.get("OpenROAD.STAPostPNR")
    sta_step = STAPostPNR(config=state["config"], state_in=state["rcx_state_out"])
    sta_step.start()
    st.write("#### STA Timing Violation Summary")
    sta_results = []
    value_re = re.compile(r":\s*(-?[\d\.]+)")
    reports_to_find = ["tns.max.rpt", "tns.min.rpt", "wns.max.rpt", "wns.min.rpt"]
    all_wns = []

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
                        if "Wns Max" in metric:
                           all_wns.append(value)

    worst_wns = min(all_wns) if all_wns else 0
    st.write(f"**Worst Negative Slack (WNS) across all corners: {worst_wns:.2f} ps**")

    if sta_results:
        df_sta = pd.DataFrame(sta_results, columns=["Corner", "Metric", "Value (ps)"])
        pivoted_df = df_sta.pivot(index='Metric', columns='Corner', values='Value (ps)')
        def style_violations(val):
            try:
                color = 'green' if float(val) >= 0 else 'red'
                return f'color: {color}'
            except (ValueError, TypeError): return ''
        styled_df = pivoted_df.style.applymap(style_violations).format("{:.2f}")
        st.dataframe(styled_df)
    else:
        st.warning("Could not parse key STA report files (TNS, WNS).")

    return {"sta_state_out": sta_step.state_out, "worst_wns": worst_wns / 1000.0} # Convert ps to ns


# NEW AGENT
def sta_correction_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🤖 Agent: STA Corrector")
    st.error("❌ Timing violations detected! Attempting to fix by adjusting clock period.")

    current_config = state["config"]
    current_period = float(current_config["CLOCK_PERIOD"])
    worst_wns_ns = state["worst_wns"]

    # Decide on the new clock period
    if abs(worst_wns_ns) > current_period * 0.5:
        # High violation: increase period dramatically
        new_period = current_period * 10
        st.warning(f"HIGH violation detected (WNS = {worst_wns_ns:.2f} ns). Increasing clock period 10x.")
    else:
        # Smaller violation: increase by a smaller factor
        new_period = current_period * 2
        st.warning(f"Small violation detected (WNS = {worst_wns_ns:.2f} ns). Increasing clock period 2x.")

    st.write(f"Old Clock Period: {current_period:.2f} ns")
    st.write(f"**New Clock Period: {new_period:.2f} ns**")

    # Create a mutable copy of the configuration to update it
    new_config = dict(current_config)
    new_config["CLOCK_PERIOD"] = new_period

    feedback = state.get("feedback_log", []) + [f"STA failed with WNS={worst_wns_ns:.2f}ns. Increased clock period from {current_period}ns to {new_period}ns and re-running PnR."]

    return {"config": new_config, "feedback_log": feedback}


def stream_out_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💾 Agent 19: GDSII Stream Out")
    st.write("This step converts the final layout into GDSII format, the file that is sent to the foundry for fabrication.")
    StreamOut = Step.factory.get("KLayout.StreamOut")
    gds_step = StreamOut(config=state["config"], state_in=state["sta_state_out"])
    gds_step.start()
    return {"stream_out_state_out": gds_step.state_out}

def drc_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ✅ Agent 20: Design Rule Check (DRC)")
    st.write("Checks if the final layout violates any of the foundry's manufacturing rules.")
    DRC = Step.factory.get("Magic.DRC")
    drc_step = DRC(config=state["config"], state_in=state["stream_out_state_out"])
    drc_step.start()
    st.write("#### DRC Violation Report")
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
            else: st.text(content)
    except FileNotFoundError: st.warning("DRC report file not found.")
    return {"drc_state_out": drc_step.state_out}

def spice_extraction_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⚡ Agent 21: SPICE Extraction")
    st.write("Extracts a SPICE netlist from the final GDSII layout. This is needed for the LVS check.")
    SpiceExtraction = Step.factory.get("Magic.SpiceExtraction")
    spx_step = SpiceExtraction(config=state["config"], state_in=state["drc_state_out"])
    spx_step.start()
    return {"spice_extraction_state_out": spx_step.state_out}

def lvs_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ↔️ Agent 22: Layout vs. Schematic (LVS)")
    st.write("Compares the extracted SPICE netlist (from the layout) against the original Verilog netlist to ensure they match.")
    LVS = Step.factory.get("Netgen.LVS")
    lvs_step = LVS(config=state["config"], state_in=state["spice_extraction_state_out"])
    lvs_step.start()
    st.write("#### LVS Report Summary")
    report_path = os.path.join(lvs_step.step_dir, "reports", "lvs.netgen.rpt")
    try:
        with open(report_path) as f:
            content = f.read()
            summary_match = re.search(r"Subcircuit summary:(.*?)Final result:", content, re.DOTALL)
            final_result_match = re.search(r"Final result:\s*(.*)", content)
            if summary_match: st.text(summary_match.group(1).strip())
            if final_result_match:
                result = final_result_match.group(1).strip()
                if "Circuits match uniquely" in result: st.success(f"✅ **Final Result:** {result}")
                else: st.error(f"❌ **Final Result:** {result}")
            else: st.warning("Could not parse LVS final result.")
    except FileNotFoundError: st.warning("LVS report file not found.")
    return {"lvs_state_out": lvs_step.state_out}

def render_step_image(state: AgentState, state_key_in: str, caption: str):
    st.write(f"### 🖼️ Rendering: {caption}")
    Render = Step.factory.get("KLayout.Render")
    render_step = Render(config=state["config"], state_in=state[state_key_in])
    render_step.start()
    image_path = os.path.join(render_step.step_dir, "out.png")
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width=400)
    else:
        st.warning(f"Image not found for {caption} at: {image_path}")
    return {}

# --- Conditional Logic ---

def check_simulation(state: AgentState) -> str:
    if state["simulation_passed"]:
        st.success("✅ Simulation Passed. Proceeding to OpenLane flow.")
        return "continue_to_synthesis"
    else:
        st.error("❌ Simulation Failed.")
        if state.get("update_attempt", 0) > 1: # Allow one correction attempt
            st.warning("Looping back to Verilog Corrector for another attempt.")
            return "fix_verilog"
        else:
            st.error("Simulation failed after correction attempt. Halting.")
            return "end"

def check_floorplan(state: AgentState) -> str:
    width_ok = state['die_width_mm'] <= state['max_die_width_mm']
    height_ok = state['die_height_mm'] <= state['max_die_height_mm']

    if width_ok and height_ok:
        st.success("✅ Die size is within limits. Proceeding with Place and Route.")
        return "continue_to_pnr"
    else:
        st.error("❌ Die size exceeds maximum limits. Looping back to Verilog Corrector.")
        feedback = state.get("feedback_log", []) + [f"Floorplan failed. Die size {state['die_width_mm']:.3f}x{state['die_height_mm']:.3f}mm exceeds limit of {state['max_die_width_mm']:.3f}x{state['max_die_height_mm']:.3f}mm. Please simplify the design further."]
        return "fix_verilog"

# NEW CONDITIONAL FUNCTION
def check_sta_violations(state: AgentState) -> str:
    worst_wns = state.get("worst_wns", 0.0) # WNS is in ns
    if worst_wns < 0:
        st.error(f"❌ STA VIOLATION DETECTED (WNS={worst_wns:.2f} ns).")
        return "fix_sta"
    else:
        st.success(f"✅ Timing constraints met (WNS={worst_wns:.2f} ns). Proceeding to final signoff.")
        return "continue_to_signoff"


# --- Build the graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("file_processing", file_processing_agent)
workflow.add_node("icarus_simulation", icarus_simulation_agent)
workflow.add_node("verilog_corrector", verilog_corrector_agent)
workflow.add_node("code_decomposer", code_decomposer_agent)
workflow.add_node("testbench_corrector", testbench_corrector_agent)
workflow.add_node("file_saver", file_saver_agent)
workflow.add_node("setup", setup_agent)
workflow.add_node("synthesis", synthesis_agent)
workflow.add_node("floorplan", floorplan_agent)
workflow.add_node("tap_endcap", tap_endcap_agent)
workflow.add_node("io_placement", io_placement_agent)
workflow.add_node("generate_pdn", generate_pdn_agent)
workflow.add_node("global_placement", global_placement_agent)
workflow.add_node("detailed_placement", detailed_placement_agent)
workflow.add_node("cts", cts_agent)
workflow.add_node("global_routing", global_routing_agent)
workflow.add_node("detailed_routing", detailed_routing_agent)
workflow.add_node("fill_insertion", fill_insertion_agent)
workflow.add_node("rcx", rcx_agent)
workflow.add_node("sta", sta_agent)
workflow.add_node("sta_correction", sta_correction_agent) # New node
workflow.add_node("stream_out", stream_out_agent)
workflow.add_node("drc", drc_agent)
workflow.add_node("spice_extraction", spice_extraction_agent)
workflow.add_node("lvs", lvs_agent)

# Render nodes
workflow.add_node("render_floorplan", lambda s: render_step_image(s, "floorplan_state_out", "Floorplan"))
workflow.add_node("render_routing", lambda s: render_step_image(s, "detailed_routing_state_out", "Detailed Routing"))
workflow.add_node("render_gds", lambda s: render_step_image(s, "stream_out_state_out", "Final GDSII Layout"))


# --- Define Edges for the new flow ---
workflow.add_edge(START, "file_processing")
workflow.add_edge("file_processing", "icarus_simulation")

workflow.add_conditional_edges(
    "icarus_simulation",
    check_simulation,
    {"continue_to_synthesis": "setup", "fix_verilog": "verilog_corrector", "end": END}
)

workflow.add_edge("setup", "synthesis")
workflow.add_edge("synthesis", "floorplan")
workflow.add_edge("floorplan", "render_floorplan")

workflow.add_conditional_edges(
    "render_floorplan",
    check_floorplan,
    {"continue_to_pnr": "tap_endcap", "fix_verilog": "verilog_corrector"}
)

# Correction Loop
workflow.add_edge("verilog_corrector", "code_decomposer")
workflow.add_edge("code_decomposer", "testbench_corrector")
workflow.add_edge("testbench_corrector", "file_saver")
workflow.add_edge("file_saver", "icarus_simulation") # Loop back to simulation

# PNR Flow
pnr_chain = [
    "tap_endcap", "io_placement", "generate_pdn", "global_placement",
    "detailed_placement", "cts", "global_routing", "detailed_routing",
    "render_routing"
]
for i in range(len(pnr_chain) - 1):
    workflow.add_edge(pnr_chain[i], pnr_chain[i+1])

# Post-PNR, Pre-Signoff Flow
workflow.add_edge("render_routing", "fill_insertion")
workflow.add_edge("fill_insertion", "rcx")
workflow.add_edge("rcx", "sta")

# NEW: STA Violation Check and Loop
workflow.add_conditional_edges(
    "sta",
    check_sta_violations,
    {
        "continue_to_signoff": "stream_out", # If STA passes, continue
        "fix_sta": "sta_correction"          # If STA fails, trigger correction
    }
)
workflow.add_edge("sta_correction", "setup") # Loop back to setup with new config

# Final Signoff Flow
signoff_chain = [
    "stream_out", "render_gds", "drc", "spice_extraction", "lvs"
]
for i in range(len(signoff_chain) - 1):
    workflow.add_edge(signoff_chain[i], signoff_chain[i+1])

workflow.add_edge(signoff_chain[-1], END)

app = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 LLM for Chip Design Automation (v6 - With STA Corrector)")
st.write("This application uses a multi-agent workflow to automate the chip design process, including intelligent loops for fixing floorplan and timing violations.")

st.write("### Agentic Workflow Graph")
st.graphviz_chart(
    """
    digraph {
        graph [splines=ortho, nodesep=0.5, ranksep=1];
        node [shape=box, style="rounded,filled", fillcolor="#a9def9", width=2.5, height=0.6, fontsize=10];
        edge [color="#555555", arrowhead=vee];

        subgraph cluster_prep {
            label = "Initial Verification";
            style=filled;
            color=lightgrey;
            file_processing [label="1. File Processing"];
            icarus_simulation [label="2. Icarus Simulation", fillcolor="#fcf6bd"];
        }

        subgraph cluster_pnr {
            label = "Synthesis & PnR";
            style=filled;
            color=lightgrey;
            setup [label="3. OpenLane Setup"];
            synthesis [label="4. Synthesis"];
            floorplan [label="5. Floorplanning"];
            pnr_flow [label="6. PnR Flow"];
        }

        subgraph cluster_correction {
            label = "Optimization Loops";
            style=filled;
            color="#ffe8e8";
            verilog_corrector [label="A. Verilog Corrector (LLM)", fillcolor="#f6d5f7"];
            sta_corrector [label="E. STA Corrector (Agent)", fillcolor="#fbc4ab"];
        }

        subgraph cluster_signoff {
            label = "Signoff";
            style=filled;
            color=lightgrey;
            sta [label="7. Static Timing Analysis", fillcolor="#fcf6bd"];
            signoff_flow [label="8. Final Signoff"];
        }

        end_node [label="END", shape=ellipse, fillcolor="#d4edda"];

        // Edges
        file_processing -> icarus_simulation;
        icarus_simulation -> setup [label="Sim OK"];

        setup -> synthesis -> floorplan -> pnr_flow -> sta;

        // Conditional Edges
        sta -> signoff_flow [label="Timing OK", color=green];
        signoff_flow -> end_node;

        // Loop Edges
        icarus_simulation -> verilog_corrector [label="Sim Fail", style=dashed, color=red];
        floorplan -> verilog_corrector [label="Size Too Big", style=dashed, color=red, constraint=false];
        verilog_corrector -> icarus_simulation [label="Re-Verify", style=dashed, color=orange, constraint=false];

        sta -> sta_corrector [label="Timing Violation", style=dashed, color=red];
        sta_corrector -> setup [label="Re-Run PnR", style=dashed, color=blue];
    }
    """
)


st.sidebar.header("1. Upload Your Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload Verilog design (.v) and testbench (tb.v) files", accept_multiple_files=True
)

if uploaded_files:
    verilog_file_names = [f.name for f in uploaded_files if f.name.endswith((".v", ".vh")) and "tb" not in f.name.lower()]
    if not verilog_file_names:
        st.sidebar.warning("Please upload at least one Verilog design file (not a testbench).")
    else:
        top_level_module = st.sidebar.selectbox(
            "Select the top-level module",
            options=[name.replace(".v", "").replace(".vh", "") for name in verilog_file_names],
        )

        st.sidebar.header("2. Set Constraints")
        max_w = st.sidebar.number_input("Max Die Width (mm)", min_value=0.01, value=1.0, step=0.01, format="%.3f")
        max_h = st.sidebar.number_input("Max Die Height (mm)", min_value=0.01, value=1.0, step=0.01, format="%.3f")

        if st.sidebar.button("🚀 Run Agentic Flow"):
            if not llm:
                st.error("Cannot run flow: Gemini LLM is not initialized.")
            else:
                original_cwd = os.getcwd()
                try:
                    with st.spinner("🚀 Agents at work... This flow involves LLM calls and a full PnR, it will take several minutes."):
                        initial_state = {
                            "uploaded_files": uploaded_files,
                            "top_level_module": top_level_module,
                            "max_die_width_mm": max_w,
                            "max_die_height_mm": max_h,
                        }
                        # Use a high recursion limit for the feedback loops
                        app.invoke(initial_state, {"recursion_limit": 150})
                    st.success("✅ Agentic flow completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during the flow: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    os.chdir(original_cwd)
                    st.write(f"✅ Restored working directory to: `{os.getcwd()}`")
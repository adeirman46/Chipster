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
try:
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
    modified_verilog_code: Optional[str]
    decomposed_files: Dict[str, str]
    testbench_file: Optional[str]
    original_testbench_code: Optional[str]
    modified_testbench_code: Optional[str]
    config: Dict[str, Any]
    run_path: str
    update_attempt: int
    # Constraints
    max_die_width_mm: float
    max_die_height_mm: float
    max_pins: int
    # Metrics
    die_width_mm: float
    die_height_mm: float
    pin_count: int
    # Flow Control
    simulation_passed: bool
    simulation_output: str
    feedback_log: List[str]
    # OpenLane States
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
    lvs_step_dir: Optional[str]
    worst_tns: Optional[float]
    worst_wns: Optional[float]


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
    st.write("### 🧠 Agent 2: Verilog Area/Sim Corrector (LLM)")
    st.info("This agent uses an LLM to rewrite Verilog to fix simulation errors or reduce the design's physical area.")

    if not llm:
        st.error("Gemini LLM not initialized. Skipping correction.")
        return {"modified_verilog_code": "\n".join(state["original_verilog_code"].values())}

    feedback = "\n".join(state['feedback_log'])
    st.write("#### Feedback for Correction (Area/Simulation):")
    st.code(feedback, language='text')

    prompt = f"""
    You are an expert Verilog designer. Your task is to optimize the given Verilog code based on the following feedback from a failed EDA tool run.
    The primary goal is to **simplify the design to reduce its area** or fix simulation errors.

    **Feedback from Tools:**
    {feedback}

    **Optimization Strategies (Apply in order of priority):**
    1.  **Operator Strength Reduction:** Replace expensive operators like multipliers (`*`) with a series of additions or bit-shifts if possible.
    2.  **Module Simplification/Removal:** If a module is used with constant inputs, replace its instantiation with the pre-calculated result. Simplify or remove modules if their full functionality is not required.
    3.  **Aggressive Bit-width Reduction:** This is a critical step for area reduction. Analyze the logic and drastically reduce the bit-width of registers, wires, and parameters. For example, if a 32-bit register only ever holds values up to 100, reduce it to 7 bits (`[6:0]`). You MUST ensure this change is propagated to all connected modules and calculations.

    **RULES:**
    - You MUST generate pure, synthesizable Verilog-2001 compatible code.
    - Combine all Verilog modules into a single, monolithic block of code.
    - Do NOT include the testbench.
    - Your output MUST be only the Verilog code, enclosed in a single markdown block.

    **Original Verilog Code:**
    ---
    """
    code_to_correct = state.get("decomposed_files") or state["original_verilog_code"]
    for filename, code in code_to_correct.items():
        prompt += f"--- {filename} ---\n{code}\n"
    prompt += "---"

    st.write("🤖 Asking Gemini to optimize the Verilog code for Area/Sim...")
    try:
        response = llm.invoke(prompt)
        response_content = response.content
        st.write("#### Gemini's Raw Response:")
        st.markdown(response_content)

        modified_code_match = re.search(r"```(?:verilog)?\s*\n(.*?)```", response_content, re.DOTALL)
        if not modified_code_match:
            st.error("LLM response parsing failed. Could not find a valid Verilog code block. Falling back to previous code version.")
            return {"modified_verilog_code": "\n".join(code_to_correct.values())}

        modified_verilog_code = modified_code_match.group(1).strip()
        if not modified_verilog_code:
            st.error("LLM response parsing failed. The Verilog code block was empty. Falling back to previous code version.")
            return {"modified_verilog_code": "\n".join(code_to_correct.values())}

        st.success("✅ Successfully extracted optimized Verilog code from LLM response.")
        return {"modified_verilog_code": modified_verilog_code}

    except Exception as e:
        st.error(f"An error occurred while communicating with the Gemini API: {e}")
        import traceback
        st.code(traceback.format_exc())
        return {"modified_verilog_code": "\n".join(code_to_correct.values())}

def code_decomposer_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧩 Agent 3: Code Decomposer (LLM-Powered)")
    st.info("After an LLM generates a single block of corrected Verilog, this agent intelligently splits it back into separate files, one for each module.")

    monolithic_code = state.get("modified_verilog_code")
    if not monolithic_code:
        st.error("No modified Verilog code found to decompose.")
        return {"decomposed_files": state.get("decomposed_files", state["original_verilog_code"])}

    st.write("Decomposing LLM-generated code into separate files using Gemini...")

    prompt = f"""
    You are an expert Verilog refactoring tool.
    Your task is to analyze the following monolithic Verilog code and decompose it into multiple files.

    **RULES:**
    1.  Separate each `module` into its own file. The filename should be the module name with a `.v` extension (e.g., `module_name.v`).
    2.  Return a single, valid JSON object where keys are the filenames and values are the complete code content for that file.
    3.  Your final output **MUST** be only the JSON object, enclosed in a markdown block.

    **MONOLITHIC VERILOG CODE:**
    ```verilog
    {monolithic_code}
    ```
    """

    response = llm.invoke(prompt)
    st.write("#### Decomposer LLM Response:")
    st.markdown(response.content)

    try:
        json_str = None
        match = re.search(r"```json\s*(\{.*?\})\s*```", response.content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = response.content.find('{')
            end = response.content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = response.content[start:end+1]

        if not json_str:
            raise json.JSONDecodeError("No valid JSON object found in the LLM response.", response.content, 0)

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

# --- NEW AGENT TO FIX THE BUG ---
def design_name_updater_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📝 Agent 3.5: Design Name Updater")
    st.info("This agent analyzes the decomposed files to find the new top-level module name after code modifications.")
    
    decomposed_files = state["decomposed_files"]
    if not decomposed_files:
        st.warning("No decomposed files to analyze. Keeping original design name.")
        return {}

    # Find all module definitions (module names are filenames without extension)
    defined_modules = {Path(f).stem for f in decomposed_files.keys()}

    # Find all module instantiations
    instantiated_modules = set()
    # This regex finds patterns like "module_name instance_name (" or "#("
    instantiation_re = re.compile(r"\s*(\w+)\s+(?:#\s*\(.*\)\s*)?\w+\s*\(", re.MULTILINE)

    for content in decomposed_files.values():
        matches = instantiation_re.findall(content)
        for module_name in matches:
            # Exclude common keywords that might look like instantiations
            if module_name not in ["module", "input", "output", "wire", "reg"]:
                instantiated_modules.add(module_name)

    # The top-level module is the one that is defined but never instantiated
    top_level_candidates = defined_modules - instantiated_modules
    
    new_design_name = state["design_name"] # Default to old name

    if len(top_level_candidates) == 1:
        new_design_name = top_level_candidates.pop()
        if new_design_name != state["design_name"]:
            st.success(f"✅ New top-level module detected: **{new_design_name}**")
        else:
            st.write("✅ Top-level module name remains the same.")
    elif len(top_level_candidates) > 1:
        st.warning(f"Multiple top-level candidates found: {top_level_candidates}. Defaulting to the previous name: {new_design_name}")
    else:
        st.warning(f"Could not determine a unique top-level module. Defaulting to the previous name: {new_design_name}")

    return {
        "design_name": new_design_name,
        "top_level_module": new_design_name
    }

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

    **RULES:**
    - You MUST generate pure Verilog-2001 compatible code for the testbench.
    - DO NOT use any SystemVerilog features.

    **Design Modules:**
    ---
    """
    for filename, code in state['decomposed_files'].items():
        prompt += f"--- {filename} ---\n{code}\n"

    prompt += f"""
    ---
    **Original Testbench Code (`{os.path.basename(state['testbench_file'])}`):**
    ---
    {tb_to_correct}
    ---
    Provide the updated, complete, and corrected testbench code in a single Verilog code block.
    """

    st.write("🤖 Asking Gemini to update the testbench...")
    response = llm.invoke(prompt)
    st.write("#### Gemini's Response:")
    st.markdown(response.content)

    modified_code = re.search(r"```(?:verilog)?\s*\n(.*?)```", response.content, re.DOTALL)
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
        saved_verilog_files.append(os.path.relpath(file_path, state['run_path']))
        st.write(f"  - Saved `{tb_filename}`")

    return {
        "verilog_files": saved_verilog_files,
        "update_attempt": update_attempt
    }


def icarus_simulation_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 6: Icarus Simulation")
    st.info("This agent compiles and runs a simulation using the Icarus Verilog simulator to functionally verify the design's behavior.")

    if not state.get('testbench_file'):
        st.warning("No testbench file found. Skipping simulation.")
        return {"simulation_passed": True, "simulation_output": "No testbench provided."}

    run_path = state['run_path']
    verilog_files_to_sim = [os.path.join(run_path, f) for f in state['verilog_files']]
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
    st.info("This agent initializes the OpenLane 2.0 configuration for the design.")

    config_or_dict = state.get('config')
    design_name = state["design_name"] # Use the potentially updated design name

    # On loops, the config exists. We need to create a new one if the design name changed.
    if config_or_dict and config_or_dict["DESIGN_NAME"] != design_name:
        st.warning(f"Design name changed from '{config_or_dict['DESIGN_NAME']}' to '{design_name}'. Re-initializing configuration.")
        config_or_dict = None # Force re-initialization

    if config_or_dict:
        st.write("♻️ Looping back: Using existing (potentially modified) configuration.")
        if isinstance(config_or_dict, dict):
            config = Config(config_or_dict)
        else:
            config = config_or_dict

        for item in os.listdir(state['run_path']):
            if item.startswith('runs'):
                shutil.rmtree(os.path.join(state['run_path'], item))
                st.write(f"🧹 Removed old OpenLane run directory: {item}")
    else:
        st.write("🚀 Initial run or Design Name changed: Creating new OpenLane configuration.")
        config = Config.interactive(
            design_name, PDK="gf180mcuC", # Use the correct design name
            CLOCK_PORT="clk", CLOCK_NET="clk", CLOCK_PERIOD=1000,
            PRIMARY_GDSII_STREAMOUT_TOOL="klayout",
        )
    st.write("✅ OpenLane configuration loaded.")
    st.info(f"**Design Name for this run: {config['DESIGN_NAME']}**")
    st.info(f"**Clock Period set to: {config['CLOCK_PERIOD']} ns**")
    return {"config": config}


def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 8: Synthesis")
    st.info("This agent converts the high-level Verilog RTL into a gate-level netlist.")

    synthesizable_files = [f for f in state["verilog_files"] if "_tb" not in f.lower() and "tb." not in f.lower()]
    st.write("Synthesizing the following files:")
    for f in synthesizable_files:
        st.write(f"- `{f}`")

    Synthesis = Step.factory.get("Yosys.Synthesis")
    # Pass the potentially updated design name to the synthesis step
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
    st.info("This agent defines the overall chip dimensions (die area).")
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
    st.info("This agent inserts special cells to prevent latch-up issues.")
    TapEndcap = Step.factory.get("OpenROAD.TapEndcapInsertion")
    tap_step = TapEndcap(config=state["config"], state_in=state["floorplan_state_out"])
    tap_step.start()
    return {"tap_endcap_state_out": tap_step.state_out}

def io_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📍 Agent 11: I/O Pin Placement")
    st.info("This agent performs the detailed placement of the I/O pads.")
    IOPlacement = Step.factory.get("OpenROAD.IOPlacement")
    ioplace_step = IOPlacement(config=state["config"], state_in=state["tap_endcap_state_out"])
    ioplace_step.start()
    return {"io_placement_state_out": ioplace_step.state_out}

def generate_pdn_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⚡ Agent 12: Power Distribution Network (PDN)")
    st.info("This agent generates the grid of power and ground stripes.")
    GeneratePDN = Step.factory.get("OpenROAD.GeneratePDN")
    pdn_step = GeneratePDN(config=state["config"], state_in=state["io_placement_state_out"], FP_PDN_VWIDTH=2, FP_PDN_HWIDTH=2, FP_PDN_VPITCH=30, FP_PDN_HPITCH=30)
    pdn_step.start()
    return {"pdn_state_out": pdn_step.state_out}

def global_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌍 Agent 13: Global Placement")
    st.info("This agent determines the approximate locations for all standard cells.")
    GlobalPlacement = Step.factory.get("OpenROAD.GlobalPlacement")
    gpl_step = GlobalPlacement(config=state["config"], state_in=state["pdn_state_out"])
    gpl_step.start()
    return {"global_placement_state_out": gpl_step.state_out}

def detailed_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📐 Agent 14: Detailed Placement")
    st.info("This agent refines placement, legalizing all cell positions.")
    DetailedPlacement = Step.factory.get("OpenROAD.DetailedPlacement")
    dpl_step = DetailedPlacement(config=state["config"], state_in=state["global_placement_state_out"])
    dpl_step.start()
    return {"detailed_placement_state_out": dpl_step.state_out}

def cts_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌳 Agent 15: Clock Tree Synthesis (CTS)")
    st.info("This agent builds the clock tree to distribute the clock signal.")
    CTS = Step.factory.get("OpenROAD.CTS")
    cts_step = CTS(config=state["config"], state_in=state["detailed_placement_state_out"])
    cts_step.start()
    return {"cts_state_out": cts_step.state_out}

def global_routing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🗺️ Agent 16: Global Routing")
    st.info("This agent plans the paths for the interconnect wires.")
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
    st.write("### ✍️ Agent 17: Detailed Routing")
    st.info("This agent performs the final, exact routing of all wires.")
    DetailedRouting = Step.factory.get("OpenROAD.DetailedRouting")
    drt_step = DetailedRouting(config=state["config"], state_in=state["global_routing_state_out"])
    drt_step.start()
    return {"detailed_routing_state_out": drt_step.state_out}

def fill_insertion_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧱 Agent 18: Fill Insertion")
    st.info("This agent adds 'filler' cells to ensure metal density uniformity.")
    FillInsertion = Step.factory.get("OpenROAD.FillInsertion")
    fill_step = FillInsertion(config=state["config"], state_in=state["detailed_routing_state_out"])
    fill_step.start()
    return {"fill_insertion_state_out": fill_step.state_out}

def rcx_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔌 Agent 19: Parasitics Extraction (RCX)")
    st.info("This agent extracts the parasitic resistance (R) and capacitance (C) of wires.")
    RCX = Step.factory.get("OpenROAD.RCX")
    rcx_step = RCX(config=state["config"], state_in=state["fill_insertion_state_out"])
    rcx_step.start()
    return {"rcx_state_out": rcx_step.state_out}

def sta_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⏱️ Agent 20: Static Timing Analysis (STA)")
    st.info("This analysis step verifies that the chip meets its timing constraints.")
    STAPostPNR = Step.factory.get("OpenROAD.STAPostPNR")
    sta_step = STAPostPNR(config=state["config"], state_in=state["rcx_state_out"])
    sta_step.start()
    st.write("#### STA Timing Violation Summary")
    sta_results = []
    value_re = re.compile(r":\s*(-?[\d\.]+)")
    reports_to_find = ["tns.max.rpt", "tns.min.rpt", "wns.max.rpt", "wns.min.rpt"]
    all_tns, all_wns = [], []

    for root, _, files in os.walk(sta_step.step_dir):
        for file in files:
            if file in reports_to_find:
                corner = os.path.basename(root)
                metric_name = file.replace(".rpt", "").replace(".", " ").title()
                with open(os.path.join(root, file)) as f:
                    content = f.read()
                    match = value_re.search(content)
                    if match:
                        value = float(match.group(1))
                        sta_results.append([corner, metric_name, value])
                        if "Tns Max" in metric_name: all_tns.append(value)
                        if "Wns Max" in metric_name: all_wns.append(value)

    worst_tns = min(all_tns) if all_tns else 0
    worst_wns = min(all_wns) if all_wns else 0
    st.info(f"**Worst Total Negative Slack (TNS): {worst_tns:.2f} ps** | **Worst Negative Slack (WNS): {worst_wns:.2f} ps**")

    if sta_results:
        df_sta = pd.DataFrame(sta_results, columns=["Corner", "Metric", "Value (ps)"])
        pivoted_df = df_sta.pivot(index='Metric', columns='Corner', values='Value (ps)').fillna(0)
        styled_df = pivoted_df.style.applymap(lambda val: f'color: {"red" if val < 0 else "green"}').format("{:.2f}")
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("Could not parse key STA report files (TNS, WNS).")

    return {
        "sta_state_out": sta_step.state_out,
        "worst_tns": worst_tns / 1000.0,
        "worst_wns": worst_wns / 1000.0
    }

def sta_correction_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🤖 Agent 21: STA Corrector")
    st.info("If timing violations are found, this agent attempts to fix them by increasing the clock period.")
    st.error("❌ Timing violations detected! Attempting to fix by adjusting clock period.")

    config_dict = dict(state["config"])
    current_period = float(config_dict["CLOCK_PERIOD"])
    worst_tns_ns = state["worst_tns"]
    worst_wns_ns = state["worst_wns"]
    feedback_msg, new_period = "", current_period

    if abs(worst_tns_ns) > 500:
        new_period, feedback_msg = current_period * 10, f"CRITICAL TNS ({worst_tns_ns:.2f} ns). Drastically increasing clock period 10x."
    elif abs(worst_tns_ns) > 50:
        new_period, feedback_msg = current_period * 2, f"HIGH TNS ({worst_tns_ns:.2f} ns). Increasing clock period 2x."
    elif worst_tns_ns < 0:
        new_period, feedback_msg = current_period * 1.5, f"Small TNS ({worst_tns_ns:.2f} ns). Increasing clock period 1.5x."
    elif worst_wns_ns < 0:
        new_period, feedback_msg = current_period * 1.15, f"TNS OK, but WNS violation ({worst_wns_ns:.2f} ns). Slightly increasing clock period by 15%."

    st.warning(feedback_msg)
    st.write(f"Old Clock Period: {current_period:.2f} ns")
    st.success(f"**New Clock Period: {new_period:.2f} ns**")

    config_dict["CLOCK_PERIOD"] = new_period
    feedback = state.get("feedback_log", []) + [f"STA Correction: {feedback_msg} Changed clock from {current_period}ns to {new_period}ns."]
    return {"config": Config(config_dict), "feedback_log": feedback}


def stream_out_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💾 Agent 22: GDSII Stream Out")
    st.info("This agent generates the final GDSII file for manufacturing.")
    StreamOut = Step.factory.get("KLayout.StreamOut")
    gds_step = StreamOut(config=state["config"], state_in=state["sta_state_out"])
    gds_step.start()
    return {"stream_out_state_out": gds_step.state_out}

def drc_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ✅ Agent 23: Design Rule Check (DRC)")
    st.info("This agent checks if the final layout adheres to the foundry's geometric and electrical rules.")
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
    st.info("This agent extracts a detailed SPICE netlist from the final layout.")
    SpiceExtraction = Step.factory.get("Magic.SpiceExtraction")
    spx_step = SpiceExtraction(config=state["config"], state_in=state["drc_state_out"])
    spx_step.start()
    return {"spice_extraction_state_out": spx_step.state_out}

def lvs_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ↔️ Agent 25: Layout vs. Schematic (LVS)")
    st.info("Compares the extracted SPICE netlist (layout) against the original Verilog netlist.")
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
    
    return {
        "lvs_state_out": lvs_step.state_out,
        "lvs_step_dir": lvs_step.step_dir
    }


# --- PIN COUNTING AND REDUCTION AGENTS (Pin Counter is CORRECTED) ---

def pin_counter_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔢 Agent 26: Pin Counter")
    st.info("This agent inspects the LVS report to count the number of I/O pins.")
    
    lvs_step_dir = state.get("lvs_step_dir")
    design_name = state["design_name"]
    if not lvs_step_dir:
        st.error("LVS step directory not found in state. Cannot count pins.")
        return {"pin_count": -1}

    json_report_path = os.path.join(lvs_step_dir, "reports", "lvs.netgen.json")
    pin_count = 0

    if not os.path.exists(json_report_path):
        st.error(f"LVS JSON report not found at: {json_report_path}. Cannot count pins.")
        return {"pin_count": -1}

    try:
        with open(json_report_path, 'r') as f:
            lvs_data = json.load(f)
            
            # CORRECTED LOGIC: Find the dictionary for the top-level module
            top_module_data = None
            if isinstance(lvs_data, list):
                for item in lvs_data:
                    # Check if the item is a dictionary and has the 'name' key
                    if isinstance(item, dict) and 'name' in item:
                        # The 'name' key contains a list of two names, should be the same
                        if item['name'][0] == design_name:
                            top_module_data = item
                            break
            
            if top_module_data:
                pin_list = top_module_data["pins"][0]
                # Expanded set of common power/ground pins to exclude
                power_ground_pins = {'vccd1', 'vssd1', 'vccd', 'vssd', 'gnd', 'vdd', 'vpw', 'vnw'}
                
                core_pins = [p for p in pin_list if p.lower() not in power_ground_pins]
                pin_count = len(core_pins)
                st.success(f"✅ Successfully parsed LVS report for '{design_name}'. Found {pin_count} I/O pins.")
            else:
                st.error(f"Could not find pin data for top-level module '{design_name}' in the LVS JSON report.")
                pin_count = -1
                
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        st.error(f"Error reading or parsing LVS JSON report: {e}")
        pin_count = -1

    st.write(f"Pin Count: **{pin_count}** / Max Allowed: **{state['max_pins']}**")
    return {"pin_count": pin_count}


def pin_reduction_corrector_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧠 Agent 27: Pin Reduction Corrector (LLM)")
    st.info("This specialized agent uses an LLM to rewrite the Verilog code to reduce the number of I/O pins.")

    feedback = "\n".join(state['feedback_log'])
    st.write("#### Feedback for Correction (Pin Count):")
    st.code(feedback, language='text')

    prompt = f"""
    You are an expert Verilog designer specializing in I/O optimization. Your task is to rewrite the given Verilog code to drastically reduce the number of input/output pins.
    The design currently has **{state['pin_count']}** pins, but the maximum allowed is **{state['max_pins']}**.

    **Feedback from Tools:**
    {feedback}

    **REQUIRED Pin Reduction Strategies (Apply one or more):**
    1.  **Serialization (Most Important):** Convert parallel data buses into serial interfaces.
        - For inputs, implement a **Serial-In, Parallel-Out (SIPO)** shift register. You will need to add a serial data input pin (`ser_in`) and a control signal (e.g., `load_en`).
        - For outputs, implement a **Parallel-In, Serial-Out (PISO)** shift register. You will need a serial data output pin (`ser_out`).
        - This will require state machines or counters to manage the shifting process over multiple clock cycles.
    2.  **Multiplexing:** Use a smaller number of shared data lines with selection bits to choose which internal signal is being driven or read.
    3.  **Encoding:** If there are multiple single-bit inputs/outputs that are mutually exclusive, encode them. For example, 8 one-hot signals can be replaced by a 3-bit binary signal.
    4.  **Use a Standard Protocol (Advanced):** If applicable, implement a simple SPI-like interface with `sclk`, `mosi`, `miso`, and `cs` pins.

    **RULES:**
    - You MUST modify the top-level module's port list.
    - The new logic MUST be functionally equivalent to the old logic, just with a different I/O method.
    - You MUST generate pure, synthesizable Verilog-2001 compatible code.
    - Combine all Verilog modules into a single, monolithic block of code.
    - Do NOT include the testbench.
    - Your output MUST be only the Verilog code, enclosed in a single markdown block.

    **Original Verilog Code:**
    ---
    """
    code_to_correct = state.get("decomposed_files")
    for filename, code in code_to_correct.items():
        prompt += f"--- {filename} ---\n{code}\n"
    prompt += "---"

    st.write("🤖 Asking Gemini to optimize the Verilog code for Pin Reduction...")
    response = llm.invoke(prompt)
    st.write("#### Gemini's Raw Response:")
    st.markdown(response.content)

    modified_code_match = re.search(r"```(?:verilog)?\s*\n(.*?)```", response.content, re.DOTALL)
    if not modified_code_match:
        st.error("LLM response parsing failed. Could not find a valid Verilog code block. Falling back.")
        return {"modified_verilog_code": "\n".join(code_to_correct.values())}

    modified_verilog_code = modified_code_match.group(1).strip()
    st.success("✅ Successfully extracted pin-optimized Verilog code from LLM response.")
    return {"modified_verilog_code": modified_verilog_code}

def pin_reduction_testbench_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧠 Agent 28: Pin Reduction Testbench Corrector (LLM)")
    st.info("This agent rewrites the testbench to work with the new, pin-reduced (likely serial) I/O of the design.")

    if not state.get("original_testbench_code"):
        return {}

    prompt = f"""
    You are an expert Verilog testbench writer. The design under test has just been significantly modified to reduce its I/O pin count.
    This means parallel buses have likely been replaced with serial interfaces (e.g., using shift registers).
    Your task is to rewrite the original testbench to correctly drive and verify this new serial interface.

    **Key Tasks:**
    - Analyze the new design's module definition to understand the new ports (`ser_in`, `ser_out`, control signals, etc.).
    - Modify the testbench stimulus generation. Instead of assigning a parallel value in one step, you must now create a loop or task to shift in data bit-by-bit over multiple clock cycles.
    - Modify the verification logic to capture serial output data and reconstruct it for comparison against expected values.

    **New Design Modules:**
    ---
    """
    for filename, code in state['decomposed_files'].items():
        prompt += f"--- {filename} ---\n{code}\n"

    prompt += f"""
    ---
    **Original Testbench Code (for the old parallel design):**
    ---
    {state['original_testbench_code']}
    ---
    Provide the updated, complete, and corrected testbench code in a single Verilog code block. It MUST correctly interact with the new serial interface.
    """

    st.write("🤖 Asking Gemini to create a new testbench for the pin-reduced design...")
    response = llm.invoke(prompt)
    st.write("#### Gemini's Response:")
    st.markdown(response.content)

    modified_code = re.search(r"```(?:verilog)?\s*\n(.*?)```", response.content, re.DOTALL)
    if not modified_code:
        st.error("Could not extract corrected testbench code from LLM response.")
        return {"modified_testbench_code": state['original_testbench_code']}

    corrected_tb_code = modified_code.group(1).strip()
    st.success("✅ Successfully generated new testbench for pin-reduced design.")
    return {"modified_testbench_code": corrected_tb_code}


def pin_reduction_decomposer_agent(state: AgentState) -> Dict[str, Any]:
    # This agent can be the same as the original decomposer, but we create a new node for clarity in the graph
    st.write("---")
    st.write("### 🧩 Agent 29: Pin Reduction Code Decomposer")
    st.info("Splitting the pin-reduced Verilog back into separate files.")
    return code_decomposer_agent(state)

def pin_reduction_saver_agent(state: AgentState) -> Dict[str, Any]:
    # This agent can be the same as the original saver, but we create a new node for clarity
    st.write("---")
    st.write("### 💾 Agent 30: Pin Reduction File Saver")
    st.info("Saving the pin-reduced Verilog and new testbench to a new versioned directory.")
    # Reset modified testbench code to ensure the new one is picked up
    state['modified_testbench_code'] = state.get('modified_testbench_code')
    return file_saver_agent(state)


def render_step_image(state: AgentState, state_key_in: str, caption: str):
    st.write(f"#### 🖼️ Visualizing: {caption}")
    Render = Step.factory.get("KLayout.Render")
    render_step = Render(config=state["config"], state_in=state[state_key_in])
    render_step.start()
    image_path = os.path.join(render_step.step_dir, "out.png")
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width=400)
    else:
        st.warning(f"Could not render image for {caption}")
    return {}

# --- Conditional Logic ---

def check_simulation(state: AgentState) -> str:
    if state["simulation_passed"]:
        st.success("✅ Simulation Passed. Proceeding to OpenLane flow.")
        return "continue_to_synthesis"
    else:
        st.error("❌ Simulation Failed.")
        feedback = state.get("feedback_log", []) + [f"Icarus simulation failed. Fix the Verilog code:\n{state['simulation_output']}"]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 5:
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
        feedback = state.get("feedback_log", []) + [f"Floorplan failed. Die size {state['die_width_mm']:.3f}x{state['die_height_mm']:.3f}mm exceeds limit. Simplify the design to reduce area."]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 10:
            st.error("Die size too large after multiple correction attempts. Halting.")
            return "end"
        return "fix_verilog"

def check_sta_violations(state: AgentState) -> str:
    worst_tns = state.get("worst_tns", 0.0)
    worst_wns = state.get("worst_wns", 0.0)
    if worst_tns < 0 or worst_wns < 0:
        st.error(f"❌ STA VIOLATION (TNS={worst_tns:.2f} ns, WNS={worst_wns:.2f} ns).")
        if state.get("update_attempt", 0) > 15:
             st.error("Could not meet timing after multiple attempts. Halting.")
             return "end"
        return "fix_sta"
    else:
        st.success(f"✅ Timing constraints met. Proceeding to final signoff.")
        return "continue_to_signoff"

def check_pin_count(state: AgentState) -> str:
    if state["pin_count"] < 0: # Error case
        st.error("Halting due to pin counting error.")
        return "end"
    if state["pin_count"] <= state["max_pins"]:
        st.success("✅ Pin count is within limits. Flow complete!")
        return "end"
    else:
        st.error(f"❌ Pin count ({state['pin_count']}) exceeds maximum of {state['max_pins']}.")
        feedback = state.get("feedback_log", []) + [f"LVS passed, but pin count {state['pin_count']} exceeds limit of {state['max_pins']}. You must reduce the number of I/O ports using serialization or other techniques."]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 20:
             st.error("Could not meet pin count constraint after multiple attempts. Halting.")
             return "end"
        st.warning("Looping back to Verilog Pin Reduction Corrector.")
        return "fix_pins"

# --- Build the graph ---
workflow = StateGraph(AgentState)

# Add Nodes
node_definitions = {
    "file_processing": file_processing_agent,
    "icarus_simulation": icarus_simulation_agent,
    "setup": setup_agent,
    "synthesis": synthesis_agent,
    "floorplan": floorplan_agent,
    "tap_endcap": tap_endcap_agent,
    "io_placement": io_placement_agent,
    "generate_pdn": generate_pdn_agent,
    "global_placement": global_placement_agent,
    "detailed_placement": detailed_placement_agent,
    "cts": cts_agent,
    "global_routing": global_routing_agent,
    "detailed_routing": detailed_routing_agent,
    "fill_insertion": fill_insertion_agent,
    "rcx": rcx_agent,
    "sta": sta_agent,
    "sta_correction": sta_correction_agent,
    "stream_out": stream_out_agent,
    "drc": drc_agent,
    "spice_extraction": spice_extraction_agent,
    "lvs": lvs_agent,
    # Area/Sim correction loop
    "verilog_corrector": verilog_corrector_agent,
    "code_decomposer": code_decomposer_agent,
    "design_name_updater": design_name_updater_agent, # <-- ADDED NEW AGENT
    "testbench_corrector": testbench_corrector_agent,
    "file_saver": file_saver_agent,
    # Pin reduction loop
    "pin_counter": pin_counter_agent,
    "pin_reduction_corrector": pin_reduction_corrector_agent,
    "pin_reduction_decomposer": pin_reduction_decomposer_agent,
    "pin_reduction_testbench": pin_reduction_testbench_agent,
    "pin_reduction_saver": pin_reduction_saver_agent,
    # Visualization nodes
    "render_floorplan": lambda s: render_step_image(s, "floorplan_state_out", "Floorplan Layout"),
    "render_tap_endcap": lambda s: render_step_image(s, "tap_endcap_state_out", "Tap/Endcap Insertion"),
    "render_io": lambda s: render_step_image(s, "io_placement_state_out", "I/O Placement"),
    "render_pdn": lambda s: render_step_image(s, "pdn_state_out", "Power Distribution Network"),
    "render_global_placement": lambda s: render_step_image(s, "global_placement_state_out", "Global Placement"),
    "render_detailed_placement": lambda s: render_step_image(s, "detailed_placement_state_out", "Detailed Placement"),
    "render_cts": lambda s: render_step_image(s, "cts_state_out", "Clock Tree Synthesis"),
    "render_routing": lambda s: render_step_image(s, "detailed_routing_state_out", "Detailed Routing"),
    "render_fill": lambda s: render_step_image(s, "fill_insertion_state_out", "Fill Insertion"),
    "render_gds": lambda s: render_step_image(s, "stream_out_state_out", "Final GDSII Layout")
}
for name, func in node_definitions.items():
    workflow.add_node(name, func)

# --- Define Edges ---
workflow.add_edge(START, "file_processing")
workflow.add_edge("file_processing", "icarus_simulation")

# Conditional Edge 1: Simulation Check
workflow.add_conditional_edges("icarus_simulation", check_simulation,
    {"continue_to_synthesis": "setup", "fix_verilog": "verilog_corrector", "end": END})

# Correction Loop 1: Verilog Area/Sim Fix
workflow.add_edge("verilog_corrector", "code_decomposer")
workflow.add_edge("code_decomposer", "design_name_updater") # <-- FIX: Insert updater
workflow.add_edge("design_name_updater", "testbench_corrector")
workflow.add_edge("testbench_corrector", "file_saver")
workflow.add_edge("file_saver", "icarus_simulation") # Loop back to re-verify

# Main Flow Path
workflow.add_edge("setup", "synthesis")
workflow.add_edge("synthesis", "floorplan")
workflow.add_edge("floorplan", "render_floorplan")

# Conditional Edge 2: Floorplan/Area Check
workflow.add_conditional_edges("render_floorplan", check_floorplan,
    {"continue_to_pnr": "tap_endcap", "fix_verilog": "verilog_corrector", "end": END})

# PNR Chain
pnr_chain = ["tap_endcap", "render_tap_endcap", "io_placement", "render_io", "generate_pdn", "render_pdn",
             "global_placement", "render_global_placement", "detailed_placement", "render_detailed_placement",
             "cts", "render_cts", "global_routing", "detailed_routing", "render_routing"]
for i in range(len(pnr_chain) - 1):
    workflow.add_edge(pnr_chain[i], pnr_chain[i+1])

workflow.add_edge("render_routing", "fill_insertion")
workflow.add_edge("fill_insertion", "render_fill")
workflow.add_edge("render_fill", "rcx")
workflow.add_edge("rcx", "sta")

# Conditional Edge 3: STA/Timing Check
workflow.add_conditional_edges("sta", check_sta_violations,
    {"continue_to_signoff": "stream_out", "fix_sta": "sta_correction", "end": END})

# Correction Loop 2: STA/Timing fix
workflow.add_edge("sta_correction", "setup") # Loop back to setup with new config

# Final Signoff Chain
signoff_chain = ["stream_out", "render_gds", "drc", "spice_extraction", "lvs"]
for i in range(len(signoff_chain) - 1):
    workflow.add_edge(signoff_chain[i], signoff_chain[i+1])

# Conditional Edge 4: Pin Count Check
workflow.add_edge("lvs", "pin_counter")
workflow.add_conditional_edges("pin_counter", check_pin_count,
    {"fix_pins": "pin_reduction_corrector", "end": END})

# Correction Loop 3: Pin Reduction Fix
workflow.add_edge("pin_reduction_corrector", "pin_reduction_decomposer")
workflow.add_edge("pin_reduction_decomposer", "design_name_updater") # <-- FIX: Insert updater
# The name updater will flow into the testbench corrector
workflow.add_edge("pin_reduction_testbench", "pin_reduction_saver")
workflow.add_edge("pin_reduction_saver", "icarus_simulation") # Loop all the way back to the start

app = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 LLM for Chip Design Automation")
st.write("This application uses a multi-agent workflow to automate the digital chip design flow, from RTL to GDSII. It includes intelligent feedback loops to correct functional, area, timing, and I/O pin count violations.")

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
        max_p = st.sidebar.number_input("Max I/O Pins", min_value=4, value=30, step=1)

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
                            "max_pins": max_p,
                        }
                        app.invoke(initial_state, {"recursion_limit": 200})
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
        label="1. Pre-Processing & Verification"; style="rounded,filled"; color="#e3f2fd"; node[fillcolor="#bbdefb"];
        file_processing [label="1. File Processing"];
        icarus_simulation [label="6. Icarus Simulation", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }

    subgraph cluster_correction_verilog {
        label="A. Area/Sim Correction Loop"; style="rounded,filled"; color="#ffebee"; node[fillcolor="#ffcdd2"];
        verilog_corrector [label="2. Verilog Area/Sim Corrector"];
        code_decomposer [label="3. Code Decomposer"];
        design_name_updater [label="3.5. Design Name Updater", shape=septagon, style=filled, fillcolor="#f8bbd0"];
        testbench_corrector [label="4. Testbench Corrector"];
        file_saver [label="5. File Saver"];
    }

    subgraph cluster_pnr {
        label="2. Physical Design (PnR) & Timing"; style="rounded,filled"; color="#e8f5e9"; node[fillcolor="#c8e6c9"];
        setup [label="7. OpenLane Setup"];
        synthesis [label="8. Synthesis"];
        floorplan [label="9. Floorplan", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
        pnr_group [label="PnR & Vis Steps (10-19)"];
        sta [label="20. Static Timing Analysis", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }

    subgraph cluster_correction_sta {
        label="B. Timing Correction Loop"; style="rounded,filled"; color="#fff3e0"; node[fillcolor="#ffe0b2"];
        sta_correction [label="21. STA Corrector"];
    }

    subgraph cluster_signoff {
        label="3. Final Signoff"; style="rounded,filled"; color="#f3e5f5"; node[fillcolor="#e1bee7"];
        stream_out [label="22. GDSII Stream Out"];
        drc [label="23. DRC"];
        spice_extraction [label="24. SPICE Extraction"];
        lvs [label="25. LVS"];
        pin_counter [label="26. Pin Counter", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }
    
    subgraph cluster_correction_pins {
        label="C. Pin Reduction Loop"; style="rounded,filled"; color="#dcedc8"; node[fillcolor="#c5e1a5"];
        pin_reduction_corrector [label="27. Pin Reduction Corrector"];
        pin_reduction_decomposer [label="28. Pin Reduction Decomposer"];
        pin_reduction_testbench [label="29. Pin Reduction Testbench Agent"];
        pin_reduction_saver [label="30. Pin Reduction Saver"];
    }

    end_node [label="Flow Complete", shape=ellipse, style=filled, fillcolor="#b2dfdb"];

    // Main Flow
    file_processing -> icarus_simulation;
    icarus_simulation -> setup [label=" Sim OK", color=darkgreen, fontcolor=darkgreen];
    setup -> synthesis -> floorplan;
    floorplan -> pnr_group [label=" Area OK", color=darkgreen, fontcolor=darkgreen];
    pnr_group -> sta;
    sta -> stream_out [label=" Timing OK", color=darkgreen, fontcolor=darkgreen];
    stream_out -> drc -> spice_extraction -> lvs -> pin_counter;
    pin_counter -> end_node [label=" Pins OK", color=darkgreen, fontcolor=darkgreen, style=bold];

    // Loop 1: Verilog Fix (Red)
    icarus_simulation -> verilog_corrector [label=" Sim FAIL", style=dashed, color=red, fontcolor=red, constraint=false];
    floorplan -> verilog_corrector [label=" Area TOO BIG", style=dashed, color=red, fontcolor=red, constraint=false];
    verilog_corrector -> code_decomposer -> design_name_updater -> testbench_corrector -> file_saver -> icarus_simulation [style=dashed, color=red, arrowhead=normal];

    // Loop 2: STA Fix (Blue)
    sta -> sta_correction [label=" Timing FAIL", style=dashed, color=blue, fontcolor=blue, constraint=false];
    sta_correction -> setup [style=dashed, color=blue, arrowhead=normal, label=" Re-run PnR w/ new clock"];
    
    // Loop 3: Pin Reduction (Dark Orange)
    pin_counter -> pin_reduction_corrector [label=" Too Many Pins", style=dashed, color="#E65100", fontcolor="#E65100", constraint=false];
    pin_reduction_corrector -> pin_reduction_decomposer;
    pin_reduction_decomposer -> design_name_updater [style=dashed, color="#E65100"]; // Reuse updater
    // The existing edge from design_name_updater to testbench_corrector works for both loops
    pin_reduction_testbench -> pin_reduction_saver -> icarus_simulation [style=dashed, color="#E65100", arrowhead=normal, label="Re-verify new I/O"];
}
""")
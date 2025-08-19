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
    # Use a more advanced model for better reasoning in code generation
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    st.sidebar.success("Gemini 2.5 Pro Initialized")
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
    current_src_dir: str # Tracks the latest source code directory
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
    simulation_verified: bool
    feedback_log: List[str]
    lvs_passed: bool
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
    """
    Initial agent to set up the environment, process uploaded files,
    and separate design files from the testbench.
    """
    st.write("---")
    st.write("### 📂 Agent 1: File Processing")
    st.info("This agent processes uploaded Verilog files, creates a dedicated run directory, and separates design files from the testbench.")
    uploaded_files = state["uploaded_files"]
    top_level_module = state["top_level_module"]
    design_name = top_level_module

    # Create a clean, dedicated directory for this run
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

    # Process and save each uploaded file
    for file in uploaded_files:
        file_path = os.path.join(src_dir, file.name)
        file_content_buffer = file.getbuffer()
        with open(file_path, "wb") as f:
            f.write(file_content_buffer)

        if file.name.endswith((".v", ".vh")):
            decoded_content = file_content_buffer.tobytes().decode('utf-8', errors='ignore')
            # Heuristic to identify testbench files
            if "tb" in file.name.lower() or "testbench" in file.name.lower():
                    testbench_file = file_path
                    original_testbench_code = decoded_content
            else:
                    verilog_files.append(file_path)
                    original_verilog_code[file.name] = decoded_content

    st.write(f"✅ Top-level module '{top_level_module}' selected.")
    st.write(f"✅ Verilog files saved in: `{src_dir}`")
    if testbench_file:
        st.write(f"✅ Testbench file found: `{os.path.basename(testbench_file)}`")
    else:
        st.warning("⚠️ No testbench file detected. Simulation will be skipped.")

    # Change the current working directory to the run path for tool compatibility
    os.chdir(run_path)
    st.write(f"✅ Changed working directory to: `{os.getcwd()}`")

    # Ensure testbench_file path is relative if it exists
    relative_tb_path = os.path.relpath(testbench_file, os.getcwd()) if testbench_file else None

    return {
        "design_name": design_name,
        "verilog_files": [os.path.relpath(p, os.getcwd()) for p in verilog_files],
        "original_verilog_code": original_verilog_code,
        "decomposed_files": original_verilog_code, # Initially, decomposed is same as original
        "testbench_file": relative_tb_path,
        "original_testbench_code": original_testbench_code,
        "run_path": os.getcwd(),
        "current_src_dir": src_dir, # Set the initial src directory
        "feedback_log": ["Starting the design flow."],
        "update_attempt": 0,
    }

def verilog_corrector_agent(state: AgentState) -> Dict[str, Any]:
    """
    Uses an LLM to rewrite Verilog code based on feedback from failed tool runs,
    focusing on reducing area or fixing simulation errors.
    """
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

        # Robustly extract Verilog code from the markdown response
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
    """
    After an LLM generates a single block of Verilog, this agent splits it
    back into separate files, one for each module.
    """
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
    2.  Any file that starts with `// --- Start of content from <filename> ---` should be extracted into its own file with that <filename>.
    3.  Return a single, valid JSON object where keys are the filenames and values are the complete code content for that file.
    4.  Your final output **MUST** be only the JSON object, enclosed in a markdown block.

    **MONOLITHIC VERILOG CODE:**
    ```verilog
    {monolithic_code}
    ```
    """

    response = llm.invoke(prompt)
    st.write("#### Decomposer LLM Response:")
    st.markdown(response.content)

    try:
        # Robustly extract JSON from the markdown response
        json_str = None
        match = re.search(r"```json\s*(\{.*?\})\s*```", response.content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else: # Fallback for cases where the markdown block is missing
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

def design_name_updater_agent(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes decomposed files to find the new top-level module name after
    code modifications, as the LLM might rename it.
    """
    st.write("---")
    st.write("### 📝 Agent 3.5: Design Name Updater")
    st.info("This agent analyzes the decomposed files to find the new top-level module name after code modifications.")

    decomposed_files = state["decomposed_files"]
    if not decomposed_files:
        st.warning("No decomposed files to analyze. Keeping original design name.")
        return {}

    # Find all defined modules (from filenames, excluding .vh files)
    defined_modules = {Path(f).stem for f in decomposed_files.keys() if f.endswith(".v")}

    # Find all instantiated modules by scanning the code
    instantiated_modules = set()
    # Regex to find module instantiations, ignoring keywords and port connections
    instantiation_re = re.compile(r"^\s*(\w+)\s+(?:#\s*\(.*\)\s*)?\w+\s*\(", re.MULTILINE)

    for content in decomposed_files.values():
        matches = instantiation_re.findall(content)
        for module_name in matches:
            # Filter out Verilog keywords that might look like instantiations
            if module_name not in ["module", "input", "output", "wire", "reg", "always", "assign", "case", "if", "for", "function"]:
                instantiated_modules.add(module_name)

    # The top-level module is defined but never instantiated
    top_level_candidates = defined_modules - instantiated_modules

    new_design_name = state["design_name"]

    if len(top_level_candidates) == 1:
        new_design_name = top_level_candidates.pop()
        if new_design_name != state["design_name"]:
            st.success(f"✅ New top-level module detected: **{new_design_name}**")
        else:
            st.write("✅ Top-level module name remains the same.")
    elif len(top_level_candidates) > 1:
        st.warning(f"Multiple top-level candidates found: {top_level_candidates}. Defaulting to the previous name: {new_design_name}")
    else: # No candidates found, might be a single-module design
        if len(defined_modules) == 1:
            new_design_name = defined_modules.pop()
            st.write(f"✅ Single module design detected. Setting top-level to: **{new_design_name}**")
        else:
            st.warning(f"Could not determine a unique top-level module. Defaulting to the previous name: {new_design_name}")

    return {
        "design_name": new_design_name,
        "top_level_module": new_design_name
    }

def testbench_corrector_agent(state: AgentState) -> Dict[str, Any]:
    """
    Uses an LLM to update the testbench to match the (potentially modified)
    design and adds explicit, timed self-checking logic.
    """
    st.write("---")
    st.write("### 🧠 Agent 4: Testbench Corrector (LLM)")
    st.info("This agent updates the testbench to match the design and adds explicit, timed self-checking.")

    if not llm or not state.get("original_testbench_code"):
        st.warning("No LLM or original testbench found. Skipping testbench correction.")
        return {}

    # Use the most recently modified testbench if available, otherwise use the original
    tb_to_correct = state.get("modified_testbench_code") or state["original_testbench_code"]

    prompt = f"""
    You are an expert Verilog testbench writer. Your task is to create a robust, self-checking testbench compatible with the provided design modules. The design may be pipelined and require time to produce a valid output.

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

    **CRITICAL INSTRUCTIONS FOR A ROBUST TESTBENCH:**
    1.  **Update Instantiation:** Ensure the testbench correctly instantiates and connects to the design under test, especially if the module name or ports have changed. The top-level module is `{state['top_level_module']}`.
    2.  **Initial Reset:** The testbench MUST start with a proper reset sequence for the design.
    3.  **Wait for Pipeline Latency:** After applying the stimulus (inputs), you MUST add a significant delay (e.g., `#100` or more) before checking the outputs. This allows time for the pipelined logic to compute the result.
    4.  **Self-Checking Logic:** At the end of the simulation, the testbench must determine if the test passed or failed based on the final output values.
    5.  **Clear Output Message:** You **MUST** add a final block that uses `$display` to print **EXACTLY** `"Result: PASSED"` on a new line if all checks are correct. Do not add any other characters or emojis.
    6.  If any check fails, it must use `$display` to print **EXACTLY** `"Result: FAILED"` on a new line, and then immediately call `$finish`.
    7.  The testbench must end with a `$finish;` statement after the final check.

    Provide the updated, complete, and robust self-checking testbench code in a single Verilog code block.
    """

    st.write("🤖 Asking Gemini to update the testbench with self-checking and proper timing...")
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
    """
    Saves the corrected and decomposed Verilog files to a versioned subdirectory
    to ensure a clean state for the next tool run.
    """
    st.write("---")
    st.write("### 💾 Agent 5: File Saver")
    st.info("This agent saves the newly corrected and decomposed Verilog files to a versioned subdirectory, ensuring a clean state for the next simulation or layout attempt.")

    update_attempt = state.get("update_attempt", 0) + 1

    # Create a new versioned directory for this update attempt
    save_path = os.path.join(state['run_path'], f"updated_codes_{update_attempt}")
    os.makedirs(save_path, exist_ok=True)
    st.write(f"Saving updated files to: `{save_path}`")

    verilog_to_save = state["decomposed_files"]
    tb_to_save = state.get("modified_testbench_code") or state["original_testbench_code"]

    saved_verilog_files = []

    # Save design files
    for filename, content in verilog_to_save.items():
        file_path = os.path.join(save_path, filename)
        with open(file_path, 'w') as f: f.write(content)
        saved_verilog_files.append(os.path.relpath(file_path, state['run_path']))
        st.write(f"  - Saved `{filename}`")

    # Save testbench file if it exists
    if state.get("testbench_file") and tb_to_save:
        tb_filename = os.path.basename(state["testbench_file"])
        file_path = os.path.join(save_path, tb_filename)
        with open(file_path, 'w') as f: f.write(tb_to_save)
        st.write(f"  - Saved `{tb_filename}`")
        # Update the state to point to the new testbench location
        state["testbench_file"] = os.path.relpath(file_path, state['run_path'])


    return {
        "verilog_files": saved_verilog_files,
        "update_attempt": update_attempt,
        "current_src_dir": save_path # IMPORTANT: Update the current source directory
    }


def icarus_simulation_agent(state: AgentState) -> Dict[str, Any]:
    """
    Compiles and runs a simulation using the Icarus Verilog simulator.
    """
    st.write("---")
    st.write("### 🔬 Agent 6: Icarus Simulation")
    st.info("This agent compiles and runs a simulation using the Icarus Verilog simulator to functionally verify the design's behavior.")

    if not state.get('testbench_file'):
        st.warning("No testbench file found. Skipping simulation.")
        # Treat as passed so the flow can continue to synthesis
        return {"simulation_passed": True, "simulation_verified": True, "simulation_output": "No testbench provided."}

    run_path = state['run_path']
    current_src_dir = state['current_src_dir'] # Get the current source directory

    # Files are already relative to the run_path, so we just need the directory for the -I flag
    verilog_files_to_sim = [os.path.join(run_path, f) for f in state['verilog_files']]
    testbench_path = os.path.join(run_path, state['testbench_file'])
    all_files_for_sim = verilog_files_to_sim + [testbench_path]

    output_vvp_file = os.path.join(run_path, "design.vvp")

    # Use the current_src_dir for the include path
    compile_command = ["iverilog", "-g2005-sv", "-o", output_vvp_file, "-I", current_src_dir] + all_files_for_sim

    try:
        st.write(f"Running compilation: `{' '.join(compile_command)}`")
        # Use a timeout to prevent hanging
        compile_process = subprocess.run(compile_command, capture_output=True, text=True, check=True, timeout=60)
        st.write("✅ Compilation successful.")

        sim_command = ["vvp", output_vvp_file]
        st.write(f"Running simulation: `{' '.join(sim_command)}`")
        sim_process = subprocess.run(sim_command, capture_output=True, text=True, check=True, timeout=60)

        st.write("✅ Simulation process completed.")
        st.text_area("Simulation Output", sim_process.stdout, height=150, key=f"sim_output_{state.get('update_attempt', 0)}")
        return {"simulation_passed": True, "simulation_output": sim_process.stdout}

    except subprocess.CalledProcessError as e:
        # This catches errors where the tool runs but returns a non-zero exit code
        error_message = f"ERROR during {'compilation' if 'iverilog' in ' '.join(e.cmd) else 'simulation'}:\n{e.stderr or e.stdout}"
        st.error(error_message)
        return {"simulation_passed": False, "simulation_output": error_message}
    except subprocess.TimeoutExpired as e:
        # This catches if the simulation takes too long
        error_message = f"ERROR: {'Compilation' if 'iverilog' in ' '.join(e.cmd) else 'Simulation'} timed out."
        st.error(error_message)
        return {"simulation_passed": False, "simulation_output": error_message}

def simulation_verifier_agent(state: AgentState) -> Dict[str, Any]:
    """
    Checks the simulation output. On the first run, it is lenient.
    On subsequent runs, it strictly checks for a 'PASSED' or 'SUCCESS' message.
    """
    st.write("---")
    st.write("### 🧐 Agent 6.5: Simulation Verifier")
    st.info("This agent checks the simulation output for a 'PASSED' or 'SUCCESS' message from the self-checking testbench.")

    update_attempt = state.get("update_attempt", 0)
    simulation_output = state.get("simulation_output", "")
    ascii_output = simulation_output.encode('ascii', 'ignore').decode('ascii').strip()

    # On the very first run (attempt 0), if the simulation process completed but
    # produced no output, we will pass it. This handles cases where the initial
    # testbench is not self-checking. The loop will later add self-checking if needed.
    if update_attempt == 0 and state["simulation_passed"] and not ascii_output:
        st.warning("⚠️ First simulation run produced no output. Passing to allow the flow to proceed. The testbench will be updated if a later stage fails and triggers a correction loop.")
        return {"simulation_verified": True}

    # For all subsequent runs, or if the first run produced any output,
    # we strictly enforce that the testbench must be self-checking.
    if re.search(r'Result:\s*PASSED', ascii_output, re.IGNORECASE):
        st.success("✅ Verification PASSED: 'Result: PASSED' found in simulation output.")
        return {"simulation_verified": True}
    else:
        st.error("❌ Verification FAILED: 'Result: PASSED' not found in simulation output.")
        st.write("Sanitized output that was checked:")
        st.code(ascii_output if ascii_output else "[No output produced by simulation]")
        return {"simulation_verified": False}


def setup_agent(state: AgentState) -> Dict[str, Any]:
    """
    Initializes or updates the OpenLane 2.0 configuration for the design.
    """
    st.write("---")
    st.write("### 🛠️ Agent 7: OpenLane Setup")
    st.info("This agent initializes the OpenLane 2.0 configuration for the design.")

    config_or_dict = state.get('config')
    design_name = state["design_name"]

    # If the design name has changed, we must re-initialize the config
    if config_or_dict and config_or_dict.get("DESIGN_NAME") != design_name:
        st.warning(f"Design name changed from '{config_or_dict['DESIGN_NAME']}' to '{design_name}'. Re-initializing configuration.")
        config_or_dict = None

    if config_or_dict:
        st.write("♻️ Looping back: Using existing (potentially modified) configuration.")
        if isinstance(config_or_dict, dict):
            config = Config(config_or_dict)
        else:
            config = config_or_dict

        # Clean up old OpenLane run directories to ensure a fresh start
        for item in os.listdir(state['run_path']):
            if item.startswith('runs'):
                shutil.rmtree(os.path.join(state['run_path'], item))
                st.write(f"🧹 Removed old OpenLane run directory: {item}")
    else:
        st.write("🚀 Initial run or Design Name changed: Creating new OpenLane configuration.")
        # Create a default configuration
        config = Config.interactive(
            design_name, PDK="gf180mcuC",
            CLOCK_PORT="clk", CLOCK_NET="clk", CLOCK_PERIOD=1000.0, # Start with a reasonable clock period
            PRIMARY_GDSII_STREAMOUT_TOOL="klayout",
        )
    st.write("✅ OpenLane configuration loaded.")
    st.info(f"**Design Name for this run: {config['DESIGN_NAME']}**")
    st.info(f"**Clock Period set to: {config['CLOCK_PERIOD']} ns**")
    return {"config": config}


def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Converts the high-level Verilog RTL into a gate-level netlist using Yosys.
    """
    st.write("---")
    st.write("### 🔬 Agent 8: Synthesis")
    st.info("This agent converts the high-level Verilog RTL into a gate-level netlist.")

    # Use the current source directory from the state
    src_path = state['current_src_dir']

    synthesizable_files = [
        os.path.join(src_path, f) for f in os.listdir(src_path)
        if f.endswith(('.v', '.vh')) and 'tb' not in f.lower() and 'testbench' not in f.lower()
    ]

    st.write(f"Synthesizing files from: `{src_path}`")
    for f in synthesizable_files:
        st.write(f"- `{os.path.basename(f)}`")

    Synthesis = Step.factory.get("Yosys.Synthesis")
    synthesis_step = Synthesis(config=state["config"], state_in=State(), VERILOG_FILES=synthesizable_files)
    synthesis_step.start()
    report_path = os.path.join(synthesis_step.step_dir, "reports", "stat.json")
    with open(report_path) as f: metrics = json.load(f)
    st.write("#### Synthesis Metrics")
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=["Value"]).astype(str))
    return {"synthesis_state_out": synthesis_step.state_out}


def floorplan_agent(state: AgentState) -> Dict[str, Any]:
    """
    Defines the chip's dimensions (die area) and core area.
    """
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

    # Extract die dimensions from the metrics report
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

# --- Placeholder agents for standard PnR steps ---
# These agents simply run their corresponding OpenLane step.

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
    """
    Performs Static Timing Analysis (STA) to check for timing violations.
    """
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

    # Walk through the STA step directory to find all timing reports
    for root, _, files in os.walk(sta_step.step_dir):
        for file in files:
            if file in reports_to_find:
                corner = os.path.basename(root) # e.g., 'ss_100C_1v60'
                metric_name = file.replace(".rpt", "").replace(".", " ").title()
                with open(os.path.join(root, file)) as f:
                    content = f.read()
                    match = value_re.search(content)
                    if match:
                        value = float(match.group(1))
                        sta_results.append([corner, metric_name, value])
                        if "Tns Max" in metric_name: all_tns.append(value)
                        if "Wns Max" in metric_name: all_wns.append(value)

    # Find the worst (most negative) slack values across all corners
    worst_tns = min(all_tns) if all_tns else 0
    worst_wns = min(all_wns) if all_wns else 0
    st.info(f"**Worst Total Negative Slack (TNS): {worst_tns:.2f} ps** | **Worst Negative Slack (WNS): {worst_wns:.2f} ps**")

    if sta_results:
        df_sta = pd.DataFrame(sta_results, columns=["Corner", "Metric", "Value (ps)"])
        # Pivot the table for better readability
        pivoted_df = df_sta.pivot(index='Metric', columns='Corner', values='Value (ps)').fillna(0)
        # Color-code the results for quick analysis
        styled_df = pivoted_df.style.applymap(lambda val: f'color: {"red" if val < 0 else "green"}').format("{:.2f}")
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("Could not parse key STA report files (TNS, WNS).")

    return {
        "sta_state_out": sta_step.state_out,
        "worst_tns": worst_tns / 1000.0, # Convert to ns for correction logic
        "worst_wns": worst_wns / 1000.0, # Convert to ns
    }

def sta_correction_agent(state: AgentState) -> Dict[str, Any]:
    """
    If timing violations are found, this agent attempts to fix them by
    increasing the clock period (i.e., slowing down the clock).
    """
    st.write("---")
    st.write("### 🤖 Agent 21: STA Corrector")
    st.info("If timing violations are found, this agent attempts to fix them by increasing the clock period.")
    st.error("❌ Timing violations detected! Attempting to fix by adjusting clock period.")

    config_dict = dict(state["config"])
    current_period = float(config_dict["CLOCK_PERIOD"])
    worst_tns_ns = state["worst_tns"]
    worst_wns_ns = state["worst_wns"]
    feedback_msg, new_period = "", current_period

    # Apply increasingly aggressive corrections based on the severity of the violation
    if abs(worst_tns_ns) > 500:
        new_period, feedback_msg = current_period * 10, f"CRITICAL TNS ({worst_tns_ns:.2f} ns). Drastically increasing clock period 10x."
    elif abs(worst_tns_ns) > 50:
        new_period, feedback_msg = current_period * 2, f"HIGH TNS ({worst_tns_ns:.2f} ns). Increasing clock period 2x."
    elif worst_tns_ns < 0:
        new_period, feedback_msg = current_period * 1.5, f"Small TNS ({worst_tns_ns:.2f} ns). Increasing clock period 1.5x."
    elif worst_wns_ns < 0:
        # TNS might be ok, but WNS (the single worst path) is failing
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
                st.text_area("DRC Report", content, height=200, key=f"drc_report_{state.get('update_attempt', 0)}")
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

    return {
        "lvs_state_out": lvs_step.state_out,
        "lvs_step_dir": lvs_step.step_dir
    }

def lvs_verifier_agent(state: AgentState) -> Dict[str, Any]:
    """
    Parses the LVS report to determine if the layout and schematic match.
    """
    st.write("---")
    st.write("### 🧐 Agent 25.5: LVS Verifier")
    st.info("This agent parses the LVS report to verify that the layout and schematic match.")

    lvs_step_dir = state.get("lvs_step_dir")
    if not lvs_step_dir:
        st.error("LVS step directory not found in state. Cannot verify LVS.")
        return {"lvs_passed": False}

    report_path = os.path.join(lvs_step_dir, "reports", "lvs.netgen.rpt")
    lvs_passed = False

    try:
        with open(report_path) as f:
            content = f.read()
            st.text_area("LVS Report", content, height=200, key=f"lvs_report_{state.get('update_attempt', 0)}")
            # The key phrase for a successful LVS run in Netgen
            if "Circuits match uniquely" in content:
                st.success("✅ LVS PASSED: Circuits match uniquely.")
                lvs_passed = True
            else:
                st.error("❌ LVS FAILED: Circuits do not match.")
                lvs_passed = False
    except FileNotFoundError:
        st.error(f"LVS report file not found at: {report_path}")
        lvs_passed = False

    return {"lvs_passed": lvs_passed}


def pin_counter_agent(state: AgentState) -> Dict[str, Any]:
    """
    Inspects the LVS report to count the number of non-power/ground I/O pins.
    """
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

            top_module_data = None
            # The JSON report is a list of dictionaries, one for each module
            if isinstance(lvs_data, list):
                for item in lvs_data:
                    if isinstance(item, dict) and 'name' in item:
                        # Find the dictionary corresponding to our top-level module
                        if item['name'][0] == design_name:
                            top_module_data = item
                            break

            if top_module_data:
                pin_list = top_module_data["pins"][0]
                # Define a set of common power/ground pin names to exclude
                power_ground_pins = {'vccd1', 'vssd1', 'vccd', 'vssd', 'gnd', 'vdd', 'vpw', 'vnw'}

                # Count pins that are not in the power/ground set
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
    """
    A specialized LLM agent that creates a new top-level module to integrate a
    serial interface with the original design, reducing pin count.
    """
    st.write("---")
    st.write("### 🧠 Agent 27: Pin Reduction Corrector (LLM)")
    st.info("This agent uses an LLM to build a serial interface and a new top-level module to reduce I/O pins.")

    feedback = "\n".join(state['feedback_log'])
    st.write("#### Feedback for Correction (Pin Count):")
    st.code(feedback, language='text')

    original_top_module = state['top_level_module']

    prompt = f"""
    You are an expert Verilog designer specializing in I/O optimization. Your goal is to reduce the pin count of a design by creating a new top-level module that integrates a serial communication block.
    The design currently has **{state['pin_count']}** pins, but the maximum allowed is **{state['max_pins']}**.

    **REQUIRED STRATEGY: MODULAR SERIAL INTEGRATION**
    You MUST NOT modify the original Verilog modules. Instead, you will create two new modules:
    1. A generic, reusable serial communication module (e.g., an SPI slave or a simple UART).
    2. A new top-level module that instantiates BOTH the serial module AND the original design, connecting them together.

    **INSTRUCTIONS:**
    1.  **Preserve Original Code:** Keep all the original Verilog modules provided below completely unchanged.
    2.  **Create a Serial Interface Module:**
        -   Design a self-contained module for serial communication. Let's call it `spi_interface`.
        -   This module will have serial pins (e.g., `sclk`, `cs`, `mosi`, `miso`) on one side and parallel data buses on the other side to connect to the main design.
    3.  **Create a New Top-Level Module:**
        -   Create a new module named `{original_top_module}_with_serial`. This will be the new top-level for the entire system.
        -   The port list for this new top-level module MUST have fewer than {state['max_pins']} pins. It will expose the serial interface pins.
        -   Inside this new top-level, you MUST instantiate two sub-modules:
            a. The `spi_interface` module you just created.
            b. The original top-level design: `{original_top_module}`.
        -   Connect the parallel data buses from the serial module to the corresponding inputs/outputs of the original design.
    4.  **Final Output:** Your output MUST be a single, monolithic block of synthesizable Verilog-2001 code. It must contain:
        - The **UNCHANGED original modules**.
        - The **NEW serial interface module**.
        - The **NEW top-level module** that connects everything.
    5.  Enclose the final complete Verilog code in a single markdown block. Do NOT include a testbench.

    **Original Verilog Code (Do NOT modify this part):**
    ---
    """
    code_to_correct = state.get("decomposed_files")
    for filename, code in code_to_correct.items():
        prompt += f"--- {filename} ---\n{code}\n"
    prompt += "---"

    st.write("🤖 Asking Gemini to create a serial interface and new top-level to reduce pin count...")
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
    """
    A specialized LLM agent that rewrites the testbench to work with the new,
    pin-reduced serial wrapper of the design.
    """
    st.write("---")
    st.write("### 🧠 Agent 28: Pin Reduction Testbench Corrector (LLM)")
    st.info("This agent rewrites the testbench to work with the new serial wrapper interface.")

    if not llm or not state.get("original_testbench_code"):
        st.warning("No LLM or original testbench found. Skipping testbench correction.")
        return {}

    tb_to_correct = state.get("modified_testbench_code") or state["original_testbench_code"]

    # The new top-level module is the wrapper
    new_top_level = state['top_level_module']
    original_top_module = state.get("top_level_module").replace("_with_serial", "")


    prompt = f"""
    You are an expert Verilog testbench writer. The design under test has been integrated into a new top-level module with a serial interface to reduce its pin count. You MUST update the testbench to communicate using this new serial protocol.

    **New Design Modules (including the serial interface and new top-level):**
    ---
    """
    for filename, code in state['decomposed_files'].items():
        prompt += f"--- {filename} ---\n{code}\n"

    prompt += f"""
    ---
    **Original Testbench Code (for the parallel version):**
    ---
    {tb_to_correct}
    ---

    **CRITICAL INSTRUCTIONS FOR THE NEW TESTBENCH:**
    1.  **Instantiate the New Top-Level:** The testbench must now instantiate the new top-level module, which is `{new_top_level}` (e.g., `{original_top_module}_with_serial`). This module contains both the serial interface and the original design.
    2.  **Implement Serial Communication:**
        -   You must drive the serial interface pins (`sclk`, `cs`, `mosi`, etc.) correctly.
        -   To send data to the DUT, create a task or loop that drives the `mosi` pin one bit at a time on each clock edge while `cs` is active.
        -   After sending all bits, allow time for the internal core logic to compute.
        -   To receive data, create a similar loop to capture the `miso` pin's value on each clock edge.
    3.  **Maintain Core Test Logic:** The fundamental test (the data you send and the expected result) remains the same as the original testbench. You are only changing the *method* of communication from parallel to serial at the boundary of the new top-level module.
    4.  **Verify Correctness:** After receiving the full serial output, reconstruct the parallel data and compare it with the expected result.
    5.  **Self-Checking:** The testbench MUST use `$display` to print **EXACTLY** `"Result: PASSED"` or `"Result: FAILED"` and then call `$finish`.

    Provide the updated, complete, and robust self-checking testbench for the **new serially-integrated design** in a single Verilog code block.
    """

    st.write("🤖 Asking Gemini to update the testbench for the new serial interface...")
    response = llm.invoke(prompt)
    st.write("#### Gemini's Response:")
    st.markdown(response.content)

    modified_code = re.search(r"```(?:verilog)?\s*\n(.*?)```", response.content, re.DOTALL)
    if not modified_code:
        st.error("Could not extract corrected testbench code from LLM response.")
        return {"modified_testbench_code": tb_to_correct}

    corrected_tb_code = modified_code.group(1).strip()

    st.write("#### Testbench Changes (Serial vs. Parallel):")
    diff = difflib.unified_diff(
        tb_to_correct.splitlines(keepends=True),
        corrected_tb_code.splitlines(keepends=True),
        fromfile='original_parallel_tb', tofile='modified_serial_tb',
    )
    st.code(''.join(diff), language='diff')

    return {"modified_testbench_code": corrected_tb_code}

def pin_reduction_decomposer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Decomposes the pin-reduced Verilog code.
    This is a wrapper around the main decomposer agent.
    """
    st.write("---")
    st.write("### 🧩 Agent 29: Pin Reduction Code Decomposer")
    st.info("Splitting the pin-reduced Verilog back into separate files.")
    return code_decomposer_agent(state)

def pin_reduction_saver_agent(state: AgentState) -> Dict[str, Any]:
    """
    Saves the pin-reduced Verilog and new testbench.
    This is a wrapper around the main file saver agent.
    """
    st.write("---")
    st.write("### 💾 Agent 30: Pin Reduction File Saver")
    st.info("Saving the pin-reduced Verilog and new testbench to a new versioned directory.")
    return file_saver_agent(state)


def render_step_image(state: AgentState, state_key_in: str, caption: str):
    """
    A utility agent that generates a PNG image of the chip layout at various stages.
    """
    st.write(f"#### 🖼️ Visualizing: {caption}")
    Render = Step.factory.get("KLayout.Render")
    # Use a try-except block to prevent rendering failures from halting the flow
    try:
        render_step = Render(config=state["config"], state_in=state[state_key_in])
        render_step.start()
        image_path = os.path.join(render_step.step_dir, "out.png")
        if os.path.exists(image_path):
            st.image(image_path, caption=caption, width=400)
        else:
            st.warning(f"Could not render image for {caption}")
    except Exception as e:
        st.warning(f"Image rendering failed for {caption}: {e}")
    return {}

# --- Conditional Logic for Graph Branching ---

def check_simulation_exit_code(state: AgentState) -> str:
    """
    Checks if the simulation process itself ran without crashing.
    """
    if state["simulation_passed"]:
        return "verify_simulation_output" # Proceed to check the content of the output
    else:
        st.error("❌ Simulation process failed to execute.")
        feedback = state.get("feedback_log", []) + [f"Icarus simulation failed to compile or run. Fix the Verilog code:\n{state['simulation_output']}"]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 5:
            st.error("Simulation failed after multiple correction attempts. Halting.")
            return "end"
        st.warning("Looping back to Verilog Corrector for another attempt.")
        return "fix_verilog"

def check_simulation_results(state: AgentState) -> str:
    """
    Checks if the testbench reported a 'PASSED' status.
    """
    if state["simulation_verified"]:
        return "continue_to_synthesis" # Success, move to the PnR flow
    else:
        st.error("❌ Testbench reported failure.")
        feedback = state.get("feedback_log", []) + [f"Testbench did not report PASSED or SUCCESS. The design may have functional bugs. Please fix the Verilog code based on the simulation output:\n{state['simulation_output']}"]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 5:
            st.error("Simulation verification failed after multiple correction attempts. Halting.")
            return "end"
        st.warning("Looping back to Verilog Corrector for another attempt.")
        return "fix_verilog"

def check_floorplan(state: AgentState) -> str:
    """
    Checks if the die size from floorplanning is within the user-defined constraints.
    """
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
        return "fix_verilog" # Go back to the Verilog corrector to reduce area

def check_sta_violations(state: AgentState) -> str:
    """
    Checks if the design has any timing violations (negative slack).
    """
    worst_tns = state.get("worst_tns", 0.0)
    worst_wns = state.get("worst_wns", 0.0)
    if worst_tns < 0 or worst_wns < 0:
        st.error(f"❌ STA VIOLATION (TNS={worst_tns:.2f} ns, WNS={worst_wns:.2f} ns).")
        if state.get("update_attempt", 0) > 15:
                        st.error("Could not meet timing after multiple attempts. Halting.")
                        return "end"
        return "fix_sta" # Go to the STA corrector agent
    else:
        st.success(f"✅ Timing constraints met. Proceeding to final signoff.")
        return "continue_to_signoff"

def check_lvs_results(state: AgentState) -> str:
    """
    Checks if the LVS verifier agent passed.
    """
    if state["lvs_passed"]:
        st.success("✅ LVS passed. Proceeding to pin count check.")
        return "continue_to_pin_count"
    else:
        st.error("❌ LVS check failed. The layout does not match the schematic.")
        feedback = state.get("feedback_log", []) + ["LVS check failed. This is a critical error indicating a problem with the physical design tools or the Verilog code that is causing a mismatch. Attempting to fix the Verilog code."]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 18:
                        st.error("LVS failed after multiple correction attempts. Halting.")
                        return "end"
        st.warning("Looping back to Verilog Corrector to attempt a fix.")
        return "fix_verilog"

def check_pin_count(state: AgentState) -> str:
    """
    Checks if the final pin count is within the user-defined constraints.
    """
    if state["pin_count"] < 0: # Error case from the pin counter agent
        st.error("Halting due to pin counting error.")
        return "end"
    if state["pin_count"] <= state["max_pins"]:
        st.success("✅ Pin count is within limits. Flow complete!")
        return "end" # Final success
    else:
        st.error(f"❌ Pin count ({state['pin_count']}) exceeds maximum of {state['max_pins']}.")
        feedback = state.get("feedback_log", []) + [f"LVS passed, but pin count {state['pin_count']} exceeds limit of {state['max_pins']}. You must reduce the number of I/O ports using serialization or other techniques."]
        state['feedback_log'] = feedback
        if state.get("update_attempt", 0) > 20:
                        st.error("Could not meet pin count constraint after multiple attempts. Halting.")
                        return "end"
        st.warning("Looping back to Verilog Pin Reduction Corrector.")
        return "fix_pins" # Go to the specialized pin reduction loop

# --- Build the StateGraph ---
workflow = StateGraph(AgentState)

# Add Nodes for all agents
node_definitions = {
    "file_processing": file_processing_agent,
    "verilog_corrector": verilog_corrector_agent,
    "code_decomposer": code_decomposer_agent,
    "design_name_updater": design_name_updater_agent,
    "testbench_corrector": testbench_corrector_agent,
    "file_saver": file_saver_agent,
    "icarus_simulation": icarus_simulation_agent,
    "simulation_verifier": simulation_verifier_agent,
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
    "lvs_verifier": lvs_verifier_agent,
    "pin_counter": pin_counter_agent,
    "pin_reduction_corrector": pin_reduction_corrector_agent,
    "pin_reduction_decomposer": pin_reduction_decomposer_agent,
    # FIX: Add a uniquely named node that calls the same function to resolve ambiguity
    "pin_reduction_design_name_updater": design_name_updater_agent,
    "pin_reduction_testbench": pin_reduction_testbench_agent,
    "pin_reduction_saver": pin_reduction_saver_agent,
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

# --- Define Edges and Graph Structure ---

# 1. Initial File Processing and Simulation
workflow.add_edge(START, "file_processing")
workflow.add_edge("file_processing", "icarus_simulation")

# 2. Simulation Check and Correction Loop
workflow.add_conditional_edges(
    "icarus_simulation",
    check_simulation_exit_code,
    {"verify_simulation_output": "simulation_verifier", "fix_verilog": "verilog_corrector", "end": END},
)
workflow.add_conditional_edges(
    "simulation_verifier",
    check_simulation_results,
    {"continue_to_synthesis": "setup", "fix_verilog": "verilog_corrector", "end": END},
)

# Define the standard Verilog correction flow
workflow.add_edge("verilog_corrector", "code_decomposer")
workflow.add_edge("code_decomposer", "design_name_updater")
workflow.add_edge("design_name_updater", "testbench_corrector")
workflow.add_edge("testbench_corrector", "file_saver")
workflow.add_edge("file_saver", "icarus_simulation") # Loop back to re-simulate

# 3. Main PnR Flow
workflow.add_edge("setup", "synthesis")
workflow.add_edge("synthesis", "floorplan")
workflow.add_edge("floorplan", "render_floorplan")

# 4. Floorplan (Area) Check and Correction Loop
workflow.add_conditional_edges("render_floorplan", check_floorplan,
    {"continue_to_pnr": "tap_endcap", "fix_verilog": "verilog_corrector", "end": END})

# Chain together the PnR and rendering steps for a linear flow
pnr_chain = ["tap_endcap", "render_tap_endcap", "io_placement", "render_io", "generate_pdn", "render_pdn",
             "global_placement", "render_global_placement", "detailed_placement", "render_detailed_placement",
             "cts", "render_cts", "global_routing", "detailed_routing", "render_routing"]
for i in range(len(pnr_chain) - 1):
    workflow.add_edge(pnr_chain[i], pnr_chain[i+1])

workflow.add_edge("render_routing", "fill_insertion")
workflow.add_edge("fill_insertion", "render_fill")
workflow.add_edge("render_fill", "rcx")
workflow.add_edge("rcx", "sta")

# 5. STA (Timing) Check and Correction Loop
workflow.add_conditional_edges("sta", check_sta_violations,
    {"continue_to_signoff": "stream_out", "fix_sta": "sta_correction", "end": END})
workflow.add_edge("sta_correction", "setup") # Loop back to re-run PnR with new clock period

# 6. Final Signoff Flow
signoff_chain = ["stream_out", "render_gds", "drc", "spice_extraction", "lvs"]
for i in range(len(signoff_chain) - 1):
    workflow.add_edge(signoff_chain[i], signoff_chain[i+1])

# 7. LVS Check and Correction Loop
workflow.add_edge("lvs", "lvs_verifier")
workflow.add_conditional_edges("lvs_verifier", check_lvs_results,
    {"continue_to_pin_count": "pin_counter", "fix_verilog": "verilog_corrector", "end": END})

# 8. Pin Count Check and Correction Loop
workflow.add_conditional_edges("pin_counter", check_pin_count,
    {"fix_pins": "pin_reduction_corrector", "end": END})

# Define the pin reduction correction flow
workflow.add_edge("pin_reduction_corrector", "pin_reduction_decomposer")
# FIX: Use the new, uniquely named node to create a deterministic path
workflow.add_edge("pin_reduction_decomposer", "pin_reduction_design_name_updater")
workflow.add_edge("pin_reduction_design_name_updater", "pin_reduction_testbench")
workflow.add_edge("pin_reduction_testbench", "pin_reduction_saver")
workflow.add_edge("pin_reduction_saver", "icarus_simulation") # Loop all the way back to re-verify

# Compile the graph
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
                        # Invoke the graph with a high recursion limit to allow for many loops
                        app.invoke(initial_state, {"recursion_limit": 200})
                    st.success("✅ Agentic flow completed!")
                except Exception as e:
                    st.error(f"An error occurred during the flow: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    # Restore the original working directory
                    os.chdir(original_cwd)
                    st.write(f"✅ Restored working directory to: `{os.getcwd()}`")

# --- Detailed Workflow Graph Visualization ---
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
        simulation_verifier [label="6.5 Simulation Verifier", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }

    subgraph cluster_correction_verilog {
        label="A. Area/Sim/LVS Correction Loop"; style="rounded,filled"; color="#ffebee"; node[fillcolor="#ffcdd2"];
        verilog_corrector [label="2. Verilog Corrector"];
        code_decomposer [label="3. Code Decomposer"];
        design_name_updater [label="3.5. Design Name Updater"];
        testbench_corrector [label="4. Testbench Corrector"];
        file_saver [label="5. File Saver"];
    }

    subgraph cluster_pnr {
        label="2. Physical Design (PnR) & Timing"; style="rounded,filled"; color="#e8f5e9"; node[fillcolor="#c8e6c9"];
        setup [label="7. OpenLane Setup"];
        synthesis [label="8. Synthesis"];
        floorplan [label="9. Floorplan", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
        pnr_group [label="PnR & Vis Steps (10-19)"];
        sta [label="20. STA", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
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
        lvs_verifier [label="25.5 LVS Verifier", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
        pin_counter [label="26. Pin Counter", shape=diamond, style="rounded,filled", fillcolor="#fff9c4"];
    }

    subgraph cluster_correction_pins {
        label="C. Pin Reduction Loop"; style="rounded,filled"; color="#dcedc8"; node[fillcolor="#c5e1a5"];
        pin_reduction_corrector [label="27. Pin Reduction Corrector"];
        pin_reduction_decomposer [label="28. Decomposer"];
        pin_reduction_testbench [label="29. Testbench Corrector"];
        pin_reduction_saver [label="30. File Saver"];
    }

    end_node [label="Flow Complete", shape=ellipse, style=filled, fillcolor="#b2dfdb"];

    // Main Flow
    file_processing -> icarus_simulation;
    icarus_simulation -> simulation_verifier [label="Sim OK", color=darkgreen];
    simulation_verifier -> setup [label="Verify OK", color=darkgreen];
    setup -> synthesis -> floorplan;
    floorplan -> pnr_group [label="Area OK", color=darkgreen];
    pnr_group -> sta;
    sta -> stream_out [label="Timing OK", color=darkgreen];
    stream_out -> drc -> spice_extraction -> lvs -> lvs_verifier;
    lvs_verifier -> pin_counter [label="LVS OK", color=darkgreen];
    pin_counter -> end_node [label="Pins OK", color=darkgreen, style=bold];

    // Correction Loops
    icarus_simulation -> verilog_corrector [label="Sim FAIL", style=dashed, color=red, fontcolor=red, constraint=false];
    simulation_verifier -> verilog_corrector [label="Verify FAIL", style=dashed, color=red, fontcolor=red, constraint=false];
    floorplan -> verilog_corrector [label="Area TOO BIG", style=dashed, color=red, fontcolor=red, constraint=false];
    lvs_verifier -> verilog_corrector [label="LVS FAIL", style=dashed, color=red, fontcolor=red, constraint=false];
    verilog_corrector -> code_decomposer -> design_name_updater -> testbench_corrector -> file_saver -> icarus_simulation [style=dashed, color=red];

    sta -> sta_correction [label="Timing FAIL", style=dashed, color=blue, fontcolor=blue, constraint=false];
    sta_correction -> setup [style=dashed, color=blue, label="Re-run PnR"];

    pin_counter -> pin_reduction_corrector [label="Too Many Pins", style=dashed, color="#E65100", fontcolor="#E65100", constraint=false];
    pin_reduction_corrector -> pin_reduction_decomposer -> design_name_updater -> pin_reduction_testbench -> pin_reduction_saver -> icarus_simulation [style=dashed, color="#E65100", label="Re-verify new I/O"];
}
""")

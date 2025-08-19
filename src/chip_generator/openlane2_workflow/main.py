import streamlit as st
import os
import json
import pandas as pd
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, List, Dict, Any
import shutil
from openlane.state import State
from openlane.steps import Step
from openlane.config import Config
from pathlib import Path
import re 

# --- Agentic Workflow using LangGraph ---

class AgentState(TypedDict):
    uploaded_files: List[Any]
    top_level_module: str
    design_name: str
    verilog_files: List[str]
    config: Dict[str, Any]
    run_path: str
    synthesis_state_out: State
    floorplan_state_out: State
    tap_endcap_state_out: State
    io_placement_state_out: State
    pdn_state_out: State
    global_placement_state_out: State
    detailed_placement_state_out: State
    cts_state_out: State
    global_routing_state_out: State
    detailed_routing_state_out: State
    fill_insertion_state_out: State
    rcx_state_out: State
    sta_state_out: State
    stream_out_state_out: State
    drc_state_out: State
    spice_extraction_state_out: State
    lvs_state_out: State


# Agent 1: File Processing and Setup
def file_processing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📂 Agent 1: File Processing")
    uploaded_files = state["uploaded_files"]
    top_level_module = state["top_level_module"]
    design_name = top_level_module
    
    run_path = os.path.abspath(os.path.join("..", "..", "examples", "generated_chips", f"generated_{design_name}"))
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path, exist_ok=True)

    src_dir = os.path.join(run_path, "src")
    os.makedirs(src_dir, exist_ok=True)
    verilog_files = []
    for file in uploaded_files:
        file_path = os.path.join(src_dir, file.name)
        with open(file_path, "wb") as f: f.write(file.getbuffer())
        if file.name.endswith((".v", ".vh")): verilog_files.append(file_path)

    st.write(f"✅ Top-level module '{top_level_module}' selected.")
    st.write(f"✅ Verilog files saved in: `{src_dir}`")
    
    os.chdir(run_path)
    st.write(f"✅ Changed working directory to: `{os.getcwd()}`")

    return {
        "design_name": design_name,
        "verilog_files": [os.path.relpath(p, os.getcwd()) for p in verilog_files],
        "run_path": os.getcwd(),
    }

# Agent 2: OpenLane Setup
def setup_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🛠️ Agent 2: OpenLane Setup")
    config = Config.interactive(
        state["design_name"],
        PDK="gf180mcuC",
        CLOCK_PORT="clk", CLOCK_NET="clk", CLOCK_PERIOD=10,
        PRIMARY_GDSII_STREAMOUT_TOOL="klayout",
    )
    st.write("✅ OpenLane configuration created successfully.")
    return {"config": config}


# Physical Step Agents ...
def synthesis_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔬 Agent 3: Synthesis")
    st.write("""Converting high-level Verilog to a netlist of standard cells.""")
    Synthesis = Step.factory.get("Yosys.Synthesis")
    synthesis_step = Synthesis(config=state["config"], state_in=State(), VERILOG_FILES=state["verilog_files"])
    synthesis_step.start()
    report_path = os.path.join(synthesis_step.step_dir, "reports", "stat.json")
    with open(report_path) as f: metrics = json.load(f)
    st.write("#### Synthesis Metrics")
    st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=["Value"]).astype(str))
    return {"synthesis_state_out": synthesis_step.state_out}

def floorplan_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🏗️ Agent 4: Floorplanning")
    st.write("""Determining the chip's dimensions and creating the cell placement grid.""")
    Floorplan = Step.factory.get("OpenROAD.Floorplan")
    floorplan_step = Floorplan(config=state["config"], state_in=state["synthesis_state_out"])
    floorplan_step.start()
    metrics_path = os.path.join(floorplan_step.step_dir, "or_metrics_out.json")
    with open(metrics_path) as f: metrics = json.load(f)
    st.write("#### Floorplan Metrics")
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).astype(str))
    return {"floorplan_state_out": floorplan_step.state_out}

def tap_endcap_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💠 Agent 5: Tap/Endcap Insertion")
    st.write("""Placing tap and endcap cells for power stability.""")
    TapEndcap = Step.factory.get("OpenROAD.TapEndcapInsertion")
    tap_step = TapEndcap(config=state["config"], state_in=state["floorplan_state_out"])
    tap_step.start()
    return {"tap_endcap_state_out": tap_step.state_out}

def io_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📍 Agent 6: I/O Pin Placement")
    st.write("Placing I/O pins at the edges of the design.")
    IOPlacement = Step.factory.get("OpenROAD.IOPlacement")
    ioplace_step = IOPlacement(config=state["config"], state_in=state["tap_endcap_state_out"])
    ioplace_step.start()
    return {"io_placement_state_out": ioplace_step.state_out}

def generate_pdn_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ⚡ Agent 7: Power Distribution Network (PDN) Generation")
    st.write("""Creating the metal grid for power and ground.""")
    GeneratePDN = Step.factory.get("OpenROAD.GeneratePDN")
    pdn_step = GeneratePDN(config=state["config"], state_in=state["io_placement_state_out"], FP_PDN_VWIDTH=2, FP_PDN_HWIDTH=2, FP_PDN_VPITCH=30, FP_PDN_HPITCH=30)
    pdn_step.start()
    return {"pdn_state_out": pdn_step.state_out}

def global_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌍 Agent 8: Global Placement")
    st.write("""Finding an approximate location for all standard cells.""")
    GlobalPlacement = Step.factory.get("OpenROAD.GlobalPlacement")
    gpl_step = GlobalPlacement(config=state["config"], state_in=state["pdn_state_out"])
    gpl_step.start()
    return {"global_placement_state_out": gpl_step.state_out}

def detailed_placement_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 📐 Agent 9: Detailed Placement")
    st.write("""Snapping cells to the legal manufacturing grid.""")
    DetailedPlacement = Step.factory.get("OpenROAD.DetailedPlacement")
    dpl_step = DetailedPlacement(config=state["config"], state_in=state["global_placement_state_out"])
    dpl_step.start()
    return {"detailed_placement_state_out": dpl_step.state_out}

def cts_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🌳 Agent 10: Clock Tree Synthesis (CTS)")
    st.write("""Building the clock distribution network.""")
    CTS = Step.factory.get("OpenROAD.CTS")
    cts_step = CTS(config=state["config"], state_in=state["detailed_placement_state_out"])
    cts_step.start()
    return {"cts_state_out": cts_step.state_out}

def global_routing_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🗺️ Agent 11: Global Routing")
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
    st.write("### ✍️ Agent 12: Detailed Routing")
    st.write("""Creating the final physical wires on the metal layers.""")
    DetailedRouting = Step.factory.get("OpenROAD.DetailedRouting")
    drt_step = DetailedRouting(config=state["config"], state_in=state["global_routing_state_out"])
    drt_step.start()
    return {"detailed_routing_state_out": drt_step.state_out}

def fill_insertion_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🧱 Agent 13: Fill Insertion")
    st.write("""Filling empty gaps in the design with 'fill cells' for manufacturability.""")
    FillInsertion = Step.factory.get("OpenROAD.FillInsertion")
    fill_step = FillInsertion(config=state["config"], state_in=state["detailed_routing_state_out"])
    fill_step.start()
    return {"fill_insertion_state_out": fill_step.state_out}

def rcx_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 🔌 Agent 14: Parasitics Extraction (RCX)")
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
    st.write("### ⏱️ Agent 15: Static Timing Analysis (STA)")
    st.write("""This final analysis step verifies that the chip meets its timing constraints to run at the rated clock speed.""")
    STAPostPNR = Step.factory.get("OpenROAD.STAPostPNR")
    sta_step = STAPostPNR(config=state["config"], state_in=state["rcx_state_out"])
    sta_step.start()
    st.write("#### STA Timing Violation Summary")
    sta_results = []
    value_re = re.compile(r":\s*(-?[\d\.]+)")
    reports_to_find = ["tns.max.rpt", "tns.min.rpt", "wns.max.rpt", "wns.min.rpt", "ws.max.rpt", "ws.min.rpt"]
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
        st.warning("Could not parse key STA report files (TNS, WNS, WS).")
    return {"sta_state_out": sta_step.state_out}

def stream_out_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### 💾 Agent 16: GDSII Stream Out")
    st.write("This step converts the final layout into GDSII format, the file that is sent to the foundry for fabrication.")
    StreamOut = Step.factory.get("KLayout.StreamOut")
    gds_step = StreamOut(config=state["config"], state_in=state["sta_state_out"])
    gds_step.start()
    return {"stream_out_state_out": gds_step.state_out}

def drc_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ✅ Agent 17: Design Rule Check (DRC)")
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
    st.write("### ⚡ Agent 18: SPICE Extraction")
    st.write("Extracts a SPICE netlist from the final GDSII layout. This is needed for the LVS check.")
    SpiceExtraction = Step.factory.get("Magic.SpiceExtraction")
    spx_step = SpiceExtraction(config=state["config"], state_in=state["drc_state_out"])
    spx_step.start()
    return {"spice_extraction_state_out": spx_step.state_out}

def lvs_agent(state: AgentState) -> Dict[str, Any]:
    st.write("---")
    st.write("### ↔️ Agent 19: Layout vs. Schematic (LVS)")
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

# RENDER AGENT (Generic)
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

# Build the graph
workflow = StateGraph(AgentState)
nodes = [
    ("file_processing", file_processing_agent), ("setup", setup_agent),
    ("synthesis", synthesis_agent), ("floorplan", floorplan_agent),
    ("render_floorplan", lambda s: render_step_image(s, "floorplan_state_out", "Floorplan")),
    ("tap_endcap", tap_endcap_agent),
    ("render_tap_endcap", lambda s: render_step_image(s, "tap_endcap_state_out", "Tap/Endcap Insertion")),
    ("io_placement", io_placement_agent),
    ("render_io", lambda s: render_step_image(s, "io_placement_state_out", "I/O Placement")),
    ("generate_pdn", generate_pdn_agent),
    ("render_pdn", lambda s: render_step_image(s, "pdn_state_out", "PDN")),
    ("global_placement", global_placement_agent),
    ("render_global_placement", lambda s: render_step_image(s, "global_placement_state_out", "Global Placement")),
    ("detailed_placement", detailed_placement_agent),
    ("render_detailed_placement", lambda s: render_step_image(s, "detailed_placement_state_out", "Detailed Placement")),
    ("cts", cts_agent),
    ("render_cts", lambda s: render_step_image(s, "cts_state_out", "Clock Tree Synthesis")),
    ("global_routing", global_routing_agent),
    ("detailed_routing", detailed_routing_agent),
    ("render_detailed_routing", lambda s: render_step_image(s, "detailed_routing_state_out", "Detailed Routing")),
    ("fill_insertion", fill_insertion_agent),
    ("render_fill", lambda s: render_step_image(s, "fill_insertion_state_out", "Fill Insertion")),
    ("rcx", rcx_agent),
    ("sta", sta_agent),
    ("stream_out", stream_out_agent),
    ("render_gds", lambda s: render_step_image(s, "stream_out_state_out", "Final GDSII Layout")),
    ("drc", drc_agent),
    ("spice_extraction", spice_extraction_agent),
    ("lvs", lvs_agent)
]
for name, node in nodes:
    workflow.add_node(name, node)

# Define the sequential flow
chain = [
    "file_processing", "setup", "synthesis", "floorplan", "render_floorplan",
    "tap_endcap", "render_tap_endcap", "io_placement", "render_io",
    "generate_pdn", "render_pdn", "global_placement", "render_global_placement",
    "detailed_placement", "render_detailed_placement", "cts", "render_cts",
    "global_routing", "detailed_routing", "render_detailed_routing",
    "fill_insertion", "render_fill", "rcx", "sta",
    "stream_out", "render_gds", "drc", "spice_extraction", "lvs"
]
workflow.add_edge(START, chain[0])
for i in range(len(chain) - 1):
    workflow.add_edge(chain[i], chain[i+1])
workflow.add_edge(chain[-1], END)

app = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 LLM for Chip Design Automation")
st.write("Welcome! This application uses an agentic AI to guide you through the full ASIC PnR flow using OpenLane 2.")

st.write("### Agentic Workflow Graph")
st.graphviz_chart(
    """
    digraph {
        graph [splines=ortho, nodesep=0.4, ranksep=0.8];
        node [shape=box, style="rounded,filled", fillcolor="#a9def9", width=2, height=0.5, fontsize=10];
        edge [color="#555555", arrowhead=vee];

        // Define Ranks for horizontal layout
        { rank=same; prep_1; prep_2; prep_3; }
        { rank=same; fp_1; fp_2; fp_3; fp_4;}
        { rank=same; place_1; place_2; place_3;}
        { rank=same; route_1; route_2; route_3;}
        { rank=same; signoff_1; signoff_2; signoff_3; signoff_4; signoff_5;}

        // Nodes
        prep_1 [label="File Processing"];
        prep_2 [label="Setup"];
        prep_3 [label="Synthesis"];
        
        fp_1 [label="Floorplanning"];
        fp_2 [label="Tap/Endcap"];
        fp_3 [label="I/O Placement"];
        fp_4 [label="PDN"];
        
        place_1 [label="Global Placement"];
        place_2 [label="Detailed Placement"];
        place_3 [label="CTS"];
        
        route_1 [label="Global Routing"];
        route_2 [label="Detailed Routing"];
        route_3 [label="Fill Insertion"];

        signoff_1 [label="Stream Out (GDS)"];
        signoff_2 [label="RCX"];
        signoff_3 [label="STA"];
        signoff_4 [label="DRC"];
        signoff_5 [label="LVS"];

        // Edges
        prep_1 -> prep_2 -> prep_3 -> fp_1;
        fp_1 -> fp_2 -> fp_3 -> fp_4 -> place_1;
        place_1 -> place_2 -> place_3 -> route_1;
        route_1 -> route_2 -> route_3 -> signoff_1;
        signoff_1 -> signoff_2 -> signoff_3 -> signoff_4 -> signoff_5;
    }
    """
)

st.sidebar.header("1. Upload Your Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload your Verilog files (.v, .vh)", accept_multiple_files=True
)

if uploaded_files:
    verilog_file_names = [f.name for f in uploaded_files if f.name.endswith(".v")]
    top_level_module = st.sidebar.selectbox(
        "Select the top-level module",
        options=[name.replace(".v", "") for name in verilog_file_names],
    )

    if st.sidebar.button("🚀 Run Agentic Flow"):
        original_cwd = os.getcwd()
        try:
            with st.spinner("🚀 Agents at work... This flow is long and will take several minutes."):
                initial_state = { "uploaded_files": uploaded_files, "top_level_module": top_level_module }
                app.invoke(initial_state, {"recursion_limit": 100})
            st.success("✅ Agentic flow completed successfully!")
        except Exception as e:
            st.error(f"An error occurred during the flow: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            os.chdir(original_cwd)
            st.write(f"✅ Restored working directory to: `{os.getcwd()}`")
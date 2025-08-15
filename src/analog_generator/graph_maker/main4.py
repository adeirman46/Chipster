import streamlit as st
import os
import networkx as nx
import matplotlib.pyplot as plt
import traceback
import numpy as np
from io import BytesIO
import json
from typing import TypedDict, List, Dict
from collections import defaultdict
from lcapy import Circuit
import graphviz

# LangChain and Gemini specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# --- State Definition for the Graph ---
class AppState(TypedDict, total=False):
    pyspice_code: str
    analysis_results: str
    components: List[Dict]
    graph: nx.Graph
    json_output: Dict
    json_file_path: str
    lcapy_string: str
    lcapy_circuit: Circuit
    graph_image: BytesIO
    schematic_image: BytesIO
    pyspice_retriever: Runnable
    lcapy_retriever: Runnable
    error_message: str

# --- Agent Definitions ---

@st.cache_resource
def get_retrievers():
    """Loads and caches the FAISS vector stores to avoid reloading."""
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # IMPORTANT: Update these paths to the correct location of your FAISS indexes
    pyspice_faiss_path = "../../../notebooks/faiss_index_pyspice_docs"
    lcapy_faiss_path = "../../../notebooks/faiss_index_lcapy_docs"

    if not os.path.exists(pyspice_faiss_path):
        st.error(f"FAISS index for PySpice not found at: {pyspice_faiss_path}")
        return None, None
    if not os.path.exists(lcapy_faiss_path):
        st.error(f"FAISS index for Lcapy not found at: {lcapy_faiss_path}")
        return None, None

    try:
        pyspice_vectorstore = FAISS.load_local(pyspice_faiss_path, embeddings, allow_dangerous_deserialization=True)
        lcapy_vectorstore = FAISS.load_local(lcapy_faiss_path, embeddings, allow_dangerous_deserialization=True)
        return pyspice_vectorstore.as_retriever(), lcapy_vectorstore.as_retriever()
    except Exception as e:
        st.error(f"Failed to load FAISS indexes: {e}")
        return None, None

def setup_agent(state: AppState) -> AppState:
    """Loads FAISS retrievers into the state."""
    st.write("🛠️ Setting up resources...")
    pyspice_retriever, lcapy_retriever = get_retrievers()
    if pyspice_retriever is None or lcapy_retriever is None:
        return {"error_message": "Failed to load FAISS indexes. Workflow cannot continue."}
    return {"pyspice_retriever": pyspice_retriever, "lcapy_retriever": lcapy_retriever}

@st.cache_resource
def generate_workflow_graph_image():
    """Generates and caches a visualization of the agent workflow."""
    dot_string = """
    digraph G {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fillcolor="#ddeeff", fontname="Helvetica", fontsize=10];
        edge [fontname="Helvetica", color="#4a4a4a", fontsize=9];

        start [label="PySpice\\nCode", shape=invhouse, fillcolor="#ffddc1"];
        setup [label="Setup\\n(Load FAISS)"];
        pyspice_rag [label="PySpice\\nRAG"];
        parser [label="Parse\\nComponents"];
        builder [label="Build\\nGraph"];
        json_gen [label="Generate\\nJSON"];
        lcapy_rag [label="Lcapy\\nRAG"];
        schematic_gen [label="Generate\\nSchematic"];
        graph_drawer [label="Draw\\nGraphs"];
        final_output [label="Display\\nResults", shape=ellipse, fillcolor="#d4edda"];

        start -> setup -> pyspice_rag -> parser -> builder -> json_gen -> lcapy_rag -> schematic_gen -> graph_drawer -> final_output;
    }
    """
    try:
        graph = graphviz.Source(dot_string)
        png_data = graph.pipe(format='png')
        return BytesIO(png_data)
    except graphviz.backend.ExecutableNotFound:
        st.warning("Graphviz executable not found. Cannot visualize workflow. Please install Graphviz.")
        return None
    except Exception as e:
        st.error(f"Failed to visualize workflow: {e}")
        return None


def pyspice_rag_agent(state: AppState) -> AppState:
    """Analyzes PySpice code using a pre-loaded RAG chain."""
    st.write("🔬 Analyzing PySpice code with RAG...")
    pyspice_code = state['pyspice_code']
    retriever = state['pyspice_retriever']
    
    try:
        template = """You are a meticulous PySpice code analyzer. Your task is to analyze the provided code and explain the parameters for each component with high accuracy.
**Instructions:**
1.  Identify each line of code that defines a component (`circuit.V`, `circuit.R`, `circuit.I`, `circuit.MOSFET`).
2.  Use the standard PySpice syntax below as a reference template.
3.  For each component instance in the code, list its parameters one-by-one, matching them to the reference syntax. **Do not miss any parameters.**
---
**Reference Syntax Template:**
* `circuit.V(name, n_plus, n_minus, dc_value)`
* `circuit.I(name, n_plus, n_minus, dc_value)`
* `circuit.R(name, node1, node2, resistance_value)`
* `circuit.MOSFET(name, drain, gate, source, body, model='model_name')`
---
**Code Context to Analyze:**
{context}
**Your Detailed Analysis:**
{question}
"""
        prompt = PromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        analysis = rag_chain.invoke(pyspice_code)
        return {"analysis_results": analysis}
    except Exception as e:
        return {"error_message": f"PySpice RAG Agent Error: {e}"}

def component_parser_agent(state: AppState) -> AppState:
    """Parses PySpice code to extract a structured list of components."""
    st.write("🔩 Parsing components from code...")
    pyspice_code = state['pyspice_code']
    components = []
    lines = pyspice_code.split('\n')
    
    for line in lines:
        if ('circuit.' in line or 'self.' in line) and '(' in line:
            try:
                clean_line = line.split('#')[0].strip()
                if not clean_line or clean_line.startswith('#'): continue
                
                parts = clean_line.split('(')
                comp_type_part = parts[0].split('.')[-1]
                args_part = parts[1].split(')')[0]
                
                args = []
                for arg in args_part.split(','):
                    arg = arg.strip().strip("'\"")
                    if '@' in arg: arg = arg.split('@')[0].strip()
                    if '=' in arg: arg = arg.split('=')[1].strip().strip("'\"")
                    args.append(arg)

                terminals, connections = [], []
                if comp_type_part in ["V", "I"] and len(args) >= 4:
                    connections, terminals = args[1:3], ['n+', 'n-']
                elif comp_type_part == "R" and len(args) >= 4:
                    connections, terminals = args[1:3], ['n1', 'n2']
                elif comp_type_part == "MOSFET" and len(args) >= 5:
                    connections, terminals = args[1:5], ['D', 'G', 'S', 'B']

                if connections:
                    cleaned_connections = [c.replace("circuit.gnd", "gnd").replace("self.gnd", "gnd") for c in connections]
                    components.append({
                        "name": args[0], 
                        "type": comp_type_part, 
                        "connections": list(zip(terminals, cleaned_connections))
                    })
            except (IndexError, ValueError):
                continue
    return {"components": components}

def graph_builder_agent(state: AppState) -> AppState:
    """Builds a networkx MultiGraph from the component list."""
    st.write("🕸️ Building connection graph...")
    components = state.get('components', [])
    if not components: return {"graph": None}
    
    g = nx.MultiGraph()
    for comp in components:
        comp_name = f"{comp['type']}_{comp['name']}"
        g.add_node(comp_name, type='component')
        for terminal, conn_net in comp['connections']:
            g.add_node(conn_net, type='net')
            g.add_edge(comp_name, conn_net, label=terminal)
    return {"graph": g}

def json_generator_agent(state: AppState) -> AppState:
    """Generates a JSON representation of the graph."""
    st.write("📄 Generating JSON output...")
    graph = state.get('graph')
    pyspice_code = state.get('pyspice_code')
    
    module_name = "DefaultCircuit"
    for line in pyspice_code.split('\n'):
        if "class" in line and "SubCircuitFactory" in line:
            try:
                module_name = line.split('(')[0].split(' ')[1]
                break
            except IndexError:
                continue

    if graph is None: return {"json_output": None}

    output_dir = f"analog_generation_{module_name}"
    os.makedirs(output_dir, exist_ok=True)
    json_data = nx.node_link_data(graph)
    output_path = os.path.join(output_dir, f"{module_name}.json")
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    return {"json_output": json_data, "json_file_path": output_path}

def lcapy_rag_agent(state: AppState) -> AppState:
    """Generates an lcapy schematic definition using a pre-loaded RAG chain with a one-shot example."""
    st.write("⚡ Converting to Lcapy schematic with RAG...")
    json_rep = json.dumps(state.get('json_output'), indent=2)
    retriever = state['lcapy_retriever']
    
    try:
        # --- ENHANCED PROMPT WITH ONE-SHOT EXAMPLE ---
        template = """You are an expert in `lcapy`. Your task is to convert the provided JSON graph representation into a valid `lcapy` circuit definition string. Use the documentation context and the example below to ensure correct syntax.

---
**Lcapy Documentation Context:**
{context}
---
**Example:**
*JSON Input:*
```json
{{
  "nodes": [
    {{"id": "V_dd", "type": "component"}},
    {{"id": "R_load", "type": "component"}},
    {{"id": "Vdd", "type": "net"}},
    {{"id": "gnd", "type": "net"}},
    {{"id": "Vout", "type": "net"}}
  ],
  "links": [
    {{"source": "V_dd", "target": "Vdd", "label": "n+"}},
    {{"source": "V_dd", "target": "gnd", "label": "n-"}},
    {{"source": "R_load", "target": "Vout", "label": "n1"}},
    {{"source": "R_load", "target": "gnd", "label": "n2"}}
  ]
}}
```
*Correct `lcapy` Circuit Definition Output:*
```
Vdd Vdd gnd; down
Rload Vout gnd; right
```
---
**Your Task:**
Now, convert the following JSON input into an `lcapy` circuit definition.

**JSON Input:**
{question}
---
**Your `lcapy` Circuit Definition:**"""
        prompt = PromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        lcapy_string = rag_chain.invoke(json_rep).strip().replace("```", "")
        return {"lcapy_string": lcapy_string}
    except Exception as e:
        return {"error_message": f"Lcapy RAG Agent Error: {e}"}

def schematic_generator_agent(state: AppState) -> AppState:
    """Generates and draws the lcapy circuit schematic, with robust error handling."""
    st.write("🎨 Drawing schematic...")
    lcapy_string = state.get('lcapy_string')
    
    try:
        cct = Circuit(lcapy_string)
        buf = BytesIO()
        cct.draw(buf=buf, format='png', style='american', draw_nodes='connections')
        buf.seek(0)
        return {"lcapy_circuit": cct, "schematic_image": buf}
    except Exception as e:
        st.error(f"Could not draw Lcapy schematic: {e}")
        st.info("The generated (invalid) lcapy string was:")
        st.code(lcapy_string, language='text')
        # Set error message to prevent crash
        return {"error_message": f"Lcapy drawing failed: {e}"}

def graph_drawer_agent(state: AppState) -> AppState:
    """Draws the networkx connection graph."""
    st.write("🖌️ Drawing connection graph...")
    g = state.get('graph')
    if g is None: return {"graph_image": None}
    
    fig, ax = plt.subplots(figsize=(18, 14))
    pos = nx.spring_layout(g, k=3.5, iterations=120, seed=42)

    component_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'component']
    net_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'net']
    
    nx.draw_networkx_nodes(g, pos, nodelist=component_nodes, node_shape='s', node_color='skyblue', node_size=3500, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_nodes(g, pos, nodelist=net_nodes, node_shape='o', node_color='lightgreen', node_size=2000, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold')
    
    edge_labels = nx.get_edge_attributes(g, 'label')
    nx.draw_networkx_edges(g, pos, width=2, alpha=0.7, edge_color='gray')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='red', font_size=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    
    ax.set_title("Connection Graph", fontsize=20, fontweight='bold')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return {"graph_image": buf}

# --- Conditional Logic for Workflow ---
def should_continue(state: AppState) -> str:
    """Determines the next step based on whether an error has occurred."""
    if state.get("error_message"):
        return "end"
    return "continue"

# --- LangGraph Workflow Setup ---
workflow = StateGraph(AppState)
workflow.add_node("setup", setup_agent)
workflow.add_node("pyspice_rag", pyspice_rag_agent)
workflow.add_node("parser", component_parser_agent)
workflow.add_node("builder", graph_builder_agent)
workflow.add_node("json_generator", json_generator_agent)
workflow.add_node("lcapy_rag", lcapy_rag_agent)
workflow.add_node("schematic_generator", schematic_generator_agent)
workflow.add_node("graph_drawer", graph_drawer_agent)

workflow.set_entry_point("setup")
workflow.add_edge("setup", "pyspice_rag")
workflow.add_edge("pyspice_rag", "parser")
workflow.add_edge("parser", "builder")
workflow.add_edge("builder", "json_generator")
workflow.add_edge("json_generator", "lcapy_rag")
workflow.add_edge("lcapy_rag", "schematic_generator")

# Conditional edge: only proceed to draw if schematic generation was successful
workflow.add_conditional_edges(
    "schematic_generator",
    should_continue,
    {"continue": "graph_drawer", "end": END}
)
workflow.add_edge("graph_drawer", END)

app_graph = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="LLM for Chip Design Automation", layout="wide")
st.title("🤖 Advanced Circuit Visualizer and Analyzer")

st.sidebar.header("About")
st.sidebar.info("This application uses an agentic workflow to analyze, visualize, and generate schematics from PySpice code, powered by FAISS-based RAG agents.")

# Generate and display the workflow graph at the top
workflow_image = generate_workflow_graph_image()
if workflow_image:
    st.header("📊 Agentic Workflow")
    st.image(workflow_image, caption="Visualization of the Agent Workflow", use_container_width=True)
    st.divider()

if 'final_state' not in st.session_state:
    st.session_state.final_state = {}

pyspice_code_input = st.text_area(
    "**Enter your PySpice Sub-Circuit Code Below:**",
    height=400,
    value="""from PySpice.Unit import *
from PySpice.Spice.Netlist import SubCircuitFactory

class CommonDrainAmp(SubCircuitFactory):
    NAME = ('CommonDrainAmp')
    NODES = ('Vin', 'Vout')
    def __init__(self):
        super().__init__()
        self.model('nmos_model', 'nmos', level=1, kp=100e-6, vto=0.5)
        self.V('dd', 'Vdd', self.gnd, 5.0)
        self.MOSFET('1', 'Vdd', 'Vin', 'Vout', self.gnd, model='nmos_model')
        self.R('load', 'Vout', self.gnd, 1@u_kOhm)
"""
)

if st.button("🚀 Generate Full Circuit Analysis", use_container_width=True, type="primary"):
    if pyspice_code_input:
        with st.spinner("Executing agentic workflow... Please wait."):
            initial_state = {"pyspice_code": pyspice_code_input}
            try:
                final_state = app_graph.invoke(initial_state)
                st.session_state.final_state = final_state
                if final_state.get("error_message"):
                    st.error(f"Workflow stopped due to an error: {final_state['error_message']}")
                else:
                    st.success("Workflow completed successfully!")
            except Exception as e:
                st.error(f"A critical error occurred during the workflow: {e}")
                st.error(traceback.format_exc())
    else:
        st.warning("Please enter PySpice code to analyze.")

if st.session_state.final_state:
    final_state = st.session_state.final_state
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("🔗 Connection Graph")
        if final_state.get('graph_image'):
            st.image(final_state['graph_image'], caption="Component and Net Connectivity", use_container_width=True)
        else:
            st.warning("Could not generate the connection graph.")
            
    with col2:
        st.header("⚡ Circuit Schematic (Lcapy)")
        if final_state.get('schematic_image'):
            st.image(final_state['schematic_image'], caption="Generated Circuit Schematic", use_container_width=True)
        else:
            st.warning("Could not generate the schematic.")
            if final_state.get('lcapy_string'):
                st.info("The generated (and possibly invalid) lcapy string was:")
                st.code(final_state.get('lcapy_string'), language='text')

    st.divider()
    
    with st.expander("🔬 View Detailed RAG Analysis & Outputs", expanded=False):
        st.subheader("PySpice RAG Analysis")
        st.markdown(final_state.get('analysis_results', 'No analysis available.'))
        
        st.subheader("Generated JSON")
        st.json(final_state.get('json_output', 'No JSON output.'))

        st.subheader("Generated Lcapy String")
        st.code(final_state.get('lcapy_string', 'No Lcapy string generated.'), language='text')

import streamlit as st
import os
import networkx as nx
import matplotlib.pyplot as plt
import traceback
import numpy as np
from io import BytesIO
from typing import TypedDict, List, Dict
from collections import defaultdict
import re
import subprocess
import sys
import tempfile
import time
import graphviz

# LangChain and Gemini specific imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()


# --- Agent Definitions ---

def pyspice_parser_agent(state):
    """Parses PySpice code to extract components and their named connections."""
    pyspice_code = state['pyspice_code']
    components = []
    lines = pyspice_code.split('\n')
    
    for line in lines:
        # Handle both circuit. and self. formats
        if ('circuit.' in line or 'self.' in line) and '(' in line:
            try:
                clean_line = line.split('#')[0].strip()
                
                if not clean_line or clean_line.startswith('#'):
                    continue
                
                if '(' not in clean_line or ')' not in clean_line:
                    continue
                    
                parts = clean_line.split('(')
                if len(parts) < 2:
                    continue
                    
                comp_type_part = parts[0].split('.')[-1]
                args_part = parts[1].split(')')[0]
                
                args = []
                for arg in args_part.split(','):
                    arg = arg.strip().strip("'\"")
                    if '@' in arg:
                        arg = arg.split('@')[0].strip()
                    if '=' in arg:
                        arg = arg.split('=')[1].strip().strip("'\"")
                    args.append(arg)

                terminals, connections = [], []
                
                if comp_type_part in ["V", "I"]:
                    if len(args) >= 4:
                        connections = args[1:3]
                        terminals = ['n+', 'n-']
                elif comp_type_part == "R":
                    if len(args) >= 4:
                        connections = args[1:3]
                        terminals = ['n1', 'n2']
                elif comp_type_part == "MOSFET":
                    if len(args) >= 5:
                        connections = args[1:5]
                        terminals = ['D', 'G', 'S', 'B']

                if connections:
                    cleaned_connections = [conn.replace("circuit.gnd", "gnd").replace("self.gnd", "gnd") for conn in connections]
                    named_connections = list(zip(terminals, cleaned_connections))
                    components.append({
                        "name": args[0], 
                        "type": comp_type_part, 
                        "connections": named_connections
                    })
                    
            except (IndexError, ValueError):
                continue
            
    return {"components": components}


def graph_builder_agent(state):
    """Builds a networkx MultiGraph to allow for parallel edges."""
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

def lcapy_converter_agent(state):
    """Converts the parsed components to LCapy netlist format with smart node naming."""
    components = state.get('components', [])
    if not components:
        return {"lcapy_netlist": None}

    net_to_node = {}
    node_counter = 0
    if 'gnd' not in net_to_node:
        net_to_node['gnd'] = 0
        node_counter = 1
    
    all_nets = sorted({net for comp in components for _, net in comp['connections']})
    for net in all_nets:
        if net not in net_to_node:
            net_to_node[net] = node_counter
            node_counter += 1

    lcapy_lines = []
    for comp in components:
        comp_name = comp['name']
        comp_type = comp['type']
        
        node_strings = [str(net_to_node[net]) for _, net in comp['connections']]
        connections_str = ' '.join(node_strings)
        
        if comp_type == 'V':
            lcapy_lines.append(f"cct.add('V{comp_name} {connections_str}; down')")
        elif comp_type == 'I':
            lcapy_lines.append(f"cct.add('I{comp_name} {connections_str}; down')")
        elif comp_type == 'R':
            lcapy_lines.append(f"cct.add('R{comp_name} {connections_str}; right')")
        elif comp_type == 'MOSFET':
            lcapy_lines.append(f"cct.add('M{comp_name} {connections_str}; up')")

    return {
        "lcapy_netlist": '\n'.join(lcapy_lines),
        "net_to_node_mapping": net_to_node
    }

def lcapy_corrector_agent(line_to_fix, all_previous_lines, error_str):
    """
    Attempts to correct a failing LCapy netlist line with a node-first strategy.
    """
    st.warning("LCapy error detected. Attempting intelligent correction...")
    
    # --- Strategy 1: Fix Node Connection Conflicts (Primary) ---
    if "already connected" in error_str or "Cannot connect" in error_str:
        # Find all nodes used in previous, successful lines
        node_usage_counts = defaultdict(int)
        for line in all_previous_lines:
            # Find nodes like '1', '0', '3_2', etc. in the connection part of the string
            # This regex captures nodes that are numbers, optionally with a _suffix
            nodes_in_line = re.findall(r'\b(\d+(?:_\d+)?)\b', line.split(';')[0])
            for node in nodes_in_line:
                # We only care about non-MOSFET components for this kind of error
                if not line.strip().startswith("cct.add('M"):
                    node_usage_counts[node] += 1
        
        # Find the nodes in the line that is causing the error
        match = re.search(r"cct\.add\('.*?\s+(.*?);", line_to_fix)
        if match:
            offending_nodes_str = match.group(1)
            # Find nodes in the failing line. We want to find the specific nodes to replace.
            nodes_to_check = re.findall(r'\b\d+\b', offending_nodes_str) 
            
            # Build the corrected connection string part by part
            new_connection_parts = []
            original_connection_parts = offending_nodes_str.split()
            was_corrected = False

            # We need to track which nodes from the failing line we've already suffixed
            suffixed_this_turn = set()

            for part in original_connection_parts:
                # Check if the part is a node that needs suffixing
                if part.isdigit() and part not in suffixed_this_turn:
                    # A node needs suffixing if it's a non-ground node that's already been used
                    if node_usage_counts[part] > 0 and part != '0':
                        # Find the next available suffix (e.g., if 3 is used, next is 3_2)
                        next_suffix = 2
                        while f"{part}_{next_suffix}" in node_usage_counts:
                            next_suffix += 1
                        
                        new_node = f"{part}_{next_suffix}"
                        new_connection_parts.append(new_node)
                        st.info(f"Correction Applied: Node '{part}' is busy. Renaming to '{new_node}'.")
                        
                        # Mark this node as corrected for this line
                        suffixed_this_turn.add(part)
                        node_usage_counts[new_node] += 1 # Immediately add to counts for this context
                        was_corrected = True
                    else:
                        new_connection_parts.append(part) # No correction needed for this node
                else:
                    new_connection_parts.append(part) # Not a node, or already handled

            if was_corrected:
                new_connections_str = ' '.join(new_connection_parts)
                # Replace the whole connection string to be safe
                return line_to_fix.replace(offending_nodes_str, new_connections_str)

    # --- Strategy 2: Change Orientation (Fallback) ---
    st.info("Node correction did not apply. Trying to change component orientation.")
    orientations = ['down', 'up', 'right', 'left']
    for orient in orientations:
        if orient in line_to_fix:
            next_orient = orientations[(orientations.index(orient) + 1) % len(orientations)]
            corrected_line = re.sub(r";\s*\w+", f"; {next_orient}", line_to_fix)
            st.info(f"Fallback Correction: Changed orientation to '{next_orient}'.")
            return corrected_line
            
    return line_to_fix # Return original if no strategy worked

def lcapy_visualizer_agent(state):
    """
    Iteratively builds and corrects an LCapy schematic, yielding images at each step.
    """
    lcapy_netlist = state.get('lcapy_netlist')
    if lcapy_netlist is None:
        return {"schematic_images": [], "final_lcapy_netlist": ""}

    initial_lines = lcapy_netlist.split('\n')
    corrected_lines = list(initial_lines)
    images = []
    
    # Placeholders for the real-time UI updates
    schematic_placeholder = st.empty()
    log_placeholder = st.empty()

    for i in range(len(initial_lines)):
        while True: # Loop indefinitely until the line is drawn successfully
            # Build the script with lines up to the current one
            current_lines_to_run = corrected_lines[:i+1]
            script_to_run = [
                "from lcapy import Circuit",
                "import matplotlib.pyplot as plt",
                "cct = Circuit()",
                *current_lines_to_run,
                "cct.draw(style='american', backend='matplotlib')",
                "plt.savefig('temp_schematic.png', dpi=150, bbox_inches='tight')",
                "plt.close()"
            ]
            
            try:
                # Execute the script in a subprocess for safety
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write('\n'.join(script_to_run))
                    temp_script_path = f.name

                result = subprocess.run(
                    [sys.executable, temp_script_path],
                    capture_output=True, text=True, timeout=20, check=False
                )

                # Check for errors from the subprocess
                if result.returncode != 0 or "Error" in result.stderr or "failed" in result.stderr.lower():
                    raise RuntimeError(result.stderr)
                
                # If successful, read the generated image
                with open('temp_schematic.png', 'rb') as img_file:
                    buf = BytesIO(img_file.read())
                    images.append(buf)
                    
                    # Update the UI in "real-time"
                    schematic_placeholder.image(buf, caption=f"Step {i+1}/{len(initial_lines)}: Drawn `{corrected_lines[i]}`", use_container_width=True)
                    log_placeholder.success(f"Step {i+1} successful.")
                    time.sleep(0.5) # Pause for visualization effect
                
                # Cleanup and break the infinite loop for this line
                os.unlink(temp_script_path)
                os.unlink('temp_schematic.png')
                break 

            except Exception as e:
                error_message = str(e)
                log_placeholder.error(f"Error at step {i+1} on line: `{corrected_lines[i]}`. Retrying...")
                
                # Cleanup any lingering temp files from the failed attempt
                if 'temp_script_path' in locals() and os.path.exists(temp_script_path):
                    os.unlink(temp_script_path)
                if os.path.exists('temp_schematic.png'):
                    os.unlink('temp_schematic.png')

                # Call the corrector agent to get a new, potentially fixed line
                # Pass the successfully drawn lines as context
                successful_previous_lines = corrected_lines[:i]
                corrected_line = lcapy_corrector_agent(corrected_lines[i], successful_previous_lines, error_message)
                corrected_lines[i] = corrected_line
                time.sleep(1) # Pause before retrying

    return {
        "schematic_images": images,
        "final_lcapy_netlist": '\n'.join(corrected_lines)
    }


def graph_drawer_agent(state):
    """Draws the MultiGraph with separate visible edges for each connection."""
    g = state.get('graph')
    if g is None: return {"image": None}
    
    fig, ax = plt.subplots(figsize=(18, 14))
    pos = nx.spring_layout(g, k=2.5, iterations=100, seed=42)

    component_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'component']
    net_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'net']
    
    nx.draw_networkx_nodes(g, pos, nodelist=component_nodes, node_shape='s', node_color='lightblue', node_size=5000, ax=ax, edgecolors='navy', linewidths=2)
    nx.draw_networkx_nodes(g, pos, nodelist=net_nodes, node_shape='o', node_color='lightgreen', node_size=3000, ax=ax, edgecolors='darkgreen', linewidths=2)
    nx.draw_networkx_labels(g, pos, font_size=9, font_weight='bold', ax=ax)

    edge_groups = defaultdict(list)
    for u, v, key in g.edges(keys=True):
        edge_key = tuple(sorted((u, v)))
        edge_groups[edge_key].append((u, v, key))

    for edge_key, edges in edge_groups.items():
        u_base, v_base = edge_key
        pos_u, pos_v = np.array(pos[u_base]), np.array(pos[v_base])
        num_edges = len(edges)
        
        for i, (u, v, key) in enumerate(edges):
            label = g[u][v][key].get('label', '')
            connectionstyle = f"arc3,rad={0.15 + (i // 2) * 0.1 * (-1 if i % 2 else 1)}" if num_edges > 1 else "arc3,rad=0"
            
            nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], width=2, edge_color='gray', alpha=0.8, ax=ax, connectionstyle=connectionstyle)
            
            # Simplified label positioning
            midpoint = (pos_u + pos_v) / 2
            ax.text(midpoint[0], midpoint[1], label, fontsize=9, fontweight='bold', color='red', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='none', alpha=0.7))

    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Component', markerfacecolor='lightblue', markersize=15, markeredgecolor='navy'),
        plt.Line2D([0], [0], marker='o', color='w', label='Net', markerfacecolor='lightgreen', markersize=15, markeredgecolor='darkgreen')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax.set_title("Circuit Graph Visualization", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return {"image": buf}


# --- RAG Agent Setup ---
def create_rag_chain(pyspice_code):
    """Creates a RAG chain to analyze component parameters."""
    docs = [Document(page_content=pyspice_code)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    template = """You are a meticulous PySpice code analyzer. Your task is to analyze the provided code and explain the parameters for each component with high accuracy.
**Instructions:**
1.  Identify each line of code that defines a component (`circuit.V`, `circuit.R`, `circuit.I`, `circuit.MOSFET`).
2.  Use the standard PySpice syntax below as a reference template.
3.  For each component instance in the code, list its parameters one-by-one, matching them to the reference syntax. **Do not miss any parameters.**
4.  **LCapy Note:** LCapy often requires unique nodes for each two-terminal component connected to a net. If multiple resistors connect to node '3', they might need to be defined as connecting to '3', '3_2', '3_3', etc., to draw correctly. This is a common requirement for schematic drawing tools.
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
    return rag_chain

# --- LangGraph Workflow ---
class GraphState(TypedDict, total=False):
    pyspice_code: str
    components: List[dict]
    graph: nx.Graph
    image: BytesIO
    lcapy_netlist: str
    net_to_node_mapping: dict
    schematic_images: List[BytesIO]
    final_lcapy_netlist: str

# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("parser", pyspice_parser_agent)
workflow.add_node("builder", graph_builder_agent)
workflow.add_node("drawer", graph_drawer_agent)
workflow.add_node("lcapy_converter", lcapy_converter_agent)
workflow.add_node("lcapy_visualizer", lcapy_visualizer_agent)

# Set up the edges
workflow.set_entry_point("parser")
workflow.add_edge("parser", "builder")
workflow.add_edge("builder", "drawer")
workflow.add_edge("drawer", "lcapy_converter")
workflow.add_edge("lcapy_converter", "lcapy_visualizer")
workflow.add_edge("lcapy_visualizer", END)

app_graph = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="LLM for Chip Design Automation", layout="wide")
st.title("Circuit Visualizer and Analyzer")

# --- Workflow Visualization ---
st.header("🤖 Agent Workflow Graph")
st.graphviz_chart(
    """
    digraph {
        graph [rankdir="LR", splines=ortho, nodesep=0.5, ranksep=1.0];
        node [shape=box, style="rounded,filled", fillcolor="#e1f5fe", width=3, height=0.8, fontsize=10];
        edge [color="#333333", arrowhead=vee, fontsize=9];

        // Main flow nodes
        parser [label="PySpice Parser Agent\\n(pyspice_parser_agent)"];
        builder [label="Graph Builder Agent\\n(graph_builder_agent)"];
        drawer [label="Graph Drawer Agent\\n(graph_drawer_agent)"];
        lcapy_converter [label="LCapy Netlist Generator\\n(lcapy_converter_agent)"];
        
        // Loop nodes
        lcapy_visualizer [label="LCapy Visualizer\\n(lcapy_visualizer_agent)"];
        lcapy_corrector [label="LCapy Corrector Agent\\n(lcapy_corrector_agent)", fillcolor="#ffcdd2"];

        // Main Edges
        parser -> builder -> drawer -> lcapy_converter -> lcapy_visualizer;
        
        // Correction Loop Edges
        lcapy_visualizer -> lcapy_corrector [label="On Error", style=dashed];
        lcapy_corrector -> lcapy_visualizer [label="Retry", style=dashed];
    }
    """
)


col1, col2 = st.columns(2)

with col1:
    st.header("PySpice Code Input")
    pyspice_code_input = st.text_area(
        "Enter your PySpice code here:",
        height=450,
        value="""from PySpice.Unit import *
from PySpice.Spice.Netlist import SubCircuitFactory

class CommonDrainAmp(SubCircuitFactory):
    NAME = ('CommonDrainAmp')
    NODES = ('Vin', 'Vout')
    def __init__(self):
        super().__init__()
        self.model('nmos_model', 'nmos', level=1, kp=100e-6, vto=0.5)
        self.V('dd', 'Vdd', self.gnd, 5.0)
        # This will cause an error for LCapy initially, which the corrector will fix
        self.R('source', 'Vout', self.gnd, 1@u_kOhm) 
        self.MOSFET('1', 'Vdd', 'Vin', 'Vout', self.gnd, model='nmos_model')
        self.R('load', 'Vout', self.gnd, 2@u_kOhm)
"""
    )
    if st.button("Generate Circuit Visualizations", use_container_width=True):
        with st.spinner("Generating visualizations... This may take a moment."):
            initial_state = {"pyspice_code": pyspice_code_input}
            # We invoke the graph. The visualizer agent will handle UI updates.
            final_state = app_graph.invoke(initial_state)
            
            # Store results in session state for other parts of the UI
            if final_state:
                st.session_state.graph_image = final_state.get('image')
                st.session_state.schematic_images = final_state.get('schematic_images')
                st.session_state.final_lcapy_netlist = final_state.get('final_lcapy_netlist')
                st.session_state.net_mapping = final_state.get('net_to_node_mapping')
                st.session_state.pyspice_code_for_rag = pyspice_code_input
                st.success("Visualizations generated successfully!")
            else:
                st.error("Failed to generate visualizations. Check code for errors.")

with col2:
    st.header("Graph Visualization")
    if 'graph_image' in st.session_state and st.session_state.graph_image:
        st.image(st.session_state.graph_image, caption="Component & Net Graph", use_container_width=True)
    else:
        st.info("Graph will be displayed here once generated.")

# --- Iterative Schematic Display ---
st.divider()
st.header("Schematic Visualization (Iterative Drawing)")
if 'schematic_images' in st.session_state and st.session_state.schematic_images:
    # This section is now primarily handled by the lcapy_visualizer_agent itself
    st.info("The visualization above was built step-by-step. The final corrected schematic is shown.")
    st.image(st.session_state.schematic_images[-1], caption="Final Corrected Circuit Schematic", use_container_width=True)
else:
    st.info("Schematic will be displayed here once generated.")


# LCapy Netlist Display
st.divider()
st.header("🔌 Generated & Corrected LCapy Netlist")
if 'final_lcapy_netlist' in st.session_state and st.session_state.final_lcapy_netlist:
    col_netlist, col_mapping = st.columns([2, 1])
    
    with col_netlist:
        st.code(st.session_state.final_lcapy_netlist, language='python')
    
    with col_mapping:
        st.subheader("Node Mapping")
        if 'net_mapping' in st.session_state:
            mapping_text = ""
            for net, node in sorted(st.session_state.net_mapping.items(), key=lambda x: x[1]):
                mapping_text += f"**{net}** → Node **{node}**\n"
            st.markdown(mapping_text)
else:
    st.info("LCapy netlist will be displayed here once generated.")

st.divider()
st.header("⚙️ Detailed Component Parameter Analysis (RAG)")
if 'pyspice_code_for_rag' in st.session_state and st.session_state.pyspice_code_for_rag:
    if st.button("Analyze Component Parameters", use_container_width=True):
        with st.spinner("Performing detailed analysis..."):
            try:
                rag_chain = create_rag_chain(st.session_state.pyspice_code_for_rag)
                fixed_question = "Provide a detailed analysis of the component parameters found in the code."
                answer = rag_chain.invoke(fixed_question)
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(traceback.format_exc())
else:
    st.info("Generate visualizations first to enable parameter analysis.")

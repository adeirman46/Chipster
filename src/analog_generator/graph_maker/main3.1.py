import streamlit as st
import os
import networkx as nx
import matplotlib.pyplot as plt
import traceback
import numpy as np
from io import BytesIO
from typing import TypedDict, List
from collections import defaultdict
import re
import tempfile
import subprocess
import sys

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
                
                # Skip lines that are just comments or empty
                if not clean_line or clean_line.startswith('#'):
                    continue
                
                # Find the component call
                if '(' not in clean_line or ')' not in clean_line:
                    continue
                    
                parts = clean_line.split('(')
                if len(parts) < 2:
                    continue
                    
                comp_type_part = parts[0].split('.')[-1]
                args_part = parts[1].split(')')[0]
                
                # Handle arguments that might contain @ symbol (PySpice units)
                args = []
                for arg in args_part.split(','):
                    arg = arg.strip().strip("'\"")
                    # Handle PySpice unit syntax like 1@u_kOhm
                    if '@' in arg:
                        # Use regex to handle different unit formats
                        match = re.search(r'([\d\.\-eE]+)', arg)
                        if match:
                            arg = match.group(1)
                    # Handle keyword arguments by taking only the value
                    if '=' in arg:
                        arg = arg.split('=')[1].strip().strip("'\"")
                    args.append(arg)

                terminals, connections = [], []
                
                # Parse different component types
                if comp_type_part in ["V", "I"]:
                    if len(args) >= 4:  # name, n+, n-, value
                        connections = args[1:3]
                        terminals = ['n+', 'n-']
                elif comp_type_part == "R":
                    if len(args) >= 4:  # name, n1, n2, value
                        connections = args[1:3]
                        terminals = ['n1', 'n2']
                elif comp_type_part == "MOSFET":
                    if len(args) >= 5:  # name, drain, gate, source, body, [model, ...]
                        connections = args[1:5]
                        terminals = ['D', 'G', 'S', 'B']

                # Clean up connection names
                if connections:
                    cleaned_connections = []
                    for conn in connections:
                        # Replace both circuit.gnd and self.gnd with gnd
                        conn = conn.replace("circuit.gnd", "gnd").replace("self.gnd", "gnd")
                        cleaned_connections.append(conn)
                    
                    named_connections = list(zip(terminals, cleaned_connections))
                    components.append({
                        "name": args[0], 
                        "type": comp_type_part, 
                        "connections": named_connections
                    })
                    
            except (IndexError, ValueError) as e:
                # Skip malformed lines
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
    """Converts the parsed components to LCapy netlist format with sequential ground node naming as requested."""
    components = state.get('components', [])
    if not components:
        return {"lcapy_netlist": None, "net_to_node_mapping": {}}

    # 1. Map all non-ground nets to unique numbers, starting from 1.
    net_to_node = {}
    node_counter = 1
    all_nets_set = {net for comp in components for _, net in comp['connections']}
    for net in sorted(list(all_nets_set)):
        if net != 'gnd':
            if net not in net_to_node:
                net_to_node[net] = node_counter
                node_counter += 1

    # 2. Build the LCapy netlist and the new mapping.
    lcapy_lines = [
        "# LCapy Netlist generated from PySpice",
        "# WARNING: Sequential ground naming (g1, g2, ...) may create a disconnected circuit.",
        "from lcapy import Circuit",
        "cct = Circuit()",
        ""
    ]
    
    ground_instance_counter = 1
    # Create a copy of the mapping to modify for display
    display_mapping = {k: v for k, v in net_to_node.items()}

    for comp in components:
        comp_name = comp['name']
        comp_type = comp['type']
        
        node_strings = []
        for _, net in comp['connections']:
            if net == 'gnd':
                g_node_name = f"g{ground_instance_counter}"
                node_strings.append(g_node_name)
                # Add this specific instance to the display mapping
                display_mapping[f"gnd_instance_{ground_instance_counter}"] = g_node_name
                ground_instance_counter += 1
            else:
                node_strings.append(str(net_to_node.get(net, net)))
        
        connections_str = ' '.join(node_strings)
        
        # Format the LCapy `add` command
        if comp_type == 'V':
            lcapy_lines.append(f"cct.add('V{comp_name} {connections_str}; down')")
        elif comp_type == 'I':
            lcapy_lines.append(f"cct.add('I{comp_name} {connections_str}; down')")
        elif comp_type == 'R':
            lcapy_lines.append(f"cct.add('R{comp_name} {connections_str}; right')")
        elif comp_type == 'MOSFET':
            mosfet_connections = ' '.join(node_strings[:3])
            lcapy_lines.append(f"cct.add('M{comp_name} {mosfet_connections}; up')")

    lcapy_lines.extend(["", "# Draw the circuit", "cct.draw()"])
    lcapy_netlist = '\n'.join(lcapy_lines)
    
    # Clean up the display mapping keys for better readability
    final_display_mapping = {}
    gnd_count = 1
    # Sort by value to group gnd instances if they are not contiguous
    sorted_items = sorted(display_mapping.items(), key=lambda item: str(item[1]))
    
    # Process non-ground items first
    for k, v in sorted_items:
        if not str(k).startswith('gnd_instance'):
            final_display_mapping[k] = v
            
    # Process ground items to append them at the end
    for k, v in sorted_items:
        if str(k).startswith('gnd_instance'):
            final_display_mapping[f"gnd (conn {gnd_count})"] = v
            gnd_count += 1

    return {
        "lcapy_netlist": lcapy_netlist,
        "net_to_node_mapping": final_display_mapping
    }


def create_simple_schematic_plot(state):
    """Creates a simple schematic-like plot as a fallback if LCapy fails."""
    components = state.get('components', [])
    net_to_node = state.get('net_to_node_mapping', {})
    
    if not components: return {"schematic_image": None}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simple schematic layout
    for i, comp in enumerate(components):
        y_pos = len(components) - i - 1
        
        symbol_map = {'V': ('blue', 'V'), 'R': ('green', 'R'), 'I': ('red', 'I'), 'MOSFET': ('purple', 'M')}
        color, symbol = symbol_map.get(comp['type'], ('black', '?'))

        # Draw component symbol
        rect = plt.Rectangle((0.75, y_pos-0.2), 0.5, 0.4, fill=False, color=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(1, y_pos, symbol, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(1, y_pos - 0.4, comp['name'], ha='center', va='center', fontsize=8)

        # Draw connection lines and labels
        for j, (terminal, net) in enumerate(comp['connections']):
            # Find the corresponding node name from the complex mapping
            node_num = '?'
            if net == 'gnd':
                # This part is tricky because the mapping is now complex. We find the first gX.
                # This is just for the fallback plot, so an approximation is okay.
                for k, v in net_to_node.items():
                    if 'gnd' in str(k):
                        node_num = v
                        break
            else:
                node_num = net_to_node.get(net, net)

            line_x_start = 0.75 if j < len(comp['connections']) / 2 else 1.25
            line_y = y_pos + (0.15 * (1-j) if len(comp['connections'])==2 else 0.2 - j*0.13)
            
            ax.plot([line_x_start, line_x_start - 0.25], [line_y, line_y], 'k-', linewidth=1)
            ax.text(line_x_start - 0.3, line_y, f"{terminal}({node_num})", fontsize=8, ha='right', va='center')

    ax.set_title("Fallback Schematic Representation", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, len(components))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    legend_text = "Node Mapping:\n" + "\n".join([f"{net} → {node}" for net, node in sorted(net_to_node.items(), key=lambda x: str(x[1]))])
    ax.text(1.5, len(components)-0.5, legend_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8), va='top')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return {"schematic_image": buf}


def lcapy_plotter_agent(state):
    """Plots the schematic using LCapy, with a fallback to a simple plot."""
    lcapy_netlist = state.get('lcapy_netlist')
    if lcapy_netlist is None:
        return {"schematic_image": None}

    temp_script_path = None
    output_image_path = 'schematic_output.png'
    
    try:
        # Write the LCapy code to a temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            lcapy_code_to_run = f"""
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from lcapy import Circuit

# --- Netlist from Agent ---
{lcapy_netlist.replace("cct.draw()", "")}
# --- End Netlist ---

cct.draw(style='american', filename='{output_image_path}', dpi=300)
plt.close()
"""
            f.write(lcapy_code_to_run)
            temp_script_path = f.name
        
        # Execute the script in a subprocess for safety and isolation
        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True, text=True, timeout=20, check=False
        )
        
        # If script ran successfully and created the file, read it
        if result.returncode == 0 and os.path.exists(output_image_path):
            with open(output_image_path, 'rb') as img_file:
                img_data = img_file.read()
            return {"schematic_image": BytesIO(img_data)}
        else:
            # If LCapy fails, trigger the fallback plot
            st.warning("LCapy failed to generate the schematic. Displaying a fallback visualization.")
            st.code(f"LCapy Subprocess Error:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}", language='bash')
            return create_simple_schematic_plot(state)
        
    except Exception as e:
        # On any other exception, also use the fallback
        st.warning(f"An exception occurred during LCapy plotting: {e}. Displaying fallback.")
        return create_simple_schematic_plot(state)
    finally:
        # Cleanup all temporary files
        if temp_script_path and os.path.exists(temp_script_path):
            os.unlink(temp_script_path)
        if os.path.exists(output_image_path):
            os.unlink(output_image_path)


def graph_drawer_agent(state):
    """
    Draws the MultiGraph with net-based numbering and special 'g' labels for ground.
    """
    g = state.get('graph')
    if g is None: return {"image": None}
    
    fig, ax = plt.subplots(figsize=(18, 14))
    pos = nx.spring_layout(g, k=2.5, iterations=100, seed=42)

    component_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'component']
    net_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'net']
    
    nx.draw_networkx_nodes(g, pos, nodelist=component_nodes, 
                           node_shape='s', node_color='lightblue', 
                           node_size=5000, ax=ax, edgecolors='navy', linewidths=2)
    nx.draw_networkx_nodes(g, pos, nodelist=net_nodes, 
                           node_shape='o', node_color='lightgreen', 
                           node_size=3000, ax=ax, edgecolors='darkgreen', linewidths=2)

    nx.draw_networkx_labels(g, pos, font_size=9, font_weight='bold', ax=ax)

    # --- Net-Based and Ground-Specific Numbering Logic ---
    net_base_number, net_edge_count = {}, defaultdict(int)
    bundle_counter, ground_counter = 1, 1
    edge_number_labels = {}
    
    # Sort edges for deterministic numbering
    sorted_edges = sorted(list(g.edges(keys=True)))
    for u, v, key in sorted_edges:
        net_node = u if g.nodes[u]['type'] == 'net' else v
        
        if net_node == 'gnd':
            edge_number_labels[(u, v, key)] = f"g{ground_counter}"
            ground_counter += 1
        else:
            if net_node not in net_base_number:
                net_base_number[net_node] = bundle_counter
                bundle_counter += 1
            
            base_num = net_base_number[net_node]
            count = net_edge_count[net_node]
            edge_number_labels[(u, v, key)] = str(base_num) if count == 0 else f"{base_num}_{count + 1}"
            net_edge_count[net_node] += 1

    # --- Draw Edges and Labels ---
    edge_groups = defaultdict(list)
    for u, v, key in g.edges(keys=True):
        edge_key = tuple(sorted((u, v)))
        edge_groups[edge_key].append((u, v, key))

    for edge_key, edges in edge_groups.items():
        pos_u, pos_v = np.array(pos[edge_key[0]]), np.array(pos[edge_key[1]])
        
        for i, (u, v, key) in enumerate(edges):
            label = g[u][v][key].get('label', '')
            edge_num_label = edge_number_labels.get((u, v, key), '')

            if len(edges) == 1:
                nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], width=2, edge_color='gray', alpha=0.8, ax=ax)
                midpoint = (pos_u + pos_v) / 2
                edge_vector = pos_v - pos_u
                perp_vector = np.array([-edge_vector[1], edge_vector[0]]) / (np.linalg.norm(edge_vector) or 1)
                label_pos, num_label_pos = midpoint + perp_vector * 0.08, midpoint - perp_vector * 0.12
            else:
                curve = 0.15 + (i // 2) * 0.1
                connectionstyle = f"arc3,rad={curve if i % 2 == 0 else -curve}"
                nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], width=2, edge_color='gray', alpha=0.8, ax=ax, connectionstyle=connectionstyle)
                
                edge_vector = pos_v - pos_u
                straight_mid = (pos_u + pos_v) / 2
                perp_vector = np.array([-edge_vector[1], edge_vector[0]]) / (np.linalg.norm(edge_vector) or 1)
                curve_mid = straight_mid + perp_vector * (curve if i % 2 == 0 else -curve) * np.linalg.norm(edge_vector) * 0.5
                label_pos, num_label_pos = curve_mid + perp_vector * 0.05, curve_mid - perp_vector * 0.08

            ax.annotate(label, xy=label_pos, c='red', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='red', lw=1, alpha=0.8))
            ax.annotate(edge_num_label, xy=num_label_pos, c='blue', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='blue', lw=1, alpha=0.8))

    legend_elements = [plt.Line2D([0], [0], marker='s', c='w', label='Component', mfc='lightblue', ms=15, mec='navy'),
                       plt.Line2D([0], [0], marker='o', c='w', label='Net', mfc='lightgreen', ms=15, mec='darkgreen')]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title("Circuit Graph Visualization", fontsize=16, fontweight='bold')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return {"image": buf}


# --- RAG Agent Setup ---
def create_rag_chain(pyspice_code):
    """Creates a RAG chain to analyze component parameters from the PySpice code."""
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-latest", temperature=0)
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
    schematic_image: BytesIO

workflow = StateGraph(GraphState)
workflow.add_node("parser", pyspice_parser_agent)
workflow.add_node("builder", graph_builder_agent)
workflow.add_node("drawer", graph_drawer_agent)
workflow.add_node("lcapy_converter", lcapy_converter_agent)
workflow.add_node("lcapy_plotter", lcapy_plotter_agent)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "builder")
workflow.add_edge("builder", "drawer")
workflow.add_edge("drawer", "lcapy_converter")
workflow.add_edge("lcapy_converter", "lcapy_plotter")
workflow.add_edge("lcapy_plotter", END)

app_graph = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="LLM for Chip Design Automation", layout="wide")
st.title("Circuit Visualizer and Analyzer")

col1, col2, col3 = st.columns([1, 1, 1])

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
        # Define the MOSFET model
        self.model('nmos_model', 'nmos', level=1, kp=100e-6, vto=0.5)
        # Power Supply for the power
        self.V('dd', 'Vdd', self.gnd, 5.0)
        # Common-Drain Amplifier with Resistor Load
        self.MOSFET('1', 'Vdd', 'Vin', 'Vout', self.gnd, model='nmos_model')
        self.R('load', 'Vout', self.gnd, 1@u_kOhm)
"""
    )
    if st.button("Generate Circuit Visualizations", use_container_width=True):
        with st.spinner("🔬 Analyzing and visualizing..."):
            initial_state = {"pyspice_code": pyspice_code_input}
            try:
                final_state = app_graph.invoke(initial_state)
                st.session_state.graph_image = final_state.get('image')
                st.session_state.schematic_image = final_state.get('schematic_image')
                st.session_state.lcapy_netlist = final_state.get('lcapy_netlist')
                st.session_state.net_mapping = final_state.get('net_to_node_mapping')
                st.session_state.pyspice_code_for_rag = pyspice_code_input
                st.rerun()
            except Exception as e:
                st.error("Failed to generate visualizations. Please check the code for errors.")
                st.error(traceback.format_exc())

with col2:
    st.header("Connectivity Graph")
    if 'graph_image' in st.session_state:
        st.image(st.session_state.graph_image, caption="Component and Net Connectivity Graph", use_container_width=True)
    else:
        st.info("Graph will be displayed here.")

with col3:
    st.header("Circuit Schematic")
    if 'schematic_image' in st.session_state:
        st.image(st.session_state.schematic_image, caption="Generated Circuit Schematic (via LCapy)", use_container_width=True)
    else:
        st.info("Schematic will be displayed here.")

st.divider()
if 'lcapy_netlist' in st.session_state:
    st.header("🔌 Generated LCapy Netlist & Node Mapping")
    col_netlist, col_mapping = st.columns([2, 1])
    with col_netlist:
        st.code(st.session_state.lcapy_netlist, language='python')
    with col_mapping:
        if 'net_mapping' in st.session_state:
            st.markdown("##### Node Mapping")
            # The sorting key is adjusted to handle both numeric and string node names correctly
            mapping_text = "\n".join([f"- **{net}** → Node `{node}`" for net, node in sorted(st.session_state.net_mapping.items(), key=lambda x: str(x[0]))])
            st.markdown(mapping_text)

st.divider()
st.header("⚙️ Detailed Component Parameter Analysis (RAG)")
if 'pyspice_code_for_rag' in st.session_state:
    if st.button("Analyze Component Parameters", use_container_width=True):
        with st.spinner("Performing detailed analysis with AI..."):
            try:
                rag_chain = create_rag_chain(st.session_state.pyspice_code_for_rag)
                fixed_question = "Provide a detailed analysis of the component parameters found in the code."
                answer = rag_chain.invoke(fixed_question)
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.error(traceback.format_exc())
else:
    st.info("Generate visualizations first to enable parameter analysis.")

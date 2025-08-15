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
                    # Handle PySpice unit syntax like 1@u_mA
                    if '@' in arg:
                        arg = arg.split('@')[0].strip()
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


def calculate_label_offset(angle, offset_distance=0.15):
    """Calculate offset position for edge labels to avoid overlaps."""
    return offset_distance * np.cos(angle), offset_distance * np.sin(angle)


def lcapy_converter_agent(state):
    """Converts the parsed components to LCapy netlist format with smart node naming."""
    components = state.get('components', [])
    if not components:
        return {"lcapy_netlist": None}

    # 1. Create base node mapping (gnd -> 0, Vdd -> 1, etc.)
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

    # 2. Pre-scan components to understand how nets are used
    other_node_usage = defaultdict(int)
    for comp in components:
        if comp['type'] != 'MOSFET':
            for _, net in comp['connections']:
                base_node = net_to_node[net]
                other_node_usage[base_node] += 1

    # 3. Build the LCapy netlist with smart node naming
    lcapy_lines = [
        "# LCapy Netlist",
        "from lcapy import Circuit",
        "cct = Circuit()",
        ""
    ]
    node_suffix_counters = defaultdict(lambda: 2)

    for comp in components:
        comp_name = comp['name']
        comp_type = comp['type']
        
        node_strings = []
        for _, net in comp['connections']:
            base_node = net_to_node[net]
            node_name = ""

            if comp_type == 'MOSFET':
                # MOSFETs always use the simple, base node name
                node_name = str(base_node)
            else:
                # For other components, check if the net is "busy"
                if other_node_usage[base_node] > 1:
                    # If a net connects to more than one non-MOSFET, it needs a unique, suffixed name
                    suffix = node_suffix_counters[base_node]
                    node_name = f"{base_node}_{suffix}"
                    node_suffix_counters[base_node] += 1
                else:
                    # Otherwise, use the simple base name
                    node_name = str(base_node)
            
            node_strings.append(node_name)

        # Format the LCapy `add` command based on component type
        connections_str = ' '.join(node_strings)
        if comp_type == 'V':
            lcapy_lines.append(f"cct.add('V{comp_name} {connections_str}; down')")
        elif comp_type == 'I':
            lcapy_lines.append(f"cct.add('I{comp_name} {connections_str}; down')")
        elif comp_type == 'R':
            lcapy_lines.append(f"cct.add('R{comp_name} {connections_str}; right')")
        elif comp_type == 'MOSFET':
            lcapy_lines.append(f"cct.add('M{comp_name} {connections_str}; up')")

    lcapy_lines.extend(["", "# Draw the circuit", "cct.draw()"])
    lcapy_netlist = '\n'.join(lcapy_lines)

    return {
        "lcapy_netlist": lcapy_netlist,
        "net_to_node_mapping": net_to_node
    }


def lcapy_plotter_agent(state):
    """Plots the schematic using matplotlib by executing LCapy code."""
    lcapy_netlist = state.get('lcapy_netlist')
    if lcapy_netlist is None:
        return {"schematic_image": None}
    
    try:
        # Create a temporary module to execute the LCapy code
        import tempfile
        import subprocess
        import sys
        
        # Create temporary file for the LCapy script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write the LCapy code with matplotlib integration
            lcapy_code = f"""
{lcapy_netlist}

# Save the schematic to a file
import matplotlib.pyplot as plt
plt.savefig('temp_schematic.png', dpi=300, bbox_inches='tight')
plt.close()

# Read the image back as bytes
with open('temp_schematic.png', 'rb') as img_file:
    img_data = img_file.read()
    
# Save to a location we can access
with open('schematic_output.png', 'wb') as out_file:
    out_file.write(img_data)
"""
            f.write(lcapy_code)
            temp_script = f.name
        
        # Execute the script in a subprocess (safer than exec)
        try:
            result = subprocess.run([sys.executable, temp_script], 
                                    capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Try to read the generated image
                try:
                    with open('schematic_output.png', 'rb') as img_file:
                        img_data = img_file.read()
                        buf = BytesIO(img_data)
                        return {"schematic_image": buf}
                except FileNotFoundError:
                    pass
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        
        # If LCapy fails, create a simple matplotlib representation
        return create_simple_schematic_plot(state)
        
    except Exception as e:
        # Fallback: create a simple schematic representation
        return create_simple_schematic_plot(state)
    finally:
        # Cleanup temporary files
        try:
            os.unlink(temp_script)
            os.unlink('temp_schematic.png')
            os.unlink('schematic_output.png')
        except:
            pass


def create_simple_schematic_plot(state):
    """Creates a simple schematic-like plot when LCapy is not available."""
    components = state.get('components', [])
    net_to_node = state.get('net_to_node_mapping', {})
    
    if not components:
        return {"schematic_image": None}
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simple schematic layout
    y_pos = 0
    component_positions = {}
    
    for i, comp in enumerate(components):
        comp_name = f"{comp['type']}_{comp['name']}"
        y_pos = len(components) - i - 1
        
        if comp['type'] == 'V':
            # Draw voltage source as circle with +/-
            circle = plt.Circle((1, y_pos), 0.3, fill=False, color='blue', linewidth=2)
            ax.add_patch(circle)
            ax.text(1, y_pos, 'V', ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(0.5, y_pos, comp['name'], ha='center', va='center', fontsize=8)
            
        elif comp['type'] == 'R':
            # Draw resistor as zigzag rectangle
            rect = plt.Rectangle((0.5, y_pos-0.1), 1, 0.2, fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            ax.text(1, y_pos, 'R', ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(0.3, y_pos, comp['name'], ha='center', va='center', fontsize=8)
            
        elif comp['type'] == 'I':
            # Draw current source as circle with arrow
            circle = plt.Circle((1, y_pos), 0.3, fill=False, color='red', linewidth=2)
            ax.add_patch(circle)
            ax.text(1, y_pos, 'I', ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(0.5, y_pos, comp['name'], ha='center', va='center', fontsize=8)
            
        elif comp['type'] == 'MOSFET':
            # Draw MOSFET as rectangle with terminals
            rect = plt.Rectangle((0.5, y_pos-0.2), 1, 0.4, fill=False, color='purple', linewidth=2)
            ax.add_patch(rect)
            ax.text(1, y_pos, 'M', ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(0.3, y_pos, comp['name'], ha='center', va='center', fontsize=8)
            
            # Add terminal labels
            connections = comp['connections']
            ax.text(1.7, y_pos+0.1, f"D:{net_to_node.get(connections[0][1], connections[0][1])}", fontsize=7)
            ax.text(1.7, y_pos, f"G:{net_to_node.get(connections[1][1], connections[1][1])}", fontsize=7)
            ax.text(1.7, y_pos-0.1, f"S:{net_to_node.get(connections[2][1], connections[2][1])}", fontsize=7)
        
        # Draw connection lines and labels
        for j, (terminal, net) in enumerate(comp['connections']):
            node_num = net_to_node.get(net, net)
            connection_x = 2.5 + j * 0.5
            ax.plot([1.5, connection_x], [y_pos, y_pos], 'k-', linewidth=1)
            ax.text(connection_x, y_pos + 0.15, f"{terminal}", fontsize=8, ha='center', color='red')
            ax.text(connection_x, y_pos - 0.15, f"({node_num})", fontsize=8, ha='center', color='blue')
    
    # Add title and formatting
    ax.set_title("Circuit Schematic Representation", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, len(components))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend for node mapping
    legend_text = "Node Mapping:\n"
    for net, node in sorted(net_to_node.items(), key=lambda x: x[1]):
        legend_text += f"{net} → {node}\n"
    
    ax.text(4.5, len(components)-1, legend_text, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return {"schematic_image": buf}

def graph_drawer_agent(state):
    """
    Draws the MultiGraph with separate visible edges for each connection.
    """
    g = state.get('graph')
    if g is None: return {"image": None}
    
    # Use a larger figure for better readability
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Use a layout that spreads nodes better
    pos = nx.spring_layout(g, k=2.5, iterations=100, seed=42)

    # Draw nodes with better spacing and colors
    component_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'component']
    net_nodes = [n for n, d in g.nodes(data=True) if d.get('type') == 'net']
    
    nx.draw_networkx_nodes(g, pos, nodelist=component_nodes, 
                           node_shape='s', node_color='lightblue', 
                           node_size=5000, ax=ax, edgecolors='navy', linewidths=2)
    nx.draw_networkx_nodes(g, pos, nodelist=net_nodes, 
                           node_shape='o', node_color='lightgreen', 
                           node_size=3000, ax=ax, edgecolors='darkgreen', linewidths=2)

    # Draw node labels
    nx.draw_networkx_labels(g, pos, font_size=9, font_weight='bold', ax=ax)

    # --- Draw separate edges for each connection ---
    # Group edges by connected nodes to handle multiple edges
    edge_groups = defaultdict(list)
    for u, v, key in g.edges(keys=True):
        edge_key = (u, v) if u < v else (v, u)  # Normalize edge direction for grouping
        edge_groups[edge_key].append((u, v, key))

    # Draw each edge separately with proper curves for multiple connections
    for edge_key, edges in edge_groups.items():
        u_base, v_base = edge_key
        pos_u = np.array(pos[u_base])
        pos_v = np.array(pos[v_base])
        
        num_edges = len(edges)
        
        for i, (u, v, key) in enumerate(edges):
            # Get the label for this edge
            label = g[u][v][key].get('label', '')
            
            if num_edges == 1:
                # Single edge: draw straight line
                nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], 
                                     width=2, edge_color='gray', alpha=0.8, ax=ax)
                
                # Place label at midpoint
                midpoint = (pos_u + pos_v) / 2
                edge_vector = pos_v - pos_u
                edge_length = np.linalg.norm(edge_vector)
                if edge_length > 0:
                    perp_vector = np.array([-edge_vector[1], edge_vector[0]]) / edge_length
                    label_pos = midpoint + perp_vector * 0.08
                else:
                    label_pos = midpoint
            else:
                # Multiple edges: draw with curves to separate them visually
                # Calculate curve radius based on edge index
                curve_offset = 0.15 + (i // 2) * 0.1
                if i % 2 == 1:
                    curve_offset = -curve_offset
                
                # Draw curved edge
                connectionstyle = f"arc3,rad={curve_offset}"
                nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], 
                                     width=2, edge_color='gray', alpha=0.8, ax=ax,
                                     connectionstyle=connectionstyle)
                
                # Calculate label position on the curve
                # For curved edges, we need to calculate the midpoint of the curve
                edge_vector = pos_v - pos_u
                edge_length = np.linalg.norm(edge_vector)
                if edge_length > 0:
                    # Calculate the curved midpoint
                    straight_midpoint = (pos_u + pos_v) / 2
                    perp_vector = np.array([-edge_vector[1], edge_vector[0]]) / edge_length
                    curve_midpoint = straight_midpoint + perp_vector * curve_offset * edge_length * 0.5
                    
                    # Add small offset for label readability
                    label_offset = perp_vector * 0.05
                    label_pos = curve_midpoint + label_offset
                else:
                    label_pos = pos_u
            
            # Draw the label with background for better readability
            ax.annotate(label, xy=label_pos, fontsize=9, fontweight='bold',
                        color='red', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                  edgecolor='red', alpha=0.8))

    # Improve the legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Component', 
                   markerfacecolor='lightblue', markersize=15, markeredgecolor='navy', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', label='Net', 
                   markerfacecolor='lightgreen', markersize=15, markeredgecolor='darkgreen', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    # Set title and clean up the plot
    ax.set_title("Circuit Graph Visualization", fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    
    # Remove axes ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid for better visual reference
    ax.grid(True, alpha=0.3)
    
    # Adjust margins
    plt.tight_layout()
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
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

# Create three columns for better layout
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
        self.V('dd', 'Vdd', self.gnd, 5.0)  # 5V power supply
        # Common-Drain Amplifier with Resistor Load
        self.MOSFET('1', 'Vdd', 'Vin', 'Vout', self.gnd, model='nmos_model', w=50e-6, l=1e-6)
        self.R('load', 'Vout', self.gnd, 1@u_kOhm)
"""
    )
    if st.button("Generate Circuit Visualizations", use_container_width=True):
        with st.spinner("Generating visualizations..."):
            initial_state = {"pyspice_code": pyspice_code_input}
            final_state = app_graph.invoke(initial_state)
            if final_state:
                if final_state.get('image'):
                    st.session_state.graph_image = final_state.get('image')
                if final_state.get('schematic_image'):
                    st.session_state.schematic_image = final_state.get('schematic_image')
                if final_state.get('lcapy_netlist'):
                    st.session_state.lcapy_netlist = final_state.get('lcapy_netlist')
                if final_state.get('net_to_node_mapping'):
                    st.session_state.net_mapping = final_state.get('net_to_node_mapping')
                st.session_state.pyspice_code_for_rag = pyspice_code_input
                st.rerun()
            else:
                st.error("Failed to generate visualizations. Check code for errors.")

with col2:
    st.header("Graph Visualization")
    if 'graph_image' in st.session_state and st.session_state.graph_image:
        st.image(st.session_state.graph_image, caption="Circuit Graph", use_container_width=True)
    else:
        st.write("Graph will be displayed here once generated.")

with col3:
    st.header("Schematic Visualization")
    if 'schematic_image' in st.session_state and st.session_state.schematic_image:
        st.image(st.session_state.schematic_image, caption="Circuit Schematic", use_container_width=True)
    else:
        st.write("Schematic will be displayed here once generated.")

# LCapy Netlist Display
st.divider()
st.header("🔌 Generated LCapy Netlist")
if 'lcapy_netlist' in st.session_state and st.session_state.lcapy_netlist:
    col_netlist, col_mapping = st.columns([2, 1])
    
    with col_netlist:
        st.code(st.session_state.lcapy_netlist, language='python')
    
    with col_mapping:
        st.subheader("Node Mapping")
        if 'net_mapping' in st.session_state:
            mapping_text = ""
            for net, node in sorted(st.session_state.net_mapping.items(), key=lambda x: x[1]):
                mapping_text += f"**{net}** → Node {node}\n"
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

import streamlit as st
import os
import networkx as nx
import matplotlib.pyplot as plt
import traceback
import numpy as np
from io import BytesIO
from typing import TypedDict, List
from collections import defaultdict

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
        if 'circuit.' in line and '(' in line:
            try:
                clean_line = line.split('#')[0]
                parts = clean_line.split('(')
                comp_type_part = parts[0].split('.')[-1]
                args_part = parts[1].split(')')[0]
                args = [arg.strip().strip("'") for arg in args_part.split(',')]

                terminals, connections = [], []
                if comp_type_part in ["V", "I"]:
                    connections, terminals = args[1:3], ['n+', 'n-']
                elif comp_type_part == "R":
                    connections, terminals = args[1:3], ['n1', 'n2']
                elif comp_type_part == "MOSFET":
                    connections, terminals = args[1:5], ['D', 'G', 'S', 'B']

                if connections:
                    named_connections = list(zip(terminals, [c.replace("circuit.gnd", "gnd") for c in connections]))
                    components.append({"name": args[0], "type": comp_type_part, "connections": named_connections})
            except IndexError:
                pass
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
workflow = StateGraph(GraphState)
workflow.add_node("parser", pyspice_parser_agent)
workflow.add_node("builder", graph_builder_agent)
workflow.add_node("drawer", graph_drawer_agent)
workflow.set_entry_point("parser")
workflow.add_edge("parser", "builder")
workflow.add_edge("builder", "drawer")
workflow.add_edge("drawer", END)
app_graph = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="LLM for Chip Design Automation", layout="wide")
st.title("Circuit Visualizer and Analyzer")
col1, col2 = st.columns(2)
with col1:
    st.header("PySpice Code Input")
    pyspice_code_input = st.text_area(
        "Enter your PySpice code here:",
        height=450,
        value="""from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
circuit = Circuit('Cascode Current Mirror')
# Define the MOSFET models
circuit.model('nmos_model', 'nmos', level=1, kp=100e-6, vto=0.5)
# Power Supply
circuit.V('dd', 'Vdd', circuit.gnd, 5.0)
# Reference Current Source
circuit.I('ref', 'Vdd', 'Iref', 1@u_mA)
# Input Side: M1 is diode-connected. M2 S/B are tied to ground.
circuit.MOSFET('1', 'Iref', 'Iref', 'Source2', 'Source2', model='nmos_model')
circuit.MOSFET('2', 'Source2', 'Iref', circuit.gnd, circuit.gnd, model='nmos_model')
# Output Side: M4 S/B are tied to ground.
circuit.MOSFET('3', 'Iout', 'Iref', 'Source4', 'Source4', model='nmos_model')
circuit.MOSFET('4', 'Source4', 'Iref', circuit.gnd, circuit.gnd, model='nmos_model')
# Load Resistor
circuit.R('load', 'Iout', 'Vdd', 1@u_kΩ)
"""
    )
    if st.button("Generate Circuit Graph", use_container_width=True):
        with st.spinner("Generating graph..."):
            initial_state = {"pyspice_code": pyspice_code_input}
            final_state = app_graph.invoke(initial_state)
            if final_state and final_state.get('image'):
                st.session_state.graph_image = final_state.get('image')
                st.session_state.pyspice_code_for_rag = pyspice_code_input
                st.rerun()
            else:
                st.error("Failed to generate graph. Check code for errors.")
with col2:
    st.header("Circuit Graph")
    if 'graph_image' in st.session_state and st.session_state.graph_image:
        st.image(st.session_state.graph_image, caption="Generated Circuit Graph", use_container_width=True)
    else:
        st.write("Graph will be displayed here once generated.")
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
    st.info("Generate a graph first to enable parameter analysis.")
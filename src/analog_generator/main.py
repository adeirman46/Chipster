import streamlit as st
import os
import asyncio
import traceback

# LangChain and Gemini specific imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv

# Core simulation and plotting imports
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt
import numpy as np

# --- App Configuration & Initialization ---

st.set_page_config(layout="wide", page_title="LLM for Analog Chip Design ⚡️")
load_dotenv()

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("🚨 GOOGLE_API_KEY environment variable not found. Please create a .env file with your key.")
    st.stop()

# Initialize Session State Variables
for key in ['pyspice_code', 'schematic_code', 'simulation_code', 'initial_prompt']:
    if key not in st.session_state:
        st.session_state[key] = ""

# --- RAG Setup with Persistent FAISS Index ---

FAISS_INDEX_PATH = "../../data/analog_datasets/pyspice_index"

@st.cache_resource
def get_or_create_retriever():
    """Loads a FAISS index from disk if it exists, otherwise creates and saves it."""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        if os.path.exists(FAISS_INDEX_PATH):
            st.info(f"Loading existing FAISS index from '{FAISS_INDEX_PATH}'...")
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("FAISS index loaded successfully.")
            return vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            st.info(f"FAISS index not found. Creating a new one from the 'datasets' directory...")
            if not os.path.exists("datasets"):
                st.error("The 'datasets' directory was not found. Please create it and add your Python files.")
                return None

            loader = DirectoryLoader('./datasets/', glob="**/*.py", show_progress=True)
            raw_documents = loader.load()

            if not raw_documents:
                st.error("No .py files found in the 'datasets' directory. Cannot build knowledge base.")
                return None

            text_splitter = RecursiveCharacterTextSplitter.from_language(language="python", chunk_size=1000, chunk_overlap=100)
            documents = text_splitter.split_documents(raw_documents)

            st.info(f"Creating embeddings for {len(documents)} document chunks. This may take a moment...")
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            st.success(f"New FAISS index created and saved to '{FAISS_INDEX_PATH}'.")
            return vectorstore.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"Failed to create or load the RAG retriever: {e}")
        st.code(traceback.format_exc())
        return None

# Initialize LLM and Retriever
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
RETRIEVER = get_or_create_retriever()


# --- Core Functions & Prompts ---

def extract_code_from_response(response: str) -> str:
    """Extracts Python code from a markdown formatted string."""
    if "```python" in response:
        return response.split("```python\n")[1].split("```")[0].strip()
    elif "```" in response:
        return response.split("```")[1].strip()
    return response

def generate_with_rag(prompt_template, query: str, retriever=RETRIEVER):
    """Generates code using a RAG chain, expecting a 'query'."""
    if not LLM or not retriever:
        st.error("ERROR: LLM or Retriever not initialized.")
        return ""
    rag_chain = (
        {"context": retriever.get_relevant_documents, "query": RunnablePassthrough()}
        | prompt_template
        | LLM
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)
    return extract_code_from_response(response)

def generate_from_template(prompt_template, input_dict: dict):
    """Generates code using a simple prompt-to-LLM chain, no RAG."""
    if not LLM:
        st.error("ERROR: LLM not initialized.")
        return ""
    simple_chain = prompt_template | LLM | StrOutputParser()
    response = simple_chain.invoke(input_dict)
    return extract_code_from_response(response)

# --- Prompt Templates ---

PYSPICE_PROMPT = PromptTemplate.from_template("""
You are an expert in analog circuit design using PySpice. Generate a complete, runnable PySpice script based on the user's request, using the provided context for guidance.
CONTEXT: {context}
USER REQUEST: {query}
INSTRUCTIONS:
1. Create a single, complete Python script defining the circuit.
2. Import `PySpice.Spice.Netlist.Circuit` and `PySpice.Unit`.
3. Define necessary models (e.g., `circuit.model(...)`).
4. Construct the full circuit including all sources and components.
5. DO NOT include `simulator` or any analysis calls.
6. Output ONLY the Python code inside a markdown block.
""")

# --- FIX: Updated Schematic Prompt ---
SCHEMATIC_PROMPT = PromptTemplate.from_template("""
You are a `schemdraw` expert. Convert the following PySpice code into a `schemdraw` visualization.

PYSPICE CODE:
{pyspice_code}

MODIFICATION REQUEST:
{modification_prompt}

INSTRUCTIONS:
1. Create a `schemdraw.Drawing()` object named `d`.
2. Accurately represent all components and connections.
3. **IMPORTANT**: If the PySpice code defines a 4-terminal MOSFET (with a bulk connection), you MUST use `elm.NFet(bulk=True)` or `elm.PFet(bulk=True)`.
4. Label key nodes and components (e.g., 'Vin', 'Vout', 'R1', 'M1').
5. The final line of the script MUST be `d.draw()`. The application will handle rendering.
6. Output ONLY the Python code inside a markdown block.
""")

SIMULATION_PROMPT = PromptTemplate.from_template("""
You are a PySpice simulation expert. Generate a Python script to perform a DC sweep on the input 'Vin' of the given circuit and plot the 'Vout' using Matplotlib.
PYSPICE CODE:
{pyspice_code}
MODIFICATION REQUEST:
{modification_prompt}
INSTRUCTIONS:
1. Assume the `circuit` object from the PySpice code already exists.
2. Create the simulator and run a `.dc()` analysis on `Vin` from 0V to 5V.
3. Use `matplotlib.pyplot` to create the plot. Name the figure object `figure`.
4. Set a clear title for the plot and labels for the x and y axes.
5. Output ONLY the Python code for simulation and plotting inside a markdown block.
""")


# --- Streamlit UI ---

st.title("LLM for Analog Chip Design ⚡️")
st.write("Your AI assistant for analog circuit design, powered by LLM.")

st.header("1. Describe Your Circuit")
initial_prompt = st.text_area(
    "Start by describing the circuit you want to design (e.g., 'a common-source amplifier with a 4kOhm load').",
    height=100,
    key="initial_prompt_input"
)

if st.button("🚀 Generate Design", use_container_width=True, type="primary"):
    if initial_prompt and RETRIEVER:
        st.session_state.initial_prompt = initial_prompt
        with st.spinner("Step 1/3: Generating PySpice circuit..."):
            st.session_state.pyspice_code = generate_with_rag(PYSPICE_PROMPT, initial_prompt)
        with st.spinner("Step 2/3: Visualizing with Schemdraw..."):
            schematic_input = {"pyspice_code": st.session_state.pyspice_code, "modification_prompt": "None"}
            st.session_state.schematic_code = generate_from_template(SCHEMATIC_PROMPT, schematic_input)
        with st.spinner("Step 3/3: Simulating DC characteristics..."):
            simulation_input = {"pyspice_code": st.session_state.pyspice_code, "modification_prompt": "None"}
            st.session_state.simulation_code = generate_from_template(SIMULATION_PROMPT, simulation_input)
    elif not initial_prompt:
        st.warning("Please describe the circuit you want to build.")
    else:
        st.error("Retriever is not available. Please check the console for errors.")

if st.session_state.pyspice_code:
    st.header("2. Review and Refine Your Design")

    col_code1, col_code2, col_code3 = st.columns(3)
    with col_code1:
        st.subheader("PySpice Code")
        st.session_state.pyspice_code = st.text_area("Circuit Definition", value=st.session_state.pyspice_code, height=300, key="pyspice_editor")
    with col_code2:
        st.subheader("Schematic Code")
        st.session_state.schematic_code = st.text_area("`schemdraw` Visualization", value=st.session_state.schematic_code, height=300, key="schematic_editor")
    with col_code3:
        st.subheader("Simulation Code")
        st.session_state.simulation_code = st.text_area("`matplotlib` Plotting", value=st.session_state.simulation_code, height=300, key="simulation_editor")

    st.header("3. Visualize and Simulate")

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.subheader("Circuit Schematic")
        if st.session_state.schematic_code:
            try:
                # --- FIX: Correctly render schemdraw by capturing the Matplotlib figure ---
                plt.figure() # Create a new figure to draw on
                exec_globals = {'schemdraw': schemdraw, 'elm': elm, 'plt': plt}
                exec(st.session_state.schematic_code, exec_globals)
                fig = plt.gcf() # Get the current figure that schemdraw drew on
                st.pyplot(fig)
            except Exception:
                st.error("Error in schematic code:")
                st.code(traceback.format_exc())

    with col_vis2:
        st.subheader("DC Simulation Plot")
        if st.session_state.pyspice_code and st.session_state.simulation_code:
            try:
                full_code = st.session_state.pyspice_code + "\n" + st.session_state.simulation_code
                exec_globals = {
                    'Circuit': Circuit, 'u_kOhm': u_kOhm, 'u_V': u_V, 'u_uA': u_uA,
                    'u_uF': u_uF, 'u_nH': u_nH, 'u_pF': u_pF, 'u_Ohm': u_Ohm,
                    'u_mH': u_mH, 'u_mV': u_mV, '@u_kΩ': u_kOhm,
                    'plt': plt, 'np': np
                }
                exec(full_code, exec_globals)
                fig = exec_globals.get('figure')
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Could not find a `figure` object in the simulation code to plot.")
            except Exception:
                st.error("Error during simulation:")
                st.code(traceback.format_exc())

import streamlit as st
import os
import glob
import pandas as pd
from typing import List, TypedDict, Dict
import torch
import re
import json
import graphviz
import subprocess
import difflib
import base64

from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langgraph.graph import StateGraph, END

# For the web agent
import asyncio
import nest_asyncio
from crawl4ai import AsyncWebCrawler
from googlesearch import search
from langchain.docstore.document import Document

# For Waveform visualization
from sootty import WireTrace, Visualizer, Style

# --- Configuration & Setup ---

load_dotenv()
nest_asyncio.apply()

st.set_page_config(page_title="Chipster Agent", layout="wide")
st.title("🤖 Chipster Agent: A Self-Correcting Verilog Designer")
st.markdown("Powered by LangGraph and Gemini 2.5 Pro")

try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
    st.error("🚨 GOOGLE_API_KEY not found! Please create a .env file with your key.")
    st.stop()

# --- Part 1: FAISS Index & Model Loading ---

DATASET_PATH = "../../data/verilog_datasets"
INDEX_PATH_DATASET = os.path.join(DATASET_PATH, "faiss_verilog_db")
INDEX_PATH_QFT = os.path.join(DATASET_PATH, "faiss_qft_verieval") # NEW: Path for the second index
GENERATED_CODE_PATH = "../../examples/verilog_designs"
MAX_RETRIES = 10 # Maximum number of correction attempts

@st.cache_resource
def get_embedding_model():
    """Loads the local HuggingFace embedding model, cached for performance."""
    st.write("Loading Local Embedding Model (all-MiniLM-L6-v2)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f"Using device: {device}")
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': device})

@st.cache_resource
def load_dataset_vectorstore():
    """Loads the main dataset FAISS index if it exists."""
    if os.path.exists(INDEX_PATH_DATASET):
        st.write(f"Loading existing dataset FAISS index from '{INDEX_PATH_DATASET}'...")
        return FAISS.load_local(INDEX_PATH_DATASET, get_embedding_model(), allow_dangerous_deserialization=True)
    else:
        st.warning(f"Local dataset index not found at '{INDEX_PATH_DATASET}'. This data source will be skipped.")
        return None

@st.cache_resource
def load_qft_vectorstore():
    """Loads the QFT and VerilogEval FAISS index if it exists."""
    if os.path.exists(INDEX_PATH_QFT):
        st.write(f"Loading existing QFT/VeriEval FAISS index from '{INDEX_PATH_QFT}'...")
        return FAISS.load_local(INDEX_PATH_QFT, get_embedding_model(), allow_dangerous_deserialization=True)
    else:
        st.warning(f"Local QFT index not found at '{INDEX_PATH_QFT}'. This data source will be skipped.")
        return None

db_verilog_dataset = load_dataset_vectorstore()
db_qft_verieval = load_qft_vectorstore() # NEW: Load the second database


# --- Part 2: LangGraph Multi-Agent Setup ---

class GraphState(TypedDict):
    query: str
    log: List[str]
    documents: List[Document]
    generation: str
    decomposed_files: Dict[str, str]
    testbench_code: Dict[str, str]
    output_path: str
    simulation_output: str
    error_count: int
    top_module_name: str
    summary: str
    theory: str
    waveform_svg: str


def get_graph_viz(active_node: str = None):
    """Generates a Graphviz object to visualize the agent workflow."""
    dot = graphviz.Digraph(comment='Chipster Agent Workflow')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightgrey')
    dot.attr(rankdir='TB', splines='ortho')

    nodes = {
        "dataset_retriever": "1. Dataset Retriever",
        "web_retriever": "2. Web Researcher",
        "code_generator": "3. Verilog Generator",
        "decomposer": "4. Decomposer & Header Extractor", # UPDATED
        "testbench_generator": "5. Testbench Writer",
        "file_writer": "6. File Writer",
        "simulator": "7. Icarus Simulator",
        "check_simulation": "8. Check Results",
        "module_corrector": "9a. Module Corrector",
        "testbench_corrector": "9b. Testbench Corrector",
        "summarizer": "10. Code Summarizer",
        "theory_researcher": "11. Theory Researcher",
        "waveform_viewer": "12. Waveform Viewer"
    }
    for name, label in nodes.items():
        if name == active_node:
            dot.node(name, label, shape='square', style='filled,bold', fillcolor='#FFFF99', fontcolor='black') # Yellow highlight
        else:
            dot.node(name, label, shape='box', style='rounded,filled', fillcolor='#E0E0E0', fontcolor='black') # Light Grey

    # Main flow
    dot.edge("dataset_retriever", "web_retriever")
    dot.edge("web_retriever", "code_generator")
    dot.edge("code_generator", "decomposer")
    dot.edge("decomposer", "testbench_generator")
    dot.edge("testbench_generator", "file_writer")
    dot.edge("file_writer", "simulator")
    dot.edge("simulator", "check_simulation")

    # Success Path
    dot.edge("check_simulation", "summarizer", label="Success", color="green", style="bold")
    dot.edge("summarizer", "theory_researcher")
    dot.edge("theory_researcher", "waveform_viewer")
     
    # Add an END node for clarity
    dot.node("END", "🏁 END", shape="ellipse", style="filled", fillcolor="palegreen")
    dot.edge("waveform_viewer", "END")


    # Conditional Edges from Router
    dot.edge("check_simulation", "testbench_corrector", label="Fix Testbench", color="orange", style="dashed")
    dot.edge("check_simulation", "module_corrector", label="Fix Design", color="red", style="dashed")

    # Correction loop paths
    dot.edge("testbench_corrector", "file_writer", style="dashed")
    dot.edge("module_corrector", "file_writer", style="dashed")

    return dot

# --- Helper Functions ---
def log_code_changes(log: List[str], filename: str, old_code: str, new_code: str) -> List[str]:
    """Generates a diff and adds it to the log."""
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    diff_str = "".join(diff)
    if diff_str:
        log.append(f"🔍 Code changes for `{filename}`:\n```diff\n{diff_str}```")
    else:
        log.append(f"🔍 No functional changes detected for `{filename}`.")
    return log

# --- Agent Nodes ---

def dataset_retriever_node(state):
    query = state["query"]
    log = state.get("log", []) + ["\n--- AGENT: Dataset Retriever ---"]
    all_docs = []

    # No change to this node, keeping it concise
    if db_verilog_dataset:
        docs1 = db_verilog_dataset.as_retriever(search_kwargs={"k": 10}).invoke(query)
        all_docs.extend(docs1)
        log.append(f"Found {len(docs1)} docs in 'faiss_verilog_db'.")
    if db_qft_verieval:
        docs2 = db_qft_verieval.as_retriever(search_kwargs={"k": 10}).invoke(query)
        all_docs.extend(docs2)
        log.append(f"Found {len(docs2)} docs in 'faiss_qft_verieval'.")
     
    log.append(f"Total documents retrieved from local DBs: {len(all_docs)}")
    return {"documents": all_docs, "log": log}

def web_retriever_node(state):
    return asyncio.run(web_retriever_node_async(state))

async def web_retriever_node_async(state):
    """
    UPDATED NODE: This node has an improved search and crawling strategy
    to find more relevant Verilog code on GitHub.
    """
    query = state["query"]
    existing_docs = state.get("documents", [])
    log = state.get("log", []) + ["\n--- AGENT: Web Researcher ---"]
    embeddings = get_embedding_model()
    sanitized_prompt = re.sub(r'\W+', '_', query).lower()
    index_name = f"faiss_github_{sanitized_prompt}"
    INDEX_PATH_WEB = os.path.join(DATASET_PATH, index_name)
    log.append(f"Checking for cached web index: '{INDEX_PATH_WEB}'")
    web_vectorstore = None
    if os.path.exists(INDEX_PATH_WEB):
        log.append("✅ Cached index found! Loading.")
        web_vectorstore = FAISS.load_local(INDEX_PATH_WEB, embeddings, allow_dangerous_deserialization=True)
    else:
        log.append("❌ No cache. Searching and crawling web...")

        # --- IMPROVED SEARCH LOGIC ---
        # Broader search query to find repositories and code
        search_query = f'"{query}" verilog source code OR design files site:github.com'
        log.append(f"Executing Google search with query: '{search_query}'")
        # Increase search results to get more diverse code examples
        urls = list(search(search_query, num_results=10, lang="en"))
        log.append(f"Found {len(urls)} potential URLs from Google.")
        # Log the first few URLs for debugging
        for i, url in enumerate(urls[:5]):
            log.append(f"  - URL {i+1}: {url}")
        # --- END IMPROVEMENT ---

        if not urls:
             log.append("⚠️ No relevant URLs found on Google search.")
             return {"documents": existing_docs, "log": log}

        new_web_docs = []
        crawled_count = 0
        async with AsyncWebCrawler() as crawler:
            # --- IMPROVED CRAWLING LOGIC ---
            # Process all found URLs instead of just a subset
            log.append(f"Crawling up to {len(urls)} URLs...")
            for url in urls:
                if url and "github.com" in url: # Ensure it's a GitHub link
                    try:
                        result = await crawler.arun(url=url)
                        if result and result.markdown:
                            # Add a check for code content to avoid empty READMEs
                            if "```" in result.markdown or "module" in result.markdown or "input" in result.markdown:
                                new_web_docs.append(Document(page_content=result.markdown, metadata={"source": url}))
                                crawled_count += 1
                                log.append(f"  - ✅ Successfully crawled: {url}")
                            else:
                                log.append(f"  - 🟡 Skipped (no code indicators): {url}")
                        else:
                            log.append(f"  - ⚠️ Crawled but no markdown content: {url}")
                    except Exception as e:
                        log.append(f"  - ❌ Failed to crawl {url}: {e}")
            # --- END IMPROVEMENT ---

        if new_web_docs:
            log.append(f"Successfully extracted content from {crawled_count} URLs.")
            split_docs = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(new_web_docs)
            web_vectorstore = FAISS.from_documents(split_docs, embeddings)
            web_vectorstore.save_local(INDEX_PATH_WEB)
            log.append(f"✅ New web index saved with {len(split_docs)} document chunks.")
        else:
            log.append("Could not retrieve any valid documents from the web.")

    docs_from_web = []
    if web_vectorstore:
        # Retrieve more documents to give the generator more context
        retriever = web_vectorstore.as_retriever(search_kwargs={"k": 15}) # Increased k
        docs_from_web = retriever.invoke(query)
        log.append(f"✅ Retrieved {len(docs_from_web)} relevant document chunks from web cache for the query.")
    else:
        log.append("⚠️ No web vectorstore available to retrieve from.")

    return {"documents": existing_docs + docs_from_web, "log": log}

def code_generator_node(state):
    query = state["query"]
    documents = state["documents"]
    log = state.get("log", []) + ["\n--- AGENT: Verilog Generator ---"]
    log.append("✍️ Generating monolithic code from scratch...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)
     
    prompt_template = """You are an expert Verilog HDL designer.
Based on the context from reference documents and the user's request, generate the complete, monolithic Verilog code.
The code should be well-structured and include any necessary `define` macros or parameters at the top.
Your output **MUST** be only the Verilog code, enclosed in a single markdown block. Do not include any other text.

**CONTEXT:**
{context}

**REQUEST:**
{question}

**GENERATED VERILOG CODE:**
```verilog
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
     
    def format_docs(docs):
        if not docs: return "No context documents found."
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in docs)
         
    rag_chain = ({"context": lambda x: format_docs(x["documents"]), "question": RunnablePassthrough()}| prompt | llm | StrOutputParser())
    generation = rag_chain.invoke({"documents": documents, "question": query}).replace("```verilog", "").replace("```", "").strip()
    log.append("✅ Monolithic code generated.")
     
    return {"generation": generation, "log": log, "simulation_output": ""}

def module_corrector_node(state):
    log = state.get("log", []) + ["\n--- AGENT: Verilog Module Corrector ---"]
    log.append("♻️ Attempting to fix previous design error...")
     
    decomposed_files = state["decomposed_files"]
    error_log = state["simulation_output"]
     
    # Improved logic to find the faulty file
    faulty_filename = None
    for fname in decomposed_files.keys():
        # Icarus often reports errors with file:line format
        if fname in error_log:
            faulty_filename = fname
            break
     
    if not faulty_filename:
        log.append("⚠️ Could not identify a specific faulty module from the error log. No correction applied.")
        return {"decomposed_files": decomposed_files, "log": log}

    faulty_code = decomposed_files[faulty_filename]
    log.append(f"Identified faulty file: `{faulty_filename}`")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)
     
    prompt_template = """You are an expert Verilog debugger.
**TASK:** You are given a single Verilog module that failed during simulation. Analyze the error message and the code, identify the bug, and provide a corrected version of **only that module's code**.
Your output **MUST** be only the corrected Verilog code for the module, enclosed in a single markdown block.

**FAULTY VERILOG MODULE (`{faulty_filename}`):**
```verilog
{faulty_code}
```

**SIMULATION ERROR LOG:**
```
{error_log}
```

**YOUR RESPONSE (Corrected, Complete Verilog Code for the Module Only):**
```verilog
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
     
    corrected_module_code = chain.invoke({
        "faulty_filename": faulty_filename,
        "faulty_code": faulty_code,
        "error_log": error_log
    }).replace("```verilog", "").replace("```", "").strip()

    updated_files = decomposed_files.copy()
    updated_files[faulty_filename] = corrected_module_code
    log.append(f"✅ Design correction generated for `{faulty_filename}`.")
     
    log = log_code_changes(log, faulty_filename, faulty_code, corrected_module_code)

    return {"decomposed_files": updated_files, "log": log}


def decomposer_node(state):
    """
    UPDATED NODE: This node now also extracts `define` macros and parameters
    into a separate .vh header file and adds `include` statements where needed.
    """
    generation = state["generation"]
    log = state.get("log", []) + ["\n--- AGENT: Decomposer & Header Extractor ---"]
    log.append("Decomposing code and extracting headers...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=GOOGLE_API_KEY)
     
    decomposer_prompt_template = """You are an expert Verilog refactoring tool.
Your task is to analyze monolithic Verilog code and decompose it into multiple files.

**RULES:**
1.  Identify the top-level module.
2.  Separate each `module` into its own file (e.g., `module_name.v`).
3.  **Crucially: Identify all `` `define `` macros and shared `parameter` declarations. Move them into a single header file (e.g., `shared_header.vh`).**
4.  In each `.v` file that uses a macro/parameter from the header, add the `` `include "shared_header.vh" `` directive at the top.
5.  Return a single, valid JSON object with two keys: "top_module_name" and "files".
6.  `files` must be an object where keys are filenames (`.v` or `.vh`) and values are the code content.
7.  Your final output **MUST** be only the JSON object.

**USER REQUEST:** {query}
**MONOLITHIC VERILOG CODE:**
```verilog
{verilog_code}
```

**RESPONSE (Valid JSON object only):**
"""
    decomposer_prompt = ChatPromptTemplate.from_template(decomposer_prompt_template)
     
    chain = decomposer_prompt | llm | StrOutputParser()
    response = chain.invoke({"verilog_code": generation, "query": state["query"]})
     
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in the LLM response.", response, 0)
         
        json_str = json_match.group(0)
        parsed_json = json.loads(json_str)
         
        decomposed_files = parsed_json.get("files", {})
        top_module_name = parsed_json.get("top_module_name", "")

        if not decomposed_files or not top_module_name:
             raise ValueError("Parsed JSON is missing 'files' or 'top_module_name' keys.")

        log.append(f"✅ Decomposed into {len(decomposed_files)} files. Top module: `{top_module_name}`")
        if any(".vh" in f for f in decomposed_files.keys()):
            log.append("✅ Header file extracted successfully.")

    except (json.JSONDecodeError, ValueError) as e:
        log.append(f"❌ Failed to parse valid JSON from decomposer. Error: {e}. Falling back to monolithic code.")
        log.append(f"   Raw LLM Response: {response}")
        top_module_match = re.search(r'module\s+([\w#\(\)]+)', generation)
        top_module_name = top_module_match.group(1).split('#')[0].strip() if top_module_match else "unknown_module"
        decomposed_files = {f"{top_module_name}.v": generation}
         
    return {"decomposed_files": decomposed_files, "top_module_name": top_module_name, "log": log}

def testbench_generator_node(state):
    """
    UPDATED NODE: This node now instructs the LLM to create a self-checking
    testbench that explicitly prints PASSED, SUCCESS, or FAILED.
    """
    log = state.get("log", []) + ["\n--- AGENT: Testbench Writer ---"]
    decomposed_files = state["decomposed_files"]
    top_module_name = state["top_module_name"]

    if not decomposed_files:
        log.append("❌ Cannot generate testbench: No decomposed module files were provided.")
        return {"testbench_code": {}, "log": log}

    log.append("✍️ Generating new self-checking testbench...")
    top_module_code = decomposed_files.get(f"{top_module_name}.v", list(decomposed_files.values())[0])
    header_file = next((f for f in decomposed_files if f.endswith('.vh')), None)
    header_include_line = f'- **Include the header file: `` `include "{header_file}" ``**' if header_file else ''

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)
     
    prompt_template = """You are an expert in Verilog testbench design.
**TASK:** Write a comprehensive, self-checking testbench for the provided top-level module.

**CRITICAL REQUIREMENTS:**
1.  The testbench module name **MUST** be `{top_module_name}_tb`.
2.  Instantiate the DUT, provide realistic stimuli, and use `$display` or `$monitor` to show results.
3.  It must include a clock signal if needed and terminate automatically using `$finish`.
4.  **Self-Checking:** At the end of the simulation, the testbench must determine if the test passed or failed.
5.  **Clear Output Message:** You **MUST** add a final block that uses `$display` to print **EXACTLY** `"SIMULATION PASSED"` on a new line if all checks are correct.
6.  If any check fails, it must use `$display` to print **EXACTLY** `"SIMULATION FAILED"` on a new line, and then immediately call `$finish`.
7.  **Waveform Generation:** You MUST include these two lines at the start of the initial block:
    `$dumpfile("design.vcd");`
    `$dumpvars(0, {top_module_name}_tb);`
{header_include}
8.  Your final output **MUST** be a single, valid JSON object with one key-value pair: the key is the testbench filename (`{top_module_name}_tb.v`) and the value is the complete testbench code. **DO NOT** include the DUT's code in your response.

**TOP-LEVEL MODULE CODE (for context only):**
```verilog
{top_module_code}
```
**RESPONSE (Valid JSON object containing only the testbench code):**
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
     
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "top_module_name": top_module_name,
        "top_module_code": top_module_code,
        "header_include": header_include_line
    })

    try:
        json_str = response[response.find('{'):response.rfind('}')+1]
        testbench_json = json.loads(json_str)
        log.append(f"✅ Self-checking testbench generated: `{list(testbench_json.keys())[0]}`")
    except Exception as e:
        log.append(f"❌ Failed to generate valid testbench JSON. Error: {e}")
        testbench_json = {}
         
    return {"testbench_code": testbench_json, "log": log}


def testbench_corrector_node(state):
    log = state.get("log", []) + ["\n--- AGENT: Testbench Corrector ---"]
    log.append("♻️ Attempting to fix previous testbench error...")

    decomposed_files = state["decomposed_files"]
    top_module_name = state["top_module_name"]
    faulty_tb_code_dict = state["testbench_code"]
    error_log = state["simulation_output"]

    top_module_code = decomposed_files.get(f"{top_module_name}.v", list(decomposed_files.values())[0])
    faulty_tb_filename = list(faulty_tb_code_dict.keys())[0] if faulty_tb_code_dict else f"{top_module_name}_tb.v"
    faulty_tb_code = list(faulty_tb_code_dict.values())[0] if faulty_tb_code_dict else "# Faulty testbench code was not found"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)
     
    prompt_template = """You are an expert Verilog testbench debugger.
**TASK:** You are given a testbench that failed during simulation. Analyze the error message, the testbench code, and the module it is testing (DUT). Provide a corrected version of **only the testbench code**.

**CRITICAL REQUIREMENTS:**
1.  **Self-Checking:** Ensure the corrected testbench explicitly prints `"SIMULATION PASSED"` or `"SIMULATION FAILED"` at the end.
2.  **Waveform Generation:** Ensure the corrected testbench includes `$dumpfile("design.vcd");` and `$dumpvars(0, {top_module_name}_tb);`.
3.  Your final output **MUST** be a single JSON object containing the corrected testbench. The key must be the original testbench filename.

**SIMULATION ERROR LOG:**
```
{error_log}
```

**FAULTY TESTBENCH CODE (`{faulty_tb_filename}`):**
```verilog
{faulty_tb_code}
```

**DEVICE UNDER TEST (DUT) CODE (`{top_module_name}.v`) (for context only):**
```verilog
{top_module_code}
```

**RESPONSE (Valid JSON object containing only the corrected testbench code):**
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
     
    response = chain.invoke({
        "top_module_name": top_module_name,
        "top_module_code": top_module_code,
        "error_log": error_log,
        "faulty_tb_filename": faulty_tb_filename,
        "faulty_tb_code": faulty_tb_code
    })

    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in the LLM response.", response, 0)
         
        json_str = json_match.group(0)
        corrected_testbench_json = json.loads(json_str)

        if not isinstance(corrected_testbench_json, dict) or not corrected_testbench_json:
            raise ValueError("Parsed JSON is not a valid, non-empty dictionary.")
         
        corrected_tb_filename = list(corrected_testbench_json.keys())[0]
        corrected_tb_code = corrected_testbench_json[corrected_tb_filename]

        log.append(f"✅ Testbench correction generated for: `{corrected_tb_filename}`")
        log = log_code_changes(log, corrected_tb_filename, faulty_tb_code, corrected_tb_code)

    except (json.JSONDecodeError, ValueError) as e:
        log.append(f"❌ Failed to generate valid corrected testbench JSON. Error: {e}")
        log.append(f"   Raw LLM Response: {response}")
        corrected_testbench_json = faulty_tb_code_dict
         
    return {"testbench_code": corrected_testbench_json, "log": log}


def file_writer_node(state):
    log = state.get("log", []) + ["\n--- AGENT: File Writer ---"]
    query = state["query"]
    decomposed_files = state["decomposed_files"]
    testbench_code = state.get("testbench_code", {})
    sanitized_prompt = re.sub(r'\W+', '_', query).lower()
    output_path = os.path.join(GENERATED_CODE_PATH, f"generated_{sanitized_prompt}")
    os.makedirs(output_path, exist_ok=True)
    log.append(f"Writing files to: `{output_path}`")

    all_files_to_write = {**decomposed_files, **testbench_code}
    for filename, content in all_files_to_write.items():
        # Sanitize filename just in case
        safe_filename = re.sub(r'[^\w\.\-]', '_', filename)
        if isinstance(content, str) and content.strip():
            with open(os.path.join(output_path, safe_filename), 'w') as f:
                f.write(content)
            log.append(f"  - Wrote `{safe_filename}`")
        else:
            log.append(f"  - ⚠️ Skipped writing `{safe_filename}` due to invalid content.")
             
    return {"output_path": output_path, "log": log}

def simulator_node(state):
    """
    UPDATED NODE: Runs simulation and checks stdout for explicit
    PASSED, SUCCESS, or FAILED messages to determine the outcome.
    """
    log = state.get("log", []) + ["\n--- AGENT: Icarus Simulator ---"]
    output_path = state["output_path"]
    log.append(f"Preparing to simulate files in `{output_path}`")

    verilog_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(output_path, "*.v"))]
    if not verilog_filenames:
        log.append("❌ No `.v` files found to simulate.")
        return {"simulation_output": "Error: No Verilog files found.", "log": log}
     
    output_vvp_filename = "design.vvp"
    command = ["iverilog", "-o", output_vvp_filename] + verilog_filenames
     
    simulation_output = ""
    try:
        log.append(f"Running compilation in `{output_path}`...")
        log.append(f"Command: `{' '.join(command)}`")
        compile_process = subprocess.run(
            command, capture_output=True, text=True, timeout=30, cwd=output_path
        )
         
        if compile_process.returncode != 0:
            raise subprocess.CalledProcessError(compile_process.returncode, compile_process.args, output=compile_process.stdout, stderr=compile_process.stderr)
             
        log.append("✅ Compilation successful.")
         
        sim_command = ["vvp", output_vvp_filename]
        log.append(f"Running simulation in `{output_path}`...")
        log.append(f"Command: `{' '.join(sim_command)}`")
        sim_process = subprocess.run(
            sim_command, capture_output=True, text=True, check=True, timeout=30, cwd=output_path
        )
        
        # --- NEW VERIFICATION LOGIC ---
        stdout = sim_process.stdout
        log.append("✅ Simulation finished. Verifying output...")
        log.append(f"```\n{stdout}\n```")

        if re.search(r'SIMULATION PASSED|SUCCESS', stdout, re.IGNORECASE):
            log.append("✅ Verification PASSED.")
            simulation_output = "" # Clear output on success
        elif re.search(r'SIMULATION FAILED|FAIL', stdout, re.IGNORECASE):
            log.append("❌ Verification FAILED.")
            simulation_output = f"ERROR: Testbench reported failure.\n{stdout}"
        else:
            log.append("⚠️ Verification AMBIGUOUS: No explicit PASS/FAIL message found.")
            # Treat ambiguous as failure to enforce good testbench practice
            simulation_output = f"ERROR: Ambiguous result. Testbench did not report PASS or FAIL.\n{stdout}"
        # --- END NEW LOGIC ---

    except subprocess.CalledProcessError as e:
        error_message = e.stderr or e.stdout or "Unknown simulation error."
        simulation_output = f"ERROR during {'compilation' if 'iverilog' in ' '.join(e.cmd) else 'simulation'}:\n{error_message}"
        log.append(f"❌ Simulation Failed.")
    except subprocess.TimeoutExpired:
        simulation_output = "ERROR: Simulation timed out. The testbench may have an infinite loop or is not finishing with `$finish`."
        log.append(f"❌ {simulation_output}")

    if "ERROR" not in simulation_output and not simulation_output:
        return {"simulation_output": "", "log": log}
    else:
        error_count = state.get("error_count", 0) + 1
        return {"simulation_output": simulation_output, "error_count": error_count, "log": log}


def check_simulation_results_node(state):
    log = state.get("log", []) + ["\n--- ROUTER: Checking Results ---"]
    simulation_output = state.get("simulation_output", "")
    error_count = state.get("error_count", 0)
     
    if not simulation_output:
        log.append("✅ Success! Routing to final documentation.")
        return "success"

    log.append(f"⚠️ Error detected on attempt {error_count + 1}.")
    if error_count >= MAX_RETRIES:
        log.append(f"❌ Maximum retry limit ({MAX_RETRIES}) reached. Halting workflow.")
        return "end"
     
    tb_files = [f for f in state.get("testbench_code", {}).keys()]
    # Check for testbench file names in the error, timeout, or explicit failure messages
    is_tb_error = (
        any(tb_file in simulation_output for tb_file in tb_files if tb_file) or
        "timeout" in simulation_output.lower() or
        "reported failure" in simulation_output.lower() or
        "ambiguous result" in simulation_output.lower()
    )

    if is_tb_error:
        log.append("Routing to: Testbench Corrector")
        return "fix_testbench"
    else:
        log.append("Routing to: Module Corrector")
        return "fix_design"

def summarizer_node(state):
    log = state.get("log", []) + ["\n--- AGENT: Code Summarizer ---"]
    log.append("Generating code summary...")
     
    top_module_name = state["top_module_name"]
    top_module_code = state["decomposed_files"].get(f"{top_module_name}.v", "")

    if not top_module_code:
        log.append("⚠️ Top module code not found for summarization.")
        return {"summary": "Could not generate summary because the top-level module code was not found.", "log": log}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        """You are a technical writer for hardware design. Based on the Verilog code, create a concise summary.
        Include:
        1.  **Purpose**: One sentence on what the module does.
        2.  **Ports**: Lists of input and output ports with bit widths.
        3.  **Functionality**: A short paragraph on its behavior.

        **Top-Level Module Code (`{module_name}.v`):**
        ```verilog
        {module_code}
        ```
        **Your Summary:**
        """
    )
     
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"module_name": top_module_name, "module_code": top_module_code})
     
    log.append("✅ Summary generated.")
    return {"summary": summary, "log": log}

async def theory_researcher_node_async(state):
    log = state.get("log", []) + ["\n--- AGENT: Theory Researcher ---"]
    query = state["query"]
    log.append(f"Researching theory for: '{query}'...")

    search_query = f"explain {query} digital logic design"
    urls = list(search(search_query, num_results=1, lang="en"))

    if not urls:
        log.append("⚠️ No relevant theory explanation found on the web.")
        return {"theory": "Could not find a relevant theoretical explanation for this topic.", "log": log}

    explanation_content = ""
    async with AsyncWebCrawler() as crawler:
        if urls[0]:
            try:
                result = await crawler.arun(url=urls[0])
                if result and result.markdown:
                    explanation_content = result.markdown
            except Exception as e:
                log.append(f"⚠️ Failed to crawl {urls[0]}: {e}")
                return {"theory": "Failed to retrieve information from the web.", "log": log}

    if not explanation_content:
        log.append("⚠️ Crawled page has no content.")
        return {"theory": "Could not extract content from the web page.", "log": log}

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_template(
        """You are an expert in digital logic. Based on the text provided, write a concise explanation of the concept requested by the user.
        Focus on fundamental principles.

        **User's Request:** {original_query}
        **Content from Webpage:**
        ```
        {web_content}
        ```
        **Your Concise Explanation:**
        """
    )
    chain = prompt | llm | StrOutputParser()
    theory = chain.invoke({"original_query": query, "web_content": explanation_content})

    log.append("✅ Theoretical explanation generated.")
    return {"theory": theory, "log": log}

def theory_researcher_node(state):
    return asyncio.run(theory_researcher_node_async(state))

def waveform_viewer_node(state):
    """Generates an SVG waveform from the VCD file using Sootty."""
    log = state.get("log", []) + ["\n--- AGENT: Waveform Viewer ---"]
    output_path = state["output_path"]
    vcd_path = os.path.join(output_path, "design.vcd")
     
    log.append(f"Looking for VCD file at: `{vcd_path}`")

    if not os.path.exists(vcd_path) or os.path.getsize(vcd_path) == 0:
        log.append("⚠️ VCD file not found or is empty. Skipping waveform generation.")
        return {"waveform_svg": "", "log": log}
         
    log.append("✅ Found VCD file. Generating waveform image with Sootty...")
     
    try:
        wiretrace = WireTrace.from_vcd(vcd_path)
        # Render all wires for a comprehensive view
        wires_to_render = wiretrace.get_wires()
        image = Visualizer(Style.Dark).to_svg(wiretrace, start=0, length=2000, wires=wires_to_render)
        # Convert SVG object to a base64 encoded string for reliable display
        svg_string = image.decode('utf-8')
        log.append("✅ Waveform SVG generated successfully.")
        return {"waveform_svg": svg_string, "log": log}
    except Exception as e:
        log.append(f"❌ Failed to generate waveform with Sootty: {e}")
        return {"waveform_svg": "", "log": log}

# --- Graph Definition ---
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("dataset_retriever", dataset_retriever_node)
    workflow.add_node("web_retriever", web_retriever_node)
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("module_corrector", module_corrector_node)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("testbench_generator", testbench_generator_node)
    workflow.add_node("testbench_corrector", testbench_corrector_node)
    workflow.add_node("file_writer", file_writer_node)
    workflow.add_node("simulator", simulator_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("theory_researcher", theory_researcher_node)
    workflow.add_node("waveform_viewer", waveform_viewer_node)

    workflow.set_entry_point("dataset_retriever")
    workflow.add_edge("dataset_retriever", "web_retriever")
    workflow.add_edge("web_retriever", "code_generator")
    workflow.add_edge("code_generator", "decomposer")
    workflow.add_edge("decomposer", "testbench_generator")
    workflow.add_edge("testbench_generator", "file_writer")
    workflow.add_edge("file_writer", "simulator")
     
    workflow.add_edge("summarizer", "theory_researcher")
    workflow.add_edge("theory_researcher", "waveform_viewer")
    workflow.add_edge("waveform_viewer", END)

    workflow.add_edge("module_corrector", "file_writer")
    workflow.add_edge("testbench_corrector", "file_writer")
     
    workflow.add_conditional_edges(
        "simulator",
        check_simulation_results_node,
        {
            "success": "summarizer",
            "fix_testbench": "testbench_corrector",
            "fix_design": "module_corrector",
            "end": END
        }
    )
     
    # The recursion limit is set during the stream/invoke call, not at compile time.
    return workflow.compile()

app = build_graph()


# --- Part 3: Streamlit UI ---
st.sidebar.header("Controls")
user_query = st.sidebar.text_area("Describe the Verilog module you want to build:", height=150, value="risc v 32 bit")

if st.sidebar.button("✨ Generate & Verify Code", use_container_width=True):
    if not user_query:
        st.sidebar.error("Please enter a description for the Verilog module.")
    else:
        st.subheader("🚀 Agent Workflow Visualization")
        graph_placeholder = st.empty()
         
        col1, col2 = st.columns([1, 2])
        with col1:
            log_expander = st.expander("Agent Activity Log", expanded=True)
            log_container = log_expander.container()
         
        with col2:
            results_expander = st.expander("Final Results & Files", expanded=True)
            results_placeholder = results_expander.container()


        graph_placeholder.graphviz_chart(get_graph_viz())
        inputs = {"query": user_query, "error_count": 0, "summary": "", "theory": "", "waveform_svg": ""}
         
        with st.spinner("Chipster Agent is thinking... This may take a few minutes for complex designs."):
            final_result = None
            log_messages = []
             
            # Set the recursion limit for this specific run in the config.
            config = {"recursion_limit": 100}
            for s in app.stream(inputs, config=config, stream_mode="values"):
                active_node = list(s.keys())[-1]
                graph_placeholder.graphviz_chart(get_graph_viz(active_node))
                final_result = s
                if "log" in final_result:
                    new_logs = final_result["log"][len(log_messages):]
                    for msg in new_logs:
                        log_container.markdown(f"{msg.strip()}", unsafe_allow_html=True)
                        log_messages.append(msg)
             
            graph_placeholder.graphviz_chart(get_graph_viz("END"))
             
            results_placeholder.subheader("🏁 Final Outcome")
            if final_result.get("simulation_output"): # Check if error output exists
                results_placeholder.error(f"Workflow halted with an error after {final_result.get('error_count', 0)} retries.")
                with results_placeholder.expander("🚨 View Final Simulation Error", expanded=True):
                    st.code(final_result.get("simulation_output"), language='bash')
            else:
                st.balloons()
                results_placeholder.success(f"✅ All Verilog files generated, verified, and saved to: `{final_result.get('output_path', 'N/A')}`")
             
            if final_result.get("summary"):
                with results_placeholder.expander("📝 Code Summary", expanded=True):
                    st.markdown(final_result["summary"])
             
            if final_result.get("theory"):
                with results_placeholder.expander("🎓 Theory & Explanation", expanded=True):
                    st.markdown(final_result["theory"])
             
            # Display Waveform if it exists
            if final_result.get("waveform_svg"):
                with results_placeholder.expander("📈 Waveform Visualization", expanded=True):
                    # Display SVG directly for better rendering
                    st.image(final_result["waveform_svg"], use_column_width=True)
            elif not final_result.get("simulation_output"): # Only show if successful
                 with results_placeholder.expander("📈 Waveform Visualization", expanded=True):
                    st.warning("Waveform data was not generated. This can happen if the testbench does not properly exercise the design or if the simulation is very short.")


            results_placeholder.write("---")
            results_placeholder.subheader("Generated Files & Content")
            all_files = {
                **final_result.get("decomposed_files", {}),
                **final_result.get("testbench_code", {})
            }
            if all_files:
                # Sort files to show headers first, then top module, then others
                sorted_files = sorted(all_files.items(), key=lambda item: (not item[0].endswith('.vh'), item[0] != f"{final_result.get('top_module_name')}.v", item[0]))
                for filename, content in sorted_files:
                    icon = "📄"
                    if filename.endswith("_tb.v"):
                        icon = "🧪"
                    elif filename.endswith(".vh"):
                        icon = "📚"
                    with results_placeholder.expander(f"{icon} **{filename}**"):
                        st.code(content, language='verilog')
else:
    st.info("Enter your Verilog design requirements in the sidebar and click 'Generate & Verify'.")

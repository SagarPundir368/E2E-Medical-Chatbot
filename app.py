# app.py
import os
import torch
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage, AIMessage
import os

# Import your custom modules explicitly
from src.config import RERANKER_MODEL_NAME, DATA_DIR
from src.config import SUPPORTED_PROVIDERS, PROVIDER_MODELS
from src.embedding import get_embedding_model
from src.retrieval import load_production_hybrid_retriever
from src.rag_service import MedicalRAGService

# Set professional page layouts
st.set_page_config(page_title="Clinical RAG Portal", page_icon="🩺", layout="wide")

# Initialize session structures at the absolute top to avoid layout reset loops
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Primary Consultation Thread": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Primary Consultation Thread"

# --- 1. DYNAMIC MODEL INITIALIZER FACTORY ---
def build_selected_llm(provider: str, model_name: str, api_key: str):
    """Factory function that returns the correct LangChain model instance based on user input."""
    if provider == "Groq":
        return ChatGroq(model=model_name, temperature=0.1, api_key=api_key)
    elif provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=0.1, api_key=api_key)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


# --- 2. SIDEBAR INTERFACE (Authentication & Model Parameters) ---
with st.sidebar:
    st.title("⚙️ Engine Configuration")
    
    # Provider selection
    selected_provider = st.selectbox("Select LLM Provider:", options=SUPPORTED_PROVIDERS)
    
    # Models dropdown list changes dynamically based on selection above
    available_models = PROVIDER_MODELS[selected_provider]
    selected_model = st.selectbox("Select Model:", options=available_models)
    
    # Password text-input field hides secret key strings
    user_api_key = st.text_input(
        f"Enter your {selected_provider} API Key:",
        type="password",
        placeholder=f"Paste your {selected_provider.lower()} key here..."
    )

    if selected_provider == "Groq":
        st.caption("🔑 Missing credentials? [Get your Groq API(Free) Key here](https://console.groq.com/keys).")
    elif selected_provider == "OpenAI":
        st.caption("🔑 Missing credentials? [Get your OpenAI API($Paid$) Key here](https://platform.openai.com/api-keys).")
    
    st.divider()
    st.title("📁 Patient Case Files")
    st.subheader("Manage Active Sessions")
    
   # Cleaner session tracking selector
    st.session_state.current_session = st.selectbox(
        "Select Active Consultation:",
        options=list(st.session_state.sessions.keys()),
        index=list(st.session_state.sessions.keys()).index(st.session_state.current_session)
    )
        
    new_thread = st.text_input("Open New Consultation Case File:")
    if st.button("Initialize Case File") and new_thread.strip():
        if new_thread not in st.session_state.sessions:
            st.session_state.sessions[new_thread] = []
            st.session_state.current_session = new_thread
            st.rerun()
    
    st.divider()
    device_type = "CUDA Tensor Cores" if torch.cuda.is_available() else "Local CPU Computing"
    st.info(f"💡 **Engine Details:** Running BGE-M3 Embeddings + Hybrid RRF Fusion Search via {device_type}.")

# --- 4. MAIN DISPLAY ENTRY & AUTHENTICATION GUARDRAIL ---
st.title("🩺 Advanced Medical Expert AI Agent")

if not user_api_key.strip():
    st.warning("🔑 Please enter your API Key in the sidebar configuration layout to unlock the RAG system dashboard.")
    st.stop()  

# --- 3. THE ARCHITECTURAL SPLIT: CACHE STATIC RETRIEVAL ASSETS ---
@st.cache_resource
def load_retrieval_assets():
    """Loads massive neural databases and local encoders into RAM exactly once.

    Takes no arguments, meaning it NEVER reloads during runtime.
    """
    st.write("Initializing static clinical database layers...")
    
    st.write("Loading Embeddings...")
    embeddings_model = get_embedding_model()
    
    st.write("Loading Reranker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, device=device)
    
    st.write("Loading Retriever Data Chunks...")
    chunks_pickle_path = os.path.join(DATA_DIR, "processed_chunk", "semantic_chunks.pkl")
    hybrid_retriever = load_production_hybrid_retriever(embeddings_model, chunks_pickle_path)
    
    return embeddings_model, hybrid_retriever, reranker_model


# Execute the static data loader immediately on application boot
st.write("Starting App...")
st.success("Streamlit Engine Core Loaded")
embeddings, retriever, reranker = load_retrieval_assets()


# # --- 4. SECURE AUTHENTICATION GUARDRAIL ---
# if not user_api_key.strip():
#     st.title("🩺 Advanced Medical Expert AI Agent")
#     st.warning("🔑 Enter your API Key in the sidebar configuration layout to unlock the RAG system dashboard.")
#     st.stop()

# --- 5. DYNAMIC STEP-BY-STEP SERVICE ASSEMBLE ---
# This block runs instantly on user interaction because the heavy database components are already in RAM!
try:
    # A. Build the lightweight LLM connection layer
    llm = build_selected_llm(selected_provider, selected_model, user_api_key)
    
    # B. Bind the new LLM to the existing frozen memory pointers
    rag_service = MedicalRAGService(
        llm=llm, 
        hybrid_retriever=retriever, 
        reranker_model=reranker
    )
except Exception as e:
    st.error(f"Failed to compile dynamic RAG orchestration layer: {str(e)}")
    st.stop()


# --- 6. MAIN CHAT CONTAINER PRESENTATION ---
st.caption(f"Active Case Track: `{st.session_state.current_session}` | Model: `{selected_model}`")
active_history = st.session_state.sessions[st.session_state.current_session]

# Render historical messages within the open viewport panel
for message in active_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Prompt execution chat submission line
if user_input := st.chat_input("Enter your symptoms or clinical query..."):
    # Display human interaction element immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Store text input explicitly inside memory trace array
    active_history.append(HumanMessage(content=user_input))
    
    # Trigger RAG processing operations pipeline
    with st.chat_message("assistant"):
        # UI Progress spinners communicate background actions gracefully
        with st.spinner("Analyzing query context, scanning textbooks, and verifying clinical evidence..."):
            try:
                # Execute your brilliant multi-stage execution pipeline function
                result_payload = rag_service.generate_response(
                    user_query=user_input,
                    chat_history=active_history[:-1] # Pass full history except current query entry
                )
                
                final_answer = result_payload["answer"]
                
                # Render the final parsed output answer text block
                st.markdown(final_answer)
                
                # Append finalized output text record to system state tracker arrays
                active_history.append(AIMessage(content=final_answer))
                
            except Exception as e:
                error_msg = f"An architectural processing runtime failure occurred: `{str(e)}`"
                st.error(error_msg)
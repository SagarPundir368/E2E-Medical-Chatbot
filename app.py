# app.py
import os
import torch
import streamlit as st
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

# Import your custom modules explicitly
from src.config import GENERATION_LLM_NAME, RERANKER_MODEL_NAME, DATA_DIR
from src.embedding import get_embedding_model
from src.retrieval import load_production_hybrid_retriever
from src.rag_service import MedicalRAGService

# Set professional page layouts
st.set_page_config(page_title="Clinical RAG Portal", page_icon="🩺", layout="wide")

# --- 1. MEMORY CACHING ENGINE ---
@st.cache_resource
def initialize_core_rag_service():
    """Loads massive neural models and database vector maps into RAM exactly once."""
    st.write("Initializing clinical pipeline infrastructure...")
    
    # Init underlying model abstractions
    st.write("Loading LLM...")
    llm = ChatGroq(api_key=api_key, model=GENERATION_LLM_NAME, temperature=0.1)
    
    st.write("Loading Embeddings...")
    embeddings_model = get_embedding_model()
    
    st.write("Loading Reranker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_model = CrossEncoder(
        RERANKER_MODEL_NAME,
        device=device
    )
    
    # Load your optimized pre-computed hybrid retriever instance
    st.write("Loading Retriever...")
    chunks_pickle_path = os.path.join(DATA_DIR,"processed_chunk", "semantic_chunks.pkl")
    hybrid_retriever = load_production_hybrid_retriever(embeddings_model, chunks_pickle_path)
    
    # Return initialized master system orchestration class
    return MedicalRAGService(
        llm=llm, 
        hybrid_retriever=hybrid_retriever, 
        reranker_model=reranker_model
    )

st.write("Starting App...")
st.success("Streamlit Loaded")
# Instantiate engine asset
rag_service = initialize_core_rag_service()

# --- 2. MULTI-SESSION TRACKER STATE ---
if "sessions" not in st.session_state:
    st.session_state.sessions = {"Primary Consultation Thread": []}
if "current_session" not in st.session_state:
    st.session_state.current_session = "Primary Consultation Thread"

# --- 3. SIDEBAR INTERFACE ---
with st.sidebar:
    st.title("📁 Patient Case Files")
    st.subheader("Manage Active Sessions")
    
    new_thread = st.text_input("Open New Consultation Case File:")
    if st.button("Initialize Case File") and new_thread.strip():
        if new_thread not in st.session_state.sessions:
            st.session_state.sessions[new_thread] = []
            st.session_state.current_session = new_thread
            st.rerun()
            
    st.session_state.current_session = st.selectbox(
        "Select Active Consultation:",
        options=list(st.session_state.sessions.keys()),
        index=list(st.session_state.sessions.keys()).index(st.session_state.current_session)
    )
    
    st.divider()
    st.info("💡 **Engine Details:** Running BGE-M3 Embeddings + Hybrid RRF Fusion Search via local CPU computing.")
# Bind active history array pointer trace reference
active_history = st.session_state.sessions[st.session_state.current_session]

# --- 4. MAIN CHAT CONTAINER PRESENTATION ---
st.title("🩺 Advanced Medical Expert AI Agent")
st.caption(f"Active Case Track: `{st.session_state.current_session}`")

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
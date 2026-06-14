import os
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
# Core configuration pathways
from src.config import INDEX_DIR, ENSEMBLE_DENSE_WEIGHT, ENSEMBLE_SPARSE_WEIGHT

def load_production_hybrid_retriever(embeddings_model, chunks_pickle_path: str) -> EnsembleRetriever:
    """Loads a pre-built FAISS index from disk and assembles an optimized hybrid pipeline.

    Bypasses vector recalculations entirely by pulling localized index files, 
    maximizing operational runtime efficiency for user interactions.

    Args:
        embeddings_model: The global HuggingFaceEmbeddings model wrapper object (BGE-M3).
        chunks_pickle_path (str): Path to your saved 'semantic_chunks.pkl' asset.

    Returns:
        EnsembleRetriever: A fully initialized, high-performance RRF hybrid retriever.
    """
    print("⚡ Loading pre-computed FAISS vector database index from disk...")
    
    # 1. Load the pre-saved Dense Vector Store (No re-calculation!)
    dense_store = FAISS.load_local(
        folder_path=INDEX_DIR,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True # Required to load local pkl headers securely
    )
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": 10})
    
    # 2. Load your raw text chunks out of the backup pickle file to build BM25
    print("📖 Loading serialized text chunks for keyword search index...")
    with open(chunks_pickle_path, "rb") as f:
        cached_chunks = pickle.load(f)
        
    sparse_retriever = BM25Retriever.from_documents(cached_chunks)
    sparse_retriever.k=10
    
    print("🚀 Blending search layers into a 60/40 Ensemble Retriever array...")
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[ENSEMBLE_DENSE_WEIGHT, ENSEMBLE_SPARSE_WEIGHT]
    )


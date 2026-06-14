import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "FAISS","medical_embed")

# Model Parameters
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
GENERATION_LLM_NAME = "llama-3.3-70b-versatile" # Or our local path/ollama tag

# Retrieval Weights
ENSEMBLE_DENSE_WEIGHT = 0.6
ENSEMBLE_SPARSE_WEIGHT = 0.4
RERANK_TOP_N = 5

# Clean Display Mapper
BOOK_NAME_MAPPER = {
    "Medical_book.pdf": "The Gale Encyclopedia of Medicine",
    "AnatomyAndPhysiology-LR.pdf": "OpenStax Anatomy and Physiology"
}


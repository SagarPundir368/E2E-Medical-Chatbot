import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "FAISS","medical_embed")

# Model Parameters
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# Provider Mappings
SUPPORTED_PROVIDERS = ["Groq", "OpenAI"]

# Grouped Model Allocations
PROVIDER_MODELS = {
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ],
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo"
    ]
}

# Retrieval Weights
ENSEMBLE_DENSE_WEIGHT = 0.6
ENSEMBLE_SPARSE_WEIGHT = 0.4
RERANK_TOP_N = 5

# Clean Display Mapper
BOOK_NAME_MAPPER = {
    "Medical_book.pdf": "The Gale Encyclopedia of Medicine",
    "AnatomyAndPhysiology-LR.pdf": "OpenStax Anatomy and Physiology"
}


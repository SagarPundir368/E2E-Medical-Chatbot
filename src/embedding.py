import os
import torch
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Core configuration configurations
from src.config import EMBEDDING_MODEL_NAME


def get_embedding_model():
# Initialize the BGE-M3 model on my RTX 4060 GPU tensor cores
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device}
    )

    return embeddings

def find_embed(chunks,embed_model,embed_path):
    vectorstore = FAISS.from_documents(chunks, embed_model)
    ## SAVING LOCALLY
    os.makedirs(embed_path, exist_ok=True)
    vectorstore.save_local(embed_path)
    print(f"Vector store assets successfully indexed and saved to: {embed_path}")
    return vectorstore


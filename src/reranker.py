from src.config import RERANK_TOP_N


def run_gpu_reranker(query, documents, raw_rerank_model, top_n=RERANK_TOP_N):
    """Executes deep cross-attention ranking across retrieved candidate slices."""
    # Score pairs dynamically via the RTX 4060 GPU
    pairs = [[query, doc.page_content] for doc in documents]
    scores = raw_rerank_model.predict(pairs)
    
    # Sort documents by their scoring output descending
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_n]]


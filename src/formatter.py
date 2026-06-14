import os
from src.config import BOOK_NAME_MAPPER, ENSEMBLE_DENSE_WEIGHT, ENSEMBLE_SPARSE_WEIGHT


def format_docs(docs):
    """Strips hard-drive file paths out and replaces them with clean book titles."""
    formatted_blocks = []
    for i, doc in enumerate(docs):
        filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
        clean_title = BOOK_NAME_MAPPER.get(filename, filename)
        page = doc.metadata.get('page', 'N/A')
        
        block = f"[Document {i+1}]\nSource: {clean_title}\nPage: {page}\n\n{doc.page_content.strip()}"
        formatted_blocks.append(block)
    return "\n\n".join(formatted_blocks)
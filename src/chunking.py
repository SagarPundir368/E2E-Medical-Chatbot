import os
import pickle
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Safe relative module references based on your project directory layout
from src.config import DATA_DIR

def load_pdf_files(data_path: str) -> List[Document]:
    """Discovers and parses all PDF documents within a target directory.

    Utilizes PyMuPDF under the hood for highly accurate text and layout extraction 
    from academic textbooks or reference materials.

    Args:
        data_path (str): Absolute or relative filesystem path to the data folder.

    Returns:
        List[Document]: A raw list of un-chunked LangChain Document objects.
    """
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    raw_pages = loader.load()
    return raw_pages

def filter_docs_metadata(docs: List[Document]) -> List[Document]:
    """Cleans extracted document metadata by stripping irrelevant properties.

    Retains only essential mapping vectors ('source' and 'page') to limit token bloat
    and enhance vector store indexing efficiency.

    Args:
        docs (List[Document]): Raw document pages extracted by the directory loader.

    Returns:
        List[Document]: A minimized list of documents containing pristine metadata pairs.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
            )
        ) 
    return minimal_docs

def hybrid_chunking(clean_docs: List[Document],embedding_model) -> List[Document]:
    """Transforms raw text documents into highly accurate semantic context chunks.

    Processes documents using semantic boundary analysis via the BAAI/bge-m3 model,
    breaking text precisely where thematic context shifts occur. Once processed, 
    the final chunk data array is cached to disk as a serialized pickle asset.

    Args:
        clean_docs (List[Document]): Minimal documents containing sanitized metadata.

    Returns:
        List[Document]: The final generated semantic chunk list array.
    """
    # Performance Optimization: Passing documents directly to Semantic Chunker 
    # protects your GPU VRAM from loop thrashing while maintaining elite recall accuracy.
    recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n","\n",". "," ",""]
    )

    recursive_docs = recursive_splitter.split_documents(clean_docs)

    semantic_chunker = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    print("Initiating semantic chunk division on GPU via BGE-M3...")
    final_chunks = semantic_chunker.split_documents(recursive_docs)
    print(f"Successfully generated {len(final_chunks)} semantic chunks!")

    # Safe system-agnostic file path configuration
    pickle_output_path = os.path.join(DATA_DIR,"processed_chunk", "semantic_chunks.pkl")
    
    with open(pickle_output_path, "wb") as f:
        pickle.dump(final_chunks, f)
    print(f"Chunk layout asset serialized and saved to: {pickle_output_path}")

    return final_chunks
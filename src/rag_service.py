# src/rag_service.py
import os
from typing import List, Dict, Any

from src.config import RERANK_TOP_N
from src.reranker import run_gpu_reranker
from src.formatter import format_docs
from src.chains import get_query_enhancer_chain, get_history_aware_query_chain, get_final_rag_chain

class MedicalRAGService:
    def __init__(self, llm, hybrid_retriever, reranker_model):
        """Orchestrates sequential multi-stage pre-retrieval and generation layers."""
        self.llm = llm
        self.hybrid_retriever = hybrid_retriever
        self.reranker_model = reranker_model
        
        # Factory chain allocations
        self.query_enhancer_chain = get_query_enhancer_chain(llm)
        self.history_aware_query_chain = get_history_aware_query_chain(llm)
        self.final_rag_chain = get_final_rag_chain(llm)
        
    def enhance_query(self, query: str,chat_history: List) -> Dict[str, Any]:
        """Resolves conversational pronouns first, then expands synonyms for vector lookups."""
        # Step 1: Coreference resolution (Converts "how is it treated?" -> "treatment for Diabetes")
        condensed_query = self.history_aware_query_chain.invoke(
            {
                "query": query,
                "chat_history": chat_history 
            }
        )
        
        # Step 2: Pass the clean standalone text into your JSON query decomposition block
        enhanced_output = self.query_enhancer_chain.invoke({"query": condensed_query})
        
        # Return BOTH queries so downstream components can use them appropriately
        return {
            "condensed_query": condensed_query,
            "enhanced_query": enhanced_output
        }

    def retrieve_documents(self, enhanced_query: str) -> List[Any]:
        """Gathers dense and sparse candidates from the index structure."""
        return self.hybrid_retriever.invoke(enhanced_query)

    def rerank_documents(self, original_query: str, docs: List[Any]) -> List[Any]:
        """Reranks candidates on your RTX 4060 using the original user query intent."""
        return run_gpu_reranker(
            query=original_query,
            documents=docs,
            raw_rerank_model=self.reranker_model,
            top_n=RERANK_TOP_N
        )
    
    def generate_answer(self, query: str, context: str, chat_history: List[Any]) -> str:
        """Invokes the final grounded text generation chain."""
        return self.final_rag_chain.invoke({
            "query": query,
            "context": context,
            "chat_history": chat_history
        })
    
    def generate_response(self, user_query: str, chat_history: List[Any]) -> Dict[str, Any]:
        """Executes the master operational RAG pipeline timeline."""
        # 1. Enhance and unpack query
        enhanced_output = self.enhance_query(user_query,chat_history)
        condensed_query = enhanced_output['condensed_query']
        enhanced_query = enhanced_output['enhanced_query'].get("rewritten_query", user_query)

        # 2. Retrieve
        retrieved_docs = self.retrieve_documents(enhanced_query)

        # 3. Rerank (Uses user_original_query for precision)
        reranked_docs = self.rerank_documents(condensed_query, retrieved_docs)

        # 4. Format
        formatted_context = format_docs(reranked_docs)

        # 5. Generate Response (Uses enhanced query for perfect synthesis alignment)
        answer = self.generate_answer(
            query=enhanced_query,
            context=formatted_context,
            chat_history=chat_history
        )

        return {
            "answer": answer,
            "documents": reranked_docs,
            "enhanced_query": enhanced_query
        }
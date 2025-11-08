"""Service for answer generation"""
from typing import Dict
from src.services.vector_store_service import get_vector_store_service
from src.core import RAGAnswerSystem


class AnswerService:
    """Service for answer generation"""
    
    def __init__(self):
        self.vector_store_service = get_vector_store_service()
    
    def generate_answer(
        self,
        question: str,
        k: int,
        mode: str,
        semantic_weight: float,
        bm25_weight: float,
        temperature: float,
        max_tokens: int,
        model: str = None,
        use_reranking: bool = None,
        rerank_top_k: int = None
    ) -> Dict:
        """Generate answer using RAG with optional reranking"""
        rag_system = self.vector_store_service.get_rag_system(model=model)
        if rag_system is None:
            raise ValueError("RAG system not available. Check OPENAI_API_KEY.")
        
        # Get retrieval system to use reranking
        retrieval_system = self.vector_store_service.get_retrieval_system()
        if retrieval_system is None:
            raise ValueError("Retrieval system not available.")
        
        # Perform retrieval with reranking
        retrieved_chunks = retrieval_system.search(
            query=question,
            k=k,
            mode=mode,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            use_reranking=use_reranking,
            rerank_top_k=rerank_top_k
        )
        
        # Format chunks for LLM (convert to expected format)
        formatted_chunks = []
        for chunk in retrieved_chunks:
            formatted_chunks.append({
                'text': chunk['text'],
                'metadata': chunk.get('metadata', {}),
                'score': chunk.get('score', 0.0')
            })
        
        # Generate answer with retrieved chunks
        from src.core.llm import LLMAnswerGenerator
        from src.config import get_settings
        
        settings = get_settings()
        llm_generator = LLMAnswerGenerator(
            api_key=settings.OPENAI_API_KEY,
            model=model or settings.DEFAULT_LLM_MODEL
        )
        
        return llm_generator.generate_answer(
            query=question,
            context_chunks=formatted_chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )


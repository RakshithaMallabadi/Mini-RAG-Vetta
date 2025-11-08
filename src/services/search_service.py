"""Service for search operations"""
from typing import List, Dict
from src.services.vector_store_service import get_vector_store_service
from src.core import RetrievalSystem


class SearchService:
    """Service for search operations"""
    
    def __init__(self):
        self.vector_store_service = get_vector_store_service()
    
    def search(
        self,
        query: str,
        k: int,
        mode: str,
        semantic_weight: float,
        bm25_weight: float,
        use_reranking: bool = None,
        rerank_top_k: int = None
    ) -> List[Dict]:
        """Perform search with optional reranking"""
        retrieval_system = self.vector_store_service.get_retrieval_system()
        if retrieval_system is None:
            raise ValueError("Vector store not loaded")
        
        return retrieval_system.search(
            query=query,
            k=k,
            mode=mode,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            use_reranking=use_reranking,
            rerank_top_k=rerank_top_k
        )


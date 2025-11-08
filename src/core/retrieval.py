"""
Retrieval Module for Mini RAG System
Implements semantic search and hybrid retrieval (BM25 + semantic)
"""

import re
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from src.core.embeddings import EmbeddingVectorStore
from src.core.reranking import RerankingSystem
from src.config import get_settings


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25
    Converts text to lowercase and splits on whitespace and punctuation
    """
    # Convert to lowercase and split on non-word characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


class BM25Retriever:
    """BM25-based keyword retrieval"""
    
    def __init__(self, chunks: List[Dict]):
        """
        Initialize BM25 retriever with document chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25 is required. Install it with: pip install rank-bm25"
            )
        
        self.chunks = chunks
        self.texts = [chunk['text'] for chunk in chunks]
        
        # Tokenize all documents
        tokenized_docs = [tokenize(text) for text in self.texts]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search using BM25
        
        Args:
            query: Query text string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing:
            - text: Chunk text
            - score: BM25 score
            - metadata: Original chunk metadata
        """
        # Tokenize query
        query_tokens = tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                results.append({
                    'text': self.chunks[idx]['text'],
                    'score': float(scores[idx]),
                    'metadata': {
                        'chunk_index': self.chunks[idx].get('chunk_index', idx),
                        'source_file': self.chunks[idx].get('source_file', ''),
                        'source_path': self.chunks[idx].get('source_path', ''),
                        'tokens': self.chunks[idx].get('tokens', 0)
                    }
                })
        
        return results


class HybridRetriever:
    """Hybrid retrieval combining BM25 and semantic search"""
    
    def __init__(self, embedding_vector_store: EmbeddingVectorStore, chunks: List[Dict]):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_vector_store: EmbeddingVectorStore instance for semantic search
            chunks: List of chunk dictionaries
        """
        self.embedding_store = embedding_vector_store
        self.bm25_retriever = BM25Retriever(chunks)
        self.chunks = chunks
    
    def search(self, 
               query: str, 
               k: int = 5, 
               semantic_weight: float = 0.7,
               bm25_weight: float = 0.3,
               normalize_scores: bool = True) -> List[Dict]:
        """
        Hybrid search combining BM25 and semantic search
        
        Args:
            query: Query text string
            k: Number of results to return
            semantic_weight: Weight for semantic search scores (default: 0.7)
            bm25_weight: Weight for BM25 scores (default: 0.3)
            normalize_scores: Whether to normalize scores before combining (default: True)
            
        Returns:
            List of dictionaries containing:
            - text: Chunk text
            - score: Combined hybrid score
            - semantic_score: Semantic search score
            - bm25_score: BM25 score
            - metadata: Original chunk metadata
        """
        # Get semantic search results
        semantic_results = self.embedding_store.search(query, k=k * 2)  # Get more for better merging
        
        # Get BM25 results
        bm25_results = self.bm25_retriever.search(query, k=k * 2)
        
        # Create dictionaries for easy lookup
        semantic_dict = {result['metadata'].get('chunk_index', i): result 
                        for i, result in enumerate(semantic_results)}
        bm25_dict = {result['metadata'].get('chunk_index', i): result 
                    for i, result in enumerate(bm25_results)}
        
        # Get all unique chunk indices
        all_indices = set(semantic_dict.keys()) | set(bm25_dict.keys())
        
        # Normalize scores if requested
        if normalize_scores:
            semantic_scores = [r['score'] for r in semantic_results if r['score'] > 0]
            bm25_scores = [r['score'] for r in bm25_results if r['score'] > 0]
            
            if semantic_scores:
                semantic_max = max(semantic_scores)
                semantic_min = min(semantic_scores)
                semantic_range = semantic_max - semantic_min if semantic_max != semantic_min else 1
            else:
                semantic_range = 1
                semantic_min = 0
            
            if bm25_scores:
                bm25_max = max(bm25_scores)
                bm25_min = min(bm25_scores)
                bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
            else:
                bm25_range = 1
                bm25_min = 0
        else:
            semantic_range = 1
            semantic_min = 0
            bm25_range = 1
            bm25_min = 0
        
        # Combine scores
        combined_results = []
        for idx in all_indices:
            semantic_result = semantic_dict.get(idx)
            bm25_result = bm25_dict.get(idx)
            
            # Get normalized scores
            if semantic_result:
                semantic_score = semantic_result['score']
                if normalize_scores:
                    semantic_score = (semantic_score - semantic_min) / semantic_range
            else:
                semantic_score = 0.0
            
            if bm25_result:
                bm25_score = bm25_result['score']
                if normalize_scores:
                    bm25_score = (bm25_score - bm25_min) / bm25_range
            else:
                bm25_score = 0.0
            
            # Combine scores
            hybrid_score = (semantic_weight * semantic_score) + (bm25_weight * bm25_score)
            
            # Get metadata from either result
            metadata = semantic_result['metadata'] if semantic_result else bm25_result['metadata']
            text = semantic_result['text'] if semantic_result else bm25_result['text']
            
            combined_results.append({
                'text': text,
                'score': hybrid_score,
                'semantic_score': semantic_result['score'] if semantic_result else 0.0,
                'bm25_score': bm25_result['score'] if bm25_result else 0.0,
                'metadata': metadata
            })
        
        # Sort by hybrid score and return top k
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        return combined_results[:k]
    
    def search_semantic_only(self, query: str, k: int = 5) -> List[Dict]:
        """Semantic search only"""
        return self.embedding_store.search(query, k=k)
    
    def search_bm25_only(self, query: str, k: int = 5) -> List[Dict]:
        """BM25 search only"""
        return self.bm25_retriever.search(query, k=k)


class RetrievalSystem:
    """Unified retrieval system supporting semantic and hybrid search with optional reranking"""
    
    def __init__(self, 
                 embedding_vector_store: EmbeddingVectorStore,
                 reranking_enabled: Optional[bool] = None,
                 reranker_model: Optional[str] = None):
        """
        Initialize retrieval system
        
        Args:
            embedding_vector_store: EmbeddingVectorStore instance
            reranking_enabled: Whether to enable reranking (None = use settings default)
            reranker_model: Reranker model name (None = use settings default)
        """
        self.embedding_store = embedding_vector_store
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self._chunks_cache: Optional[List[Dict]] = None
        
        # Initialize reranking system
        settings = get_settings()
        enabled = reranking_enabled if reranking_enabled is not None else settings.RERANKING_ENABLED
        model = reranker_model or settings.RERANKER_MODEL
        self.reranking_system = RerankingSystem(
            model_name=model,
            enabled=enabled
        )
    
    def _get_chunks(self) -> List[Dict]:
        """Get chunks from vector store metadata"""
        if self._chunks_cache is None:
            # Load chunks from metadata
            import pickle
            import os
            
            metadata_path = "faiss_metadata.pkl"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self._chunks_cache = metadata
            else:
                self._chunks_cache = []
        
        return self._chunks_cache
    
    def _initialize_hybrid_retriever(self):
        """Initialize hybrid retriever if not already done"""
        if self.hybrid_retriever is None:
            chunks = self._get_chunks()
            if not chunks:
                raise ValueError("No chunks available. Process documents first.")
            self.hybrid_retriever = HybridRetriever(self.embedding_store, chunks)
    
    def search(self, 
               query: str, 
               k: int = 5,
               mode: str = "semantic",
               semantic_weight: float = 0.7,
               bm25_weight: float = 0.3,
               use_reranking: Optional[bool] = None,
               rerank_top_k: Optional[int] = None) -> List[Dict]:
        """
        Search with different retrieval modes and optional reranking
        
        Args:
            query: Query text string
            k: Number of results to return
            mode: Retrieval mode - "semantic", "bm25", or "hybrid" (default: "semantic")
            semantic_weight: Weight for semantic search in hybrid mode (default: 0.7)
            bm25_weight: Weight for BM25 in hybrid mode (default: 0.3)
            use_reranking: Whether to use reranking (None = use system default)
            rerank_top_k: Number of documents to retrieve before reranking (None = k * 2)
            
        Returns:
            List of search results (reranked if reranking is enabled)
        """
        # Determine if reranking should be used
        should_rerank = use_reranking if use_reranking is not None else self.reranking_system.is_available()
        
        # If reranking is enabled, retrieve more documents first
        if should_rerank:
            initial_k = rerank_top_k if rerank_top_k is not None else (k * 2)
        else:
            initial_k = k
        
        # Perform initial retrieval
        if mode == "semantic":
            results = self.embedding_store.search(query, k=initial_k)
        
        elif mode == "bm25":
            self._initialize_hybrid_retriever()
            results = self.hybrid_retriever.search_bm25_only(query, k=initial_k)
        
        elif mode == "hybrid":
            self._initialize_hybrid_retriever()
            results = self.hybrid_retriever.search(
                query, 
                k=initial_k, 
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight
            )
        
        else:
            raise ValueError(f"Unknown search mode: {mode}. Use 'semantic', 'bm25', or 'hybrid'")
        
        # Apply reranking if enabled
        if should_rerank and results:
            results = self.reranking_system.rerank(query, results, top_k=k)
        
        # Return top k results
        return results[:k]
    
    def update_chunks(self, chunks: List[Dict]):
        """Update chunks cache and reinitialize hybrid retriever"""
        self._chunks_cache = chunks
        self.hybrid_retriever = None
        if chunks:
            self._initialize_hybrid_retriever()


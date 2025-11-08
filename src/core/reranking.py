"""
Reranking Module for Mini RAG System
Implements BGE-reranker for second-stage reranking of retrieved results
"""

from typing import List, Dict, Optional
import os

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None


class BGEReranker:
    """BGE-reranker for reranking search results"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize BGE reranker
        
        Args:
            model_name: HuggingFace model name for BGE reranker
                       Options:
                       - "BAAI/bge-reranker-base" (default, ~110M params)
                       - "BAAI/bge-reranker-large" (~330M params, better but slower)
                       - "BAAI/bge-reranker-v2-m3" (multilingual)
        """
        if FlagReranker is None:
            raise ImportError(
                "FlagEmbedding is required for reranking. "
                "Install it with: pip install FlagEmbedding"
            )
        
        self.model_name = model_name
        self.reranker = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the reranker model"""
        if self.reranker is None:
            try:
                print(f"Loading BGE reranker model: {self.model_name}...")
                self.reranker = FlagReranker(self.model_name, use_fp16=False)
                print(f"✓ BGE reranker loaded successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load BGE reranker model {self.model_name}: {e}"
                )
    
    def rerank(self, 
               query: str, 
               documents: List[Dict], 
               top_k: Optional[int] = None) -> List[Dict]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Query text string
            documents: List of document dictionaries with 'text' field
                      Each dict should have at least:
                      - 'text': Document text
                      - 'score': Original retrieval score (optional)
                      - 'metadata': Metadata dict (optional)
            top_k: Number of top results to return (None = return all)
            
        Returns:
            List of reranked documents with updated 'score' field
            Documents are sorted by relevance (highest first)
        """
        if not documents:
            return []
        
        if self.reranker is None:
            self._load_model()
        
        # Extract texts for reranking
        texts = [doc.get('text', '') for doc in documents]
        
        # Prepare pairs for reranking: (query, document)
        pairs = [[query, text] for text in texts]
        
        try:
            # Get reranking scores
            scores = self.reranker.compute_score(pairs)
            
            # Handle different return types from FlagReranker
            if isinstance(scores, list):
                rerank_scores = scores
            elif hasattr(scores, 'tolist'):
                rerank_scores = scores.tolist()
            else:
                rerank_scores = [float(scores)] if len(pairs) == 1 else list(scores)
            
            # Update documents with reranking scores
            reranked_docs = []
            for i, doc in enumerate(documents):
                new_doc = doc.copy()
                new_doc['rerank_score'] = float(rerank_scores[i])
                # Keep original score for reference
                new_doc['original_score'] = doc.get('score', 0.0)
                # Use rerank_score as the primary score
                new_doc['score'] = new_doc['rerank_score']
                reranked_docs.append(new_doc)
            
            # Sort by rerank_score (descending)
            reranked_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top_k if specified
            if top_k is not None:
                return reranked_docs[:top_k]
            
            return reranked_docs
            
        except Exception as e:
            raise RuntimeError(f"Error during reranking: {e}")
    
    def rerank_batch(self,
                    queries: List[str],
                    documents_list: List[List[Dict]],
                    top_k: Optional[int] = None) -> List[List[Dict]]:
        """
        Batch rerank multiple queries
        
        Args:
            queries: List of query strings
            documents_list: List of document lists (one per query)
            top_k: Number of top results per query
            
        Returns:
            List of reranked document lists
        """
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k=top_k)
            results.append(reranked)
        return results


class RerankingSystem:
    """Wrapper for reranking functionality with optional usage"""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 enabled: bool = True):
        """
        Initialize reranking system
        
        Args:
            model_name: BGE reranker model name (None = use default)
            enabled: Whether reranking is enabled (default: True)
        """
        self.enabled = enabled
        self.reranker: Optional[BGEReranker] = None
        
        if self.enabled:
            try:
                model = model_name or "BAAI/bge-reranker-base"
                self.reranker = BGEReranker(model_name=model)
            except ImportError:
                print("Warning: FlagEmbedding not available. Reranking disabled.")
                self.enabled = False
                self.reranker = None
            except Exception as e:
                print(f"Warning: Could not initialize reranker: {e}. Reranking disabled.")
                self.enabled = False
                self.reranker = None
    
    def rerank(self,
               query: str,
               documents: List[Dict],
               top_k: Optional[int] = None) -> List[Dict]:
        """
        Rerank documents if enabled, otherwise return original documents
        
        Args:
            query: Query text
            documents: List of document dictionaries
            top_k: Number of top results to return
            
        Returns:
            Reranked documents if enabled, otherwise original documents
        """
        if not self.enabled or self.reranker is None:
            # Return original documents sorted by score
            sorted_docs = sorted(documents, key=lambda x: x.get('score', 0.0), reverse=True)
            if top_k is not None:
                return sorted_docs[:top_k]
            return sorted_docs
        
        return self.reranker.rerank(query, documents, top_k=top_k)
    
    def is_available(self) -> bool:
        """Check if reranking is available and enabled"""
        return self.enabled and self.reranker is not None


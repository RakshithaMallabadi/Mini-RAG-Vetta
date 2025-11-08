"""Service for managing vector store operations"""
import os
import pickle
from typing import Optional
from pathlib import Path

from src.core import EmbeddingVectorStore, RetrievalSystem, RAGAnswerSystem
from src.config import get_settings


class VectorStoreService:
    """Service for vector store management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_store: Optional[EmbeddingVectorStore] = None
        self.retrieval_system: Optional[RetrievalSystem] = None
        self.rag_system: Optional[RAGAnswerSystem] = None
    
    def load_vector_store(self) -> bool:
        """Load the vector store if it exists"""
        if self.vector_store is None:
            index_path = self.settings.INDEX_PATH
            metadata_path = self.settings.METADATA_PATH
            
            if index_path.exists() and metadata_path.exists():
                try:
                    self.vector_store = EmbeddingVectorStore()
                    self.vector_store.load(str(index_path), str(metadata_path))
                    
                    # Initialize retrieval system
                    self.retrieval_system = RetrievalSystem(self.vector_store)
                    
                    # Load chunks for BM25
                    if metadata_path.exists():
                        with open(metadata_path, 'rb') as f:
                            chunks = pickle.load(f)
                        self.retrieval_system.update_chunks(chunks)
                    
                    # Initialize RAG system if API key is available
                    if self.settings.OPENAI_API_KEY:
                        try:
                            self.rag_system = RAGAnswerSystem(
                                self.retrieval_system,
                                api_key=self.settings.OPENAI_API_KEY
                            )
                        except Exception as e:
                            print(f"Warning: Could not initialize RAG system: {e}")
                    
                    return True
                except Exception as e:
                    print(f"Error loading vector store: {e}")
                    return False
        return self.vector_store is not None
    
    def get_vector_store(self) -> Optional[EmbeddingVectorStore]:
        """Get the vector store instance"""
        self.load_vector_store()
        return self.vector_store
    
    def get_retrieval_system(self) -> Optional[RetrievalSystem]:
        """Get the retrieval system instance"""
        self.load_vector_store()
        if self.retrieval_system is None and self.vector_store is not None:
            import pickle
            self.retrieval_system = RetrievalSystem(self.vector_store)
            if self.settings.METADATA_PATH.exists():
                with open(self.settings.METADATA_PATH, 'rb') as f:
                    chunks = pickle.load(f)
                self.retrieval_system.update_chunks(chunks)
        return self.retrieval_system
    
    def get_rag_system(self, model: Optional[str] = None) -> Optional[RAGAnswerSystem]:
        """Get the RAG system instance"""
        self.load_vector_store()
        
        # Normalize model parameter - handle None, empty string, or invalid values
        if not model or model.strip() == '' or model == 'string':
            model = self.settings.DEFAULT_LLM_MODEL
        else:
            model = model.strip()
        
        # Check if we need to create or recreate RAG system
        # If model changed or system doesn't exist, create/recreate it
        if self.settings.OPENAI_API_KEY:
            if self.retrieval_system is None:
                self.retrieval_system = RetrievalSystem(self.vector_store)
                metadata_path = self.settings.METADATA_PATH
                if metadata_path.exists():
                    with open(metadata_path, 'rb') as f:
                        chunks = pickle.load(f)
                    self.retrieval_system.update_chunks(chunks)
            
            # Always create RAG system with the requested model
            # (Note: This creates a new instance each time, which is fine for now)
            self.rag_system = RAGAnswerSystem(
                self.retrieval_system,
                api_key=self.settings.OPENAI_API_KEY,
                model=model
            )
        
        return self.rag_system
    
    def reset(self):
        """Reset the vector store"""
        index_path = self.settings.INDEX_PATH
        metadata_path = self.settings.METADATA_PATH
        
        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        self.vector_store = None
        self.retrieval_system = None
        self.rag_system = None


# Global service instance
_vector_store_service = None


def get_vector_store_service() -> VectorStoreService:
    """Get vector store service instance (singleton)"""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service


"""Service for document processing operations"""
import os
import shutil
from pathlib import Path
from typing import List, Dict

from src.core import DocumentProcessor, EmbeddingVectorStore, RetrievalSystem
from src.config import get_settings


class DocumentService:
    """Service for document processing"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def process_directory(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        index_type: str
    ) -> Dict:
        """Process all documents in the documents directory"""
        if not self.settings.DOCUMENTS_DIR.exists():
            raise ValueError(f"Documents directory not found: {self.settings.DOCUMENTS_DIR}")
        
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = processor.process_directory(str(self.settings.DOCUMENTS_DIR))
        
        if not chunks:
            raise ValueError("No documents found or processed.")
        
        # Create vector store
        vector_store = EmbeddingVectorStore(
            embedding_model=embedding_model,
            index_type=index_type
        )
        vector_store.add_chunks(chunks)
        vector_store.save(
            str(self.settings.INDEX_PATH),
            str(self.settings.METADATA_PATH)
        )
        
        stats = vector_store.get_stats()
        
        return {
            "chunks": chunks,
            "vector_store": vector_store,
            "stats": stats
        }
    
    def process_uploaded_file(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        index_type: str,
        existing_vector_store: EmbeddingVectorStore = None
    ) -> Dict:
        """Process a single uploaded file"""
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = processor.process_document(file_path)
        
        if not chunks:
            raise ValueError("No chunks generated from document")
        
        # Load or create vector store
        if existing_vector_store is None:
            if (self.settings.INDEX_PATH.exists() and 
                self.settings.METADATA_PATH.exists()):
                vector_store = EmbeddingVectorStore(
                    embedding_model=embedding_model,
                    index_type=index_type
                )
                vector_store.load(
                    str(self.settings.INDEX_PATH),
                    str(self.settings.METADATA_PATH)
                )
            else:
                vector_store = EmbeddingVectorStore(
                    embedding_model=embedding_model,
                    index_type=index_type
                )
        else:
            vector_store = existing_vector_store
        
        vector_store.add_chunks(chunks)
        vector_store.save(
            str(self.settings.INDEX_PATH),
            str(self.settings.METADATA_PATH)
        )
        
        stats = vector_store.get_stats()
        
        return {
            "chunks": chunks,
            "vector_store": vector_store,
            "stats": stats
        }
    
    def validate_file_extension(self, filename: str) -> bool:
        """Validate file extension"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.settings.ALLOWED_EXTENSIONS


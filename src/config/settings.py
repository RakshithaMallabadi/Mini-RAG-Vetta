"""Application settings and configuration"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""
    
    # API Settings
    API_TITLE = "Mini RAG 2 API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "REST API for document processing, embeddings, semantic search, and LLM answering"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DOCUMENTS_DIR = BASE_DIR / "documents"
    # Use data directory if it exists (for Docker volumes), otherwise use base dir
    DATA_DIR = BASE_DIR / "data"
    # Ensure data directory exists
    if not DATA_DIR.exists():
        DATA_DIR = BASE_DIR
    INDEX_PATH = DATA_DIR / "faiss_index.bin"
    METADATA_PATH = DATA_DIR / "faiss_metadata.pkl"
    
    # Default Processing Settings
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_INDEX_TYPE = "flat"
    
    # Default Retrieval Settings
    DEFAULT_K = 5
    DEFAULT_MODE = "semantic"
    DEFAULT_SEMANTIC_WEIGHT = 0.7
    DEFAULT_BM25_WEIGHT = 0.3
    
    # Default LLM Settings
    DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 500
    
    # Reranking Settings
    RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    DEFAULT_RERANK_TOP_K = None  # None = rerank all retrieved documents
    
    # Thread Pool Settings
    THREAD_POOL_WORKERS = 2
    
    # Allowed File Extensions
    ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.html', '.htm']
    
    # Environment Variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        # Ensure data directory exists if using it
        if cls.DATA_DIR != cls.BASE_DIR:
            cls.DATA_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get settings instance (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


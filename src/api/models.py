"""Pydantic models for API requests and responses"""
from pydantic import BaseModel, field_validator
from typing import List, Optional


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str
    k: int = 5
    mode: str = "semantic"  # "semantic", "bm25", or "hybrid"
    semantic_weight: float = 0.7  # For hybrid mode
    bm25_weight: float = 0.3  # For hybrid mode
    use_reranking: Optional[bool] = None  # None = use system default
    rerank_top_k: Optional[int] = None  # Number of docs to retrieve before reranking (None = k * 2)


class SearchResult(BaseModel):
    """Single search result"""
    text: str
    score: float
    metadata: dict
    semantic_score: Optional[float] = None  # For hybrid mode
    bm25_score: Optional[float] = None  # For hybrid mode
    rerank_score: Optional[float] = None  # Reranking score if reranking was used
    original_score: Optional[float] = None  # Original retrieval score before reranking


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    query: str
    results: List[SearchResult]
    total_results: int


class AnswerRequest(BaseModel):
    """Request model for answer endpoint"""
    question: str
    k: int = 5
    mode: str = "semantic"  # "semantic", "bm25", or "hybrid"
    semantic_weight: float = 0.7  # For hybrid mode
    bm25_weight: float = 0.3  # For hybrid mode
    temperature: float = 0.7
    max_tokens: int = 500
    model: Optional[str] = None  # Override default model
    use_reranking: Optional[bool] = None  # None = use system default
    rerank_top_k: Optional[int] = None  # Number of docs to retrieve before reranking (None = k * 2)
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        """Normalize model field - convert empty string or 'string' to None"""
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v == '' or v == 'string':
                return None
        return v


class Citation(BaseModel):
    """Citation model"""
    number: int
    source_file: str
    chunk_index: int
    text: str
    score: float


class AnswerResponse(BaseModel):
    """Response model for answer endpoint"""
    question: str
    answer: str
    citations: List[int]
    sources: List[str]
    context_chunks: List[Citation]


class ProcessResponse(BaseModel):
    """Response model for document processing"""
    message: str
    chunks_processed: int
    total_vectors: int


class StatsResponse(BaseModel):
    """Response model for stats endpoint"""
    total_vectors: int
    embedding_dim: int
    embedding_model: str
    index_type: str
    index_path: str
    metadata_path: str
    
    class Config:
        protected_namespaces = ()


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    vector_store_loaded: bool


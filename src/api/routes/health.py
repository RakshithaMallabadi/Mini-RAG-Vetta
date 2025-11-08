"""Health check and stats routes"""
from fastapi import APIRouter, HTTPException
from src.api.models import HealthResponse, StatsResponse
from src.services.vector_store_service import get_vector_store_service
from src.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    vector_store_service = get_vector_store_service()
    is_loaded = vector_store_service.load_vector_store()
    return HealthResponse(
        status="healthy",
        vector_store_loaded=is_loaded
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the vector store"""
    vector_store_service = get_vector_store_service()
    if not vector_store_service.load_vector_store():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Process documents first."
        )
    
    settings = get_settings()
    vector_store = vector_store_service.get_vector_store()
    stats = vector_store.get_stats()
    
    return StatsResponse(
        total_vectors=stats['total_vectors'],
        embedding_dim=stats['embedding_dim'],
        embedding_model=stats['model_name'],
        index_type=stats['index_type'],
        index_path=str(settings.INDEX_PATH),
        metadata_path=str(settings.METADATA_PATH)
    )


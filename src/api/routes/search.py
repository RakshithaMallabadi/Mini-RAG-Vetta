"""Search routes"""
import sys
import traceback
from fastapi import APIRouter, HTTPException
from src.api.models import SearchRequest, SearchResponse, SearchResult
from src.services.search_service import SearchService
from src.services.vector_store_service import get_vector_store_service

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search the vector store for similar documents"""
    vector_store_service = get_vector_store_service()
    if not vector_store_service.load_vector_store():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Process documents first using /process or /upload"
        )
    
    # Validate mode
    if request.mode not in ["semantic", "bm25", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'semantic', 'bm25', or 'hybrid'"
        )
    
    # Validate weights for hybrid mode
    if request.mode == "hybrid":
        if abs(request.semantic_weight + request.bm25_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="semantic_weight + bm25_weight should equal 1.0"
            )
    
    try:
        search_service = SearchService()
        results = search_service.search(
            query=request.query,
            k=request.k,
            mode=request.mode,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight,
            use_reranking=request.use_reranking,
            rerank_top_k=request.rerank_top_k
        )
        
        search_results = [
            SearchResult(
                text=result['text'],
                score=result['score'],
                metadata=result['metadata'],
                semantic_score=result.get('semantic_score'),
                bm25_score=result.get('bm25_score'),
                rerank_score=result.get('rerank_score'),
                original_score=result.get('original_score')
            )
            for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
    
    except Exception as e:
        print(f"Error in /search: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Error during search: {str(e)}"
        )


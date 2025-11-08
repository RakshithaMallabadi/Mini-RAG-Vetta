"""Answer generation routes"""
import sys
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from src.api.models import AnswerRequest, AnswerResponse, Citation
from src.services.answer_service import AnswerService
from src.services.vector_store_service import get_vector_store_service
from src.config import get_settings

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=2)


@router.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest):
    """Answer question using RAG"""
    vector_store_service = get_vector_store_service()
    if not vector_store_service.load_vector_store():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Process documents first."
        )
    
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set"
        )
    
    # validate mode
    if request.mode not in ["semantic", "bm25", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Use 'semantic', 'bm25', or 'hybrid'"
        )
    
    # check weights for hybrid
    if request.mode == "hybrid":
        if abs(request.semantic_weight + request.bm25_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail="semantic_weight + bm25_weight should equal 1.0"
            )
    
    try:
        answer_service = AnswerService()
        loop = asyncio.get_event_loop()
        
        # handle empty model string (swagger UI sends 'string' sometimes)
        model = request.model
        if model and (model.strip() == '' or model.strip() == 'string'):
            model = None
        
        def generate_answer():
            """run in executor to avoid blocking"""
            try:
                return answer_service.generate_answer(
                    question=request.question,
                    k=request.k,
                    mode=request.mode,
                    semantic_weight=request.semantic_weight,
                    bm25_weight=request.bm25_weight,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    model=model,
                    use_reranking=request.use_reranking,
                    rerank_top_k=request.rerank_top_k
                )
            except Exception as e:
                import traceback
                error_msg = str(e) if str(e) else repr(e)
                traceback.print_exc()
                raise Exception(f"Error generating answer: {error_msg}")
        
        result = await loop.run_in_executor(executor, generate_answer)
        
        # Format response
        citations = [
            Citation(
                number=i + 1,
                source_file=chunk['source_file'],
                chunk_index=chunk['chunk_index'],
                text=chunk['text'],
                score=chunk['score']
            )
            for i, chunk in enumerate(result['context_chunks'])
        ]
        
        return AnswerResponse(
            question=request.question,
            answer=result['answer'],
            citations=result['citations'],
            sources=result['sources'],
            context_chunks=citations
        )
    
    except Exception as e:
        print(f"Error in /answer: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


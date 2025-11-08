"""Document processing routes"""
import os
import shutil
import asyncio
import sys
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from src.api.models import ProcessResponse
from src.services.document_service import DocumentService
from src.services.vector_store_service import get_vector_store_service
from src.config import get_settings

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=2)


@router.post("/process", response_model=ProcessResponse)
async def process_documents(
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
    index_type: str = Form("flat")
):
    """Process all documents in the documents directory"""
    settings = get_settings()
    document_service = DocumentService()
    vector_store_service = get_vector_store_service()
    
    if not settings.DOCUMENTS_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Documents directory not found: {settings.DOCUMENTS_DIR}"
        )
    
    try:
        loop = asyncio.get_event_loop()
        
        def process_docs():
            """Process documents in a separate thread"""
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    result = document_service.process_directory(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embedding_model=embedding_model,
                        index_type=index_type
                    )
                return result, None
            except Exception as e:
                return None, str(e)
        
        result, error = await loop.run_in_executor(executor, process_docs)
        
        if error:
            raise HTTPException(status_code=500, detail=f"Error processing documents: {error}")
        
        if not result:
            raise HTTPException(
                status_code=400,
                detail="No documents found or processed. Add documents to the documents/ directory."
            )
        
        # Update vector store service
        vector_store_service.vector_store = result["vector_store"]
        vector_store_service.retrieval_system = None  # Will be reloaded on next access
        # Reload retrieval system with new chunks
        import pickle
        if settings.METADATA_PATH.exists():
            with open(settings.METADATA_PATH, 'rb') as f:
                all_chunks = pickle.load(f)
            retrieval_system = vector_store_service.get_retrieval_system()
            if retrieval_system:
                retrieval_system.update_chunks(all_chunks)
        
        return ProcessResponse(
            message="Documents processed successfully",
            chunks_processed=len(result["chunks"]),
            total_vectors=result["stats"]["total_vectors"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in /process: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@router.post("/upload", response_model=ProcessResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
    index_type: str = Form("flat")
):
    """Upload and process a single document"""
    settings = get_settings()
    document_service = DocumentService()
    vector_store_service = get_vector_store_service()
    
    # Validate file extension
    if not document_service.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    file_path = None
    try:
        # Save uploaded file
        file_path = settings.DOCUMENTS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loop = asyncio.get_event_loop()
        
        def process_doc():
            """Process single document in executor"""
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    result = document_service.process_uploaded_file(
                        file_path=str(file_path),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        embedding_model=embedding_model,
                        index_type=index_type,
                        existing_vector_store=vector_store_service.get_vector_store()
                    )
                return result, None
            except Exception as e:
                return None, str(e)
        
        result, error = await loop.run_in_executor(executor, process_doc)
        
        if error:
            if file_path and file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Error processing document: {error}")
        
        if not result:
            if file_path and file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=400, detail="No chunks generated from document")
        
        # Update vector store service
        vector_store_service.vector_store = result["vector_store"]
        vector_store_service.retrieval_system = None
        # Reload retrieval system with new chunks
        import pickle
        if settings.METADATA_PATH.exists():
            with open(settings.METADATA_PATH, 'rb') as f:
                all_chunks = pickle.load(f)
            retrieval_system = vector_store_service.get_retrieval_system()
            if retrieval_system:
                retrieval_system.update_chunks(all_chunks)
        
        return ProcessResponse(
            message=f"Document '{file.filename}' processed and added to vector store",
            chunks_processed=len(result["chunks"]),
            total_vectors=result["stats"]["total_vectors"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        if file_path and file_path.exists():
            file_path.unlink()
        import traceback
        print(f"Error in /upload: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.delete("/reset")
async def reset_vector_store():
    """Reset the vector store"""
    vector_store_service = get_vector_store_service()
    vector_store_service.reset()
    return {
        "message": "Vector store reset successfully",
        "index_deleted": not get_settings().INDEX_PATH.exists(),
        "metadata_deleted": not get_settings().METADATA_PATH.exists()
    }


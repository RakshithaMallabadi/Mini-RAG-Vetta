"""
FastAPI Application for Mini RAG 2 System
Provides REST API endpoints for document processing, embeddings, and search
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv

from data_ingestion import DocumentProcessor
from embeddings import EmbeddingVectorStore
from retrieval import RetrievalSystem
from llm_answering import RAGAnswerSystem

# Load environment variables from .env file
load_dotenv()

# Thread pool executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=2)

# Initialize FastAPI app
app = FastAPI(
    title="Mini RAG 2 API",
    description="REST API for document processing, embeddings, and semantic search",
    version="1.0.0"
)

# Global variables for vector store, retrieval, and LLM
vector_store: Optional[EmbeddingVectorStore] = None
retrieval_system: Optional[RetrievalSystem] = None
rag_system: Optional[RAGAnswerSystem] = None
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
DOCUMENTS_DIR = "./documents/"

# Ensure documents directory exists
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    k: int = 5
    mode: str = "semantic"  # "semantic", "bm25", or "hybrid"
    semantic_weight: float = 0.7  # For hybrid mode
    bm25_weight: float = 0.3  # For hybrid mode


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict
    semantic_score: Optional[float] = None  # For hybrid mode
    bm25_score: Optional[float] = None  # For hybrid mode


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class AnswerRequest(BaseModel):
    question: str
    k: int = 5
    mode: str = "semantic"  # "semantic", "bm25", or "hybrid"
    semantic_weight: float = 0.7  # For hybrid mode
    bm25_weight: float = 0.3  # For hybrid mode
    temperature: float = 0.7
    max_tokens: int = 500
    model: Optional[str] = None  # Override default model


class Citation(BaseModel):
    number: int
    source_file: str
    chunk_index: int
    text: str
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: List[int]
    sources: List[str]
    context_chunks: List[Citation]


class ProcessResponse(BaseModel):
    message: str
    chunks_processed: int
    total_vectors: int


class StatsResponse(BaseModel):
    total_vectors: int
    embedding_dim: int
    embedding_model: str  # Changed from model_name to avoid Pydantic conflict
    index_type: str
    index_path: str
    metadata_path: str
    
    class Config:
        protected_namespaces = ()


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool


def load_vector_store():
    """Load the vector store if it exists"""
    global vector_store, retrieval_system, rag_system
    if vector_store is None:
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            try:
                vector_store = EmbeddingVectorStore()
                vector_store.load(INDEX_PATH, METADATA_PATH)
                
                # Initialize retrieval system
                retrieval_system = RetrievalSystem(vector_store)
                
                # Load chunks for BM25
                import pickle
                if os.path.exists(METADATA_PATH):
                    with open(METADATA_PATH, 'rb') as f:
                        chunks = pickle.load(f)
                    retrieval_system.update_chunks(chunks)
                
                # Initialize RAG system if API key is available
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    try:
                        rag_system = RAGAnswerSystem(retrieval_system, api_key=api_key)
                    except Exception as e:
                        print(f"Warning: Could not initialize RAG system: {e}")
                
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
    return vector_store is not None


@app.on_event("startup")
async def startup_event():
    """Load vector store and pre-load embedding model on startup"""
    print("Starting Mini RAG 2 API...")
    
    # Pre-load embedding model in background to avoid first-request delay
    def preload_model():
        try:
            from embeddings import EmbeddingGenerator
            print("Pre-loading embedding model...")
            generator = EmbeddingGenerator()
            print("✓ Embedding model pre-loaded")
        except Exception as e:
            print(f"⚠ Could not pre-load embedding model: {e}")
    
    # Run in executor to not block startup
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, preload_model)
    
    if load_vector_store():
        print("✓ Vector store loaded successfully")
    else:
        print("⚠ No existing vector store found. Process documents first.")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Mini RAG 2 API",
        "version": "1.0.0",
        "description": "REST API for document processing, embeddings, semantic search, and LLM answering",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "process_documents": "/process",
            "upload_document": "/upload",
            "search": "/search",
            "answer": "/answer",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_loaded = load_vector_store()
    return HealthResponse(
        status="healthy",
        vector_store_loaded=is_loaded
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the vector store"""
    if not load_vector_store():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Process documents first."
        )
    
    stats = vector_store.get_stats()
    return StatsResponse(
        total_vectors=stats['total_vectors'],
        embedding_dim=stats['embedding_dim'],
        embedding_model=stats['model_name'],
        index_type=stats['index_type'],
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )


@app.post("/process", response_model=ProcessResponse)
async def process_documents(
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
    index_type: str = Form("flat")
):
    """
    Process all documents in the documents directory
    
    Parameters:
    - chunk_size: Number of tokens per chunk (default: 512)
    - chunk_overlap: Number of overlapping tokens (default: 50)
    - embedding_model: Sentence transformer model name (default: all-MiniLM-L6-v2)
    - index_type: FAISS index type - "flat" or "ivf" (default: flat)
    """
    global vector_store
    
    # Check if documents directory exists and has files
    if not os.path.exists(DOCUMENTS_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"Documents directory not found: {DOCUMENTS_DIR}"
        )
    
    # Process documents
    try:
        loop = asyncio.get_event_loop()
        
        # Process documents in executor to avoid blocking and pipe issues
        def process_docs():
            """Process documents in a separate thread"""
            # Capture stdout/stderr to avoid broken pipe issues
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    processor = DocumentProcessor(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = processor.process_directory(DOCUMENTS_DIR)
                return chunks, None
            except Exception as e:
                return None, str(e)
        
        # Run document processing in executor
        chunks, error = await loop.run_in_executor(executor, process_docs)
        
        if error:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing documents: {error}"
            )
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents found or processed. Add documents to the documents/ directory."
            )
        
        # Create vector store in executor (model loading can be slow)
        def create_vector_store():
            """Create vector store in executor"""
            try:
                vs = EmbeddingVectorStore(
                    embedding_model=embedding_model,
                    index_type=index_type
                )
                vs.add_chunks(chunks)
                vs.save(INDEX_PATH, METADATA_PATH)
                return vs.get_stats()
            except Exception as e:
                raise Exception(f"Error creating vector store: {str(e)}")
        
        stats = await loop.run_in_executor(executor, create_vector_store)
        
        # Update global vector store
        vector_store = EmbeddingVectorStore(
            embedding_model=embedding_model,
            index_type=index_type
        )
        vector_store.load(INDEX_PATH, METADATA_PATH)
        
        # Update retrieval system with new chunks
        global retrieval_system
        retrieval_system = RetrievalSystem(vector_store)
        retrieval_system.update_chunks(chunks)
        
        return ProcessResponse(
            message="Documents processed successfully",
            chunks_processed=len(chunks),
            total_vectors=stats['total_vectors']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = str(e)
        # Log full error for debugging (to stderr, not stdout)
        print(f"Error in /process: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing documents: {error_msg}"
        )


@app.post("/upload", response_model=ProcessResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    embedding_model: str = Form("all-MiniLM-L6-v2"),
    index_type: str = Form("flat")
):
    """
    Upload and process a single document
    
    Parameters:
    - file: Document file to upload (PDF, DOCX, TXT, HTML)
    - chunk_size: Number of tokens per chunk (default: 512)
    - chunk_overlap: Number of overlapping tokens (default: 50)
    - embedding_model: Sentence transformer model name (default: all-MiniLM-L6-v2)
    - index_type: FAISS index type - "flat" or "ivf" (default: flat)
    """
    global vector_store
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.html', '.htm']
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loop = asyncio.get_event_loop()
        
        # Process document in executor
        def process_doc():
            """Process single document in executor"""
            f = io.StringIO()
            try:
                with redirect_stdout(f), redirect_stderr(f):
                    processor = DocumentProcessor(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = processor.process_document(file_path)
                return chunks, None
            except Exception as e:
                return None, str(e)
        
        chunks, error = await loop.run_in_executor(executor, process_doc)
        
        if error:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {error}"
            )
        
        if not chunks:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail="No chunks generated from document"
            )
        
        # Load or create vector store in executor
        def add_to_vector_store():
            """Add chunks to vector store in executor"""
            try:
                vs = None
                if vector_store is None:
                    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
                        vs = EmbeddingVectorStore(embedding_model=embedding_model, index_type=index_type)
                        vs.load(INDEX_PATH, METADATA_PATH)
                    else:
                        vs = EmbeddingVectorStore(
                            embedding_model=embedding_model,
                            index_type=index_type
                        )
                else:
                    vs = vector_store
                
                vs.add_chunks(chunks)
                vs.save(INDEX_PATH, METADATA_PATH)
                return vs.get_stats()
            except Exception as e:
                raise Exception(f"Error adding to vector store: {str(e)}")
        
        stats = await loop.run_in_executor(executor, add_to_vector_store)
        
        # Update global vector store
        if vector_store is None:
            vector_store = EmbeddingVectorStore(embedding_model=embedding_model, index_type=index_type)
            vector_store.load(INDEX_PATH, METADATA_PATH)
        
        # Update retrieval system with new chunks
        global retrieval_system
        if retrieval_system is None:
            retrieval_system = RetrievalSystem(vector_store)
        # Reload all chunks including new ones
        import pickle
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'rb') as f:
                all_chunks = pickle.load(f)
            retrieval_system.update_chunks(all_chunks)
        
        return ProcessResponse(
            message=f"Document '{file.filename}' processed and added to vector store",
            chunks_processed=len(chunks),
            total_vectors=stats['total_vectors']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        # Clean up uploaded file on error
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        import traceback
        print(f"Error in /upload: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search the vector store for similar documents
    
    Parameters:
    - query: Search query text
    - k: Number of results to return (default: 5)
    - mode: Retrieval mode - "semantic", "bm25", or "hybrid" (default: "semantic")
    - semantic_weight: Weight for semantic search in hybrid mode (default: 0.7)
    - bm25_weight: Weight for BM25 in hybrid mode (default: 0.3)
    """
    if not load_vector_store():
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
        # Use retrieval system for all modes
        global retrieval_system
        if retrieval_system is None:
            retrieval_system = RetrievalSystem(vector_store)
            # Load chunks
            import pickle
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'rb') as f:
                    chunks = pickle.load(f)
                retrieval_system.update_chunks(chunks)
        
        results = retrieval_system.search(
            query=request.query,
            k=request.k,
            mode=request.mode,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight
        )
        
        search_results = [
            SearchResult(
                text=result['text'],
                score=result['score'],
                metadata=result['metadata'],
                semantic_score=result.get('semantic_score'),
                bm25_score=result.get('bm25_score')
            )
            for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
    
    except Exception as e:
        import traceback
        print(f"Error in /search: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Error during search: {str(e)}"
        )


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation)
    
    Parameters:
    - question: User question
    - k: Number of chunks to retrieve (default: 5)
    - mode: Retrieval mode - "semantic", "bm25", or "hybrid" (default: "semantic")
    - semantic_weight: Weight for semantic search in hybrid mode (default: 0.7)
    - bm25_weight: Weight for BM25 in hybrid mode (default: 0.3)
    - temperature: LLM temperature (default: 0.7)
    - max_tokens: Maximum tokens in response (default: 500)
    - model: LLM model name (optional, uses default if not provided)
    """
    if not load_vector_store():
        raise HTTPException(
            status_code=404,
            detail="Vector store not found. Process documents first using /process or /upload"
        )
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set. Please set it to use the answer endpoint."
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
        # Initialize RAG system if not already done
        global rag_system, retrieval_system
        if rag_system is None:
            if retrieval_system is None:
                retrieval_system = RetrievalSystem(vector_store)
                import pickle
                if os.path.exists(METADATA_PATH):
                    with open(METADATA_PATH, 'rb') as f:
                        chunks = pickle.load(f)
                    retrieval_system.update_chunks(chunks)
            
            rag_system = RAGAnswerSystem(
                retrieval_system,
                api_key=api_key,
                model=request.model or "gpt-3.5-turbo"
            )
        
        # Generate answer
        loop = asyncio.get_event_loop()
        
        def generate_answer():
            """Generate answer in executor"""
            try:
                return rag_system.answer(
                    query=request.question,
                    k=request.k,
                    mode=request.mode,
                    semantic_weight=request.semantic_weight,
                    bm25_weight=request.bm25_weight,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
            except Exception as e:
                raise Exception(f"Error generating answer: {str(e)}")
        
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
        import traceback
        print(f"Error in /answer: {traceback.format_exc()}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


@app.delete("/reset")
async def reset_vector_store():
    """Reset the vector store (delete index and metadata files)"""
    global vector_store
    
    try:
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)
        
        vector_store = None
        retrieval_system = None
        rag_system = None
        
        return {
            "message": "Vector store reset successfully",
            "index_deleted": os.path.exists(INDEX_PATH) == False,
            "metadata_deleted": os.path.exists(METADATA_PATH) == False
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting vector store: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
FastAPI Application for Mini RAG 2 System
Main entry point for the modularized application
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn

from src.api import create_app
from src.config import get_settings
from src.core import EmbeddingGenerator

# Thread pool executor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=2)

# Create FastAPI app
app = create_app()


@app.on_event("startup")
async def startup_event():
    """Load vector store and pre-load embedding model on startup"""
    settings = get_settings()
    print("Starting Mini RAG 2 API...")
    
    # Pre-load embedding model in background to avoid first-request delay
    def preload_model():
        try:
            print("Pre-loading embedding model...")
            generator = EmbeddingGenerator()
            print("✓ Embedding model pre-loaded")
        except Exception as e:
            print(f"⚠ Could not pre-load embedding model: {e}")
    
    # Run in executor to not block startup
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, preload_model)
    
    # Load vector store
    from src.services.vector_store_service import get_vector_store_service
    vector_store_service = get_vector_store_service()
    if vector_store_service.load_vector_store():
        print("✓ Vector store loaded successfully")
    else:
        print("⚠ No existing vector store found. Process documents first.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""API module"""
from fastapi import FastAPI

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    from src.config import get_settings
    settings = get_settings()
    
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION
    )
    
    # Register routes
    from src.api.routes import health, documents, search, answer
    
    app.include_router(health.router, tags=["Health"])
    app.include_router(documents.router, tags=["Documents"])
    app.include_router(search.router, tags=["Search"])
    app.include_router(answer.router, tags=["Answer"])
    
    # Root endpoint
    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint with API information"""
        return {
            "name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "description": settings.API_DESCRIPTION,
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
    
    return app


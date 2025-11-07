#!/bin/bash
# Start the FastAPI server

echo "Starting Mini RAG 2 API Server..."
echo "API will be available at: http://localhost:8000"
echo "Interactive docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload


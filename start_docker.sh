#!/bin/bash
# Script to start Docker container for Mini RAG 2

echo "=== Starting Mini RAG 2 Docker Container ==="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker daemon is not running!"
    echo ""
    echo "Please start Docker:"
    echo "  - macOS: Open Docker Desktop application"
    echo "  - Linux: sudo systemctl start docker"
    exit 1
fi

echo "✅ Docker daemon is running"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "Creating .env file template..."
    echo "OPENAI_API_KEY=your-api-key-here" > .env
    echo "⚠️  Please edit .env file and add your OPENAI_API_KEY"
    echo ""
fi

# Build and start containers
echo "Building and starting containers..."
docker-compose up --build -d

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "📊 Container status:"
    docker-compose ps
    echo ""
    echo "📝 View logs:"
    echo "   docker-compose logs -f"
    echo ""
    echo "🌐 Access API:"
    echo "   http://localhost:8000"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "🛑 Stop container:"
    echo "   docker-compose down"
else
    echo ""
    echo "❌ Failed to start container"
    echo "Check logs with: docker-compose logs"
    exit 1
fi


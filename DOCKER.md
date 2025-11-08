# Docker Deployment Guide

This guide explains how to run the Mini RAG FastAPI application using Docker.

## Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, for easier management)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### Option 2: Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t mini-rag:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name mini-rag-api \
     -p 8000:8000 \
     -e OPENAI_API_KEY=your-api-key-here \
     -v $(pwd)/documents:/app/documents \
     -v $(pwd)/data:/app/data \
     mini-rag:latest
   ```

3. **View logs:**
   ```bash
   docker logs -f mini-rag-api
   ```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` (required for /answer endpoint): Your OpenAI API key
- `PORT` (optional): Port to run the server on (default: 8000)

### Volumes

The Docker setup mounts two volumes:

1. **`./documents:/app/documents`** - Document storage directory
   - Add your documents here for processing
   - Documents persist between container restarts

2. **`./data:/app/data`** - Data directory for vector store
   - FAISS index and metadata files are stored here
   - Persists vector store between restarts

## Docker Commands

### Build
```bash
docker-compose build
# or
docker build -t mini-rag:latest .
```

### Start
```bash
docker-compose up -d
# or
docker start mini-rag-api
```

### Stop
```bash
docker-compose down
# or
docker stop mini-rag-api
```

### View Logs
```bash
docker-compose logs -f
# or
docker logs -f mini-rag-api
```

### Restart
```bash
docker-compose restart
# or
docker restart mini-rag-api
```

### Remove
```bash
docker-compose down -v
# or
docker stop mini-rag-api && docker rm mini-rag-api
```

## Production Deployment

### Using Docker Compose

1. **Create production .env file:**
   ```bash
   OPENAI_API_KEY=your-production-api-key
   ```

2. **Run in detached mode:**
   ```bash
   docker-compose up -d
   ```

3. **Set up reverse proxy (optional):**
   Use nginx or traefik to handle SSL and routing.

### Using Docker Swarm or Kubernetes

The Dockerfile is compatible with orchestration platforms. You'll need to:
- Configure secrets for `OPENAI_API_KEY`
- Set up persistent volumes for documents and data
- Configure health checks (already included)

## Health Checks

The container includes a health check that monitors the `/health` endpoint:
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds (allows time for model loading)

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs mini-rag-api

# Check if port is already in use
lsof -i :8000
```

### Vector store not persisting
- Ensure the `./data` volume is mounted correctly
- Check file permissions in the data directory

### Documents not found
- Ensure the `./documents` volume is mounted
- Check that documents are in the mounted directory

### API key issues
- Verify `OPENAI_API_KEY` is set in environment
- Check logs for authentication errors

## Development with Docker

For development, you can mount the source code:

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  mini-rag-api:
    build: .
    volumes:
      - .:/app
      - ./documents:/app/documents
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Run with:
```bash
docker-compose -f docker-compose.dev.yml up
```

## Image Size Optimization

The Dockerfile uses a multi-stage build to minimize image size:
- Build stage: Installs build dependencies
- Runtime stage: Only includes runtime dependencies

Final image size: ~1.5GB (includes Python, dependencies, and models)

## Security Considerations

1. **Never commit `.env` files** - Use Docker secrets or environment variables
2. **Use non-root user** (optional, can be added to Dockerfile)
3. **Keep dependencies updated** - Regularly rebuild images
4. **Limit resource usage** - Set memory/CPU limits in production

## Example: Full Workflow

```bash
# 1. Clone and navigate
cd Mini_RAG

# 2. Set up environment
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Add documents
cp your-documents/* documents/

# 4. Start container
docker-compose up -d

# 5. Process documents
curl -X POST "http://localhost:8000/process" \
  -F "chunk_size=512" \
  -F "chunk_overlap=50"

# 6. Test search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "k": 5}'

# 7. View logs
docker-compose logs -f
```

## Support

For issues or questions:
- Check logs: `docker logs mini-rag-api`
- Review README.md for general usage
- Check API docs: http://localhost:8000/docs


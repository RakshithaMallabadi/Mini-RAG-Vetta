# Mini RAG - Modular RAG System

A complete, modular Retrieval-Augmented Generation (RAG) system with document processing, semantic search, and LLM-powered question answering.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [API Documentation](#api-documentation)
6. [Retrieval System](#retrieval-system)
7. [LLM Answering](#llm-answering)
8. [Reranking](#reranking)
9. [Docker Deployment](#docker-deployment)
10. [Architecture](#architecture)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)

## Features

- **Multi-format Document Processing**: Extract and process PDF, DOCX, HTML, and TXT files
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Hybrid Retrieval**: Combine semantic search with BM25 for better results
- **LLM Answering**: Generate answers with automatic citation tracking using OpenAI
- **Reranking**: Optional two-stage retrieval with BGE-reranker for improved accuracy
- **REST API**: FastAPI-based API for easy integration
- **Modular Architecture**: Clean, maintainable code structure

## Project Structure

```
Mini_RAG/
├── src/
│   ├── api/              # API layer (routes, models)
│   ├── core/             # Core modules (ingestion, embeddings, retrieval, LLM)
│   ├── services/         # Business logic services
│   └── config/           # Configuration management
├── documents/            # Document storage directory
├── app.py               # FastAPI application entry point
├── run_pipeline.py      # Complete pipeline runner
├── run_demo.py          # Interactive demo
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Mini_RAG
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## Quick Start

### Option 1: Run with Docker (Recommended for Production)

```bash
# Using Docker Compose
docker-compose up --build

# Or using Docker directly
docker build -t mini-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key mini-rag
```

See [Docker Deployment](#docker-deployment) section for detailed instructions.

### Option 2: Run the Demo

```bash
python run_demo.py
```

This will:
1. Process documents in the `documents/` directory
2. Create embeddings and vector store
3. Demonstrate search and Q&A functionality

### Option 3: Use the API

1. **Start the server:**
   ```bash
   uvicorn app:app --reload
   ```
   Or run directly with uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API:**
   - API Base: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

3. **Process documents:**
   ```bash
   curl -X POST "http://localhost:8000/process" \
     -F "chunk_size=512" \
     -F "chunk_overlap=50" \
     -F "embedding_model=all-MiniLM-L6-v2" \
     -F "index_type=flat"
   ```

4. **Ask questions:**
   ```bash
   curl -X POST "http://localhost:8000/answer" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is this about?", "k": 5}'
   ```

### Option 4: Use Programmatically

```python
from src.core.ingestion import DocumentProcessor
from src.core.embeddings import EmbeddingVectorStore
from src.core.retrieval import RetrievalSystem

# Process documents
processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
chunks = processor.process_directory("./documents/")

# Create vector store
vector_store = EmbeddingVectorStore()
vector_store.add_chunks(chunks)
vector_store.save("faiss_index.bin", "faiss_metadata.pkl")

# Search
vector_store.load("faiss_index.bin", "faiss_metadata.pkl")
results = vector_store.search("your query", k=5)
```

## API Documentation

### Quick Start

**Start the Server:**

```bash
# Option 1: Using Python
python app.py

# Option 2: Using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

#### 2. Health Check
**GET** `/health`

Check API health and vector store status.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_loaded": true
}
```

#### 3. Get Statistics
**GET** `/stats`

Get statistics about the vector store.

**Response:**
```json
{
  "total_vectors": 100,
  "embedding_dim": 384,
  "embedding_model": "all-MiniLM-L6-v2",
  "index_type": "flat",
  "index_path": "faiss_index.bin",
  "metadata_path": "faiss_metadata.pkl"
}
```

#### 4. Process Documents
**POST** `/process`

Process all documents in the `documents/` directory.

**Form Parameters:**
- `chunk_size` (int, default: 512): Tokens per chunk
- `chunk_overlap` (int, default: 50): Overlapping tokens
- `embedding_model` (str, default: "all-MiniLM-L6-v2"): Embedding model
- `index_type` (str, default: "flat"): FAISS index type ("flat" or "ivf")

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "chunk_size=512" \
  -F "chunk_overlap=50" \
  -F "embedding_model=all-MiniLM-L6-v2" \
  -F "index_type=flat"
```

**Response:**
```json
{
  "message": "Documents processed successfully",
  "chunks_processed": 10,
  "total_vectors": 10
}
```

#### 5. Upload Document
**POST** `/upload`

Upload and process a single document file.

**Form Parameters:**
- `file` (file, required): Document file (PDF, DOCX, TXT, HTML)
- `chunk_size` (int, default: 512): Tokens per chunk
- `chunk_overlap` (int, default: 50): Overlapping tokens
- `embedding_model` (str, default: "all-MiniLM-L6-v2"): Embedding model
- `index_type` (str, default: "flat"): FAISS index type

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "chunk_size=512" \
  -F "chunk_overlap=50"
```

**Response:**
```json
{
  "message": "Document 'document.pdf' processed and added to vector store",
  "chunks_processed": 5,
  "total_vectors": 15
}
```

#### 6. Answer Questions
**POST** `/answer`

Generate answer with citations using RAG.

**Request Body:**
```json
{
  "question": "What is a workflow job?",
  "k": 5,
  "mode": "semantic",
  "semantic_weight": 0.7,
  "bm25_weight": 0.3,
  "temperature": 0.7,
  "max_tokens": 500,
  "model": "gpt-3.5-turbo",
  "use_reranking": false,
  "rerank_top_k": 20
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a workflow job?",
    "k": 5
  }'
```

**Response:**
```json
{
  "question": "What is a workflow job?",
  "answer": "A workflow job is started using the POST /api/v1/jobs/start endpoint [Citation 1]...",
  "citations": [1],
  "sources": ["api_reference.txt"],
  "context_chunks": [
    {
      "number": 1,
      "source_file": "api_reference.txt",
      "chunk_index": 0,
      "text": "POST /api/v1/jobs/start Starts a new workflow job...",
      "score": 0.85
    }
  ]
}
```

#### 7. Reset Vector Store
**DELETE** `/reset`

Delete the vector store index and metadata files.

**Example using curl:**
```bash
curl -X DELETE "http://localhost:8000/reset"
```

### Client Examples

#### Python Client

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Process documents
response = requests.post(
    f"{BASE_URL}/process",
    data={
        "chunk_size": 512,
        "chunk_overlap": 50
    }
)
print(response.json())

# 2. Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f},
        data={"chunk_size": 512}
    )
print(response.json())

# 3. Ask a question
response = requests.post(
    f"{BASE_URL}/answer",
    json={"question": "your question", "k": 5}
)
results = response.json()
print(f"Answer: {results['answer']}")

# 4. Get statistics
response = requests.get(f"{BASE_URL}/stats")
print(response.json())
```

#### JavaScript/TypeScript Example

```javascript
// Process documents
const processResponse = await fetch('http://localhost:8000/process', {
  method: 'POST',
  body: new FormData().append('chunk_size', '512')
});
const processData = await processResponse.json();

// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('chunk_size', '512');

const uploadResponse = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});
const uploadData = await uploadResponse.json();

// Ask question
const answerResponse = await fetch('http://localhost:8000/answer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: 'your question', k: 5 })
});
const answerData = await answerResponse.json();
```

### Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (vector store not found)
- `500`: Internal Server Error

Error responses include a `detail` field with the error message:

```json
{
  "detail": "Vector store not found. Process documents first."
}
```

## Retrieval System

The system supports three retrieval modes:

### 1. Semantic Search (Default)

Uses sentence embeddings and cosine similarity to find semantically similar documents.

**Best for:**
- Finding documents with similar meaning
- Handling synonyms and related concepts
- Understanding context and intent

**Example:**
```json
{
  "query": "workflow job",
  "k": 5,
  "mode": "semantic"
}
```

### 2. BM25 Search

Uses keyword matching with BM25 ranking algorithm.

**Best for:**
- Exact keyword matching
- Finding documents with specific terms
- Fast keyword-based retrieval

**Example:**
```json
{
  "query": "API endpoint",
  "k": 5,
  "mode": "bm25"
}
```

### 3. Hybrid Search

Combines semantic and BM25 scores with configurable weights.

**Best for:**
- Getting the best of both worlds
- Balancing semantic understanding with keyword matching
- Production use cases requiring robust retrieval

**Example:**
```json
{
  "query": "workflow job",
  "k": 5,
  "mode": "hybrid",
  "semantic_weight": 0.7,
  "bm25_weight": 0.3
}
```

### How Hybrid Search Works

1. **Semantic Search**: Generates embeddings and finds similar vectors using FAISS
2. **BM25 Search**: Tokenizes query and documents, calculates BM25 scores
3. **Score Normalization**: Normalizes both scores to 0-1 range
4. **Score Combination**: Combines scores using weighted average:
   ```
   hybrid_score = (semantic_weight × semantic_score) + (bm25_weight × bm25_score)
   ```
5. **Ranking**: Returns top-k results sorted by hybrid score

### Choosing Weights

- **High semantic_weight (0.7-0.9)**: Better for understanding intent, synonyms, context
- **High bm25_weight (0.7-0.9)**: Better for exact keyword matching, specific terms
- **Balanced (0.5-0.5)**: Good general-purpose setting

## LLM Answering

The system includes LLM-based question answering with automatic citation tracking.

### Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Automatic citation extraction
- ✅ Source file tracking
- ✅ Configurable retrieval modes (semantic, BM25, hybrid)
- ✅ Customizable LLM parameters (temperature, max_tokens, model)

### How It Works

#### 1. Retrieval Phase
- User question is used to retrieve top-k relevant chunks
- Supports semantic, BM25, or hybrid retrieval modes
- Chunks are ranked by relevance score

#### 2. Prompt Construction
- Retrieved chunks are formatted with citation numbers
- Each chunk includes source file and chunk index
- Prompt instructs LLM to cite sources

#### 3. LLM Generation
- LLM generates answer based on retrieved context
- Model is instructed to cite sources using [Citation X] format
- Temperature and max_tokens are configurable

#### 4. Citation Extraction
- System extracts citation numbers from LLM response
- Maps citations to source files and chunks
- Returns structured citation information

### Parameters

#### Retrieval Parameters
- **k**: Number of chunks to retrieve (default: 5)
- **mode**: "semantic", "bm25", or "hybrid" (default: "semantic")
- **semantic_weight**: Weight for semantic search in hybrid mode (default: 0.7)
- **bm25_weight**: Weight for BM25 in hybrid mode (default: 0.3)

#### LLM Parameters
- **temperature**: Sampling temperature, 0.0-2.0 (default: 0.7)
  - Lower = more focused, deterministic
  - Higher = more creative, diverse
- **max_tokens**: Maximum tokens in response (default: 500)
- **model**: Model name (default: "gpt-3.5-turbo")
  - Options: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", etc.

### Best Practices

1. **Retrieval Mode Selection**:
   - Use "semantic" for conceptual questions
   - Use "bm25" for exact keyword matching
   - Use "hybrid" for best overall results

2. **Chunk Count (k)**:
   - Start with k=5 for most questions
   - Increase to k=10 for complex questions
   - Decrease to k=3 for simple questions

3. **Temperature**:
   - Use 0.3-0.5 for factual, precise answers
   - Use 0.7 for balanced responses
   - Use 0.9+ for creative, exploratory answers

4. **Model Selection**:
   - gpt-3.5-turbo: Fast, cost-effective, good for most use cases
   - gpt-4: More accurate, better reasoning, higher cost
   - gpt-4-turbo: Best quality, highest cost

## Reranking

Reranking is a two-stage retrieval approach that improves search quality.

### Overview

1. **First Stage**: Initial retrieval using semantic search, BM25, or hybrid search
2. **Second Stage**: Reranking of retrieved results using a cross-encoder model (BGE-reranker)

### Why Reranking?

- **Better Relevance**: Cross-encoder models consider query-document pairs together, providing more accurate relevance scores
- **Improved Accuracy**: Reranking can significantly improve the quality of top-k results
- **Flexible**: Can be enabled/disabled per request or globally

### BGE-Reranker

The system uses **BGE-reranker** from FlagEmbedding, which is a state-of-the-art reranking model developed by BAAI.

**Available Models:**
- `BAAI/bge-reranker-base` (default, ~110M parameters) - Fast and efficient
- `BAAI/bge-reranker-large` (~330M parameters) - Better accuracy but slower
- `BAAI/bge-reranker-v2-m3` - Multilingual support

### Architecture

```
Query → Initial Retrieval (Semantic/BM25/Hybrid) → Reranker → Final Top-K Results
```

1. **Initial Retrieval**: Retrieves more documents (typically k * 2) using semantic/BM25/hybrid search
2. **Reranking**: BGE-reranker scores each query-document pair
3. **Final Selection**: Returns top-k documents based on reranking scores

### Configuration

#### Environment Variables

```bash
# Enable/disable reranking globally (default: false)
RERANKING_ENABLED=false

# Reranker model name (default: BAAI/bge-reranker-base)
RERANKER_MODEL=BAAI/bge-reranker-base
```

#### API Parameters

- **`use_reranking`** (optional): 
  - `true`: Enable reranking for this request
  - `false`: Disable reranking for this request
  - `null`/omitted: Use system default (`RERANKING_ENABLED`)

- **`rerank_top_k`** (optional):
  - Number of documents to retrieve in first stage before reranking
  - If `null`/omitted: Defaults to `k * 2`
  - Example: If `k=5` and `rerank_top_k=20`, retrieve 20 docs, rerank them, return top 5

### Performance Considerations

- **Without Reranking**: Faster, but may miss highly relevant documents
- **With Reranking**: Slower (adds ~100-500ms depending on model and number of documents), but significantly better accuracy

### Recommendations

1. **For Production**: Use `bge-reranker-base` for good balance of speed and accuracy
2. **For High Accuracy**: Use `bge-reranker-large` if latency is acceptable
3. **For Speed**: Disable reranking if sub-100ms latency is critical
4. **Hybrid Approach**: Use reranking only for important queries, disable for simple keyword searches

### Installation

Reranking requires the `FlagEmbedding` package:

```bash
pip install FlagEmbedding>=1.2.0
```

## Docker Deployment

### Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- Docker Compose (optional, for easier management)

### Quick Start

#### Option 1: Using Docker Compose (Recommended)

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

#### Option 2: Using Docker directly

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

### Configuration

#### Environment Variables

- `OPENAI_API_KEY` (required for /answer endpoint): Your OpenAI API key
- `PORT` (optional): Port to run the server on (default: 8000)

#### Volumes

The Docker setup mounts two volumes:

1. **`./documents:/app/documents`** - Document storage directory
   - Add your documents here for processing
   - Documents persist between container restarts

2. **`./data:/app/data`** - Data directory for vector store
   - FAISS index and metadata files are stored here
   - Persists vector store between restarts

### Docker Commands

**Build:**
```bash
docker-compose build
# or
docker build -t mini-rag:latest .
```

**Start:**
```bash
docker-compose up -d
# or
docker start mini-rag-api
```

**Stop:**
```bash
docker-compose down
# or
docker stop mini-rag-api
```

**View Logs:**
```bash
docker-compose logs -f
# or
docker logs -f mini-rag-api
```

**Restart:**
```bash
docker-compose restart
# or
docker restart mini-rag-api
```

**Remove:**
```bash
docker-compose down -v
# or
docker stop mini-rag-api && docker rm mini-rag-api
```

### Production Deployment

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

### Health Checks

The container includes a health check that monitors the `/health` endpoint:
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds (allows time for model loading)

## Architecture

### Modular Structure

The codebase is organized into a modular structure:

```
Mini_RAG/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py          # FastAPI app factory
│   │   ├── models.py            # Pydantic request/response models
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py        # Health check and stats endpoints
│   │       ├── documents.py     # Document processing endpoints
│   │       └── answer.py       # Answer generation endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Document processing
│   │   ├── embeddings.py       # Embedding generation
│   │   ├── retrieval.py        # Retrieval system
│   │   ├── llm.py              # LLM answering
│   │   └── reranking.py        # Reranking system
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vector_store_service.py  # Vector store management
│   │   ├── document_service.py     # Document processing logic
│   │   ├── search_service.py       # Search operations
│   │   └── answer_service.py      # Answer generation logic
│   └── config/
│       ├── __init__.py
│       └── settings.py         # Application configuration
├── app.py                       # Main entry point
├── run_pipeline.py             # Pipeline runner
├── run_demo.py                 # Interactive demo
└── ... (other files)
```

### Key Components

#### 1. Configuration Module (`src/config/`)
- Centralized all configuration in `settings.py`
- Uses environment variables from `.env`
- Provides singleton pattern for settings access

#### 2. API Layer (`src/api/`)
- **Models**: All Pydantic models separated from routes
- **Routes**: Each endpoint group in its own file
- **App Factory**: `create_app()` function for clean app initialization

#### 3. Core Modules (`src/core/`)
- **ingestion.py**: Document processing and chunking
- **embeddings.py**: Embedding generation and FAISS vector store
- **retrieval.py**: Semantic, BM25, and hybrid retrieval
- **llm.py**: LLM answer generation with citations
- **reranking.py**: BGE-reranker integration

#### 4. Service Layer (`src/services/`)
- **VectorStoreService**: Manages vector store lifecycle
- **DocumentService**: Handles document processing logic
- **SearchService**: Encapsulates search operations
- **AnswerService**: Manages answer generation

### Benefits

1. **Separation of Concerns**: Each module has a single responsibility
2. **Maintainability**: Easier to find and modify code
3. **Testability**: Services can be tested independently
4. **Scalability**: Easy to add new features without affecting existing code
5. **Reusability**: Services can be used across different parts of the application

## Configuration

### Default Settings

- **Chunk Size**: 512 tokens
- **Chunk Overlap**: 50 tokens
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Index Type**: `flat` (FAISS)
- **LLM Model**: `gpt-3.5-turbo`
- **Reranking**: Disabled by default

### Customization

Edit `src/config/settings.py` or pass parameters via API endpoints.

### Environment Variables

```bash
# Required for LLM answering
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Reranking configuration
RERANKING_ENABLED=false
RERANKER_MODEL=BAAI/bge-reranker-base

# Optional: Server port
PORT=8000
```

## Supported File Formats

- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- HTML (`.html`, `.htm`)
- Plain Text (`.txt`)

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Development

### Running Tests

```bash
# Run pipeline
python run_pipeline.py ./documents/

# Run examples
python example_usage.py
```

## Troubleshooting

### Common Issues

1. **PyTorch compatibility**: If you encounter meta tensor errors, ensure you have compatible versions:
   ```bash
   pip install torch==2.2.2 numpy<2
   ```

2. **OpenAI API Key**: Make sure `.env` file contains `OPENAI_API_KEY`

3. **Vector Store Not Found**: Process documents first using `/process` or `/upload`

4. **Reranking Not Working**: 
   - Ensure `FlagEmbedding` is installed: `pip install FlagEmbedding`
   - Check model download (first use downloads ~400MB)
   - Check logs for reranking-related messages

5. **Docker Issues**:
   - Check if port 8000 is already in use: `lsof -i :8000`
   - Verify volumes are mounted correctly
   - Check container logs: `docker logs mini-rag-api`

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- Sentence Transformers for embeddings
- FAISS for vector storage
- FastAPI for the API framework
- OpenAI for LLM capabilities
- BAAI for BGE-reranker model

# Mini RAG - Modular RAG System

A complete, modular Retrieval-Augmented Generation (RAG) system with document processing, semantic search, and LLM-powered question answering.

## Features

- **Multi-format Document Processing**: Extract and process PDF, DOCX, HTML, and TXT files
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Hybrid Retrieval**: Combine semantic search with BM25 for better results
- **LLM Answering**: Generate answers with automatic citation tracking using OpenAI
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

See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

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
   Or use the provided script:
   ```bash
   ./start_api.sh
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

4. **Search:**
   ```bash
   curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "your search query", "k": 5}'
   ```

5. **Ask questions:**
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

## API Endpoints

### Health & Stats
- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - Vector store statistics

### Document Processing
- `POST /process` - Process all documents in documents/ directory
- `POST /upload` - Upload and process a single document
- `DELETE /reset` - Reset vector store

### Search & Answer
- `POST /search` - Semantic/hybrid search
- `POST /answer` - Generate answer with citations

See http://localhost:8000/docs for interactive API documentation.

## Configuration

### Default Settings

- **Chunk Size**: 512 tokens
- **Chunk Overlap**: 50 tokens
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Index Type**: `flat` (FAISS)
- **LLM Model**: `gpt-3.5-turbo`

### Customization

Edit `src/config/settings.py` or pass parameters via API endpoints.

## Supported File Formats

- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- HTML (`.html`, `.htm`)
- Plain Text (`.txt`)

## Retrieval Modes

1. **Semantic**: Vector similarity search (default)
2. **BM25**: Keyword-based search
3. **Hybrid**: Weighted combination of semantic and BM25

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Development

### Project Structure

- `src/api/` - FastAPI routes and models
- `src/core/` - Core RAG components
- `src/services/` - Business logic layer
- `src/config/` - Configuration management

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

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- Sentence Transformers for embeddings
- FAISS for vector storage
- FastAPI for the API framework
- OpenAI for LLM capabilities

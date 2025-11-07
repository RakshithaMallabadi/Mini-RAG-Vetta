# Retrieval System Documentation

## Overview

The Mini RAG 2 system now supports three retrieval modes:
1. **Semantic Search** - Uses FAISS vector similarity search
2. **BM25 Search** - Keyword-based retrieval using BM25 algorithm
3. **Hybrid Search** - Combines semantic and BM25 for best results

## Retrieval Modes

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

## API Usage

### Search Endpoint

**POST** `/search`

**Request Body:**
```json
{
  "query": "your search query",
  "k": 5,
  "mode": "semantic",  // "semantic", "bm25", or "hybrid"
  "semantic_weight": 0.7,  // Only for hybrid mode
  "bm25_weight": 0.3  // Only for hybrid mode
}
```

**Response:**
```json
{
  "query": "workflow job",
  "total_results": 2,
  "results": [
    {
      "text": "Document chunk text...",
      "score": 0.85,
      "semantic_score": 0.90,  // Present in hybrid mode
      "bm25_score": 0.75,  // Present in hybrid mode
      "metadata": {
        "chunk_index": 0,
        "source_file": "document.pdf",
        "source_path": "./documents/document.pdf",
        "tokens": 512
      }
    }
  ]
}
```

### Example Requests

**Semantic Search:**
```bash
curl -X POST 'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "workflow job",
    "k": 5,
    "mode": "semantic"
  }'
```

**BM25 Search:**
```bash
curl -X POST 'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "API endpoint",
    "k": 5,
    "mode": "bm25"
  }'
```

**Hybrid Search:**
```bash
curl -X POST 'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "workflow job",
    "k": 5,
    "mode": "hybrid",
    "semantic_weight": 0.7,
    "bm25_weight": 0.3
  }'
```

## Python Usage

```python
from embeddings import EmbeddingVectorStore
from retrieval import RetrievalSystem

# Load vector store
vector_store = EmbeddingVectorStore()
vector_store.load('faiss_index.bin', 'faiss_metadata.pkl')

# Initialize retrieval system
retrieval = RetrievalSystem(vector_store)

# Load chunks
import pickle
with open('faiss_metadata.pkl', 'rb') as f:
    chunks = pickle.load(f)
retrieval.update_chunks(chunks)

# Semantic search
results = retrieval.search("workflow job", k=5, mode="semantic")

# BM25 search
results = retrieval.search("API endpoint", k=5, mode="bm25")

# Hybrid search
results = retrieval.search(
    "workflow job",
    k=5,
    mode="hybrid",
    semantic_weight=0.7,
    bm25_weight=0.3
)
```

## How Hybrid Search Works

1. **Semantic Search**: Generates embeddings and finds similar vectors using FAISS
2. **BM25 Search**: Tokenizes query and documents, calculates BM25 scores
3. **Score Normalization**: Normalizes both scores to 0-1 range
4. **Score Combination**: Combines scores using weighted average:
   ```
   hybrid_score = (semantic_weight × semantic_score) + (bm25_weight × bm25_score)
   ```
5. **Ranking**: Returns top-k results sorted by hybrid score

## Choosing Weights

- **High semantic_weight (0.7-0.9)**: Better for understanding intent, synonyms, context
- **High bm25_weight (0.7-0.9)**: Better for exact keyword matching, specific terms
- **Balanced (0.5-0.5)**: Good general-purpose setting

## Performance Notes

- **Semantic Search**: Slower (requires embedding generation), but better for semantic understanding
- **BM25 Search**: Faster (keyword matching), but limited to exact term matches
- **Hybrid Search**: Combines both, slightly slower but most accurate

## Implementation Details

- **BM25**: Uses `rank-bm25` library with default parameters
- **Tokenization**: Simple word tokenization (lowercase, alphanumeric)
- **Score Normalization**: Min-max normalization to ensure fair combination
- **Chunk Storage**: Chunks are stored in metadata for BM25 indexing


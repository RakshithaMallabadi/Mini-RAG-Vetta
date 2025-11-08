# Mini RAG - FastAPI Documentation

## Quick Start

### Start the Server

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

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

### 2. Health Check
**GET** `/health`

Check API health and vector store status.

**Response:**
```json
{
  "status": "healthy",
  "vector_store_loaded": true
}
```

### 3. Get Statistics
**GET** `/stats`

Get statistics about the vector store.

**Response:**
```json
{
  "total_vectors": 100,
  "embedding_dim": 384,
  "model_name": "all-MiniLM-L6-v2",
  "index_type": "flat",
  "index_path": "faiss_index.bin",
  "metadata_path": "faiss_metadata.pkl"
}
```

### 4. Process Documents
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

### 5. Upload Document
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

### 6. Search
**POST** `/search`

Search the vector store for similar documents.

**Request Body:**
```json
{
  "query": "your search query",
  "k": 5
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "workflow job",
    "k": 5
  }'
```

**Response:**
```json
{
  "query": "workflow job",
  "total_results": 3,
  "results": [
    {
      "text": "Document chunk text...",
      "score": 0.85,
      "metadata": {
        "source_file": "document.pdf",
        "chunk_index": 0,
        "tokens": 512,
        "source_path": "./documents/document.pdf"
      }
    }
  ]
}
```

### 7. Reset Vector Store
**DELETE** `/reset`

Delete the vector store index and metadata files.

**Example using curl:**
```bash
curl -X DELETE "http://localhost:8000/reset"
```

## Python Client Example

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

# 3. Search
response = requests.post(
    f"{BASE_URL}/search",
    json={"query": "your query", "k": 5}
)
results = response.json()
for result in results["results"]:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")

# 4. Get statistics
response = requests.get(f"{BASE_URL}/stats")
print(response.json())
```

## JavaScript/TypeScript Example

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

// Search
const searchResponse = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'your query', k: 5 })
});
const searchData = await searchResponse.json();
```

## Error Handling

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

## Notes

- The vector store is automatically loaded on startup if it exists
- Uploaded files are saved to the `documents/` directory
- The vector store persists between server restarts
- Use `/reset` to clear the vector store and start fresh


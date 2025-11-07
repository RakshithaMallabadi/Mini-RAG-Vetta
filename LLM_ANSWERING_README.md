# LLM Answering with Citations

## Overview

The Mini RAG 2 system now includes LLM-based question answering with automatic citation tracking. The system:

1. **Retrieves** top-k relevant chunks using semantic/BM25/hybrid search
2. **Generates** answers using an LLM (OpenAI GPT models)
3. **Cites** evidence from retrieved chunks automatically

## Features

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Automatic citation extraction
- ✅ Source file tracking
- ✅ Configurable retrieval modes (semantic, BM25, hybrid)
- ✅ Customizable LLM parameters (temperature, max_tokens, model)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add to your `.env` file or environment.

## API Usage

### Answer Endpoint

**POST** `/answer`

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
  "model": "gpt-3.5-turbo"
}
```

**Response:**
```json
{
  "question": "What is a workflow job?",
  "answer": "A workflow job is started using the POST /api/v1/jobs/start endpoint [Citation 1]. The request body must include workflow_id and optional payload, and the response returns job_id and execution status [Citation 1].",
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

### Example Requests

**Basic Question:**
```bash
curl -X POST 'http://localhost:8000/answer' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is a workflow job?",
    "k": 5
  }'
```

**With Hybrid Retrieval:**
```bash
curl -X POST 'http://localhost:8000/answer' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How do I start a workflow?",
    "k": 5,
    "mode": "hybrid",
    "semantic_weight": 0.7,
    "bm25_weight": 0.3
  }'
```

**Custom LLM Parameters:**
```bash
curl -X POST 'http://localhost:8000/answer' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Explain the API endpoint",
    "k": 3,
    "temperature": 0.5,
    "max_tokens": 300,
    "model": "gpt-4"
  }'
```

## Python Usage

```python
from embeddings import EmbeddingVectorStore
from retrieval import RetrievalSystem
from llm_answering import RAGAnswerSystem
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key"

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

# Initialize RAG system
rag = RAGAnswerSystem(retrieval, api_key=os.getenv("OPENAI_API_KEY"))

# Generate answer
result = rag.answer(
    question="What is a workflow job?",
    k=5,
    mode="semantic"
)

print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
print(f"Sources: {result['sources']}")
```

## How It Works

### 1. Retrieval Phase
- User question is used to retrieve top-k relevant chunks
- Supports semantic, BM25, or hybrid retrieval modes
- Chunks are ranked by relevance score

### 2. Prompt Construction
- Retrieved chunks are formatted with citation numbers
- Each chunk includes source file and chunk index
- Prompt instructs LLM to cite sources

### 3. LLM Generation
- LLM generates answer based on retrieved context
- Model is instructed to cite sources using [Citation X] format
- Temperature and max_tokens are configurable

### 4. Citation Extraction
- System extracts citation numbers from LLM response
- Maps citations to source files and chunks
- Returns structured citation information

## Citation Format

The LLM is instructed to use citations in the format:
- `[Citation 1]` - Preferred format
- `[Citation1]` - Alternative format
- `(Citation 1)` - Alternative format

The system automatically extracts these citations and maps them to:
- Source file names
- Chunk indices
- Original chunk text
- Relevance scores

## Parameters

### Retrieval Parameters
- **k**: Number of chunks to retrieve (default: 5)
- **mode**: "semantic", "bm25", or "hybrid" (default: "semantic")
- **semantic_weight**: Weight for semantic search in hybrid mode (default: 0.7)
- **bm25_weight**: Weight for BM25 in hybrid mode (default: 0.3)

### LLM Parameters
- **temperature**: Sampling temperature, 0.0-2.0 (default: 0.7)
  - Lower = more focused, deterministic
  - Higher = more creative, diverse
- **max_tokens**: Maximum tokens in response (default: 500)
- **model**: Model name (default: "gpt-3.5-turbo")
  - Options: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", etc.

## Error Handling

- If no relevant chunks found: Returns message indicating no information available
- If API key not set: Returns 500 error with instructions
- If vector store not loaded: Returns 404 error
- If LLM call fails: Returns 500 error with details

## Best Practices

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

## Example Workflow

1. **Process Documents**:
   ```bash
   curl -X POST 'http://localhost:8000/process' \
     -F "chunk_size=512" \
     -F "chunk_overlap=50"
   ```

2. **Ask Question**:
   ```bash
   curl -X POST 'http://localhost:8000/answer' \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is a workflow job?"}'
   ```

3. **Review Answer with Citations**:
   - Check the `answer` field for the generated response
   - Review `citations` to see which chunks were cited
   - Check `sources` for source file names
   - Examine `context_chunks` for full context

## Notes

- Citations are automatically extracted from LLM responses
- The system ensures answers are based on retrieved context
- If information is not in the context, LLM will indicate this
- All retrieved chunks are included in the response for transparency


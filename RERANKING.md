# Reranking Support in Mini RAG 2

This document describes the reranking functionality added to the Mini RAG 2 system.

## Overview

Reranking is a two-stage retrieval approach that improves search quality:
1. **First Stage**: Initial retrieval using semantic search, BM25, or hybrid search
2. **Second Stage**: Reranking of retrieved results using a cross-encoder model (BGE-reranker)

## Why Reranking?

- **Better Relevance**: Cross-encoder models (like BGE-reranker) consider query-document pairs together, providing more accurate relevance scores
- **Improved Accuracy**: Reranking can significantly improve the quality of top-k results
- **Flexible**: Can be enabled/disabled per request or globally

## Implementation

### BGE-Reranker

The system uses **BGE-reranker** from FlagEmbedding, which is a state-of-the-art reranking model developed by BAAI (Beijing Academy of Artificial Intelligence).

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

## Configuration

### Environment Variables

```bash
# Enable/disable reranking globally (default: true)
RERANKING_ENABLED=true

# Reranker model name (default: BAAI/bge-reranker-base)
RERANKER_MODEL=BAAI/bge-reranker-base
```

### Settings

Reranking can be configured in `src/config/settings.py`:

```python
RERANKING_ENABLED = True  # Global default
RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANK_TOP_K = None  # None = retrieve k * 2 before reranking
```

## API Usage

### Search Endpoint

```json
POST /search
{
  "query": "What is machine learning?",
  "k": 5,
  "mode": "semantic",
  "use_reranking": true,        // Optional: override global setting
  "rerank_top_k": 20            // Optional: number of docs to retrieve before reranking
}
```

### Answer Endpoint

```json
POST /answer
{
  "question": "What is machine learning?",
  "k": 5,
  "mode": "hybrid",
  "use_reranking": true,        // Optional: override global setting
  "rerank_top_k": 20            // Optional: number of docs to retrieve before reranking
}
```

### Parameters

- **`use_reranking`** (optional): 
  - `true`: Enable reranking for this request
  - `false`: Disable reranking for this request
  - `null`/omitted: Use system default (`RERANKING_ENABLED`)

- **`rerank_top_k`** (optional):
  - Number of documents to retrieve in first stage before reranking
  - If `null`/omitted: Defaults to `k * 2`
  - Example: If `k=5` and `rerank_top_k=20`, retrieve 20 docs, rerank them, return top 5

## Response Format

When reranking is used, the response includes additional fields:

```json
{
  "results": [
    {
      "text": "...",
      "score": 0.95,              // Final score (rerank_score if reranking used)
      "rerank_score": 0.95,       // Reranking score (if reranking used)
      "original_score": 0.82,     // Original retrieval score (if reranking used)
      "semantic_score": 0.82,     // Semantic search score (if hybrid mode)
      "bm25_score": 0.15,         // BM25 score (if hybrid mode)
      "metadata": {...}
    }
  ]
}
```

## Performance Considerations

### Speed vs Accuracy Trade-off

- **Without Reranking**: Faster, but may miss highly relevant documents
- **With Reranking**: Slower (adds ~100-500ms depending on model and number of documents), but significantly better accuracy

### Recommendations

1. **For Production**: Use `bge-reranker-base` for good balance of speed and accuracy
2. **For High Accuracy**: Use `bge-reranker-large` if latency is acceptable
3. **For Speed**: Disable reranking if sub-100ms latency is critical
4. **Hybrid Approach**: Use reranking only for important queries, disable for simple keyword searches

### Optimization Tips

- Set `rerank_top_k` appropriately: Too high = slower, too low = may miss relevant docs
- Typical values: `rerank_top_k = k * 2` to `k * 5`
- For `k=5`, retrieving 10-20 documents before reranking is usually optimal

## Installation

Reranking requires the `FlagEmbedding` package:

```bash
pip install FlagEmbedding>=1.2.0
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Example Usage

### Python Code

```python
from src.core.reranking import BGEReranker

# Initialize reranker
reranker = BGEReranker(model_name="BAAI/bge-reranker-base")

# Rerank documents
query = "What is machine learning?"
documents = [
    {"text": "Machine learning is a subset of AI...", "score": 0.8},
    {"text": "Deep learning uses neural networks...", "score": 0.7},
]

reranked = reranker.rerank(query, documents, top_k=5)
```

### API Example

```bash
# Search with reranking
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 5,
    "mode": "semantic",
    "use_reranking": true,
    "rerank_top_k": 20
  }'
```

## Troubleshooting

### Reranking Not Working

1. **Check Installation**: Ensure `FlagEmbedding` is installed
   ```bash
   pip install FlagEmbedding
   ```

2. **Check Model Download**: First use will download the model (~400MB for base model)
   - Check internet connection
   - Check disk space
   - Model is cached in `~/.cache/huggingface/`

3. **Check Logs**: Look for reranking-related messages in application logs

### Performance Issues

1. **Model Too Large**: Switch to `bge-reranker-base` instead of `large`
2. **Too Many Documents**: Reduce `rerank_top_k` value
3. **Memory Issues**: Ensure sufficient RAM (base model needs ~500MB)

## Future Enhancements

Potential improvements:
- Support for ColBERT reranking
- Support for other reranking models
- Caching of reranking results
- Batch reranking optimization
- GPU acceleration support


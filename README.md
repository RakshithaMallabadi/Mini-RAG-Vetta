# Mini RAG 2 - Data Ingestion Module

A robust data ingestion system for extracting, cleaning, and chunking documents for RAG (Retrieval-Augmented Generation) applications.

## Features

- **Multi-format Support**: Extract text from PDF, DOCX, HTML, and TXT files
- **Text Cleaning**: Remove boilerplate, noise, URLs, emails, and excessive whitespace
- **Token-based Chunking**: Intelligent chunking with configurable overlap using tiktoken
- **Error Handling**: Robust error handling for various edge cases

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from data_ingestion import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=512,      # Tokens per chunk
    chunk_overlap=50     # Overlapping tokens between chunks
)

# Process a single document
chunks = processor.process_document("document.pdf")

# Process all documents in a directory
chunks = processor.process_directory("./documents/")
```

### Command Line Usage

```bash
# Process a single file
python data_ingestion.py document.pdf

# Process all documents in a directory
python data_ingestion.py ./documents/
```

### Example Output

Each chunk contains:
- `text`: The chunk text content
- `tokens`: Number of tokens in the chunk
- `chunk_index`: Index of the chunk
- `source_file`: Original filename
- `source_path`: Full path to source file

## Configuration

You can customize the processor:

```python
processor = DocumentProcessor(
    encoding_model="cl100k_base",  # Tokenizer model (default for GPT)
    chunk_size=512,                 # Tokens per chunk
    chunk_overlap=50                # Overlap between chunks
)
```

## Supported Formats

- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- HTML (`.html`, `.htm`)
- Plain Text (`.txt`)

## Text Cleaning Features

The cleaning process removes:
- Excessive whitespace
- Email addresses
- URLs
- Common boilerplate (page numbers, copyright notices)
- Special control characters
- Very short lines (likely noise)


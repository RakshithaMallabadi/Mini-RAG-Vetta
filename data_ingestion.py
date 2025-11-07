"""
Data Ingestion Module for Mini RAG System
Handles document extraction, cleaning, and chunking
"""

import os
import re
import chardet
from typing import List, Dict, Optional
from pathlib import Path
import tiktoken

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import html2text
except ImportError:
    html2text = None


class DocumentProcessor:
    """Processes documents: extracts, cleans, and chunks text"""
    
    def __init__(self, encoding_model: str = "cl100k_base", chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor
        
        Args:
            encoding_model: Tokenizer model name (default: cl100k_base for GPT models)
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.encoding_model = encoding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_model)
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer: {e}")
        
        # Initialize HTML converter
        if html2text:
            self.html_converter = html2text.HTML2Text()
            self.html_converter.ignore_links = True
            self.html_converter.ignore_images = True
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install it with: pip install pypdf2")
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path}: {e}")
        
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX processing. Install it with: pip install python-docx")
        
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise ValueError(f"Error reading DOCX {file_path}: {e}")
    
    def extract_text_from_html(self, file_path: str) -> str:
        """Extract text from HTML file"""
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required for HTML processing. Install it with: pip install beautifulsoup4 lxml")
        
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                content = file.read()
            
            # Use BeautifulSoup to parse and clean HTML
            soup = BeautifulSoup(content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # If html2text is available, use it for better conversion
            if html2text:
                text = self.html_converter.handle(content)
            
            return text
        except Exception as e:
            raise ValueError(f"Error reading HTML {file_path}: {e}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                text = file.read()
            return text
        except Exception as e:
            raise ValueError(f"Error reading text file {file_path}: {e}")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a document based on file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif extension in ['.docx', '.doc']:
            return self.extract_text_from_docx(str(file_path))
        elif extension in ['.html', '.htm']:
            return self.extract_text_from_html(str(file_path))
        elif extension == '.txt':
            return self.extract_text_from_txt(str(file_path))
        else:
            # Try as plain text
            return self.extract_text_from_txt(str(file_path))
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing boilerplate and noise
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace (multiple spaces, tabs, newlines)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common boilerplate patterns
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove common header/footer patterns (page numbers, copyright, etc.)
        text = re.sub(r'\b(?:Page\s+\d+|©|Copyright|All rights reserved)\b', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Remove special characters that are likely noise
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove very short lines (likely noise)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        text = '\n'.join(cleaned_lines)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Chunk text into overlapping token-based chunks
        
        Args:
            text: Cleaned text to chunk
            
        Returns:
            List of chunks, each containing:
                - text: The chunk text
                - tokens: Token count
                - chunk_index: Index of the chunk
        """
        if not text:
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [{
                'text': text,
                'tokens': len(tokens),
                'chunk_index': 0
            }]
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            # Get chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'tokens': len(chunk_tokens),
                'chunk_index': chunk_index
            })
            
            # Move start index forward with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            chunk_index += 1
            
            # Prevent infinite loop
            if start_idx >= len(tokens):
                break
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, any]]:
        """
        Complete pipeline: extract, clean, and chunk a document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed chunks
        """
        # Extract text
        raw_text = self.extract_text(file_path)
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text)
        
        # Add metadata to each chunk
        file_name = Path(file_path).name
        for chunk in chunks:
            chunk['source_file'] = file_name
            chunk['source_path'] = str(file_path)
        
        return chunks
    
    def process_directory(self, directory_path: str, extensions: Optional[List[str]] = None) -> List[Dict[str, any]]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            extensions: List of file extensions to process (e.g., ['.pdf', '.txt'])
                        If None, processes all supported formats
        
        Returns:
            List of all processed chunks from all documents
        """
        if extensions is None:
            extensions = ['.pdf', '.docx', '.txt', '.html', '.htm']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_chunks = []
        
        for ext in extensions:
            for file_path in directory.glob(f'*{ext}'):
                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                    # Use logging or suppress print to avoid broken pipe issues in API context
                    import sys
                    if sys.stdout.isatty():  # Only print if running in terminal
                        print(f"Processed {file_path.name}: {len(chunks)} chunks")
                except Exception as e:
                    import sys
                    if sys.stderr.isatty():  # Only print if running in terminal
                        print(f"Error processing {file_path.name}: {e}", file=sys.stderr)
        
        return all_chunks


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_ingestion.py <file_path_or_directory>")
        print("Example: python data_ingestion.py document.pdf")
        print("Example: python data_ingestion.py ./documents/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Process file or directory
    if os.path.isfile(path):
        chunks = processor.process_document(path)
        print(f"\nProcessed {path}")
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1} ({chunk['tokens']} tokens):")
            print(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
    elif os.path.isdir(path):
        chunks = processor.process_directory(path)
        print(f"\nProcessed directory: {path}")
        print(f"Total chunks: {len(chunks)}")
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()


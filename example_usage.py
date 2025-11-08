"""
Example usage of the data ingestion module
"""

from collections import defaultdict
from src.core.ingestion import DocumentProcessor

def example_single_file():
    """Example: Process a single document"""
    print("=" * 60)
    print("Example 1: Processing a single document")
    print("=" * 60)
    
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Replace with your actual file path
    file_path = "example_document.txt"
    
    try:
        chunks = processor.process_document(file_path)
        print(f"\n✓ Successfully processed: {file_path}")
        print(f"✓ Total chunks created: {len(chunks)}")
        print(f"\nFirst chunk preview:")
        if chunks:
            print(f"  Tokens: {chunks[0]['tokens']}")
            print(f"  Text preview: {chunks[0]['text'][:150]}...")
    except FileNotFoundError:
        print(f"⚠ File not found: {file_path}")
        print("   Create a test file or update the path")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_directory():
    """Example: Process all documents in a directory"""
    print("\n" + "=" * 60)
    print("Example 2: Processing a directory")
    print("=" * 60)
    
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Replace with your actual directory path
    directory_path = "./documents"
    
    try:
        chunks = processor.process_directory(directory_path)
        print(f"\n✓ Successfully processed directory: {directory_path}")
        print(f"✓ Total chunks created: {len(chunks)}")
        
        # Group by source file
        chunks_by_file = defaultdict(int)
        for chunk in chunks:
            chunks_by_file[chunk['source_file']] += 1
        
        print(f"\nChunks per file:")
        for filename, count in chunks_by_file.items():
            print(f"  {filename}: {count} chunks")
    except FileNotFoundError:
        print(f"⚠ Directory not found: {directory_path}")
        print("   Create a documents directory or update the path")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_custom_config():
    """Example: Using custom configuration"""
    print("\n" + "=" * 60)
    print("Example 3: Custom configuration")
    print("=" * 60)
    
    # Custom chunk size and overlap
    processor = DocumentProcessor(
        encoding_model="cl100k_base",
        chunk_size=256,      # Smaller chunks
        chunk_overlap=25     # Smaller overlap
    )
    
    print(f"✓ Processor configured with:")
    print(f"  - Chunk size: {processor.chunk_size} tokens")
    print(f"  - Chunk overlap: {processor.chunk_overlap} tokens")
    print(f"  - Encoding model: {processor.encoding_model}")


def example_text_cleaning():
    """Example: Text cleaning demonstration"""
    print("\n" + "=" * 60)
    print("Example 4: Text cleaning demonstration")
    print("=" * 60)
    
    processor = DocumentProcessor()
    
    # Sample dirty text
    dirty_text = """
    This is a sample text with    excessive    whitespace.
    
    Email: test@example.com
    URL: https://www.example.com/page
    
    Page 1
    
    Copyright © 2024 All rights reserved
    
    Some actual content here that we want to keep.
    """
    
    cleaned = processor.clean_text(dirty_text)
    
    print("Original text:")
    print(repr(dirty_text))
    print("\nCleaned text:")
    print(repr(cleaned))
    print("\n✓ Text cleaned successfully!")


if __name__ == "__main__":
    # Run examples
    example_single_file()
    example_directory()
    example_custom_config()
    example_text_cleaning()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


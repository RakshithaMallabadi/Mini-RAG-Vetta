#!/usr/bin/env python3
"""
Interactive Demo for Mini RAG 2 System
Demonstrates document processing, search, and Q&A functionality
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.ingestion import DocumentProcessor
from src.core.embeddings import EmbeddingVectorStore
from src.core.retrieval import RetrievalSystem
from src.core.llm import RAGAnswerSystem
from src.config import get_settings


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_document_processing():
    """Demonstrate document processing"""
    print_section("STEP 1: Document Processing")
    
    settings = get_settings()
    documents_dir = settings.DOCUMENTS_DIR
    
    if not documents_dir.exists() or not any(documents_dir.iterdir()):
        print(f"⚠ No documents found in {documents_dir}")
        print("   Please add some documents (.txt, .pdf, .docx) to the documents/ directory")
        return None
    
    print(f"Processing documents from: {documents_dir}")
    print(f"Found documents: {[f.name for f in documents_dir.iterdir() if f.is_file()]}")
    
    processor = DocumentProcessor(
        chunk_size=settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
    )
    
    chunks = processor.process_directory(str(documents_dir))
    
    if not chunks:
        print("❌ No chunks generated from documents")
        return None
    
    print(f"✅ Successfully processed {len(chunks)} chunks")
    
    # Show sample chunk
    if chunks:
        print(f"\nSample chunk:")
        print(f"  Source: {chunks[0]['source_file']}")
        print(f"  Tokens: {chunks[0]['tokens']}")
        print(f"  Text preview: {chunks[0]['text'][:150]}...")
    
    return chunks


def demo_embeddings(chunks):
    """Demonstrate embedding generation and vector store"""
    print_section("STEP 2: Creating Embeddings & Vector Store")
    
    if not chunks:
        print("⚠ No chunks to process")
        return None
    
    settings = get_settings()
    
    print(f"Generating embeddings using model: {settings.DEFAULT_EMBEDDING_MODEL}")
    print("This may take a moment...")
    
    vector_store = EmbeddingVectorStore(
        embedding_model=settings.DEFAULT_EMBEDDING_MODEL,
        index_type=settings.DEFAULT_INDEX_TYPE
    )
    
    vector_store.add_chunks(chunks)
    
    # Save vector store
    vector_store.save(
        str(settings.INDEX_PATH),
        str(settings.METADATA_PATH)
    )
    
    stats = vector_store.get_stats()
    print(f"✅ Vector store created successfully!")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Embedding dimension: {stats['embedding_dim']}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Index type: {stats['index_type']}")
    
    return vector_store


def demo_search(vector_store):
    """Demonstrate search functionality"""
    print_section("STEP 3: Semantic Search")
    
    if vector_store is None:
        print("⚠ Vector store not available")
        return None
    
    # Initialize retrieval system
    import pickle
    settings = get_settings()
    
    retrieval_system = RetrievalSystem(vector_store)
    if settings.METADATA_PATH.exists():
        with open(settings.METADATA_PATH, 'rb') as f:
            chunks = pickle.load(f)
        retrieval_system.update_chunks(chunks)
    
    # Interactive search
    print("Enter search queries (type 'exit' to continue to Q&A)")
    print("-" * 70)
    
    while True:
        query = input("\n🔍 Search query: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        print(f"\nSearching for: '{query}'...")
        
        # Semantic search
        results = retrieval_system.search(query, k=3, mode="semantic")
        
        if results:
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"Result {i} (Score: {result['score']:.4f})")
                print(f"  Source: {result['metadata']['source_file']}")
                print(f"  Text: {result['text'][:200]}...")
                print()
        else:
            print("No results found.")
    
    return retrieval_system


def demo_qa(retrieval_system):
    """Demonstrate Q&A functionality"""
    print_section("STEP 4: Question Answering with LLM")
    
    if retrieval_system is None:
        print("⚠ Retrieval system not available")
        return
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OPENAI_API_KEY not set in .env file")
        print("   Skipping Q&A demo. Set OPENAI_API_KEY to enable this feature.")
        return
    
    try:
        rag_system = RAGAnswerSystem(retrieval_system, api_key=api_key)
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    print("Enter questions (type 'exit' to finish)")
    print("-" * 70)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        if not question:
            continue
        
        print(f"\n🤔 Thinking...")
        
        try:
            result = rag_system.answer(question, k=3, mode="semantic")
            
            print(f"\n💡 Answer:")
            print(f"   {result['answer']}")
            
            if result['citations']:
                print(f"\n📚 Citations: {result['citations']}")
                print(f"📁 Sources: {', '.join(result['sources'])}")
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")


def main():
    """Main demo function"""
    print_section("Mini RAG 2 - Interactive Demo")
    print("This demo will walk you through:")
    print("  1. Document processing")
    print("  2. Embedding generation")
    print("  3. Semantic search")
    print("  4. Question answering with LLM")
    
    input("\nPress Enter to start...")
    
    # Step 1: Document Processing
    chunks = demo_document_processing()
    if not chunks:
        print("\n⚠ Demo cannot continue without processed documents")
        return
    
    input("\nPress Enter to continue to embeddings...")
    
    # Step 2: Embeddings
    vector_store = demo_embeddings(chunks)
    if not vector_store:
        print("\n⚠ Demo cannot continue without vector store")
        return
    
    input("\nPress Enter to continue to search...")
    
    # Step 3: Search
    retrieval_system = demo_search(vector_store)
    
    input("\nPress Enter to continue to Q&A (if OpenAI API key is set)...")
    
    # Step 4: Q&A
    demo_qa(retrieval_system)
    
    print_section("Demo Complete!")
    print("✅ You've successfully demonstrated:")
    print("   - Document processing and chunking")
    print("   - Embedding generation and vector storage")
    print("   - Semantic search")
    if os.getenv("OPENAI_API_KEY"):
        print("   - LLM-powered question answering")
    print("\nTo use the API, run: uvicorn app:app --reload")
    print("Then visit: http://localhost:8000/docs")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


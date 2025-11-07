"""
Complete RAG Pipeline Runner
Integrates data ingestion, embeddings, and vector store
"""

from data_ingestion import DocumentProcessor
from embeddings import EmbeddingVectorStore
import os


def run_complete_pipeline(documents_path: str = "./documents/",
                          chunk_size: int = 512,
                          chunk_overlap: int = 50,
                          embedding_model: str = "all-MiniLM-L6-v2",
                          index_type: str = "flat",
                          save_index: bool = True,
                          index_path: str = "faiss_index.bin",
                          metadata_path: str = "faiss_metadata.pkl"):
    """
    Run the complete RAG pipeline: ingestion -> embeddings -> vector store
    
    Args:
        documents_path: Path to documents directory
        chunk_size: Token chunk size
        chunk_overlap: Token overlap between chunks
        embedding_model: Sentence transformer model name
        index_type: FAISS index type ("flat" or "ivf")
        save_index: Whether to save the index to disk
        index_path: Path to save FAISS index
        metadata_path: Path to save metadata
    """
    print("=" * 70)
    print("Mini RAG 2 - Complete Pipeline")
    print("=" * 70)
    
    # Step 1: Data Ingestion
    print("\n" + "=" * 70)
    print("STEP 1: Data Ingestion")
    print("=" * 70)
    
    if not os.path.exists(documents_path):
        print(f"✗ Error: Documents directory not found: {documents_path}")
        return None
    
    processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = processor.process_directory(documents_path)
    
    if not chunks:
        print("✗ No chunks generated. Check your documents.")
        return None
    
    print(f"\n✓ Generated {len(chunks)} chunks from documents")
    
    # Step 2: Generate Embeddings & Store in FAISS
    print("\n" + "=" * 70)
    print("STEP 2: Generating Embeddings & Vector Store")
    print("=" * 70)
    
    vector_store = EmbeddingVectorStore(
        embedding_model=embedding_model,
        index_type=index_type
    )
    
    vector_store.add_chunks(chunks)
    
    # Step 3: Save Vector Store
    if save_index:
        print("\n" + "=" * 70)
        print("STEP 3: Saving Vector Store")
        print("=" * 70)
        vector_store.save(index_path, metadata_path)
    
    # Display Statistics
    print("\n" + "=" * 70)
    print("PIPELINE STATISTICS")
    print("=" * 70)
    stats = vector_store.get_stats()
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")
    print(f"  Model: {stats['model_name']}")
    print(f"  Index type: {stats['index_type']}")
    print(f"  Chunks processed: {len(chunks)}")
    
    print("\n" + "=" * 70)
    print("✓ Pipeline completed successfully!")
    print("=" * 70)
    
    return vector_store


def test_search(vector_store: EmbeddingVectorStore, query: str, k: int = 5):
    """
    Test the search functionality
    
    Args:
        vector_store: The EmbeddingVectorStore instance
        query: Search query
        k: Number of results
    """
    print("\n" + "=" * 70)
    print("TESTING SEARCH")
    print("=" * 70)
    print(f"Query: '{query}'")
    print(f"Retrieving top {k} results...\n")
    
    results = vector_store.search(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"{'─' * 70}")
        print(f"Result {i} (Similarity Score: {result['score']:.4f})")
        print(f"Source: {result['metadata']['source_file']}")
        print(f"Chunk Index: {result['metadata']['chunk_index']}")
        print(f"Tokens: {result['metadata']['tokens']}")
        print(f"\nText:")
        print(f"  {result['text']}")
        print()


def main():
    """Main entry point"""
    import sys
    
    # Default settings
    documents_path = "./documents/"
    test_query = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        documents_path = sys.argv[1]
    if len(sys.argv) > 2:
        test_query = sys.argv[2]
    
    # Run pipeline
    vector_store = run_complete_pipeline(documents_path=documents_path)
    
    if vector_store is None:
        return
    
    # Test search if query provided
    if test_query:
        test_search(vector_store, test_query)
    else:
        # Default test query
        print("\n" + "=" * 70)
        print("Try a search query:")
        print("=" * 70)
        print("Example: python run_pipeline.py ./documents/ 'your query here'")
        print("\nOr use the vector store programmatically:")
        print("  from embeddings import EmbeddingVectorStore")
        print("  vector_store = EmbeddingVectorStore()")
        print("  vector_store.load('faiss_index.bin', 'faiss_metadata.pkl')")
        print("  results = vector_store.search('your query', k=5)")


if __name__ == "__main__":
    main()


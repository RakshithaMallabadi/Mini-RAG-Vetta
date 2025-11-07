"""
Embeddings and Vector Store Module for Mini RAG System
Handles sentence embeddings generation and FAISS vector storage
"""

import os
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np

# Lazy import for sentence-transformers to avoid TensorFlow dependency issues
# Set environment variables to handle TensorFlow issues
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress TensorFlow logs
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')  # Disable oneDNN optimizations

SentenceTransformer = None
try:
    # Suppress warnings during import
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        from sentence_transformers import SentenceTransformer
except (ImportError, OSError, RuntimeError, Exception) as e:
    # If import fails due to TensorFlow, try to work around it
    # This is a known issue where transformers tries to import TensorFlow
    # even though sentence-transformers doesn't need it
    print(f"Warning: Could not import sentence_transformers: {type(e).__name__}")
    print("This may be due to a TensorFlow dependency issue.")
    print("Try: pip uninstall tensorflow tensorflow-macos")
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


class EmbeddingGenerator:
    """Generates sentence embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
                       Default: "all-MiniLM-L6-v2" (fast, good quality)
                       Alternatives: "all-mpnet-base-v2" (better quality, slower)
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}...")
        # Workaround for PyTorch meta tensor issue
        # Set environment variables before importing/loading
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        
        try:
            import torch
            # Disable meta device to avoid meta tensor issues
            # This forces PyTorch to materialize tensors immediately
            if hasattr(torch, '_C'):
                # Try to disable meta device usage
                try:
                    torch._C._set_print_stack_traces_on_fatal_signal(False)
                except:
                    pass
            
            # Load model with explicit device and avoid lazy loading
            device = 'cpu'
            # Use device_map='cpu' to force CPU and avoid meta tensors
            self.model = SentenceTransformer(
                model_name,
                device=device
            )
            # Force model to CPU explicitly after loading
            if hasattr(self.model, 'to'):
                self.model = self.model.to(device)
            # Also move the underlying model if it exists
            if hasattr(self.model, '_modules'):
                for module in self.model._modules.values():
                    if hasattr(module, 'to'):
                        module.to(device)
                        
        except Exception as e:
            # Fallback: try simple loading
            print(f"Warning: Error with device specification: {e}")
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e2:
                print(f"Error loading model: {e2}")
                raise
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_texts, embedding_dim) containing embeddings
        """
        if not texts:
            return np.array([])
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Single text string to embed
            
        Returns:
            numpy array of shape (embedding_dim,) containing the embedding
        """
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]


class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of the embeddings
            index_type: Type of FAISS index
                       - "flat": Exact search (IndexFlatL2 or IndexFlatIP)
                       - "ivf": Inverted file index (faster for large datasets)
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required. Install it with: pip install faiss-cpu"
            )
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create FAISS index (using cosine similarity with normalized vectors)
        # For normalized vectors, dot product = cosine similarity
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        elif index_type == "ivf":
            # IVF index for faster search on large datasets
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = 100  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            self.index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Store metadata for each vector
        self.metadata: List[Dict] = []
        self.is_trained = False
    
    def add_vectors(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add vectors and their metadata to the index
        
        Args:
            embeddings: numpy array of shape (n_vectors, embedding_dim)
            metadata: List of metadata dictionaries, one per vector
        """
        if len(embeddings) != len(metadata):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(metadata)} metadata entries")
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.is_trained:
            print("Training FAISS index...")
            self.index.train(embeddings)
            self.is_trained = True
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        print(f"✓ Added {len(embeddings)} vectors to FAISS index. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector of shape (embedding_dim,)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices, metadata_list)
            - distances: Similarity scores (higher is more similar for cosine)
            - indices: Indices of similar vectors
            - metadata_list: Metadata for the retrieved vectors
        """
        if self.index.ntotal == 0:
            return np.array([]), np.array([]), []
        
        # Ensure query is float32 and normalized
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Get metadata for retrieved vectors
        metadata_list = [self.metadata[idx] for idx in indices[0]]
        
        return distances[0], indices[0], metadata_list
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save the FAISS index and metadata to disk
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"✓ Saved index to {index_path}")
        print(f"✓ Saved metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load the FAISS index and metadata from disk
        
        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.is_trained = True  # Loaded index is already trained
        print(f"✓ Loaded index from {index_path}")
        print(f"✓ Loaded metadata from {metadata_path}")
        print(f"  Total vectors: {self.index.ntotal}")
    
    def get_total_vectors(self) -> int:
        """Get the total number of vectors in the index"""
        return self.index.ntotal


class EmbeddingVectorStore:
    """Combined class for generating embeddings and storing in FAISS"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_type: str = "flat",
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """
        Initialize the combined embedding and vector store system
        
        Args:
            embedding_model: Sentence transformer model name
            index_type: FAISS index type ("flat" or "ivf")
            index_path: Optional path to load existing index
            metadata_path: Optional path to load existing metadata
        """
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_generator.embedding_dim,
            index_type=index_type
        )
        
        # Load existing index if provided
        if index_path and metadata_path and os.path.exists(index_path) and os.path.exists(metadata_path):
            self.vector_store.load(index_path, metadata_path)
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries from DocumentProcessor
                   Each chunk should have at least a 'text' field
        """
        if not chunks:
            print("No chunks to add")
            return
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Prepare metadata (preserve all chunk information)
        metadata = []
        for chunk in chunks:
            metadata.append({
                'text': chunk.get('text', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'source_file': chunk.get('source_file', ''),
                'source_path': chunk.get('source_path', ''),
                'tokens': chunk.get('tokens', 0)
            })
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, metadata)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Query text string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing:
            - text: Chunk text
            - score: Similarity score
            - metadata: Original chunk metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search
        distances, indices, metadata_list = self.vector_store.search(query_embedding, k=k)
        
        # Format results
        results = []
        for i, (distance, meta) in enumerate(zip(distances, metadata_list)):
            results.append({
                'text': meta['text'],
                'score': float(distance),
                'metadata': meta
            })
        
        return results
    
    def save(self, index_path: str = "faiss_index.bin", metadata_path: str = "faiss_metadata.pkl") -> None:
        """Save the vector store to disk"""
        self.vector_store.save(index_path, metadata_path)
    
    def load(self, index_path: str = "faiss_index.bin", metadata_path: str = "faiss_metadata.pkl") -> None:
        """Load the vector store from disk"""
        self.vector_store.load(index_path, metadata_path)
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.vector_store.get_total_vectors(),
            'embedding_dim': self.embedding_generator.embedding_dim,
            'model_name': self.embedding_generator.model_name,
            'index_type': self.vector_store.index_type
        }


def main():
    """Example usage"""
    import sys
    from data_ingestion import DocumentProcessor
    
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py <documents_directory>")
        print("Example: python embeddings.py ./documents/")
        sys.exit(1)
    
    documents_path = sys.argv[1]
    
    # Step 1: Process documents
    print("=" * 70)
    print("Step 1: Processing documents...")
    print("=" * 70)
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    chunks = processor.process_directory(documents_path)
    print(f"✓ Processed {len(chunks)} chunks from documents\n")
    
    # Step 2: Generate embeddings and store in FAISS
    print("=" * 70)
    print("Step 2: Generating embeddings and storing in FAISS...")
    print("=" * 70)
    vector_store = EmbeddingVectorStore()
    vector_store.add_chunks(chunks)
    
    # Step 3: Save the vector store
    print("\n" + "=" * 70)
    print("Step 3: Saving vector store...")
    print("=" * 70)
    vector_store.save()
    
    # Step 4: Test search
    print("\n" + "=" * 70)
    print("Step 4: Testing search...")
    print("=" * 70)
    test_query = "API workflow job"
    results = vector_store.search(test_query, k=3)
    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i} (score: {result['score']:.4f}):")
        print(f"  Source: {result['metadata']['source_file']}")
        print(f"  Text: {result['text'][:100]}...")
        print()
    
    # Show stats
    stats = vector_store.get_stats()
    print("=" * 70)
    print("Vector Store Statistics:")
    print("=" * 70)
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


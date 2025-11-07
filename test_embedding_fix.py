#!/usr/bin/env python3
"""Test if embedding model can be loaded with the fix"""
import sys
import os

# Add workaround environment variables
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

try:
    from embeddings import EmbeddingGenerator
    print("Testing EmbeddingGenerator initialization...")
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    print("✅ SUCCESS: EmbeddingGenerator initialized successfully!")
    print(f"   Embedding dimension: {generator.embedding_dim}")
    
    # Test generating an embedding
    test_text = "This is a test"
    embedding = generator.generate_embedding(test_text)
    print(f"✅ SUCCESS: Generated embedding of shape {embedding.shape}")
    sys.exit(0)
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


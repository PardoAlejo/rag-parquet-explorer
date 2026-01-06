#!/usr/bin/env python
"""
Quick demo script to test the RAG system components
Run this to verify everything is working without needing Ollama/UI
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("🚀 RAG System Demo")
print("=" * 50)
print()

# Test 1: Check Parquet files
print("📊 Step 1: Checking Parquet files...")
from rag.config import PARQUET_FILES

for pf in PARQUET_FILES:
    if pf.exists():
        size_mb = pf.stat().st_size / 1024 / 1024
        print(f"  ✓ Found: {pf.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ Missing: {pf.name}")

print()

# Test 2: Load documents
print("📄 Step 2: Loading documents from Parquet...")
try:
    from rag.retrieval import ParquetRetriever
    from rag.embeddings import EmbeddingGenerator

    # Create mock embedding generator (no actual model loading)
    class MockEmbeddingGen:
        def generate_embeddings(self, texts, **kwargs):
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384).astype(np.float32)

        def get_embedding_dimension(self):
            return 384

    embedding_gen = MockEmbeddingGen()

    retriever = ParquetRetriever(
        parquet_files=[PARQUET_FILES[0]],  # Just first file for demo
        embedding_generator=embedding_gen,
        top_k=3
    )

    # Load a sample
    docs = retriever.load_documents(sample_size=100)
    print(f"  ✓ Loaded {len(docs)} documents (sample)")

    if docs:
        print(f"  ✓ Sample document ID: {docs[0]['id']}")
        print(f"  ✓ Sample document source: {docs[0]['source']}")
        print(f"  ✓ Content fields: {list(docs[0]['content'].keys())}")

    print()

except Exception as e:
    print(f"  ✗ Error: {e}")
    print()

# Test 3: Build embeddings
print("🔢 Step 3: Testing embedding generation (mock)...")
try:
    embeddings = retriever.build_embeddings()
    print(f"  ✓ Generated embeddings: shape {embeddings.shape}")
    print()
except Exception as e:
    print(f"  ✗ Error: {e}")
    print()

# Test 4: Test search
print("🔍 Step 4: Testing semantic search...")
try:
    results = retriever.search("test query", top_k=3)
    print(f"  ✓ Search returned {len(results)} results")

    for i, result in enumerate(results[:2], 1):
        print(f"  ✓ Result {i}:")
        print(f"    - Similarity: {result['similarity_score']:.3f}")
        print(f"    - Source: {result['source']}")
        print(f"    - Text preview: {result['text'][:80]}...")

    print()
except Exception as e:
    print(f"  ✗ Error: {e}")
    print()

# Test 5: Test context generation
print("📝 Step 5: Testing context generation...")
try:
    context = retriever.get_context_for_query("sample query", top_k=2)
    print(f"  ✓ Generated context ({len(context)} characters)")
    print(f"  ✓ Context preview:")
    print(f"    {context[:200]}...")
    print()
except Exception as e:
    print(f"  ✗ Error: {e}")
    print()

# Test 6: Check Ollama (optional)
print("🤖 Step 6: Checking Ollama availability (optional)...")
try:
    from rag.llm import LLMInterface

    llm = LLMInterface()
    available = llm.check_availability()

    if available:
        print("  ✓ Ollama is available and model is installed")
        print("  ✓ You can use the full RAG system!")
    else:
        print("  ℹ Ollama not available (that's OK for this demo)")
        print("  ℹ To use RAG system: ollama pull llama3.2")

    print()
except Exception as e:
    print(f"  ℹ Ollama check skipped: {e}")
    print()

# Summary
print("=" * 50)
print("✅ Demo Complete!")
print()
print("Next steps:")
print("  1. Launch Parquet Explorer:")
print("     cd tools/parquet_explorer && streamlit run app.py")
print()
print("  2. Launch RAG System (requires Ollama):")
print("     ollama serve  # In separate terminal")
print("     streamlit run rag/app.py")
print()
print("  3. Run tests:")
print("     ./run_tests.sh")
print()

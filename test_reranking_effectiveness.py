"""
Quick test to check if re-ranking actually improves results on real data
"""

from pathlib import Path
from rag.embeddings import EmbeddingGenerator
from rag.retrieval import ParquetRetriever
from rag.config import PARQUET_FILES, RERANKER_MODEL

def test_reranking_effectiveness():
    """Test if re-ranking changes document order significantly"""

    # Test queries - adjust these for your domain
    test_queries = [
        "What are the consequences of independence movements?",
        "Legal framework for constitutional changes",
        "Economic impact of political decisions",
        "State of democracy in Europe",
    ]

    # Test with larger pool to see if that helps
    POOL_SIZE = 50

    print("=" * 80)
    print("RE-RANKING EFFECTIVENESS TEST (with Pool System)")
    print("=" * 80)
    print(f"\nPool size: {POOL_SIZE} documents")
    print(f"Comparing top 5 from each pool\n")

    # Initialize retriever
    print("\nInitializing RAG system...")
    embedding_gen = EmbeddingGenerator()
    retriever = ParquetRetriever(
        parquet_files=PARQUET_FILES,
        embedding_generator=embedding_gen,
        top_k=10,
        use_reranking=False,
        reranker_model=RERANKER_MODEL
    )

    retriever.load_documents()
    retriever.build_embeddings()

    print(f"Loaded {len(retriever.documents)} documents")

    total_changes = 0
    total_queries = 0

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print("="*80)

        # Get original ranking from pool (cosine similarity)
        pool_original = retriever.search(
            query,
            top_k=POOL_SIZE,
            use_reranking=False,
            min_similarity=0.3
        )

        # Get re-ranked pool
        pool_reranked = retriever.search(
            query,
            top_k=POOL_SIZE,
            use_reranking=True,
            min_similarity=0.3
        )

        # Take top 5 from each pool for comparison
        results_original = pool_original[:5]
        results_reranked = pool_reranked[:5]

        if not results_original or not results_reranked:
            print("⚠️  No results found for this query")
            continue

        total_queries += 1

        # Compare top 5 results
        print("\n📊 ORIGINAL RANKING (Cosine Similarity):")
        print("-" * 80)
        original_ids = []
        for i, doc in enumerate(results_original[:5], 1):
            title = doc.get('content', {}).get('title', doc.get('source', 'N/A'))[:60]
            score = doc.get('similarity_score', 0)
            doc_id = doc.get('id', 'unknown')
            original_ids.append(doc_id)
            print(f"{i}. [{score:.3f}] {title}")

        print(f"\n🔄 RE-RANKED RESULTS (Cross-Encoder from pool of {POOL_SIZE}):")
        print("-" * 80)
        reranked_ids = []
        for i, doc in enumerate(results_reranked[:5], 1):
            title = doc.get('content', {}).get('title', doc.get('source', 'N/A'))[:60]
            score = doc.get('similarity_score', 0)
            doc_id = doc.get('id', 'unknown')
            reranked_ids.append(doc_id)

            # Find original position in the pool
            try:
                orig_doc_ids = [d['id'] for d in pool_original]
                orig_pos = orig_doc_ids.index(doc_id) + 1
                if orig_pos <= 5:
                    change_indicator = f"(was #{orig_pos})"
                else:
                    change_indicator = f"(was #{orig_pos} - SURFACED from pool!)"
            except ValueError:
                change_indicator = "(NOT FOUND in pool)"

            print(f"{i}. [{score:.3f}] {title} {change_indicator}")

        # Calculate how many positions changed
        changes = sum(1 for i in range(min(5, len(original_ids), len(reranked_ids)))
                     if original_ids[i] != reranked_ids[i])

        total_changes += changes

        print(f"\n📈 ANALYSIS:")
        print(f"   Documents that changed position: {changes}/5")
        print(f"   Order preserved: {5 - changes}/5")

        if changes == 0:
            print("   ⚠️  Re-ranking had NO effect - same order!")
        elif changes <= 2:
            print("   ⚠️  Re-ranking had MINIMAL effect")
        elif changes <= 4:
            print("   ✅ Re-ranking had MODERATE effect")
        else:
            print("   ✅ Re-ranking SIGNIFICANTLY changed order")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    if total_queries > 0:
        avg_changes = total_changes / total_queries
        print(f"Average documents changed per query: {avg_changes:.1f}/5")
        print(f"Average preservation rate: {(5 - avg_changes) / 5 * 100:.1f}%")

        if avg_changes < 1:
            print("\n❌ RE-RANKING IS NOT EFFECTIVE")
            print("   Recommendation: Disable re-ranking, it's just adding latency")
        elif avg_changes < 2:
            print("\n⚠️  RE-RANKING HAS MINIMAL IMPACT")
            print("   Recommendation: Consider disabling or using different model")
        elif avg_changes < 4:
            print("\n✅ RE-RANKING IS MODERATELY EFFECTIVE")
            print("   Recommendation: Keep enabled for production")
        else:
            print("\n✅ RE-RANKING IS HIGHLY EFFECTIVE")
            print("   Recommendation: Always use re-ranking")
    else:
        print("⚠️  No queries returned results")

if __name__ == "__main__":
    test_reranking_effectiveness()

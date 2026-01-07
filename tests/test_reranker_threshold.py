"""
Tests for re-ranker with similarity threshold interaction
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


class TestRerankerThresholdInteraction:
    """Test that similarity threshold is applied correctly with re-ranking"""

    @pytest.fixture
    def temp_parquet_file(self, tmp_path):
        """Create a temporary parquet file with sample data"""
        data = {
            'text': [f'Document {i}' for i in range(10)],
            'id': list(range(10))
        }
        table = pa.table(data)
        file_path = tmp_path / "test.parquet"
        pq.write_table(table, file_path)
        return file_path

    @pytest.fixture
    def mock_embedding_generator(self):
        """Mock embedding generator"""
        mock = MagicMock()
        # Return fixed embeddings
        mock.generate_embeddings.return_value = np.random.randn(10, 384).astype(np.float32)
        return mock

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder for re-ranking"""
        with patch('sentence_transformers.CrossEncoder') as mock:
            mock_model = MagicMock()
            # Return cross-encoder scores (NOT in 0-1 range)
            # These are typical scores from ms-marco model
            mock_model.predict.return_value = np.array([2.5, 1.8, 0.3, -0.5, -1.2])
            mock.return_value = mock_model
            yield mock

    def test_threshold_applied_before_reranking(self, temp_parquet_file, mock_embedding_generator, mock_cross_encoder):
        """Test that similarity threshold filters on cosine similarity, not cross-encoder scores"""
        from rag.retrieval import ParquetRetriever

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_embedding_generator,
            use_reranking=True
        )

        # Load and build embeddings
        retriever.load_documents()
        retriever.build_embeddings()

        # Mock cosine similarities - some above, some below threshold
        cosine_sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        with patch.object(retriever, '_cosine_similarity', return_value=cosine_sims):
            # Search with threshold of 0.65
            # Without threshold fix, cross-encoder scores would all be filtered out
            results = retriever.search("test query", top_k=3, min_similarity=0.65)

            # Should only get documents with cosine similarity >= 0.65
            # That's documents with scores: 0.9, 0.8, 0.7 (3 documents)
            # Even if cross-encoder gives them negative scores, they should pass
            assert len(results) == 3

            # All results should have cosine_similarity preserved
            for result in results:
                assert 'cosine_similarity' in result
                assert result['cosine_similarity'] >= 0.65

    def test_threshold_zero_with_reranking(self, temp_parquet_file, mock_embedding_generator, mock_cross_encoder):
        """Test that threshold=0 works correctly with re-ranking"""
        from rag.retrieval import ParquetRetriever

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_embedding_generator,
            use_reranking=True
        )

        retriever.load_documents()
        retriever.build_embeddings()

        # Search with no threshold
        results = retriever.search("test query", top_k=5, min_similarity=0.0)

        # Should get 5 results (limited by top_k)
        assert len(results) == 5

    def test_cosine_similarity_preserved_after_reranking(self, temp_parquet_file, mock_embedding_generator, mock_cross_encoder):
        """Test that original cosine similarity is preserved after re-ranking"""
        from rag.retrieval import ParquetRetriever

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_embedding_generator,
            use_reranking=True
        )

        retriever.load_documents()
        retriever.build_embeddings()

        cosine_sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        with patch.object(retriever, '_cosine_similarity', return_value=cosine_sims):
            results = retriever.search("test query", top_k=3, min_similarity=0.0)

            # Check that both scores are present
            for result in results:
                assert 'similarity_score' in result  # Cross-encoder score
                assert 'cosine_similarity' in result  # Original cosine score
                assert result.get('reranked') is True

                # Cosine similarity should be different from similarity_score
                # (because one is cosine, the other is cross-encoder)
                # But both should be present

    def test_high_threshold_filters_all_documents(self, temp_parquet_file, mock_embedding_generator, mock_cross_encoder):
        """Test that very high threshold filters out all documents"""
        from rag.retrieval import ParquetRetriever

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_embedding_generator,
            use_reranking=True
        )

        retriever.load_documents()
        retriever.build_embeddings()

        cosine_sims = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01])

        with patch.object(retriever, '_cosine_similarity', return_value=cosine_sims):
            # Threshold higher than all similarities
            results = retriever.search("test query", top_k=5, min_similarity=0.9)

            # Should get no results
            assert len(results) == 0

    def test_without_reranking_threshold_still_works(self, temp_parquet_file, mock_embedding_generator):
        """Test that threshold works correctly when re-ranking is disabled"""
        from rag.retrieval import ParquetRetriever

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_embedding_generator,
            use_reranking=False  # No re-ranking
        )

        retriever.load_documents()
        retriever.build_embeddings()

        cosine_sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

        with patch.object(retriever, '_cosine_similarity', return_value=cosine_sims):
            results = retriever.search("test query", top_k=5, min_similarity=0.65)

            # Should get 3 documents (0.9, 0.8, 0.7)
            assert len(results) == 3

            # Results should not be marked as reranked
            for result in results:
                assert result.get('reranked') is not True
                assert result['similarity_score'] >= 0.65

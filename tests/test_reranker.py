"""
Tests for the re-ranking system
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from rag.retrieval.reranker import Reranker, NoOpReranker


class TestReranker:
    """Tests for the Reranker class"""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder model"""
        with patch('sentence_transformers.CrossEncoder') as mock:
            # Create a mock model instance
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0.9, 0.7, 0.5, 0.3])
            mock.return_value = mock_model
            yield mock

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {
                'id': 'doc1',
                'text': 'Python is a programming language',
                'similarity_score': 0.8
            },
            {
                'id': 'doc2',
                'text': 'JavaScript is also a programming language',
                'similarity_score': 0.7
            },
            {
                'id': 'doc3',
                'text': 'Machine learning uses Python',
                'similarity_score': 0.6
            },
            {
                'id': 'doc4',
                'text': 'Web development with JavaScript',
                'similarity_score': 0.5
            }
        ]

    def test_reranker_initialization(self):
        """Test reranker initializes correctly"""
        reranker = Reranker(model_name="test-model")
        assert reranker.model_name == "test-model"
        assert reranker.model is None
        assert not reranker._initialized

    def test_lazy_initialization(self, mock_cross_encoder):
        """Test reranker lazy loads model on first use"""
        reranker = Reranker()
        assert not reranker._initialized

        # Trigger lazy initialization
        reranker._lazy_init()

        assert reranker._initialized
        assert reranker.model is not None
        mock_cross_encoder.assert_called_once()

    def test_rerank_with_model(self, mock_cross_encoder, sample_documents):
        """Test re-ranking with cross-encoder model"""
        reranker = Reranker()
        query = "Python programming"

        # Re-rank documents
        results = reranker.rerank(query, sample_documents)

        # Check that model was used
        assert reranker._initialized
        assert reranker.model.predict.called

        # Check results
        assert len(results) == 4
        assert all('reranked' in doc for doc in results)
        assert all('original_score' in doc for doc in results)

        # Check scores are updated and sorted
        scores = [doc['similarity_score'] for doc in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_with_top_k(self, mock_cross_encoder, sample_documents):
        """Test re-ranking with top_k limit"""
        reranker = Reranker()
        query = "Python programming"

        # Re-rank with top_k=2
        results = reranker.rerank(query, sample_documents, top_k=2)

        assert len(results) == 2
        # Should return the top 2 after re-ranking
        assert all('reranked' in doc for doc in results)

    def test_rerank_empty_documents(self, mock_cross_encoder):
        """Test re-ranking with empty document list"""
        reranker = Reranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_preserves_original_score(self, mock_cross_encoder, sample_documents):
        """Test that original scores are preserved"""
        reranker = Reranker()
        results = reranker.rerank("query", sample_documents)

        for i, result in enumerate(results):
            assert 'original_score' in result
            # Original score should match the input document's score
            # (we need to find the matching doc since order may change)
            original_doc = next(d for d in sample_documents if d['id'] == result['id'])
            assert result['original_score'] == original_doc['similarity_score']

    def test_rerank_fallback_on_model_load_failure(self):
        """Test that reranker falls back gracefully when model fails to load"""
        with patch('sentence_transformers.CrossEncoder', side_effect=Exception("Model not found")):
            reranker = Reranker()
            docs = [{'id': '1', 'text': 'test', 'similarity_score': 0.5}]

            # Should not raise exception, just return original docs
            results = reranker.rerank("query", docs)
            assert results == docs

    def test_rerank_fallback_on_prediction_failure(self, sample_documents):
        """Test fallback when prediction fails"""
        with patch('sentence_transformers.CrossEncoder') as mock_encoder:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("Prediction failed")
            mock_encoder.return_value = mock_model

            reranker = Reranker()
            results = reranker.rerank("query", sample_documents)

            # Should return original documents
            assert len(results) == len(sample_documents)

    def test_is_available(self, mock_cross_encoder):
        """Test is_available method"""
        reranker = Reranker()

        # Should trigger lazy init and return True
        assert reranker.is_available()
        assert reranker._initialized

    def test_is_available_when_model_fails(self):
        """Test is_available when model fails to load"""
        with patch('sentence_transformers.CrossEncoder', side_effect=Exception("Failed")):
            reranker = Reranker()
            assert not reranker.is_available()

    def test_rerank_creates_pairs_correctly(self, mock_cross_encoder, sample_documents):
        """Test that query-document pairs are created correctly"""
        reranker = Reranker()
        query = "test query"

        reranker.rerank(query, sample_documents)

        # Check that predict was called with correct pairs
        call_args = reranker.model.predict.call_args[0][0]
        assert len(call_args) == len(sample_documents)

        for i, pair in enumerate(call_args):
            assert pair[0] == query
            assert pair[1] == sample_documents[i]['text']


class TestNoOpReranker:
    """Tests for the NoOpReranker class"""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {'id': 'doc1', 'text': 'test1', 'similarity_score': 0.9},
            {'id': 'doc2', 'text': 'test2', 'similarity_score': 0.7},
            {'id': 'doc3', 'text': 'test3', 'similarity_score': 0.5}
        ]

    def test_noop_reranker_returns_original(self, sample_documents):
        """Test NoOpReranker returns documents unchanged"""
        reranker = NoOpReranker()
        results = reranker.rerank("query", sample_documents)

        assert results == sample_documents

    def test_noop_reranker_with_top_k(self, sample_documents):
        """Test NoOpReranker respects top_k parameter"""
        reranker = NoOpReranker()
        results = reranker.rerank("query", sample_documents, top_k=2)

        assert len(results) == 2
        assert results == sample_documents[:2]

    def test_noop_reranker_with_empty_docs(self):
        """Test NoOpReranker with empty document list"""
        reranker = NoOpReranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_noop_reranker_is_available(self):
        """Test NoOpReranker is always available"""
        reranker = NoOpReranker()
        assert reranker.is_available()


class TestRerankerIntegration:
    """Integration tests for reranker"""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder model"""
        with patch('sentence_transformers.CrossEncoder') as mock:
            mock_model = MagicMock()
            # Return scores that reverse the original order
            mock_model.predict.return_value = np.array([0.3, 0.5, 0.7, 0.9])
            mock.return_value = mock_model
            yield mock

    def test_reranking_changes_order(self, mock_cross_encoder):
        """Test that re-ranking actually changes document order"""
        documents = [
            {'id': 'doc1', 'text': 'first', 'similarity_score': 0.9},
            {'id': 'doc2', 'text': 'second', 'similarity_score': 0.7},
            {'id': 'doc3', 'text': 'third', 'similarity_score': 0.5},
            {'id': 'doc4', 'text': 'fourth', 'similarity_score': 0.3}
        ]

        reranker = Reranker()
        results = reranker.rerank("query", documents)

        # With mock scores [0.3, 0.5, 0.7, 0.9], order should be reversed
        assert results[0]['id'] == 'doc4'  # Got highest score (0.9)
        assert results[1]['id'] == 'doc3'  # Got second highest (0.7)
        assert results[2]['id'] == 'doc2'  # Got third highest (0.5)
        assert results[3]['id'] == 'doc1'  # Got lowest score (0.3)

    def test_reranking_with_partial_results(self, mock_cross_encoder):
        """Test re-ranking followed by top_k selection"""
        documents = [
            {'id': f'doc{i}', 'text': f'text{i}', 'similarity_score': 1.0 - i*0.1}
            for i in range(10)
        ]

        # Mock returns descending scores
        mock_cross_encoder.return_value.predict.return_value = np.arange(10, 0, -1) / 10.0

        reranker = Reranker()
        results = reranker.rerank("query", documents, top_k=3)

        assert len(results) == 3
        # Check that we got the top 3 after re-ranking
        assert all('reranked' in doc for doc in results)

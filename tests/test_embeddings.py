"""
Tests for rag/embeddings/generator.py
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rag.embeddings.generator import EmbeddingGenerator


@pytest.mark.unit
class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator class"""

    def test_init_default_model(self):
        """Test initialization with default model"""
        generator = EmbeddingGenerator()
        assert generator.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert generator.model is None

    def test_init_custom_model(self):
        """Test initialization with custom model"""
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        generator = EmbeddingGenerator(model_name=custom_model)
        assert generator.model_name == custom_model
        assert generator.model is None

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_load_model_success(self, mock_transformer):
        """Test successful model loading"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        generator.load_model()

        mock_transformer.assert_called_once_with(generator.model_name)
        assert generator.model == mock_model

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_load_model_only_once(self, mock_transformer):
        """Test model is loaded only once (lazy loading)"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        generator.load_model()
        generator.load_model()  # Call again

        # Should only be called once due to lazy loading
        assert mock_transformer.call_count == 1

    def test_load_model_import_error(self):
        """Test error handling when sentence-transformers not installed"""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            generator = EmbeddingGenerator()
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                generator.load_model()

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_generate_embeddings_single_text(self, mock_transformer):
        """Test embedding generation for single text"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        text = "This is a test"
        embeddings = generator.generate_embeddings(text)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 384
        mock_model.encode.assert_called_once()

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_generate_embeddings_multiple_texts(self, mock_transformer):
        """Test embedding generation for multiple texts"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator.generate_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_generate_embeddings_with_batch_size(self, mock_transformer):
        """Test embedding generation with custom batch size"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(10, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        texts = [f"Text {i}" for i in range(10)]
        embeddings = generator.generate_embeddings(texts, batch_size=2)

        # Check that batch_size was passed
        call_args = mock_model.encode.call_args
        assert call_args[1]['batch_size'] == 2

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_generate_embeddings_progress_bar(self, mock_transformer):
        """Test progress bar parameter"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        texts = [f"Text {i}" for i in range(5)]

        # Test with progress bar enabled
        generator.generate_embeddings(texts, show_progress=True)
        call_args = mock_model.encode.call_args
        assert call_args[1]['show_progress_bar'] is True

        # Test with progress bar disabled
        generator.generate_embeddings(texts, show_progress=False)
        call_args = mock_model.encode.call_args
        assert call_args[1]['show_progress_bar'] is False

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_transformer):
        """Test getting embedding dimension"""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        dim = generator.get_embedding_dimension()

        assert dim == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_embeddings_are_numpy_arrays(self, mock_transformer):
        """Test that embeddings are returned as numpy arrays"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator.generate_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_empty_text_list(self, mock_transformer):
        """Test handling of empty text list"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([]).reshape(0, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        embeddings = generator.generate_embeddings([])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_model_loading_triggers_on_first_use(self, mock_transformer):
        """Test that model is loaded on first use of generate_embeddings"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        generator = EmbeddingGenerator()
        assert generator.model is None

        generator.generate_embeddings("Test")
        assert generator.model is not None

    @pytest.mark.requires_models
    @pytest.mark.slow
    def test_real_embedding_generation(self):
        """Integration test with real model (requires sentence-transformers)"""
        try:
            generator = EmbeddingGenerator()
            text = "This is a real test"
            embeddings = generator.generate_embeddings(text)

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == 1
            assert embeddings.shape[1] > 0  # Actual dimension
            assert not np.all(embeddings == 0)  # Not all zeros
        except ImportError:
            pytest.skip("sentence-transformers not installed")

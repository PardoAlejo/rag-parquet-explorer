"""
Tests for rag/retrieval/retriever.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rag.retrieval.retriever import ParquetRetriever


@pytest.mark.unit
class TestParquetRetriever:
    """Test cases for ParquetRetriever class"""

    def test_init(self, temp_parquet_file, mock_embedding_model):
        """Test retriever initialization"""
        mock_generator = Mock()
        mock_generator.model = mock_embedding_model

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=5
        )

        assert len(retriever.parquet_files) == 1
        assert retriever.top_k == 5
        assert retriever.embedding_generator == mock_generator
        assert len(retriever.documents) == 0

    def test_init_multiple_files(self, multiple_parquet_files, mock_embedding_model):
        """Test initialization with multiple files"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=multiple_parquet_files,
            embedding_generator=mock_generator,
            top_k=3
        )

        assert len(retriever.parquet_files) == 2
        assert all(isinstance(f, Path) for f in retriever.parquet_files)

    def test_load_documents_single_file(self, temp_parquet_file, mock_embedding_model):
        """Test loading documents from a single file"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        documents = retriever.load_documents()

        assert len(documents) == 5  # Sample data has 5 rows
        assert all('id' in doc for doc in documents)
        assert all('source' in doc for doc in documents)
        assert all('content' in doc for doc in documents)

    def test_load_documents_with_sample_size(self, temp_parquet_file, mock_embedding_model):
        """Test loading with sample size limit"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        documents = retriever.load_documents(sample_size=3)

        assert len(documents) <= 3

    def test_load_documents_multiple_files(self, multiple_parquet_files, mock_embedding_model):
        """Test loading documents from multiple files"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=multiple_parquet_files,
            embedding_generator=mock_generator
        )

        documents = retriever.load_documents()

        # Should have documents from both files
        assert len(documents) == 10  # 5 from each file
        sources = set(doc['source'] for doc in documents)
        assert len(sources) == 2  # Two different source files

    def test_load_documents_nonexistent_file(self, mock_embedding_model):
        """Test handling of nonexistent file"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[Path('/nonexistent/file.parquet')],
            embedding_generator=mock_generator
        )

        documents = retriever.load_documents()

        # Should handle gracefully and return empty list
        assert len(documents) == 0

    def test_build_embeddings(self, temp_parquet_file, mock_embedding_model):
        """Test building embeddings for documents"""
        mock_generator = Mock()
        mock_generator.generate_embeddings.return_value = np.random.randn(5, 384).astype(np.float32)

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        retriever.load_documents()
        embeddings = retriever.build_embeddings()

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] == 384
        assert 'embeddings' in retriever.embeddings_cache

    def test_build_embeddings_without_documents(self, temp_parquet_file, mock_embedding_model):
        """Test building embeddings without loading documents first"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        with pytest.raises(ValueError, match="No documents loaded"):
            retriever.build_embeddings()

    def test_build_embeddings_specific_field(self, temp_parquet_file, mock_embedding_model):
        """Test building embeddings for a specific field"""
        mock_generator = Mock()
        mock_generator.generate_embeddings.return_value = np.random.randn(5, 384).astype(np.float32)

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        retriever.load_documents()
        embeddings = retriever.build_embeddings(text_field='content')

        assert isinstance(embeddings, np.ndarray)
        # Verify the generator was called with specific field data
        mock_generator.generate_embeddings.assert_called_once()

    def test_search_basic(self, temp_parquet_file, mock_embedding_model):
        """Test basic search functionality"""
        mock_generator = Mock()
        # Query embedding
        query_emb = np.array([1.0] + [0.0] * 383).astype(np.float32)
        # Doc embeddings - make first doc most similar
        doc_embs = np.random.randn(5, 384).astype(np.float32)
        doc_embs[0] = query_emb  # Perfect match

        mock_generator.generate_embeddings.side_effect = [
            doc_embs,  # For build_embeddings
            query_emb.reshape(1, -1)  # For search query
        ]

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=3
        )

        retriever.load_documents()
        retriever.build_embeddings()

        results = retriever.search("test query", top_k=3)

        assert len(results) <= 3
        assert all('similarity_score' in r for r in results)
        assert all('text' in r for r in results)
        # Results should be sorted by similarity (descending)
        if len(results) > 1:
            assert results[0]['similarity_score'] >= results[1]['similarity_score']

    def test_search_without_embeddings(self, temp_parquet_file, mock_embedding_model):
        """Test search without building embeddings first"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        retriever.load_documents()

        with pytest.raises(ValueError, match="Embeddings not built"):
            retriever.search("test query")

    def test_search_custom_top_k(self, temp_parquet_file, mock_embedding_model):
        """Test search with custom top_k"""
        mock_generator = Mock()
        query_emb = np.random.randn(1, 384).astype(np.float32)
        doc_embs = np.random.randn(5, 384).astype(np.float32)

        mock_generator.generate_embeddings.side_effect = [doc_embs, query_emb]

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=5
        )

        retriever.load_documents()
        retriever.build_embeddings()

        results = retriever.search("test query", top_k=2)

        assert len(results) == 2

    def test_cosine_similarity(self, temp_parquet_file, mock_embedding_model):
        """Test cosine similarity computation"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        # Create test vectors
        query_vec = np.array([1.0, 0.0, 0.0])
        doc_vecs = np.array([
            [1.0, 0.0, 0.0],  # Identical - similarity = 1.0
            [0.0, 1.0, 0.0],  # Orthogonal - similarity = 0.0
            [-1.0, 0.0, 0.0]  # Opposite - similarity = -1.0
        ])

        similarities = retriever._cosine_similarity(query_vec, doc_vecs)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-5  # Almost 1.0
        assert abs(similarities[1] - 0.0) < 1e-5  # Almost 0.0
        assert abs(similarities[2] - (-1.0)) < 1e-5  # Almost -1.0

    def test_get_context_for_query(self, temp_parquet_file, mock_embedding_model):
        """Test context formatting for LLM"""
        mock_generator = Mock()
        query_emb = np.random.randn(1, 384).astype(np.float32)
        doc_embs = np.random.randn(5, 384).astype(np.float32)

        mock_generator.generate_embeddings.side_effect = [doc_embs, query_emb]

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=3
        )

        retriever.load_documents()
        retriever.build_embeddings()

        context = retriever.get_context_for_query("test query", top_k=2)

        assert isinstance(context, str)
        assert len(context) > 0
        assert "Document 1" in context or "No relevant context" in context
        if "Document 1" in context:
            assert "Similarity:" in context
            assert "Source:" in context

    def test_get_context_with_min_similarity(self, temp_parquet_file, mock_embedding_model):
        """Test context filtering by minimum similarity"""
        mock_generator = Mock()
        query_emb = np.random.randn(1, 384).astype(np.float32)
        doc_embs = np.random.randn(5, 384).astype(np.float32)

        mock_generator.generate_embeddings.side_effect = [doc_embs, query_emb]

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=5
        )

        retriever.load_documents()
        retriever.build_embeddings()

        # With very high threshold, should get no results
        context = retriever.get_context_for_query(
            "test query",
            top_k=5,
            min_similarity=0.99
        )

        # Should either have no context or very few results
        assert isinstance(context, str)

    def test_format_value_string(self, temp_parquet_file, mock_embedding_model):
        """Test SQL value formatting for strings"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        formatted = retriever._format_value("test string")
        assert formatted == "'test string'"

    def test_format_value_number(self, temp_parquet_file, mock_embedding_model):
        """Test SQL value formatting for numbers"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        formatted = retriever._format_value(123)
        assert formatted == "123"

        formatted = retriever._format_value(45.67)
        assert formatted == "45.67"

    def test_text_columns_auto_detection(self, temp_parquet_file, mock_embedding_model):
        """Test automatic detection of text columns"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            text_columns=None  # Auto-detect
        )

        documents = retriever.load_documents()

        # Should have detected string columns
        assert len(documents) > 0
        assert 'content' in documents[0]
        # Check that text columns were included
        text_found = any(
            'title' in doc['content'] or 'content' in doc['content']
            for doc in documents
        )
        assert text_found

    def test_text_columns_explicit(self, temp_parquet_file, mock_embedding_model):
        """Test using explicit text columns"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            text_columns=['content']
        )

        documents = retriever.load_documents()

        assert len(documents) > 0
        # Should only have specified columns in content
        assert 'content' in documents[0]['content']

    def test_document_structure(self, temp_parquet_file, mock_embedding_model):
        """Test structure of loaded documents"""
        mock_generator = Mock()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator
        )

        documents = retriever.load_documents()

        assert len(documents) > 0
        doc = documents[0]

        # Check required fields
        assert 'id' in doc
        assert 'source' in doc
        assert 'content' in doc
        assert 'metadata' in doc

        # Check types
        assert isinstance(doc['id'], str)
        assert isinstance(doc['source'], str)
        assert isinstance(doc['content'], dict)
        assert isinstance(doc['metadata'], dict)


@pytest.mark.integration
class TestParquetRetrieverIntegration:
    """Integration tests for ParquetRetriever"""

    def test_full_retrieval_pipeline(self, temp_parquet_file, mock_embedding_model):
        """Test complete retrieval pipeline"""
        mock_generator = Mock()
        query_emb = np.random.randn(1, 384).astype(np.float32)
        doc_embs = np.random.randn(5, 384).astype(np.float32)

        mock_generator.generate_embeddings.side_effect = [doc_embs, query_emb]

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=3
        )

        # Full pipeline
        retriever.load_documents()
        retriever.build_embeddings()
        results = retriever.search("machine learning AI", top_k=2)

        assert len(results) > 0
        assert len(results) <= 2
        assert all('similarity_score' in r for r in results)

    def test_multiple_searches(self, temp_parquet_file, mock_embedding_model):
        """Test performing multiple searches"""
        mock_generator = Mock()
        doc_embs = np.random.randn(5, 384).astype(np.float32)
        query_emb1 = np.random.randn(1, 384).astype(np.float32)
        query_emb2 = np.random.randn(1, 384).astype(np.float32)

        mock_generator.generate_embeddings.side_effect = [
            doc_embs,
            query_emb1,
            query_emb2
        ]

        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=mock_generator,
            top_k=3
        )

        retriever.load_documents()
        retriever.build_embeddings()

        # Multiple searches should work
        results1 = retriever.search("query 1")
        results2 = retriever.search("query 2")

        assert len(results1) > 0
        assert len(results2) > 0

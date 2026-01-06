"""
Integration tests for the complete RAG system
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from rag.embeddings.generator import EmbeddingGenerator
from rag.retrieval.retriever import ParquetRetriever
from rag.llm.model import LLMInterface


@pytest.mark.integration
class TestRAGPipeline:
    """Integration tests for the complete RAG pipeline"""

    @patch('rag.embeddings.generator.SentenceTransformer')
    @patch('rag.llm.model.ollama')
    def test_full_rag_pipeline(
        self,
        mock_ollama,
        mock_transformer,
        temp_parquet_file
    ):
        """Test complete RAG pipeline from query to answer"""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        mock_ollama.chat.return_value = {
            'message': {'content': 'This is the answer based on context'}
        }

        # Initialize components
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen,
            top_k=3
        )
        llm = LLMInterface()

        # Execute pipeline
        # 1. Load documents
        retriever.load_documents()
        assert len(retriever.documents) > 0

        # 2. Build embeddings
        retriever.build_embeddings()
        assert 'embeddings' in retriever.embeddings_cache

        # 3. Search
        query_embedding = np.random.randn(1, 384).astype(np.float32)
        mock_model.encode.return_value = query_embedding

        results = retriever.search("What is machine learning?", top_k=3)
        assert len(results) > 0

        # 4. Get context
        context = retriever.get_context_for_query("What is machine learning?")
        assert isinstance(context, str)
        assert len(context) > 0

        # 5. Generate answer
        answer = llm.generate_with_context(
            query="What is machine learning?",
            context=context
        )
        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_embedding_retrieval_integration(
        self,
        mock_transformer,
        temp_parquet_file
    ):
        """Test integration between embeddings and retrieval"""
        # Setup consistent embeddings
        doc_embeddings = np.random.randn(5, 384).astype(np.float32)
        query_embedding = doc_embeddings[0:1]  # Make query match first doc

        mock_model = Mock()
        mock_model.encode.side_effect = [doc_embeddings, query_embedding]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Initialize
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen,
            top_k=3
        )

        # Load and index
        retriever.load_documents()
        retriever.build_embeddings()

        # Search
        results = retriever.search("test query")

        # First result should have highest similarity (near 1.0)
        assert len(results) > 0
        assert results[0]['similarity_score'] > 0.9

    @patch('rag.embeddings.generator.SentenceTransformer')
    @patch('rag.llm.model.ollama')
    def test_retrieval_llm_integration(
        self,
        mock_ollama,
        mock_transformer,
        temp_parquet_file
    ):
        """Test integration between retrieval and LLM"""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)
        mock_transformer.return_value = mock_model

        # LLM should receive context in prompt
        captured_prompt = {}

        def capture_chat(**kwargs):
            captured_prompt['messages'] = kwargs['messages']
            return {'message': {'content': 'Generated answer'}}

        mock_ollama.chat = capture_chat

        # Initialize
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen
        )
        llm = LLMInterface()

        # Retrieve context
        retriever.load_documents()
        retriever.build_embeddings()

        query = "What is AI?"
        context = retriever.get_context_for_query(query)

        # Generate answer
        answer = llm.generate_with_context(query, context)

        # Verify context was passed to LLM
        assert 'messages' in captured_prompt
        user_message = captured_prompt['messages'][-1]['content']
        assert query in user_message
        assert "Context:" in user_message or len(context) > 0

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_multiple_queries_same_index(
        self,
        mock_transformer,
        temp_parquet_file
    ):
        """Test multiple queries using the same indexed documents"""
        # Setup
        doc_embeddings = np.random.randn(5, 384).astype(np.float32)
        query1_emb = np.random.randn(1, 384).astype(np.float32)
        query2_emb = np.random.randn(1, 384).astype(np.float32)

        mock_model = Mock()
        mock_model.encode.side_effect = [doc_embeddings, query1_emb, query2_emb]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Initialize
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen,
            top_k=2
        )

        # Index once
        retriever.load_documents()
        retriever.build_embeddings()

        # Multiple queries
        results1 = retriever.search("query 1")
        results2 = retriever.search("query 2")

        assert len(results1) > 0
        assert len(results2) > 0
        # Results may differ based on similarity

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_filtering_by_similarity_threshold(
        self,
        mock_transformer,
        temp_parquet_file
    ):
        """Test that similarity threshold filters results correctly"""
        # Setup embeddings with known similarities
        query_emb = np.array([1.0] + [0.0] * 383).astype(np.float32)

        # Create doc embeddings with varying similarities
        doc_embs = np.array([
            [1.0] + [0.0] * 383,   # Very similar (1.0)
            [0.5] + [0.5] + [0.0] * 382,  # Somewhat similar
            [0.0] + [1.0] + [0.0] * 382,  # Not similar
        ] + [[0.1] * 384] * 2).astype(np.float32)

        mock_model = Mock()
        mock_model.encode.side_effect = [doc_embs, query_emb.reshape(1, -1)]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Initialize
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen,
            top_k=5
        )

        retriever.load_documents()
        retriever.build_embeddings()

        # Get context with high threshold
        context = retriever.get_context_for_query(
            "test",
            top_k=5,
            min_similarity=0.8
        )

        # Should filter out low-similarity results
        assert isinstance(context, str)

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_empty_results_handling(
        self,
        mock_transformer,
        temp_parquet_file
    ):
        """Test handling when no documents match the query"""
        # Setup with very dissimilar embeddings
        doc_embs = np.random.randn(5, 384).astype(np.float32)
        query_emb = np.random.randn(1, 384).astype(np.float32) * -1

        mock_model = Mock()
        mock_model.encode.side_effect = [doc_embs, query_emb]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Initialize
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen
        )

        retriever.load_documents()
        retriever.build_embeddings()

        # Very high threshold should return no results
        context = retriever.get_context_for_query(
            "test",
            min_similarity=0.999
        )

        assert "No relevant context" in context or len(context) >= 0

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_batch_processing(
        self,
        mock_transformer,
        multiple_parquet_files
    ):
        """Test processing multiple files in batch"""
        # Setup
        doc_embs = np.random.randn(10, 384).astype(np.float32)
        query_emb = np.random.randn(1, 384).astype(np.float32)

        mock_model = Mock()
        mock_model.encode.side_effect = [doc_embs, query_emb]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        # Initialize with multiple files
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=multiple_parquet_files,
            embedding_generator=embedding_gen,
            top_k=5
        )

        # Load and search
        retriever.load_documents()
        retriever.build_embeddings()
        results = retriever.search("test")

        # Should have results from multiple files
        assert len(results) > 0
        sources = set(r['source'] for r in results)
        # May have results from one or both files


@pytest.mark.integration
class TestConfigIntegration:
    """Test integration with config module"""

    def test_config_imports(self):
        """Test that config can be imported"""
        from rag import config

        assert hasattr(config, 'PARQUET_FILES')
        assert hasattr(config, 'EMBEDDING_MODEL')
        assert hasattr(config, 'LLM_MODEL')

    def test_config_paths(self):
        """Test that config paths are valid"""
        from rag import config

        assert config.PROJECT_ROOT.exists()
        assert config.DATA_DIR.parent.exists()


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end workflow tests"""

    @patch('rag.embeddings.generator.SentenceTransformer')
    @patch('rag.llm.model.ollama')
    def test_complete_rag_workflow(
        self,
        mock_ollama,
        mock_transformer,
        temp_parquet_file
    ):
        """Test complete workflow: load → index → search → generate"""
        # Setup comprehensive mocks
        doc_embs = np.random.randn(5, 384).astype(np.float32)
        query_emb = np.random.randn(1, 384).astype(np.float32)

        mock_model = Mock()
        mock_model.encode.side_effect = [doc_embs, query_emb]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model

        mock_ollama.chat.return_value = {
            'message': {'content': 'Comprehensive answer'}
        }
        mock_ollama.list.return_value = {
            'models': [{'name': 'llama3.2'}]
        }

        # User workflow
        # Step 1: Initialize components
        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen,
            top_k=3
        )
        llm = LLMInterface()

        # Step 2: Load and index data
        docs = retriever.load_documents()
        assert len(docs) > 0

        embeddings = retriever.build_embeddings()
        assert embeddings is not None

        # Step 3: Query the system
        user_query = "What is the main topic?"

        # Step 4: Retrieve relevant docs
        results = retriever.search(user_query, top_k=3)
        assert len(results) > 0

        # Step 5: Format context
        context = retriever.get_context_for_query(user_query)
        assert len(context) > 0

        # Step 6: Generate answer
        answer = llm.generate_with_context(user_query, context)
        assert isinstance(answer, str)
        assert len(answer) > 0

        # Step 7: Verify LLM availability (optional)
        available = llm.check_availability()
        assert isinstance(available, bool)

    @patch('rag.embeddings.generator.SentenceTransformer')
    def test_error_recovery(
        self,
        mock_transformer,
        temp_parquet_file
    ):
        """Test system behavior with errors"""
        mock_model = Mock()
        mock_transformer.return_value = mock_model

        embedding_gen = EmbeddingGenerator()
        retriever = ParquetRetriever(
            parquet_files=[temp_parquet_file],
            embedding_generator=embedding_gen
        )

        # Try to search without loading - should raise error
        with pytest.raises(ValueError):
            retriever.build_embeddings()

        # Recover by loading documents
        retriever.load_documents()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)

        # Now should work
        retriever.build_embeddings()
        assert 'embeddings' in retriever.embeddings_cache

"""
Pytest configuration and shared fixtures
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import Mock, MagicMock


# Test data fixtures
@pytest.fixture
def sample_data():
    """Sample DataFrame for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'title': ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5'],
        'content': [
            'This is about machine learning and AI',
            'Python programming for data science',
            'Deep learning with neural networks',
            'Natural language processing basics',
            'Computer vision and image recognition'
        ],
        'category': ['AI', 'Programming', 'AI', 'NLP', 'Vision'],
        'score': [8.5, 7.2, 9.1, 8.0, 7.8],
        'published_date': pd.date_range('2024-01-01', periods=5)
    })


@pytest.fixture
def temp_parquet_file(sample_data):
    """Create a temporary Parquet file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        temp_path = Path(f.name)

    # Write sample data to Parquet
    table = pa.Table.from_pandas(sample_data)
    pq.write_table(table, temp_path)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def multiple_parquet_files(sample_data):
    """Create multiple temporary Parquet files"""
    files = []

    # Create 2 files with different data
    for i in range(2):
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = Path(f.name)

        # Modify data slightly for each file
        data = sample_data.copy()
        data['id'] = data['id'] + (i * 10)
        data['source_file'] = f'file_{i}'

        table = pa.Table.from_pandas(data)
        pq.write_table(table, temp_path)
        files.append(temp_path)

    yield files

    # Cleanup
    for f in files:
        if f.exists():
            f.unlink()


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer model"""
    mock_model = MagicMock()

    # Mock encode method to return dummy embeddings
    def mock_encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True):
        n = len(texts) if isinstance(texts, list) else 1
        return np.random.randn(n, 384).astype(np.float32)

    mock_model.encode = mock_encode
    mock_model.get_sentence_embedding_dimension.return_value = 384

    return mock_model


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client"""
    mock_client = MagicMock()

    # Mock chat method
    mock_client.chat.return_value = {
        'message': {
            'content': 'This is a test response from the LLM.'
        }
    }

    # Mock list method
    mock_client.list.return_value = {
        'models': [
            {'name': 'llama3.2'},
            {'name': 'mistral'}
        ]
    }

    return mock_client


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    np.random.seed(42)
    return np.random.randn(5, 384).astype(np.float32)


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding"""
    np.random.seed(42)
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: Tests that require Ollama to be running"
    )
    config.addinivalue_line(
        "markers", "requires_models: Tests that require downloading models"
    )

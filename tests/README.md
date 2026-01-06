# Tests Directory

This directory contains all test files for the RAG system.

## Quick Start

```bash
# From project root
./run_tests.sh

# Or directly with pytest
pytest
```

## Test Files

| File | Coverage | Description |
|------|----------|-------------|
| `conftest.py` | N/A | Shared fixtures and pytest configuration |
| `test_embeddings.py` | ~15 tests | Tests for `rag/embeddings/generator.py` |
| `test_retrieval.py` | ~25 tests | Tests for `rag/retrieval/retriever.py` |
| `test_llm.py` | ~20 tests | Tests for `rag/llm/model.py` |
| `test_utils.py` | ~15 tests | Tests for `rag/utils/helpers.py` |
| `test_integration.py` | ~10 tests | End-to-end integration tests |

**Total: ~85 tests**

## Running Tests

### Quick Commands

```bash
# All tests
pytest

# With verbose output
pytest -v

# Specific file
pytest tests/test_embeddings.py

# Specific test
pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_init_default_model

# By marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

### Using the Test Runner Script

```bash
# All tests
./run_tests.sh all

# Fast tests only (skip external dependencies)
./run_tests.sh fast

# With coverage
./run_tests.sh coverage

# Specific module
./run_tests.sh embeddings
./run_tests.sh retrieval
./run_tests.sh llm
./run_tests.sh utils

# Help
./run_tests.sh help
```

## Test Structure

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions/methods
- Mock all external dependencies
- Fast execution (<5 seconds total)
- No network calls, no file I/O (except temp files)

### Integration Tests (`@pytest.mark.integration`)
- Test multiple components together
- May use temp files and directories
- Test realistic workflows
- Medium execution time (<10 seconds)

### Markers

- `unit` - Unit tests
- `integration` - Integration tests
- `slow` - Tests taking >5 seconds
- `requires_ollama` - Needs Ollama running
- `requires_models` - Needs model downloads

## Fixtures

Available fixtures (from `conftest.py`):

- `sample_data` - Sample DataFrame for testing
- `temp_parquet_file` - Temporary Parquet file
- `multiple_parquet_files` - Multiple temp Parquet files
- `mock_embedding_model` - Mock SentenceTransformer
- `mock_ollama_client` - Mock Ollama client
- `sample_embeddings` - Pre-generated embeddings
- `temp_directory` - Temporary directory

## Writing Tests

### Template

```python
import pytest
from rag.module import YourClass

@pytest.mark.unit
class TestYourClass:
    """Test YourClass functionality"""

    def test_basic_functionality(self):
        """Test description"""
        # Arrange
        obj = YourClass()

        # Act
        result = obj.method()

        # Assert
        assert result == expected
```

### Using Fixtures

```python
def test_with_fixture(temp_parquet_file):
    """Fixtures are auto-injected"""
    assert temp_parquet_file.exists()
```

### Mocking

```python
from unittest.mock import patch, Mock

@patch('rag.module.ExternalDependency')
def test_with_mock(mock_dependency):
    """Mock external dependencies"""
    mock_dependency.return_value = Mock()
    # Test code
```

## Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=rag

# HTML report
pytest --cov=rag --cov-report=html
open htmlcov/index.html
```

### Current Coverage

- Embeddings: ~95%
- Retrieval: ~90%
- LLM: ~85%
- Utils: ~95%

**Target: >80% overall**

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Manual workflow triggers

See `.github/workflows/` for CI configuration.

## Troubleshooting

### Import Errors

```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in dev mode
pip install -e .
```

### Missing Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Run in parallel
pip install pytest-xdist
pytest -n auto
```

## Best Practices

✅ **Do:**
- Write tests for new features
- Mock external dependencies
- Use descriptive test names
- Keep tests independent
- Add docstrings

❌ **Don't:**
- Test implementation details
- Create test dependencies
- Use real models in unit tests
- Skip tests without reason
- Commit failing tests

## Resources

- [Full Testing Guide](../docs/TESTING.md)
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage Plugin](https://pytest-cov.readthedocs.io/)

---

For detailed testing documentation, see [docs/TESTING.md](../docs/TESTING.md)

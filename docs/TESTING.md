# Testing Documentation

Comprehensive testing guide for the RAG system.

## Overview

The test suite includes:
- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Mocked tests**: Test without external dependencies
- **Real tests**: Optional tests with real models (slower)

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_embeddings.py       # EmbeddingGenerator tests
├── test_retrieval.py        # ParquetRetriever tests
├── test_llm.py             # LLMInterface tests
├── test_utils.py           # Utility function tests
└── test_integration.py      # End-to-end integration tests
```

## Setup

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock pytest-asyncio
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=rag --cov-report=html
```

### Run Specific Test Files

```bash
# Test embeddings module only
pytest tests/test_embeddings.py

# Test retrieval module only
pytest tests/test_retrieval.py

# Test LLM module only
pytest tests/test_llm.py

# Test utils module only
pytest tests/test_utils.py

# Integration tests only
pytest tests/test_integration.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_embeddings.py::TestEmbeddingGenerator

# Run a specific test function
pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_init_default_model

# Run tests matching a pattern
pytest -k "embedding"
```

### Run Tests by Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests that don't require external services
pytest -m "not requires_ollama"

# Run tests that don't require model downloads
pytest -m "not requires_models"

# Skip slow tests
pytest -m "not slow"
```

## Test Markers

Tests are organized with markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (multiple components)
- `@pytest.mark.slow` - Slow tests (may take >5 seconds)
- `@pytest.mark.requires_ollama` - Requires Ollama to be running
- `@pytest.mark.requires_models` - Requires downloading models

## Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=rag

# HTML report (opens in browser)
pytest --cov=rag --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=rag --cov-report=xml
```

### Coverage Targets

Current coverage by module:
- `rag/embeddings/`: ~95%
- `rag/retrieval/`: ~90%
- `rag/llm/`: ~85%
- `rag/utils/`: ~95%

Target: >80% overall coverage

## Test Details

### Embeddings Tests (test_embeddings.py)

**What's tested:**
- Initialization with default/custom models
- Lazy model loading
- Single and batch embedding generation
- Error handling (missing dependencies)
- Parameter passing (batch_size, show_progress)
- Embedding dimensions

**Key test cases:**
- `test_init_default_model` - Default initialization
- `test_load_model_success` - Successful model loading
- `test_generate_embeddings_single_text` - Single text embedding
- `test_generate_embeddings_multiple_texts` - Batch processing
- `test_load_model_import_error` - Missing dependency handling

**Mocking:**
- Uses mock SentenceTransformer to avoid downloading models
- Returns dummy embeddings (random vectors)

### Retrieval Tests (test_retrieval.py)

**What's tested:**
- Document loading from Parquet files
- Embedding generation for documents
- Similarity search
- Context formatting for LLM
- Multiple file handling
- Sample size limiting
- Error recovery

**Key test cases:**
- `test_load_documents_single_file` - Basic document loading
- `test_build_embeddings` - Embedding creation
- `test_search_basic` - Semantic search
- `test_cosine_similarity` - Similarity computation
- `test_get_context_for_query` - Context formatting

**Mocking:**
- Uses temporary Parquet files created in fixtures
- Mock embeddings for predictable results

### LLM Tests (test_llm.py)

**What's tested:**
- Ollama client initialization
- Text generation
- RAG-specific generation (with context)
- System prompts
- Parameter customization (temperature, max_tokens)
- Model availability checking
- Error handling

**Key test cases:**
- `test_generate_basic` - Simple text generation
- `test_generate_with_system_prompt` - Custom system prompts
- `test_generate_with_context_basic` - RAG generation
- `test_check_availability_success` - Model availability
- `test_generate_custom_temperature` - Parameter overrides

**Mocking:**
- Mock Ollama client to avoid needing running server
- Returns predefined responses

### Utils Tests (test_utils.py)

**What's tested:**
- Logging setup with various levels
- Log file creation
- Directory creation (single, multiple, nested)
- Permission handling
- Error recovery

**Key test cases:**
- `test_setup_logging_default` - Default logging
- `test_setup_logging_with_file` - File logging
- `test_ensure_single_directory` - Directory creation
- `test_ensure_nested_directories` - Nested structures

**Mocking:**
- Uses temporary directories for isolation
- Real file system operations (safe with temp dirs)

### Integration Tests (test_integration.py)

**What's tested:**
- Complete RAG pipeline (end-to-end)
- Component interactions
- Multiple queries on same index
- Error recovery
- Similarity thresholding
- Batch processing

**Key test cases:**
- `test_full_rag_pipeline` - Complete workflow
- `test_embedding_retrieval_integration` - Embeddings + retrieval
- `test_retrieval_llm_integration` - Retrieval + LLM
- `test_complete_rag_workflow` - User workflow simulation

**Mocking:**
- Combines all component mocks
- Simulates realistic workflows

## Writing New Tests

### Test Template

```python
import pytest
from unittest.mock import patch, Mock

@pytest.mark.unit
def test_my_feature():
    """Test description"""
    # Arrange
    input_data = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output
```

### Using Fixtures

```python
def test_with_fixture(temp_parquet_file, mock_embedding_model):
    """Use shared fixtures from conftest.py"""
    # Fixtures are automatically available
    assert temp_parquet_file.exists()
```

### Mocking External Dependencies

```python
@patch('rag.embeddings.generator.SentenceTransformer')
def test_with_mock(mock_transformer):
    """Mock external library"""
    mock_transformer.return_value = Mock()
    # Your test code
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=rag --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Ensure you're in the project root
cd /path/to/rag_example

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests Fail with Missing Fixtures

```bash
# Ensure conftest.py is in tests/
ls tests/conftest.py

# Run pytest with verbose to see fixture issues
pytest -v --fixtures
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Skip tests requiring external services
pytest -m "not requires_ollama and not requires_models"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

### Coverage Not Working

```bash
# Reinstall pytest-cov
pip install --upgrade pytest-cov

# Ensure running from project root
pytest --cov=rag --cov-report=term

# Check .coveragerc if you have one
```

## Best Practices

### Do's ✅
- Write tests before fixing bugs (TDD)
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests independent (no shared state)
- Mock external dependencies
- Use fixtures for common setup
- Add docstrings to test functions

### Don'ts ❌
- Don't test implementation details
- Don't write tests that depend on execution order
- Don't use real models in unit tests
- Don't commit failed tests
- Don't skip tests without comments
- Don't test third-party libraries

## Test Metrics

### Current Status

```bash
# Run tests with summary
pytest --tb=short --quiet

# Count tests
pytest --collect-only | grep "test session starts"
```

**Test Count:**
- Embeddings: ~15 tests
- Retrieval: ~25 tests
- LLM: ~20 tests
- Utils: ~15 tests
- Integration: ~10 tests
- **Total: ~85 tests**

### Performance

Typical test run times:
- Unit tests: <5 seconds
- Integration tests: <10 seconds
- Full suite: <15 seconds
- With real models: 1-2 minutes

## Advanced Testing

### Testing with Real Models

```python
# Mark as requiring real models
@pytest.mark.requires_models
@pytest.mark.slow
def test_real_embedding():
    """Test with actual SentenceTransformer"""
    generator = EmbeddingGenerator()
    result = generator.generate_embeddings("test")
    assert result.shape[1] == 384
```

Run with:
```bash
pytest -m requires_models
```

### Parameterized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("world", 5),
    ("test", 4),
])
def test_length(input, expected):
    assert len(input) == expected
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Coverage Plugin](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure >80% coverage
3. Add integration tests for new workflows
4. Update this documentation
5. Run full test suite before submitting

---

**Questions?** Check the main [README.md](../README.md) or open an issue.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **local RAG (Retrieval Augmented Generation) system** for querying Parquet files using semantic search and local LLMs. It consists of two Streamlit applications:

1. **Parquet Explorer** (`tools/parquet_explorer/`) - Interactive data visualization tool
2. **RAG Query System** (`rag/`) - AI-powered semantic search with local Ollama LLM

## Key Commands

### Running the Applications

```bash
# Parquet Explorer (port 8501)
cd tools/parquet_explorer && streamlit run app.py

# RAG Query System (requires Ollama running, port 8502)
ollama serve                    # In separate terminal
streamlit run rag/app.py

# Quick validation (no Ollama needed)
python demo.py
```

### Testing

```bash
# Run all tests
./run_tests.sh

# Run specific test categories
./run_tests.sh unit          # Unit tests only
./run_tests.sh integration   # Integration tests only
./run_tests.sh fast          # Skip slow and external dependency tests
./run_tests.sh coverage      # Generate coverage report

# Run specific module tests
./run_tests.sh embeddings
./run_tests.sh retrieval
./run_tests.sh llm

# Run single test with pytest directly
pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_init_default_model -v

# Run tests by marker
pytest -m unit               # All unit tests
pytest -m "not requires_ollama"  # Skip Ollama-dependent tests
```

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python demo.py

# Check syntax
python -m py_compile rag/app.py
```

## Architecture

### RAG System Data Flow

```
User Query → EmbeddingGenerator (text → 384-dim vector)
           → ParquetRetriever (cosine similarity search)
           → Top-K Documents Retrieved
           → Context Builder (format for LLM)
           → LLMInterface (Ollama)
           → Generated Answer
```

### Component Dependencies

- **`rag/app.py`**: Streamlit UI that orchestrates all components
- **`rag/config.py`**: Single source of truth for all configuration
- **`rag/embeddings/generator.py`**: Wraps sentence-transformers models
- **`rag/retrieval/retriever.py`**: Loads Parquet files, builds embeddings, performs search
- **`rag/llm/model.py`**: Interfaces with Ollama for text generation

**Key Relationship**: `ParquetRetriever` depends on `EmbeddingGenerator` (injected) to embed both documents and queries.

### Critical Design Patterns

1. **Lazy Loading**: Models (`EmbeddingGenerator`, `LLMInterface`) only initialize when first used
2. **Streamlit Caching**: `@st.cache_resource` on `initialize_rag_system()` prevents re-initialization
3. **In-Memory Embedding Cache**: `ParquetRetriever.embeddings_cache` stores generated embeddings
4. **Configuration Injection**: All components read from `rag/config.py`

### Parquet → Documents → Embeddings Pipeline

```
Parquet Files → DuckDB (zero-copy scan)
              → Document objects (id, source, content dict, metadata dict)
              → SentenceTransformer
              → NumPy arrays (shape: [n_docs, 384])
              → Cosine similarity search
```

## Configuration

**All settings in `rag/config.py`** - edit this file to customize:

| Setting | Purpose | Impact |
|---------|---------|--------|
| `PARQUET_FILES` | List of data files to index | Add/remove Parquet files here |
| `EMBEDDING_MODEL` | HuggingFace model name | Controls search quality (default: 384-dim) |
| `LLM_MODEL` | Ollama model name | Which LLM to use (default: llama3.2) |
| `TOP_K_RESULTS` | Number of docs to retrieve | Higher = more context, slower |
| `SIMILARITY_THRESHOLD` | Min similarity score | Higher = stricter matching |
| `LLM_TEMPERATURE` | Response creativity | 0 = focused, 1 = creative |

**Important**: After changing `PARQUET_FILES` or `EMBEDDING_MODEL`, embeddings must be rebuilt (clear cache and restart app).

## Testing Philosophy

### Test Organization

Tests use **extensive mocking** to avoid external dependencies:

- `tests/conftest.py`: Shared fixtures (mock models, temp Parquet files, sample data)
- Unit tests (`@pytest.mark.unit`): Fast, isolated, mock all external deps
- Integration tests (`@pytest.mark.integration`): Multiple components, still mocked
- Real tests (`@pytest.mark.requires_models`, `@pytest.mark.requires_ollama`): Actual model downloads/Ollama

### Key Fixtures

- `temp_parquet_file`: Auto-cleaned temporary Parquet file with sample data
- `mock_embedding_model`: Returns random 384-dim vectors (no model download)
- `mock_ollama_client`: Returns canned responses (no Ollama needed)
- `sample_embeddings`: Pre-generated NumPy arrays for testing

### Running Tests Without External Services

The test suite is designed to run without Ollama or model downloads:

```bash
./run_tests.sh fast   # Skips @pytest.mark.requires_ollama and @pytest.mark.requires_models
```

Use this for CI/CD or when working offline.

## Common Development Tasks

### Adding a New Parquet File

1. Place file in `data/parquet_files/your_file.parquet`
2. Edit `rag/config.py`:
   ```python
   PARQUET_FILES = [
       DATA_DIR / "Spain_translated.parquet",
       DATA_DIR / "France_translated.parquet",
       DATA_DIR / "your_file.parquet",  # Add here
   ]
   ```
3. Restart RAG app (embeddings rebuild automatically)

### Changing the Embedding Model

1. Edit `rag/config.py`:
   ```python
   EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality, slower
   # or
   EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster
   ```
2. Clear cache and restart app

### Changing the LLM

1. Pull new model: `ollama pull mistral`
2. Edit `rag/config.py`:
   ```python
   LLM_MODEL = "mistral"
   ```
3. Restart app

### Debugging Initialization Issues

The RAG system initialization can fail silently. Common issues:

1. **Parquet files not found**: Check `PARQUET_FILES` paths in config
2. **Ollama not running**: Verify with `ollama list`
3. **Model not installed**: Run `ollama pull <model_name>`
4. **Out of memory**: Reduce sample size in `retriever.load_documents(sample_size=1000)`

Add debugging:
```python
# In rag/app.py, inside initialize_rag_system()
print(f"Loading from: {PARQUET_FILES}")
docs = retriever.load_documents()
print(f"Loaded {len(docs)} documents")
embeddings = retriever.build_embeddings()
print(f"Generated embeddings: {embeddings.shape}")
```

### Adding Tests for New Features

1. Add test file in `tests/test_<module>.py`
2. Use fixtures from `conftest.py`:
   ```python
   def test_my_feature(temp_parquet_file, mock_embedding_model):
       # temp_parquet_file has sample data
       # mock_embedding_model returns dummy embeddings
       pass
   ```
3. Mark appropriately:
   ```python
   @pytest.mark.unit
   def test_fast_isolated(): ...

   @pytest.mark.integration
   def test_multiple_components(): ...

   @pytest.mark.requires_ollama
   def test_with_real_ollama(): ...
   ```

## Performance Characteristics

- **Initial indexing**: 1-2 minutes for 100k documents (one-time cost)
- **Query search**: 0.5-1 second (cosine similarity in NumPy)
- **LLM inference**: 2-5 seconds (depends on response length and model)
- **Memory usage**: ~2-4GB for 100k documents with embeddings
- **Scalability**: Current in-memory approach handles up to ~100k documents

For larger datasets, implement FAISS vector store (placeholder in config: `USE_FAISS = True`).

## UI Customization

The RAG app uses custom CSS for answer display (`rag/app.py` lines 27-104):

```python
# Answer box styling
.answer-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-left: 5px solid #ffd700;
}
```

Modify CSS here for theming changes. The answer box uses `st.markdown()` with `unsafe_allow_html=True`.

## Data Directory Structure

```
data/
├── .gitkeep                     # Tracked in git
└── parquet_files/
    ├── .gitkeep                 # Tracked in git
    ├── Spain_translated.parquet # NOT tracked (in .gitignore)
    └── France_translated.parquet # NOT tracked (in .gitignore)
```

Parquet files are gitignored to avoid committing large data files. Only folder structure (`.gitkeep` files) is tracked.

## Important Notes

### Text Column Auto-Detection

`ParquetRetriever` auto-detects text columns if not specified:

```python
# In retriever.py load_documents()
if self.text_columns is None:
    # Auto-detect: any column with dtype 'object' or 'string'
    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
```

To specify manually:
```python
retriever = ParquetRetriever(
    parquet_files=...,
    embedding_generator=...,
    text_columns=['title', 'content']  # Explicit columns
)
```

### DuckDB Integration

Both apps use DuckDB's `parquet_scan()` for zero-copy Parquet access:

```python
query = f"SELECT * FROM parquet_scan('{file_path}') LIMIT 1000"
df = conn.execute(query).fetchdf()
```

This allows SQL queries without loading entire file into memory.

### Cosine Similarity Implementation

Search uses NumPy for fast cosine similarity (in `retriever.py`):

```python
def _cosine_similarity(query_vec, doc_vecs):
    # Normalize vectors
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    # Compute similarities
    return np.dot(doc_norms, query_norm)
```

Results sorted descending by similarity score.

## Troubleshooting

### "Module not found" errors

Ensure you're running from project root and dependencies are installed:
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Tests fail with import errors

```bash
# Verify you're in project root
pwd  # Should end with /rag_example

# Run from correct location
pytest tests/
```

### RAG app shows "Initializing..." forever

Check logs in `logs/rag.log` and verify:
1. Parquet files exist at paths in `PARQUET_FILES`
2. No errors during embedding generation
3. Sufficient memory available

### Ollama connection errors

```bash
# Verify Ollama is running
ollama list

# Check specific model exists
ollama pull llama3.2

# Test connection
python -c "from rag.llm import LLMInterface; llm = LLMInterface(); print(llm.check_availability())"
```

## Documentation

- **START_HERE.md**: Quick start guide for new users
- **EXAMPLES.md**: Detailed usage examples and workflows
- **docs/ARCHITECTURE.md**: System architecture deep dive
- **docs/TESTING.md**: Comprehensive testing guide
- **docs/SETUP.md**: Installation and setup instructions
- **tests/README.md**: Quick testing reference

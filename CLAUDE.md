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
           → Threshold Filter (on cosine scores)
           → [Optional] Reranker (cross-encoder re-scoring)
           → Top-K Documents Retrieved
           → Context Builder (format for LLM)
           → LLMInterface (Ollama)
           → Generated Answer

First Run:  Parquet → DuckDB → Documents → Embeddings → Cache to disk
Next Runs:  Cache → Load embeddings (90x faster)
```

### Component Dependencies

- **`rag/app.py`**: Streamlit UI that orchestrates all components
- **`rag/config.py`**: Single source of truth for all configuration
- **`rag/embeddings/generator.py`**: Wraps sentence-transformers models (bi-encoder)
- **`rag/retrieval/retriever.py`**: Loads Parquet files, builds embeddings, performs search
- **`rag/retrieval/reranker.py`**: Cross-encoder re-ranking for improved relevance (NEW)
- **`rag/utils/cache.py`**: Persistent embedding cache with file change detection (NEW)
- **`rag/llm/model.py`**: Interfaces with Ollama for text generation

**Key Relationships**:
- `ParquetRetriever` depends on `EmbeddingGenerator` (bi-encoder) for initial retrieval
- `ParquetRetriever` optionally uses `Reranker` (cross-encoder) for improved ordering
- `ParquetRetriever` uses `EmbeddingCache` for persistent storage

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
| `EMBEDDING_MODEL` | HuggingFace bi-encoder model | Controls search quality (default: 384-dim) |
| `LLM_MODEL` | Ollama model name | Which LLM to use (default: llama3.2) |
| `TOP_K_RESULTS` | Number of docs to retrieve | Higher = more context, slower |
| `SIMILARITY_THRESHOLD` | Min cosine similarity | Higher = stricter matching (0-1 range) |
| `LLM_TEMPERATURE` | Response creativity | 0 = focused, 1 = creative |
| `USE_RERANKING` | Enable cross-encoder re-ranking | Better relevance, ~2x slower |
| `RERANKER_MODEL` | Cross-encoder model name | Default: ms-marco-MiniLM-L-6-v2 |

**Important**:
- After changing `PARQUET_FILES` or `EMBEDDING_MODEL`, clear cache and restart app
- `SIMILARITY_THRESHOLD` applies to **cosine similarity** (0-1), NOT cross-encoder scores
- Cache directory: `.cache/` (auto-created, stores embeddings + metadata)

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

## Recent Features & Fixes

### New Features (Jan 2026)

1. **Re-ranking System** (`rag/retrieval/reranker.py`)
   - Uses cross-encoder model for improved document relevance
   - Configurable via `USE_RERANKING` in config.py
   - See `docs/features/FEATURE_SUMMARY.md` for details
   - **Known Issue FIXED**: See `docs/fixes/reranker-threshold.md`
   - ⚠️ **Effectiveness Note**: Current model not effective on Spanish/French data (see below)

2. **Instant Re-ranking Toggle** (NEW - Jan 7, 2026)
   - Smart caching allows instant switching between original and re-ranked results
   - Toggle feels instant (<100ms) vs old reload (2-5s)
   - Pre-computes both versions for seamless comparison
   - See `docs/features/instant-reranking-toggle.md` for details
   - **UX improvement**: 20-50x faster toggle operations

3. **Embedding Cache System** (`rag/utils/cache.py`)
   - Persistent caching to avoid recomputing embeddings
   - 90x speedup for app restarts (90s → <1s)
   - See `docs/features/FEATURE_SUMMARY.md` for details
   - **Known Issue FIXED**: See `docs/fixes/cache-invalidation.md`

4. **Retrieval Pool System** (NEW - Jan 7, 2026)
   - Retrieve large pool (50-100 docs) → Rank all → Show/use top-k
   - Makes re-ranking more effective by giving it more candidates
   - User-adjustable pool size (10-100) and top-k (1-20) in UI
   - Smart caching: pool size/threshold change triggers re-search
   - See `docs/features/retrieval-pool-system.md` for architecture

5. **UI Improvements** (NEW - Jan 7, 2026)
   - Lower default similarity threshold (0.7 → 0.5) for more permissive retrieval
   - Document titles shown in expanders (from 'title' field) instead of filenames
   - Dark mode support with purple gradient answer box
   - See `docs/fixes/ui-dark-mode.md` for CSS details

### Critical Bugs Fixed

**IMPORTANT: Read these before debugging related issues**

1. **Re-ranker Threshold Bug** (CRITICAL - would return 0 results)
   - **Problem**: Cross-encoder scores (-10 to +10) filtered by cosine threshold (0-1)
   - **Symptom**: Re-ranking enabled + any threshold = no results
   - **Fix**: Apply threshold BEFORE re-ranking, preserve both scores
   - **Details**: `docs/fixes/reranker-threshold.md`

2. **Cache Invalidation Bug** (HIGH - defeated caching purpose)
   - **Problem**: mtime changes (git, backups) triggered unnecessary rebuilds
   - **Symptom**: Cache rebuilt every app restart despite no file changes
   - **Fix**: Only hash-verify when mtime changes but size is same
   - **Details**: `docs/fixes/cache-invalidation.md`

3. **UI Dark Mode Bug** (MEDIUM - visibility issue)
   - **Problem**: Light green background invisible in dark theme
   - **Symptom**: Answer box barely visible
   - **Fix**: Purple gradient with white text works in all themes
   - **Details**: `docs/fixes/ui-dark-mode.md`

### Known Limitations

**Re-ranking Effectiveness (Jan 7, 2026)**

⚠️ **Current Status**: Re-ranking is NOT effective for Spanish/French news data

**Test Results:**
- Tested on 151,224 Spanish/French news articles
- 4 diverse queries with pool size of 50 documents
- Result: **0% document reordering** (exact same order as cosine similarity)
- Verdict: Re-ranking adds 2-5s latency with zero benefit

**Root Cause:**
- Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) trained on English Q&A
- Dataset mismatch: Spanish/French news vs English questions
- Embedding model (`all-MiniLM-L6-v2`) already very effective

**What We Keep:**
- ✅ Feature fully implemented and tested
- ✅ UI toggle and pool system working correctly
- ✅ Ready for improvement with better model
- ✅ Documented for future reference

**See:** `docs/features/reranking-effectiveness.md` for full analysis

**Test Script:**
```bash
python test_reranking_effectiveness.py
```

## Future Enhancements

### Near-Term Improvements

**1. Multilingual Cross-Encoder** (RECOMMENDED NEXT)
- **Priority**: HIGH
- **Effort**: 30 minutes
- **Model**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Why**: Trained on multilingual data (100+ languages including Spanish/French)
- **Expected**: 20-40% document reordering vs current 0%
- **Action**: Change `RERANKER_MODEL` in `rag/config.py` and re-test

### Long-Term Vision

**2. Graph-Based Retrieval** 🔮
- **Priority**: FUTURE (major feature)
- **Effort**: Weeks/months
- **Impact**: Transformative for complex queries

**Concept:**
- Build knowledge graph from document entities and relationships
- Combine vector similarity with graph connectivity
- Multi-hop reasoning for complex questions
- Better citation and reference tracking

**Potential Approaches:**
- Entity linking + graph traversal (e.g., "How is X related to Y?")
- Document-level graph with semantic edges
- Hybrid: Vector retrieval → Graph expansion → Re-ranking

**Technologies to Explore:**
- LlamaIndex (graph-based RAG)
- LangGraph (workflow graphs)
- Neo4j + vector search
- NetworkX for custom algorithms

**Success Metrics:**
- Better multi-document query handling
- Improved reference tracking
- More comprehensive answers requiring multiple sources

**3. Domain-Specific Fine-Tuning**
- Fine-tune cross-encoder on Spanish/French news Q&A pairs
- Create training dataset from actual queries and relevance judgments
- Expected: >50% improvement in ranking quality

**4. Hybrid Retrieval**
- Combine dense (embeddings) + sparse (BM25) + graph
- Reciprocal Rank Fusion (RRF) for merging
- Better coverage across query types

## Documentation Structure

### User-Facing Docs (Root)
- **README.md**: Main project overview
- **START_HERE.md**: Quick start guide (5 minute setup)
- **EXAMPLES.md**: Detailed usage examples and workflows
- **QUICK_REFERENCE.md**: Command cheat sheet
- **CLAUDE.md**: This file - for Claude Code instances

### Technical Docs (docs/)
- **docs/ARCHITECTURE.md**: System architecture deep dive
- **docs/TESTING.md**: Comprehensive testing guide
- **docs/SETUP.md**: Installation and setup instructions
- **docs/TEST_SUMMARY.md**: Test coverage summary

### Feature Documentation (docs/features/)
- **docs/features/FEATURE_SUMMARY.md**: Re-ranker and cache features overview
- **docs/features/instant-reranking-toggle.md**: Smart toggle implementation
- **docs/features/retrieval-pool-system.md**: Pool-based retrieval architecture
- **docs/features/reranking-effectiveness.md**: ⚠️ Effectiveness analysis and recommendations

### Fix Documentation (docs/fixes/)
- **docs/fixes/reranker-threshold.md**: Re-ranker threshold bug and fix
- **docs/fixes/cache-invalidation.md**: Cache invalidation bug and fix
- **docs/fixes/ui-dark-mode.md**: Dark mode visibility fix

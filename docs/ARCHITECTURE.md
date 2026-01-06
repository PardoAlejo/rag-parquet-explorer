# Architecture Documentation

## System Overview

This project consists of two independent but complementary tools:

1. **Parquet Explorer**: Data visualization and exploration tool
2. **RAG Query System**: Retrieval-augmented generation for querying data

Both tools are built with Streamlit for the UI and share common dependencies (PyArrow, DuckDB, Pandas).

## Component Architecture

### 1. Parquet Explorer

```
┌─────────────────────────────────────┐
│      Streamlit Web Interface        │
│  (tools/parquet_explorer/app.py)    │
└───────────┬─────────────────────────┘
            │
            ├── ParquetExplorer Class
            │   ├── File Loading (PyArrow)
            │   ├── DuckDB Query Engine
            │   └── Metadata Extraction
            │
            ├── Visualization Layer
            │   ├── Plotly Charts
            │   └── Streamlit DataFrames
            │
            └── Export Functions
                ├── CSV Export
                └── JSON Export
```

**Key Design Decisions:**

- **Lazy Loading**: Uses PyArrow to read Parquet metadata without loading full file
- **DuckDB Integration**: SQL queries run directly on Parquet files (zero-copy)
- **Streaming**: Pagination prevents loading entire dataset into memory
- **Caching**: Streamlit's `@cache_resource` for expensive operations

**Data Flow:**

```
Parquet File → PyArrow Reader → DuckDB Scanner → Query Execution → Pandas DataFrame → Streamlit UI
```

### 2. RAG Query System

```
┌─────────────────────────────────────┐
│      Streamlit Web Interface        │
│         (rag/app.py)                │
└───────────┬─────────────────────────┘
            │
            ├── Embedding Layer (rag/embeddings/)
            │   ├── SentenceTransformer Model
            │   ├── Text → Vector Conversion
            │   └── Embedding Cache
            │
            ├── Retrieval Layer (rag/retrieval/)
            │   ├── Document Loader (DuckDB + Parquet)
            │   ├── Vector Similarity Search
            │   └── Context Builder
            │
            └── LLM Layer (rag/llm/)
                ├── Ollama Client
                ├── Prompt Builder
                └── Response Generation
```

**Key Design Decisions:**

- **Modular Components**: Separate modules for embeddings, retrieval, LLM
- **Lazy Initialization**: Components loaded only when needed
- **Caching**: Embeddings cached to avoid recomputation
- **Local-First**: All computation happens locally (no API calls)

**Data Flow:**

```
User Query
    ↓
[Embedding Generation]
    ↓ (query vector)
[Similarity Search] ← [Document Embeddings]
    ↓ (top-k docs)
[Context Building]
    ↓ (formatted context)
[LLM Prompt] → [Ollama] → [Generated Answer]
    ↓
User Interface
```

## Module Breakdown

### rag/config.py

**Purpose**: Centralized configuration management

**Contains**:
- File paths (data, models, cache)
- Model configurations (embedding, LLM)
- Hyperparameters (top-k, temperature, thresholds)
- Application settings (ports, logging)

**Pattern**: All imports go through config for easy customization

### rag/embeddings/generator.py

**Purpose**: Convert text to vector embeddings

**Key Methods**:
- `load_model()`: Lazy load SentenceTransformer
- `generate_embeddings()`: Batch embedding generation
- `get_embedding_dimension()`: Return embedding size

**Model**: Uses sentence-transformers library (HuggingFace)

**Performance**: Batched processing for efficiency

### rag/retrieval/retriever.py

**Purpose**: Load, index, and search Parquet documents

**Key Methods**:
- `load_documents()`: Read Parquet files via DuckDB
- `build_embeddings()`: Generate embeddings for all documents
- `search()`: Semantic similarity search
- `get_context_for_query()`: Format results for LLM

**Search Algorithm**: Cosine similarity in embedding space

**Storage**: In-memory (suitable for up to ~100k documents)

### rag/llm/model.py

**Purpose**: Interface with local LLM via Ollama

**Key Methods**:
- `load_model()`: Connect to Ollama
- `generate()`: General text generation
- `generate_with_context()`: RAG-specific generation
- `check_availability()`: Verify Ollama and model

**Communication**: HTTP API to Ollama server

**Prompt Engineering**: System prompts guide RAG behavior

### rag/utils/helpers.py

**Purpose**: Utility functions

**Functions**:
- `setup_logging()`: Configure logging
- `ensure_directories()`: Create required directories

## Data Flow Diagrams

### RAG Query Flow

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
┌──────────────┐            ┌─────────────────┐
│ Query        │            │ Document Store  │
│ Embedding    │            │ (Parquet Files) │
└──────┬───────┘            └────────┬────────┘
       │                             │
       │                             ▼
       │                    ┌─────────────────┐
       │                    │ Document        │
       │                    │ Embeddings      │
       │                    └────────┬────────┘
       │                             │
       └─────────────┬───────────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Similarity      │
            │ Computation     │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Top-K Results   │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Context         │
            │ Builder         │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ LLM Prompt      │
            │ + Context       │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Ollama LLM      │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Generated       │
            │ Answer          │
            └─────────────────┘
```

### Parquet Explorer Flow

```
┌──────────────┐
│ File Path    │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ PyArrow          │
│ ParquetFile      │
└──────┬───────────┘
       │
       ├─────────────┬─────────────┬─────────────┐
       │             │             │             │
       ▼             ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Metadata  │  │ Schema   │  │DuckDB    │  │Filtering │
│Extraction│  │Inspection│  │Query     │  │& Export  │
└──────────┘  └──────────┘  └────┬─────┘  └──────────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │ Pandas DF    │
                           └──────┬───────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │ Streamlit UI │
                           │ + Plotly     │
                           └──────────────┘
```

## Performance Considerations

### Parquet Explorer

**Optimizations**:
1. **Zero-copy reads**: DuckDB scans Parquet directly
2. **Lazy evaluation**: Only requested data is loaded
3. **Sampling**: Option to work with subset of data
4. **Column pruning**: Only selected columns loaded

**Bottlenecks**:
- Very large files (>10GB): Consider sampling
- Complex filters: DuckDB handles efficiently
- Visualizations: Plotly can slow with >100k points

### RAG System

**Optimizations**:
1. **Embedding caching**: Generated once, reused
2. **Batch processing**: Embeddings generated in batches
3. **Streamlit caching**: `@cache_resource` for models
4. **Cosine similarity**: Efficient numpy operations

**Bottlenecks**:
- Initial indexing: ~1-2 min for 100k docs
- Embedding generation: CPU-bound (can use GPU)
- LLM inference: Depends on Ollama/model size
- Large document stores: Consider FAISS for >100k docs

## Scalability

### Current Limitations

**Parquet Explorer**:
- Designed for files up to ~10GB
- Pagination handles larger files
- Single-file focus (merging is basic)

**RAG System**:
- In-memory vectors: ~100k documents max
- No persistence: Rebuilds on restart
- Single-query processing: No batch mode

### Scaling Strategies

**For Large Datasets**:
1. Implement FAISS vector store
2. Persist embeddings to disk
3. Document chunking for long texts
4. Incremental indexing

**For Production**:
1. Add FastAPI backend (decouple from Streamlit)
2. Implement vector database (Qdrant, Weaviate)
3. Add caching layer (Redis)
4. Queue system for batch processing

## Security Considerations

**Current State**: Local-only, no authentication

**For Multi-User**:
- Add authentication layer
- Sandboxed file access
- Rate limiting on queries
- Input sanitization

**Data Privacy**:
- All data stays local
- No telemetry or external calls
- Embedding models run locally

## Extension Points

### Adding New Data Sources

**Parquet Explorer**:
- Already supports multiple files
- Could add: CSV, JSON, database connectors

**RAG System**:
- Modify `ParquetRetriever` to support other formats
- Add readers in `rag/retrieval/`

### Adding New Models

**Embeddings**:
- Change `EMBEDDING_MODEL` in config
- Any sentence-transformers model works

**LLM**:
- Change `LLM_MODEL` in config
- Any Ollama model works
- Could add: OpenAI API, Anthropic API, local transformers

### Adding Vector Databases

Replace in-memory storage:

```python
# In rag/retrieval/retriever.py
# Current: numpy arrays
# Future: FAISS, Qdrant, Chroma, etc.

from qdrant_client import QdrantClient

class ParquetRetriever:
    def __init__(self, ...):
        self.vector_db = QdrantClient(...)
```

## Technology Stack Rationale

**Streamlit**:
- Fast prototyping
- Built-in components
- Easy deployment
- Good for demos/internal tools

**DuckDB**:
- SQL on Parquet (zero-copy)
- In-process (no server)
- Fast analytical queries
- Arrow integration

**PyArrow**:
- Standard for Parquet
- Fast C++ backend
- Memory efficient
- Arrow ecosystem

**Sentence Transformers**:
- State-of-art embeddings
- Pre-trained models
- Easy to use
- HuggingFace ecosystem

**Ollama**:
- Local LLM execution
- Simple API
- Model management
- Privacy-preserving

## Future Enhancements

**Short-term**:
- [ ] Add FAISS for vector search
- [ ] Persist embeddings to disk
- [ ] Add conversation history
- [ ] Implement document chunking

**Medium-term**:
- [ ] FastAPI backend
- [ ] WebSocket for streaming responses
- [ ] Multi-query fusion
- [ ] Hybrid search (keyword + semantic)

**Long-term**:
- [ ] Multi-modal support (images in Parquet)
- [ ] Distributed retrieval
- [ ] Fine-tuning pipeline
- [ ] Production deployment (Docker, K8s)

---

**Questions?** See main [README.md](../README.md) or [SETUP.md](SETUP.md)

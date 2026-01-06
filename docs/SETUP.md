# Setup Guide

Complete setup instructions for the RAG system and Parquet Explorer.

## Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** package manager
- **Ollama** (for RAG system only) - https://ollama.ai
- **Mac** (tested on macOS, should work on Linux/Windows with minor adjustments)

## Installation Steps

### 1. Clone or Download Repository

```bash
cd ~/Projects
# (assuming you already have the code)
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

#### Option A: Parquet Explorer Only
```bash
pip install streamlit pyarrow duckdb pandas plotly numpy
```

#### Option B: Full RAG System
```bash
pip install -r requirements.txt
```

This will install all dependencies including:
- Core tools (streamlit, pyarrow, duckdb, pandas, plotly, numpy)
- RAG dependencies (sentence-transformers, ollama, torch, transformers)

### 4. Install Ollama (RAG System Only)

#### Mac
```bash
# Download from https://ollama.ai
# Or use Homebrew:
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Windows
Download installer from https://ollama.ai

### 5. Pull LLM Model

```bash
# Start Ollama service (if not auto-started)
ollama serve

# In another terminal, pull a model:
ollama pull llama3.2

# Other options:
ollama pull mistral
ollama pull llama2
ollama pull codellama
```

### 6. Verify Installation

#### Test Parquet Explorer
```bash
cd tools/parquet_explorer
streamlit run app.py
```

Should open browser to `http://localhost:8501`

#### Test RAG System
```bash
# From project root
streamlit run rag/app.py
```

Should open browser to `http://localhost:8502`

#### Test Ollama
```bash
ollama list  # Should show installed models
ollama run llama3.2 "Hello"  # Should generate a response
```

## Configuration

### Update Data Sources

Edit `rag/config.py`:

```python
# Add your Parquet files
PARQUET_FILES = [
    DATA_DIR / "your_file_1.parquet",
    DATA_DIR / "your_file_2.parquet",
]
```

Place Parquet files in `data/parquet_files/`

### Change Models

Edit `rag/config.py`:

```python
# Change embedding model (HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Change LLM model (Ollama)
LLM_MODEL = "mistral"  # Then: ollama pull mistral
```

### Adjust Performance Settings

Edit `rag/config.py`:

```python
# Retrieval
TOP_K_RESULTS = 3  # Fewer results = faster
SIMILARITY_THRESHOLD = 0.8  # Higher = stricter matching

# LLM
LLM_MAX_TOKENS = 256  # Fewer tokens = faster
LLM_TEMPERATURE = 0.5  # Lower = more focused
```

## Troubleshooting

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# If torch installation fails on Mac:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Ollama Connection Issues

```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama manually
ollama serve

# Check available models
ollama list
```

### Memory Issues

For machines with limited RAM:

```python
# In rag/config.py:
# Use smaller models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Smaller
LLM_MODEL = "llama3.2:1b"  # Smaller variant

# Load fewer documents
# In rag/app.py, modify:
retriever.load_documents(sample_size=1000)  # Limit to 1000 rows
```

### Port Already in Use

```bash
# Parquet Explorer (default: 8501)
streamlit run tools/parquet_explorer/app.py --server.port 8503

# RAG System (default: 8502)
streamlit run rag/app.py --server.port 8504
```

## Environment Variables

You can set these environment variables:

```bash
# Streamlit config
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true

# Ollama config
export OLLAMA_HOST="http://localhost:11434"

# HuggingFace cache (for embeddings)
export HF_HOME="~/.cache/huggingface"
```

## First Run

### Parquet Explorer
1. Launch app: `cd tools/parquet_explorer && streamlit run app.py`
2. In sidebar, select "Local Path"
3. Enter: `../../data/parquet_files/Spain_translated.parquet`
4. Click "Load File"
5. Explore tabs!

### RAG System
1. Ensure Ollama is running: `ollama serve`
2. Launch app: `streamlit run rag/app.py`
3. Wait for embeddings to build (first time only, ~1-2 minutes)
4. Enter a query: "What are the main themes?"
5. Click "🤖 Search & Answer"

## Next Steps

- Read the main [README.md](../README.md)
- Check out [Parquet Explorer Quick Start](../tools/parquet_explorer/QUICKSTART.md)
- Explore `rag/config.py` for customization options
- Try different Ollama models
- Experiment with embedding models

## Support

For issues:
1. Check this setup guide
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check that Ollama is running (for RAG)
5. Try with smaller data samples first

## Performance Benchmarks

Typical performance on M1 Mac:

- **Parquet Explorer**: Instant for files <500MB
- **RAG Indexing**: ~1-2 min for 100k documents
- **RAG Query**: ~2-5 seconds per query
- **Embedding Generation**: ~100-200 docs/second

Performance varies based on:
- File size
- Number of documents
- Model sizes
- Available RAM
- CPU/GPU

---

**Ready to use!** Proceed to main [README.md](../README.md) for usage instructions.

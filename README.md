# RAG System with Parquet Data Explorer

A comprehensive local RAG (Retrieval Augmented Generation) system that allows you to query Parquet files using a local language model, plus a powerful Parquet Explorer tool for data visualization.

## 🎯 Project Overview

This repository contains two main components:

1. **RAG Query System**: Ask questions about your Parquet data using semantic search and local LLMs
2. **Parquet Explorer**: Interactive web tool to visualize, filter, and analyze Parquet files

Both tools run **100% locally** on your Mac - no cloud, no accounts required.

## 📁 Project Structure

```
rag_example/
├── README.md                      # This file
├── requirements.txt               # All Python dependencies
├── .gitignore                     # Git ignore rules
│
├── data/                          # Data directory
│   └── parquet_files/            # Your Parquet files
│       ├── Spain_translated.parquet
│       └── France_translated.parquet
│
├── rag/                           # RAG application
│   ├── __init__.py
│   ├── app.py                    # Main RAG Streamlit app
│   ├── config.py                 # Configuration settings
│   ├── embeddings/               # Embedding generation
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── retrieval/                # Document retrieval
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── llm/                      # LLM integration
│   │   ├── __init__.py
│   │   └── model.py
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── helpers.py
│
├── tools/                         # Utility tools
│   └── parquet_explorer/         # Parquet Explorer
│       ├── app.py                # Main Explorer app
│       ├── run.sh                # Quick launch script
│       ├── README.md             # Explorer documentation
│       └── QUICKSTART.md         # Explorer quick start
│
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Tests
└── docs/                          # Additional documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core dependencies (Parquet Explorer only)
pip install streamlit pyarrow duckdb pandas plotly numpy

# Install full dependencies (RAG System + Explorer)
pip install -r requirements.txt
```

### 2A. Launch Parquet Explorer

```bash
cd tools/parquet_explorer
streamlit run app.py
```

Or use the convenience script:
```bash
cd tools/parquet_explorer
./run.sh
```

Then open your browser to `http://localhost:8501`

### 2B. Launch RAG Query System

First, install and start Ollama:
```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.2
```

Then launch the RAG app:
```bash
streamlit run rag/app.py
```

Open your browser to `http://localhost:8502`

## 📊 Tool 1: Parquet Explorer

### Features

- **File Loading**: Upload files, enter local paths, or merge multiple files
- **Overview**: File metadata, compression info, schema preview
- **Schema Inspector**: Detailed column statistics with search
- **Data Viewer**: Paginated table with sorting, filtering, export
- **Advanced Filters**: Numeric, text, date filtering with DuckDB speed
- **Visualizations**: Histograms, bar charts, correlation matrices
- **Export**: CSV and JSON export for filtered data

### Usage

```bash
cd tools/parquet_explorer
streamlit run app.py
```

See [`tools/parquet_explorer/QUICKSTART.md`](tools/parquet_explorer/QUICKSTART.md) for detailed usage.

## 🤖 Tool 2: RAG Query System

### Features

- **Semantic Search**: Find relevant documents using embeddings
- **Local LLM**: Uses Ollama for 100% local inference
- **Context-Aware Answers**: Retrieval augmented generation
- **Configurable**: Adjust top-k, similarity threshold, temperature
- **Interactive UI**: Streamlit-based web interface
- **Multiple Data Sources**: Query across multiple Parquet files

### How It Works

1. **Indexing**: Parquet files are loaded and text content is embedded using sentence transformers
2. **Retrieval**: User queries are embedded and matched against document embeddings
3. **Generation**: Top-k relevant documents are passed as context to local LLM
4. **Response**: LLM generates an answer based on retrieved context

### Usage

```bash
# Make sure Ollama is running with a model installed
ollama pull llama3.2

# Launch RAG app
streamlit run rag/app.py
```

### Example Queries

- "What are the main themes in the Spanish documents?"
- "Summarize the content from France"
- "Find documents mentioning [topic]"
- "Compare Spain and France data"

## ⚙️ Configuration

Edit `rag/config.py` to customize:

```python
# Parquet files to use
PARQUET_FILES = [
    DATA_DIR / "Spain_translated.parquet",
    DATA_DIR / "France_translated.parquet",
]

# Embedding model (from HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model (from Ollama)
LLM_MODEL = "llama3.2"  # or "mistral", "llama2", etc.

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# LLM settings
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
```

## 📦 Dependencies

### Core (Parquet Explorer)
- streamlit - Web UI
- pyarrow - Parquet reading
- duckdb - Fast SQL queries
- pandas - Data manipulation
- plotly - Visualizations
- numpy - Numerical operations

### RAG System
- sentence-transformers - Text embeddings
- ollama - Local LLM interface
- torch - PyTorch for embeddings
- transformers - HuggingFace transformers

### Optional
- faiss-cpu - Vector database for large datasets

## 🔧 Architecture

### RAG System Flow

```
User Query
    ↓
Embedding Generator (sentence-transformers)
    ↓
Similarity Search (cosine similarity)
    ↓
Parquet Retriever (DuckDB)
    ↓
Top-K Documents
    ↓
Context Building
    ↓
LLM Interface (Ollama)
    ↓
Generated Answer
```

### Key Components

- **EmbeddingGenerator**: Converts text to vector embeddings
- **ParquetRetriever**: Loads and searches Parquet files
- **LLMInterface**: Communicates with Ollama for text generation
- **Config**: Centralized configuration management

## 📚 Documentation

- **Parquet Explorer**: See [`tools/parquet_explorer/README.md`](tools/parquet_explorer/README.md)
- **Quick Start**: See [`tools/parquet_explorer/QUICKSTART.md`](tools/parquet_explorer/QUICKSTART.md)
- **RAG Configuration**: See [`rag/config.py`](rag/config.py)

## 🛠️ Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Data Sources

1. Place Parquet files in `data/parquet_files/`
2. Update `rag/config.py`:
   ```python
   PARQUET_FILES = [
       DATA_DIR / "your_file.parquet",
   ]
   ```
3. Restart the RAG app to reindex

### Customizing the LLM

Change the model in `rag/config.py`:
```python
LLM_MODEL = "mistral"  # or any Ollama model
```

Then pull the model:
```bash
ollama pull mistral
```

### Using Different Embedding Models

Update `rag/config.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better quality
# or
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # Faster
```

## 💡 Tips & Best Practices

### For Parquet Explorer
1. Use sampling for large files (>1GB)
2. Apply filters early to reduce data volume
3. Keep "Compute full statistics" OFF for initial exploration

### For RAG System
1. **First time setup**: Let embeddings build completely (may take a few minutes)
2. **Query tips**: Be specific in your questions
3. **Performance**: Reduce `TOP_K_RESULTS` if responses are slow
4. **Accuracy**: Increase `TOP_K_RESULTS` if answers lack context
5. **Creativity**: Adjust `LLM_TEMPERATURE` (lower=focused, higher=creative)

### Performance Optimization
- Use `sample_size` parameter when loading large Parquet files
- Consider using FAISS for very large datasets (>100k documents)
- Adjust chunk size in config for better retrieval granularity

## 🐛 Troubleshooting

### Parquet Explorer won't start
```bash
pip install -r requirements.txt --upgrade
```

### RAG System errors

**"Ollama not found"**
```bash
# Install Ollama from https://ollama.ai
# Verify it's running:
ollama list
```

**"Model not found"**
```bash
ollama pull llama3.2
```

**"Out of memory"**
- Reduce `sample_size` in `retriever.load_documents(sample_size=1000)`
- Use a smaller embedding model
- Use a smaller LLM model (e.g., `llama3.2:1b`)

**"Slow performance"**
- Reduce `TOP_K_RESULTS`
- Use GPU-enabled torch if available
- Consider using FAISS for vector search

## 🚦 Roadmap

- [ ] FAISS integration for large-scale retrieval
- [ ] Multi-query fusion for better retrieval
- [ ] Document chunking strategies
- [ ] Conversation history / chat mode
- [ ] Export Q&A sessions
- [ ] API endpoint (FastAPI)
- [ ] Docker containerization
- [ ] Support for other vector DBs (ChromaDB, Qdrant)

## 📄 License

This project is provided as-is for local data exploration and RAG applications.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [Apache Arrow](https://arrow.apache.org/)
- [DuckDB](https://duckdb.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)

---

**Questions or Issues?** Check the documentation in each tool's directory or open an issue.

**Happy Exploring & Querying!** 🚀

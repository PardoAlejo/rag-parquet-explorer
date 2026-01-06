# Quick Reference Guide

One-page reference for common operations.

## 🚀 Launch Commands

```bash
# Parquet Explorer
cd tools/parquet_explorer && streamlit run app.py

# RAG Query System
streamlit run rag/app.py

# Both with custom ports
streamlit run tools/parquet_explorer/app.py --server.port 8503
streamlit run rag/app.py --server.port 8504
```

## 📦 Installation

```bash
# Core dependencies only
pip install streamlit pyarrow duckdb pandas plotly numpy

# Full stack (RAG + Explorer)
pip install -r requirements.txt

# Ollama (for RAG)
# Mac: Download from https://ollama.ai
# Then: ollama pull llama3.2
```

## 🔧 Configuration Quick Edits

**File**: `rag/config.py`

```python
# Change data sources
PARQUET_FILES = [
    DATA_DIR / "your_file.parquet",
]

# Change models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"  # then: ollama pull llama3.2

# Tune retrieval
TOP_K_RESULTS = 5          # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score

# Tune LLM
LLM_TEMPERATURE = 0.7      # 0=focused, 1=creative
LLM_MAX_TOKENS = 512       # Response length
```

## 📂 File Locations

```
Add Parquet files here:
  data/parquet_files/*.parquet

Configuration:
  rag/config.py

Logs (auto-created):
  logs/rag.log

Cache (auto-created):
  .cache/
```

## 🛠️ Common Tasks

### Add New Parquet File

1. Copy file to `data/parquet_files/`
2. Edit `rag/config.py`:
   ```python
   PARQUET_FILES = [
       DATA_DIR / "new_file.parquet",
   ]
   ```
3. Restart RAG app

### Change LLM Model

```bash
# Pull new model
ollama pull mistral

# Edit rag/config.py
LLM_MODEL = "mistral"

# Restart app
```

### Change Embedding Model

Edit `rag/config.py`:
```python
# Smaller/faster
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Better/slower
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

Restart app to rebuild embeddings.

### Reduce Memory Usage

```python
# In rag/config.py
LLM_MODEL = "llama3.2:1b"  # Smaller variant
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# In rag/app.py, modify initialize_rag_system():
retriever.load_documents(sample_size=5000)  # Limit rows
```

## 🐛 Troubleshooting

```bash
# Python environment issues
pip install -r requirements.txt --upgrade

# Ollama not found
ollama serve  # Start Ollama
ollama list   # Check models
ollama pull llama3.2  # Install model

# Port in use
streamlit run app.py --server.port 8503

# Clear Streamlit cache
streamlit cache clear

# Check what's running
lsof -i :8501  # Parquet Explorer
lsof -i :8502  # RAG System
lsof -i :11434 # Ollama
```

## 📊 Performance Tips

### Parquet Explorer
- Use sampling for files >1GB
- Apply filters before viewing
- Keep "full stats" OFF initially

### RAG System
- First run: Wait for embedding build
- Reduce TOP_K for faster queries
- Increase TOP_K for better context
- Lower temperature for factual answers
- Higher temperature for creative answers

## 🔍 Example Queries (RAG)

```
General:
- "What are the main themes in the data?"
- "Summarize the content from Spain"

Specific:
- "Find documents mentioning [topic]"
- "Compare Spain and France data"
- "What information do we have about [entity]?"

Analytical:
- "What are the most common patterns?"
- "Identify key differences between sources"
```

## 📈 Typical Performance

**M1 Mac, 100k documents:**
- Initial indexing: 1-2 minutes
- Query (retrieval): 0.5-1 second
- Query (RAG full): 2-5 seconds
- Explorer load: <1 second

## 🔗 Useful Links

- Ollama: https://ollama.ai
- Sentence Transformers: https://www.sbert.net
- Streamlit Docs: https://docs.streamlit.io
- DuckDB: https://duckdb.org

## 📝 File Structure Reference

```
rag_example/
├── data/parquet_files/     # Put Parquet files here
├── rag/
│   ├── app.py             # RAG Streamlit app
│   └── config.py          # Edit configuration here
├── tools/parquet_explorer/
│   └── app.py             # Explorer Streamlit app
├── requirements.txt        # Dependencies
└── docs/                  # Documentation
    ├── SETUP.md
    └── ARCHITECTURE.md
```

## 💡 Best Practices

1. **Start small**: Test with sample data first
2. **Iterate**: Adjust config based on results
3. **Monitor**: Check logs/rag.log for issues
4. **Cache**: Let embeddings build once, reuse
5. **Experiment**: Try different models and settings

---

For detailed guides:
- Setup: [docs/SETUP.md](docs/SETUP.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Main docs: [README.md](README.md)

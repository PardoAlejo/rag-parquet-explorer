# 🚀 START HERE - Quick Start Guide

Welcome! This guide will get you up and running in 5 minutes.

## What You Have

This project contains:

1. **Parquet Explorer** - Interactive web tool to visualize Parquet files
2. **RAG Query System** - AI-powered Q&A over your Parquet data
3. **Comprehensive Tests** - 91 tests with ~90% coverage
4. **Full Documentation** - Architecture, setup, examples

## 🎯 Quick Start (Choose One)

### Option 1: Test Everything Works (30 seconds)

```bash
# Run the demo script
python demo.py
```

This verifies all components are working. No Ollama needed!

### Option 2: Explore Parquet Files (2 minutes)

```bash
# Navigate to explorer
cd tools/parquet_explorer

# Launch the app
streamlit run app.py

# Browser opens automatically
# Select "Local Path"
# Enter: ../../data/parquet_files/Spain_translated.parquet
# Click "Load File"
# Explore!
```

### Option 3: Query with AI (5 minutes)

```bash
# 1. Install Ollama (if not installed)
#    Download from: https://ollama.ai

# 2. Start Ollama (in a separate terminal)
ollama serve

# 3. Pull a model (first time only)
ollama pull llama3.2

# 4. Launch RAG system (from project root)
streamlit run rag/app.py

# 5. Wait for initialization (~1 minute first time)
# 6. Ask questions and get AI answers!
```

## 📁 Project Structure

```
rag_example/
├── 📄 START_HERE.md           ← You are here!
├── 📄 README.md               ← Full project overview
├── 📄 EXAMPLES.md             ← Detailed usage examples
├── 📄 QUICK_REFERENCE.md      ← Command reference
│
├── 🎯 demo.py                 ← Quick test script
├── 🧪 run_tests.sh            ← Test runner
│
├── 📂 data/
│   └── parquet_files/         ← Your Parquet files
│       ├── Spain_translated.parquet   (244 MB, 88K rows)
│       └── France_translated.parquet  (128 MB)
│
├── 📂 rag/                    ← RAG system
│   ├── app.py                 ← Main RAG app
│   ├── config.py              ← Configuration
│   ├── embeddings/            ← Embedding generation
│   ├── retrieval/             ← Document retrieval
│   ├── llm/                   ← LLM integration
│   └── utils/                 ← Utilities
│
├── 📂 tools/
│   └── parquet_explorer/      ← Parquet Explorer
│       └── app.py             ← Explorer app
│
├── 📂 tests/                  ← Test suite (91 tests)
└── 📂 docs/                   ← Documentation
    ├── SETUP.md
    ├── ARCHITECTURE.md
    ├── TESTING.md
    └── TEST_SUMMARY.md
```

## 🎮 What Can You Do?

### With Parquet Explorer:
- ✅ Visualize file metadata (size, rows, columns)
- ✅ Browse schema with statistics
- ✅ View and sample data
- ✅ Create charts and visualizations
- ✅ Filter data with advanced queries
- ✅ Export to CSV/JSON

### With RAG System:
- ✅ Ask questions about your data in natural language
- ✅ Get AI-powered answers based on retrieved documents
- ✅ Semantic search across all Parquet files
- ✅ Adjust relevance and temperature settings
- ✅ See which documents were used for answers

### With Tests:
- ✅ Run comprehensive test suite (91 tests)
- ✅ Generate coverage reports
- ✅ Test individual components
- ✅ Integration testing

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Complete project overview |
| [EXAMPLES.md](EXAMPLES.md) | Detailed usage examples |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Command cheat sheet |
| [docs/SETUP.md](docs/SETUP.md) | Installation guide |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture |
| [docs/TESTING.md](docs/TESTING.md) | Testing guide |

## 🔧 Common Commands

```bash
# Test the system
python demo.py

# Launch Parquet Explorer
cd tools/parquet_explorer && streamlit run app.py

# Launch RAG System
streamlit run rag/app.py

# Run tests
./run_tests.sh

# Run tests with coverage
./run_tests.sh coverage

# Run specific tests
./run_tests.sh embeddings
./run_tests.sh retrieval
```

## 💡 Example Workflows

### Workflow 1: Quick Data Exploration
```bash
1. cd tools/parquet_explorer
2. streamlit run app.py
3. Load: ../../data/parquet_files/Spain_translated.parquet
4. Explore tabs: Overview → Schema → Data → Stats
5. Done!
```

### Workflow 2: Ask Questions About Your Data
```bash
1. ollama serve  # In terminal 1
2. ollama pull llama3.2  # First time only
3. streamlit run rag/app.py  # In terminal 2
4. Enter query: "What are the main topics?"
5. Click "🤖 Search & Answer"
6. View results!
```

### Workflow 3: Filter and Export
```bash
1. Launch Parquet Explorer
2. Load your file
3. Sidebar → Add Filter → Configure
4. Go to "Filtered View" tab
5. Download as CSV
```

## ⚡ Performance Tips

**Parquet Explorer:**
- For large files (>1GB), use sampling
- Apply filters before viewing
- Keep "Compute full statistics" OFF initially

**RAG System:**
- First run takes ~1-2 minutes (builds embeddings)
- Subsequent runs are instant (uses cache)
- Adjust "Top K" for speed vs quality tradeoff
- Lower temperature (0.3-0.5) for factual answers
- Higher temperature (0.7-0.9) for creative answers

**Tests:**
- Use `./run_tests.sh fast` for quick validation
- Full suite runs in <1 second
- Coverage report: `./run_tests.sh coverage`

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### "Ollama not available"
```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama3.2
```

### "Port already in use"
```bash
# Use different port
streamlit run app.py --server.port 8503
```

### "File not found"
```bash
# Check file exists
ls -lh data/parquet_files/

# Use absolute path
/Users/pardoga/Projects/rag_example/data/parquet_files/Spain_translated.parquet
```

## 📞 Need Help?

1. **Check the docs:** See [README.md](README.md) and [EXAMPLES.md](EXAMPLES.md)
2. **Run the demo:** `python demo.py` to verify setup
3. **Check tests:** `./run_tests.sh` to ensure everything works
4. **Read architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 🎯 Next Steps

After you're comfortable with the basics:

1. **Customize the RAG System:**
   - Edit `rag/config.py` to add your files
   - Try different embedding models
   - Experiment with different LLMs

2. **Explore Advanced Features:**
   - Multi-file merging
   - Complex filtering
   - Statistical analysis

3. **Integrate Into Your Workflow:**
   - Export filtered data
   - Build custom queries
   - Create reports

4. **Contribute:**
   - Add new features
   - Improve tests
   - Enhance documentation

## 🎉 You're Ready!

Choose one of the quick start options above and start exploring!

**Most Popular First Steps:**

1. Run `python demo.py` to verify everything works
2. Launch Parquet Explorer: `cd tools/parquet_explorer && streamlit run app.py`
3. Explore your Parquet files visually
4. When ready for AI, set up Ollama and launch RAG system

**Have fun exploring your data!** 🚀

---

*For detailed information, see [README.md](README.md) | For examples, see [EXAMPLES.md](EXAMPLES.md)*

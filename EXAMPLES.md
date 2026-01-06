# Examples & Usage Guide

Practical examples showing how to use the Parquet Explorer and RAG Query System.

## 🎯 Quick Links

- [Parquet Explorer Example](#1-parquet-explorer-example)
- [RAG Query System Example](#2-rag-query-system-example)
- [Running Tests](#3-running-tests)
- [Common Workflows](#4-common-workflows)

---

## 1. Parquet Explorer Example

### Launch the Explorer

```bash
cd tools/parquet_explorer
streamlit run app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

### Using the Explorer

#### Step 1: Load a Parquet File

When the browser opens, you'll see the file selection sidebar:

**Option A: Upload a file**
- Click "Browse files" button
- Select your Parquet file

**Option B: Use local path**
```
1. Select "Local Path" from dropdown
2. Enter path: ../../data/parquet_files/Spain_translated.parquet
3. Click "Load File"
```

#### Step 2: Explore Your Data

**Overview Tab:**
- See file size: 243.92 MB
- Row count: 88,324 rows
- Column count: 14 columns
- Compression: SNAPPY
- Row groups: metadata info

**Schema Tab:**
- Click on "Schema" tab
- See all columns with:
  - Data types
  - Null percentages
  - Distinct counts
  - Min/max values
  - Top values

**Data Tab:**
- View actual data in table format
- Select page size (50/100/500/1000 rows)
- Sort by any column
- Choose sampling method (first/random)
- Export to CSV or JSON

**Stats Tab:**
- Select 1-4 columns to visualize
- See histograms (numeric columns)
- See bar charts (categorical columns)
- View correlation matrix

**Filtered View:**
- In sidebar, click "➕ Add Filter"
- Example filter:
  ```
  Column: score
  Operator: >
  Value: 7.5
  ```
- Go to "Filtered View" tab
- Download filtered data

### Example Session

```bash
# Terminal
cd tools/parquet_explorer
./run.sh

# Then in browser (http://localhost:8501):
# 1. Select "Local Path"
# 2. Enter: ../../data/parquet_files/Spain_translated.parquet
# 3. Click "Load File"
# 4. Explore tabs!
```

---

## 2. RAG Query System Example

### Prerequisites

**Step 1: Install Ollama**
```bash
# Mac (download from https://ollama.ai)
# Or via Homebrew:
brew install ollama

# Verify installation
ollama --version
```

**Step 2: Start Ollama and Pull Model**
```bash
# Start Ollama (in one terminal)
ollama serve

# In another terminal, pull a model
ollama pull llama3.2

# Verify model is available
ollama list
```

**Step 3: Install Python Dependencies**
```bash
# From project root
pip install -r requirements.txt
```

### Launch the RAG System

```bash
# From project root
streamlit run rag/app.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
  Network URL: http://192.168.1.x:8502
```

### Using the RAG System

#### First Time Setup (Automatic)

When you first launch, the system will:

```
Initializing RAG system...
✓ Loading documents from Parquet files...
  Loaded 88,324 documents from 2 files

✓ Generating embeddings...
  Progress: [████████████████████] 100%
  Generated embeddings for 88,324 documents

✓ RAG system initialized successfully!
```

**Note:** This takes 1-2 minutes the first time. Subsequent launches reuse cached embeddings.

#### Step 1: Simple Search

In the browser:

```
Query box:
┌─────────────────────────────────────────────┐
│ What are the main topics in the Spanish    │
│ documents?                                  │
└─────────────────────────────────────────────┘

Click: 🔍 Search
```

**Results:**
```
📄 Retrieved Documents

Document 1 - Spain_translated.parquet (Similarity: 0.87)
└─ Content: Information about tourism, culture, and history...

Document 2 - Spain_translated.parquet (Similarity: 0.82)
└─ Content: Economic data and regional statistics...

Document 3 - Spain_translated.parquet (Similarity: 0.79)
└─ Content: Cultural traditions and festivals...
```

#### Step 2: Get AI Answer

Click: **🤖 Search & Answer**

**Expected Response:**
```
🤖 AI-Generated Answer
┌─────────────────────────────────────────────────┐
│ Based on the retrieved documents, the main      │
│ topics in the Spanish documents include:        │
│                                                  │
│ 1. Tourism and cultural heritage                │
│ 2. Economic indicators and regional statistics  │
│ 3. Traditional festivals and customs            │
│                                                  │
│ The documents provide detailed information      │
│ about Spain's diverse regions and their         │
│ unique characteristics.                          │
└─────────────────────────────────────────────────┘
```

### Example Queries

#### Query 1: Specific Information
```
Query: "Find information about tourism in France"

Results:
- Top 5 relevant documents from France_translated.parquet
- AI answer summarizing tourism information
```

#### Query 2: Comparison
```
Query: "Compare Spain and France data"

Results:
- Documents from both files
- AI answer highlighting differences and similarities
```

#### Query 3: Detailed Question
```
Query: "What are the economic indicators mentioned in the data?"

Results:
- Relevant economic data passages
- AI summary of key economic metrics
```

### Adjusting Settings

In the sidebar, expand "Advanced Settings":

```
Top K Results: [5]     ← More = more context, slower
                       ← Fewer = faster, less context

Similarity Threshold: [0.7]  ← Higher = stricter matching
                             ← Lower = more results

LLM Temperature: [0.7]  ← Higher = creative
                        ← Lower = focused/factual
```

### Example Session

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model (first time only)
ollama pull llama3.2

# Terminal 3: Launch RAG app
cd /Users/pardoga/Projects/rag_example
streamlit run rag/app.py

# Wait for initialization (1-2 minutes first time)
# Then open http://localhost:8502

# In browser:
# 1. Enter query: "What is the main content about?"
# 2. Click "🤖 Search & Answer"
# 3. View retrieved documents and AI answer
```

---

## 3. Running Tests

### Quick Test Run

```bash
# From project root
./run_tests.sh
```

**Expected Output:**
```
🧪 RAG System Test Runner
==========================

Running all tests...

===== test session starts =====
collected 91 items

tests/test_embeddings.py .............. [ 15%]
tests/test_retrieval.py ............................ [ 47%]
tests/test_llm.py ...................... [ 72%]
tests/test_utils.py ................... [ 95%]
tests/test_integration.py .......... [100%]

===== 91 passed in 0.45s =====

✅ Tests completed successfully!
```

### Test with Coverage

```bash
./run_tests.sh coverage
```

**Expected Output:**
```
Running tests with coverage report...

===== test session starts =====
...
===== 91 passed in 0.52s =====

Name                          Stmts   Miss  Cover
-------------------------------------------------
rag/__init__.py                  1      0   100%
rag/embeddings/generator.py     45      2    96%
rag/retrieval/retriever.py     128     12    91%
rag/llm/model.py                67      8    88%
rag/utils/helpers.py            18      1    95%
-------------------------------------------------
TOTAL                          259     23    91%

📊 Coverage report generated in htmlcov/index.html
```

### Fast Tests Only

```bash
./run_tests.sh fast
```

This skips tests requiring:
- Ollama to be running
- Model downloads
- Slow operations (>5 seconds)

---

## 4. Common Workflows

### Workflow 1: Explore New Data

```bash
# 1. Add your Parquet file
cp /path/to/your/data.parquet data/parquet_files/

# 2. Launch Parquet Explorer
cd tools/parquet_explorer
streamlit run app.py

# 3. In browser:
#    - Load file
#    - Check Overview for metadata
#    - Browse Schema for column details
#    - Sample data in Data tab
#    - Create visualizations in Stats tab
```

### Workflow 2: Query Your Data with AI

```bash
# 1. Update config (optional)
# Edit rag/config.py:
PARQUET_FILES = [
    DATA_DIR / "your_file.parquet",
]

# 2. Start Ollama
ollama serve  # In separate terminal

# 3. Launch RAG system
streamlit run rag/app.py

# 4. Wait for initialization
# 5. Ask questions and get AI-powered answers!
```

### Workflow 3: Filter and Export Data

```bash
# Launch Explorer
cd tools/parquet_explorer
streamlit run app.py

# In browser:
# 1. Load your Parquet file
# 2. Add filters in sidebar:
#    - Column: age > 25
#    - Column: country = "Spain"
# 3. Go to "Filtered View" tab
# 4. Click "📥 Download Filtered Data as CSV"
# 5. Open CSV in Excel/Google Sheets
```

### Workflow 4: Compare Multiple Files

```bash
# In Parquet Explorer:
# 1. Select "Multiple Files (Merge)"
# 2. Enter paths:
#    - File 1: data/parquet_files/Spain_translated.parquet
#    - File 2: data/parquet_files/France_translated.parquet
# 3. Click "Load & Merge Files"
# 4. Explore combined data across all tabs

# In RAG System:
# Files are automatically merged
# Ask: "Compare Spain and France data"
```

### Workflow 5: Development Testing

```bash
# 1. Make code changes
# 2. Run tests
./run_tests.sh fast

# 3. If tests pass, run full suite
./run_tests.sh

# 4. Check coverage
./run_tests.sh coverage
open htmlcov/index.html

# 5. Commit changes
git add .
git commit -m "Add new feature"
```

---

## 5. Troubleshooting Examples

### Issue: Can't Find Parquet File

```bash
# Check file exists
ls -lh data/parquet_files/

# Use absolute path in Explorer
/Users/pardoga/Projects/rag_example/data/parquet_files/Spain_translated.parquet

# Or use relative path from project root
../../data/parquet_files/Spain_translated.parquet
```

### Issue: RAG System Won't Start

```bash
# 1. Check Ollama is running
ollama list

# If not running:
ollama serve

# 2. Check model is installed
ollama pull llama3.2

# 3. Check dependencies
pip install -r requirements.txt

# 4. Try launching again
streamlit run rag/app.py
```

### Issue: Tests Failing

```bash
# 1. Check you're in project root
pwd  # Should end with /rag_example

# 2. Install test dependencies
pip install pytest pytest-cov pytest-mock

# 3. Run specific failing test
pytest tests/test_embeddings.py -v

# 4. Check detailed error
pytest tests/test_embeddings.py -v --tb=short
```

### Issue: Port Already in Use

```bash
# Check what's using the port
lsof -i :8501  # Parquet Explorer
lsof -i :8502  # RAG System

# Use different port
streamlit run app.py --server.port 8503

# Or kill the process
kill <PID>
```

---

## 6. Complete Example Session

Here's a complete end-to-end example:

```bash
# ========================================
# TERMINAL SESSION
# ========================================

# Step 1: Setup (first time only)
cd /Users/pardoga/Projects/rag_example
pip install -r requirements.txt
ollama pull llama3.2

# Step 2: Explore data with Parquet Explorer
cd tools/parquet_explorer
streamlit run app.py
# Browser opens at http://localhost:8501
# Load: ../../data/parquet_files/Spain_translated.parquet
# Explore all tabs
# Ctrl+C to stop

# Step 3: Query data with RAG system
cd ../..
ollama serve &  # Start in background
streamlit run rag/app.py
# Browser opens at http://localhost:8502
# Wait for initialization
# Enter query: "What are the main topics?"
# Click "🤖 Search & Answer"
# View results

# Step 4: Run tests
./run_tests.sh coverage
# All tests pass!

# Step 5: Clean up
pkill ollama  # Stop Ollama
```

---

## 7. Video Walkthrough Script

If you want to record a demo, here's a script:

```
1. [0:00-0:30] Show project structure
   $ tree -L 2 -I '__pycache__|*.pyc'

2. [0:30-2:00] Launch Parquet Explorer
   $ cd tools/parquet_explorer
   $ streamlit run app.py
   [Show Overview, Schema, Data, Stats tabs]

3. [2:00-4:00] Launch RAG System
   $ streamlit run rag/app.py
   [Wait for initialization]
   [Enter query: "What is this data about?"]
   [Show results and AI answer]

4. [4:00-5:00] Run tests
   $ ./run_tests.sh
   [Show passing tests]

5. [5:00-5:30] Show documentation
   $ cat README.md | head -50
```

---

## 8. Next Steps

After running these examples:

1. **Customize the RAG System:**
   - Edit `rag/config.py`
   - Add your own Parquet files
   - Try different LLM models

2. **Explore Advanced Features:**
   - Multi-file merging
   - Complex filtering
   - Custom visualizations

3. **Integrate into Your Workflow:**
   - Export data for analysis
   - Use RAG for Q&A on your data
   - Build custom applications

4. **Learn More:**
   - Read [ARCHITECTURE.md](docs/ARCHITECTURE.md)
   - Study [TESTING.md](docs/TESTING.md)
   - Check [SETUP.md](docs/SETUP.md)

---

## 🎉 You're Ready!

You now have:
- ✅ Working Parquet Explorer
- ✅ Working RAG Query System
- ✅ Comprehensive test suite
- ✅ Complete documentation

**Start exploring your Parquet files!** 🚀

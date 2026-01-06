# Quick Start Guide

## 🚀 Launch in 3 Steps

### 1. Ensure dependencies are installed (first time only)
```bash
pip install -r requirements.txt
```

### 2. Launch the application
```bash
streamlit run app.py
```

Or use the convenience script:
```bash
./run.sh
```

### 3. Open your browser
The app will automatically open at: `http://localhost:8501`

## 📝 First Time Usage

When the app opens:

1. **In the sidebar**, you'll see "File Selection"
2. Choose a loading method:
   - **File Uploader**: Drag & drop a `.parquet` file
   - **Local Path**: Enter a file path (e.g., `parquet_files/Spain_translated.parquet`)
   - **Multiple Files (Merge)**: Load 2+ files to compare

3. For quick testing, try:
   ```
   parquet_files/Spain_translated.parquet
   ```

4. Click **"Load File"**

## 🎯 Exploring Your Data

Once loaded, use the tabs:

### 📋 Overview
- See file metadata instantly
- Row count, column count, file size
- Compression info, row groups
- Full schema preview

### 🔍 Schema
- Detailed column statistics
- Data types, null percentages
- Min/max values, distinct counts
- Top values for each column
- **Search** columns by name

### 📊 Data
- Interactive table viewer
- **Page size**: 50/100/500/1000 rows
- **Sampling**: First rows or random sample
- **Sort** by any column
- **Select** which columns to show
- **Export** to CSV or JSON

### 📈 Stats
- Select 1-4 columns to visualize
- **Histograms** for numeric data
- **Bar charts** for categorical data
- **Correlation matrix** for numeric columns
- Adjust sample size (100-50,000 rows)

### 🔎 Filtered View
- In sidebar: Click **"➕ Add Filter"**
- Build complex filters:
  - Numeric: `=`, `!=`, `<`, `>`, `between`
  - Text: `contains`, `startswith`, `endswith`
  - Null checks: `is null`, `is not null`
- View filtered results
- **Export** filtered data as CSV

## 💡 Pro Tips

### For Large Files (>1GB)
1. Use **sampling** in Data tab
2. Keep **"Compute full statistics"** unchecked in Schema tab
3. Apply **filters early** to reduce data volume
4. Lower **sample size** in Stats tab

### For Fast Exploration
1. Start with **Overview** to understand structure
2. Use **Schema search** to find columns quickly
3. Apply **filters** before viewing data

### For Data Export
1. Build filters in sidebar
2. Go to **Filtered View** tab
3. Adjust max rows slider
4. Click **"📥 Download Filtered Data as CSV"**

## 🧪 Example Workflows

### Workflow 1: Quick Data Preview
```
1. Load file → Data tab
2. Select page size: 100
3. Choose sampling: first
4. Browse data
```

### Workflow 2: Statistical Analysis
```
1. Load file → Stats tab
2. Select 2-3 numeric columns
3. Adjust sample size: 10,000
4. Review histograms & correlation
```

### Workflow 3: Filter & Export
```
1. Load file
2. Sidebar → Add Filter
   - Column: age
   - Operator: >
   - Value: 25
3. Add another filter
   - Column: country
   - Operator: =
   - Value: Spain
4. Go to Filtered View tab
5. Download CSV
```

## 🎨 UI Features

- **Clean, modern interface** with tabbed navigation
- **Progress indicators** for long operations
- **Responsive design** for different screen sizes
- **Interactive plots** (zoom, pan, download)
- **Data export** in multiple formats

## ⚙️ Configuration

The app auto-configures based on your data, but you can:

- Toggle **"Compute full statistics"** (Schema tab) for sample vs. full stats
- Adjust **sample sizes** (Stats tab) from 100 to 50,000 rows
- Set **page size** (Data tab) for table viewing
- Choose **sampling method** (Data tab): first rows or random

## 🛠️ Troubleshooting

### App won't start
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version (need 3.8+)
python --version
```

### File won't load
- Use **absolute paths**: `/Users/username/data/file.parquet`
- Or **relative paths** from project root: `parquet_files/file.parquet`
- Check file permissions

### Slow performance
- Enable **sampling** in Data tab
- Reduce **sample size** in Stats tab
- Keep **"Compute full statistics"** OFF for large files

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out all features in each tab
- Try filtering and exporting data
- Explore visualizations in the Stats tab

## 🎉 You're Ready!

Start exploring your Parquet files with powerful, local, interactive visualizations!

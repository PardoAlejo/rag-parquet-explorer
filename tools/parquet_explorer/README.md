# Parquet Explorer 📊

A powerful local tool to visualize and explore Apache Parquet files with an interactive web interface. Built with Streamlit, PyArrow, and DuckDB for efficient data exploration.

## Features

### 🎯 Core Capabilities

- **File Loading**
  - Load Parquet files via CLI argument, file uploader, or local path
  - Support for multiple files with optional merging
  - Efficient lazy loading using PyArrow and DuckDB (doesn't load full dataset into RAM)
  - Sampling options: first N rows, random sample, or stratified sampling

- **Overview Tab**
  - File metadata: name, size, row count, column count
  - Compression and encoding information
  - Row group details and format version
  - Complete schema preview

- **Schema Tab**
  - Detailed column information with statistics
  - Data types, null percentages, distinct counts
  - Min/max values for numeric/date columns
  - Top-K values for categorical columns
  - Column search functionality
  - Option to compute full stats or sample stats

- **Data Tab**
  - Paginated table viewer (50/100/500/1000 rows per page)
  - Column sorting (ascending/descending)
  - Column selection (show/hide)
  - Sampling methods (first rows or random)
  - Export to CSV and JSON

- **Advanced Filtering**
  - Multiple filter support with AND logic
  - Numeric filters: =, !=, <, <=, >, >=, between
  - Text filters: contains, startswith, endswith
  - Null checks: is null, is not null
  - Date/timestamp filtering
  - Compiled to DuckDB queries for high performance

- **Stats & Visualizations**
  - Interactive histograms for numeric columns
  - Bar charts for categorical data (top 20 values)
  - Value distribution analysis
  - Correlation matrix for numeric columns
  - Configurable sample sizes
  - Summary statistics (mean, median, std dev, min, max)

- **Performance Optimizations**
  - Lazy data loading with DuckDB
  - Efficient querying without loading full dataset
  - Progress indicators for long operations
  - Sample-based statistics for large files
  - In-memory DuckDB connection for fast filtering

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - streamlit (≥1.28.0) - Web UI framework
   - pyarrow (≥14.0.0) - Parquet file reading
   - duckdb (≥0.9.0) - Fast SQL queries on Parquet
   - pandas (≥2.0.0) - Data manipulation
   - plotly (≥5.17.0) - Interactive visualizations
   - numpy (≥1.24.0) - Numerical operations

## Usage

### Method 1: Run with file path as CLI argument

```bash
streamlit run app.py -- /path/to/your/file.parquet
```

Note: This is planned functionality. Currently, use Method 2 or 3.

### Method 2: Run and use file uploader

```bash
streamlit run app.py
```

Then use the sidebar to upload a Parquet file or enter a local path.

### Method 3: Quick start with example files

If you have Parquet files in a `parquet_files` directory:

```bash
streamlit run app.py
```

Then in the sidebar:
1. Select "Local Path" as loading method
2. Enter: `parquet_files/Spain_translated.parquet`
3. Click "Load File"

### Example Workflows

#### Exploring a Single File

1. Launch the app: `streamlit run app.py`
2. Use the sidebar file uploader or enter a local path
3. Navigate through tabs:
   - **Overview**: Get file metadata and schema
   - **Schema**: Deep dive into column statistics
   - **Data**: Browse and sort your data
   - **Stats**: Visualize distributions and correlations
   - **Filtered View**: Apply filters and export results

#### Comparing Multiple Files

1. Select "Multiple Files (Merge)" in the sidebar
2. Enter paths to 2-10 Parquet files
3. Click "Load & Merge Files"
4. Explore merged data across all tabs

#### Filtering and Exporting Data

1. Load your Parquet file
2. In the sidebar, click "➕ Add Filter"
3. Configure filters:
   - Select column
   - Choose operator (=, >, contains, etc.)
   - Enter value
4. Go to "Filtered View" tab to see results
5. Adjust the maximum rows slider
6. Click "📥 Download Filtered Data as CSV" to export

#### Statistical Analysis

1. Load your file
2. Go to "Stats" tab
3. Select 1-4 columns to visualize
4. Adjust sample size slider (100-50,000 rows)
5. View:
   - Histograms for numeric columns
   - Bar charts for categorical columns
   - Summary statistics tables
   - Correlation matrix (for 2+ numeric columns)

## Architecture

### Technology Stack

- **Streamlit**: Provides the interactive web interface
- **PyArrow**: Reads Parquet files efficiently with lazy loading
- **DuckDB**: Executes SQL queries directly on Parquet files without full data load
- **Pandas**: Data manipulation and export functionality
- **Plotly**: Interactive, publication-quality visualizations

### Key Design Decisions

1. **Lazy Loading**: Uses DuckDB's `parquet_scan()` to query files without loading everything into memory
2. **Sampling**: Offers first-N and random sampling to handle large files
3. **In-Memory Database**: Creates a DuckDB in-memory connection per session for fast repeated queries
4. **Progress Indicators**: Shows progress bars for long-running operations
5. **Modular Design**: Separate functions for each tab make the code maintainable

## File Structure

```
rag_example/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── parquet_files/        # Example Parquet files
    ├── Spain_translated.parquet
    └── France_translated.parquet
```

## Features Deep Dive

### Filter Panel

The filter panel supports complex filtering logic:

**Numeric Columns:**
- `=`, `!=`: Exact match or not match
- `<`, `<=`, `>`, `>=`: Comparison operators
- `between`: Range filtering (requires two values)

**Text Columns:**
- `=`, `!=`: Exact match
- `contains`: Substring search
- `startswith`: Prefix matching
- `endswith`: Suffix matching

**Date/Timestamp Columns:**
- Same operators as numeric columns
- `between`: Date range filtering

**All Columns:**
- `is null`: Find null/missing values
- `is not null`: Find non-null values

All filters are compiled into optimized DuckDB SQL queries for fast execution.

### Nested Types Support

- Parquet files with nested types (lists, structs) are supported
- Schema tab shows nested structure
- Data tab displays nested data (JSON stringified for preview)

### Error Handling

The app provides clear, actionable error messages for:
- Missing files
- Corrupted Parquet files
- Unsupported Parquet features
- Query errors
- Missing dependencies

## Performance Tips

1. **For Large Files (>1GB)**
   - Use sampling instead of viewing all rows
   - In Schema tab, keep "Compute full statistics" unchecked
   - Use filters to narrow down data before viewing
   - Lower the sample size in Stats tab (5,000-10,000 rows)

2. **For Fast Exploration**
   - Start with Overview tab to understand structure
   - Use Schema tab's search to find columns quickly
   - Apply filters early to reduce data volume

3. **For Complex Analysis**
   - Export filtered data to CSV for external tools
   - Use correlation matrix on sample data first
   - Select only needed columns in Data tab

## Troubleshooting

### "Package not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### "File not found" errors
- Use absolute paths: `/Users/username/data/file.parquet`
- Or relative from project root: `parquet_files/file.parquet`

### Slow performance on large files
- Enable sampling in Data tab
- Reduce sample size in Stats tab
- Keep "Compute full statistics" unchecked in Schema tab

### Memory errors
- Close other applications
- Use filtering to reduce data volume
- Sample smaller chunks of data

## Limitations

- Merging multiple files requires compatible schemas
- Very large files (>10GB) may require increased sampling
- Correlation matrix limited to numeric columns
- Regex filtering not yet implemented for text columns

## Future Enhancements

Potential features for future versions:
- CLI argument support for direct file loading
- Partition-aware browsing for partitioned datasets
- Custom SQL query interface
- Data profiling reports (PDF export)
- Column type inference and casting
- Regex support for text filtering
- GROUP BY aggregations interface

## License

This tool is provided as-is for local data exploration. Feel free to modify and extend for your needs.

## Contributing

Suggestions and improvements welcome! Key areas for contribution:
- Performance optimizations for very large files
- Additional visualization types
- Export formats (Excel, Parquet, etc.)
- Advanced filtering (regex, complex expressions)

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - The fastest way to build data apps
- [Apache Arrow](https://arrow.apache.org/) - Language-agnostic columnar data format
- [DuckDB](https://duckdb.org/) - In-process SQL OLAP database
- [Plotly](https://plotly.com/) - Interactive graphing library

---

**Happy Exploring!** 🚀

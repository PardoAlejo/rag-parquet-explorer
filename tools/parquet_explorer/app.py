"""
Parquet Explorer - A local tool to visualize and explore Apache Parquet files
"""

import streamlit as st
import pyarrow.parquet as pq
import duckdb
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List, Dict, Any, Optional


# Page configuration
st.set_page_config(
    page_title="Parquet Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


class ParquetExplorer:
    """Main class for exploring Parquet files"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.parquet_file = pq.ParquetFile(file_path)
        self.metadata = self.parquet_file.metadata
        self.schema = self.parquet_file.schema_arrow
        self.num_rows = self.metadata.num_rows
        self.num_columns = len(self.schema)

        # Initialize DuckDB connection for efficient querying
        self.conn = duckdb.connect(':memory:')

    def get_file_size(self) -> str:
        """Get human-readable file size"""
        size_bytes = Path(self.file_path).stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def get_compression_info(self) -> str:
        """Get compression codec information"""
        compressions = set()
        for i in range(self.metadata.num_row_groups):
            rg = self.metadata.row_group(i)
            for j in range(rg.num_columns):
                col = rg.column(j)
                compressions.add(col.compression)
        return ", ".join(compressions)

    def read_sample(self, n_rows: int = 1000, method: str = 'first') -> pd.DataFrame:
        """Read a sample of rows efficiently"""
        if method == 'first':
            # Read first n_rows
            table = self.parquet_file.read(columns=None, use_threads=True)
            return table.slice(0, min(n_rows, self.num_rows)).to_pandas()
        elif method == 'random':
            # Read all and sample (for small files) or read first N*10 and sample
            if self.num_rows <= n_rows * 10:
                table = self.parquet_file.read()
                df = table.to_pandas()
                if len(df) > n_rows:
                    return df.sample(n=n_rows, random_state=42)
                return df
            else:
                # Read a larger chunk and sample from it
                table = self.parquet_file.read(columns=None, use_threads=True)
                df = table.slice(0, min(n_rows * 10, self.num_rows)).to_pandas()
                return df.sample(n=min(n_rows, len(df)), random_state=42)
        return pd.DataFrame()

    def get_column_stats(self, column_name: str, sample_size: int = 10000) -> Dict[str, Any]:
        """Get statistics for a specific column"""
        stats = {
            'name': column_name,
            'dtype': str(self.schema.field(column_name).type),
            'null_count': 0,
            'null_percent': 0.0,
            'distinct_count': None,
            'min': None,
            'max': None,
            'top_values': []
        }

        try:
            # Use DuckDB for efficient stats computation
            query = f"""
            SELECT
                COUNT(*) as total_count,
                COUNT("{column_name}") as non_null_count,
                COUNT(DISTINCT "{column_name}") as distinct_count
            FROM parquet_scan('{self.file_path}')
            LIMIT {sample_size}
            """
            result = self.conn.execute(query).fetchone()

            if result:
                total_count, non_null_count, distinct_count = result
                stats['null_count'] = total_count - non_null_count
                stats['null_percent'] = (stats['null_count'] / total_count * 100) if total_count > 0 else 0
                stats['distinct_count'] = distinct_count

            # Get min/max for numeric and date columns
            col_type = str(self.schema.field(column_name).type)
            if any(t in col_type.lower() for t in ['int', 'float', 'double', 'decimal', 'date', 'timestamp']):
                try:
                    query = f"""
                    SELECT
                        MIN("{column_name}") as min_val,
                        MAX("{column_name}") as max_val
                    FROM parquet_scan('{self.file_path}')
                    """
                    result = self.conn.execute(query).fetchone()
                    if result:
                        stats['min'], stats['max'] = result
                except Exception:
                    pass

            # Get top values for categorical columns
            if 'string' in col_type.lower() or 'int' in col_type.lower():
                try:
                    query = f"""
                    SELECT "{column_name}", COUNT(*) as count
                    FROM parquet_scan('{self.file_path}')
                    WHERE "{column_name}" IS NOT NULL
                    GROUP BY "{column_name}"
                    ORDER BY count DESC
                    LIMIT 5
                    """
                    result = self.conn.execute(query).fetchall()
                    stats['top_values'] = [(str(val), count) for val, count in result]
                except Exception:
                    pass

        except Exception as e:
            st.warning(f"Error computing stats for {column_name}: {str(e)}")

        return stats

    def apply_filters(self, filters: List[Dict], limit: int = 1000) -> pd.DataFrame:
        """Apply filters and return filtered dataframe using DuckDB"""
        where_clauses = []

        for f in filters:
            col = f['column']
            op = f['operator']
            val = f['value']

            if op == '=':
                where_clauses.append(f'"{col}" = {self._format_value(val)}')
            elif op == '!=':
                where_clauses.append(f'"{col}" != {self._format_value(val)}')
            elif op == '<':
                where_clauses.append(f'"{col}" < {self._format_value(val)}')
            elif op == '<=':
                where_clauses.append(f'"{col}" <= {self._format_value(val)}')
            elif op == '>':
                where_clauses.append(f'"{col}" > {self._format_value(val)}')
            elif op == '>=':
                where_clauses.append(f'"{col}" >= {self._format_value(val)}')
            elif op == 'contains':
                where_clauses.append(f'"{col}" LIKE \'%{val}%\'')
            elif op == 'startswith':
                where_clauses.append(f'"{col}" LIKE \'{val}%\'')
            elif op == 'endswith':
                where_clauses.append(f'"{col}" LIKE \'%{val}\'')
            elif op == 'is null':
                where_clauses.append(f'"{col}" IS NULL')
            elif op == 'is not null':
                where_clauses.append(f'"{col}" IS NOT NULL')
            elif op == 'between' and len(f.get('value2', '')) > 0:
                where_clauses.append(f'"{col}" BETWEEN {self._format_value(val)} AND {self._format_value(f["value2"])}')

        where_clause = ' AND '.join(where_clauses) if where_clauses else '1=1'

        query = f"""
        SELECT * FROM parquet_scan('{self.file_path}')
        WHERE {where_clause}
        LIMIT {limit}
        """

        try:
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            st.error(f"Error applying filters: {str(e)}")
            return pd.DataFrame()

    def _format_value(self, val: Any) -> str:
        """Format value for SQL query"""
        if isinstance(val, str):
            return f"'{val}'"
        return str(val)

    def get_column_data_for_viz(self, column_name: str, sample_size: int = 10000) -> pd.Series:
        """Get column data for visualization"""
        query = f"""
        SELECT "{column_name}"
        FROM parquet_scan('{self.file_path}')
        WHERE "{column_name}" IS NOT NULL
        LIMIT {sample_size}
        """
        try:
            result = self.conn.execute(query).fetchdf()
            return result[column_name]
        except Exception as e:
            st.error(f"Error fetching data for {column_name}: {str(e)}")
            return pd.Series()


def load_parquet_file(file_path: str) -> Optional[ParquetExplorer]:
    """Load a parquet file and return explorer instance"""
    try:
        with st.spinner(f"Loading {Path(file_path).name}..."):
            explorer = ParquetExplorer(file_path)
        st.success(f"Successfully loaded {Path(file_path).name}")
        return explorer
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def render_overview_tab(explorer: ParquetExplorer):
    """Render the Overview tab"""
    st.markdown('<p class="sub-header">File Information</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("File Name", explorer.file_name)
    with col2:
        st.metric("File Size", explorer.get_file_size())
    with col3:
        st.metric("Total Rows", f"{explorer.num_rows:,}")
    with col4:
        st.metric("Total Columns", explorer.num_columns)

    st.markdown('<p class="sub-header">Metadata</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Compression", explorer.get_compression_info())
    with col2:
        st.metric("Row Groups", explorer.metadata.num_row_groups)
    with col3:
        st.metric("Format Version", explorer.metadata.format_version)

    # Schema preview
    st.markdown('<p class="sub-header">Schema Preview</p>', unsafe_allow_html=True)
    schema_data = []
    for i, field in enumerate(explorer.schema):
        schema_data.append({
            'Column': field.name,
            'Type': str(field.type),
            'Nullable': field.nullable
        })
    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, height=400)


def render_schema_tab(explorer: ParquetExplorer):
    """Render the Schema tab with detailed column information"""
    st.markdown('<p class="sub-header">Column Details</p>', unsafe_allow_html=True)

    # Search box
    search_term = st.text_input("🔍 Search columns by name", "")

    # Compute stats toggle
    compute_full = st.checkbox("Compute full statistics (may be slow for large files)", value=False)
    sample_size = 10000 if not compute_full else explorer.num_rows

    # Filter columns by search term
    columns_to_show = [field.name for field in explorer.schema
                       if search_term.lower() in field.name.lower()]

    if not columns_to_show:
        st.warning("No columns match your search.")
        return

    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    stats_data = []
    for idx, col_name in enumerate(columns_to_show):
        status_text.text(f"Computing statistics for {col_name}... ({idx+1}/{len(columns_to_show)})")
        progress_bar.progress((idx + 1) / len(columns_to_show))

        stats = explorer.get_column_stats(col_name, sample_size)

        top_vals_str = ""
        if stats['top_values']:
            top_vals_str = ", ".join([f"{val} ({cnt})" for val, cnt in stats['top_values'][:3]])

        stats_data.append({
            'Column': stats['name'],
            'Type': stats['dtype'],
            'Null %': f"{stats['null_percent']:.2f}%",
            'Distinct': stats['distinct_count'] if stats['distinct_count'] is not None else 'N/A',
            'Min': str(stats['min']) if stats['min'] is not None else 'N/A',
            'Max': str(stats['max']) if stats['max'] is not None else 'N/A',
            'Top Values': top_vals_str if top_vals_str else 'N/A'
        })

    progress_bar.empty()
    status_text.empty()

    # Display stats table
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, height=600)


def render_data_tab(explorer: ParquetExplorer):
    """Render the Data tab with table viewer"""
    st.markdown('<p class="sub-header">Data Preview</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        page_size = st.selectbox("Rows per page", [50, 100, 500, 1000], index=1)

    with col2:
        sample_method = st.selectbox("Sampling method", ['first', 'random'])

    with col3:
        selected_columns = st.multiselect(
            "Select columns to display (leave empty for all)",
            options=[field.name for field in explorer.schema],
            default=[]
        )

    # Load data
    df = explorer.read_sample(n_rows=page_size, method=sample_method)

    if selected_columns:
        df = df[selected_columns]

    # Sorting
    col1, col2 = st.columns(2)
    with col1:
        sort_column = st.selectbox("Sort by column", ['None'] + list(df.columns))
    with col2:
        sort_order = st.radio("Sort order", ['Ascending', 'Descending'], horizontal=True)

    if sort_column != 'None':
        df = df.sort_values(by=sort_column, ascending=(sort_order == 'Ascending'))

    # Display dataframe
    st.dataframe(df, use_container_width=True, height=500)

    # Export options
    st.markdown('<p class="sub-header">Export Data</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Export to CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{Path(explorer.file_name).stem}_export.csv",
            mime="text/csv"
        )

    with col2:
        # Export to JSON
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"{Path(explorer.file_name).stem}_export.json",
            mime="application/json"
        )


def render_filter_panel(explorer: ParquetExplorer) -> List[Dict]:
    """Render filter panel and return list of filters"""
    st.sidebar.markdown("### 🔍 Filters")

    filters = []

    # Initialize session state for filters
    if 'num_filters' not in st.session_state:
        st.session_state.num_filters = 0

    # Add filter button
    if st.sidebar.button("➕ Add Filter"):
        st.session_state.num_filters += 1

    # Clear all filters
    if st.session_state.num_filters > 0 and st.sidebar.button("🗑️ Clear All Filters"):
        st.session_state.num_filters = 0

    # Render each filter
    for i in range(st.session_state.num_filters):
        st.sidebar.markdown(f"**Filter {i+1}**")

        col = st.sidebar.selectbox(
            f"Column",
            options=[field.name for field in explorer.schema],
            key=f"filter_col_{i}"
        )

        # Determine operators based on column type
        col_type = str(explorer.schema.field(col).type)

        if 'string' in col_type.lower():
            operators = ['=', '!=', 'contains', 'startswith', 'endswith', 'is null', 'is not null']
        elif any(t in col_type.lower() for t in ['int', 'float', 'double', 'decimal']):
            operators = ['=', '!=', '<', '<=', '>', '>=', 'between', 'is null', 'is not null']
        elif any(t in col_type.lower() for t in ['date', 'timestamp']):
            operators = ['=', '!=', '<', '<=', '>', '>=', 'between', 'is null', 'is not null']
        else:
            operators = ['=', '!=', 'is null', 'is not null']

        op = st.sidebar.selectbox(
            "Operator",
            options=operators,
            key=f"filter_op_{i}"
        )

        filter_dict = {'column': col, 'operator': op}

        if op not in ['is null', 'is not null']:
            val = st.sidebar.text_input(
                "Value",
                key=f"filter_val_{i}"
            )
            filter_dict['value'] = val

            if op == 'between':
                val2 = st.sidebar.text_input(
                    "Value 2",
                    key=f"filter_val2_{i}"
                )
                filter_dict['value2'] = val2

        filters.append(filter_dict)
        st.sidebar.markdown("---")

    return filters


def render_stats_tab(explorer: ParquetExplorer):
    """Render the Stats tab with visualizations"""
    st.markdown('<p class="sub-header">Statistics & Visualizations</p>', unsafe_allow_html=True)

    # Column selection
    selected_cols = st.multiselect(
        "Select columns to visualize",
        options=[field.name for field in explorer.schema],
        max_selections=4
    )

    if not selected_cols:
        st.info("Please select one or more columns to visualize.")
        return

    # Sample size
    sample_size = st.slider("Sample size for visualizations", 100, 50000, 10000, step=100)

    # Create visualizations
    for col_name in selected_cols:
        st.markdown(f"#### {col_name}")

        col_type = str(explorer.schema.field(col_name).type)
        data = explorer.get_column_data_for_viz(col_name, sample_size)

        if len(data) == 0:
            st.warning(f"No data available for {col_name}")
            continue

        col1, col2 = st.columns(2)

        with col1:
            # Determine visualization type based on data type
            if any(t in col_type.lower() for t in ['int', 'float', 'double', 'decimal']):
                # Histogram for numeric data
                fig = px.histogram(
                    data,
                    x=data,
                    nbins=50,
                    title=f"Distribution of {col_name}",
                    labels={col_name: col_name, 'count': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
            elif 'string' in col_type.lower():
                # Bar chart for categorical data (top 20)
                value_counts = data.value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 20 Values in {col_name}",
                    labels={'x': col_name, 'y': 'Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Visualization not supported for type: {col_type}")

        with col2:
            # Statistics summary
            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Unique', 'Null Count', 'Null %'],
                'Value': [
                    len(data),
                    data.nunique(),
                    data.isna().sum(),
                    f"{data.isna().sum() / len(data) * 100:.2f}%"
                ]
            })

            if any(t in col_type.lower() for t in ['int', 'float', 'double', 'decimal']):
                numeric_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{data.mean():.2f}",
                        f"{data.median():.2f}",
                        f"{data.std():.2f}",
                        f"{data.min():.2f}",
                        f"{data.max():.2f}"
                    ]
                })
                stats_df = pd.concat([stats_df, numeric_stats], ignore_index=True)

            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("---")

    # Correlation matrix for numeric columns
    numeric_cols = [col for col in selected_cols
                    if any(t in str(explorer.schema.field(col).type).lower()
                          for t in ['int', 'float', 'double', 'decimal'])]

    if len(numeric_cols) >= 2:
        st.markdown("#### Correlation Matrix")

        # Fetch data for correlation
        query = f"""
        SELECT {', '.join([f'"{col}"' for col in numeric_cols])}
        FROM parquet_scan('{explorer.file_path}')
        LIMIT {sample_size}
        """
        df_corr = explorer.conn.execute(query).fetchdf()

        corr_matrix = df_corr.corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_filtered_data_view(explorer: ParquetExplorer, filters: List[Dict]):
    """Render filtered data view"""
    if not filters or all(f.get('value', '') == '' and f['operator'] not in ['is null', 'is not null'] for f in filters):
        st.info("Add filters from the sidebar to see filtered results.")
        return

    st.markdown('<p class="sub-header">Filtered Data</p>', unsafe_allow_html=True)

    # Display active filters
    st.markdown("**Active Filters:**")
    for f in filters:
        if f.get('value', '') or f['operator'] in ['is null', 'is not null']:
            if f['operator'] == 'between':
                st.text(f"• {f['column']} {f['operator']} {f.get('value', '')} AND {f.get('value2', '')}")
            elif f['operator'] in ['is null', 'is not null']:
                st.text(f"• {f['column']} {f['operator']}")
            else:
                st.text(f"• {f['column']} {f['operator']} {f.get('value', '')}")

    # Apply filters
    limit = st.slider("Maximum rows to display", 100, 10000, 1000, step=100)

    df_filtered = explorer.apply_filters(filters, limit=limit)

    if df_filtered.empty:
        st.warning("No rows match the specified filters.")
    else:
        st.success(f"Found {len(df_filtered)} rows (limited to {limit})")
        st.dataframe(df_filtered, use_container_width=True, height=500)

        # Export filtered data
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"{Path(explorer.file_name).stem}_filtered.csv",
            mime="text/csv"
        )


def main():
    """Main application"""

    # Header
    st.markdown('<p class="main-header">📊 Parquet Explorer</p>', unsafe_allow_html=True)
    st.markdown("A local tool to explore and visualize Apache Parquet files")

    # Sidebar - File selection
    st.sidebar.title("File Selection")

    # Method selection
    load_method = st.sidebar.radio(
        "Choose loading method",
        ["File Uploader", "Local Path", "Multiple Files (Merge)"]
    )

    explorer = None
    explorers = []

    if load_method == "File Uploader":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a Parquet file",
            type=['parquet'],
            accept_multiple_files=False
        )

        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            explorer = load_parquet_file(temp_path)

    elif load_method == "Local Path":
        file_path = st.sidebar.text_input(
            "Enter file path",
            placeholder="/path/to/your/file.parquet"
        )

        if file_path and st.sidebar.button("Load File"):
            if Path(file_path).exists():
                explorer = load_parquet_file(file_path)
            else:
                st.sidebar.error("File not found!")

    elif load_method == "Multiple Files (Merge)":
        num_files = st.sidebar.number_input("Number of files", min_value=2, max_value=10, value=2)
        file_paths = []

        for i in range(num_files):
            path = st.sidebar.text_input(f"File {i+1} path", key=f"file_{i}")
            if path:
                file_paths.append(path)

        if len(file_paths) >= 2 and st.sidebar.button("Load & Merge Files"):
            valid_paths = [p for p in file_paths if Path(p).exists()]

            if len(valid_paths) < 2:
                st.sidebar.error("At least 2 valid files are required for merging!")
            else:
                try:
                    # Load all files
                    for path in valid_paths:
                        exp = load_parquet_file(path)
                        if exp:
                            explorers.append(exp)

                    if len(explorers) >= 2:
                        st.info("Multiple files loaded. Merged view is shown below.")
                        # Use first explorer for schema reference
                        explorer = explorers[0]
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")

    # Main content
    if explorer:
        # Render filter panel
        filters = render_filter_panel(explorer)

        # Tabs
        tabs = st.tabs(["📋 Overview", "🔍 Schema", "📊 Data", "📈 Stats", "🔎 Filtered View"])

        with tabs[0]:
            render_overview_tab(explorer)

        with tabs[1]:
            render_schema_tab(explorer)

        with tabs[2]:
            render_data_tab(explorer)

        with tabs[3]:
            render_stats_tab(explorer)

        with tabs[4]:
            render_filtered_data_view(explorer, filters)

        # Show merged data info if multiple files
        if len(explorers) > 1:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Loaded Files")
            for exp in explorers:
                st.sidebar.text(f"• {exp.file_name}")
                st.sidebar.text(f"  Rows: {exp.num_rows:,}")

    else:
        # Welcome message
        st.info("""
        👈 **Get Started:**

        1. Choose a loading method from the sidebar
        2. Upload a file or provide a local path
        3. Explore your Parquet data with interactive visualizations!

        **Features:**
        - 📋 File overview with metadata
        - 🔍 Detailed schema inspection
        - 📊 Interactive data preview with sorting
        - 📈 Statistical visualizations
        - 🔎 Advanced filtering capabilities
        - 📥 Export filtered data
        """)

        # Show example
        st.markdown("### Quick Start Example")
        st.code("""
# Via CLI argument:
streamlit run app.py -- /path/to/your/file.parquet

# Or use the sidebar to upload or browse to a file
        """, language="bash")


if __name__ == "__main__":
    main()

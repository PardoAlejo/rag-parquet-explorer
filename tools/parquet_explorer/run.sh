#!/bin/bash
# Quick launch script for Parquet Explorer

echo "🚀 Starting Parquet Explorer..."
echo ""

# Check if dependencies are installed
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Launch Streamlit app
streamlit run app.py

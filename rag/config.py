"""
Configuration settings for the RAG application
"""

from pathlib import Path
from typing import List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "parquet_files"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Parquet files
PARQUET_FILES = [
    DATA_DIR / "Spain_translated.parquet",
    DATA_DIR / "France_translated.parquet",
]

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, fast
EMBEDDING_DIM = 384
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# LLM configuration
LLM_MODEL = "llama3.2"  # For use with Ollama
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
LLM_CONTEXT_WINDOW = 4096

# Retrieval configuration
RETRIEVAL_POOL_SIZE = 50  # Number of documents to retrieve initially (before ranking)
TOP_K_RESULTS = 5  # Number of top documents to show and use for LLM
SIMILARITY_THRESHOLD = 0.7

# Re-ranking configuration
USE_RERANKING = False  # Enable/disable re-ranking by default
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model for re-ranking

# Vector store configuration
VECTOR_DB_PATH = CACHE_DIR / "vector_db"
USE_FAISS = True  # Set to False to use in-memory search

# Application configuration
APP_TITLE = "RAG Query System"
APP_PORT = 8502  # Different from Parquet Explorer (8501)

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "rag.log"

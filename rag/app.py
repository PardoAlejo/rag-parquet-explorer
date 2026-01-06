"""
RAG Application - Main Streamlit interface for querying Parquet data with LLM
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import *
from rag.embeddings import EmbeddingGenerator
from rag.retrieval import ParquetRetriever
from rag.llm import LLMInterface
from rag.utils import setup_logging, ensure_directories


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark mode support
st.markdown("""
<style>
    /* Light theme (default) */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }

    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #ffd700;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }

    .context-box {
        background-color: rgba(31, 119, 180, 0.15);
        color: inherit;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        backdrop-filter: blur(10px);
    }

    .query-box {
        font-size: 1.1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(46, 204, 113, 0.1);
        border: 2px solid #2ecc71;
        margin: 1rem 0;
    }

    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #64b5f6;
        }

        .answer-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .context-box {
            background-color: rgba(100, 181, 246, 0.15);
            border-left-color: #64b5f6;
        }

        .query-box {
            background-color: rgba(46, 204, 113, 0.15);
            border-color: #4caf50;
        }
    }

    /* Streamlit dark theme specific */
    [data-testid="stAppViewContainer"][data-theme="dark"] .answer-box,
    .stApp[data-theme="dark"] .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
    }

    /* Enhanced readability */
    .answer-box p, .answer-box li, .answer-box span {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components (cached)"""
    with st.spinner("Initializing RAG system..."):
        # Ensure directories exist
        ensure_directories([MODELS_DIR, CACHE_DIR, LOG_FILE.parent])

        # Initialize components
        embedding_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
        retriever = ParquetRetriever(
            parquet_files=PARQUET_FILES,
            embedding_generator=embedding_gen,
            top_k=TOP_K_RESULTS
        )
        llm = LLMInterface(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        # Load documents
        retriever.load_documents()

        # Build embeddings
        retriever.build_embeddings()

        st.success("RAG system initialized successfully!")

        return retriever, llm


def main():
    """Main application"""

    # Header
    st.markdown('<p class="main-header">🤖 RAG Query System</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your Parquet data using local LLM with retrieval augmented generation")

    # Sidebar - Configuration
    st.sidebar.title("⚙️ Configuration")

    st.sidebar.markdown("### Data Sources")
    for pf in PARQUET_FILES:
        if pf.exists():
            st.sidebar.success(f"✓ {pf.name}")
        else:
            st.sidebar.error(f"✗ {pf.name}")

    st.sidebar.markdown("### Model Settings")
    st.sidebar.text(f"Embedding: {EMBEDDING_MODEL.split('/')[-1]}")
    st.sidebar.text(f"LLM: {LLM_MODEL}")

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        top_k = st.slider("Top K Results", 1, 10, TOP_K_RESULTS)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.05)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, LLM_TEMPERATURE, 0.1)

    # Initialize RAG system
    try:
        retriever, llm = initialize_rag_system()
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.info("Make sure all dependencies are installed:\n\n```\npip install sentence-transformers ollama\n```")
        return

    # Check LLM availability
    with st.sidebar.expander("Check LLM Status"):
        if st.button("Test Ollama Connection"):
            if llm.check_availability():
                st.success("✓ Ollama connected and model available")
            else:
                st.error("✗ Ollama not available or model not found")

    # Main content
    st.markdown("---")

    # Query input
    query = st.text_area(
        "Enter your query:",
        placeholder="e.g., What are the main themes in the Spanish documents?",
        height=100
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    with col2:
        search_and_answer = st.button("🤖 Search & Answer", type="primary", use_container_width=True)

    # Process query
    if (search_button or search_and_answer) and query.strip():
        # Update retriever settings
        retriever.top_k = top_k

        # Retrieve relevant documents
        with st.spinner("Searching for relevant documents..."):
            results = retriever.search(query, top_k=top_k)

        # Filter by similarity threshold
        results = [r for r in results if r['similarity_score'] >= similarity_threshold]

        if not results:
            st.warning("No relevant documents found. Try lowering the similarity threshold.")
            return

        # Display results
        st.markdown("### 📄 Retrieved Documents")

        for i, result in enumerate(results, 1):
            with st.expander(
                f"Document {i} - {result['source']} (Similarity: {result['similarity_score']:.3f})",
                expanded=(i == 1)
            ):
                st.markdown(f"**Content:**")
                st.markdown(f"```\n{result['text']}\n```")

                if result['metadata']:
                    st.markdown("**Metadata:**")
                    st.json(result['metadata'])

        # Generate answer if requested
        if search_and_answer:
            st.markdown("---")
            st.markdown("### 🤖 AI-Generated Answer")

            # Get context
            context = retriever.get_context_for_query(
                query,
                top_k=top_k,
                min_similarity=similarity_threshold
            )

            # Generate answer
            with st.spinner("Generating answer with LLM..."):
                try:
                    answer = llm.generate_with_context(query, context)

                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.info("""
                    Make sure Ollama is running and the model is installed:

                    1. Install Ollama: https://ollama.ai
                    2. Pull the model: `ollama pull llama3.2`
                    3. Verify it's running: `ollama list`
                    """)

    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Quick Start:**

        1. Enter your question in the text box above
        2. Click **"🔍 Search"** to find relevant documents only
        3. Click **"🤖 Search & Answer"** to get an AI-generated answer based on retrieved documents

        **Tips:**

        - Use the **Top K Results** slider to control how many documents are retrieved
        - Adjust **Similarity Threshold** to filter out low-relevance results
        - Modify **LLM Temperature** for more creative (higher) or focused (lower) responses

        **Requirements:**

        - Ollama must be installed and running: https://ollama.ai
        - Model must be pulled: `ollama pull llama3.2`
        """)

    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Statistics")
    st.sidebar.metric("Total Documents", len(retriever.documents))
    st.sidebar.metric("Embedding Dimension", EMBEDDING_DIM)


if __name__ == "__main__":
    setup_logging(LOG_LEVEL, LOG_FILE)
    main()

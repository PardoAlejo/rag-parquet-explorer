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
            top_k=TOP_K_RESULTS,
            use_reranking=False,  # Always False - we'll apply re-ranking dynamically
            reranker_model=RERANKER_MODEL
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
        pool_size = st.slider(
            "Retrieval Pool Size",
            10, 100,
            RETRIEVAL_POOL_SIZE,
            step=10,
            help="Number of documents to retrieve before ranking (larger = better re-ranking, slower)"
        )
        top_k = st.slider(
            "Top K for LLM",
            1, 20,
            TOP_K_RESULTS,
            help="Number of top-ranked documents to show and use for answer generation"
        )
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.05)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, LLM_TEMPERATURE, 0.1)

    # Initialize session state for results caching
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    # Re-ranking toggle (no expander, visible toggle)
    st.sidebar.markdown("### 🔄 Re-Ranking")
    use_reranking = st.sidebar.toggle(
        "Enable Re-ranking",
        value=USE_RERANKING,
        help="Toggle between original and re-ranked results (instant switch, no recomputation)",
        key="reranking_toggle"
    )

    if use_reranking:
        st.sidebar.info("✨ Showing re-ranked results (cross-encoder scores)")
    else:
        st.sidebar.info("📊 Showing original results (cosine similarity)")

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

    # Cache management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Cache Management")

    # Display cache info
    cache_info = retriever.cache.get_cache_info()
    if cache_info['exists']:
        st.sidebar.success(f"✓ Cache exists ({cache_info['size_mb']:.2f} MB)")
    else:
        st.sidebar.info("No cache found")

    # Manual recalculation button
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Rebuild Cache", help="Force recalculation of all embeddings"):
            with st.spinner("Rebuilding embeddings..."):
                retriever.build_embeddings(force_rebuild=True)
            st.success("✓ Embeddings rebuilt!")
            st.rerun()

    with col2:
        if st.button("🗑️ Clear Cache", help="Remove cached embeddings"):
            retriever.clear_cache()
            st.success("✓ Cache cleared!")
            st.rerun()

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

    # Check if we should show results (button clicked OR just toggling with existing results)
    cache_key = f"{query}|{pool_size}|{similarity_threshold}"
    has_cached_results = (
        st.session_state.current_query == cache_key and
        st.session_state.results_cache
    )

    # Auto-trigger re-search if pool_size or threshold changed (but same query text)
    previous_cache_key = st.session_state.current_query
    auto_rerun = False
    if previous_cache_key and previous_cache_key != cache_key and query.strip():
        # Extract previous query text (before first |)
        prev_query = previous_cache_key.split('|')[0] if '|' in previous_cache_key else previous_cache_key
        if prev_query == query:
            # Same query, but pool_size or threshold changed - auto re-run
            auto_rerun = True

    # Process query (button clicked OR cached results OR auto-rerun)
    if ((search_button or search_and_answer) and query.strip()) or has_cached_results or auto_rerun:
        # Update retriever settings
        retriever.top_k = top_k

        # Check if we need to recompute (query changed or no cache)
        query_changed = cache_key != st.session_state.current_query

        if query_changed:
            st.session_state.current_query = cache_key
            st.session_state.results_cache = {}  # Clear old results

            # Show info message if auto-rerun triggered
            if auto_rerun:
                st.info(f"🔄 Pool size or threshold changed - retrieving new pool of {pool_size} documents...")

            # Retrieve initial pool of documents (without re-ranking)
            with st.spinner(f"Retrieving pool of {pool_size} documents..."):
                pool_original = retriever.search(
                    query,
                    top_k=pool_size,
                    use_reranking=False,
                    min_similarity=similarity_threshold
                )

            if not pool_original:
                st.warning("No relevant documents found. Try lowering the similarity threshold.")
                return

            # Cache full original pool
            st.session_state.results_cache['original_pool'] = pool_original

            # Compute re-ranked pool for instant switching
            with st.spinner(f"Re-ranking {len(pool_original)} documents..."):
                pool_reranked = retriever.search(
                    query,
                    top_k=pool_size,
                    use_reranking=True,
                    min_similarity=similarity_threshold
                )

            # Cache full re-ranked pool
            st.session_state.results_cache['reranked_pool'] = pool_reranked

        # Normalize re-ranked scores to 0-1 range for easy comparison
        def normalize_scores(results_list):
            """Normalize similarity scores to 0-1 range using min-max normalization"""
            if not results_list:
                return results_list

            # Get all scores
            scores = [r['similarity_score'] for r in results_list]
            min_score = min(scores)
            max_score = max(scores)

            # Avoid division by zero
            if max_score == min_score:
                for r in results_list:
                    r['normalized_score'] = 1.0
                return results_list

            # Min-max normalization
            for r in results_list:
                normalized = (r['similarity_score'] - min_score) / (max_score - min_score)
                r['normalized_score'] = normalized

            return results_list

        # Get appropriate pool based on toggle, then slice to top_k
        if use_reranking:
            pool = st.session_state.results_cache.get('reranked_pool', [])
            # Normalize re-ranked scores for display
            pool = normalize_scores(pool)
            if not query_changed and has_cached_results:
                st.success("⚡ Switched to re-ranked results instantly (no recomputation needed!)")
        else:
            pool = st.session_state.results_cache.get('original_pool', [])
            if not query_changed and has_cached_results:
                st.success("⚡ Switched to original results instantly (no recomputation needed!)")

        # Slice pool to top_k for display and LLM use
        results = pool[:top_k] if pool else []

        if not results:
            st.warning("No relevant documents found. Try lowering the similarity threshold.")
            return

        # Display results
        reranked_badge = " 🔄" if use_reranking else ""
        st.markdown(f"### 📄 Top {len(results)} Documents{reranked_badge}")

        # Show pool info
        pool_info = f"Retrieved {len(pool)} documents from pool"
        if use_reranking:
            pool_info += " → Re-ranked with cross-encoder"
        pool_info += f" → Showing top {len(results)} for LLM"
        st.caption(pool_info)

        if use_reranking:
            st.info("✨ Scores normalized to 0-1 for easy comparison")

        for i, result in enumerate(results, 1):
            # Show normalized scores when re-ranking is ON, otherwise show cosine similarity
            if use_reranking and 'normalized_score' in result:
                score_info = f"Score: {result['normalized_score']:.3f}"
            else:
                score_info = f"Score: {result.get('similarity_score', 0):.3f}"

            # Try to get title from content or metadata
            doc_title = (
                result.get('content', {}).get('title') or
                result.get('content', {}).get('title_trans') or
                result.get('metadata', {}).get('title') or
                result.get('metadata', {}).get('title_trans') or
                result.get('source', 'Unknown')
            )

            with st.expander(
                f"Document {i}: {doc_title} ({score_info})",
                expanded=False
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

            # Build context from the top_k results we're displaying
            context_parts = []
            for i, result in enumerate(results, 1):
                score = result.get('normalized_score', result.get('similarity_score', 0))
                context_parts.append(f"[Document {i}] (Score: {score:.3f})")
                context_parts.append(f"Source: {result['source']}")
                context_parts.append(f"Content: {result['text']}")
                context_parts.append("")

            context = "\n".join(context_parts)

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

        **Cache Management:**

        - Embeddings are automatically cached to speed up subsequent launches
        - Cache is invalidated when files change (detected by file hash)
        - Use **🔄 Rebuild Cache** to force recalculation of embeddings
        - Use **🗑️ Clear Cache** to remove cached data

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

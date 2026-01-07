"""
Retrieval system for querying Parquet files
"""

from typing import List, Dict, Any, Optional
import duckdb
import numpy as np
from pathlib import Path
from .reranker import Reranker, NoOpReranker
from rag.utils import EmbeddingCache


class ParquetRetriever:
    """
    Retrieve relevant documents from Parquet files using vector similarity.

    This class handles loading data from Parquet files, generating embeddings,
    and performing similarity search to find relevant context for queries.
    """

    def __init__(
        self,
        parquet_files: List[Path],
        embedding_generator,
        top_k: int = 5,
        text_columns: Optional[List[str]] = None,
        use_reranking: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the retriever.

        Args:
            parquet_files: List of paths to Parquet files
            embedding_generator: EmbeddingGenerator instance
            top_k: Number of top results to return
            text_columns: Columns to use for text retrieval (if None, auto-detect)
            use_reranking: Whether to use re-ranking (default: False)
            reranker_model: Cross-encoder model to use for re-ranking
            cache_dir: Directory for caching embeddings (if None, use default)
        """
        self.parquet_files = [Path(f) for f in parquet_files]
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.text_columns = text_columns
        self.conn = duckdb.connect(':memory:')
        self.embeddings_cache = {}
        self.documents = []
        self.use_reranking = use_reranking

        # Initialize reranker (lazy loading)
        if use_reranking:
            self.reranker = Reranker(model_name=reranker_model)
        else:
            self.reranker = NoOpReranker()

        # Initialize cache system
        if cache_dir is None:
            from rag.config import CACHE_DIR
            cache_dir = CACHE_DIR
        self.cache = EmbeddingCache(cache_dir)

    def load_documents(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load documents from Parquet files.

        Args:
            sample_size: Optional limit on number of rows to load per file

        Returns:
            List of document dictionaries
        """
        documents = []

        for parquet_file in self.parquet_files:
            if not parquet_file.exists():
                print(f"Warning: File not found: {parquet_file}")
                continue

            # Build query
            query = f"SELECT * FROM parquet_scan('{parquet_file}')"
            if sample_size:
                query += f" LIMIT {sample_size}"

            # Load data
            df = self.conn.execute(query).fetchdf()

            # Auto-detect text columns if not specified
            if self.text_columns is None:
                # Use string columns
                text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
            else:
                text_cols = self.text_columns

            # Create documents
            for idx, row in df.iterrows():
                doc = {
                    'id': f"{parquet_file.stem}_{idx}",
                    'source': parquet_file.name,
                    'content': {},
                    'metadata': {}
                }

                # Add text content
                for col in text_cols:
                    if col in row and row[col] is not None:
                        doc['content'][col] = str(row[col])

                # Add other columns as metadata
                for col in df.columns:
                    if col not in text_cols:
                        doc['metadata'][col] = row[col]

                documents.append(doc)

        self.documents = documents
        print(f"Loaded {len(documents)} documents from {len(self.parquet_files)} files")
        return documents

    def build_embeddings(self, text_field: str = None, force_rebuild: bool = False) -> np.ndarray:
        """
        Build embeddings for all documents, using cache when possible.

        Args:
            text_field: Specific field to use for embeddings (if None, concatenate all text)
            force_rebuild: If True, ignore cache and rebuild embeddings

        Returns:
            numpy array of embeddings
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        # Check if we can use cached embeddings
        if not force_rebuild:
            cached_data = self._try_load_from_cache()
            if cached_data:
                return cached_data['embeddings']

        # Extract texts
        texts = []
        for doc in self.documents:
            if text_field and text_field in doc['content']:
                text = doc['content'][text_field]
            else:
                # Concatenate all content fields
                text = " | ".join(
                    f"{k}: {v}" for k, v in doc['content'].items()
                )
            texts.append(text)

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_generator.generate_embeddings(texts)

        self.embeddings_cache['texts'] = texts
        self.embeddings_cache['embeddings'] = embeddings

        # Save to persistent cache
        self._save_to_cache(embeddings, texts)

        return embeddings

    def _try_load_from_cache(self) -> Optional[Dict[str, Any]]:
        """
        Try to load embeddings from cache if files haven't changed.

        Returns:
            Cached data if valid, None otherwise
        """
        # Load cache
        cached_data = self.cache.load_cache()
        if not cached_data:
            print("No cache found, will generate embeddings...")
            return None

        # Get current file metadata
        current_metadata = {}
        for pf in self.parquet_files:
            if pf.exists():
                current_metadata[str(pf)] = self.cache.get_file_metadata(pf)

        # Check if any files have changed
        cached_file_metadata = cached_data.get('file_metadata', {})
        files_changed = []

        for file_path, current_meta in current_metadata.items():
            cached_meta = cached_file_metadata.get(file_path, {})
            if self.cache.has_file_changed(Path(file_path), cached_meta):
                files_changed.append(file_path)

        # Check for new or removed files
        cached_files = set(cached_file_metadata.keys())
        current_files = set(current_metadata.keys())
        new_files = current_files - cached_files
        removed_files = cached_files - current_files

        if files_changed or new_files or removed_files:
            if files_changed:
                print(f"Files changed: {[Path(f).name for f in files_changed]}")
            if new_files:
                print(f"New files: {[Path(f).name for f in new_files]}")
            if removed_files:
                print(f"Removed files: {[Path(f).name for f in removed_files]}")
            print("Cache invalidated, will regenerate embeddings...")
            return None

        # Cache is valid, use it
        print("✓ Loading embeddings from cache...")
        self.embeddings_cache['texts'] = cached_data['texts']
        self.embeddings_cache['embeddings'] = cached_data['embeddings']

        # Verify document count matches
        if len(cached_data['documents']) != len(self.documents):
            print("Warning: Document count mismatch, regenerating embeddings...")
            return None

        print(f"✓ Loaded {len(cached_data['texts'])} embeddings from cache")
        return cached_data

    def _save_to_cache(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Save embeddings to persistent cache.

        Args:
            embeddings: Document embeddings
            texts: Document texts
        """
        # Get file metadata
        file_metadata = {}
        for pf in self.parquet_files:
            if pf.exists():
                file_metadata[str(pf)] = self.cache.get_file_metadata(pf)

        # Save cache
        self.cache.save_cache(embeddings, texts, self.documents, file_metadata)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear_cache()
        self.embeddings_cache = {}
        print("✓ Cache cleared successfully")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return (uses self.top_k if None)
            use_reranking: Override the default reranking setting (None = use self.use_reranking)
            min_similarity: Minimum cosine similarity threshold (applied before re-ranking)

        Returns:
            List of relevant documents with similarity scores
        """
        if 'embeddings' not in self.embeddings_cache:
            raise ValueError("Embeddings not built. Call build_embeddings() first.")

        k = top_k or self.top_k
        apply_reranking = use_reranking if use_reranking is not None else self.use_reranking

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings(
            query,
            show_progress=False
        )

        # Compute similarities (cosine similarity)
        doc_embeddings = self.embeddings_cache['embeddings']
        similarities = self._cosine_similarity(query_embedding[0], doc_embeddings)

        # Filter by minimum similarity threshold BEFORE re-ranking
        # This is important because cross-encoder scores are on a different scale
        if min_similarity > 0:
            # Get indices where similarity meets threshold
            valid_indices = np.where(similarities >= min_similarity)[0]
            if len(valid_indices) == 0:
                return []  # No documents meet the threshold

            # Filter similarities and sort
            valid_similarities = similarities[valid_indices]
            sorted_positions = np.argsort(valid_similarities)[::-1]

            # Get top-k from valid documents
            retrieve_k = k * 2 if apply_reranking else k
            top_positions = sorted_positions[:min(retrieve_k, len(sorted_positions))]
            top_indices = valid_indices[top_positions]
        else:
            # No threshold, use original logic
            retrieve_k = k * 2 if apply_reranking else k
            top_indices = np.argsort(similarities)[::-1][:retrieve_k]

        # Build results with original cosine similarity scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(similarities[idx])
            doc['cosine_similarity'] = float(similarities[idx])  # Preserve original score
            doc['text'] = self.embeddings_cache['texts'][idx]
            results.append(doc)

        # Apply re-ranking if enabled
        # Re-ranker will update similarity_score but cosine_similarity is preserved
        if apply_reranking and len(results) > 0:
            results = self.reranker.rerank(query, results, top_k=k)

        return results

    def _cosine_similarity(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors"""
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)

        # Compute similarities
        similarities = np.dot(doc_norms, query_norm)

        return similarities

    def get_context_for_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.0
    ) -> str:
        """
        Get formatted context for a query to pass to LLM.

        Args:
            query: Search query
            top_k: Number of results to include
            min_similarity: Minimum similarity threshold

        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=top_k)

        # Filter by similarity threshold
        results = [r for r in results if r['similarity_score'] >= min_similarity]

        if not results:
            return "No relevant context found."

        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}] (Similarity: {result['similarity_score']:.3f})")
            context_parts.append(f"Source: {result['source']}")
            context_parts.append(f"Content: {result['text']}")
            context_parts.append("")

        return "\n".join(context_parts)

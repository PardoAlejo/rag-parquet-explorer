"""
Retrieval system for querying Parquet files
"""

from typing import List, Dict, Any, Optional
import duckdb
import numpy as np
from pathlib import Path
from .reranker import Reranker, NoOpReranker


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
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

    def build_embeddings(self, text_field: str = None) -> np.ndarray:
        """
        Build embeddings for all documents.

        Args:
            text_field: Specific field to use for embeddings (if None, concatenate all text)

        Returns:
            numpy array of embeddings
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

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

        return embeddings

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return (uses self.top_k if None)
            use_reranking: Override the default reranking setting (None = use self.use_reranking)

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

        # Get top-k indices (retrieve more if reranking to allow for better reranking)
        # Retrieve 2x more documents for reranking to have better candidates
        retrieve_k = k * 2 if apply_reranking else k
        top_indices = np.argsort(similarities)[::-1][:retrieve_k]

        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc['similarity_score'] = float(similarities[idx])
            doc['text'] = self.embeddings_cache['texts'][idx]
            results.append(doc)

        # Apply re-ranking if enabled
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

"""
Re-ranking system for improving document retrieval quality.

This module provides cross-encoder based re-ranking to improve the quality
of retrieved documents beyond what's possible with bi-encoder similarity alone.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class Reranker:
    """
    Re-rank retrieved documents using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders (used for initial retrieval)
    but slower, so they're applied only to the top-k results from initial retrieval.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model from HuggingFace
        """
        self.model_name = model_name
        self.model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization to avoid loading model until needed"""
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading cross-encoder model: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
                self._initialized = True
                print("Re-ranker model loaded successfully!")
            except Exception as e:
                print(f"Warning: Failed to load re-ranker model: {e}")
                self._initialized = False

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder scores.

        Args:
            query: The search query
            documents: List of document dictionaries with 'text' field
            top_k: Number of top documents to return (None = return all)

        Returns:
            Re-ranked list of documents with updated similarity scores
        """
        if not documents:
            return documents

        # Lazy load model
        self._lazy_init()

        if not self._initialized or self.model is None:
            # If model failed to load, return original ranking
            print("Re-ranker not available, returning original ranking")
            return documents

        # Prepare pairs for cross-encoder
        pairs = []
        for doc in documents:
            text = doc.get('text', '')
            pairs.append([query, text])

        # Get cross-encoder scores
        try:
            scores = self.model.predict(pairs)

            # Update documents with new scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy()
                # Store original score for reference
                doc_copy['original_score'] = doc_copy.get('similarity_score', 0.0)
                # Update with cross-encoder score
                doc_copy['similarity_score'] = float(score)
                doc_copy['reranked'] = True
                reranked_docs.append(doc_copy)

            # Sort by new scores (descending)
            reranked_docs.sort(key=lambda x: x['similarity_score'], reverse=True)

            # Return top-k if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]

            return reranked_docs

        except Exception as e:
            print(f"Error during re-ranking: {e}")
            return documents

    def is_available(self) -> bool:
        """Check if the re-ranker is available and initialized"""
        if not self._initialized:
            self._lazy_init()
        return self._initialized and self.model is not None


class NoOpReranker:
    """
    No-op reranker that just returns the original documents.
    Used when re-ranking is disabled.
    """

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return documents as-is"""
        if top_k is not None:
            return documents[:top_k]
        return documents

    def is_available(self) -> bool:
        """Always returns True since this is a pass-through"""
        return True

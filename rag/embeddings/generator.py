"""
Embedding generation using sentence transformers
"""

from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """
    Generate embeddings from text using sentence transformers.

    This class handles the creation of vector embeddings from text data
    stored in Parquet files for use in retrieval.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the embedding model (lazy loading)"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        self.load_model()
        return self.model.get_sentence_embedding_dimension()

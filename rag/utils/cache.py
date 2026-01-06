"""
Cache management for embeddings to avoid recomputation
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


class EmbeddingCache:
    """
    Manage persistent cache for document embeddings.

    This cache stores embeddings along with file metadata to detect changes
    and avoid recomputing embeddings for unchanged files.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.metadata_file = self.cache_dir / "cache_metadata.json"

    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Get metadata for a file including hash, size, and modification time.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return {}

        stat = file_path.stat()
        return {
            'path': str(file_path),
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'hash': self.compute_file_hash(file_path)
        }

    def has_file_changed(self, file_path: Path, cached_metadata: Dict[str, Any]) -> bool:
        """
        Check if a file has changed since it was cached.

        Args:
            file_path: Path to the file
            cached_metadata: Previously cached metadata

        Returns:
            True if file has changed, False otherwise
        """
        if not cached_metadata:
            return True

        current_metadata = self.get_file_metadata(file_path)

        # Quick check: size or mtime changed
        if (current_metadata.get('size') != cached_metadata.get('size') or
            current_metadata.get('mtime') != cached_metadata.get('mtime')):
            return True

        # Thorough check: hash changed
        return current_metadata.get('hash') != cached_metadata.get('hash')

    def load_cache(self) -> Optional[Dict[str, Any]]:
        """
        Load cached embeddings and metadata.

        Returns:
            Dictionary with cached data or None if cache doesn't exist
        """
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                cache_data['file_metadata'] = metadata

            return cache_data
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None

    def save_cache(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        documents: List[Dict[str, Any]],
        file_metadata: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Save embeddings and metadata to cache.

        Args:
            embeddings: Document embeddings array
            texts: List of text strings
            documents: List of document dictionaries
            file_metadata: Metadata for each source file
        """
        cache_data = {
            'embeddings': embeddings,
            'texts': texts,
            'documents': documents
        }

        try:
            # Save embeddings and documents with pickle
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            # Save file metadata as JSON for easy inspection
            with open(self.metadata_file, 'w') as f:
                json.dump(file_metadata, f, indent=2)

            print(f"✓ Cache saved successfully to {self.cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def clear_cache(self) -> None:
        """Remove all cached data."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"✓ Removed cache file: {self.cache_file}")

        if self.metadata_file.exists():
            self.metadata_file.unlink()
            print(f"✓ Removed metadata file: {self.metadata_file}")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.

        Returns:
            Dictionary with cache information
        """
        info = {
            'exists': self.cache_file.exists(),
            'cache_file': str(self.cache_file),
            'metadata_file': str(self.metadata_file)
        }

        if self.cache_file.exists():
            stat = self.cache_file.stat()
            info['size_bytes'] = stat.st_size
            info['size_mb'] = stat.st_size / (1024 * 1024)
            info['last_modified'] = stat.st_mtime

        return info

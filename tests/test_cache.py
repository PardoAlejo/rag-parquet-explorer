"""
Tests for the embedding cache system
"""

import pytest
import numpy as np
import json
import pickle
import time
from pathlib import Path
from rag.utils.cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for the EmbeddingCache class"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create an EmbeddingCache instance"""
        return EmbeddingCache(temp_cache_dir)

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings array"""
        return np.random.randn(10, 384).astype(np.float32)

    @pytest.fixture
    def sample_texts(self):
        """Sample text list"""
        return [f"Document {i}" for i in range(10)]

    @pytest.fixture
    def sample_documents(self):
        """Sample document list"""
        return [
            {'id': f'doc{i}', 'text': f'text{i}', 'source': 'test.parquet'}
            for i in range(10)
        ]

    @pytest.fixture
    def temp_parquet_file(self, tmp_path):
        """Create a temporary parquet file"""
        file_path = tmp_path / "test.parquet"
        file_path.write_text("test content")
        return file_path

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly"""
        cache = EmbeddingCache(temp_cache_dir)
        assert cache.cache_dir == temp_cache_dir
        assert cache.cache_file == temp_cache_dir / "embeddings_cache.pkl"
        assert cache.metadata_file == temp_cache_dir / "cache_metadata.json"

    def test_cache_dir_creation(self, tmp_path):
        """Test cache directory is created if it doesn't exist"""
        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()

        cache = EmbeddingCache(cache_dir)
        assert cache_dir.exists()

    def test_compute_file_hash(self, cache, temp_parquet_file):
        """Test file hash computation"""
        hash1 = cache.compute_file_hash(temp_parquet_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex string length

        # Same file should produce same hash
        hash2 = cache.compute_file_hash(temp_parquet_file)
        assert hash1 == hash2

    def test_compute_file_hash_changes(self, cache, temp_parquet_file):
        """Test hash changes when file content changes"""
        hash1 = cache.compute_file_hash(temp_parquet_file)

        # Modify file
        temp_parquet_file.write_text("different content")

        hash2 = cache.compute_file_hash(temp_parquet_file)
        assert hash1 != hash2

    def test_get_file_metadata(self, cache, temp_parquet_file):
        """Test getting file metadata"""
        metadata = cache.get_file_metadata(temp_parquet_file)

        assert 'path' in metadata
        assert 'size' in metadata
        assert 'mtime' in metadata
        assert 'hash' in metadata

        assert metadata['path'] == str(temp_parquet_file)
        assert metadata['size'] > 0
        assert metadata['mtime'] > 0
        assert len(metadata['hash']) == 64

    def test_get_file_metadata_nonexistent(self, cache, tmp_path):
        """Test getting metadata for non-existent file"""
        fake_file = tmp_path / "nonexistent.parquet"
        metadata = cache.get_file_metadata(fake_file)
        assert metadata == {}

    def test_has_file_changed_no_cached_metadata(self, cache, temp_parquet_file):
        """Test file change detection with no cached metadata"""
        assert cache.has_file_changed(temp_parquet_file, {})

    def test_has_file_changed_size_change(self, cache, temp_parquet_file):
        """Test file change detection when size changes"""
        metadata1 = cache.get_file_metadata(temp_parquet_file)

        # Change file size
        temp_parquet_file.write_text("much longer content now")

        assert cache.has_file_changed(temp_parquet_file, metadata1)

    def test_has_file_changed_mtime_change(self, cache, temp_parquet_file):
        """Test file change detection when modification time changes but content doesn't"""
        # Write specific content
        temp_parquet_file.write_text("test content")
        metadata1 = cache.get_file_metadata(temp_parquet_file)

        # Wait a bit and touch the file (changes mtime but not content)
        time.sleep(0.01)
        temp_parquet_file.touch()

        # Size and content are same, only mtime changed
        # Should NOT be detected as changed (hash will verify content is same)
        assert not cache.has_file_changed(temp_parquet_file, metadata1)

    def test_has_file_changed_content_change(self, cache, temp_parquet_file):
        """Test file change detection when content changes with same size"""
        # Write initial content
        temp_parquet_file.write_text("a" * 100)
        metadata1 = cache.get_file_metadata(temp_parquet_file)

        # Write different content with same size (edge case)
        temp_parquet_file.write_text("b" * 100)

        # Hash should detect the change even though size is same
        assert cache.has_file_changed(temp_parquet_file, metadata1)

    def test_has_file_changed_mtime_and_content_change(self, cache, temp_parquet_file):
        """Test file change detection when both mtime and content change"""
        temp_parquet_file.write_text("original content")
        metadata1 = cache.get_file_metadata(temp_parquet_file)

        # Wait and change content (mtime and hash both change)
        time.sleep(0.01)
        temp_parquet_file.write_text("modified content")

        # Should be detected as changed
        assert cache.has_file_changed(temp_parquet_file, metadata1)

    def test_has_file_not_changed(self, cache, temp_parquet_file):
        """Test that unchanged file is detected correctly"""
        metadata1 = cache.get_file_metadata(temp_parquet_file)
        assert not cache.has_file_changed(temp_parquet_file, metadata1)

    def test_save_and_load_cache(self, cache, sample_embeddings, sample_texts, sample_documents):
        """Test saving and loading cache"""
        file_metadata = {
            'test.parquet': {
                'path': 'test.parquet',
                'size': 1000,
                'mtime': time.time(),
                'hash': 'abc123'
            }
        }

        # Save cache
        cache.save_cache(sample_embeddings, sample_texts, sample_documents, file_metadata)

        # Check files were created
        assert cache.cache_file.exists()
        assert cache.metadata_file.exists()

        # Load cache
        loaded = cache.load_cache()

        assert loaded is not None
        assert 'embeddings' in loaded
        assert 'texts' in loaded
        assert 'documents' in loaded
        assert 'file_metadata' in loaded

        # Check embeddings match
        np.testing.assert_array_equal(loaded['embeddings'], sample_embeddings)

        # Check texts match
        assert loaded['texts'] == sample_texts

        # Check documents match
        assert loaded['documents'] == sample_documents

        # Check file metadata matches
        assert loaded['file_metadata'] == file_metadata

    def test_load_cache_nonexistent(self, cache):
        """Test loading cache when it doesn't exist"""
        loaded = cache.load_cache()
        assert loaded is None

    def test_load_cache_corrupt_file(self, cache):
        """Test loading cache with corrupted file"""
        # Create a corrupt cache file
        cache.cache_file.write_bytes(b"not a valid pickle file")

        loaded = cache.load_cache()
        assert loaded is None

    def test_clear_cache(self, cache, sample_embeddings, sample_texts, sample_documents):
        """Test clearing cache"""
        file_metadata = {'test.parquet': {'hash': 'abc'}}

        # Save cache
        cache.save_cache(sample_embeddings, sample_texts, sample_documents, file_metadata)
        assert cache.cache_file.exists()
        assert cache.metadata_file.exists()

        # Clear cache
        cache.clear_cache()
        assert not cache.cache_file.exists()
        assert not cache.metadata_file.exists()

    def test_clear_cache_when_empty(self, cache):
        """Test clearing cache when no cache exists"""
        # Should not raise an error
        cache.clear_cache()

    def test_get_cache_info_no_cache(self, cache):
        """Test getting cache info when no cache exists"""
        info = cache.get_cache_info()

        assert info['exists'] is False
        assert 'cache_file' in info
        assert 'metadata_file' in info
        assert 'size_bytes' not in info

    def test_get_cache_info_with_cache(self, cache, sample_embeddings, sample_texts, sample_documents):
        """Test getting cache info when cache exists"""
        file_metadata = {'test.parquet': {'hash': 'abc'}}
        cache.save_cache(sample_embeddings, sample_texts, sample_documents, file_metadata)

        info = cache.get_cache_info()

        assert info['exists'] is True
        assert 'cache_file' in info
        assert 'metadata_file' in info
        assert 'size_bytes' in info
        assert 'size_mb' in info
        assert 'last_modified' in info

        # Check size calculations
        assert info['size_bytes'] > 0
        assert info['size_mb'] == info['size_bytes'] / (1024 * 1024)

    def test_cache_metadata_is_json(self, cache, sample_embeddings, sample_texts, sample_documents):
        """Test that metadata file is valid JSON"""
        file_metadata = {
            'test.parquet': {
                'path': 'test.parquet',
                'size': 1000,
                'mtime': 12345.67,
                'hash': 'abc123'
            }
        }

        cache.save_cache(sample_embeddings, sample_texts, sample_documents, file_metadata)

        # Load and verify JSON
        with open(cache.metadata_file, 'r') as f:
            loaded_metadata = json.load(f)

        assert loaded_metadata == file_metadata

    def test_cache_embeddings_are_pickle(self, cache, sample_embeddings, sample_texts, sample_documents):
        """Test that embeddings file is valid pickle"""
        file_metadata = {'test.parquet': {'hash': 'abc'}}
        cache.save_cache(sample_embeddings, sample_texts, sample_documents, file_metadata)

        # Load and verify pickle
        with open(cache.cache_file, 'rb') as f:
            loaded_data = pickle.load(f)

        assert 'embeddings' in loaded_data
        assert 'texts' in loaded_data
        assert 'documents' in loaded_data
        np.testing.assert_array_equal(loaded_data['embeddings'], sample_embeddings)

    def test_large_file_hashing(self, cache, tmp_path):
        """Test hashing of large files (chunked reading)"""
        # Create a large file (1MB)
        large_file = tmp_path / "large.parquet"
        large_file.write_bytes(b"x" * (1024 * 1024))

        hash1 = cache.compute_file_hash(large_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Same file should produce same hash
        hash2 = cache.compute_file_hash(large_file)
        assert hash1 == hash2


class TestEmbeddingCacheIntegration:
    """Integration tests for embedding cache"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        return tmp_path / "cache"

    @pytest.fixture
    def temp_data_files(self, tmp_path):
        """Create temporary parquet files"""
        files = []
        for i in range(3):
            file_path = tmp_path / f"data{i}.parquet"
            file_path.write_text(f"Data file {i}")
            files.append(file_path)
        return files

    def test_cache_workflow(self, temp_cache_dir, temp_data_files):
        """Test complete cache workflow"""
        cache = EmbeddingCache(temp_cache_dir)

        # Create sample data
        embeddings = np.random.randn(100, 384).astype(np.float32)
        texts = [f"text {i}" for i in range(100)]
        documents = [{'id': f'doc{i}'} for i in range(100)]

        # Get file metadata
        file_metadata = {}
        for file in temp_data_files:
            file_metadata[str(file)] = cache.get_file_metadata(file)

        # Save cache
        cache.save_cache(embeddings, texts, documents, file_metadata)

        # Load cache
        loaded = cache.load_cache()
        assert loaded is not None

        # Verify no files have changed
        for file_path, cached_meta in loaded['file_metadata'].items():
            assert not cache.has_file_changed(Path(file_path), cached_meta)

        # Modify one file
        temp_data_files[0].write_text("Modified content")

        # Check that file change is detected
        file_path = str(temp_data_files[0])
        cached_meta = loaded['file_metadata'][file_path]
        assert cache.has_file_changed(temp_data_files[0], cached_meta)

    def test_cache_invalidation_on_new_file(self, temp_cache_dir, temp_data_files, tmp_path):
        """Test cache invalidation when new file is added"""
        cache = EmbeddingCache(temp_cache_dir)

        # Create and save cache with 3 files
        embeddings = np.random.randn(10, 384).astype(np.float32)
        texts = ["text"] * 10
        documents = [{'id': 'doc'}] * 10

        file_metadata = {}
        for file in temp_data_files:
            file_metadata[str(file)] = cache.get_file_metadata(file)

        cache.save_cache(embeddings, texts, documents, file_metadata)

        # Add a new file
        new_file = tmp_path / "data3.parquet"
        new_file.write_text("New file")

        # Load cache
        loaded = cache.load_cache()

        # Check that new file is detected
        cached_files = set(loaded['file_metadata'].keys())
        current_files = {str(f) for f in temp_data_files + [new_file]}
        assert current_files != cached_files

    def test_multiple_cache_instances(self, temp_cache_dir, tmp_path):
        """Test multiple cache instances share the same storage"""
        # Create two cache instances for same directory
        cache1 = EmbeddingCache(temp_cache_dir)
        cache2 = EmbeddingCache(temp_cache_dir)

        # Save with cache1
        embeddings = np.random.randn(5, 384).astype(np.float32)
        texts = ["text"] * 5
        documents = [{'id': 'doc'}] * 5
        file_metadata = {}

        cache1.save_cache(embeddings, texts, documents, file_metadata)

        # Load with cache2
        loaded = cache2.load_cache()
        assert loaded is not None
        np.testing.assert_array_equal(loaded['embeddings'], embeddings)

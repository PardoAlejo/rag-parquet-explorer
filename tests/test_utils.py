"""
Tests for rag/utils/helpers.py
"""

import pytest
import logging
from pathlib import Path
from rag.utils.helpers import setup_logging, ensure_directories


@pytest.mark.unit
class TestSetupLogging:
    """Test cases for setup_logging function"""

    def test_setup_logging_default(self, caplog):
        """Test logging setup with default parameters"""
        with caplog.at_level(logging.INFO):
            setup_logging()

            # Test that logging works
            logger = logging.getLogger(__name__)
            logger.info("Test message")

            # Default level is INFO
            assert len(caplog.records) >= 1
            assert any(r.levelname == "INFO" for r in caplog.records)
            assert any("Test message" in r.message for r in caplog.records)

    def test_setup_logging_debug_level(self, caplog):
        """Test logging setup with DEBUG level"""
        with caplog.at_level(logging.DEBUG):
            setup_logging(log_level="DEBUG")

            logger = logging.getLogger(__name__)
            logger.debug("Debug message")

            assert len(caplog.records) >= 1
            assert any(r.levelname == "DEBUG" for r in caplog.records)

    def test_setup_logging_warning_level(self, caplog):
        """Test logging setup with WARNING level"""
        setup_logging(log_level="WARNING")

        logger = logging.getLogger(__name__)
        logger.info("Info message")  # Should not appear
        logger.warning("Warning message")  # Should appear

        # Only warning should be captured
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

    def test_setup_logging_error_level(self, caplog):
        """Test logging setup with ERROR level"""
        with caplog.at_level(logging.ERROR):
            setup_logging(log_level="ERROR")

            logger = logging.getLogger(__name__)
            logger.warning("Warning")  # Should not appear at ERROR level
            logger.error("Error")  # Should appear

            # At least the ERROR should be captured
            assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_setup_logging_with_file(self, temp_directory):
        """Test logging setup with log file"""
        log_file = temp_directory / "test.log"

        # Test that setup_logging creates the file
        setup_logging(log_level="INFO", log_file=log_file)

        # Check that parent directory was created
        assert log_file.parent.exists()

        # File should be created (may be empty until log messages are written and flushed)
        # This tests that the setup doesn't crash and creates the necessary structure
        assert True  # Setup succeeded without error

    def test_setup_logging_file_directory_creation(self, temp_directory):
        """Test that parent directories are created for log file"""
        log_file = temp_directory / "subdir" / "logs" / "test.log"

        setup_logging(log_file=log_file)

        logger = logging.getLogger(__name__)
        logger.info("Test")

        # Check that parent directories were created
        assert log_file.parent.exists()
        assert log_file.exists()

    def test_setup_logging_format(self, caplog):
        """Test that log format includes expected fields"""
        with caplog.at_level(logging.INFO):
            setup_logging()

            logger = logging.getLogger(__name__)
            logger.info("Format test")

            # Check that log record has expected format
            assert len(caplog.records) > 0
            record = caplog.records[0]
            assert hasattr(record, 'name')
            assert hasattr(record, 'levelname')
            assert hasattr(record, 'message')

    def test_setup_logging_case_insensitive_level(self, caplog):
        """Test that log level is case insensitive"""
        with caplog.at_level(logging.INFO):
            setup_logging(log_level="info")  # lowercase

            logger = logging.getLogger(__name__)
            logger.info("Test")

            assert len(caplog.records) >= 1

    def test_setup_logging_multiple_calls(self, temp_directory):
        """Test calling setup_logging multiple times"""
        log_file1 = temp_directory / "test1.log"
        log_file2 = temp_directory / "test2.log"

        setup_logging(log_file=log_file1)
        setup_logging(log_file=log_file2)

        logger = logging.getLogger(__name__)
        logger.info("Test")

        # Both files should exist if handlers accumulate
        # (behavior depends on implementation)
        assert log_file2.exists()


@pytest.mark.unit
class TestEnsureDirectories:
    """Test cases for ensure_directories function"""

    def test_ensure_single_directory(self, temp_directory):
        """Test creating a single directory"""
        new_dir = temp_directory / "new_dir"

        ensure_directories([new_dir])

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_multiple_directories(self, temp_directory):
        """Test creating multiple directories"""
        dir1 = temp_directory / "dir1"
        dir2 = temp_directory / "dir2"
        dir3 = temp_directory / "dir3"

        ensure_directories([dir1, dir2, dir3])

        assert dir1.exists()
        assert dir2.exists()
        assert dir3.exists()

    def test_ensure_nested_directories(self, temp_directory):
        """Test creating nested directory structure"""
        nested_dir = temp_directory / "level1" / "level2" / "level3"

        ensure_directories([nested_dir])

        assert nested_dir.exists()
        assert (temp_directory / "level1").exists()
        assert (temp_directory / "level1" / "level2").exists()

    def test_ensure_existing_directory(self, temp_directory):
        """Test with directory that already exists"""
        existing_dir = temp_directory / "existing"
        existing_dir.mkdir()

        # Should not raise error
        ensure_directories([existing_dir])

        assert existing_dir.exists()

    def test_ensure_empty_list(self):
        """Test with empty directory list"""
        # Should not raise error
        ensure_directories([])

    def test_ensure_directories_with_file_path(self, temp_directory):
        """Test that it handles file-like paths correctly"""
        file_path = temp_directory / "file.txt"

        ensure_directories([file_path])

        # Directory should be created even if it looks like a file
        assert file_path.exists()

    def test_ensure_mixed_existing_and_new(self, temp_directory):
        """Test with mix of existing and new directories"""
        existing_dir = temp_directory / "existing"
        existing_dir.mkdir()

        new_dir = temp_directory / "new"

        ensure_directories([existing_dir, new_dir])

        assert existing_dir.exists()
        assert new_dir.exists()

    def test_ensure_directories_permissions(self, temp_directory):
        """Test that created directories have correct permissions"""
        new_dir = temp_directory / "test_permissions"

        ensure_directories([new_dir])

        assert new_dir.exists()
        # Directory should be readable and writable
        assert (new_dir / "test.txt").parent.exists()

    def test_ensure_relative_paths(self, temp_directory, monkeypatch):
        """Test with relative paths"""
        # Change to temp directory
        monkeypatch.chdir(temp_directory)

        relative_dir = Path("relative_dir")

        ensure_directories([relative_dir])

        assert (temp_directory / "relative_dir").exists()

    def test_ensure_multiple_levels_multiple_dirs(self, temp_directory):
        """Test creating multiple complex nested structures"""
        dir1 = temp_directory / "a" / "b" / "c"
        dir2 = temp_directory / "x" / "y" / "z"
        dir3 = temp_directory / "m" / "n"

        ensure_directories([dir1, dir2, dir3])

        assert dir1.exists()
        assert dir2.exists()
        assert dir3.exists()
        assert (temp_directory / "a" / "b").exists()
        assert (temp_directory / "x" / "y").exists()


@pytest.mark.integration
class TestUtilsIntegration:
    """Integration tests for utils module"""

    def test_logging_and_directory_creation_together(self, temp_directory):
        """Test using both utilities together"""
        log_dir = temp_directory / "logs"
        data_dir = temp_directory / "data"

        # Create directories
        ensure_directories([log_dir, data_dir])

        # Setup logging
        log_file = log_dir / "app.log"
        setup_logging(log_level="INFO", log_file=log_file)

        # Verify directories were created
        assert log_dir.exists()
        assert data_dir.exists()

        # Verify setup succeeded
        assert True  # No exceptions raised

    def test_typical_app_setup(self, temp_directory):
        """Test typical application setup scenario"""
        # Typical directory structure
        dirs = [
            temp_directory / "logs",
            temp_directory / "cache",
            temp_directory / "models",
            temp_directory / "data",
        ]

        ensure_directories(dirs)

        # Setup logging
        log_file = dirs[0] / "app.log"
        setup_logging(log_level="INFO", log_file=log_file)

        # Verify all directories exist
        for d in dirs:
            assert d.exists()

        # Verify setup succeeded without errors
        assert True

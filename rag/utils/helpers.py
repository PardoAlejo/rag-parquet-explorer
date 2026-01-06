"""
Helper utility functions
"""

import logging
from pathlib import Path
from typing import List


def setup_logging(log_level: str = "INFO", log_file: Path = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def ensure_directories(directories: List[Path]):
    """
    Ensure all specified directories exist.

    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

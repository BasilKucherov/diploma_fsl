"""Logging utilities for training."""

import logging
import sys
from pathlib import Path


def setup_logging(log_dir=None, log_file="training.log", level=logging.INFO):
    """Setup logging to file and console.

    Args:
        log_dir: Directory to save log file (optional, if None only console logging)
        log_file: Name of log file (default: 'training.log')
        level: Logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

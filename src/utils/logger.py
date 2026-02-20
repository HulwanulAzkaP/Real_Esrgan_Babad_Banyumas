"""
Logger utility for the Real-ESRGAN manuscript restoration project.
Provides a consistent logging interface writing to both console and file.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger writing to stdout and results/training.log.

    Args:
        name: Name for the logger, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    log_dir = Path("results")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
"""
Logging configuration module
"""
import logging
import logging.handlers
import os
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "foam_orchestrator.log"


def setup_logger():
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

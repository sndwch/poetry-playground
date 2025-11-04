"""Logging configuration for generativepoetry."""

import logging
import sys


def setup_logger(name="generativepoetry", level=logging.INFO):
    """Set up a logger with console output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.setLevel(level)

    return logger


# Create a default logger instance
logger = setup_logger()


def set_log_level(quiet: bool = False, verbose: bool = False) -> None:
    """Set logging level based on quiet/verbose flags.

    Args:
        quiet: If True, set to WARNING (suppress INFO)
        verbose: If True, set to DEBUG (show all details)

    Note: quiet takes precedence over verbose
    """
    if quiet:
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

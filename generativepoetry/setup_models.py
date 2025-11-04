"""Automated setup and download of required NLTK data and spaCy models.

This module provides centralized model management to eliminate new user friction.
All models are downloaded automatically when needed, with clear progress reporting.
"""

import subprocess
import sys
from typing import List, Tuple

import nltk

from .logger import logger

# Required NLTK data packages
REQUIRED_NLTK_DATA = [
    ("tokenizers/punkt", "punkt", "Punkt sentence tokenizer"),
    ("corpora/words", "words", "Words corpus"),
    ("corpora/brown", "brown", "Brown corpus"),
    ("corpora/wordnet", "wordnet", "WordNet lexical database"),
    ("corpora/stopwords", "stopwords", "Stopwords corpus"),
]

# Required spaCy models
REQUIRED_SPACY_MODELS = [
    ("en_core_web_sm", "English language model (small)"),
]


def check_nltk_data(data_path: str) -> bool:
    """Check if NLTK data package is available.

    Args:
        data_path: NLTK data path (e.g., 'tokenizers/punkt')

    Returns:
        True if data is available, False otherwise
    """
    try:
        nltk.data.find(data_path)
        return True
    except LookupError:
        return False


def download_nltk_data(package_name: str, description: str, quiet: bool = False) -> bool:
    """Download NLTK data package.

    Args:
        package_name: Package name to download (e.g., 'punkt')
        description: Human-readable description
        quiet: Suppress output

    Returns:
        True if successful, False otherwise
    """
    try:
        if not quiet:
            logger.info(f"Downloading NLTK {description}...")
        return nltk.download(package_name, quiet=quiet)
    except Exception as e:
        logger.error(f"Failed to download NLTK {description}: {e}")
        return False


def check_spacy_model(model_name: str) -> bool:
    """Check if spaCy model is installed.

    Args:
        model_name: Model name (e.g., 'en_core_web_sm')

    Returns:
        True if model is installed, False otherwise
    """
    try:
        import spacy

        spacy.load(model_name)
        return True
    except (OSError, ImportError):
        return False


def download_spacy_model(model_name: str, description: str, quiet: bool = False) -> bool:
    """Download spaCy model.

    Args:
        model_name: Model name to download (e.g., 'en_core_web_sm')
        description: Human-readable description
        quiet: Suppress output

    Returns:
        True if successful, False otherwise
    """
    try:
        if not quiet:
            logger.info(f"Downloading spaCy {description}...")
            logger.info("This may take a few minutes...")

        # Use python -m spacy download to ensure it uses the correct Python environment
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            if not quiet:
                logger.info(f"Successfully downloaded spaCy {description}")
            return True
        else:
            logger.error(f"Failed to download spaCy {description}")
            if result.stderr and not quiet:
                logger.error(f"Error: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Failed to download spaCy {description}: {e}")
        return False


def ensure_nltk_data(quiet: bool = False) -> Tuple[List[str], List[str]]:
    """Ensure all required NLTK data is available, downloading if needed.

    Args:
        quiet: Suppress output

    Returns:
        Tuple of (already_installed, newly_downloaded) package names
    """
    already_installed = []
    newly_downloaded = []

    for data_path, package_name, description in REQUIRED_NLTK_DATA:
        if check_nltk_data(data_path):
            already_installed.append(package_name)
            if not quiet:
                logger.debug(f"NLTK {description} already installed")
        else:
            if download_nltk_data(package_name, description, quiet=quiet):
                newly_downloaded.append(package_name)
            else:
                logger.warning(f"Could not download NLTK {description}")

    return already_installed, newly_downloaded


def ensure_spacy_models(quiet: bool = False) -> Tuple[List[str], List[str]]:
    """Ensure all required spaCy models are available, downloading if needed.

    Args:
        quiet: Suppress output

    Returns:
        Tuple of (already_installed, newly_downloaded) model names
    """
    already_installed = []
    newly_downloaded = []

    for model_name, description in REQUIRED_SPACY_MODELS:
        if check_spacy_model(model_name):
            already_installed.append(model_name)
            if not quiet:
                logger.debug(f"spaCy {description} already installed")
        else:
            if download_spacy_model(model_name, description, quiet=quiet):
                newly_downloaded.append(model_name)
            else:
                logger.warning(f"Could not download spaCy {description}")

    return already_installed, newly_downloaded


def setup(quiet: bool = False, force: bool = False) -> bool:
    """Complete setup: download all required NLTK data and spaCy models.

    This is the main entry point for automated setup. Call this to ensure
    all dependencies are properly installed.

    Args:
        quiet: Suppress output
        force: Force re-download even if already installed

    Returns:
        True if all models are available (either already installed or newly downloaded)

    Example:
        >>> from generativepoetry.setup_models import setup
        >>> setup()  # Downloads all required models
        True
    """
    if not quiet:
        logger.info("Checking and installing required models...")
        logger.info("")

    success = True

    # NLTK data
    if not quiet:
        logger.info("NLTK Data Packages:")

    nltk_installed, nltk_downloaded = ensure_nltk_data(quiet=quiet)

    if not quiet:
        if nltk_installed:
            logger.info(f"  Already installed: {', '.join(nltk_installed)}")
        if nltk_downloaded:
            logger.info(f"  Newly downloaded: {', '.join(nltk_downloaded)}")
        logger.info("")

    # Check if all NLTK packages were successfully handled
    all_nltk_packages = {pkg for _, pkg, _ in REQUIRED_NLTK_DATA}
    handled_packages = set(nltk_installed) | set(nltk_downloaded)

    if not handled_packages.issuperset(all_nltk_packages):
        # Some packages failed to download
        missing_packages = all_nltk_packages - handled_packages
        for package_name in missing_packages:
            success = False
            logger.error(f"  Missing: {package_name}")

    # spaCy models
    if not quiet:
        logger.info("spaCy Models:")

    spacy_installed, spacy_downloaded = ensure_spacy_models(quiet=quiet)

    if not quiet:
        if spacy_installed:
            logger.info(f"  Already installed: {', '.join(spacy_installed)}")
        if spacy_downloaded:
            logger.info(f"  Newly downloaded: {', '.join(spacy_downloaded)}")
        logger.info("")

    # Check if all spaCy models were successfully handled
    all_spacy_models = {model for model, _ in REQUIRED_SPACY_MODELS}
    handled_models = set(spacy_installed) | set(spacy_downloaded)

    if not handled_models.issuperset(all_spacy_models):
        # Some models failed to download
        missing_models = all_spacy_models - handled_models
        for model_name in missing_models:
            success = False
            logger.error(f"  Missing: {model_name}")

    if success and not quiet:
        logger.info("All required models are installed and ready!")
    elif not success:
        logger.error("Some models failed to install. Please check the errors above.")

    return success


def lazy_ensure_nltk_data(data_path: str, package_name: str, description: str) -> bool:
    """Lazy-load NLTK data: check if available, download if needed.

    This is useful for importing at module level - it only downloads
    the first time the data is needed.

    Args:
        data_path: NLTK data path (e.g., 'tokenizers/punkt')
        package_name: Package name (e.g., 'punkt')
        description: Human-readable description

    Returns:
        True if data is available (either already or after download)
    """
    if check_nltk_data(data_path):
        return True

    logger.info(f"Downloading required NLTK {description}...")
    return download_nltk_data(package_name, description, quiet=False)


def lazy_ensure_spacy_model(model_name: str, description: str) -> bool:
    """Lazy-load spaCy model: check if available, download if needed.

    This is useful for importing at module level - it only downloads
    the first time the model is needed.

    Args:
        model_name: Model name (e.g., 'en_core_web_sm')
        description: Human-readable description

    Returns:
        True if model is available (either already or after download)
    """
    if check_spacy_model(model_name):
        return True

    logger.warning(f"spaCy {description} not found. Attempting to download...")
    logger.warning("This may take a few minutes on first run.")
    return download_spacy_model(model_name, description, quiet=False)

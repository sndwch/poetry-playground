"""Utility functions for working with Project Gutenberg texts using gutenbergpy."""

import re

from bs4 import BeautifulSoup
from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import strip_headers


def clean_gutenberg_text(text):
    """Clean up Gutenberg text by removing metadata and extra whitespace."""
    # Strip headers if not already done
    text = strip_headers(text)

    # Decode bytes to string if needed
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    # Remove Project Gutenberg headers/footers that might remain
    text = re.sub(r"\*\*\* START OF.*?\*\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*\* END OF.*?\*\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"End of the Project Gutenberg.*", "", text, flags=re.DOTALL)

    # Basic HTML cleaning if needed
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

    # Clean up whitespace
    text = re.sub(r"\r\n|\r", "\n", text)  # Normalize line endings
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
    text = re.sub(r"[ \t]+", " ", text)  # Collapse spaces/tabs
    text = re.sub(r" +\n", "\n", text)  # Remove trailing spaces
    text = re.sub(r"\n +", "\n", text)  # Remove leading spaces on lines

    return text.strip()


def get_gutenberg_metadata_cache():
    """Get or create the Gutenberg metadata cache."""
    cache = GutenbergCache()
    # This will download the cache if not present (~30MB)
    if not cache.exists():
        print("Downloading Gutenberg metadata cache (this is a one-time operation)...")
        cache.create()
    return cache


def get_document_id_from_url(url):
    """Extract document ID from a Gutenberg URL."""
    import re
    from urllib.parse import urlsplit

    # Match various Gutenberg URL patterns
    patterns = [
        r"/(?:files|ebooks|epub)/(\d+)",
        r"/(\d+)(?:/|$)",
        r"[?&]id=(\d+)",
    ]

    urlsplit(url).path
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return int(match.group(1))

    raise ValueError(f"Could not extract document ID from URL: {url}")

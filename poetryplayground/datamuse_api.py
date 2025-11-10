"""Centralized Datamuse API wrapper with caching and error handling.

This module provides a singleton interface to the Datamuse API with built-in
caching, retry logic, and consistent error handling. All modules should use
this wrapper instead of creating their own Datamuse instances.

Example:
    >>> from poetryplayground.datamuse_api import DatamuseAPI
    >>> api = DatamuseAPI()
    >>> results = api.words(ml="fire", max=10)  # Means-like "fire"
    >>> antonyms = api.words(rel_ant="hot", max=5)  # Antonyms of "hot"
"""

from functools import lru_cache
from typing import Dict, List

from datamuse import Datamuse

from poetryplayground.cache import cached_api_call
from poetryplayground.logger import logger


@lru_cache(maxsize=1)
def get_datamuse_instance() -> Datamuse:
    """Get singleton Datamuse API instance.

    Returns:
        Datamuse: Shared API instance

    Note:
        Uses LRU cache to ensure only one instance is created across
        the entire application.
    """
    logger.debug("Creating Datamuse API instance")
    return Datamuse()


class DatamuseAPI:
    """Wrapper for Datamuse API with caching and consistent interface.

    This class provides a clean interface to the Datamuse API with automatic
    caching of all requests via the @cached_api_call decorator. All API
    calls go through the cache layer with 24-hour TTL.

    Common relationship codes:
        - ml: means-like (synonyms)
        - sl: sounds-like (near-rhymes)
        - rel_ant: antonyms
        - rel_syn: synonyms
        - rel_trg: triggers (words that often appear together)
        - lc: left context (words that follow)
        - rc: right context (words that precede)

    Example:
        >>> api = DatamuseAPI()
        >>> # Get synonyms
        >>> synonyms = api.words(ml="happy")
        >>> # Get rhymes
        >>> rhymes = api.words(sl="fire")
        >>> # Get antonyms
        >>> antonyms = api.words(rel_ant="hot")
        >>> # Get words that follow "burning"
        >>> followers = api.words(lc="burning")
    """

    def __init__(self):
        """Initialize DatamuseAPI wrapper."""
        self._api = get_datamuse_instance()

    def words(self, **kwargs) -> List[Dict[str, any]]:
        """Query Datamuse API with caching.

        This is the main interface to the Datamuse API. All parameters
        are passed through to the underlying API.

        Args:
            **kwargs: Datamuse API parameters (ml, sl, rel_ant, etc.)

        Returns:
            List of dicts with 'word' and 'score' keys

        Example:
            >>> api = DatamuseAPI()
            >>> api.words(ml="ocean", max=5)
            [{'word': 'sea', 'score': 67890}, ...]

        Note:
            Results are automatically cached with 24-hour TTL via the
            cache.py infrastructure.
        """
        # Convert kwargs to a hashable cache key
        cache_key = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

        return self._cached_words_call(cache_key, **kwargs)

    @cached_api_call(endpoint="datamuse.words", ttl=86400)
    def _cached_words_call(self, cache_key: str, **kwargs) -> List[Dict[str, any]]:
        """Internal cached API call.

        Args:
            cache_key: Hashable key for caching (unused, just for cache decorator)
            **kwargs: Datamuse API parameters

        Returns:
            API response list
        """
        try:
            results = self._api.words(**kwargs)
            logger.debug(f"Datamuse API call with {kwargs} returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Datamuse API call failed with {kwargs}: {e}")
            return []

    # Convenience methods for common queries

    def means_like(self, word: str, max_results: int = 20) -> List[Dict[str, any]]:
        """Get words with similar meaning (synonyms).

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of similar words with scores
        """
        return self.words(ml=word, max=max_results)

    def sounds_like(self, word: str, max_results: int = 20) -> List[Dict[str, any]]:
        """Get words that sound similar (rhymes, near-rhymes).

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of similar-sounding words with scores
        """
        return self.words(sl=word, max=max_results)

    def antonyms(self, word: str, max_results: int = 10) -> List[Dict[str, any]]:
        """Get antonyms of a word.

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of antonyms with scores
        """
        return self.words(rel_ant=word, max=max_results)

    def synonyms(self, word: str, max_results: int = 20) -> List[Dict[str, any]]:
        """Get synonyms of a word.

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of synonyms with scores
        """
        return self.words(rel_syn=word, max=max_results)

    def triggers(self, word: str, max_results: int = 20) -> List[Dict[str, any]]:
        """Get words that often appear together (collocations).

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of related words with scores
        """
        return self.words(rel_trg=word, max=max_results)

    def left_context(self, word: str, max_results: int = 20) -> List[Dict[str, any]]:
        """Get words that often follow this word.

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of following words with scores
        """
        return self.words(lc=word, max=max_results)

    def right_context(self, word: str, max_results: int = 20) -> List[Dict[str, any]]:
        """Get words that often precede this word.

        Args:
            word: Target word
            max_results: Maximum number of results

        Returns:
            List of preceding words with scores
        """
        return self.words(rc=word, max=max_results)


# Convenience function for quick access
def get_datamuse_api() -> DatamuseAPI:
    """Get a DatamuseAPI instance.

    This is a convenience function for obtaining an API instance.
    You can also instantiate DatamuseAPI() directly.

    Returns:
        DatamuseAPI instance

    Example:
        >>> api = get_datamuse_api()
        >>> results = api.means_like("fire")
    """
    return DatamuseAPI()

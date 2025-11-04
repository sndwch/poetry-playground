"""Enhanced caching utilities for API calls with persistent storage.

Uses diskcache for efficient persistent caching of Datamuse API and CMU
pronouncing dictionary lookups. Includes exponential backoff, retry logic,
and offline mode support.
"""

import hashlib
import json
import random
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from diskcache import Cache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    Cache = None

from .config import config
from .logger import logger

# API version for cache invalidation
API_VERSION = "1.0"


class CacheKeyGenerator:
    """Generate consistent cache keys for API calls."""

    @staticmethod
    def generate(endpoint: str, params: dict, version: str = API_VERSION) -> str:
        """Generate cache key from endpoint, params, and version.

        Args:
            endpoint: API endpoint or function name
            params: Parameters passed to the API
            version: API version for cache invalidation

        Returns:
            MD5 hash of the key components
        """
        key_data = {
            'endpoint': endpoint,
            'params': sorted(params.items()) if isinstance(params, dict) else params,
            'version': version
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


class RetryStrategy:
    """Exponential backoff retry strategy for API calls."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        """Initialize retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def wait(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base * 2^attempt
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)

        # Add jitter (±20%) to prevent thundering herd
        jitter = delay * 0.2 * (random.random() * 2 - 1)

        return max(0, delay + jitter)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self.wait(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")

        raise last_exception


class PersistentAPICache:
    """Persistent API cache using diskcache with offline mode support."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl: int = 86400,
        offline_mode: bool = False
    ):
        """Initialize persistent cache.

        Args:
            cache_dir: Directory to store cache
            ttl: Time to live in seconds (default: 24 hours)
            offline_mode: If True, never make API calls, only use cache
        """
        self.cache_dir = cache_dir or config.cache_dir
        self.ttl = ttl
        self.enabled = config.enable_cache
        self.offline_mode = offline_mode
        self.retry_strategy = RetryStrategy(
            max_retries=config.max_api_retries,
            base_delay=config.api_retry_delay
        )

        # Initialize cache backend
        if self.enabled and DISKCACHE_AVAILABLE:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = Cache(str(self.cache_dir))
            self._backend = 'diskcache'
            logger.debug(f"Initialized diskcache at {self.cache_dir}")
        else:
            if self.enabled and not DISKCACHE_AVAILABLE:
                logger.warning("diskcache not available, caching disabled. Install with: pip install diskcache")
            self.cache = None
            self._backend = 'none'

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled or self.cache is None:
            return None

        try:
            value = self.cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit: {key[:16]}...")
                return value
            else:
                logger.debug(f"Cache miss: {key[:16]}...")
                return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if not specified)
        """
        if not self.enabled or self.cache is None:
            return

        try:
            expire_time = ttl or self.ttl
            self.cache.set(key, value, expire=expire_time)
            logger.debug(f"Cached: {key[:16]}... (TTL: {expire_time}s)")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        if self.enabled and self.cache is not None:
            try:
                self.cache.clear()
                logger.info("Cache cleared")
            except Exception as e:
                logger.warning(f"Cache clear error: {e}")

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or self.cache is None:
            return {'enabled': False, 'backend': self._backend}

        try:
            volume = self.cache.volume()
            size = len(self.cache)
            return {
                'enabled': True,
                'backend': self._backend,
                'size': size,
                'volume_bytes': volume,
                'offline_mode': self.offline_mode
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {e}")
            return {'enabled': True, 'backend': self._backend, 'error': str(e)}


def cached_api_call(
    endpoint: str,
    ttl: int = 3600,
    offline_mode: bool = False
):
    """Decorator to cache API call results with retry logic.

    Args:
        endpoint: Name of the API endpoint (for cache key)
        ttl: Time to live in seconds (default: 1 hour)
        offline_mode: If True, only use cache, never make API calls

    Example:
        @cached_api_call(endpoint='datamuse.words', ttl=86400)
        def get_similar_words(word, max_results=20):
            return datamuse_api.words(ml=word, max=max_results)
    """
    cache = PersistentAPICache(ttl=ttl, offline_mode=offline_mode)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from endpoint and parameters
            params = {'args': args, 'kwargs': kwargs}
            cache_key = CacheKeyGenerator.generate(endpoint, params)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # If offline mode and no cache hit, raise error
            if offline_mode:
                logger.error(f"Offline mode: no cache entry for {endpoint}")
                return None  # Could raise exception instead

            # Call the function with retry logic
            try:
                result = cache.retry_strategy.execute(func, *args, **kwargs)

                # Cache the result if successful
                if result is not None:
                    cache.set(cache_key, result, ttl=ttl)

                return result
            except Exception as e:
                logger.error(f"API call failed after retries: {endpoint} - {e}")
                raise

        return wrapper

    return decorator


# Global cache instance
api_cache = PersistentAPICache()


def get_cache_stats() -> dict:
    """Get global cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    return api_cache.stats()


def clear_cache() -> None:
    """Clear global cache."""
    api_cache.clear()

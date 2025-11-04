"""Caching utilities for API calls."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps

from .config import config
from .logger import logger


class APICache:
    """Simple file-based cache for API responses."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl: int = 86400):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir or config.cache_dir
        self.ttl = ttl
        self.enabled = config.enable_cache

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired."""
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)

            # Check if cache is expired
            if time.time() - cached_data['timestamp'] > self.ttl:
                cache_path.unlink()  # Delete expired cache
                return None

            logger.debug(f"Cache hit for {cache_key}")
            return cached_data['data']

        except (json.JSONDecodeError, KeyError, IOError):
            # Corrupted cache file, remove it
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, cache_key: str, value: Any) -> None:
        """Store value in cache."""
        if not self.enabled:
            return

        cache_path = self._get_cache_path(cache_key)
        cached_data = {
            'timestamp': time.time(),
            'data': value
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(cached_data, f)
            logger.debug(f"Cached {cache_key}")
        except (IOError, TypeError) as e:
            logger.warning(f"Failed to cache {cache_key}: {e}")

    def clear(self) -> None:
        """Clear all cache files."""
        if self.enabled and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")


def cached_api_call(ttl: int = 3600):
    """Decorator to cache API call results.

    Args:
        ttl: Time to live in seconds (default: 1 hour)
    """
    cache = APICache(ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._get_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call the function
            result = func(*args, **kwargs)

            # Cache the result
            if result is not None:
                cache.set(cache_key, result)

            return result

        return wrapper

    return decorator


# Global cache instance
api_cache = APICache()
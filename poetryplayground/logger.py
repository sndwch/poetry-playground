"""Logging, profiling, and observability for poetryplayground.

This module provides:
- Structured logging with contextual information
- Performance profiling with timing decorators
- Cache hit/miss tracking
- API call counting
- Beautiful profiling reports
"""

import functools
import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

# Global profiler instance
_profiler: Optional["Profiler"] = None


def setup_logger(name="poetryplayground", level=logging.INFO):
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


class Profiler:
    """Performance profiler for tracking timing, cache hits, and API calls.

    Collects structured performance data and generates reports.
    """

    def __init__(self, enabled: bool = False):
        """Initialize profiler.

        Args:
            enabled: Whether profiling is active
        """
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.cache_hits: Dict[str, int] = defaultdict(int)
        self.cache_misses: Dict[str, int] = defaultdict(int)
        self.api_calls: Dict[str, int] = defaultdict(int)
        self.metadata: Dict[str, Any] = {}

    def reset(self) -> None:
        """Clear all profiling data."""
        self.timings.clear()
        self.cache_hits.clear()
        self.cache_misses.clear()
        self.api_calls.clear()
        self.metadata.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata (seed, procedure name, etc.).

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations.

        Args:
            operation: Name of the operation being timed

        Example:
            with profiler.timer("generate_poem"):
                poem = generate_poem()
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[operation].append(elapsed)

    def record_cache_hit(self, cache_type: str) -> None:
        """Record a cache hit.

        Args:
            cache_type: Type of cache (e.g., "datamuse", "cmudict")
        """
        if self.enabled:
            self.cache_hits[cache_type] += 1

    def record_cache_miss(self, cache_type: str) -> None:
        """Record a cache miss.

        Args:
            cache_type: Type of cache
        """
        if self.enabled:
            self.cache_misses[cache_type] += 1

    def record_api_call(self, api_name: str) -> None:
        """Record an API call.

        Args:
            api_name: Name of the API (e.g., "datamuse", "gutenberg")
        """
        if self.enabled:
            self.api_calls[api_name] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary as structured data.

        Returns:
            Dictionary with profiling statistics
        """
        summary = {
            "metadata": self.metadata.copy(),
            "timings": {},
            "cache_stats": {},
            "api_calls": dict(self.api_calls),
        }

        # Aggregate timing stats
        for operation, times in self.timings.items():
            if times:
                summary["timings"][operation] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }

        # Aggregate cache stats
        all_caches = set(self.cache_hits.keys()) | set(self.cache_misses.keys())
        for cache in all_caches:
            hits = self.cache_hits[cache]
            misses = self.cache_misses[cache]
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0.0

            summary["cache_stats"][cache] = {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hit_rate,
            }

        return summary

    def print_report(self) -> None:
        """Print a beautiful profiling report to console."""
        if not self.enabled:
            logger.info("Profiling was not enabled for this session")
            return

        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING REPORT".center(80))
        print("=" * 80)

        # Metadata
        if summary["metadata"]:
            print("\n" + "Metadata".center(80, "-"))
            for key, value in summary["metadata"].items():
                print(f"  {key}: {value}")

        # Timing Report
        if summary["timings"]:
            print("\n" + "Timing Statistics".center(80, "-"))
            print(
                f"{'Operation':<40} {'Count':>8} {'Total':>10} {'Mean':>10} {'Min':>10} {'Max':>10}"
            )
            print("-" * 80)

            # Sort by total time descending
            sorted_timings = sorted(
                summary["timings"].items(), key=lambda x: x[1]["total"], reverse=True
            )

            for operation, stats in sorted_timings:
                print(
                    f"{operation:<40} {stats['count']:>8} "
                    f"{stats['total']:>9.3f}s {stats['mean']:>9.3f}s "
                    f"{stats['min']:>9.3f}s {stats['max']:>9.3f}s"
                )

            # Total time
            total_time = sum(stats["total"] for _, stats in sorted_timings)
            print("-" * 80)
            print(f"{'TOTAL TIME':<40} {'':<8} {total_time:>9.3f}s")

        # Cache Statistics
        if summary["cache_stats"]:
            print("\n" + "Cache Statistics".center(80, "-"))
            print(f"{'Cache Type':<40} {'Hits':>10} {'Misses':>10} {'Total':>10} {'Hit Rate':>10}")
            print("-" * 80)

            for cache, stats in sorted(summary["cache_stats"].items()):
                print(
                    f"{cache:<40} {stats['hits']:>10} {stats['misses']:>10} "
                    f"{stats['total']:>10} {stats['hit_rate']:>9.1f}%"
                )

        # API Calls
        if summary["api_calls"]:
            print("\n" + "API Calls".center(80, "-"))
            print(f"{'API':<60} {'Calls':>18}")
            print("-" * 80)

            for api, count in sorted(
                summary["api_calls"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"{api:<60} {count:>18}")

        print("=" * 80 + "\n")


def timed(operation: Optional[str] = None) -> Callable:
    """Decorator to time function execution.

    Args:
        operation: Name for the operation (defaults to function name)

    Example:
        @timed("poem_generation")
        def generate_poem():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            op_name = operation or f"{func.__module__}.{func.__name__}"

            with profiler.timer(op_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_profiler() -> Profiler:
    """Get the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler(enabled=False)
    return _profiler


def enable_profiling() -> Profiler:
    """Enable profiling and return profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler(enabled=True)
    else:
        _profiler.enabled = True
        _profiler.reset()
    return _profiler


def disable_profiling() -> None:
    """Disable profiling."""
    global _profiler
    if _profiler is not None:
        _profiler.enabled = False

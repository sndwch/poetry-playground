"""Tests for profiling and observability functionality."""

import unittest
from unittest.mock import MagicMock

from generativepoetry.logger import (
    Profiler,
    disable_profiling,
    enable_profiling,
    get_profiler,
    timed,
)


class TestProfiler(unittest.TestCase):
    """Test Profiler class functionality."""

    def setUp(self):
        """Reset profiler state before each test."""
        self.profiler = Profiler(enabled=True)

    def test_profiler_initialization(self):
        """Test profiler initializes with correct state."""
        profiler = Profiler(enabled=False)
        self.assertFalse(profiler.enabled)
        self.assertEqual(len(profiler.timings), 0)
        self.assertEqual(len(profiler.cache_hits), 0)
        self.assertEqual(len(profiler.cache_misses), 0)
        self.assertEqual(len(profiler.api_calls), 0)
        self.assertEqual(len(profiler.metadata), 0)

    def test_profiler_enabled(self):
        """Test profiler can be enabled and disabled."""
        profiler = Profiler(enabled=True)
        self.assertTrue(profiler.enabled)

        profiler.enabled = False
        self.assertFalse(profiler.enabled)

    def test_timer_records_timing(self):
        """Test timer context manager records timing data."""
        import time

        with self.profiler.timer("test_operation"):
            time.sleep(0.01)  # Small sleep to have measurable time

        self.assertIn("test_operation", self.profiler.timings)
        self.assertEqual(len(self.profiler.timings["test_operation"]), 1)
        self.assertGreater(self.profiler.timings["test_operation"][0], 0)

    def test_timer_disabled_no_recording(self):
        """Test timer doesn't record when profiler is disabled."""
        profiler = Profiler(enabled=False)

        with profiler.timer("test_operation"):
            pass

        self.assertEqual(len(profiler.timings), 0)

    def test_record_cache_hit(self):
        """Test recording cache hits."""
        self.profiler.record_cache_hit("test_cache")
        self.assertEqual(self.profiler.cache_hits["test_cache"], 1)

        self.profiler.record_cache_hit("test_cache")
        self.assertEqual(self.profiler.cache_hits["test_cache"], 2)

    def test_record_cache_miss(self):
        """Test recording cache misses."""
        self.profiler.record_cache_miss("test_cache")
        self.assertEqual(self.profiler.cache_misses["test_cache"], 1)

        self.profiler.record_cache_miss("test_cache")
        self.assertEqual(self.profiler.cache_misses["test_cache"], 2)

    def test_record_api_call(self):
        """Test recording API calls."""
        self.profiler.record_api_call("test_api")
        self.assertEqual(self.profiler.api_calls["test_api"], 1)

        self.profiler.record_api_call("test_api")
        self.assertEqual(self.profiler.api_calls["test_api"], 2)

    def test_cache_disabled_no_recording(self):
        """Test cache operations don't record when profiler is disabled."""
        profiler = Profiler(enabled=False)

        profiler.record_cache_hit("test")
        profiler.record_cache_miss("test")
        profiler.record_api_call("test")

        self.assertEqual(len(profiler.cache_hits), 0)
        self.assertEqual(len(profiler.cache_misses), 0)
        self.assertEqual(len(profiler.api_calls), 0)

    def test_set_metadata(self):
        """Test setting metadata."""
        self.profiler.set_metadata("seed", 42)
        self.profiler.set_metadata("procedure", "markov")

        self.assertEqual(self.profiler.metadata["seed"], 42)
        self.assertEqual(self.profiler.metadata["procedure"], "markov")

    def test_reset_clears_all_data(self):
        """Test reset clears all profiling data."""
        self.profiler.record_cache_hit("cache1")
        self.profiler.record_api_call("api1")
        self.profiler.set_metadata("key", "value")

        with self.profiler.timer("operation"):
            pass

        self.profiler.reset()

        self.assertEqual(len(self.profiler.timings), 0)
        self.assertEqual(len(self.profiler.cache_hits), 0)
        self.assertEqual(len(self.profiler.cache_misses), 0)
        self.assertEqual(len(self.profiler.api_calls), 0)
        self.assertEqual(len(self.profiler.metadata), 0)

    def test_get_summary_structure(self):
        """Test get_summary returns correct structure."""
        self.profiler.record_cache_hit("cache1")
        self.profiler.record_cache_miss("cache1")
        self.profiler.record_api_call("api1")
        self.profiler.set_metadata("seed", 123)

        with self.profiler.timer("op1"):
            pass

        summary = self.profiler.get_summary()

        self.assertIn("metadata", summary)
        self.assertIn("timings", summary)
        self.assertIn("cache_stats", summary)
        self.assertIn("api_calls", summary)

        self.assertEqual(summary["metadata"]["seed"], 123)
        self.assertIn("op1", summary["timings"])
        self.assertIn("cache1", summary["cache_stats"])
        self.assertEqual(summary["api_calls"]["api1"], 1)

    def test_get_summary_timing_stats(self):
        """Test get_summary calculates timing statistics correctly."""
        import time

        for _ in range(3):
            with self.profiler.timer("test_op"):
                time.sleep(0.001)

        summary = self.profiler.get_summary()
        stats = summary["timings"]["test_op"]

        self.assertEqual(stats["count"], 3)
        self.assertGreater(stats["total"], 0)
        self.assertGreater(stats["mean"], 0)
        self.assertGreater(stats["min"], 0)
        self.assertGreater(stats["max"], 0)
        self.assertLessEqual(stats["min"], stats["mean"])
        self.assertLessEqual(stats["mean"], stats["max"])

    def test_get_summary_cache_hit_rate(self):
        """Test get_summary calculates cache hit rate correctly."""
        self.profiler.record_cache_hit("cache1")
        self.profiler.record_cache_hit("cache1")
        self.profiler.record_cache_hit("cache1")
        self.profiler.record_cache_miss("cache1")

        summary = self.profiler.get_summary()
        cache_stats = summary["cache_stats"]["cache1"]

        self.assertEqual(cache_stats["hits"], 3)
        self.assertEqual(cache_stats["misses"], 1)
        self.assertEqual(cache_stats["total"], 4)
        self.assertEqual(cache_stats["hit_rate"], 75.0)

    def test_get_summary_multiple_operations(self):
        """Test get_summary handles multiple operations correctly."""
        with self.profiler.timer("op1"):
            pass
        with self.profiler.timer("op2"):
            pass

        self.profiler.record_cache_hit("cache1")
        self.profiler.record_cache_hit("cache2")

        summary = self.profiler.get_summary()

        self.assertEqual(len(summary["timings"]), 2)
        self.assertEqual(len(summary["cache_stats"]), 2)

    def test_print_report_disabled(self):
        """Test print_report handles disabled profiler gracefully."""
        profiler = Profiler(enabled=False)
        # Should not raise an exception
        profiler.print_report()


class TestTimedDecorator(unittest.TestCase):
    """Test @timed decorator functionality."""

    def setUp(self):
        """Enable profiling before each test."""
        self.profiler = enable_profiling()

    def tearDown(self):
        """Disable profiling after each test."""
        disable_profiling()

    def test_timed_decorator_records_timing(self):
        """Test @timed decorator records function timing."""

        @timed("test_function")
        def sample_function():
            return "result"

        result = sample_function()

        self.assertEqual(result, "result")
        summary = self.profiler.get_summary()
        self.assertIn("test_function", summary["timings"])

    def test_timed_decorator_with_args(self):
        """Test @timed decorator works with function arguments."""

        @timed("test_add")
        def add(a, b):
            return a + b

        result = add(2, 3)

        self.assertEqual(result, 5)
        summary = self.profiler.get_summary()
        self.assertIn("test_add", summary["timings"])

    def test_timed_decorator_default_name(self):
        """Test @timed decorator uses function name when no name provided."""

        @timed()
        def my_function():
            return "test"

        result = my_function()

        self.assertEqual(result, "test")
        summary = self.profiler.get_summary()
        # Should contain function module and name
        matching_keys = [k for k in summary["timings"].keys() if "my_function" in k]
        self.assertGreater(len(matching_keys), 0)


class TestGlobalProfiler(unittest.TestCase):
    """Test global profiler management functions."""

    def test_get_profiler_creates_instance(self):
        """Test get_profiler creates a profiler instance."""
        profiler = get_profiler()
        self.assertIsInstance(profiler, Profiler)

    def test_get_profiler_returns_same_instance(self):
        """Test get_profiler returns the same global instance."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        self.assertIs(profiler1, profiler2)

    def test_enable_profiling_enables_and_resets(self):
        """Test enable_profiling enables profiler and resets data."""
        profiler = enable_profiling()
        self.assertTrue(profiler.enabled)

        # Add some data
        profiler.record_cache_hit("test")

        # Re-enable should reset
        profiler2 = enable_profiling()
        self.assertIs(profiler, profiler2)
        self.assertTrue(profiler2.enabled)
        self.assertEqual(len(profiler2.cache_hits), 0)

    def test_disable_profiling_disables(self):
        """Test disable_profiling disables the profiler."""
        profiler = enable_profiling()
        self.assertTrue(profiler.enabled)

        disable_profiling()
        self.assertFalse(profiler.enabled)


if __name__ == "__main__":
    unittest.main()

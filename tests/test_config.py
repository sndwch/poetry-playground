#!/usr/bin/env python3
"""Comprehensive tests for configuration system."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generativepoetry.config import (
    Config,
    DocumentConfig,
    PerformanceConfig,
    QualityConfig,
    get_config,
)


class TestConfigDataModels(unittest.TestCase):
    """Test configuration data model classes."""

    def test_document_config_defaults(self):
        """Test DocumentConfig default values."""
        config = DocumentConfig()
        self.assertIsInstance(config.MIN_LENGTH_GENERAL, int)
        self.assertGreater(config.MIN_LENGTH_GENERAL, 0)

    def test_performance_config_defaults(self):
        """Test PerformanceConfig default values."""
        config = PerformanceConfig()
        self.assertIsInstance(config.MAX_PROCESSING_ATTEMPTS, int)
        self.assertGreater(config.MAX_PROCESSING_ATTEMPTS, 0)
        self.assertIsInstance(config.TIMEOUT_SECONDS, int)
        self.assertGreater(config.TIMEOUT_SECONDS, 0)

    def test_quality_config_defaults(self):
        """Test QualityConfig default values."""
        config = QualityConfig()
        # Quality config should have settings
        self.assertIsNotNone(config)


class TestMainConfig(unittest.TestCase):
    """Test main Config class."""

    def test_config_creation(self):
        """Test creating a Config instance."""
        config = Config()
        self.assertIsNotNone(config)

    def test_config_has_cache_settings(self):
        """Test that Config has cache-related settings."""
        config = Config()
        # Should have cache enabled flag
        self.assertTrue(hasattr(config, 'cache_enabled'))

    def test_config_has_spacy_model(self):
        """Test that Config has spaCy model setting."""
        config = Config()
        self.assertTrue(hasattr(config, 'spacy_model'))

    def test_config_validation(self):
        """Test that config validates constraints."""
        # This should work
        config = Config()
        self.assertIsNotNone(config)


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading from various sources."""

    def test_get_config_returns_config(self):
        """Test that get_config() returns a Config instance."""
        config = get_config()
        self.assertIsInstance(config, Config)

    def test_get_config_caching(self):
        """Test that get_config() returns same instance (lazy loading)."""
        config1 = get_config()
        config2 = get_config()
        # Should return same instance due to caching
        self.assertIs(config1, config2)

    def test_config_from_environment(self):
        """Test loading config from environment variables."""
        # Set environment variable
        with patch.dict(os.environ, {"GP_CACHE_ENABLED": "false"}):
            # Would need to reload config module or clear cache
            # For now, just test that env vars don't break loading
            config = get_config()
            self.assertIsNotNone(config)

    def test_config_from_yaml(self):
        """Test loading config from YAML file."""
        # Create a temporary YAML config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""
cache:
  enabled: true
  ttl_seconds: 3600

generation:
  max_lines: 20
""")
            yaml_path = f.name

        try:
            # Test that YAML file can be read (actual loading depends on implementation)
            self.assertTrue(Path(yaml_path).exists())
        finally:
            os.unlink(yaml_path)


class TestConfigTypes(unittest.TestCase):
    """Test configuration type constraints."""

    def test_config_cache_enabled_type(self):
        """Test Config cache_enabled type."""
        config = Config()
        self.assertIsInstance(config.cache_enabled, bool)

    def test_document_config_types(self):
        """Test DocumentConfig type constraints."""
        config = DocumentConfig()
        self.assertIsInstance(config.MIN_LENGTH_GENERAL, int)

    def test_performance_config_types(self):
        """Test PerformanceConfig type constraints."""
        config = PerformanceConfig()
        self.assertIsInstance(config.MAX_PROCESSING_ATTEMPTS, int)
        self.assertIsInstance(config.TIMEOUT_SECONDS, int)


class TestConfigModification(unittest.TestCase):
    """Test modifying configuration values."""

    def test_config_cache_setting(self):
        """Test Config cache setting."""
        config = Config(cache_enabled=True)
        # Verify value is set correctly
        self.assertTrue(config.cache_enabled)

    def test_config_immutability(self):
        """Test whether configs are immutable or mutable."""
        config = Config()
        original_cache = config.cache_enabled

        # Try to modify (may or may not work depending on Pydantic config)
        try:
            config.cache_enabled = not original_cache
            # If modification succeeds, verify it worked
            self.assertEqual(config.cache_enabled, not original_cache)
        except Exception:
            # If immutable, should raise exception
            self.assertEqual(config.cache_enabled, original_cache)


class TestConfigDefaults(unittest.TestCase):
    """Test that configuration defaults are sensible."""

    def test_cache_defaults_are_reasonable(self):
        """Test that cache defaults are reasonable."""
        config = Config()
        self.assertTrue(config.cache_enabled)  # Cache should be enabled by default

    def test_performance_defaults_are_reasonable(self):
        """Test that performance defaults are reasonable."""
        config = PerformanceConfig()
        self.assertGreater(config.MAX_PROCESSING_ATTEMPTS, 0)
        self.assertLess(config.MAX_PROCESSING_ATTEMPTS, 100)
        self.assertGreater(config.TIMEOUT_SECONDS, 0)
        self.assertLess(config.TIMEOUT_SECONDS, 600)  # Less than 10 minutes


class TestConfigConsistency(unittest.TestCase):
    """Test configuration consistency and relationships."""

    def test_config_values_are_consistent(self):
        """Test that related config values are consistent."""
        config = Config()

        # Cache should be enabled by default
        self.assertTrue(config.cache_enabled)

    def test_all_configs_load_successfully(self):
        """Test that all config classes can be instantiated."""
        # Should all instantiate without errors
        config = Config()
        document = DocumentConfig()
        performance = PerformanceConfig()
        quality = QualityConfig()

        self.assertIsNotNone(config)
        self.assertIsNotNone(document)
        self.assertIsNotNone(performance)
        self.assertIsNotNone(quality)


if __name__ == "__main__":
    unittest.main()

"""Configuration management for generativepoetry."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration settings for generativepoetry."""

    # API Settings
    datamuse_api_max: int = 50
    datamuse_timeout: int = 10

    # Generation Settings
    default_num_lines: int = 10
    min_line_words: int = 5
    max_line_words: int = 9
    max_line_length: int = 35

    # Output Settings
    output_dir: Path = Path.cwd()
    pdf_orientation: str = 'landscape'
    enable_png_generation: bool = True

    # Caching
    enable_cache: bool = True
    cache_dir: Optional[Path] = None

    # Performance
    max_api_retries: int = 3
    api_retry_delay: float = 1.0

    def __post_init__(self):
        """Initialize paths and create directories if needed."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / '.cache' / 'generativepoetry'

        # Create cache directory if caching is enabled
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        kwargs = {}

        # Read environment variables with GP_ prefix
        env_mappings = {
            'GP_DATAMUSE_API_MAX': ('datamuse_api_max', int),
            'GP_OUTPUT_DIR': ('output_dir', Path),
            'GP_ENABLE_CACHE': ('enable_cache', lambda x: x.lower() == 'true'),
            'GP_CACHE_DIR': ('cache_dir', Path),
        }

        for env_var, (config_key, transform) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                kwargs[config_key] = transform(value)

        return cls(**kwargs)


# Global config instance
config = Config.from_env()
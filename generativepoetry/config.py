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

    # Reproducibility
    seed: Optional[int] = None  # Random seed for deterministic generation

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
            'GP_SEED': ('seed', int),
            'GP_OUTPUT_DIR': ('output_dir', Path),
            'GP_ENABLE_CACHE': ('enable_cache', lambda x: x.lower() == 'true'),
            'GP_CACHE_DIR': ('cache_dir', Path),
        }

        for env_var, (config_key, transform) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                kwargs[config_key] = transform(value)

        return cls(**kwargs)


# Document retrieval configuration
class DocumentConfig:
    """Document length requirements for different extraction types"""

    # Minimum document lengths for different purposes
    MIN_LENGTH_GENERAL = 1000          # General single document retrieval
    MIN_LENGTH_IDEAS = 2000            # Poetry idea extraction
    MIN_LENGTH_METAPHORS = 3000        # Metaphor pattern extraction
    MIN_LENGTH_FRAGMENTS = 5000        # Fragment mining and resonant patterns
    MIN_LENGTH_LIBRARY_DEFAULT = 5000  # Default for document library diverse retrieval

    # Batch sizes for adaptive scaling
    INITIAL_BATCH_SIZE_SMALL = 2       # Conservative starting batch
    INITIAL_BATCH_SIZE_MEDIUM = 3      # Standard starting batch
    INITIAL_BATCH_SIZE_LARGE = 5       # Aggressive starting batch

    MAX_ADAPTIVE_BATCH = 5             # Maximum documents in adaptive scaling batch
    MIN_ADAPTIVE_BATCH = 2             # Minimum documents in adaptive scaling batch


class QualityConfig:
    """Quality thresholds and scoring parameters"""

    # Fragment quality thresholds
    FRAGMENT_QUALITY_THRESHOLD = 0.65  # Minimum quality score for fragments

    # Metaphor quality thresholds
    METAPHOR_BASE_SCORE = 0.5          # Base metaphor quality score
    METAPHOR_MAX_SCORE = 1.0           # Maximum metaphor quality score

    # Text processing limits
    MAX_SENTENCES_PER_TEXT = 100       # Maximum sentences to process per document
    MAX_FRAGMENTS_PER_TEXT = 10        # Maximum fragments to extract per document
    MAX_METAPHORS_PER_TEXT = 10        # Maximum metaphors to extract per document
    MAX_IDEAS_PER_TEXT = 8             # Maximum ideas to extract per document


class PerformanceConfig:
    """Performance-related settings"""

    # API rate limiting
    API_DELAY_SECONDS = 0.2            # Delay between API calls
    API_DELAY_IDEAS = 0.3              # Longer delay for idea generation

    # Cache settings
    MAX_DOCUMENT_CACHE = 20            # Maximum cached documents in memory
    MAX_RECENT_TRACKING = 50           # Maximum recently used documents to track

    # Processing limits
    MAX_PROCESSING_ATTEMPTS = 20       # Maximum attempts for document processing
    MAX_RETRY_ATTEMPTS = 5             # Maximum retries for failed operations


# Quick access to commonly used values
DEFAULT_FRAGMENT_COUNT = 100
DEFAULT_METAPHOR_COUNT = 15
DEFAULT_IDEA_COUNT = 20

# Common batch multipliers for adaptive scaling
SCALING_FACTOR_CONSERVATIVE = 3      # remaining_needed // 3
SCALING_FACTOR_AGGRESSIVE = 5        # remaining_needed // 5


# Global config instance
config = Config.from_env()
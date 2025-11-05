"""Configuration management for generativepoetry.

Supports loading configuration from multiple sources with priority:
1. CLI flags (highest priority)
2. --config CONFIG.yml file
3. pyproject.toml [tool.generativepoetry]
4. Environment variables (GP_* prefix)
5. Defaults (lowest priority)
"""

import os
from contextlib import suppress
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for Python 3.8-3.10

import yaml
from pydantic import BaseModel, Field, field_validator


class OutputFormat(str, Enum):
    """Valid output formats."""

    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    TXT = "txt"


class PDFOrientation(str, Enum):
    """Valid PDF orientations."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class SpaCyModel(str, Enum):
    """Available spaCy model sizes."""

    SMALL = "sm"  # en_core_web_sm (~13MB) - Fast, good for most uses
    MEDIUM = "md"  # en_core_web_md (~40MB) - Better accuracy
    LARGE = "lg"  # en_core_web_lg (~560MB) - Best accuracy, slower


class PoemForm(str, Enum):
    """Available syllable-constrained poem forms."""

    HAIKU = "haiku"  # 5-7-5 syllables, traditional Japanese
    TANKA = "tanka"  # 5-7-5-7-7 syllables, Japanese emotional expression
    SENRYU = "senryu"  # 5-7-5 syllables, focus on human nature
    FREE = "free"  # No syllable constraints (default)


class Config(BaseModel):
    """Main configuration model with validation."""

    # API Settings
    datamuse_api_max: int = Field(
        default=50, ge=1, le=1000, description="Maximum results from Datamuse API"
    )
    datamuse_timeout: int = Field(
        default=10, ge=1, le=60, description="Timeout for Datamuse API calls (seconds)"
    )

    # Generation Settings
    default_num_lines: int = Field(
        default=10, ge=1, le=100, description="Default number of lines to generate"
    )
    min_line_words: int = Field(default=5, ge=1, le=20, description="Minimum words per line")
    max_line_words: int = Field(default=9, ge=1, le=50, description="Maximum words per line")
    max_line_length: int = Field(
        default=35, ge=10, le=200, description="Maximum line length in characters"
    )
    poem_form: PoemForm = Field(
        default=PoemForm.FREE,
        description="Poem form with syllable constraints (haiku/tanka/senryu/free)",
    )

    # Reproducibility
    seed: Optional[int] = Field(
        default=None, description="Random seed for deterministic generation"
    )

    # Output Settings
    output_dir: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Directory for output files",
    )
    output_format: OutputFormat = Field(default=OutputFormat.PDF, description="Output format")
    pdf_orientation: PDFOrientation = Field(
        default=PDFOrientation.LANDSCAPE, description="PDF orientation"
    )
    enable_png_generation: bool = Field(default=True, description="Enable PNG generation from PDFs")

    # spaCy Model Selection
    spacy_model: SpaCyModel = Field(
        default=SpaCyModel.SMALL,
        description="spaCy model size (sm/md/lg)",
    )

    # CLI Behavior
    quiet: bool = Field(default=False, description="Suppress non-essential output")
    verbose: bool = Field(default=False, description="Show detailed output")
    no_color: bool = Field(default=False, description="Disable colored output")
    dry_run: bool = Field(default=False, description="Preview without generating files")
    profile: bool = Field(
        default=False, description="Enable performance profiling and timing reports"
    )

    # Caching
    enable_cache: bool = Field(default=True, description="Enable API response caching")
    cache_dir: Optional[Path] = Field(default=None, description="Cache directory path")

    # Performance
    max_api_retries: int = Field(default=3, ge=0, le=10, description="Maximum API retry attempts")
    api_retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Delay between API retries (seconds)"
    )

    @field_validator("min_line_words", "max_line_words")
    @classmethod
    def validate_line_words(cls, v, info):
        """Ensure min <= max for line words."""
        if (
            info.field_name == "max_line_words"
            and "min_line_words" in info.data
            and v < info.data["min_line_words"]
        ):
            raise ValueError("max_line_words must be >= min_line_words")
        return v

    @field_validator("quiet", "verbose")
    @classmethod
    def validate_quiet_verbose(cls, v, info):
        """Ensure quiet and verbose are not both True."""
        if info.field_name == "verbose" and v and info.data.get("quiet"):
            raise ValueError("Cannot set both quiet and verbose to True")
        if info.field_name == "quiet" and v and info.data.get("verbose"):
            raise ValueError("Cannot set both quiet and verbose to True")
        return v

    def model_post_init(self, __context):
        """Initialize paths and create directories if needed."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "generativepoetry"

        # Create cache directory if caching is enabled
        if self.enable_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure output directory exists
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables with GP_ prefix."""
        kwargs = {}

        # Environment variable mappings
        env_mappings = {
            "GP_DATAMUSE_API_MAX": ("datamuse_api_max", int),
            "GP_DATAMUSE_TIMEOUT": ("datamuse_timeout", int),
            "GP_DEFAULT_NUM_LINES": ("default_num_lines", int),
            "GP_MIN_LINE_WORDS": ("min_line_words", int),
            "GP_MAX_LINE_WORDS": ("max_line_words", int),
            "GP_MAX_LINE_LENGTH": ("max_line_length", int),
            "GP_SEED": ("seed", int),
            "GP_OUTPUT_DIR": ("output_dir", Path),
            "GP_OUTPUT_FORMAT": ("output_format", str),
            "GP_PDF_ORIENTATION": ("pdf_orientation", str),
            "GP_ENABLE_PNG_GENERATION": ("enable_png_generation", lambda x: x.lower() == "true"),
            "GP_SPACY_MODEL": ("spacy_model", str),
            "GP_QUIET": ("quiet", lambda x: x.lower() == "true"),
            "GP_VERBOSE": ("verbose", lambda x: x.lower() == "true"),
            "GP_NO_COLOR": ("no_color", lambda x: x.lower() == "true"),
            "GP_DRY_RUN": ("dry_run", lambda x: x.lower() == "true"),
            "GP_PROFILE": ("profile", lambda x: x.lower() == "true"),
            "GP_ENABLE_CACHE": ("enable_cache", lambda x: x.lower() == "true"),
            "GP_CACHE_DIR": ("cache_dir", Path),
            "GP_MAX_API_RETRIES": ("max_api_retries", int),
            "GP_API_RETRY_DELAY": ("api_retry_delay", float),
        }

        for env_var, (config_key, transform) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Skip invalid environment variables
                with suppress(ValueError, TypeError):
                    kwargs[config_key] = transform(value)

        return cls(**kwargs)

    @classmethod
    def from_pyproject(cls, pyproject_path: Optional[Path] = None) -> "Config":
        """Load config from pyproject.toml [tool.generativepoetry] section."""
        if pyproject_path is None:
            # Look for pyproject.toml in current directory or parent directories
            current = Path.cwd()
            while current != current.parent:
                pyproject_path = current / "pyproject.toml"
                if pyproject_path.exists():
                    break
                current = current.parent
            else:
                # Not found, return defaults
                return cls()

        if not pyproject_path.exists():
            return cls()

        try:
            # Try tomllib (Python 3.11+) first, fall back to tomli
            if tomllib:
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
            else:
                import tomli

                with open(pyproject_path, "rb") as f:
                    data = tomli.load(f)

            tool_config = data.get("tool", {}).get("generativepoetry", {})
            return cls(**tool_config)
        except Exception:
            # If parsing fails, return defaults
            return cls()

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load config from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def load(
        cls,
        config_file: Optional[Path] = None,
        cli_overrides: Optional[dict] = None,
    ) -> "Config":
        """Load config from all sources with proper priority.

        Priority (highest to lowest):
        1. CLI overrides
        2. Config file (if specified)
        3. pyproject.toml
        4. Environment variables
        5. Defaults

        Args:
            config_file: Optional path to YAML config file
            cli_overrides: Optional dict of CLI flag overrides

        Returns:
            Merged Config instance
        """
        # Start with defaults
        config_dict = {}

        # Layer 1: pyproject.toml
        try:
            pyproject_config = cls.from_pyproject()
            config_dict.update(pyproject_config.model_dump(exclude_unset=False))
        except Exception:
            pass

        # Layer 2: Environment variables
        try:
            env_config = cls.from_env()
            # Only update with values that were actually set in env
            for key, value in env_config.model_dump(exclude_unset=False).items():
                # Check if env var was actually set
                env_key = f"GP_{key.upper()}"
                if env_key in os.environ:
                    config_dict[key] = value
        except Exception:
            pass

        # Layer 3: Config file (if specified)
        if config_file:
            try:
                file_config = cls.from_yaml(config_file)
                config_dict.update(file_config.model_dump(exclude_unset=False))
            except Exception as e:
                raise ValueError(f"Failed to load config file: {e}") from e

        # Layer 4: CLI overrides (highest priority)
        if cli_overrides:
            config_dict.update({k: v for k, v in cli_overrides.items() if v is not None})

        return cls(**config_dict)

    def save_yaml(self, path: Path):
        """Save current config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="python"), f, default_flow_style=False, sort_keys=False)


# Document retrieval configuration (kept as simple classes for now)
class DocumentConfig:
    """Document length requirements for different extraction types"""

    MIN_LENGTH_GENERAL = 1000
    MIN_LENGTH_IDEAS = 2000
    MIN_LENGTH_METAPHORS = 3000
    MIN_LENGTH_FRAGMENTS = 5000
    MIN_LENGTH_LIBRARY_DEFAULT = 5000

    INITIAL_BATCH_SIZE_SMALL = 2
    INITIAL_BATCH_SIZE_MEDIUM = 3
    INITIAL_BATCH_SIZE_LARGE = 5

    MAX_ADAPTIVE_BATCH = 5
    MIN_ADAPTIVE_BATCH = 2


class QualityConfig:
    """Quality thresholds and scoring parameters"""

    FRAGMENT_QUALITY_THRESHOLD = 0.65
    METAPHOR_BASE_SCORE = 0.5
    METAPHOR_MAX_SCORE = 1.0

    MAX_SENTENCES_PER_TEXT = 100
    MAX_FRAGMENTS_PER_TEXT = 10
    MAX_METAPHORS_PER_TEXT = 10
    MAX_IDEAS_PER_TEXT = 8


class PerformanceConfig:
    """Performance-related settings"""

    API_DELAY_SECONDS = 0.2
    API_DELAY_IDEAS = 0.3

    MAX_DOCUMENT_CACHE = 20
    MAX_RECENT_TRACKING = 50

    MAX_PROCESSING_ATTEMPTS = 20
    MAX_RETRY_ATTEMPTS = 5


# Quick access constants
DEFAULT_FRAGMENT_COUNT = 100
DEFAULT_METAPHOR_COUNT = 15
DEFAULT_IDEA_COUNT = 20

SCALING_FACTOR_CONSERVATIVE = 3
SCALING_FACTOR_AGGRESSIVE = 5


# Global config instance - will be initialized by CLI
config: Optional[Config] = None


def init_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict] = None,
) -> Config:
    """Initialize the global config instance.

    This should be called by the CLI at startup.
    """
    global config
    config = Config.load(config_file=config_file, cli_overrides=cli_overrides)
    return config


# For backward compatibility during migration
def get_config() -> Config:
    """Get the global config instance."""
    global config
    if config is None:
        # Initialize with defaults if not yet initialized
        config = Config.load()
    return config

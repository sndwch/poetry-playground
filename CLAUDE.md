# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Poetry Playground is a **personal experimental fork** of [generativepoetry-py](https://github.com/sndwch/generativepoetry-py), which was originally created by [Corey Bobco](https://github.com/coreybobco). This is a sandbox for exploring generative poetry techniques and computational creativity. It's not intended as a production library.

## Common Commands

### Setup and Installation
```bash
# Install in development mode
pip install -e ".[dev]"

# Download NLTK data and spaCy models (one-time setup)
poetry-playground --setup
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_forms.py

# Run single test function
pytest tests/test_forms.py::test_haiku_generation

# Run with coverage
pytest tests/ --cov=poetryplayground --cov-report=term

# Run with profiling enabled (see timing data)
pytest tests/ --profile
```

### Linting and Formatting
```bash
# Run ruff linter
ruff check poetryplayground/

# Run ruff formatter
ruff format poetryplayground/

# Run type checker (informational, not blocking)
mypy poetryplayground/ --ignore-missing-imports

# Install pre-commit hooks (runs ruff + mypy on commits)
pre-commit install
```

### Running the Application
```bash
# Launch CLI menu (main interface)
poetry-playground

# Launch TUI (Terminal User Interface)
poetry-playground-tui

# Run with specific configuration
poetry-playground --config myconfig.yml --seed 42

# List available procedures
poetry-playground --list-procedures

# Enable profiling to see performance metrics
poetry-playground --profile
```

## Architecture Overview

### Entry Points and UI Layer
- **CLI**: `poetryplayground/cli.py` - Menu-driven interface using Rich for terminal output
- **TUI**: `poetryplayground/tui/app.py` - Textual-based full-screen interface
- Both interfaces use the same underlying generation modules

### Configuration System (Multi-Source Priority)
- **Priority order** (highest to lowest): CLI flags > YAML file > `pyproject.toml` > Environment vars (`PP_*` prefix) > Defaults
- **Implementation**: `poetryplayground/config.py` using Pydantic v2 for validation
- **Global singleton**: Access via `get_config()` throughout codebase
- **Override examples**: `--seed 42`, `--spacy-model lg`, `--format png`

### Core Data Flow Pattern
```
CLI/TUI (entry points)
    ↓
Config (global singleton with priority cascade)
    ↓
Cache Layer (persistent disk cache for API calls)
    ↓
Lexigen API (unified word generation interface)
    ↓
Generators (haiku, metaphor, visual poems, etc.)
    ↓
Output (PDF/PNG/terminal rendering)
```

### Caching Architecture (Three Levels)
1. **API Response Cache** (`cache.py`): All external API calls wrapped with `@cached_api_call` decorator
   - Uses diskcache for persistence
   - Default TTL: 24 hours
   - Exponential backoff retry with jitter
2. **POS Vocabulary Cache** (`pos_vocabulary.py`): Brown corpus analysis cached to `~/.poetryplayground/pos_word_bank.pkl`
3. **Document Library Cache** (`document_library.py`): In-memory LRU cache of Project Gutenberg texts

### Word Generation and Vocabulary
- **Lexigen** (`lexigen.py`): Facade over CMU Pronouncing Dictionary and Datamuse API
  - All functions return cleaned, validated word lists
  - Cached at source for performance
- **Shared Vocabulary** (`vocabulary.py`): 2280-line curated word lists organized by theme
  - Properties: `evocative_verbs`, `atmospheric_nouns`, `emotional_keywords`, `poetic_attributes`
  - Lazy-loaded on first access
- **POS Vocabulary** (`pos_vocabulary.py`): Words organized as `{POS_TAG: {syllable_count: [words]}}`
  - Built from NLTK Brown corpus + curated lists
  - Enables template-based generation with syllable constraints

### Template-Based Generation
- **Grammatical Templates** (`grammatical_templates.py`): POS patterns like `['ADJ', 'NOUN']` → "lonely mountain"
- **Form Generators** (`forms.py`): Haiku/tanka/senryu using templates + syllable counting
- **Two-phase process**:
  1. Select grammatical template matching syllable target
  2. Fill template with words from POS vocabulary
  3. Validate syllable count and return poem + validation report

### Generator Modules (Self-Contained Classes)
Each generator is a standalone class in its own module:
- `line_seeds.py` - LineSeedGenerator: Opening lines, pivots, closings
- `metaphor_generator.py` - MetaphorGenerator: 6 metaphor types using Gutenberg + spaCy
- `idea_generator.py` - PoetryIdeaGenerator: Mines classic literature across 10 categories
- `causal_poetry.py` - ResonantFragmentMiner: 26 patterns across 5 categories
- `six_degrees.py` - SixDegreesExplorer: Semantic convergence paths
- `semantic_geodesic.py` - Finds semantic paths through meaning-space (linear/Bezier/shortest-path)
- `conceptual_cloud.py` - **NEW**: Multi-dimensional word associations (semantic, contextual, opposite, phonetic, imagery, rare)
- `ship_of_theseus.py` - PoemTransformer: Gradual transformation via word substitution
- `corpus_analyzer.py` - PersonalCorpusAnalyzer: Stylistic fingerprinting
- `pdf.py` - Visual poem generators (Futurist, Chaotic, Markov, etc.)

### Seed Management and Reproducibility
- **Global seed** (`seed_manager.py`): `set_global_seed(seed)` affects Python's `random` module
- Seeds echoed at start/end of CLI sessions
- Same seed = identical output across runs
- Critical for testing and reproducible creative experiments

## Key Implementation Patterns

### Adding a New Generator
1. Create class in new module (e.g., `my_generator.py`)
2. Implement generation method(s)
3. Add to CLI menu in `cli.py` (search for `PROCEDURES` dict)
4. Add to TUI in `tui/screens/config_form.py` if desired
5. Write tests in `tests/test_my_generator.py`

### Adding a New Form (e.g., Limerick)
1. Add templates to `TemplateLibrary._load_default_templates()` in `grammatical_templates.py`
2. Create syllable pattern (e.g., [8, 8, 5, 5, 8])
3. Add method to `FormGenerator` class in `forms.py`
4. Wire into CLI via `--form limerick` support

### Adding Cached API Functionality
```python
from poetryplayground.cache import cached_api_call

@cached_api_call(endpoint="myapi.function", ttl=86400)
def my_api_function(param: str) -> list:
    # Make actual API call
    return results
```

### Using Configuration
```python
from poetryplayground.config import get_config

cfg = get_config()
model_size = cfg.spacy_model.value  # 'sm', 'md', or 'lg'
num_lines = cfg.default_num_lines
```

### Profiling Code Sections
```python
from poetryplayground.logger import profiler

with profiler.timer("my_operation"):
    # Code to time
    result = expensive_function()
```

## Testing Infrastructure

### Test Organization
- `conftest.py`: Fixtures including `mock_api_responses`, `disable_network_calls`, `temp_dir`
- All tests mock external APIs to avoid network dependency
- Autouse fixture blocks real network calls globally

### Running Specific Tests
```bash
# Test a specific feature
pytest tests/test_forms.py -v

# Test determinism (reproducibility)
pytest tests/test_deterministic.py

# Test with real APIs (requires network, slow)
pytest tests/ -m "not network"  # Skip network tests
```

### Test Patterns Used
- **Mocked API responses**: No real API calls in tests
- **Seed-based determinism**: Same seed produces same output
- **Captured Rich output**: Tests verify CLI formatting
- **Property-based**: Syllable counting accuracy validation

## Important Gotchas

### spaCy Model Trade-offs
- **sm** (small, 13MB): Fast but less accurate embeddings
- **md** (medium, 40MB): Balanced speed/accuracy
- **lg** (large, 560MB): Best embeddings but slower startup
- Configure via `--spacy-model {sm|md|lg}` or in `pyproject.toml`

### External Dependencies
- **NLTK data**: Auto-downloads on first use via `setup_models.py`
- **Poppler**: Required only for PDF→PNG conversion (optional)
  - macOS: `brew install poppler`
  - Ubuntu: `apt-get install poppler-utils`
- **Datamuse API**: Rate limits apply, hence aggressive caching

### Environment Variable Prefix
- Configuration from environment uses `PP_` prefix (not `GP_`)
- Example: `export PP_SEED=42`, `export PP_VERBOSE=true`

### Seed Management
- Seeds are global state affecting all randomness
- Set once per session, not per generator
- Reset between test runs to ensure isolation

## Module Quick Reference

### Configuration & Infrastructure
- `config.py` - Multi-source configuration with Pydantic validation
- `cache.py` - Disk-based API response caching with retry logic
- `seed_manager.py` - Global seed management for reproducibility
- `logger.py` - Profiling and timing infrastructure

### Word Generation & Vocabulary
- `lexigen.py` - Unified API for word relationships (rhymes, similarity, etc.)
- `vocabulary.py` - Curated word lists by theme (2280 lines)
- `pos_vocabulary.py` - POS-tagged words organized by syllable count
- `word_validator.py` - Quality filtering (profanity, proper nouns, frequency)

### Template & Form Generation
- `grammatical_templates.py` - POS pattern templates for coherent phrases
- `forms.py` - Haiku/tanka/senryu generation with syllable validation
- `decomposer.py` - Markov chains and text processing

### Ideation Generators
- `line_seeds.py` - Opening lines and pivotal fragments
- `metaphor_generator.py` - 6 types of metaphors using classic literature
- `idea_generator.py` - Creative seeds across 10 categories
- `causal_poetry.py` - Resonant fragment mining with 26 patterns
- `six_degrees.py` - Semantic convergence path finding
- `semantic_geodesic.py` - Semantic path finding through meaning-space (linear/Bezier/shortest)
- `conceptual_cloud.py` - Multi-dimensional word associations (6 cluster types: semantic, contextual, opposite, phonetic, imagery, rare)

### Visual & Transformation
- `pdf.py` - Visual concrete poem generators (6 types)
- `poemgen.py` - Terminal-based visual puzzle poems
- `poem_transformer.py` - Ship of Theseus gradual transformation

### Analysis & Utilities
- `corpus_analyzer.py` - Stylistic analysis of personal poetry collections
- `finders.py` - Equidistant word finder (Levenshtein distance)
- `document_library.py` - Project Gutenberg retrieval with anti-repetition

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
1. **Lint** - Ruff checks on all Python code
2. **Type Check** - mypy (informational, non-blocking)
3. **Test Matrix** - pytest on Python 3.9-3.12 across Ubuntu/macOS
4. **Example Generation** - Creates sample poems with fixed seeds
5. **Docker Build** - Validates containerized deployment

All checks run on push to master and on pull requests.

## Development Workflow

1. Make changes to code
2. Run tests: `pytest tests/`
3. Check linting: `ruff check poetryplayground/`
4. Format code: `ruff format poetryplayground/`
5. Type check (optional): `mypy poetryplayground/ --ignore-missing-imports`
6. Commit (pre-commit hooks run automatically if installed)

## Project Attribution

This is a personal fork. For production use, refer users to the original [generativepoetry-py](https://github.com/sndwch/generativepoetry-py) project which is actively maintained and feature-complete.

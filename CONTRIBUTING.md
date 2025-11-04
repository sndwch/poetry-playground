# Contributing to Generative Poetry

Thank you for your interest in contributing to Generative Poetry! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Quick Setup (One Command)

Get started with a single command that sets up everything you need:

```bash
pip install -e ".[dev]" && python -m generativepoetry.setup_models && pytest
```

This command:
1. Installs the package in editable mode with all development dependencies
2. Downloads required NLTK data and spaCy models
3. Runs the test suite to verify everything works

## Detailed Setup

If you prefer a step-by-step approach:

### 1. Clone the Repository

```bash
git clone https://github.com/sndwch/generativepoetry-py.git
cd generativepoetry-py
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- The `generativepoetry` package in editable mode
- All runtime dependencies (nltk, spacy, datamuse, etc.)
- Development tools (pytest, ruff, mypy, black, pre-commit)

### 3. Download Required Models

```bash
python -m generativepoetry.setup_models
```

Or using the CLI:

```bash
generative-poetry-cli --setup
```

This downloads:
- NLTK data: punkt, words, brown, wordnet, stopwords
- spaCy model: en_core_web_sm (~40MB)

### 4. Run Tests

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=generativepoetry --cov-report=html
```

## Development Workflow

### Code Style and Linting

This project uses:
- **black** for code formatting
- **ruff** for linting
- **mypy** for type checking
- **pre-commit** hooks for automatic checks

Install pre-commit hooks:

```bash
pre-commit install
```

Run linting manually:

```bash
ruff check generativepoetry/
```

Format code:

```bash
black generativepoetry/ tests/
```

Type check:

```bash
mypy generativepoetry/
```

### Running Tests

Run all tests:

```bash
pytest
```

Run specific test file:

```bash
pytest tests/test_cache.py
```

Run tests with verbose output:

```bash
pytest -v
```

Run tests and show print statements:

```bash
pytest -s
```

Run tests that match a pattern:

```bash
pytest -k "test_cache"
```

### Testing with Different Configurations

Test with profiling enabled:

```bash
generative-poetry-cli --profile
```

Test in dry-run mode:

```bash
generative-poetry-cli --dry-run
```

Test with different spaCy models:

```bash
generative-poetry-cli --spacy-model sm  # Small (fast, 13MB)
generative-poetry-cli --spacy-model md  # Medium (balanced, 40MB)
generative-poetry-cli --spacy-model lg  # Large (accurate, 560MB)
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Or for bug fixes:

```bash
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clear, documented code
- Follow existing code style and patterns
- Add type hints to function signatures
- Write docstrings for public functions and classes

### 3. Add Tests

- Add tests for new functionality in the appropriate `tests/test_*.py` file
- Ensure all tests pass: `pytest`
- Aim for high test coverage for new code

### 4. Update Documentation

- Update docstrings if you change function signatures
- Update README.md if you add user-facing features
- Update this CONTRIBUTING.md if you change the development workflow

### 5. Commit Your Changes

We use conventional commit messages:

```bash
git commit -m "feat: add new poem generation algorithm"
git commit -m "fix: resolve cache invalidation issue"
git commit -m "docs: update API documentation"
git commit -m "test: add tests for profiling module"
git commit -m "refactor: simplify cache key generation"
```

Commit prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring without behavior changes
- `perf:` - Performance improvements
- `chore:` - Build process or auxiliary tool changes

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to any related issues
- Screenshots/examples if applicable

## Project Structure

```
generativepoetry-py/
├── generativepoetry/          # Main package
│   ├── __init__.py
│   ├── cache.py               # API caching with diskcache
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration management
│   ├── lexigen.py             # Word generation via Datamuse/CMU Dict
│   ├── poemgen.py             # Core poem generation algorithms
│   ├── logger.py              # Logging and profiling
│   ├── pdf.py                 # PDF generation
│   ├── jolastic.py            # Markov chain word generation
│   ├── corpus_analyzer.py     # Personal corpus analysis
│   ├── metaphor_generator.py  # Metaphor generation
│   ├── line_seeds.py          # Line seed generation
│   ├── idea_generator.py      # Poetry idea generation
│   └── ...
├── tests/                     # Test suite
│   ├── test_cache.py
│   ├── test_config.py
│   ├── test_profiling.py
│   └── ...
├── docs/                      # Documentation
├── example_images/            # Example outputs
├── pyproject.toml             # Project configuration and dependencies
├── README.md                  # User documentation
└── CONTRIBUTING.md            # This file
```

## Key Modules

### Core Modules

- **lexigen.py**: Word generation and manipulation using Datamuse API and CMU pronouncing dictionary
- **poemgen.py**: Main poem generation algorithms (Markov chains, visual poems)
- **cache.py**: Persistent caching layer with exponential backoff retry logic
- **config.py**: Configuration system with multiple sources (CLI, environment, files)

### Ideation Tools

- **line_seeds.py**: Generate opening lines, pivotal fragments, sonic patterns
- **metaphor_generator.py**: AI-assisted metaphor creation
- **idea_generator.py**: Mine classic literature for creative prompts
- **corpus_analyzer.py**: Analyze personal poetry collections for style insights

### Infrastructure

- **logger.py**: Structured logging and performance profiling
- **cli.py**: Interactive menu-driven CLI
- **pdf.py**: PDF generation with reportlab
- **setup_models.py**: Download and install NLTK/spaCy resources

## Testing Philosophy

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test module interactions
- **Deterministic tests**: Use seeds for reproducible results
- **Snapshot tests**: Verify PDF/SVG structure (not pixel-perfect)
- **Cache tests**: Validate cache hit/miss behavior
- **CI compatibility**: Tests must run without network/interactive input

## Performance Profiling

Enable profiling to measure performance:

```bash
generative-poetry-cli --profile
```

This tracks:
- Timing for poem generation stages
- Cache hit/miss rates
- API call counts
- Detailed performance reports

Add profiling to new code using decorators:

```python
from generativepoetry.logger import timed

@timed("operation_name")
def my_function():
    # Your code here
    pass
```

## Common Tasks

### Adding a New Configuration Option

1. Add field to `Config` class in `config.py`
2. Add CLI argument in `cli.py`
3. Add environment variable mapping in `Config.from_env()`
4. Add tests in `tests/test_config.py`
5. Update documentation

### Adding a New API Endpoint

1. Create cached wrapper function using `@cached_api_call` decorator
2. Add error handling and retry logic
3. Add profiling for API calls
4. Add tests with mocking
5. Document the endpoint

### Adding a New Poem Generation Algorithm

1. Add method to `PoemGenerator` class in `poemgen.py`
2. Add timing decorator: `@timed("poem_generation.algorithm_name")`
3. Add corresponding PDF generator in `pdf.py` if needed
4. Add CLI menu item in `cli.py`
5. Add tests in `tests/test_generativepoetry.py`
6. Add example to `example_images/`

## Getting Help

- **Issues**: Check [GitHub Issues](https://github.com/sndwch/generativepoetry-py/issues) for known problems
- **Discussions**: Start a [GitHub Discussion](https://github.com/sndwch/generativepoetry-py/discussions) for questions
- **Documentation**: See [README.md](README.md) for user documentation

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing to Generative Poetry, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to Generative Poetry! Your efforts help make procedural poetry generation accessible and powerful for everyone.

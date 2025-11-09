# Poetry Playground

A Python playground for experimenting with generative poetry techniques and computational creativity.

## About This Project

This is a personal experimental fork exploring various approaches to generative poetry and concrete poetry generation. It's a sandbox for trying out different linguistic algorithms, visual poetry layouts, and creative text manipulation techniques.

### Attribution

This project is forked from [generativepoetry-py](https://github.com/sndwch/generativepoetry-py) by [@sndwch](https://github.com/sndwch), which itself was originally created by [Corey Bobco](https://github.com/coreybobco).

The original generativepoetry-py library is a comprehensive and well-architected tool for procedurally generating visual concrete poems. That project has evolved into a feature-rich library with:
- Multiple visual poem generation procedures
- Creative writing and ideation tools
- Poetry corpus analysis capabilities
- A modern configuration system
- Both CLI and TUI interfaces
- Extensive documentation and examples

**This fork (poetry-playground) is a personal experiment** and diverges from the original project's goals. For production use or a full-featured generative poetry toolkit, please use the original [generativepoetry-py](https://github.com/sndwch/generativepoetry-py) project instead.

## Quick Start

### Installation

```bash
# Clone this repository
git clone https://github.com/jparker/poetry-playground.git
cd poetry-playground
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('words'); nltk.download('brown'); nltk.download('wordnet')"

# Download spaCy English model
python -m spacy download en_core_web_sm

# Install the package
pip install -e .
```

### Running the CLI

```bash
# Launch the CLI menu
poetry-playground

# Or use the TUI (Terminal User Interface)
poetry-playground-tui
```

## Features

This playground includes tools for:
- **Visual Poetry Generation** - Futurist poems, concrete poems, typographic experiments
- **Creative Ideation** - Line seeds, metaphor generation, idea mining from classic literature
- **Text Analysis** - Corpus analysis, word relationships, semantic exploration
- **Experimental Transformations** - Markov chains, cut-ups, systematic poem transformation

See the original [generativepoetry-py documentation](https://github.com/sndwch/generativepoetry-py) for detailed feature descriptions, as many features are inherited from that project.

## Requirements

- Python 3.8+
- System dependencies (optional):
  - `poppler-utils` - For PDF to PNG conversion
  - `hunspell` - For enhanced spell checking

### Installing System Dependencies

#### macOS
```bash
brew install poppler hunspell
```

#### Ubuntu/Debian
```bash
sudo apt-get install poppler-utils hunspell hunspell-en-us libhunspell-dev
```

## Configuration

This project inherits the configuration system from generativepoetry-py. Configuration can be set via:
1. CLI flags
2. YAML config file (`--config`)
3. `pyproject.toml` (`[tool.poetryplayground]` section)
4. Environment variables (with `PP_` prefix)

See `pyproject.toml` for available configuration options.

## Development

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check poetryplayground/

# Run type checking
mypy poetryplayground/ --ignore-missing-imports

# Install pre-commit hooks
pre-commit install
```

## Acknowledgements

### Direct Attribution
- **[@sndwch](https://github.com/sndwch)** - Author of the generativepoetry-py fork this project is based on
- **[Corey Bobco](https://github.com/coreybobco)** - Original creator of generativepoetry-py

### Technical & Creative Influences
(Inherited from generativepoetry-py)
- **Allison Parrish** - Creator of the pronouncing library
- **Leonard Richardson** - olipy library for generative text
- **Project Gutenberg** - Public domain literary corpus
- **Datamuse API** - Word relationship data
- **CMU Pronouncing Dictionary** - Phonetic and rhyme data

### Literary & Artistic Influences
- Oulipo (Queneau, Calvino, Perec)
- Concrete Poetry Movement (Gomringer, de Campos)
- F.T. Marinetti's Futurist manifestos
- William S. Burroughs' cut-up method
- Fluxus instruction-based art
- Language Poetry constraint systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- **Original Project**: [generativepoetry-py by sndwch](https://github.com/sndwch/generativepoetry-py)
- **Original Original Project**: [generativepoetry-py by coreybobco](https://github.com/coreybobco/generativepoetry-py)
- **This Playground**: [poetry-playground](https://github.com/jparker/poetry-playground)

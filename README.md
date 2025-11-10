# Poetry Playground

A Python playground for experimenting with generative poetry techniques and computational creativity.

## About This Project

This is a personal experimental fork exploring various approaches to generative poetry and concrete poetry generation. It's a sandbox for trying out different linguistic algorithms, visual poetry layouts, and creative text manipulation techniques.

### Attribution

This project began as a fork of [generativepoetry-py](https://github.com/coreybobco/generativepoetry-py) by [@sndwch](https://github.com/coreybobco).

**This repo (poetry-playground) diverges from the original project's goals.**

This project has evolved to focus on intelligent ideation and creative scaffolding. It's a suite of "smart" tools designed to be a true creative partner, helping you explore thematic connections, analyze your own style, and build the "bones" of a poem.

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

## Core Features

This playground is now organized around a few powerful, interconnected concepts:

### 1. High-Level Creative Strategies 🧠

* **Poem Scaffold Generator**: The "conductor" for your ideas. Give it a start concept ("rust") and an end concept ("forgiveness"), and it orchestrates all the other smart generators to build a complete, multi-stanza thematic scaffold for your poem.
* **Bridge Two Concepts**: A focused strategy that finds the semantic path, key metaphors, and conceptual cloud to bridge two words.

### 2. Personalized Generation & Analysis 🔬

* **Personal Corpus Analyzer**: Ingests your own body of work to create a StyleFingerprint, analyzing your unique vocabulary (using Affinity Scoring), syntactic habits, and thematic tendencies.
* **Personalized Line Seeds**: Uses your StyleFingerprint to generate new line seeds, pivots, and images in your own voice, matching your style based on a 70/30 quality-vs-style score.

### 3. Conceptual & Semantic Tools 🗺️

* **Definitional Finder**: An incredibly powerful "lateral search" tool. It finds words by searching the definitions of other words in the dictionary (e.g., searching "bean" finds "chocolate," "casserole," and "Carver").
* **Conceptual Cloud**: A "poet's radar" that maps the 6-dimensional neighborhood of a word: Semantic, Contextual, Opposite, Phonetic, Imagery, and Rare (your "strange orbit" words).
* **Semantic Pathfinders**: Tools to find the "semantic journey" between two words, with methods like "bezier" curves for more creative, non-linear paths.
* **Equidistant Finder**: The "smart" version that finds words orthographically or phonetically between two anchors, now ranked by creative quality, not just frequency (so you get "fog" instead of "the").

### 4. Literature Mining & Transformation 📚

* **"Smart" Metaphor Generator**: Mines Project Gutenberg for poetic metaphors. This uses a robust pipeline (LoCC filtering, text cleaning, POS-first analysis, NER filtering, and semantic distance checks) to find actual poetic metaphors, not just literal chapter headings.
* **Poetry Idea Generator**: Mines classic literature for 10 different categories of creative seeds (vivid imagery, philosophical fragments, etc.).
* **Ship of Theseus Transformer**: Gradually transforms a poem by replacing words one by one, allowing you to see it evolve.

### A Note on Visual Poetry

This fork originally included several "chaotic" visual poetry generators (pdf.py, poemgen.py). As the project's focus has shifted entirely to intelligent, semantic, and structural generation, these legacy tools are no longer the primary goal. They are still present, but the core of the playground is now the "smart" engine for creative ideation.

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

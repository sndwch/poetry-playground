# Generative Poetry

A Python library for procedurally generating visual concrete poems

[![PyPI version](https://badge.fury.io/py/generativepoetry.svg)](https://badge.fury.io/py/generativepoetry)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

### Installation

```bash
# Install from PyPI
pip install generativepoetry

# Or install from source
git clone https://github.com/sndwch/generativepoetry-py.git
cd generativepoetry-py
pip install -e .
```

### Running the Interactive CLI

```bash
generative-poetry-cli
```

This launches an interactive menu where you can:
- Generate Futurist poems (PDF/Image)
- Create Stochastic Jolastic (Markov) poems
- Make Chaotic Concrete poems
- Generate Character Soup poems
- Create Stop Word Soup poems
- Run Visual Puzzle poems in terminal
- Check system dependencies

## What is this?

This library provides constraint-based procedures for stochastically generating [concrete poetry](https://en.wikipedia.org/wiki/Concrete_poetry) (visual poetry) as PDFs or terminal output. The procedures are interactive and designed to be reused as you learn to exploit their rules.

When you provide input words, algorithms find related words:
- Phonetically similar (rhymes, near-rhymes)
- Semantically related (similar meaning, contextual links)
- Frequently co-occurring words

These are combined using various connectors depending on the procedure: random conjunctions, mathematical symbols, variable spacing, etc.

## Requirements

- **Python 3.8+** (tested on 3.12)
- **System Dependencies (optional but recommended):**
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

#### Windows
Windows users should use WSL2 with Ubuntu, or use the library without these optional dependencies.

## Features

### Word Generation & Relations

The `lexigen` module provides various word relationship functions:

```python
from generativepoetry.lexigen import *

# Get rhymes
rhymes('cool')                    # All rhymes
rhymes('cool', sample_size=6)     # 6 random rhymes
rhyme('cool')                      # 1 random rhyme

# Similar sounding words (not quite rhymes)
similar_sounding_words('cool')
similar_sounding_word('cool')

# Similar meaning words
similar_meaning_words('vampire')
similar_meaning_word('vampire')

# Contextually linked words (often appear together)
contextually_linked_words('metamorphosis')  # Returns words like 'Kafka'

# Frequently following words
frequently_following_words('the')

# Phonetically related words (rhymes + similar sounding)
phonetically_related_words('slimy')
```

### Text Decomposition & Transformation

The `decomposer` module can transform existing texts:

```python
from generativepoetry.decomposer import *

# Get text from Project Gutenberg (using gutenbergpy)
text = get_gutenberg_document(11)  # Alice in Wonderland

# Parse and sample text
parsed = ParsedText(text)
parsed.random_sentence()
parsed.random_paragraph(minimum_sentences=5)

# Markov chain generation
output = markov(text, ngram_size=2, num_output_sentences=5)

# Cut-up method (Burroughs-style)
cutouts = cutup(text, min_cutout_words=3, max_cutout_words=7)
```

## Poem Types

### Futurist Poems
Inspired by F.T. Marinetti's 1912 manifesto, connects phonetically related words with mathematical operators.

### Stochastic Jolastic (Markov)
Complex generation using phonetic relations, meaning transformations, and forced rhyme schemes. Creates Joyce-like wordplay.

### Chaotic Concrete Poems
Abstract spatial arrangements focusing on visual rather than semantic relationships.

### Character Soup & Stop Word Soup
Pure visual experiments with random character or stop word placement.

## Configuration

The library now includes enhanced word validation to filter out:
- Proper nouns and names
- Non-English words
- Very rare or archaic words
- Abbreviations and acronyms

Word frequency thresholds and exclusion lists can be customized in `generativepoetry/word_validator.py`.

## Development

### Running Tests
```bash
pytest tests/
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Building
```bash
pip install build
python -m build
```

## Acknowledgements

This project was inspired by:
- **Oulipo** - Workshop of potential literature (Queneau, Calvino, Perec)
- **Allison Parrish** - Creator of the pronouncing library
- **Leonard Richardson's olipy** - Similar generative text experiments
- **Marjorie Perloff** - Work on concrete poetry and "Unoriginal Genius"

## API Integrations

- **[Datamuse API](https://www.datamuse.com/api/)** - Word relationships and similarities
- **[CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)** - Rhyme detection
- **[Project Gutenberg](https://www.gutenberg.org/)** - Public domain texts (via gutenbergpy)

## Troubleshooting

### Check System Dependencies
Run `generative-poetry-cli` and select "Check System Dependencies" to verify your setup.

### Common Issues

**pkg_resources deprecation warning**: This is suppressed automatically but doesn't affect functionality.

**PDF to PNG conversion fails**: Install poppler-utils for your system.

**Poor word quality**: The word validator filters aggressively. Adjust thresholds in `word_validator.py` if needed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [Original Repository](https://github.com/coreybobco/generativepoetry-py)
- [PyPI Package](https://pypi.org/project/generativepoetry/)
- [Documentation](https://generativepoetry.readthedocs.io)
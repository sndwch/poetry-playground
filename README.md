# Generative Poetry

A comprehensive Python library for procedurally generating visual concrete poems and exploring literary creativity through computational methods.

[![PyPI version](https://badge.fury.io/py/generativepoetry.svg)](https://badge.fury.io/py/generativepoetry)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Quick Start

### Installation

```bash
# Install from source (recommended)
git clone https://github.com/sndwch/generativepoetry-py.git
cd generativepoetry-py
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('words'); nltk.download('brown'); nltk.download('wordnet')"

# Download spaCy English model
python -m spacy download en_core_web_sm

# Install the package
pip install -e .
```

### Running the Interactive CLI

```bash
generative-poetry-cli
```

This launches a comprehensive interactive menu with 13 different creative tools:

#### 🎨 **Visual Poem Generation**
- **Futurist Poems** - Marinetti-inspired mathematical word connections (PDF/Image)
- **Stochastic Jolastic (Markov) Poems** - Joyce-like wordplay with forced rhyme schemes (Image)
- **Chaotic Concrete Poems** - Abstract spatial arrangements (Image)
- **Character Soup & Stop Word Soup** - Pure visual typographic experiments (Image)
- **Visual Puzzle Poems** - Interactive terminal-based concrete poetry

#### 💡 **Creative Ideation & Writing Tools**
- **🌱 Line Seeds Generator** - Opening lines and pivotal fragments to spark new poems
- **🔮 Metaphor Generator** - AI-assisted metaphor creation with adaptive scaling
- **💡 Poetry Idea Generator** - Mine classic literature for creative seeds to beat writer's block
- **🔍 Resonant Fragment Miner** - Extract poetic sentence fragments from Project Gutenberg texts
- **🔗 Six Degrees Word Convergence** - Explore unexpected connections between concepts

#### 📊 **Analysis & Transformation Tools**
- **📊 Poetry Corpus Analyzer** - Analyze patterns in your personal poetry collection
- **🚢 Ship of Theseus Transformer** - Gradually transform existing poems into new works
- **🔧 System Dependencies Checker** - Verify your installation and troubleshoot issues

## What is this?

This library provides constraint-based procedures for stochastically generating [concrete poetry](https://en.wikipedia.org/wiki/Concrete_poetry) (visual poetry) and exploring literary creativity through computational methods. The tools are interactive and designed to be reused as you develop your poetic practice.

## Core Features & Use Cases

### 🎨 Visual Poetry Generation
Perfect for creating concrete/visual poems and experimental typography:

**Futurist Poems** - Generate Marinetti-style manifestos connecting words through mathematical operators and phonetic relationships. Great for creating bold, energetic visual statements.

**Stochastic Jolastic (Markov)** - Complex generation using phonetic relations, meaning transformations, and forced rhyme schemes. Creates Joyce-like wordplay and linguistic experiments.

**Chaotic Concrete** - Abstract spatial arrangements focusing on visual rather than semantic relationships. Perfect for pure typographic art.

**Character/Stop Word Soup** - Minimalist visual experiments with strategic placement of characters or common words.

### 💡 Creative Writing & Ideation
Breakthrough writer's block and discover new directions:

**🌱 Line Seeds Generator** - Get opening lines, pivotal fragments, and closing thoughts tailored to your input words. Perfect for starting new poems or finding fresh angles.

**🔮 Metaphor Generator** - AI-powered metaphor creation that mines classic literature for unexpected comparisons. Uses adaptive scaling to ensure high-quality results.

**💡 Poetry Idea Generator** - Extract creative seeds from Project Gutenberg's classic literature across 10 categories:
- Emotional moments & character situations
- Vivid imagery & setting descriptions
- Philosophical fragments & dialogue sparks
- Sensory details & conflict scenarios
- Opening lines & metaphysical concepts

**🔍 Resonant Fragment Miner** - Discover evocative sentence fragments from classic literature using 26 specialized patterns across 5 categories (causality, temporal, universal, singular, modal). Request 10-200 fragments for deep literary exploration.

**🔗 Six Degrees Word Convergence** - Explore unexpected pathways between any two concepts using semantic relationships. Discover surprising connections that can inspire new poems.

### 📊 Analysis & Transformation
Work with existing poetry and understand patterns:

**📊 Poetry Corpus Analyzer** - Upload your personal poetry collection to analyze themes, word frequency, emotional patterns, and stylistic trends. Gain insights into your creative evolution.

**🚢 Ship of Theseus Transformer** - Gradually transform existing poems by systematically replacing words while maintaining structure. Perfect for creating variations or exploring how meaning shifts with word choice.

### 🔧 Technical Foundation
When you provide input words, the algorithms find related words through:
- **Phonetic similarity** (rhymes, near-rhymes, similar sounds)
- **Semantic relationships** (similar meaning, contextual links, co-occurring words)
- **Linguistic patterns** (frequently following words, rare related terms)

These are combined using various strategies: mathematical symbols, spatial arrangements, constraint-based generation, and adaptive scaling to maintain quality while increasing variety.

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

## Example Workflows

### For Writers Seeking Inspiration
1. **Start with Fragment Mining**: Use "🔍 Resonant Fragment Miner" to extract 50-100 evocative sentence fragments from classic literature
2. **Generate Line Seeds**: Use "🌱 Line Seeds Generator" with key words from your chosen fragments
3. **Create Metaphors**: Use "🔮 Metaphor Generator" to develop unexpected comparisons
4. **Find Ideas**: Use "💡 Poetry Idea Generator" to discover 10-20 creative seeds across different categories
5. **Explore Connections**: Use "🔗 Six Degrees" to find surprising pathways between concepts

### For Visual Artists & Typographers
1. **Futurist Style**: Generate bold manifestos with mathematical word connections
2. **Concrete Experiments**: Create abstract spatial arrangements with "Chaotic Concrete"
3. **Minimalist Compositions**: Explore "Character Soup" and "Stop Word Soup" for pure typography
4. **Interactive Pieces**: Use "Visual Puzzle Poems" for terminal-based concrete poetry

### For Poetry Analysis & Transformation
1. **Corpus Analysis**: Upload your poetry collection to discover patterns and evolution
2. **Systematic Transformation**: Use "Ship of Theseus" to create variations of existing poems
3. **Literary Exploration**: Mine classic texts for fragments that resonate with your style

### For Computational Poetry Researchers
1. **Study Word Relationships**: Explore the lexigen API for phonetic and semantic connections
2. **Text Processing**: Use decomposer tools for Markov chains and cut-up methods
3. **Pattern Recognition**: Analyze how different constraint systems produce varying outputs

## API Reference

### Word Generation & Relations (`lexigen` module)

```python
from generativepoetry.lexigen import *

# Phonetic relationships
rhymes('cool')                           # All rhymes
rhymes('cool', sample_size=6)           # 6 random rhymes
similar_sounding_words('cool')          # Near-rhymes
phonetically_related_words('slimy')    # Combined phonetic relations

# Semantic relationships
similar_meaning_words('vampire')        # Synonyms and related concepts
contextually_linked_words('metamorphosis')  # Co-occurring words
frequently_following_words('the')       # Common word sequences
related_rare_words('mystery')          # Uncommon but related terms
```

### Text Processing & Generation (`decomposer` module)

```python
from generativepoetry.decomposer import *
from generativepoetry.document_library import get_diverse_gutenberg_documents

# Advanced document retrieval with diversity guarantees
texts = get_diverse_gutenberg_documents(count=5, min_length=3000)

# Parse and analyze text
parsed = ParsedText(text)
parsed.random_sentence()
parsed.random_paragraph(minimum_sentences=5)

# Generative methods
output = markov(text, ngram_size=2, num_output_sentences=5)
cutouts = cutup(text, min_cutout_words=3, max_cutout_words=7)
```

### Creative Generation Tools

```python
from generativepoetry.idea_generator import PoetryIdeaGenerator
from generativepoetry.metaphor_generator import MetaphorGenerator
from generativepoetry.causal_poetry import ResonantFragmentMiner
from generativepoetry.six_degrees import SixDegreesExplorer

# Generate creative ideas from literature
idea_gen = PoetryIdeaGenerator()
ideas = idea_gen.generate_ideas(num_ideas=20)

# Create metaphors with adaptive scaling
metaphor_gen = MetaphorGenerator()
metaphors = metaphor_gen.generate_metaphor_batch(['ocean', 'memory'], count=10)

# Mine resonant fragments from classic texts
fragment_miner = ResonantFragmentMiner()
fragments = fragment_miner.mine_fragments(target_count=100, num_texts=10)

# Explore word convergence paths
explorer = SixDegreesExplorer()
path = explorer.find_convergence_path('love', 'mathematics')
```

## Advanced Configuration

### Centralized Settings
The library uses centralized configuration classes for consistent behavior:

```python
from generativepoetry.config import DocumentConfig, QualityConfig, PerformanceConfig

# Document retrieval settings
DocumentConfig.MIN_LENGTH_IDEAS = 2000      # Ideas need shorter texts
DocumentConfig.MIN_LENGTH_FRAGMENTS = 5000  # Fragments need longer texts
DocumentConfig.MIN_LENGTH_METAPHORS = 3000  # Metaphors need medium texts

# Quality thresholds
QualityConfig.FRAGMENT_QUALITY_THRESHOLD = 0.65  # High quality fragments only
QualityConfig.MAX_FRAGMENTS_PER_TEXT = 10        # Limit extraction per text

# Performance optimization
PerformanceConfig.MAX_DOCUMENT_CACHE = 20        # Cache frequently used texts
PerformanceConfig.API_DELAY_SECONDS = 0.2        # Rate limiting for APIs
```

### Adaptive Scaling Principle
All text processing modules follow the core principle: **"Never lower standards, always scale documents"**

Instead of compromising quality when yield is low, the system automatically:
- Retrieves additional diverse documents from Project Gutenberg
- Maintains quality thresholds (0.65+ for fragments, strict validation for ideas)
- Uses anti-repetition tracking to ensure document diversity
- Scales batch sizes intelligently based on remaining needs

### Word Quality Filtering
Enhanced word validation filters out:
- Proper nouns and names
- Non-English words
- Very rare or archaic words (< 1 in 1M frequency)
- Abbreviations and acronyms
- Technical jargon and specialist terms

Customize filtering in `generativepoetry/word_validator.py` and `generativepoetry/vocabulary.py`.

## Recent Improvements & Architecture

### Version 2024.11 Major Updates
- **Adaptive Scaling System**: Intelligent document retrieval that maintains quality while scaling quantity
- **Centralized Document Library**: Anti-repetition tracking ensures diverse text sources across all modules
- **Configuration Management**: Centralized constants and thresholds for consistent behavior
- **Enhanced Fragment Mining**: 26 specialized patterns across 5 categories with quality scoring
- **Comprehensive DRY Refactoring**: Eliminated duplicate code and consolidated shared functionality
- **Vocabulary Centralization**: 600+ evocative verbs and 400+ atmospheric nouns from extensive research

### Development

This is primarily a personal creative tool. For development:

```bash
# Clone and install in development mode
git clone https://github.com/sndwch/generativepoetry-py.git
cd generativepoetry-py
pip install -e .

# Run tests
pytest tests/

# Check dependencies
generative-poetry-cli  # Select option 13 to verify setup
```

## Philosophy & Acknowledgements

### Core Principles
1. **"Never lower standards, always scale documents"** - Maintain quality by increasing variety, not compromising filters
2. **Constraint-based creativity** - Use systematic limitations to spark unexpected discoveries
3. **Iterative exploration** - Tools designed for repeated use as you develop your poetic practice
4. **Computational augmentation** - Technology as creative partner, not replacement

### Inspiration & Influences
- **Oulipo** - Workshop of potential literature (Queneau, Calvino, Perec)
- **Concrete Poetry Movement** - Visual-textual synthesis (Gomringer, de Campos brothers)
- **Futurist Manifestos** - F.T. Marinetti's revolutionary typography and mathematical poetry
- **Cut-up Method** - William S. Burroughs' systematic textual intervention
- **Fluxus** - Instruction-based art and systematic chance operations
- **Language Poetry** - Constraint-based compositional methods
- **Digital Humanities** - Distant reading and computational text analysis

### Technical Acknowledgements
- **Allison Parrish** - Creator of the pronouncing library for phonetic relationships
- **Leonard Richardson's olipy** - Pioneering generative text experiments
- **Marjorie Perloff** - Scholarship on "Unoriginal Genius" and conceptual writing
- **Project Gutenberg** - Invaluable public domain literary corpus
- **Datamuse API** - Sophisticated word relationship data

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
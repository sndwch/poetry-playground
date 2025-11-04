# Installation Guide for Generative Poetry

## Prerequisites

### Python Requirements
- Python 3.8 or higher
- pip

### System Dependencies (Recommended)

#### For PDF to PNG conversion (optional but recommended)
The library can generate PNG images from PDFs if Poppler is installed:

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**Fedora/RHEL/CentOS:**
```bash
sudo yum install poppler-utils
```

**Windows:**
Download and install Poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/
Add the `bin` directory to your PATH.

#### For spellchecking (optional)
If you want to use the spellchecking features:

**macOS:**
```bash
brew install hunspell
pip install hunspell
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libhunspell-dev hunspell-en-us
pip install hunspell
```

## Installation

### From PyPI (when available)
```bash
pip install generativepoetry
```

### From Source (Development)
```bash
# Clone the repository
git clone https://github.com/coreybobco/generativepoetry-py.git
cd generativepoetry-py

# Install in editable mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[spellcheck]"
```

### Installing Required NLTK Data
After installation, you may need to download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### Installing spaCy Language Model
The library uses spaCy for NLP processing:

```bash
python -m spacy download en_core_web_sm
```

## Troubleshooting

### Font Issues
If you encounter font-related errors, the library will automatically fall back to built-in PDF fonts.

### PNG Generation
If PNG generation fails with "Poppler not installed" message:
- Install Poppler using the instructions above
- The PDF will still be generated even without Poppler

### Hunspell Issues
If you see warnings about hunspell:
- The library will work without it, but spellchecking features will be disabled
- To enable spellchecking, install hunspell system libraries first, then `pip install hunspell`

### Import Errors
If you get import errors after installation:
1. Make sure you're using Python 3.8 or higher
2. Try reinstalling with: `pip install --upgrade --force-reinstall generativepoetry`

## Verifying Installation

Test your installation:

```bash
# Run the CLI
generative-poetry-cli

# Or test in Python
python -c "from generativepoetry import *; print('Installation successful!')"
```

## Optional Features

The package has optional features that can be installed separately:

- **spellcheck**: Adds spellchecking capabilities
  ```bash
  pip install generativepoetry[spellcheck]
  ```

## Notes

- The package will work without Poppler, but PDF to PNG conversion will be disabled
- Spellchecking is optional and the package will work without hunspell
- On first run, the package may download additional data (NLTK corpora, spaCy models)

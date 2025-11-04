# Dockerfile for generativepoetry with pre-cached models
# This provides a complete, ready-to-use environment with all NLTK and spaCy data pre-downloaded

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY generativepoetry/ ./generativepoetry/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Download and cache NLTK data
RUN python -c "\
import nltk; \
import ssl; \
try: \
    _create_unverified_https_context = ssl._create_unverified_context; \
except AttributeError: \
    pass; \
else: \
    ssl._create_default_https_context = _create_unverified_https_context; \
for package in ['punkt', 'words', 'brown', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']: \
    try: \
        nltk.download(package, quiet=False); \
        print(f'Downloaded {package}'); \
    except Exception as e: \
        print(f'Warning: Failed to download {package}: {e}'); \
"

# Download and cache spaCy model
RUN python -m spacy download en_core_web_sm && \
    echo "spaCy model en_core_web_sm downloaded successfully"

# Run the automated setup to ensure everything is configured
RUN generative-poetry-cli --setup || true

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GP_ENABLE_CACHE=true

# Create directory for outputs
RUN mkdir -p /output

# Set default working directory to output
WORKDIR /output

# Default command shows help
CMD ["generative-poetry-cli", "--help"]

# Build instructions:
# docker build -t generativepoetry:latest .
#
# Usage examples:
# docker run --rm generativepoetry:latest generative-poetry-cli --list-procedures
# docker run --rm generativepoetry:latest generative-poetry-cli --version
# docker run --rm -v $(pwd)/output:/output generativepoetry:latest generative-poetry-cli --dry-run
#
# Interactive mode:
# docker run --rm -it generativepoetry:latest bash

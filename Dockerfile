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

# Download and cache NLTK data and spaCy models using the built-in setup
RUN generative-poetry-cli --setup

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

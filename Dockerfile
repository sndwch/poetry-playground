# Dockerfile for poetry-playground with pre-cached models
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
COPY poetryplayground/ ./poetryplayground/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Download and cache NLTK data and spaCy models using the built-in setup
RUN poetry-playground --setup

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PP_ENABLE_CACHE=true

# Create directory for outputs
RUN mkdir -p /output

# Set default working directory to output
WORKDIR /output

# Default command shows help
CMD ["poetry-playground", "--help"]

# Build instructions:
# docker build -t poetry-playground:latest .
#
# Usage examples:
# docker run --rm poetry-playground:latest poetry-playground --list-procedures
# docker run --rm poetry-playground:latest poetry-playground --version
# docker run --rm -v $(pwd)/output:/output poetry-playground:latest poetry-playground --dry-run
#
# Interactive mode:
# docker run --rm -it poetry-playground:latest bash

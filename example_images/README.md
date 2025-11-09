# Example Images & Commands

This directory contains example outputs from the generative poetry CLI. Below are copyable commands you can use to generate similar poems.

## Visual Poem Examples

### Markov Poem (markov_pdf.png)

Generate a Markov chain poem with forced rhyme schemes:

```bash
# Basic Markov poem
poetry-playground --procedure markov --input-words "ocean" "storm" "silence"

# With profiling enabled
poetry-playground --procedure markov --input-words "ocean" "storm" "silence" --profile

# With custom seed for reproducibility
poetry-playground --procedure markov --input-words "ocean" "storm" "silence" --seed 42

# Longer poem with more lines
poetry-playground --procedure markov --input-words "ocean" "storm" "silence" --num-lines 15
```

### Futurist Poem (futurist_pdf.png)

Generate Marinetti-inspired visual poems with mathematical operators:

```bash
# Basic futurist poem
poetry-playground --procedure futurist --input-words "velocity" "machine" "chaos"

# With profiling
poetry-playground --procedure futurist --input-words "velocity" "machine" "chaos" --profile

# With custom seed
poetry-playground --procedure futurist --input-words "velocity" "machine" "chaos" --seed 123
```

### Chaotic Concrete Poem (chaotic_concrete_pdf.png)

Generate abstract spatial arrangements:

```bash
# Basic chaotic concrete poem
poetry-playground --procedure chaotic-concrete --input-words "fragments" "space" "time"

# With profiling
poetry-playground --procedure chaotic-concrete --input-words "fragments" "space" "time" --profile

# With custom seed
poetry-playground --procedure chaotic-concrete --input-words "fragments" "space" "time" --seed 456
```

### Character Soup (character_soup_pdf.png)

Generate minimalist visual experiments with strategic character placement:

```bash
# Basic character soup
poetry-playground --procedure character-soup --input-words "abstract" "void" "echo"

# With profiling
poetry-playground --procedure character-soup --input-words "abstract" "void" "echo" --profile

# With custom seed
poetry-playground --procedure character-soup --input-words "abstract" "void" "echo" --seed 789
```

### Stop Word Soup (stopword_soup_pdf.png)

Generate visual experiments with common words:

```bash
# Basic stopword soup
poetry-playground --procedure stopword-soup --input-words "the" "and" "or"

# With profiling
poetry-playground --procedure stopword-soup --input-words "the" "and" "or" --profile

# With custom seed
poetry-playground --procedure stopword-soup --input-words "the" "and" "or" --seed 321
```

### Collage (collage.png)

Generate visual collage poems:

```bash
# Basic collage
poetry-playground --procedure collage --input-words "memory" "fragment" "dream"

# With profiling
poetry-playground --procedure collage --input-words "memory" "fragment" "dream" --profile

# With custom seed
poetry-playground --procedure collage --input-words "memory" "fragment" "dream" --seed 654
```

## Advanced Usage

### Dry Run Mode

Test generation without creating PDFs:

```bash
poetry-playground --dry-run
```

### Performance Profiling

Enable detailed performance profiling with timing and cache statistics:

```bash
poetry-playground --profile --procedure markov --input-words "test" "profiling"
```

### Custom spaCy Models

Use different spaCy models for better accuracy or performance:

```bash
# Small model (default, fast, 13MB)
poetry-playground --spacy-model sm --procedure markov --input-words "quick" "test"

# Medium model (balanced, 40MB)
poetry-playground --spacy-model md --procedure markov --input-words "balanced" "quality"

# Large model (most accurate, 560MB)
poetry-playground --spacy-model lg --procedure markov --input-words "high" "accuracy"
```

### Quiet Mode

Suppress progress messages:

```bash
poetry-playground --quiet --procedure markov --input-words "silent" "mode"
```

### List All Procedures

See all available poem generation procedures:

```bash
poetry-playground --list-procedures
```

## Reproducibility

All examples can be reproduced exactly by using the same seed value:

```bash
# Generate the same poem every time
poetry-playground --procedure markov --input-words "ocean" "storm" --seed 42
```

## Creative Ideation Tools

Beyond visual poem generation, try these creative tools:

```bash
# Generate opening lines and fragments
poetry-playground  # Select option 5: Line Seeds Generator

# Create AI-powered metaphors
poetry-playground  # Select option 6: Metaphor Generator

# Mine classic literature for ideas
poetry-playground  # Select option 7: Poetry Idea Generator

# Extract resonant fragments
poetry-playground  # Select option 8: Resonant Fragment Miner

# Explore word connections
poetry-playground  # Select option 9: Six Degrees Word Convergence

# Analyze your poetry collection
poetry-playground  # Select option 10: Poetry Corpus Analyzer

# Transform existing poems
poetry-playground  # Select option 11: Ship of Theseus Transformer
```

## Docker Usage

Run in Docker for a completely isolated environment:

```bash
# Build the Docker image
docker build -t poetryplayground:latest .

# List available procedures
docker run --rm poetryplayground:latest poetry-playground --list-procedures

# Generate a poem (output to stdout)
docker run --rm poetryplayground:latest poetry-playground --dry-run

# Save PDF output to local directory
docker run --rm -v $(pwd)/output:/output poetryplayground:latest \
  poetry-playground --procedure markov --input-words "docker" "poetry"

# Interactive mode
docker run --rm -it poetryplayground:latest bash
```

## Notes

- All PDF outputs are saved to the current directory by default
- Use `--seed` for reproducible results
- Use `--profile` to see detailed performance metrics
- Use `--dry-run` to test without creating files
- See `poetry-playground --help` for all available options

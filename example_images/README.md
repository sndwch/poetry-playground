# Example Images & Commands

This directory contains example outputs from the generative poetry CLI. Below are copyable commands you can use to generate similar poems.

## Visual Poem Examples

### Markov Poem (markov_pdf.png)

Generate a Markov chain poem with forced rhyme schemes:

```bash
# Basic Markov poem
generative-poetry-cli --procedure markov --input-words "ocean" "storm" "silence"

# With profiling enabled
generative-poetry-cli --procedure markov --input-words "ocean" "storm" "silence" --profile

# With custom seed for reproducibility
generative-poetry-cli --procedure markov --input-words "ocean" "storm" "silence" --seed 42

# Longer poem with more lines
generative-poetry-cli --procedure markov --input-words "ocean" "storm" "silence" --num-lines 15
```

### Futurist Poem (futurist_pdf.png)

Generate Marinetti-inspired visual poems with mathematical operators:

```bash
# Basic futurist poem
generative-poetry-cli --procedure futurist --input-words "velocity" "machine" "chaos"

# With profiling
generative-poetry-cli --procedure futurist --input-words "velocity" "machine" "chaos" --profile

# With custom seed
generative-poetry-cli --procedure futurist --input-words "velocity" "machine" "chaos" --seed 123
```

### Chaotic Concrete Poem (chaotic_concrete_pdf.png)

Generate abstract spatial arrangements:

```bash
# Basic chaotic concrete poem
generative-poetry-cli --procedure chaotic-concrete --input-words "fragments" "space" "time"

# With profiling
generative-poetry-cli --procedure chaotic-concrete --input-words "fragments" "space" "time" --profile

# With custom seed
generative-poetry-cli --procedure chaotic-concrete --input-words "fragments" "space" "time" --seed 456
```

### Character Soup (character_soup_pdf.png)

Generate minimalist visual experiments with strategic character placement:

```bash
# Basic character soup
generative-poetry-cli --procedure character-soup --input-words "abstract" "void" "echo"

# With profiling
generative-poetry-cli --procedure character-soup --input-words "abstract" "void" "echo" --profile

# With custom seed
generative-poetry-cli --procedure character-soup --input-words "abstract" "void" "echo" --seed 789
```

### Stop Word Soup (stopword_soup_pdf.png)

Generate visual experiments with common words:

```bash
# Basic stopword soup
generative-poetry-cli --procedure stopword-soup --input-words "the" "and" "or"

# With profiling
generative-poetry-cli --procedure stopword-soup --input-words "the" "and" "or" --profile

# With custom seed
generative-poetry-cli --procedure stopword-soup --input-words "the" "and" "or" --seed 321
```

### Collage (collage.png)

Generate visual collage poems:

```bash
# Basic collage
generative-poetry-cli --procedure collage --input-words "memory" "fragment" "dream"

# With profiling
generative-poetry-cli --procedure collage --input-words "memory" "fragment" "dream" --profile

# With custom seed
generative-poetry-cli --procedure collage --input-words "memory" "fragment" "dream" --seed 654
```

## Advanced Usage

### Dry Run Mode

Test generation without creating PDFs:

```bash
generative-poetry-cli --dry-run
```

### Performance Profiling

Enable detailed performance profiling with timing and cache statistics:

```bash
generative-poetry-cli --profile --procedure markov --input-words "test" "profiling"
```

### Custom spaCy Models

Use different spaCy models for better accuracy or performance:

```bash
# Small model (default, fast, 13MB)
generative-poetry-cli --spacy-model sm --procedure markov --input-words "quick" "test"

# Medium model (balanced, 40MB)
generative-poetry-cli --spacy-model md --procedure markov --input-words "balanced" "quality"

# Large model (most accurate, 560MB)
generative-poetry-cli --spacy-model lg --procedure markov --input-words "high" "accuracy"
```

### Quiet Mode

Suppress progress messages:

```bash
generative-poetry-cli --quiet --procedure markov --input-words "silent" "mode"
```

### List All Procedures

See all available poem generation procedures:

```bash
generative-poetry-cli --list-procedures
```

## Reproducibility

All examples can be reproduced exactly by using the same seed value:

```bash
# Generate the same poem every time
generative-poetry-cli --procedure markov --input-words "ocean" "storm" --seed 42
```

## Creative Ideation Tools

Beyond visual poem generation, try these creative tools:

```bash
# Generate opening lines and fragments
generative-poetry-cli  # Select option 5: Line Seeds Generator

# Create AI-powered metaphors
generative-poetry-cli  # Select option 6: Metaphor Generator

# Mine classic literature for ideas
generative-poetry-cli  # Select option 7: Poetry Idea Generator

# Extract resonant fragments
generative-poetry-cli  # Select option 8: Resonant Fragment Miner

# Explore word connections
generative-poetry-cli  # Select option 9: Six Degrees Word Convergence

# Analyze your poetry collection
generative-poetry-cli  # Select option 10: Poetry Corpus Analyzer

# Transform existing poems
generative-poetry-cli  # Select option 11: Ship of Theseus Transformer
```

## Docker Usage

Run in Docker for a completely isolated environment:

```bash
# Build the Docker image
docker build -t generativepoetry:latest .

# List available procedures
docker run --rm generativepoetry:latest generative-poetry-cli --list-procedures

# Generate a poem (output to stdout)
docker run --rm generativepoetry:latest generative-poetry-cli --dry-run

# Save PDF output to local directory
docker run --rm -v $(pwd)/output:/output generativepoetry:latest \
  generative-poetry-cli --procedure markov --input-words "docker" "poetry"

# Interactive mode
docker run --rm -it generativepoetry:latest bash
```

## Notes

- All PDF outputs are saved to the current directory by default
- Use `--seed` for reproducible results
- Use `--profile` to see detailed performance metrics
- Use `--dry-run` to test without creating files
- See `generative-poetry-cli --help` for all available options

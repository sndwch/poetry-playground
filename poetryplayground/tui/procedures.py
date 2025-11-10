"""Shared procedure definitions for all TUI screens.

This module provides a single source of truth for all available procedures
to prevent duplication and drift between different TUI screens.
"""

from typing import ClassVar

# Canonical list of all available procedures
# Format: (id, name, description, category)
PROCEDURES: ClassVar[list] = [
    # Visual/Concrete Poetry
    (
        "futurist",
        "Futurist Poem",
        "Marinetti-inspired mathematical word connections",
        "Visual Poetry",
    ),
    (
        "markov",
        "Stochastic Jolastic (Markov)",
        "Joyce-like wordplay with rhyme schemes",
        "Visual Poetry",
    ),
    ("chaotic", "Chaotic Concrete Poem", "Abstract spatial arrangements", "Visual Poetry"),
    ("charsoup", "Character Soup Poem", "Character-level visual chaos", "Visual Poetry"),
    ("wordsoup", "Stop Word Soup Poem", "Stop words in visual patterns", "Visual Poetry"),
    ("puzzle", "Visual Puzzle Poem", "Interactive terminal-based puzzles", "Visual Poetry"),
    # Ideation Tools
    (
        "lineseeds",
        "Line Seeds Generator",
        "Evocative incomplete phrases and line beginnings",
        "Ideation",
    ),
    (
        "personalized_lineseeds",
        "Personalized Line Seeds",
        "Style-matched seeds from your corpus fingerprint",
        "Ideation",
    ),
    (
        "metaphor",
        "Metaphor Generator",
        "Fresh metaphors from Project Gutenberg texts",
        "Ideation",
    ),
    ("corpus", "Personal Corpus Analyzer", "Analyze your existing poetry", "Ideation"),
    (
        "theseus",
        "Ship of Theseus Transformer",
        "Gradually transform existing poems",
        "Ideation",
    ),
    ("ideas", "Poetry Idea Generator", "Creative seeds from classic literature", "Ideation"),
    (
        "sixdegrees",
        "Six Degrees Word Convergence",
        "Explore connections between concepts",
        "Ideation",
    ),
    ("fragments", "Resonant Fragment Miner", "Extract poetic sentence fragments", "Ideation"),
    ("equidistant", "Equidistant Word Finder", "Find words bridging two anchors", "Ideation"),
    (
        "definitional",
        "Definitional Finder",
        "Find words by searching dictionary definitions",
        "Ideation",
    ),
    (
        "poem_scaffold",
        "Poem Scaffold Generator",
        "Generate multi-stanza thematic structure",
        "Ideation",
    ),
    (
        "semantic_path",
        "Semantic Geodesic Finder",
        "Find transitional paths through meaning-space",
        "Ideation",
    ),
    (
        "conceptual_cloud",
        "Conceptual Cloud Generator",
        "Multi-dimensional word associations",
        "Ideation",
    ),
    # Templates
    (
        "template_extract",
        "Template Extractor",
        "Extract structure from existing poems",
        "Templates",
    ),
    # Syllabic Forms
    ("haiku", "Haiku Generator", "5-7-5 syllable haiku with templates", "Forms"),
    ("tanka", "Tanka Generator", "5-7-5-7-7 syllable tanka", "Forms"),
    ("senryu", "Senryu Generator", "5-7-5 syllable senryu", "Forms"),
    # System
    ("deps", "Check System Dependencies", "Verify installation and dependencies", "System"),
]

"""Core modules for poetry generation.

This package contains all foundational, shared logic that has no dependencies
on high-level generators or features. These modules provide the stable,
tested infrastructure that all generators build upon.

Core modules:
- quality_scorer: Universal quality scoring for words and phrases
- word_validator: English word validation and filtering
- lexicon: Word frequency, concreteness, and lexical data
- lexigen: Word generation API (rhymes, synonyms, related words)
- document_library: Project Gutenberg document retrieval
- pos_vocabulary: Part-of-speech tagged vocabulary
- vocabulary: Curated word lists by theme
- gutenberg_utils: Gutenberg text processing utilities
- system_utils: System dependency checks
"""

# Quality scoring
# Document library
from poetryplayground.core.document_library import (
    DocumentInfo,
    DocumentLibrary,
    document_library,
    get_diverse_gutenberg_documents,
    random_gutenberg_document,
)

# Lexical data
from poetryplayground.core.lexicon import (
    LexiconData,
    get_lexicon_data,
)

# Word generation API
from poetryplayground.core.lexigen import (
    contextually_linked_word,
    contextually_linked_words,
    frequently_following_word,
    frequently_following_words,
    phonetically_related_words,
    related_rare_word,
    related_rare_words,
    rhyme,
    rhymes,
    similar_meaning_word,
    similar_meaning_words,
    similar_sounding_word,
    similar_sounding_words,
)
from poetryplayground.core.quality_scorer import (
    EmotionalTone,
    FormalityLevel,
    GenerationContext,
    QualityScore,
    QualityScorer,
    get_quality_scorer,
)

# Word validation
from poetryplayground.core.word_validator import (
    WordValidator,
    word_validator,
)

__all__ = [
    "DocumentInfo",
    # Document library
    "DocumentLibrary",
    "EmotionalTone",
    "FormalityLevel",
    "GenerationContext",
    # Lexical data
    "LexiconData",
    "QualityScore",
    # Quality scoring
    "QualityScorer",
    # Word validation
    "WordValidator",
    "contextually_linked_word",
    "contextually_linked_words",
    "document_library",
    "frequently_following_word",
    "frequently_following_words",
    "get_diverse_gutenberg_documents",
    "get_lexicon_data",
    "get_quality_scorer",
    "phonetically_related_words",
    "random_gutenberg_document",
    "related_rare_word",
    "related_rare_words",
    # Word generation
    "rhyme",
    "rhymes",
    "similar_meaning_word",
    "similar_meaning_words",
    "similar_sounding_word",
    "similar_sounding_words",
    "word_validator",
]

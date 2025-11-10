"""Definitional Finder: Find words by searching dictionary definitions.

This module provides a "lateral search" tool that uncovers conceptual,
associative links by searching WordNet definitions (glosses) for a given term.

Unlike semantic similarity searches that find synonyms, this finds words whose
definitions MENTION the search term, revealing strange, associative connections.

Example:
    >>> results = find_words_by_definition("threshold", limit=5)
    >>> for word, definition, score in results:
    ...     print(f"{word}: {definition[:50]}... ({score:.2f})")
    limen: the threshold of consciousness... (0.85)
    narthex: an antechamber leading to the threshold... (0.81)
    subliminal: below the threshold of sensation... (0.77)
"""

from dataclasses import dataclass
from typing import List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from poetryplayground.cache import cached_api_call
from poetryplayground.core.quality_scorer import get_quality_scorer
from poetryplayground.core.word_validator import word_validator
from poetryplayground.logger import logger


@dataclass
class DefinitionalResult:
    """A word found by searching definitions.

    Attributes:
        word: The word found in a definition
        definition: The WordNet definition (gloss) containing the search term
        quality_score: Overall quality score (0.0-1.0) from QualityScorer
        pos: Part of speech ('n', 'v', 'a', 'r')
    """

    word: str
    definition: str
    quality_score: float
    pos: str


def find_words_by_definition(
    search_term: str,
    pos_filter: Optional[str] = None,
    limit: int = 20,
    min_quality: float = 0.5,
    allow_multiword: bool = True,
    verbose: bool = True,
) -> List[DefinitionalResult]:
    """Find words by searching WordNet definitions for a search term.

    This is a "lateral search" that finds associative links rather than synonyms.
    For example, searching for "bean" finds words like "succotash", "casserole",
    or "roasting" because their definitions mention beans.

    Args:
        search_term: Word to search for in definitions (e.g., "rust", "threshold")
        pos_filter: Optional POS filter ('n', 'v', 'a', 'r' or None for all)
        limit: Maximum number of results to return (default: 20)
        min_quality: Minimum quality score required (0.0-1.0, default: 0.5)
        allow_multiword: Include multi-word terms like "iron oxide" (default: True)
        verbose: Show progress bar and logging (default: True)

    Returns:
        List of DefinitionalResult objects, sorted by quality score (descending)

    Raises:
        ImportError: If WordNet data not installed
        ValueError: If invalid pos_filter provided

    Example:
        >>> results = find_words_by_definition("rust", pos_filter='n', limit=10)
        >>> for result in results:
        ...     print(f"{result.word}: {result.quality_score:.2f}")
        smut: 0.82
        corrosion: 0.79
        oxidation: 0.75
    """
    # Import WordNet (fail early if not installed)
    try:
        from nltk.corpus import wordnet  # noqa: F401
    except (ImportError, LookupError) as e:
        raise ImportError("WordNet data not found. Please run: poetry-playground --setup") from e

    # Validate POS filter
    valid_pos = {None, "n", "v", "a", "r"}
    if pos_filter not in valid_pos:
        raise ValueError(f"Invalid pos_filter '{pos_filter}'. Must be one of: {valid_pos}")

    # Get singletons
    scorer = get_quality_scorer()

    # Normalize search term
    search_term = search_term.lower().strip()

    if verbose:
        logger.info(f"Searching WordNet definitions for '{search_term}'...")
        if pos_filter:
            logger.info(f"Filtering by POS: {pos_filter}")

    # Use cached helper function for the actual search
    found_lemmas = _search_wordnet_definitions_cached(
        search_term, pos_filter, allow_multiword, verbose
    )

    if verbose:
        logger.info(f"Found {len(found_lemmas)} candidate words")
        logger.info("Scoring and filtering results...")

    # Score and filter results
    scored_results: List[DefinitionalResult] = []
    for word, (definition, pos) in found_lemmas.items():
        # Score the word
        quality_obj = scorer.score_word(word)

        # Filter by quality threshold
        if quality_obj.overall < min_quality:
            continue

        # Create result object
        result = DefinitionalResult(
            word=word,
            definition=definition,
            quality_score=quality_obj.overall,
            pos=pos,
        )
        scored_results.append(result)

    # Sort by quality score (descending)
    scored_results.sort(key=lambda x: x.quality_score, reverse=True)

    # Return top N results
    final_results = scored_results[:limit]

    if verbose:
        logger.info(
            f"Returning {len(final_results)} results (after quality filter >= {min_quality:.2f})"
        )

    return final_results


@cached_api_call(endpoint="wordnet.definition_search", ttl=86400 * 7)  # 7 days
def _search_wordnet_definitions_cached(
    search_term: str,
    pos_filter: Optional[str],
    allow_multiword: bool,
    verbose: bool,
) -> dict:
    """Cached helper to search WordNet definitions.

    This function is cached separately to avoid re-scoring on every call.
    The cache key includes search_term, pos_filter, and allow_multiword.

    Returns:
        Dict mapping word -> (definition, pos)
    """
    from nltk.corpus import wordnet as wn

    validator = word_validator
    found_lemmas = {}

    # Get all synsets (optionally filtered by POS)
    all_synsets = list(wn.all_synsets(pos=pos_filter))

    # Use progress bar if verbose
    if verbose:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Scanning {len(all_synsets)} definitions...", total=None)

            for synset in all_synsets:
                _process_synset(synset, search_term, allow_multiword, validator, found_lemmas)

            progress.update(task, completed=True)
    else:
        # No progress bar
        for synset in all_synsets:
            _process_synset(synset, search_term, allow_multiword, validator, found_lemmas)

    return found_lemmas


def _process_synset(
    synset,
    search_term: str,
    allow_multiword: bool,
    validator,
    found_lemmas: dict,
) -> None:
    """Process a single synset, extracting matching lemmas."""
    # Get definition (gloss)
    definition = synset.definition().lower()

    # Check if search term appears in definition
    if search_term not in definition:
        return

    # Extract lemmas from this synset
    for lemma in synset.lemmas():
        # Get word and clean it
        word = lemma.name().lower()

        # Handle underscores (WordNet uses these for multi-word terms)
        has_underscore = "_" in word
        if has_underscore:
            if allow_multiword:
                word = word.replace("_", " ")
            else:
                continue  # Skip multi-word terms

        # Filter 1: Self-reference (don't return the search term itself)
        if word == search_term:
            continue

        # Filter 2: Deduplication
        if word in found_lemmas:
            continue

        # Filter 3: Validation (English word check)
        if not validator.is_valid_english_word(word, allow_rare=True):
            continue

        # Store: word -> (definition, pos)
        found_lemmas[word] = (synset.definition(), synset.pos())

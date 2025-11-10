"""Word finding algorithms for creative writing and poetry.

This module provides functions for discovering words with specific relationships
to anchor words, useful for poetic techniques like assonance, consonance, and
conceptual bridging.
"""

import functools
from typing import List, Literal, Optional, Tuple

from rapidfuzz.distance import Levenshtein

from poetryplayground.config import EquidistantHit
from poetryplayground.lexicon import LexiconData, get_lexicon_data


@functools.lru_cache(maxsize=10000)
def _get_orthographic_rime(word: str) -> str:
    """Extract the orthographic rime (last vowel + trailing consonants).

    The rime is important for rhyme and assonance detection. For example:
    - "stone" -> "one"
    - "light" -> "ight"
    - "cat" -> "at"

    Args:
        word: Word to extract rime from

    Returns:
        The rime portion of the word (empty string if no vowel found)
    """
    word = word.lower()
    # Find last vowel and return from there to end of word
    for i in range(len(word) - 1, -1, -1):
        if word[i] in "aeiouy":
            return word[i:]
    return word  # No vowel found, return whole word


def _calculate_craft_score(
    target_d: int,
    dist_a: int,
    dist_b: int,
    zipf: float,
    word_len: int,
    anchor_len_avg: float,
    word: str,
    anchor_a: str,
    anchor_b: str,
) -> float:
    """Calculate a craft-aware score for ranking equidistant results.

    This scoring function goes beyond pure mathematical distance to consider
    poetic usefulness:
    - Exactness: Bonus for being exactly at target distance
    - Quality: Uses universal Quality Scorer (novelty, non-cliché, good frequency)
    - Length: Penalize words that are very different in length from anchors
    - Rime: Bonus for sharing rime with either anchor (assonance)

    Args:
        target_d: The target Levenshtein distance
        dist_a: Actual distance from anchor A
        dist_b: Actual distance from anchor B
        zipf: Word frequency on Zipf scale (1-10) (kept for backwards compat, not used)
        word_len: Length of the candidate word
        anchor_len_avg: Average length of the two anchor words
        word: The candidate word
        anchor_a: First anchor word
        anchor_b: Second anchor word

    Returns:
        Score value (higher is better)
    """
    from poetryplayground.quality_scorer import get_quality_scorer

    scorer = get_quality_scorer()
    score = 0.0

    # Bonus for being exactly at target distance (not in window)
    if dist_a == target_d:
        score += 2.0
    if dist_b == target_d:
        score += 2.0

    # Use Quality Scorer instead of raw frequency
    # This rewards novelty, non-clichéd words, and good (not too rare, not too common) frequency
    quality = scorer.score_word(word)
    score += 5.0 * quality.overall  # Scale to match other bonuses

    # Penalty for large length deviation from anchors
    # This helps avoid finding very short or very long outliers
    score -= 0.1 * abs(word_len - anchor_len_avg)

    # Bonus for rime preservation (assonance/consonance)
    rime_w = _get_orthographic_rime(word)
    rime_a = _get_orthographic_rime(anchor_a)
    rime_b = _get_orthographic_rime(anchor_b)

    if rime_w in (rime_a, rime_b):
        score += 0.5

    return score


def find_equidistant(
    a: str,
    b: str,
    mode: Literal["orth", "phono"] = "orth",
    window: int = 0,
    min_zipf: float = 3.0,
    pos_filter: Optional[str] = None,
    syllable_filter: Optional[Tuple[int, int]] = None,
    lexicon_data: Optional[LexiconData] = None,
) -> List[EquidistantHit]:
    """Find words equidistant (or near-equidistant) from two anchor words.

    This function discovers words that are equally distant (by Levenshtein distance)
    from two anchor words, useful for:
    - Finding conceptual bridges between ideas
    - Creating word ladders or chains
    - Discovering assonance and consonance patterns
    - Generating creative variations

    The algorithm:
    1. Calculate distance d between anchor words A and B
    2. Find all words X where distance(A,X) ≈ d and distance(B,X) ≈ d
    3. Filter by linguistic properties (frequency, POS, syllables)
    4. Score and rank by poetic usefulness

    Args:
        a: First anchor word
        b: Second anchor word
        mode: 'orth' for orthographic (spelling) or 'phono' for phonetic distance
        window: Allow distance to be d±window (0 = exact, 1 = d-1 to d+1, etc.)
        min_zipf: Minimum word frequency (Zipf scale 1-10, default 3.0 = moderately common)
        pos_filter: Filter by part of speech (e.g., 'NOUN', 'VERB', 'ADJ', 'ADV')
        syllable_filter: Tuple of (min, max) syllables (inclusive), e.g., (1, 3)
        lexicon_data: Optional pre-loaded lexicon (for testing with snapshot)

    Returns:
        List of EquidistantHit objects, sorted by score (best first)

    Raises:
        ValueError: If anchor words don't have phonetic representations (mode='phono' only)

    Examples:
        >>> # Find words between "stone" and "storm" (orthographic)
        >>> hits = find_equidistant("stone", "storm")
        >>> print([h.word for h in hits[:5]])
        ['store', 'stoke', 'stomp', 'scone', 'sworn']

        >>> # Find words between "light" and "night" (phonetic, exact rhymes)
        >>> hits = find_equidistant("light", "night", mode="phono")
        >>> print([h.word for h in hits[:5]])
        ['right', 'might', 'tight', 'fight', 'sight']
    """
    # Load lexicon (cached, so this is fast)
    if lexicon_data is None:
        lexicon_data = get_lexicon_data()

    a_norm = a.lower()
    b_norm = b.lower()

    # Get anchor strings (orthographic or phonetic)
    if mode == "phono":
        str_a = lexicon_data.phoneme_cache.get(a_norm, "")
        str_b = lexicon_data.phoneme_cache.get(b_norm, "")
        if not str_a or not str_b:
            raise ValueError(
                f"Phonetic representation not found for '{a}' or '{b}'. "
                "These words may not be in the CMU pronunciation dictionary."
            )
        bucket_map = lexicon_data.words_by_phoneme_length
    else:
        str_a = a_norm
        str_b = b_norm
        bucket_map = lexicon_data.words_by_grapheme_length

    # Calculate target distance and window bounds
    d = Levenshtein.distance(str_a, str_b)
    lo_d = max(0, d - window)
    hi_d = d + window

    # Early exit for identical words with no window
    if d == 0 and window == 0:
        return []

    # Calculate length pruning bounds using triangle inequality
    # If distance(A, X) <= hi_d, then len(A) - hi_d <= len(X) <= len(A) + hi_d
    len_a = len(str_a)
    len_b = len(str_b)
    min_search_len = max(0, len_a - hi_d, len_b - hi_d)
    max_search_len = min(len_a + hi_d, len_b + hi_d)

    hits = []

    # Iterate only over relevant length buckets (optimization)
    for length in range(min_search_len, max_search_len + 1):
        for word in bucket_map.get(length, []):
            # Skip the anchor words themselves
            if word in (a_norm, b_norm):
                continue

            # Apply frequency filter
            zipf = lexicon_data.zipf_cache.get(word, 0.0)
            if zipf < min_zipf:
                continue

            # Apply POS filter
            if pos_filter is not None:
                pos = lexicon_data.pos_cache.get(word, "UNKNOWN")
                if pos != pos_filter:
                    continue

            # Apply syllable filter
            if syllable_filter is not None:
                syllables = lexicon_data.syllable_cache.get(word)
                if syllables is None:
                    continue
                min_syl, max_syl = syllable_filter
                if not (min_syl <= syllables <= max_syl):
                    continue

            # Get search string for this word
            if mode == "phono":
                str_w = lexicon_data.phoneme_cache.get(word, "")
                if not str_w:
                    continue
            else:
                str_w = word

            # Core distance checks
            d_a = Levenshtein.distance(str_a, str_w)
            if not (lo_d <= d_a <= hi_d):
                continue

            d_b = Levenshtein.distance(str_b, str_w)
            if not (lo_d <= d_b <= hi_d):
                continue

            # Passed all filters! Calculate score and add to results
            score = _calculate_craft_score(
                target_d=d,
                dist_a=d_a,
                dist_b=d_b,
                zipf=zipf,
                word_len=len(word),
                anchor_len_avg=(len(a_norm) + len(b_norm)) / 2,
                word=word,
                anchor_a=a_norm,
                anchor_b=b_norm,
            )

            hits.append(
                EquidistantHit(
                    word=word,
                    target_distance=d,
                    dist_a=d_a,
                    dist_b=d_b,
                    mode=mode,
                    zipf_frequency=zipf,
                    syllables=lexicon_data.syllable_cache.get(word),
                    pos=lexicon_data.pos_cache.get(word),
                    score=score,
                )
            )

    # Sort by score (descending) and return
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits

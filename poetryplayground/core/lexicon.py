"""Lexicon loading and caching for word-finding operations.

This module provides efficient lexicon management with pre-computed attributes
(phonemes, syllables, POS tags, frequencies) for fast lookups during word searches.
"""

import functools
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pronouncing
from wordfreq import top_n_list, zipf_frequency

# Import the existing spaCy instance from the project
try:
    from poetryplayground.pos_vocabulary import nlp
except ImportError:
    # Graceful fallback if spaCy isn't available
    nlp = None


@functools.lru_cache(maxsize=1)
def get_lexicon_data(use_snapshot_for_testing: bool = False) -> "LexiconData":
    """Get or create the cached lexicon data instance.

    Args:
        use_snapshot_for_testing: If True, uses the deterministic 10k word snapshot
            from tests/data/snapshot_lexicon.txt instead of the full 100k lexicon

    Returns:
        Cached LexiconData instance
    """
    return LexiconData(use_snapshot=use_snapshot_for_testing)


def _fallback_syllable_counter(word: str) -> int:
    """Regex-based syllable counter for words not in CMU dictionary.

    Uses a simple heuristic based on vowel groups. Not perfect, but reasonable
    for words missing from pronouncing/CMUdict.

    Args:
        word: Word to count syllables for

    Returns:
        Estimated syllable count (minimum 1)
    """
    word = word.lower()

    # Handle common single-syllable exceptions
    exceptions = {"the": 1, "a": 1, "of": 1, "is": 1, "it": 1, "in": 1, "to": 1}
    if word in exceptions:
        return exceptions[word]

    # Remove silent 'e' at the end (unless it's 'le')
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]

    # Count groups of consecutive vowels as single syllables
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)

    # Never return 0 (every word has at least one syllable)
    return max(count, 1)


class LexiconData:
    """Pre-processed lexicon with cached phonetic, syllable, and POS data.

    This class loads a word list and pre-computes expensive attributes like
    phonetic representations, syllable counts, and part-of-speech tags.
    Results are organized by length for efficient pruning during searches.

    Attributes:
        words_by_grapheme_length: Dict mapping word length -> list of words
        words_by_phoneme_length: Dict mapping phoneme length -> list of words
        phoneme_cache: Dict mapping word -> phonetic representation
        zipf_cache: Dict mapping word -> frequency score
        pos_cache: Dict mapping word -> POS tag
        syllable_cache: Dict mapping word -> syllable count
        full_lexicon: Set of all words in the lexicon
    """

    def __init__(self, use_snapshot: bool = False):
        """Initialize and pre-process the lexicon.

        Args:
            use_snapshot: If True, uses 10k word snapshot instead of 100k full lexicon
        """
        # Initialize all caches
        self.words_by_grapheme_length: Dict[int, List[str]] = defaultdict(list)
        self.words_by_phoneme_length: Dict[int, List[str]] = defaultdict(list)
        self.phoneme_cache: Dict[str, str] = {}
        self.zipf_cache: Dict[str, float] = {}
        self.pos_cache: Dict[str, str] = {}
        self.syllable_cache: Dict[str, int] = {}
        self.full_lexicon: Set[str] = set()

        # Load lexicon from appropriate source
        lexicon = self._load_snapshot_lexicon() if use_snapshot else top_n_list("en", 100000)

        # Pre-process all words with spaCy in batch for efficiency
        # nlp.pipe() is much faster than calling nlp() 100k times
        docs = nlp.pipe(lexicon) if nlp is not None else iter([None] * len(lexicon))

        # Process each word and cache all attributes
        for word, doc in zip(lexicon, docs):
            self.full_lexicon.add(word)

            # 1. Word frequency (Zipf scale 1-10)
            self.zipf_cache[word] = zipf_frequency(word, "en")

            # 2. Part of speech tagging (using spaCy)
            if doc and len(doc) > 0:
                self.pos_cache[word] = doc[0].pos_
            else:
                self.pos_cache[word] = "UNKNOWN"

            # 3. Phonetic representation and syllables
            phones_list = pronouncing.phones_for_word(word)
            if phones_list:
                # Use first pronunciation, strip stress markers (0, 1, 2)
                phone_str = phones_list[0].replace("0", "").replace("1", "").replace("2", "")
                syl_count = pronouncing.syllable_count(phones_list[0])
            else:
                # Word not in CMU dictionary
                phone_str = ""
                syl_count = _fallback_syllable_counter(word)

            self.phoneme_cache[word] = phone_str
            self.syllable_cache[word] = syl_count

            # 4. Add to length-based buckets for fast pruning
            self.words_by_grapheme_length[len(word)].append(word)
            if phone_str:  # Only bucket if we have phonetic data
                self.words_by_phoneme_length[len(phone_str)].append(word)

    def _load_snapshot_lexicon(self) -> List[str]:
        """Load the deterministic snapshot lexicon for testing.

        Returns:
            List of words from snapshot file

        Raises:
            FileNotFoundError: If snapshot file doesn't exist
        """
        # Find the snapshot file relative to this module
        # Since we're in core/, need to go up two levels to get to project root
        module_dir = Path(__file__).parent  # poetryplayground/core/
        package_dir = module_dir.parent  # poetryplayground/
        project_root = package_dir.parent  # project root
        snapshot_path = project_root / "tests" / "data" / "snapshot_lexicon.txt"

        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Snapshot lexicon not found at {snapshot_path}. "
                "Run scripts/create_snapshot_lexicon.py to create it."
            )

        with open(snapshot_path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        """Return the number of words in the lexicon."""
        return len(self.full_lexicon)

    def __contains__(self, word: str) -> bool:
        """Check if a word is in the lexicon."""
        return word.lower() in self.full_lexicon

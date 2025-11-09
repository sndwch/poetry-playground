"""POS-tagged vocabulary organized by syllable count.

This module provides a word bank where words are organized by both their
part-of-speech (POS) tag and syllable count. This enables grammatical
template-based generation for syllable-constrained poetry forms.

The word bank is built from NLTK's Brown corpus and cached to disk for
fast loading on subsequent runs.
"""

import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.corpus import brown

from .forms import count_syllables
from .logger import logger
from .vocabulary import vocabulary

# Universal POS tag mapping (simplified for poetry generation)
CORE_POS_TAGS = {
    "NOUN": ["NN", "NNS", "NNP", "NNPS"],  # Nouns
    "VERB": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],  # Verbs
    "ADJ": ["JJ", "JJR", "JJS"],  # Adjectives
    "ADV": ["RB", "RBR", "RBS"],  # Adverbs
    "DET": ["DT"],  # Determiners
    "PREP": ["IN", "TO"],  # Prepositions (TO is sometimes preposition)
    "PRON": ["PRP", "PRP$", "WP", "WP$"],  # Pronouns
    "CONJ": ["CC"],  # Conjunctions
}

# Reverse mapping: Penn Treebank tag -> Universal tag
PENN_TO_UNIVERSAL = {}
for universal_tag, penn_tags in CORE_POS_TAGS.items():
    for penn_tag in penn_tags:
        PENN_TO_UNIVERSAL[penn_tag] = universal_tag


class POSVocabulary:
    """POS-tagged word bank organized by syllables.

    Attributes:
        word_bank: Dictionary mapping POS tags to syllable counts to word lists
                  Format: {POS: {syllables: [words]}}
        cache_path: Path to the cached word bank file
    """

    def __init__(self, cache_dir: Optional[Path] = None, rebuild_cache: bool = False):
        """Initialize POS vocabulary.

        Args:
            cache_dir: Directory for cache file (default: ~/.poetryplayground/)
            rebuild_cache: If True, rebuild cache even if it exists
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".poetryplayground"

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / "pos_word_bank.pkl"

        self.word_bank: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Try to load from cache first
        if not rebuild_cache and self.cache_path.exists():
            try:
                self._load_from_cache()
                logger.info(f"Loaded POS word bank from cache: {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding...")
                self._build_word_bank()
                self._save_to_cache()
        else:
            logger.info("Building POS word bank from scratch...")
            self._build_word_bank()
            self._save_to_cache()

    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        required_data = [
            ("corpora/brown", "brown"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ]

        for data_path, data_name in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                logger.info(f"Downloading NLTK data: {data_name}")
                nltk.download(data_name, quiet=True)

    def _build_word_bank(self):
        """Build word bank from various sources."""
        self._ensure_nltk_data()

        logger.info("Building POS-tagged word bank...")

        # Source 1: Common words from vocabulary module
        self._tag_common_words()

        # Source 2: Brown corpus (tagged)
        self._tag_brown_corpus()

        # Deduplicate and filter
        self._deduplicate_and_filter()

        total_words = sum(
            len(words) for pos_dict in self.word_bank.values() for words in pos_dict.values()
        )
        logger.info(
            f"Built word bank with {total_words} total words across {len(self.word_bank)} POS tags"
        )

    def _tag_common_words(self):
        """Tag common words from vocabulary module."""
        logger.info("Tagging common words from vocabulary...")

        # Get all common words
        all_common_words = set()
        if hasattr(vocabulary, "common_words_by_syllables"):
            for word_list in vocabulary.common_words_by_syllables.values():
                all_common_words.update(word_list)

        # Tag them
        words_to_tag = list(all_common_words)
        if not words_to_tag:
            logger.warning("No common words found in vocabulary")
            return

        # Tag in batches for efficiency
        batch_size = 1000
        for i in range(0, len(words_to_tag), batch_size):
            batch = words_to_tag[i : i + batch_size]
            tagged = nltk.pos_tag(batch)

            for word, penn_tag in tagged:
                universal_tag = PENN_TO_UNIVERSAL.get(penn_tag)
                if universal_tag:
                    syllables = count_syllables(word)
                    if syllables > 0:  # Skip words with no syllables
                        self.word_bank[universal_tag][syllables].append(word)

        logger.info(f"Tagged {len(words_to_tag)} common words")

    def _tag_brown_corpus(self, max_words: int = 10000):
        """Tag words from Brown corpus.

        Args:
            max_words: Maximum number of words to process from corpus
        """
        logger.info("Tagging words from Brown corpus...")

        try:
            tagged_words = brown.tagged_words()
        except LookupError:
            logger.warning("Brown corpus not available, skipping")
            return

        # Process up to max_words unique words
        processed = set()
        count = 0

        for word, penn_tag in tagged_words:
            if count >= max_words:
                break

            # Normalize word
            word_clean = word.lower().strip()

            # Skip if already processed or not alphabetic
            if word_clean in processed or not word_clean.isalpha():
                continue

            # Convert Penn tag to universal
            universal_tag = PENN_TO_UNIVERSAL.get(penn_tag)
            if universal_tag:
                syllables = count_syllables(word_clean)
                if syllables > 0:
                    self.word_bank[universal_tag][syllables].append(word_clean)
                    processed.add(word_clean)
                    count += 1

        logger.info(f"Tagged {count} words from Brown corpus")

    def _deduplicate_and_filter(self):
        """Remove duplicates and filter out problematic words."""
        logger.info("Deduplicating and filtering word bank...")

        for pos_tag in self.word_bank:
            for syllable_count in self.word_bank[pos_tag]:
                # Deduplicate
                words = list(set(self.word_bank[pos_tag][syllable_count]))

                # Filter: keep only words with 1-7 syllables and 2+ characters
                words = [
                    w
                    for w in words
                    if 1 <= len(w) <= 20  # Reasonable length
                    and w.isalpha()  # Only letters
                    and count_syllables(w) == syllable_count  # Verify syllable count
                ]

                # Sort for consistency
                words.sort()

                self.word_bank[pos_tag][syllable_count] = words

        # Remove empty entries
        for pos_tag in list(self.word_bank.keys()):
            self.word_bank[pos_tag] = {k: v for k, v in self.word_bank[pos_tag].items() if v}
            if not self.word_bank[pos_tag]:
                del self.word_bank[pos_tag]

    def _save_to_cache(self):
        """Save word bank to cache file."""
        try:
            # Convert defaultdicts to regular dicts for pickling
            cache_data = {pos: dict(syllable_dict) for pos, syllable_dict in self.word_bank.items()}

            with open(self.cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved POS word bank to cache: {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_from_cache(self):
        """Load word bank from cache file."""
        with open(self.cache_path, "rb") as f:
            cache_data = pickle.load(f)

        # Convert back to defaultdicts
        self.word_bank = defaultdict(lambda: defaultdict(list))
        for pos, syllable_dict in cache_data.items():
            for syllables, words in syllable_dict.items():
                self.word_bank[pos][syllables] = words

    def get_words(self, pos_tag: str, syllable_count: int) -> List[str]:
        """Get words matching POS tag and syllable count.

        Args:
            pos_tag: POS tag (e.g., 'NOUN', 'VERB')
            syllable_count: Number of syllables

        Returns:
            List of words matching criteria (empty list if none found)
        """
        if pos_tag not in self.word_bank:
            return []

        return self.word_bank[pos_tag].get(syllable_count, [])

    def get_syllable_combinations(
        self, pos_pattern: List[str], target_syllables: int, max_results: int = 100
    ) -> List[Tuple[int, ...]]:
        """Find syllable distributions for a POS pattern.

        Args:
            pos_pattern: List of POS tags (e.g., ['ADJ', 'NOUN'])
            target_syllables: Target total syllables
            max_results: Maximum number of combinations to return

        Returns:
            List of tuples representing syllable distributions
            e.g., [(2, 3), (1, 4), (3, 2)] for pattern ['ADJ', 'NOUN'] with target 5

        Example:
            >>> vocab.get_syllable_combinations(['ADJ', 'NOUN'], 5)
            [(1, 4), (2, 3), (3, 2), (4, 1)]
        """
        if not pos_pattern or target_syllables <= 0:
            return []

        # Get available syllable counts for each POS
        available_syllables = []
        for pos_tag in pos_pattern:
            if pos_tag not in self.word_bank:
                return []  # Can't fill this pattern

            # Get all syllable counts available for this POS
            syllable_counts = sorted(self.word_bank[pos_tag].keys())
            if not syllable_counts:
                return []

            available_syllables.append(syllable_counts)

        # Generate all combinations
        results = []
        for combo in product(*available_syllables):
            if sum(combo) == target_syllables:
                results.append(combo)

                if len(results) >= max_results:
                    break

        return results

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about the word bank.

        Returns:
            Dictionary with stats for each POS tag
        """
        stats = {}

        for pos_tag in self.word_bank:
            total_words = sum(len(words) for words in self.word_bank[pos_tag].values())
            syllable_counts = sorted(self.word_bank[pos_tag].keys())

            stats[pos_tag] = {
                "total_words": total_words,
                "syllable_range": f"{min(syllable_counts)}-{max(syllable_counts)}"
                if syllable_counts
                else "0",
                "syllable_counts_available": len(syllable_counts),
            }

        return stats

    def rebuild(self):
        """Rebuild the word bank from scratch."""
        logger.info("Rebuilding POS word bank...")
        self.word_bank = defaultdict(lambda: defaultdict(list))
        self._build_word_bank()
        self._save_to_cache()


def create_pos_vocabulary(
    cache_dir: Optional[Path] = None, rebuild_cache: bool = False
) -> POSVocabulary:
    """Factory function to create a POSVocabulary instance.

    Args:
        cache_dir: Directory for cache file
        rebuild_cache: If True, rebuild cache even if it exists

    Returns:
        Configured POSVocabulary instance
    """
    return POSVocabulary(cache_dir=cache_dir, rebuild_cache=rebuild_cache)

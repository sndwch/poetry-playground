"""Ship of Theseus Transformer - Gradually transform poems while maintaining grammatical structure.

This module implements the Ship of Theseus thought experiment for poetry: gradually
replacing words in a poem while preserving grammatical structure, demonstrating how
much of the original can be changed while still recognizing the transformed result.

Key features:
- POS-constrained word replacement (noun→noun, verb→verb, etc.)
- Optional syllable preservation to maintain rhythm
- Gradual transformation showing progression from original to transformed
- Maintains original grammatical structure by inferring POS from input
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import nltk

from .forms import count_syllables
from .logger import logger
from .pos_vocabulary import PENN_TO_UNIVERSAL, POSVocabulary


@dataclass
class TransformationResult:
    """Result of a Ship of Theseus transformation.

    Attributes:
        original: Original text before transformation
        transformed: Text after transformation
        num_replacements: Number of words that were replaced
        replacement_ratio: Fraction of words replaced (0.0-1.0)
        preserved_syllables: Whether syllable counts were preserved
    """

    original: str
    transformed: str
    num_replacements: int
    replacement_ratio: float
    preserved_syllables: bool

    def __str__(self) -> str:
        """String representation of transformation result."""
        return (
            f"Original: {self.original}\n"
            f"Transformed: {self.transformed}\n"
            f"Replacements: {self.num_replacements} words "
            f"({self.replacement_ratio:.1%} of original)"
        )


class ShipOfTheseusTransformer:
    """Gradually transform poems while maintaining grammatical structure.

    The Ship of Theseus is a thought experiment: if you replace every plank in
    a ship, one at a time, is it still the same ship? This class applies that
    concept to poetry, replacing words while preserving grammatical structure.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the transformer.

        Args:
            cache_dir: Optional directory for POS vocabulary cache (str or Path)
        """
        # Convert string to Path if needed
        if cache_dir is not None and isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self.pos_vocab = POSVocabulary(cache_dir=cache_dir)
        logger.info("Ship of Theseus transformer initialized")

    def transform_line(
        self,
        line: str,
        replacement_ratio: float = 0.3,
        preserve_pos: bool = True,
        preserve_syllables: bool = True,
    ) -> TransformationResult:
        """Transform a line by replacing words with POS constraints.

        Args:
            line: Original line of poetry
            replacement_ratio: Fraction of words to replace (0.0-1.0)
            preserve_pos: Maintain part-of-speech when replacing (recommended)
            preserve_syllables: Keep same syllable count for rhythm

        Returns:
            TransformationResult with original and transformed text

        Raises:
            ValueError: If replacement_ratio is not between 0.0 and 1.0
        """
        if not 0.0 <= replacement_ratio <= 1.0:
            raise ValueError(
                f"replacement_ratio must be between 0.0 and 1.0, got {replacement_ratio}"
            )

        if not line.strip():
            return TransformationResult(
                original=line,
                transformed=line,
                num_replacements=0,
                replacement_ratio=0.0,
                preserved_syllables=preserve_syllables,
            )

        # Tokenize and POS tag the original line
        words = line.split()
        try:
            tagged = nltk.pos_tag(words)
        except LookupError:
            # Download NLTK data if not available
            logger.info("Downloading NLTK averaged_perceptron_tagger...")
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            tagged = nltk.pos_tag(words)

        # Select words to replace
        num_to_replace = max(1, int(len(words) * replacement_ratio))
        indices_to_replace = random.sample(range(len(words)), min(num_to_replace, len(words)))

        # Replace each selected word
        new_words = words.copy()
        num_replacements = 0

        for idx in indices_to_replace:
            word, penn_tag = tagged[idx]

            # Convert Penn Treebank tag to Universal POS
            universal_pos = PENN_TO_UNIVERSAL.get(penn_tag)

            if not universal_pos or not preserve_pos:
                # Skip if no mapping or not preserving POS
                continue

            # Get replacement candidates
            replacement = self._get_replacement(word, universal_pos, preserve_syllables)

            if replacement and replacement.lower() != word.lower():
                new_words[idx] = replacement
                num_replacements += 1

        transformed = " ".join(new_words)
        actual_ratio = num_replacements / len(words) if words else 0.0

        return TransformationResult(
            original=line,
            transformed=transformed,
            num_replacements=num_replacements,
            replacement_ratio=actual_ratio,
            preserved_syllables=preserve_syllables,
        )

    def _get_replacement(
        self, original_word: str, pos_tag: str, preserve_syllables: bool
    ) -> Optional[str]:
        """Get a replacement word with POS and optionally syllable constraints.

        Args:
            original_word: Word to replace
            pos_tag: Universal POS tag (NOUN, VERB, ADJ, ADV, etc.)
            preserve_syllables: Whether to match syllable count

        Returns:
            Replacement word or None if no suitable candidate found
        """
        if preserve_syllables:
            # Get syllable count of original word
            syllables = count_syllables(original_word)
            # Get words matching both POS and syllable count
            candidates = self.pos_vocab.get_words(pos_tag, syllables)
        else:
            # Get any words of this POS (across all syllable counts)
            candidates = []
            for syl_count in range(1, 8):  # Common syllable range
                candidates.extend(self.pos_vocab.get_words(pos_tag, syl_count))

        if not candidates:
            return None

        # Return a random candidate
        return random.choice(candidates)

    def gradual_transform(
        self,
        original: str,
        steps: int = 5,
        preserve_pos: bool = True,
        preserve_syllables: bool = True,
    ) -> List[TransformationResult]:
        """Perform gradual transformation over multiple steps.

        Demonstrates the Ship of Theseus progression from original to fully
        transformed, showing how the text changes incrementally.

        Args:
            original: Original text (single line or entire poem)
            steps: Number of transformation steps
            preserve_pos: Maintain part-of-speech when replacing
            preserve_syllables: Keep same syllable count

        Returns:
            List of TransformationResults showing progression

        Raises:
            ValueError: If steps < 1
        """
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")

        transformations = []
        current = original

        for step in range(steps):
            # Increase replacement ratio gradually
            ratio = (step + 1) / steps

            # Transform current text
            result = self.transform_line(
                current,
                replacement_ratio=ratio,
                preserve_pos=preserve_pos,
                preserve_syllables=preserve_syllables,
            )

            transformations.append(result)

            # Use transformed text as input for next step
            current = result.transformed

        return transformations

    def transform_poem(
        self,
        poem_lines: List[str],
        replacement_ratio: float = 0.3,
        preserve_pos: bool = True,
        preserve_syllables: bool = True,
    ) -> List[TransformationResult]:
        """Transform an entire poem line by line.

        Args:
            poem_lines: List of lines in the poem
            replacement_ratio: Fraction of words to replace per line
            preserve_pos: Maintain part-of-speech when replacing
            preserve_syllables: Keep same syllable count

        Returns:
            List of TransformationResults, one per line
        """
        results = []

        for line in poem_lines:
            result = self.transform_line(
                line,
                replacement_ratio=replacement_ratio,
                preserve_pos=preserve_pos,
                preserve_syllables=preserve_syllables,
            )
            results.append(result)

        return results


def create_ship_of_theseus_transformer(
    cache_dir: Optional[Union[str, Path]] = None,
) -> ShipOfTheseusTransformer:
    """Factory function to create a ShipOfTheseusTransformer.

    Args:
        cache_dir: Optional directory for POS vocabulary cache (str or Path)

    Returns:
        Configured ShipOfTheseusTransformer instance
    """
    return ShipOfTheseusTransformer(cache_dir=cache_dir)

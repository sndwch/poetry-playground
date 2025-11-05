"""Syllable-aware poem form generators.

This module provides generators for traditional poetry forms with strict syllable
constraints: haiku, tanka, and senryu. Uses CMU Pronouncing Dictionary for accurate
syllable counting with heuristic fallback for words not in the dictionary.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pronouncing

from .cache import cached_api_call
from .lexigen import contextually_linked_words, frequently_following_words, similar_meaning_words
from .logger import timed
from .vocabulary import vocabulary


# Cached syllable counting
@cached_api_call(endpoint="cmu.pronouncing.syllable_count", ttl=86400)
def _cached_syllable_count_cmu(word: str) -> Optional[int]:
    """Cached wrapper for CMU pronouncing dictionary syllable counts."""
    phones_list = pronouncing.phones_for_word(word.lower())
    if phones_list:
        # Use the first pronunciation if multiple exist
        return pronouncing.syllable_count(phones_list[0])
    return None


def count_syllables(word: str) -> int:
    """Count syllables in a word using CMU dict with heuristic fallback.

    Args:
        word: The word to count syllables for

    Returns:
        Number of syllables in the word
    """
    # Clean the word
    word_clean = word.lower().strip(".,!?;:\"'—-()[]{}").strip()
    if not word_clean:
        return 0

    # Try CMU dictionary first
    cmu_count = _cached_syllable_count_cmu(word_clean)
    if cmu_count is not None:
        return cmu_count

    # Fallback to heuristic for words not in CMU dict
    return _heuristic_syllable_count(word_clean)


def _heuristic_syllable_count(word: str) -> int:
    """Simple vowel-counting heuristic for syllable estimation.

    Args:
        word: The word to count (should be lowercase and cleaned)

    Returns:
        Estimated syllable count
    """
    if not word:
        return 0

    vowels = "aeiouy"
    syllables = 0
    prev_was_vowel = False

    for char in word:
        if char in vowels:
            if not prev_was_vowel:
                syllables += 1
            prev_was_vowel = True
        else:
            prev_was_vowel = False

    # Handle silent e
    if word.endswith("e") and syllables > 1:
        syllables -= 1

    # Special case: -le at end often adds syllable
    if len(word) >= 3 and word.endswith("le") and word[-3] not in vowels:
        syllables += 1

    return max(1, syllables)


def count_line_syllables(line: str) -> int:
    """Count total syllables in a line of text.

    Args:
        line: The line of text to count

    Returns:
        Total syllable count for the line
    """
    words = line.split()
    return sum(count_syllables(word) for word in words)


@dataclass
class FormConstraint:
    """Represents a syllable constraint for a line in a form."""

    line_number: int  # 1-indexed for display
    target_syllables: int
    actual_syllables: Optional[int] = None
    line_text: Optional[str] = None

    def is_satisfied(self) -> bool:
        """Check if constraint is satisfied."""
        if self.actual_syllables is None:
            return False
        return self.actual_syllables == self.target_syllables

    def __str__(self) -> str:
        """String representation of constraint."""
        if self.actual_syllables is None:
            return f"Line {self.line_number}: {self.target_syllables} syllables (not set)"

        status = "✓" if self.is_satisfied() else "✗"
        return f"Line {self.line_number}: {self.target_syllables} syllables (got {self.actual_syllables}) {status}"


@dataclass
class FormValidationResult:
    """Result of validating a poem against form constraints."""

    form_name: str
    lines: List[str]
    constraints: List[FormConstraint]
    valid: bool

    def get_report(self) -> str:
        """Generate a detailed validation report."""
        lines = [f"\n{self.form_name.upper()} Validation Report", "=" * 40]

        for i, (line, constraint) in enumerate(zip(self.lines, self.constraints), 1):
            lines.append(f"\nLine {i}: {line}")
            lines.append(f"  {constraint}")

        lines.append("\n" + "=" * 40)
        if self.valid:
            lines.append("✓ All constraints satisfied!")
        else:
            failed = [c for c in self.constraints if not c.is_satisfied()]
            lines.append(f"✗ {len(failed)} constraint(s) failed")
            lines.append("\nFailed constraints:")
            for constraint in failed:
                diff = constraint.actual_syllables - constraint.target_syllables
                if diff > 0:
                    lines.append(f"  Line {constraint.line_number}: {diff} too many syllables")
                else:
                    lines.append(f"  Line {constraint.line_number}: {abs(diff)} too few syllables")

        return "\n".join(lines)


class FormGenerator:
    """Generator for syllable-constrained poetry forms."""

    def __init__(self):
        """Initialize form generator."""
        pass

    @timed("forms.validate")
    def validate_form(
        self, lines: List[str], pattern: List[int], form_name: str
    ) -> FormValidationResult:
        """Validate lines against syllable pattern.

        Args:
            lines: List of lines to validate
            pattern: List of target syllable counts per line
            form_name: Name of the form being validated

        Returns:
            FormValidationResult with detailed validation info
        """
        if len(lines) != len(pattern):
            raise ValueError(
                f"Line count mismatch: got {len(lines)} lines, "
                f"expected {len(pattern)} for {form_name}"
            )

        constraints = []
        all_valid = True

        for i, (line, target) in enumerate(zip(lines, pattern), 1):
            actual = count_line_syllables(line)
            constraint = FormConstraint(
                line_number=i, target_syllables=target, actual_syllables=actual, line_text=line
            )
            constraints.append(constraint)
            if not constraint.is_satisfied():
                all_valid = False

        return FormValidationResult(
            form_name=form_name, lines=lines, constraints=constraints, valid=all_valid
        )

    def _get_word_pool(self, seed_words: List[str], last_word: Optional[str] = None) -> List[str]:
        """Get a pool of candidate words for line generation.

        Args:
            seed_words: Seed words to guide generation
            last_word: The last word added (for contextual links)

        Returns:
            List of candidate words
        """
        candidates = []

        # Try API-based word generation first (with timeout protection)
        if last_word:
            try:
                # Try to get related words (with small max to avoid slowness)
                related = contextually_linked_words(last_word, datamuse_api_max=15)
                if related:
                    candidates.extend(related[:10])
            except Exception:
                pass  # Silently fail and use fallbacks

            try:
                related = frequently_following_words(last_word, datamuse_api_max=15)
                if related:
                    candidates.extend(related[:10])
            except Exception:
                pass

        # Try seed-word-related words
        if seed_words:
            for seed in seed_words:
                try:
                    related = similar_meaning_words(seed, datamuse_api_max=10)
                    if related:
                        candidates.extend(related[:5])
                except Exception:
                    pass

        # Always include common words as fallback
        for syllable_count in [1, 2, 3]:
            if syllable_count in vocabulary.common_words_by_syllables:
                candidates.extend(vocabulary.common_words_by_syllables[syllable_count])

        # Shuffle to get variety
        random.shuffle(candidates)
        return candidates[:50]  # Limit pool size

    @timed("forms.generate_constrained")
    def generate_constrained_line(
        self, target_syllables: int, seed_words: Optional[List[str]] = None, max_attempts: int = 50
    ) -> Tuple[Optional[str], int]:
        """Generate a single line matching syllable constraint.

        Args:
            target_syllables: Target number of syllables for the line
            seed_words: Optional seed words to guide generation
            max_attempts: Maximum generation attempts before giving up

        Returns:
            Tuple of (generated line or None, actual syllable count)
        """
        seed_words = seed_words or []

        for _attempt in range(max_attempts):
            line_words = []
            current_syllables = 0

            # Start with a seed word or common word
            if seed_words and random.random() < 0.5:
                start_word = random.choice(seed_words)
            else:
                start_word = random.choice(vocabulary.common_words_by_syllables[1])

            start_syllables = count_syllables(start_word)
            if start_syllables <= target_syllables:
                line_words.append(start_word)
                current_syllables = start_syllables

            # Build the rest of the line
            last_word = start_word
            stuck_counter = 0
            max_stuck = 3

            while current_syllables < target_syllables:
                remaining = target_syllables - current_syllables

                # Get candidate words
                word_pool = self._get_word_pool(seed_words, last_word)

                # Find a word that fits
                found = False
                for candidate in word_pool:
                    syllables = count_syllables(candidate)

                    if syllables == remaining:
                        # Perfect fit!
                        line_words.append(candidate)
                        current_syllables += syllables
                        found = True
                        break
                    elif syllables < remaining:
                        # Room for more words
                        line_words.append(candidate)
                        current_syllables += syllables
                        last_word = candidate
                        found = True
                        break

                if not found:
                    # Try to fill with exact syllable count from common words
                    if remaining in vocabulary.common_words_by_syllables:
                        word = random.choice(vocabulary.common_words_by_syllables[remaining])
                        line_words.append(word)
                        current_syllables += remaining
                        break
                    else:
                        stuck_counter += 1
                        if stuck_counter >= max_stuck:
                            break  # Give up on this attempt

            # Check if we succeeded
            if current_syllables == target_syllables:
                return " ".join(line_words), current_syllables

            # Accept close matches
            if target_syllables - 1 <= current_syllables <= target_syllables:
                return " ".join(line_words), current_syllables

        # Failed to generate
        return None, 0

    @timed("forms.generate_haiku")
    def generate_haiku(
        self, seed_words: Optional[List[str]] = None, max_attempts: int = 100, strict: bool = True
    ) -> Tuple[List[str], FormValidationResult]:
        """Generate a haiku (5-7-5 syllable pattern).

        Haiku is a Japanese poetry form consisting of three lines with
        a 5-7-5 syllable pattern, traditionally focused on nature and seasons.

        Args:
            seed_words: Optional words to guide generation
            max_attempts: Maximum attempts before giving up
            strict: If True, requires exact syllable match. If False, allows ±1 syllable

        Returns:
            Tuple of (lines list, validation result)
        """
        pattern = [5, 7, 5]
        lines = []

        for target in pattern:
            line, actual = self.generate_constrained_line(target, seed_words, max_attempts)
            if line is None:
                raise ValueError(
                    f"Failed to generate line with {target} syllables after {max_attempts} attempts"
                )

            if strict and actual != target:
                raise ValueError(f"Generated line has {actual} syllables, expected {target}")

            lines.append(line)

        validation = self.validate_form(lines, pattern, "Haiku")
        return lines, validation

    @timed("forms.generate_tanka")
    def generate_tanka(
        self, seed_words: Optional[List[str]] = None, max_attempts: int = 100, strict: bool = True
    ) -> Tuple[List[str], FormValidationResult]:
        """Generate a tanka (5-7-5-7-7 syllable pattern).

        Tanka is a Japanese poetry form consisting of five lines with
        a 5-7-5-7-7 syllable pattern, often expressing personal emotions.

        Args:
            seed_words: Optional words to guide generation
            max_attempts: Maximum attempts before giving up
            strict: If True, requires exact syllable match. If False, allows ±1 syllable

        Returns:
            Tuple of (lines list, validation result)
        """
        pattern = [5, 7, 5, 7, 7]
        lines = []

        for target in pattern:
            line, actual = self.generate_constrained_line(target, seed_words, max_attempts)
            if line is None:
                raise ValueError(
                    f"Failed to generate line with {target} syllables after {max_attempts} attempts"
                )

            if strict and actual != target:
                raise ValueError(f"Generated line has {actual} syllables, expected {target}")

            lines.append(line)

        validation = self.validate_form(lines, pattern, "Tanka")
        return lines, validation

    @timed("forms.generate_senryu")
    def generate_senryu(
        self, seed_words: Optional[List[str]] = None, max_attempts: int = 100, strict: bool = True
    ) -> Tuple[List[str], FormValidationResult]:
        """Generate a senryu (5-7-5 syllable pattern, focused on human nature).

        Senryu uses the same 5-7-5 structure as haiku but focuses on human nature,
        relationships, and emotions rather than natural imagery.

        Args:
            seed_words: Optional words to guide generation
            max_attempts: Maximum attempts before giving up
            strict: If True, requires exact syllable match. If False, allows ±1 syllable

        Returns:
            Tuple of (lines list, validation result)
        """
        # Senryu has same structure as haiku, just different thematic focus
        # In practice, the generation is identical - the difference is semantic
        pattern = [5, 7, 5]
        lines = []

        for target in pattern:
            line, actual = self.generate_constrained_line(target, seed_words, max_attempts)
            if line is None:
                raise ValueError(
                    f"Failed to generate line with {target} syllables after {max_attempts} attempts"
                )

            if strict and actual != target:
                raise ValueError(f"Generated line has {actual} syllables, expected {target}")

            lines.append(line)

        validation = self.validate_form(lines, pattern, "Senryu")
        return lines, validation


def create_form_generator() -> FormGenerator:
    """Factory function to create a FormGenerator instance.

    Returns:
        Configured FormGenerator instance
    """
    return FormGenerator()

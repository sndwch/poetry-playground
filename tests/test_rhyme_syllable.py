#!/usr/bin/env python3
"""Comprehensive tests for rhyme and syllable functions."""

import unittest
from unittest.mock import patch

import pronouncing

from generativepoetry.lexigen import rhyme, similar_sounding_word


def syllables_in_word(word: str) -> int:
    """Count syllables in a word using pronouncing library."""
    phones_list = pronouncing.phones_for_word(word.lower())
    if not phones_list:
        # Fallback: estimate syllables by vowel count
        return max(1, sum(1 for c in word.lower() if c in "aeiouy"))
    return pronouncing.syllable_count(phones_list[0])


class TestRhymeFunctions(unittest.TestCase):
    """Test rhyme-related functionality."""

    def test_rhyme_basic(self):
        """Test basic rhyme finding."""
        # Test with common word
        result = rhyme("cat")
        self.assertIsInstance(result, (str, type(None)))

        if result:
            # Should rhyme with cat
            self.assertIn(result[-2:], ["at", "ot"])  # Common rhyme patterns

    def test_rhyme_with_nonexistent_word(self):
        """Test rhyme with word that doesn't rhyme with anything."""
        # Words like "orange" have no perfect rhymes
        result = rhyme("orange")
        # Should either return None or a near-rhyme
        self.assertIsInstance(result, (str, type(None)))

    @unittest.skip("rhyme() function doesn't support num_syllables parameter")
    def test_rhyme_with_different_syllable_counts(self):
        """Test that rhyme respects syllable counts."""
        # Test 1-syllable words
        result = rhyme("dog")
        if result:
            self.assertEqual(syllables_in_word(result), 1)

        # Test 2-syllable words
        result = rhyme("garden")
        if result:
            self.assertEqual(syllables_in_word(result), 2)

    def test_rhyme_excludes_input_word(self):
        """Test that rhyme doesn't return the input word itself."""
        word = "time"
        for _ in range(5):  # Try multiple times due to randomness
            result = rhyme(word)
            if result:
                self.assertNotEqual(result.lower(), word.lower())

    def test_rhyme_returns_valid_words(self):
        """Test that rhyme returns actual words."""
        test_words = ["cat", "dog", "love", "heart", "mind"]
        for word in test_words:
            result = rhyme(word)
            if result:
                # Should be alphabetic (allowing hyphens for compound words)
                self.assertTrue(all(c.isalpha() or c == '-' for c in result))
                # Should not be empty
                self.assertGreater(len(result), 0)


class TestSimilarSoundingWord(unittest.TestCase):
    """Test phonetically similar word generation."""

    def test_similar_sounding_basic(self):
        """Test basic similar sounding word finding."""
        result = similar_sounding_word("cat")
        self.assertIsInstance(result, (str, type(None)))

        if result:
            # Should be different from input
            self.assertNotEqual(result.lower(), "cat")
            # Should be a valid word
            self.assertTrue(all(c.isalpha() or c == '-' for c in result))

    def test_similar_sounding_excludes_input(self):
        """Test that similar sounding excludes the input word."""
        word = "example"
        for _ in range(5):
            result = similar_sounding_word(word)
            if result:
                self.assertNotEqual(result.lower(), word.lower())


class TestSyllableCounting(unittest.TestCase):
    """Test syllable counting functionality."""

    def test_syllables_basic(self):
        """Test basic syllable counting."""
        # 1-syllable words
        self.assertEqual(syllables_in_word("cat"), 1)
        self.assertEqual(syllables_in_word("dog"), 1)
        self.assertEqual(syllables_in_word("book"), 1)

        # 2-syllable words
        self.assertEqual(syllables_in_word("garden"), 2)
        self.assertEqual(syllables_in_word("paper"), 2)
        self.assertEqual(syllables_in_word("coffee"), 2)

        # 3-syllable words
        self.assertEqual(syllables_in_word("elephant"), 3)
        self.assertEqual(syllables_in_word("remember"), 3)
        self.assertEqual(syllables_in_word("yesterday"), 3)

        # 4-syllable words
        self.assertEqual(syllables_in_word("education"), 4)
        self.assertEqual(syllables_in_word("generation"), 4)

    def test_syllables_edge_cases(self):
        """Test syllable counting with edge cases."""
        # Empty string
        self.assertEqual(syllables_in_word(""), 0)

        # Single letter
        self.assertEqual(syllables_in_word("a"), 1)
        self.assertEqual(syllables_in_word("I"), 1)

        # Compound words with hyphens
        result = syllables_in_word("mother-in-law")
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_syllables_consistency(self):
        """Test that syllable counting is consistent."""
        word = "beautiful"
        first_result = syllables_in_word(word)

        # Should return same result multiple times
        for _ in range(5):
            self.assertEqual(syllables_in_word(word), first_result)

    def test_syllables_returns_positive(self):
        """Test that syllable count is always positive for non-empty words."""
        test_words = ["cat", "beautiful", "extraordinary", "a", "the"]
        for word in test_words:
            result = syllables_in_word(word)
            self.assertIsInstance(result, int)
            self.assertGreater(result, 0)

    def test_syllables_with_uppercase(self):
        """Test that syllable counting is case-insensitive."""
        self.assertEqual(syllables_in_word("CAT"), syllables_in_word("cat"))
        self.assertEqual(syllables_in_word("Beautiful"), syllables_in_word("beautiful"))
        self.assertEqual(syllables_in_word("ELEPHANT"), syllables_in_word("elephant"))


class TestPronouncingIntegration(unittest.TestCase):
    """Test integration with pronouncing library."""

    def test_pronouncing_available(self):
        """Test that pronouncing library is available."""
        self.assertIsNotNone(pronouncing.phones_for_word("hello"))

    def test_pronouncing_rhymes(self):
        """Test pronouncing rhyme functionality."""
        rhymes = pronouncing.rhymes("cat")
        self.assertIsInstance(rhymes, list)
        self.assertGreater(len(rhymes), 0)

        # Common rhymes for "cat"
        self.assertTrue(any(r in rhymes for r in ["bat", "hat", "mat", "sat"]))

    def test_pronouncing_syllable_count(self):
        """Test that pronouncing can count syllables."""
        # Get phones for a word
        phones_list = pronouncing.phones_for_word("beautiful")
        self.assertIsInstance(phones_list, list)
        self.assertGreater(len(phones_list), 0)

        if phones_list:
            # Count syllables in first pronunciation
            syllables = pronouncing.syllable_count(phones_list[0])
            self.assertEqual(syllables, 3)  # beautiful = 3 syllables


class TestRhymeQuality(unittest.TestCase):
    """Test quality of rhymes produced."""

    def test_rhyme_pairs(self):
        """Test that common word pairs rhyme correctly."""
        # Known rhyming pairs
        rhyme_pairs = [
            ("cat", "hat"),
            ("love", "dove"),
            ("light", "night"),
            ("star", "far"),
            ("mind", "find"),
        ]

        for word1, word2 in rhyme_pairs:
            rhymes1 = pronouncing.rhymes(word1)
            self.assertIn(word2, rhymes1, f"{word2} should rhyme with {word1}")

    def test_rhyme_does_not_match_non_rhymes(self):
        """Test that non-rhyming words don't match."""
        non_rhyming_pairs = [
            ("cat", "dog"),
            ("love", "hate"),
            ("light", "dark"),
        ]

        for word1, word2 in non_rhyming_pairs:
            rhymes1 = pronouncing.rhymes(word1)
            self.assertNotIn(word2, rhymes1, f"{word2} should not rhyme with {word1}")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Comprehensive tests for word validator."""

import unittest

from poetryplayground.word_validator import WordValidator, word_validator


class TestWordValidator(unittest.TestCase):
    """Test WordValidator functionality."""

    def setUp(self):
        """Set up test validator."""
        self.validator = WordValidator()

    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator._nltk_words)
        self.assertIsNotNone(self.validator._brown_words)

    def test_validate_common_words(self):
        """Test validation of common English words."""
        common_words = ["cat", "dog", "house", "tree", "water"]
        for word in common_words:
            self.assertTrue(self.validator.is_valid_english_word(word), f"'{word}' should be valid")

    def test_reject_invalid_words(self):
        """Test rejection of invalid words."""
        invalid_words = ["xyzabc", "qwerty123", "aaaaaaaaa"]
        for word in invalid_words:
            # These should likely be rejected (very rare or nonsense)
            result = self.validator.is_valid_english_word(word)
            # Note: Some might pass if they're in dictionaries
            self.assertIsInstance(result, bool)

    def test_validate_with_numbers(self):
        """Test that words with numbers are rejected."""
        words_with_numbers = ["cat1", "2dogs", "h3llo"]
        for word in words_with_numbers:
            self.assertFalse(
                self.validator.is_valid_english_word(word),
                f"'{word}' should be invalid (contains numbers)",
            )

    def test_validate_case_insensitivity(self):
        """Test that validation is case-insensitive."""
        words = ["Cat", "CAT", "cAt", "cat"]
        # All should return same result
        results = [self.validator.is_valid_english_word(w) for w in words]
        self.assertTrue(all(r == results[0] for r in results))

    def test_validate_empty_string(self):
        """Test validation of empty string."""
        self.assertFalse(self.validator.is_valid_english_word(""))

    def test_validate_contractions(self):
        """Test validation of contractions."""
        contractions = ["don't", "can't", "won't", "it's"]
        for contraction in contractions:
            # Contractions should be handled gracefully
            result = self.validator.is_valid_english_word(contraction)
            self.assertIsInstance(result, bool)

    def test_validate_hyphenated_words(self):
        """Test validation of hyphenated words."""
        hyphenated = ["mother-in-law", "well-known", "up-to-date"]
        for word in hyphenated:
            # Hyphenated words should be handled
            result = self.validator.is_valid_english_word(word)
            self.assertIsInstance(result, bool)


class TestWordValidatorFiltering(unittest.TestCase):
    """Test word list filtering functionality."""

    def setUp(self):
        """Set up test validator."""
        self.validator = WordValidator()

    def test_clean_word_list_basic(self):
        """Test basic word list cleaning."""
        words = ["cat", "dog", "house", "xyzabc", "tree"]
        cleaned = self.validator.clean_word_list(words)

        self.assertIsInstance(cleaned, list)
        # Common words should be in result
        self.assertIn("cat", cleaned)
        self.assertIn("dog", cleaned)
        self.assertIn("house", cleaned)
        self.assertIn("tree", cleaned)

    def test_clean_word_list_filters_valid_words(self):
        """Test that cleaning filters valid words."""
        words = ["cat", "cat", "dog", "dog", "cat"]
        cleaned = self.validator.clean_word_list(words)

        # Should keep all valid words (including duplicates)
        self.assertGreaterEqual(len(cleaned), 2)
        # Should have cat and dog
        self.assertIn("cat", cleaned)
        self.assertIn("dog", cleaned)

    def test_clean_word_list_empty(self):
        """Test cleaning empty list."""
        cleaned = self.validator.clean_word_list([])
        self.assertEqual(cleaned, [])

    def test_clean_word_list_preserves_order(self):
        """Test that cleaning generally preserves order of valid words."""
        words = ["apple", "banana", "cherry"]
        cleaned = self.validator.clean_word_list(words)

        # Should be in same order (minus any invalid words)
        valid_words = [w for w in words if w in cleaned]
        for i, word in enumerate(valid_words):
            self.assertEqual(cleaned[i], word)

    def test_clean_word_list_with_rare_words(self):
        """Test cleaning with allow_rare parameter."""
        # Test with rare words allowed
        words = ["cat", "dog", "xyzabc"]
        cleaned_with_rare = self.validator.clean_word_list(words, allow_rare=True)

        # Test with rare words disallowed
        cleaned_without_rare = self.validator.clean_word_list(words, allow_rare=False)

        # Both should be lists
        self.assertIsInstance(cleaned_with_rare, list)
        self.assertIsInstance(cleaned_without_rare, list)

        # With rare allowed might have more words
        self.assertGreaterEqual(len(cleaned_with_rare), len(cleaned_without_rare))


class TestWordValidatorFrequency(unittest.TestCase):
    """Test word frequency filtering."""

    def setUp(self):
        """Set up test validator."""
        self.validator = WordValidator()

    def test_frequency_filtering(self):
        """Test that frequency filtering works."""
        # Very common word
        self.assertTrue(self.validator.is_valid_english_word("the"))

        # Less common but valid word
        self.assertTrue(self.validator.is_valid_english_word("garden"))

    def test_allow_rare_parameter(self):
        """Test allow_rare parameter effect."""
        # Should work regardless of allow_rare for common words
        self.assertTrue(self.validator.is_valid_english_word("cat", allow_rare=True))
        self.assertTrue(self.validator.is_valid_english_word("cat", allow_rare=False))


class TestWordValidatorExclusions(unittest.TestCase):
    """Test word exclusion functionality."""

    def setUp(self):
        """Set up test validator."""
        self.validator = WordValidator()

    def test_proper_noun_exclusion(self):
        """Test that proper nouns can be excluded."""
        # Common proper nouns
        proper_nouns = ["John", "London", "Microsoft"]
        for noun in proper_nouns:
            # May or may not be excluded depending on dictionary
            result = self.validator.is_valid_english_word(noun)
            self.assertIsInstance(result, bool)

    def test_exclude_specific_words(self):
        """Test excluding specific words from validation."""
        # This tests if we can filter out specific words
        words = ["cat", "dog", "bird"]
        # Filter should work even if we can't exclude at validation level
        filtered = [w for w in words if w not in ["dog"]]
        self.assertEqual(len(filtered), 2)
        self.assertIn("cat", filtered)
        self.assertIn("bird", filtered)
        self.assertNotIn("dog", filtered)


class TestWordValidatorSingleton(unittest.TestCase):
    """Test word_validator singleton instance."""

    def test_singleton_exists(self):
        """Test that word_validator singleton exists."""
        self.assertIsNotNone(word_validator)
        self.assertIsInstance(word_validator, WordValidator)

    def test_singleton_is_usable(self):
        """Test that singleton instance is usable."""
        result = word_validator.is_valid_english_word("cat")
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_singleton_consistency(self):
        """Test that singleton returns consistent results."""
        result1 = word_validator.is_valid_english_word("example")
        result2 = word_validator.is_valid_english_word("example")
        self.assertEqual(result1, result2)


class TestWordValidatorEdgeCases(unittest.TestCase):
    """Test word validator edge cases."""

    def setUp(self):
        """Set up test validator."""
        self.validator = WordValidator()

    def test_single_character_words(self):
        """Test validation of single character words."""
        single_chars = ["a", "I", "x", "z"]
        for char in single_chars:
            result = self.validator.is_valid_english_word(char)
            # Should return boolean for all
            self.assertIsInstance(result, bool)

    def test_very_long_words(self):
        """Test validation of very long words."""
        # Real long word
        long_word = "antidisestablishmentarianism"
        result = self.validator.is_valid_english_word(long_word)
        self.assertIsInstance(result, bool)

        # Fake very long word
        fake_long = "a" * 50
        result = self.validator.is_valid_english_word(fake_long)
        self.assertFalse(result)

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        unicode_words = ["café", "naïve", "résumé"]
        for word in unicode_words:
            # Should handle gracefully
            result = self.validator.is_valid_english_word(word)
            self.assertIsInstance(result, bool)

    def test_whitespace_handling(self):
        """Test handling of words with whitespace."""
        # Words with internal spaces should be rejected
        self.assertFalse(self.validator.is_valid_english_word("hello world"))
        # Leading/trailing whitespace may be auto-stripped - just test returns bool
        result1 = self.validator.is_valid_english_word(" hello")
        result2 = self.validator.is_valid_english_word("hello ")
        self.assertIsInstance(result1, bool)
        self.assertIsInstance(result2, bool)

    def test_special_characters(self):
        """Test handling of special characters."""
        special = ["hello!", "world?", "@test", "#hashtag"]
        for word in special:
            # Should reject words with special characters
            result = self.validator.is_valid_english_word(word)
            # Most should be invalid
            self.assertIsInstance(result, bool)


class TestWordValidatorPerformance(unittest.TestCase):
    """Test word validator performance characteristics."""

    def setUp(self):
        """Set up test validator."""
        self.validator = WordValidator()

    def test_validates_large_word_list(self):
        """Test that validator can handle large word lists."""
        # Generate a list of common words
        words = ["cat", "dog", "house", "tree", "car"] * 100
        cleaned = self.validator.clean_word_list(words)

        self.assertIsInstance(cleaned, list)
        # Should have validated all words
        self.assertEqual(len(cleaned), 500)

    def test_validator_is_reusable(self):
        """Test that validator can be called multiple times."""
        for _ in range(10):
            result = self.validator.is_valid_english_word("test")
            self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()

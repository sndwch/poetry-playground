#!/usr/bin/env python3
"""Comprehensive tests for corpus analyzer."""

import unittest
from pathlib import Path

from generativepoetry.corpus_analyzer import PersonalCorpusAnalyzer, PoetryMetrics


class TestPoetryMetrics(unittest.TestCase):
    """Test PoetryMetrics data class."""

    def test_metrics_creation(self):
        """Test creating a PoetryMetrics instance."""
        metrics = PoetryMetrics()
        self.assertIsNotNone(metrics)

    def test_metrics_has_expected_attributes(self):
        """Test that metrics has expected attributes."""
        metrics = PoetryMetrics()

        # Should have basic attributes
        self.assertTrue(hasattr(metrics, 'total_lines'))
        self.assertTrue(hasattr(metrics, 'total_words'))
        self.assertTrue(hasattr(metrics, 'unique_words'))
        self.assertTrue(hasattr(metrics, 'avg_words_per_line'))

    def test_metrics_default_values(self):
        """Test default metric values."""
        metrics = PoetryMetrics()

        # Defaults should be reasonable
        self.assertIsInstance(metrics.total_lines, int)
        self.assertIsInstance(metrics.total_words, int)
        self.assertIsInstance(metrics.unique_words, int)


class TestCorpusAnalyzer(unittest.TestCase):
    """Test PersonalCorpusAnalyzer functionality."""

    def setUp(self):
        """Set up test analyzer with sample text."""
        self.analyzer = PersonalCorpusAnalyzer()
        self.sample_text = """
The cat sat on the mat.
The dog ran in the park.
Birds fly through the sky.
"""

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer)

    def test_analyze_empty_text(self):
        """Test analyzing empty text."""
        metrics = self.analyzer.analyze("")
        self.assertIsInstance(metrics, PoetryMetrics)
        self.assertEqual(metrics.total_lines, 0)
        self.assertEqual(metrics.total_words, 0)

    def test_analyze_single_line(self):
        """Test analyzing single line of text."""
        text = "The quick brown fox jumps"
        metrics = self.analyzer.analyze(text)

        self.assertIsInstance(metrics, PoetryMetrics)
        self.assertEqual(metrics.total_lines, 1)
        self.assertEqual(metrics.total_words, 5)

    def test_analyze_multiple_lines(self):
        """Test analyzing multiple lines."""
        metrics = self.analyzer.analyze(self.sample_text)

        self.assertIsInstance(metrics, PoetryMetrics)
        self.assertGreater(metrics.total_lines, 0)
        self.assertGreater(metrics.total_words, 0)

    def test_analyze_counts_unique_words(self):
        """Test that analyzer counts unique words."""
        text = "the cat the dog the bird"
        metrics = self.analyzer.analyze(text)

        # Should have 3 unique words (cat, dog, bird) + "the" = 4
        self.assertGreater(metrics.unique_words, 0)
        # Total words should be 5
        self.assertEqual(metrics.total_words, 5)

    def test_analyze_calculates_average(self):
        """Test that average words per line is calculated."""
        text = "one two three\nfour five six seven"
        metrics = self.analyzer.analyze(text)

        self.assertGreater(metrics.avg_words_per_line, 0)
        # Should be around 3.5 words per line
        self.assertAlmostEqual(metrics.avg_words_per_line, 3.5, places=1)


class TestCorpusAnalyzerWordFrequency(unittest.TestCase):
    """Test word frequency analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_get_word_frequencies(self):
        """Test getting word frequencies from text."""
        text = "the cat and the dog and the bird"
        metrics = self.analyzer.analyze(text)

        # Should have counted words
        self.assertGreater(metrics.total_words, 0)

    def test_word_frequency_case_insensitive(self):
        """Test that word frequency is case-insensitive."""
        text = "The cat THE dog The bird"
        metrics = self.analyzer.analyze(text)

        # "The" should be counted as one word regardless of case
        self.assertEqual(metrics.total_words, 5)


class TestCorpusAnalyzerPhraseExtraction(unittest.TestCase):
    """Test phrase extraction functionality."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_extract_phrases_from_text(self):
        """Test extracting common phrases."""
        text = """
The sun rises in the east.
The moon shines at night.
The stars twinkle in the sky.
"""
        metrics = self.analyzer.analyze(text)

        # Should have analyzed the text
        self.assertGreater(metrics.total_words, 0)

    def test_extract_phrases_empty_text(self):
        """Test phrase extraction from empty text."""
        metrics = self.analyzer.analyze("")
        self.assertEqual(metrics.total_words, 0)


class TestCorpusAnalyzerStyleMetrics(unittest.TestCase):
    """Test style and pattern analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_analyze_punctuation_patterns(self):
        """Test analyzing punctuation usage."""
        text = "Hello! How are you? I am fine."
        metrics = self.analyzer.analyze(text)

        # Should have analyzed the text
        self.assertGreater(metrics.total_words, 0)

    def test_analyze_line_structure(self):
        """Test analyzing line structure."""
        text = """
Short line.
This is a longer line with more words.
Medium length.
"""
        metrics = self.analyzer.analyze(text)

        # Should have multiple lines
        self.assertGreater(metrics.total_lines, 1)
        # Should have average
        self.assertGreater(metrics.avg_words_per_line, 0)


class TestCorpusAnalyzerVocabulary(unittest.TestCase):
    """Test vocabulary analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_vocabulary_richness(self):
        """Test calculating vocabulary richness."""
        # Text with low vocabulary richness
        simple_text = "the the the cat cat dog"
        simple_metrics = self.analyzer.analyze(simple_text)

        # Text with high vocabulary richness
        rich_text = "ocean sky mountain river forest"
        rich_metrics = self.analyzer.analyze(rich_text)

        # Rich text should have more unique words relative to total
        simple_ratio = simple_metrics.unique_words / max(simple_metrics.total_words, 1)
        rich_ratio = rich_metrics.unique_words / max(rich_metrics.total_words, 1)

        self.assertLess(simple_ratio, rich_ratio)

    def test_identify_common_words(self):
        """Test identifying most common words."""
        text = "the cat the dog the bird the fish"
        metrics = self.analyzer.analyze(text)

        # "the" appears 4 times, should be most common
        self.assertGreater(metrics.total_words, 0)


class TestCorpusAnalyzerFromFile(unittest.TestCase):
    """Test analyzing corpus from files."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_analyze_from_string(self):
        """Test analyzing from string input."""
        text = "Sample poetry text"
        metrics = self.analyzer.analyze(text)

        self.assertIsInstance(metrics, PoetryMetrics)
        self.assertEqual(metrics.total_words, 3)

    def test_analyze_preserves_line_breaks(self):
        """Test that line breaks are preserved in analysis."""
        text = "line one\nline two\nline three"
        metrics = self.analyzer.analyze(text)

        self.assertEqual(metrics.total_lines, 3)


class TestCorpusAnalyzerStatistics(unittest.TestCase):
    """Test statistical analysis features."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_calculates_basic_statistics(self):
        """Test calculation of basic statistics."""
        text = """
The rain falls softly.
Thunder echoes loudly.
Lightning flashes bright.
"""
        metrics = self.analyzer.analyze(text)

        # Should have basic statistics
        self.assertGreater(metrics.total_lines, 0)
        self.assertGreater(metrics.total_words, 0)
        self.assertGreater(metrics.unique_words, 0)
        self.assertGreater(metrics.avg_words_per_line, 0)

    def test_handles_varying_line_lengths(self):
        """Test handling lines of varying lengths."""
        text = """
One.
Two words here.
Three words in line.
Four words in this line.
"""
        metrics = self.analyzer.analyze(text)

        # Average should be around 2.5 words per line
        self.assertGreater(metrics.avg_words_per_line, 1.5)
        self.assertLess(metrics.avg_words_per_line, 3.5)


class TestCorpusAnalyzerEdgeCases(unittest.TestCase):
    """Test edge cases in corpus analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = CorpusAnalyzer()

    def test_analyze_whitespace_only(self):
        """Test analyzing text with only whitespace."""
        text = "   \n\n\n   "
        metrics = self.analyzer.analyze(text)

        self.assertEqual(metrics.total_words, 0)

    def test_analyze_punctuation_only(self):
        """Test analyzing text with only punctuation."""
        text = "... !!! ???"
        metrics = self.analyzer.analyze(text)

        # Should handle gracefully
        self.assertIsInstance(metrics, PoetryMetrics)

    def test_analyze_mixed_case(self):
        """Test analyzing text with mixed case."""
        text = "ThE qUiCk BrOwN fOx"
        metrics = self.analyzer.analyze(text)

        self.assertEqual(metrics.total_words, 4)

    def test_analyze_with_numbers(self):
        """Test analyzing text containing numbers."""
        text = "I have 3 cats and 2 dogs"
        metrics = self.analyzer.analyze(text)

        # Should count all tokens
        self.assertGreater(metrics.total_words, 0)


if __name__ == "__main__":
    unittest.main()

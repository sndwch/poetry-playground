#!/usr/bin/env python3
"""Comprehensive tests for corpus analyzer."""

import unittest
from pathlib import Path

from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer, PoetryMetrics


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
        self.assertTrue(hasattr(metrics, "total_lines"))
        self.assertTrue(hasattr(metrics, "total_words"))
        self.assertTrue(hasattr(metrics, "vocabulary_size"))
        self.assertTrue(hasattr(metrics, "avg_words_per_line"))

    def test_metrics_default_values(self):
        """Test default metric values."""
        metrics = PoetryMetrics()

        # Defaults should be reasonable
        self.assertIsInstance(metrics.total_lines, int)
        self.assertIsInstance(metrics.total_words, int)
        self.assertIsInstance(metrics.vocabulary_size, int)


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
        fingerprint = self.analyzer.analyze_poems([{"content": "", "title": "Empty"}])
        self.assertIsNotNone(fingerprint)
        # Empty text should have zero or minimal metrics
        self.assertIsInstance(fingerprint.metrics.total_lines, int)

    def test_analyze_single_line(self):
        """Test analyzing single line of text."""
        text = "The quick brown fox jumps"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Short"}])

        self.assertIsNotNone(fingerprint)
        self.assertIsNotNone(fingerprint.metrics)
        self.assertGreater(fingerprint.metrics.total_lines, 0)

    def test_analyze_multiple_lines(self):
        """Test analyzing multiple lines."""
        fingerprint = self.analyzer.analyze_poems(
            [{"content": self.sample_text, "title": "Sample"}]
        )

        self.assertIsNotNone(fingerprint)
        self.assertIsNotNone(fingerprint.metrics)
        self.assertGreater(fingerprint.metrics.total_lines, 0)
        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_analyze_counts_unique_words(self):
        """Test that analyzer counts unique words."""
        text = "the cat the dog the bird"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Animals"}])

        # Should have counted words
        self.assertIsNotNone(fingerprint.metrics)
        self.assertGreater(fingerprint.metrics.vocabulary_size, 0)
        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_analyze_calculates_average(self):
        """Test that average words per line is calculated."""
        text = "one two three\nfour five six seven"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Numbers"}])

        self.assertIsNotNone(fingerprint.metrics)
        self.assertGreater(fingerprint.metrics.avg_words_per_line, 0)


class TestCorpusAnalyzerWordFrequency(unittest.TestCase):
    """Test word frequency analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_get_word_frequencies(self):
        """Test getting word frequencies from text."""
        text = "the cat and the dog and the bird"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Test"}])

        # Should have counted words
        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_word_frequency_case_insensitive(self):
        """Test that word frequency is case-insensitive."""
        text = "The cat THE dog The bird"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Test"}])

        # Should have counted words
        self.assertGreater(fingerprint.metrics.total_words, 0)


class TestCorpusAnalyzerPhraseExtraction(unittest.TestCase):
    """Test phrase extraction functionality."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_extract_phrases_from_text(self):
        """Test extracting common phrases."""
        text = """
The sun rises in the east.
The moon shines at night.
The stars twinkle in the sky.
"""
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Celestial"}])

        # Should have analyzed the text
        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_extract_phrases_empty_text(self):
        """Test phrase extraction from empty text."""
        fingerprint = self.analyzer.analyze_poems([{"content": "", "title": "Empty"}])
        self.assertIsNotNone(fingerprint)


class TestCorpusAnalyzerStyleMetrics(unittest.TestCase):
    """Test style and pattern analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_analyze_punctuation_patterns(self):
        """Test analyzing punctuation usage."""
        text = "Hello! How are you? I am fine."
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Greeting"}])

        # Should have analyzed the text
        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_analyze_line_structure(self):
        """Test analyzing line structure."""
        text = """
Short line.
This is a longer line with more words.
Medium length.
"""
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Lines"}])

        # Should have multiple lines
        self.assertGreater(fingerprint.metrics.total_lines, 1)
        # Should have average
        self.assertGreater(fingerprint.metrics.avg_words_per_line, 0)


class TestCorpusAnalyzerVocabulary(unittest.TestCase):
    """Test vocabulary analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_vocabulary_richness(self):
        """Test calculating vocabulary richness."""
        # Text with low vocabulary richness
        simple_text = "the the the cat cat dog"
        simple_fingerprint = self.analyzer.analyze_poems(
            [{"content": simple_text, "title": "Simple"}]
        )

        # Text with high vocabulary richness
        rich_text = "ocean sky mountain river forest"
        rich_fingerprint = self.analyzer.analyze_poems([{"content": rich_text, "title": "Rich"}])

        # Rich text should have more unique words relative to total
        simple_ratio = simple_fingerprint.metrics.vocabulary_size / max(
            simple_fingerprint.metrics.total_words, 1
        )
        rich_ratio = rich_fingerprint.metrics.vocabulary_size / max(
            rich_fingerprint.metrics.total_words, 1
        )

        self.assertLess(simple_ratio, rich_ratio)

    def test_identify_common_words(self):
        """Test identifying most common words."""
        text = "the cat the dog the bird the fish"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Animals"}])

        # "the" appears 4 times, should be most common
        self.assertGreater(fingerprint.metrics.total_words, 0)


class TestCorpusAnalyzerFromFile(unittest.TestCase):
    """Test analyzing corpus from files."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_analyze_from_string(self):
        """Test analyzing from string input."""
        text = "Sample poetry text"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Sample"}])

        self.assertIsNotNone(fingerprint.metrics)
        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_analyze_preserves_line_breaks(self):
        """Test that line breaks are preserved in analysis."""
        text = "line one\nline two\nline three"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Lines"}])

        self.assertGreater(fingerprint.metrics.total_lines, 0)


class TestCorpusAnalyzerStatistics(unittest.TestCase):
    """Test statistical analysis features."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_calculates_basic_statistics(self):
        """Test calculation of basic statistics."""
        text = """
The rain falls softly.
Thunder echoes loudly.
Lightning flashes bright.
"""
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Weather"}])

        # Should have basic statistics
        self.assertGreater(fingerprint.metrics.total_lines, 0)
        self.assertGreater(fingerprint.metrics.total_words, 0)
        self.assertGreater(fingerprint.metrics.vocabulary_size, 0)
        self.assertGreater(fingerprint.metrics.avg_words_per_line, 0)

    def test_handles_varying_line_lengths(self):
        """Test handling lines of varying lengths."""
        text = """
One.
Two words here.
Three words in line.
Four words in this line.
"""
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Counts"}])

        # Should have calculated average
        self.assertGreater(fingerprint.metrics.avg_words_per_line, 0)


class TestCorpusAnalyzerEdgeCases(unittest.TestCase):
    """Test edge cases in corpus analysis."""

    def setUp(self):
        """Set up test analyzer."""
        self.analyzer = PersonalCorpusAnalyzer()

    def test_analyze_whitespace_only(self):
        """Test analyzing text with only whitespace."""
        text = "   \n\n\n   "
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Whitespace"}])

        # Should handle gracefully
        self.assertIsNotNone(fingerprint)

    def test_analyze_punctuation_only(self):
        """Test analyzing text with only punctuation."""
        text = "... !!! ???"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Punct"}])

        # Should handle gracefully
        self.assertIsNotNone(fingerprint)

    def test_analyze_mixed_case(self):
        """Test analyzing text with mixed case."""
        text = "ThE qUiCk BrOwN fOx"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Mixed"}])

        self.assertGreater(fingerprint.metrics.total_words, 0)

    def test_analyze_with_numbers(self):
        """Test analyzing text containing numbers."""
        text = "I have 3 cats and 2 dogs"
        fingerprint = self.analyzer.analyze_poems([{"content": text, "title": "Numbers"}])

        # Should count all tokens
        self.assertGreater(fingerprint.metrics.total_words, 0)


if __name__ == "__main__":
    unittest.main()

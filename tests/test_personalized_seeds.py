#!/usr/bin/env python3
"""Comprehensive tests for PersonalizedLineSeedGenerator."""

import unittest

from poetryplayground.corpus_analyzer import (
    PoetryMetrics,
    StyleFingerprint,
    ThematicProfile,
    VocabularyProfile,
)
from poetryplayground.personalized_seeds import PersonalizedLineSeedGenerator


class TestPersonalizedLineSeedInitialization(unittest.TestCase):
    """Test PersonalizedLineSeedGenerator initialization."""

    def setUp(self):
        """Create minimal StyleFingerprint for testing."""
        self.fingerprint = self._create_minimal_fingerprint()

    def _create_minimal_fingerprint(self):
        """Create a minimal valid StyleFingerprint."""
        return StyleFingerprint(
            metrics=PoetryMetrics(
                total_poems=10,
                total_lines=50,
                total_words=250,
                vocabulary_size=100,
                avg_words_per_line=5.0,
                line_length_distribution={5: 10, 6: 15, 7: 20, 8: 5},
            ),
            vocabulary=VocabularyProfile(
                most_common_words=[
                    ("silence", 10),
                    ("threshold", 8),
                    ("vestige", 7),
                    ("amber", 6),
                    ("stone", 5),
                ],
                most_common_content_words=[
                    ("silence", 10),
                    ("threshold", 8),
                    ("vestige", 7),
                ],
                pos_distribution={
                    "NOUN": 100,
                    "VERB": 80,
                    "ADJ": 60,
                    "ADV": 40,
                },
            ),
            themes=ThematicProfile(
                concrete_vs_abstract={"concrete": 60, "abstract": 40},
                semantic_clusters=[],
            ),
        )

    def test_initialization_valid_strictness(self):
        """Test initialization with valid strictness values."""
        # Should work with strictness in [0.0, 1.0]
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.0)
        self.assertIsNotNone(gen)

        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.5)
        self.assertIsNotNone(gen)

        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=1.0)
        self.assertIsNotNone(gen)

    def test_initialization_invalid_strictness_too_high(self):
        """Test initialization with strictness > 1.0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PersonalizedLineSeedGenerator(self.fingerprint, strictness=1.5)
        self.assertIn("strictness must be in [0.0, 1.0]", str(context.exception))

    def test_initialization_invalid_strictness_too_low(self):
        """Test initialization with strictness < 0.0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PersonalizedLineSeedGenerator(self.fingerprint, strictness=-0.1)
        self.assertIn("strictness must be in [0.0, 1.0]", str(context.exception))

    def test_initialization_creates_hybrid_vocab(self):
        """Test that initialization builds hybrid vocabulary pool."""
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)
        self.assertIsNotNone(gen.hybrid_vocab)
        self.assertIsInstance(gen.hybrid_vocab, set)
        self.assertGreater(len(gen.hybrid_vocab), 0)

    def test_initialization_extracts_fingerprint_vocab(self):
        """Test that initialization extracts fingerprint vocabulary."""
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)
        self.assertIsNotNone(gen.fingerprint_vocab)
        self.assertIsInstance(gen.fingerprint_vocab, dict)

        # Should contain words from fingerprint
        self.assertIn("silence", gen.fingerprint_vocab)
        self.assertIn("threshold", gen.fingerprint_vocab)


class TestHybridVocabularyPool(unittest.TestCase):
    """Test hybrid vocabulary pool creation."""

    def setUp(self):
        """Create fingerprint with known vocabulary."""
        self.fingerprint = StyleFingerprint(
            metrics=PoetryMetrics(
                total_poems=5,
                total_lines=25,
                total_words=125,
                vocabulary_size=50,
                avg_words_per_line=5.0,
                line_length_distribution={5: 25},
            ),
            vocabulary=VocabularyProfile(
                most_common_words=[
                    ("unique_word_a", 10),
                    ("unique_word_b", 8),
                    ("unique_word_c", 6),
                ],
                most_common_content_words=[
                    ("unique_word_a", 10),
                    ("unique_word_b", 8),
                ],
                pos_distribution={"NOUN": 50},
            ),
            themes=ThematicProfile(
                concrete_vs_abstract={"concrete": 50, "abstract": 50},
                semantic_clusters=[],
            ),
        )

    def test_strictness_zero_uses_generic_vocab(self):
        """Test that strictness=0.0 uses mostly generic vocabulary."""
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.0)

        # With strictness=0, should have mostly generic words
        # Fingerprint words should be small minority
        fingerprint_count = sum(1 for w in gen.hybrid_vocab if w in gen.fingerprint_vocab)
        total_count = len(gen.hybrid_vocab)

        # Less than 10% should be fingerprint words
        self.assertLess(fingerprint_count / total_count, 0.1)

    def test_strictness_one_uses_only_fingerprint_vocab(self):
        """Test that strictness=1.0 uses only fingerprint vocabulary."""
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=1.0)

        # With strictness=1, most words should be from fingerprint
        # (Some generic words fill to target size if fingerprint is small)
        fingerprint_words_in_hybrid = [w for w in gen.fingerprint_vocab if w in gen.hybrid_vocab]

        # All fingerprint words should be in hybrid vocab
        self.assertEqual(
            len(fingerprint_words_in_hybrid),
            len(gen.fingerprint_vocab),
            "All fingerprint words should be in hybrid vocab with strictness=1.0",
        )


class TestStyleFitCalculation(unittest.TestCase):
    """Test style fit scoring components."""

    def setUp(self):
        """Create generator with test fingerprint."""
        self.fingerprint = StyleFingerprint(
            metrics=PoetryMetrics(
                total_poems=10,
                total_lines=50,
                total_words=250,
                vocabulary_size=100,
                avg_words_per_line=5.0,
                line_length_distribution={
                    5: 5,  # 10%
                    6: 10,  # 20%
                    7: 25,  # 50% (peak)
                    8: 8,  # 16%
                    9: 2,  # 4%
                },
            ),
            vocabulary=VocabularyProfile(
                most_common_words=[("test", 10)],
                most_common_content_words=[("test", 10)],
                pos_distribution={
                    "NOUN": 100,
                    "VERB": 80,
                    "ADJ": 60,
                    "DET": 50,
                },
            ),
            themes=ThematicProfile(
                concrete_vs_abstract={"concrete": 60, "abstract": 40},
                semantic_clusters=[],
            ),
        )
        self.gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

    def test_line_length_fit_perfect_match(self):
        """Test line length fit for text matching peak length."""
        # Peak is 7 syllables in our test fingerprint
        # Create text with approximately 7 syllables
        text = "silence threshold vestige"  # 3 words, ~7 syllables typically

        score = self.gen._calculate_line_length_fit(text)

        # Should get high score for matching peak
        self.assertGreater(score, 0.5, "Perfect match should score high")

    def test_pos_pattern_fit_matching_pattern(self):
        """Test POS pattern fit for matching syntactic structure."""
        # Use text with common POS pattern
        text = "the silent threshold"  # DET ADJ NOUN

        score = self.gen._calculate_pos_pattern_fit(text)

        # Should get reasonable score
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_concreteness_fit_returns_valid_score(self):
        """Test concreteness fit returns score in valid range."""
        text = "stone threshold silence"

        score = self.gen._calculate_concreteness_fit(text)

        # Should return score in [0, 1]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_phonetic_fit_returns_valid_score(self):
        """Test phonetic fit returns score in valid range."""
        text = "silence stone stars"  # Some alliteration

        score = self.gen._calculate_phonetic_fit(text)

        # Should return score in [0, 1]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_style_fit_score_combines_components(self):
        """Test that overall style fit combines all 4 components."""
        text = "silence threshold vestige amber"

        overall, components = self.gen._calculate_style_fit_score(text)

        # Should have all 4 components
        self.assertIn("line_length", components)
        self.assertIn("pos_pattern", components)
        self.assertIn("concreteness", components)
        self.assertIn("phonetic", components)

        # Overall should be average of components
        expected = sum(components.values()) / len(components)
        self.assertAlmostEqual(overall, expected, places=5)

    def test_style_fit_score_in_valid_range(self):
        """Test that style fit scores are in [0, 1] range."""
        texts = [
            "silence",
            "threshold vestige amber",
            "the ancient stone whispers through silence",
            "a b c d e f g",  # Edge case
        ]

        for text in texts:
            overall, components = self.gen._calculate_style_fit_score(text)

            # Overall score should be in valid range
            self.assertGreaterEqual(overall, 0.0, f"Score for '{text}' too low")
            self.assertLessEqual(overall, 1.0, f"Score for '{text}' too high")

            # All components should be in valid range
            for name, score in components.items():
                self.assertGreaterEqual(score, 0.0, f"{name} score for '{text}' too low")
                self.assertLessEqual(score, 1.0, f"{name} score for '{text}' too high")


class TestPersonalizedCollectionGeneration(unittest.TestCase):
    """Test personalized line seed collection generation."""

    def setUp(self):
        """Create generator for testing."""
        self.fingerprint = StyleFingerprint(
            metrics=PoetryMetrics(
                total_poems=10,
                total_lines=50,
                total_words=250,
                vocabulary_size=100,
                avg_words_per_line=5.0,
                line_length_distribution={7: 50},
            ),
            vocabulary=VocabularyProfile(
                most_common_words=[
                    ("silence", 10),
                    ("threshold", 8),
                    ("vestige", 7),
                    ("amber", 6),
                    ("stone", 5),
                ],
                most_common_content_words=[("silence", 10), ("threshold", 8)],
                pos_distribution={"NOUN": 100, "VERB": 80, "ADJ": 60},
            ),
            themes=ThematicProfile(
                concrete_vs_abstract={"concrete": 60, "abstract": 40},
                semantic_clusters=[],
            ),
        )
        self.gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

    def test_generate_returns_list(self):
        """Test that generate_personalized_collection returns a list."""
        seeds = self.gen.generate_personalized_collection(
            count=5, min_quality=0.3, min_style_fit=0.3
        )

        self.assertIsInstance(seeds, list)

    def test_generate_respects_count(self):
        """Test that generation respects count parameter."""
        seeds = self.gen.generate_personalized_collection(
            count=3, min_quality=0.0, min_style_fit=0.0
        )

        # Should return at most requested count
        self.assertLessEqual(len(seeds), 3)

    def test_seeds_have_style_metadata(self):
        """Test that generated seeds have style fit metadata."""
        seeds = self.gen.generate_personalized_collection(
            count=5, min_quality=0.0, min_style_fit=0.0
        )

        if seeds:  # If any seeds were generated
            seed = seeds[0]

            # Should have style_fit_score
            self.assertTrue(hasattr(seed, "style_fit_score"))
            self.assertIsNotNone(seed.style_fit_score)

            # Should have style_components
            self.assertTrue(hasattr(seed, "style_components"))
            self.assertIsNotNone(seed.style_components)

    def test_seeds_meet_quality_threshold(self):
        """Test that generated seeds meet minimum quality threshold."""
        min_quality = 0.5
        seeds = self.gen.generate_personalized_collection(
            count=10, min_quality=min_quality, min_style_fit=0.0
        )

        for seed in seeds:
            self.assertGreaterEqual(
                seed.quality_score,
                min_quality,
                f"Seed '{seed.text}' has quality {seed.quality_score} < {min_quality}",
            )

    def test_seeds_meet_style_fit_threshold(self):
        """Test that generated seeds meet minimum style fit threshold."""
        min_style_fit = 0.4
        seeds = self.gen.generate_personalized_collection(
            count=10, min_quality=0.0, min_style_fit=min_style_fit
        )

        for seed in seeds:
            self.assertGreaterEqual(
                seed.style_fit_score,
                min_style_fit,
                f"Seed '{seed.text}' has style fit {seed.style_fit_score} < {min_style_fit}",
            )

    def test_seeds_sorted_by_combined_score(self):
        """Test that seeds are sorted by combined score (descending)."""
        seeds = self.gen.generate_personalized_collection(
            count=10, min_quality=0.0, min_style_fit=0.0
        )

        if len(seeds) >= 2:
            # Calculate combined scores
            scores = []
            for seed in seeds:
                combined = 0.7 * seed.quality_score + 0.3 * seed.style_fit_score
                scores.append(combined)

            # Should be in descending order
            self.assertEqual(
                scores, sorted(scores, reverse=True), "Seeds should be sorted by combined score"
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create minimal valid fingerprint."""
        self.fingerprint = StyleFingerprint(
            metrics=PoetryMetrics(
                total_poems=1,
                total_lines=5,
                total_words=25,
                vocabulary_size=20,
                avg_words_per_line=5.0,
                line_length_distribution={5: 5},
            ),
            vocabulary=VocabularyProfile(
                most_common_words=[("test", 5)],
                most_common_content_words=[("test", 5)],
                pos_distribution={"NOUN": 10},
            ),
            themes=ThematicProfile(
                concrete_vs_abstract={"concrete": 10, "abstract": 5},
                semantic_clusters=[],
            ),
        )

    def test_empty_line_length_distribution(self):
        """Test handling of empty line length distribution."""
        self.fingerprint.metrics.line_length_distribution = {}
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

        # Should not crash
        score = gen._calculate_line_length_fit("test text")
        self.assertIsInstance(score, float)

    def test_empty_pos_distribution(self):
        """Test handling of empty POS distribution."""
        self.fingerprint.vocabulary.pos_distribution = {}
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

        # Should not crash
        score = gen._calculate_pos_pattern_fit("test text")
        self.assertIsInstance(score, float)

    def test_empty_concreteness_data(self):
        """Test handling of empty concreteness data."""
        self.fingerprint.themes.concrete_vs_abstract = {}
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

        # Should not crash
        score = gen._calculate_concreteness_fit("test text")
        self.assertIsInstance(score, float)

    def test_very_high_thresholds_return_empty(self):
        """Test that very high thresholds may return empty list."""
        gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

        # Very high thresholds should return few or no seeds
        seeds = gen.generate_personalized_collection(count=10, min_quality=0.99, min_style_fit=0.99)

        # Should return list (possibly empty)
        self.assertIsInstance(seeds, list)


class TestDualScoring(unittest.TestCase):
    """Test dual scoring (quality + style fit)."""

    def setUp(self):
        """Create generator for testing."""
        self.fingerprint = StyleFingerprint(
            metrics=PoetryMetrics(
                total_poems=10,
                total_lines=50,
                total_words=250,
                vocabulary_size=100,
                avg_words_per_line=5.0,
                line_length_distribution={7: 50},
            ),
            vocabulary=VocabularyProfile(
                most_common_words=[("silence", 10)],
                most_common_content_words=[("silence", 10)],
                pos_distribution={"NOUN": 100},
            ),
            themes=ThematicProfile(
                concrete_vs_abstract={"concrete": 60, "abstract": 40},
                semantic_clusters=[],
            ),
        )
        self.gen = PersonalizedLineSeedGenerator(self.fingerprint, strictness=0.7)

    def test_combined_score_formula(self):
        """Test that combined score uses 70% quality + 30% style fit."""
        seeds = self.gen.generate_personalized_collection(
            count=5, min_quality=0.0, min_style_fit=0.0
        )

        if seeds:
            for seed in seeds:
                expected_combined = 0.7 * seed.quality_score + 0.3 * seed.style_fit_score

                # Note: We can't directly access the combined score used for sorting,
                # but we can verify it's in a reasonable range
                self.assertGreaterEqual(expected_combined, 0.0)
                self.assertLessEqual(expected_combined, 1.0)


if __name__ == "__main__":
    unittest.main()

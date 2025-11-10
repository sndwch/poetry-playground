"""Comprehensive tests for the quality scoring system."""

import pytest
from poetryplayground.quality_scorer import (
    QualityScorer,
    QualityScore,
    GenerationContext,
    EmotionalTone,
    FormalityLevel,
    get_quality_scorer,
)


@pytest.fixture
def scorer():
    """Get quality scorer instance."""
    return get_quality_scorer()


class TestQualityScorer:
    """Test suite for QualityScorer class."""

    def test_initialization(self, scorer):
        """Test that scorer initializes correctly."""
        assert scorer is not None
        assert len(scorer.cliche_phrases) > 0
        assert len(scorer.cliche_words) > 0
        assert len(scorer.concreteness_cache) > 0

    def test_cliche_detection(self, scorer):
        """Test comprehensive cliché detection."""
        # Known cliché phrases
        assert scorer.is_cliche("life is a journey", threshold=0.5)
        assert scorer.is_cliche("love is a rose", threshold=0.5)
        assert scorer.is_cliche("time is a river", threshold=0.5)

        # Known cliché words
        assert scorer.is_cliche("heart", threshold=0.5)
        assert scorer.is_cliche("soul", threshold=0.5)
        assert scorer.is_cliche("dream", threshold=0.5)

        # Fresh words should not be clichéd
        assert not scorer.is_cliche("penumbra", threshold=0.5)
        assert not scorer.is_cliche("granite", threshold=0.5)
        assert not scorer.is_cliche("vestige", threshold=0.5)

    def test_word_scoring(self, scorer):
        """Test word quality scoring."""
        # High-quality, fresh words
        penumbra_score = scorer.score_word("penumbra")
        assert isinstance(penumbra_score, QualityScore)
        assert 0.0 <= penumbra_score.overall <= 1.0
        assert penumbra_score.overall > 0.7  # Should be high quality

        # Clichéd word
        heart_score = scorer.score_word("heart")
        assert 0.0 <= heart_score.overall <= 1.0
        assert heart_score.novelty < 0.5  # Should have low novelty

        # Check all components are in range
        assert 0.0 <= penumbra_score.frequency <= 1.0
        assert 0.0 <= penumbra_score.novelty <= 1.0
        assert 0.0 <= penumbra_score.coherence <= 1.0
        assert 0.0 <= penumbra_score.register <= 1.0
        assert 0.0 <= penumbra_score.imagery <= 1.0

    def test_phrase_scoring(self, scorer):
        """Test phrase quality scoring."""
        # Fresh phrase
        fresh_score = scorer.score_phrase("silence is amber")
        assert isinstance(fresh_score, QualityScore)
        assert 0.0 <= fresh_score.overall <= 1.0
        assert fresh_score.novelty > 0.7  # Fresh phrase

        # Clichéd phrase
        cliche_score = scorer.score_phrase("life is a journey")
        assert 0.0 <= cliche_score.overall <= 1.0
        assert cliche_score.novelty < 0.3  # Clichéd phrase

    def test_frequency_scoring(self, scorer):
        """Test frequency-based scoring."""
        # Test ideal frequency range
        common_word = "the"  # Very common
        rare_word = "penumbra"  # Uncommon
        ideal_word = "silence"  # Moderate

        common_score = scorer._score_frequency(common_word)
        rare_score = scorer._score_frequency(rare_word)
        ideal_score = scorer._score_frequency(ideal_word)

        # All should be in 0-1 range
        assert 0.0 <= common_score <= 1.0
        assert 0.0 <= rare_score <= 1.0
        assert 0.0 <= ideal_score <= 1.0

    def test_concreteness_scoring(self, scorer):
        """Test concreteness ratings."""
        # Concrete words
        assert scorer.get_concreteness("stone") > 0.8
        assert scorer.get_concreteness("water") > 0.8
        assert scorer.get_concreteness("fire") > 0.8

        # Abstract words
        assert scorer.get_concreteness("truth") < 0.6
        assert scorer.get_concreteness("soul") < 0.6
        assert scorer.get_concreteness("eternity") <= 0.6

        # All should be in 0-1 range
        for word in ["stone", "water", "truth", "soul"]:
            concrete = scorer.get_concreteness(word)
            assert 0.0 <= concrete <= 1.0

    def test_novelty_scoring(self, scorer):
        """Test novelty/cliché scoring."""
        # Fresh words
        assert scorer._score_novelty_word("penumbra") > 0.8
        assert scorer._score_novelty_word("vestige") > 0.8

        # Clichéd words
        assert scorer._score_novelty_word("heart") < 0.2
        assert scorer._score_novelty_word("soul") < 0.2
        assert scorer._score_novelty_word("dream") < 0.2

    def test_imagery_scoring_with_context(self, scorer):
        """Test imagery scoring with different concreteness targets."""
        # Concrete word
        word = "stone"

        # Target concrete imagery
        concrete_context = GenerationContext(concreteness_target=0.9)
        concrete_score = scorer.score_word(word, concrete_context)

        # Target abstract imagery
        abstract_context = GenerationContext(concreteness_target=0.3)
        abstract_score = scorer.score_word(word, abstract_context)

        # Should score better when matching target
        assert concrete_score.imagery > abstract_score.imagery

    def test_context_based_scoring(self, scorer):
        """Test scoring with different contexts."""
        word = "silence"

        # Different contexts
        dark_context = GenerationContext(
            emotional_tone=EmotionalTone.DARK,
            concreteness_target=0.7,
            avoid_cliches=True
        )

        light_context = GenerationContext(
            emotional_tone=EmotionalTone.LIGHT,
            concreteness_target=0.5,
            avoid_cliches=False
        )

        dark_score = scorer.score_word(word, dark_context)
        light_score = scorer.score_word(word, light_context)

        # Both should be valid scores
        assert 0.0 <= dark_score.overall <= 1.0
        assert 0.0 <= light_score.overall <= 1.0

    def test_quality_score_grading(self):
        """Test quality score grading system."""
        score_a_plus = QualityScore(
            overall=0.95, frequency=0.9, novelty=0.9,
            coherence=0.9, register=0.9, imagery=0.9
        )
        assert score_a_plus.get_grade() == "A+"

        score_b = QualityScore(
            overall=0.75, frequency=0.7, novelty=0.7,
            coherence=0.7, register=0.7, imagery=0.7
        )
        assert score_b.get_grade() == "B"

        score_f = QualityScore(
            overall=0.3, frequency=0.3, novelty=0.3,
            coherence=0.3, register=0.3, imagery=0.3
        )
        assert score_f.get_grade() == "F"

    def test_singleton_pattern(self):
        """Test that get_quality_scorer returns same instance."""
        scorer1 = get_quality_scorer()
        scorer2 = get_quality_scorer()
        assert scorer1 is scorer2

    def test_batch_word_scoring(self, scorer):
        """Test scoring multiple words."""
        test_words = [
            "penumbra",  # High quality
            "vestige",   # High quality
            "heart",     # Clichéd
            "soul",      # Clichéd
            "stone",     # Good
        ]

        scores = [scorer.score_word(w).overall for w in test_words]

        # All scores should be in valid range
        for score in scores:
            assert 0.0 <= score <= 1.0

        # Fresh words should score higher than clichés
        assert scores[0] > scores[2]  # penumbra > heart
        assert scores[1] > scores[3]  # vestige > soul

    def test_phrase_cliche_detection(self, scorer):
        """Test phrase-level cliché detection."""
        # Exact matches
        assert scorer._score_novelty_phrase("life is a journey") < 0.2
        assert scorer._score_novelty_phrase("love is a rose") < 0.2

        # Partial matches should have some penalty
        partial_match = scorer._score_novelty_phrase("the journey of life begins")
        assert 0.2 < partial_match < 0.8  # Some penalty, not full

        # Fresh phrases should score high
        fresh = scorer._score_novelty_phrase("silence is amber")
        assert fresh > 0.8

    def test_edge_cases(self, scorer):
        """Test edge cases and error handling."""
        # Empty string
        empty_score = scorer.score_word("")
        assert 0.0 <= empty_score.overall <= 1.0

        # Single letter
        single_score = scorer.score_word("a")
        assert 0.0 <= single_score.overall <= 1.0

        # Very long word
        long_score = scorer.score_word("antidisestablishmentarianism")
        assert 0.0 <= long_score.overall <= 1.0

        # Non-English characters
        foreign_score = scorer.score_word("über")
        assert 0.0 <= foreign_score.overall <= 1.0

    def test_database_loading(self, scorer):
        """Test that databases loaded correctly."""
        # Cliché phrases database
        assert "life is a journey" in scorer.cliche_phrases
        assert "love is a rose" in scorer.cliche_phrases
        assert len(scorer.cliche_phrases) > 100

        # Cliché words database
        assert "heart" in scorer.cliche_words
        assert "soul" in scorer.cliche_words
        assert len(scorer.cliche_words) > 200

        # Concreteness cache
        assert "stone" in scorer.concreteness_cache
        assert "water" in scorer.concreteness_cache
        assert len(scorer.concreteness_cache) > 200


class TestGenerationContext:
    """Test GenerationContext configuration."""

    def test_default_context(self):
        """Test default context creation."""
        context = GenerationContext()
        assert context.emotional_tone == EmotionalTone.NEUTRAL
        assert context.concreteness_target == 0.5
        assert context.formality_level == FormalityLevel.CONVERSATIONAL
        assert context.avoid_cliches is True
        assert context.domain is None

    def test_custom_context(self):
        """Test custom context creation."""
        context = GenerationContext(
            emotional_tone=EmotionalTone.DARK,
            concreteness_target=0.8,
            formality_level=FormalityLevel.FORMAL,
            avoid_cliches=False,
            domain="nature"
        )
        assert context.emotional_tone == EmotionalTone.DARK
        assert context.concreteness_target == 0.8
        assert context.formality_level == FormalityLevel.FORMAL
        assert context.avoid_cliches is False
        assert context.domain == "nature"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

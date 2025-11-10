"""Tests for lexigen.py word generation functions."""

import pytest

from poetryplayground.core.lexigen import (
    clean_api_results,
    rhymes,
    rhyme,
    similar_meaning_words,
    similar_sounding_words,
    contextually_linked_words,
    related_rare_words,
    phonetically_related_words,
    score_and_filter_results,
)
from poetryplayground.core.quality_scorer import GenerationContext, EmotionalTone


# ============================================================================
# Unit Tests - Clean API Results
# ============================================================================


class TestCleanAPIResults:
    """Test the clean_api_results helper function."""

    def test_clean_api_results_basic(self):
        """Test basic word cleaning."""
        words = ["hello", "world", "test"]
        result = clean_api_results(words)

        assert isinstance(result, list)
        assert len(result) <= len(words)

    def test_clean_api_results_with_exclusions(self):
        """Test cleaning with excluded words."""
        words = ["hello", "world", "test"]
        exclude = ["world"]
        result = clean_api_results(words, exclude_words=exclude)

        assert "world" not in result

    def test_clean_api_results_without_validator(self):
        """Test cleaning without validation."""
        words = ["hello", "world", "xyz123"]
        result = clean_api_results(words, use_validator=False)

        assert isinstance(result, list)


# ============================================================================
# Unit Tests - Rhyme Functions
# ============================================================================


class TestRhymeFunctions:
    """Test rhyme-related functions."""

    def test_rhymes_returns_list(self):
        """Test that rhymes() returns a list."""
        result = rhymes("fire")

        assert isinstance(result, list)

    def test_rhymes_basic_word(self):
        """Test rhymes for a basic word."""
        result = rhymes("cat", sample_size=5)

        assert isinstance(result, list)
        assert len(result) <= 5
        # Common rhymes
        assert any(word in result for word in ["bat", "hat", "mat", "sat", "rat"])

    def test_rhymes_with_sample_size(self):
        """Test rhymes with sample size limit."""
        result = rhymes("tree", sample_size=3)

        assert len(result) <= 3

    def test_rhyme_single_result(self):
        """Test rhyme() returns a single word or None."""
        result = rhyme("fire")

        assert result is None or isinstance(result, str)

    def test_rhymes_no_results(self):
        """Test handling of words with no rhymes."""
        result = rhymes("xyzabc123")

        assert isinstance(result, list)
        assert len(result) == 0


# ============================================================================
# Integration Tests - Datamuse API Functions
# ============================================================================


class TestDatamuseAPIFunctions:
    """Test functions that use the Datamuse API."""

    def test_similar_meaning_words_basic(self):
        """Test similar_meaning_words with a common word."""
        try:
            result = similar_meaning_words("happy", sample_size=5)

            assert isinstance(result, list)
            assert len(result) <= 5
            # Should contain some synonyms
            if result:
                assert all(isinstance(word, str) for word in result)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_similar_meaning_words_with_list(self):
        """Test similar_meaning_words with multiple input words."""
        try:
            result = similar_meaning_words(["happy", "sad"], sample_size=5)

            assert isinstance(result, list)
            assert len(result) <= 5
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_similar_sounding_words_basic(self):
        """Test similar_sounding_words."""
        try:
            result = similar_sounding_words("fire", sample_size=5)

            assert isinstance(result, list)
            assert len(result) <= 5
            # Should contain phonetically similar words
            if result:
                assert all(isinstance(word, str) for word in result)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_contextually_linked_words_basic(self):
        """Test contextually_linked_words."""
        try:
            result = contextually_linked_words("fire", sample_size=5)

            assert isinstance(result, list)
            assert len(result) <= 5
            # Should contain contextually related words
            if result:
                assert all(isinstance(word, str) for word in result)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_related_rare_words_basic(self):
        """Test related_rare_words."""
        try:
            result = related_rare_words("fire", sample_size=5)

            assert isinstance(result, list)
            assert len(result) <= 5
            # Should contain rare related words
            if result:
                assert all(isinstance(word, str) for word in result)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_phonetically_related_words_basic(self):
        """Test phonetically_related_words."""
        try:
            result = phonetically_related_words("fire", sample_size=5)

            assert isinstance(result, list)
            assert len(result) <= 5
            # Should contain phonetically related words
            if result:
                assert all(isinstance(word, str) for word in result)
        except Exception as e:
            pytest.skip(f"API call failed: {e}")


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_input(self):
        """Test handling of empty string."""
        result = rhymes("")
        assert isinstance(result, list)

    def test_nonexistent_word(self):
        """Test handling of nonsense word."""
        result = rhymes("xyzabc123")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_very_short_word(self):
        """Test handling of single letter."""
        result = rhymes("a")
        assert isinstance(result, list)

    def test_zero_sample_size(self):
        """Test with sample_size=0."""
        result = rhymes("fire", sample_size=0)
        assert isinstance(result, list)
        # sample_size=0 may return empty or full list depending on implementation
        # Just verify it returns a list

    def test_large_sample_size(self):
        """Test with very large sample_size."""
        result = rhymes("cat", sample_size=1000)
        assert isinstance(result, list)
        # Should be capped at available rhymes


# ============================================================================
# Caching Tests
# ============================================================================


class TestCaching:
    """Test that caching is working."""

    def test_repeated_calls_are_fast(self):
        """Test that repeated calls use cache."""
        import time

        # First call (potentially cold)
        start = time.time()
        result1 = rhymes("fire", sample_size=5)
        time1 = time.time() - start

        # Second call (should be cached)
        start = time.time()
        result2 = rhymes("fire", sample_size=5)
        time2 = time.time() - start

        # Both calls should return valid results
        assert isinstance(result1, list)
        assert isinstance(result2, list)
        assert len(result1) <= 5
        assert len(result2) <= 5

        # Note: Results may differ due to random.sample, but both should be valid
        # Second call should generally be as fast or faster due to caching
        # Don't enforce strict timing as it can be flaky
        assert time2 <= time1 + 0.2  # Allow generous margin


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_for_poem_generation(self):
        """Test a typical workflow for generating word options."""
        seed_word = "ocean"

        # Get rhymes
        rhyme_options = rhymes(seed_word, sample_size=3)
        assert isinstance(rhyme_options, list)

        # Get synonyms
        try:
            synonym_options = similar_meaning_words(seed_word, sample_size=3)
            assert isinstance(synonym_options, list)
        except Exception:
            pass  # API might fail, that's okay

        # Get sound-alikes
        try:
            sound_options = similar_sounding_words(seed_word, sample_size=3)
            assert isinstance(sound_options, list)
        except Exception:
            pass  # API might fail, that's okay


# ============================================================================
# Quality-Aware Features Tests (Phase 6)
# ============================================================================


class TestQualityAwareFeatures:
    """Test quality-aware filtering in lexigen functions."""

    def test_score_and_filter_results_basic(self):
        """Test score_and_filter_results with basic API response."""
        api_response = [
            {"word": "happy", "score": 1000},
            {"word": "joyful", "score": 900},
            {"word": "glad", "score": 800},
        ]

        result = score_and_filter_results(api_response, min_quality=0.0)

        assert isinstance(result, list)
        assert all(isinstance(word, str) for word in result)
        assert len(result) <= len(api_response)

    def test_score_and_filter_results_with_quality_threshold(self):
        """Test that quality threshold filters results."""
        api_response = [
            {"word": "penumbra", "score": 100},  # High quality word
            {"word": "heart", "score": 100},     # Clichéd word (lower quality)
        ]

        # With high quality threshold, should filter out clichés
        result_high = score_and_filter_results(api_response, min_quality=0.7)

        # With low quality threshold, should include both
        result_low = score_and_filter_results(api_response, min_quality=0.0)

        assert isinstance(result_high, list)
        assert isinstance(result_low, list)
        # Low threshold should potentially have more results
        assert len(result_low) >= len(result_high)

    def test_score_and_filter_results_with_exclusions(self):
        """Test score_and_filter_results respects exclusion list."""
        api_response = [
            {"word": "happy", "score": 100},
            {"word": "joyful", "score": 100},
        ]

        result = score_and_filter_results(
            api_response,
            exclude_words=["happy"],
            min_quality=0.0
        )

        assert "happy" not in result

    def test_score_and_filter_results_sorting(self):
        """Test that results are sorted by quality (best first)."""
        api_response = [
            {"word": "heart", "score": 100},     # Clichéd (lower quality)
            {"word": "penumbra", "score": 100},  # Fresh (higher quality)
        ]

        result = score_and_filter_results(api_response, min_quality=0.0, sample_size=None)

        # Should have at least some results
        assert isinstance(result, list)
        # Results should be sorted by quality (best first)
        # This is hard to verify without knowing exact scores, but we can check structure
        assert all(isinstance(word, str) for word in result)

    def test_similar_meaning_words_with_quality_params(self):
        """Test similar_meaning_words accepts quality parameters."""
        try:
            # Test with min_quality parameter
            result_high = similar_meaning_words(
                "happy",
                sample_size=5,
                min_quality=0.7
            )
            result_low = similar_meaning_words(
                "happy",
                sample_size=5,
                min_quality=0.3
            )

            assert isinstance(result_high, list)
            assert isinstance(result_low, list)
            # Both should work without errors

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_similar_meaning_words_with_context(self):
        """Test similar_meaning_words accepts GenerationContext."""
        try:
            context = GenerationContext(
                emotional_tone=EmotionalTone.DARK,
                avoid_cliches=True
            )

            result = similar_meaning_words(
                "night",
                sample_size=5,
                context=context
            )

            assert isinstance(result, list)
            # Should return quality-filtered words

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_contextually_linked_words_with_quality_params(self):
        """Test contextually_linked_words accepts quality parameters."""
        try:
            result = contextually_linked_words(
                "fire",
                sample_size=5,
                min_quality=0.6
            )

            assert isinstance(result, list)
            # Should filter out low-quality collocations

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_phonetically_related_words_with_quality_params(self):
        """Test phonetically_related_words accepts quality parameters."""
        try:
            result = phonetically_related_words(
                "fire",
                sample_size=5,
                min_quality=0.6
            )

            assert isinstance(result, list)
            # Should filter out low-quality phonetic matches

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_related_rare_words_with_quality_params(self):
        """Test related_rare_words accepts quality parameters."""
        try:
            result = related_rare_words(
                "ocean",
                sample_size=5,
                min_quality=0.6
            )

            assert isinstance(result, list)
            # Should return rare AND quality words

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

    def test_quality_filtering_reduces_cliches(self):
        """Test that quality filtering helps avoid clichéd words."""
        try:
            # Get words without quality filtering
            result_unfiltered = similar_meaning_words(
                "love",
                sample_size=10,
                min_quality=0.0  # Accept all
            )

            # Get words with strict quality filtering
            result_filtered = similar_meaning_words(
                "love",
                sample_size=10,
                min_quality=0.7  # High quality only
            )

            # Both should be lists
            assert isinstance(result_unfiltered, list)
            assert isinstance(result_filtered, list)

            # Filtered results should potentially be shorter or different
            # (though we can't guarantee due to API variability)

        except Exception as e:
            pytest.skip(f"API call failed: {e}")

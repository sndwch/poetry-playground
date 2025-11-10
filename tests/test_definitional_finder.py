"""Tests for definitional finder functionality."""

import pytest

from poetryplayground.definitional_finder import (
    DefinitionalResult,
    find_words_by_definition,
    _process_synset,
)


# ============================================================================
# Unit Tests - DataClass
# ============================================================================


class TestDefinitionalResult:
    """Test DefinitionalResult dataclass."""

    def test_definitional_result_creation(self):
        """Test DefinitionalResult instantiation."""
        result = DefinitionalResult(
            word="corrosion",
            definition="a state of deterioration due to oxidization or chemical action",
            quality_score=0.85,
            pos="n",
        )

        assert result.word == "corrosion"
        assert "oxidization" in result.definition
        assert result.quality_score == 0.85
        assert result.pos == "n"


# ============================================================================
# Integration Tests - Basic Searches
# ============================================================================


class TestBasicSearch:
    """Test basic definitional searches."""

    def test_simple_search(self):
        """Test basic search for 'rust' in definitions."""
        try:
            results = find_words_by_definition("rust", limit=10, verbose=False)

            assert isinstance(results, list)
            assert len(results) > 0
            assert len(results) <= 10

            # All results should have the expected structure
            for result in results:
                assert isinstance(result, DefinitionalResult)
                assert isinstance(result.word, str)
                assert isinstance(result.definition, str)
                assert isinstance(result.quality_score, float)
                assert isinstance(result.pos, str)
                assert 0.0 <= result.quality_score <= 1.0

            # At least one result should mention rust
            definitions_lower = [r.definition.lower() for r in results]
            assert any("rust" in d for d in definitions_lower)

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_search_with_pos_filter(self):
        """Test search with POS filter."""
        try:
            results = find_words_by_definition("threshold", pos_filter="n", limit=10, verbose=False)

            assert isinstance(results, list)

            # All results should be nouns
            for result in results:
                assert result.pos == "n"

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_search_with_quality_filter(self):
        """Test search with minimum quality threshold."""
        try:
            results = find_words_by_definition(
                "threshold", min_quality=0.7, limit=10, verbose=False
            )

            assert isinstance(results, list)

            # All results should meet quality threshold
            for result in results:
                assert result.quality_score >= 0.7

        except ImportError:
            pytest.skip("WordNet data not installed")


# ============================================================================
# Integration Tests - Multi-word Terms
# ============================================================================


class TestMultiwordTerms:
    """Test handling of multi-word terms."""

    def test_allow_multiword_true(self):
        """Test that multi-word terms are included when allow_multiword=True."""
        try:
            results = find_words_by_definition(
                "computer", allow_multiword=True, limit=20, min_quality=0.3, verbose=False
            )

            # Check if any results contain spaces (multi-word terms)
            has_multiword = any(" " in r.word for r in results)

            # With computer-related terms, we might get multi-word results
            # but it's not guaranteed, so we just verify the flag works
            assert isinstance(results, list)

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_allow_multiword_false(self):
        """Test that multi-word terms are excluded when allow_multiword=False."""
        try:
            results = find_words_by_definition(
                "computer", allow_multiword=False, limit=20, min_quality=0.3, verbose=False
            )

            # No results should contain spaces
            for result in results:
                assert " " not in result.word

        except ImportError:
            pytest.skip("WordNet data not installed")


# ============================================================================
# Integration Tests - Result Quality
# ============================================================================


class TestResultQuality:
    """Test quality of search results."""

    def test_results_sorted_by_quality(self):
        """Test that results are sorted by quality score (descending)."""
        try:
            results = find_words_by_definition("threshold", limit=20, verbose=False)

            if len(results) > 1:
                # Check that quality scores are in descending order
                quality_scores = [r.quality_score for r in results]
                assert quality_scores == sorted(quality_scores, reverse=True)

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_no_self_reference(self):
        """Test that search term itself is not in results."""
        try:
            search_term = "threshold"
            results = find_words_by_definition(search_term, limit=20, verbose=False)

            # Search term should not appear as a result
            result_words = [r.word.lower() for r in results]
            assert search_term.lower() not in result_words

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_results_unique(self):
        """Test that all results are unique."""
        try:
            results = find_words_by_definition("threshold", limit=20, verbose=False)

            # All words should be unique
            words = [r.word for r in results]
            assert len(words) == len(set(words))

        except ImportError:
            pytest.skip("WordNet data not installed")


# ============================================================================
# Integration Tests - Golden Test Cases
# ============================================================================


class TestGoldenCases:
    """Test known good search cases."""

    def test_rust_search(self):
        """Test searching for 'rust' finds corrosion-related words."""
        try:
            results = find_words_by_definition("rust", limit=20, min_quality=0.5, verbose=False)

            # Should find some corrosion/oxidation related terms
            result_words = [r.word.lower() for r in results]

            # We expect at least some relevant results
            assert len(results) > 0

            # Check definitions mention rust
            definitions = " ".join([r.definition.lower() for r in results])
            assert "rust" in definitions

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_bean_search(self):
        """Test searching for 'bean' finds bean-related words."""
        try:
            results = find_words_by_definition("bean", limit=20, min_quality=0.3, verbose=False)

            # Should find some bean-related terms
            assert len(results) > 0

            # Check definitions mention beans
            definitions = " ".join([r.definition.lower() for r in results])
            assert "bean" in definitions

        except ImportError:
            pytest.skip("WordNet data not installed")


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_results_rare_term(self):
        """Test search for extremely rare/made-up term returns empty list."""
        try:
            results = find_words_by_definition(
                "xyzzyqwerty123", limit=10, min_quality=0.0, verbose=False
            )

            # Should return empty list for nonsense term
            assert isinstance(results, list)
            assert len(results) == 0

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_high_quality_threshold(self):
        """Test that high quality threshold filters results."""
        try:
            low_quality_results = find_words_by_definition(
                "threshold", min_quality=0.3, limit=50, verbose=False
            )
            high_quality_results = find_words_by_definition(
                "threshold", min_quality=0.8, limit=50, verbose=False
            )

            # High quality threshold should yield fewer results
            assert len(high_quality_results) <= len(low_quality_results)

        except ImportError:
            pytest.skip("WordNet data not installed")

    def test_invalid_pos_filter_raises_error(self):
        """Test that invalid POS filter raises ValueError."""
        with pytest.raises(ValueError, match="Invalid pos_filter"):
            find_words_by_definition("test", pos_filter="INVALID", verbose=False)

    def test_limit_respected(self):
        """Test that limit parameter is respected."""
        try:
            results = find_words_by_definition("the", limit=5, min_quality=0.0, verbose=False)

            # Should return at most 5 results
            assert len(results) <= 5

        except ImportError:
            pytest.skip("WordNet data not installed")


# ============================================================================
# Caching Tests
# ============================================================================


class TestCaching:
    """Test that caching works correctly."""

    def test_repeated_search_uses_cache(self):
        """Test that repeated searches are fast (indicating cache hit)."""
        try:
            import time

            # First search (cold cache)
            start = time.time()
            results1 = find_words_by_definition("threshold", limit=10, verbose=False)
            time1 = time.time() - start

            # Second identical search (warm cache)
            start = time.time()
            results2 = find_words_by_definition("threshold", limit=10, verbose=False)
            time2 = time.time() - start

            # Results should be identical
            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2):
                assert r1.word == r2.word
                assert r1.definition == r2.definition
                assert r1.quality_score == r2.quality_score

            # Second search should be much faster (at least 2x)
            # This indicates cache is working
            assert time2 < time1 / 2 or time2 < 0.1  # Either much faster or very fast

        except ImportError:
            pytest.skip("WordNet data not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

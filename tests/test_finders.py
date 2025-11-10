"""Tests for word finding algorithms (equidistant word finder)."""

import argparse

import pytest

from poetryplayground.cli import parse_syllable_range
from poetryplayground.core.lexicon import get_lexicon_data
from poetryplayground.finders import find_equidistant


@pytest.fixture(scope="module")
def test_lexicon():
    """Load the deterministic snapshot lexicon once for all tests.

    This fixture ensures:
    - Deterministic results across test runs
    - Fast test execution (10k words vs 100k)
    - No network calls required
    """
    return get_lexicon_data(use_snapshot_for_testing=True)


class TestEquidistantOrthographic:
    """Tests for orthographic (spelling-based) equidistant word finding."""

    def test_stone_storm_exact(self, test_lexicon):
        """Golden test: stone/storm should find score, shore, stare, etc."""
        hits = find_equidistant("stone", "storm", lexicon_data=test_lexicon)

        # Basic validation
        assert len(hits) > 0, "Should find at least some equidistant words"
        assert hits[0].target_distance == 2, "stone and storm are distance 2 apart"

        # Check that common expected words are in results
        words = {h.word for h in hits}
        # These words are distance 2 from both stone and storm
        # Results depend on snapshot lexicon content
        expected = {"score", "shore", "stare", "sore", "stop", "stock"}
        found = expected & words
        assert len(found) >= 2, f"Should find several common words, found: {found}"

        # Verify all hits respect distance constraints
        for hit in hits:
            assert hit.dist_a == 2, f"{hit.word} should be distance 2 from 'stone'"
            assert hit.dist_b == 2, f"{hit.word} should be distance 2 from 'storm'"

    def test_cat_bat_exact(self, test_lexicon):
        """Golden test: cat/bat are distance 1, should find many rhymes."""
        hits = find_equidistant("cat", "bat", lexicon_data=test_lexicon)

        assert len(hits) > 0
        assert hits[0].target_distance == 1

        words = {h.word for h in hits}
        # Common 1-letter changes from cat/bat
        expected = {"hat", "mat", "rat", "sat", "fat", "pat"}
        found = expected & words
        assert len(found) >= 3, f"Should find several rhymes, found: {found}"

    def test_window_parameter(self, test_lexicon):
        """Test that window parameter expands search range."""
        # No window (exact only)
        hits_exact = find_equidistant("cat", "dog", window=0, lexicon_data=test_lexicon)

        # With window ±1
        hits_window = find_equidistant("cat", "dog", window=1, lexicon_data=test_lexicon)

        # Window should find more results
        assert len(hits_window) >= len(hits_exact)

        # Check that window results respect bounds
        target_d = hits_window[0].target_distance if hits_window else 3
        for hit in hits_window:
            assert target_d - 1 <= hit.dist_a <= target_d + 1
            assert target_d - 1 <= hit.dist_b <= target_d + 1

    def test_identical_words_no_window(self, test_lexicon):
        """Identical words with no window should return empty list."""
        hits = find_equidistant("cat", "cat", lexicon_data=test_lexicon)
        assert len(hits) == 0

    def test_identical_words_with_window(self, test_lexicon):
        """Identical words with window should find words within distance."""
        hits = find_equidistant("cat", "cat", window=1, lexicon_data=test_lexicon)

        assert len(hits) > 0
        # All results should be distance 0-1 from 'cat'
        for hit in hits:
            assert hit.dist_a <= 1
            assert hit.dist_b <= 1


class TestEquidistantPhonetic:
    """Tests for phonetic (sound-based) equidistant word finding."""

    def test_light_night_phonetic(self, test_lexicon):
        """Golden test: light/night are phonetically similar (rhyme)."""
        hits = find_equidistant("light", "night", mode="phono", lexicon_data=test_lexicon)

        assert len(hits) > 0
        words = {h.word for h in hits}

        # These rhyme with light/night
        expected_rhymes = {"right", "might", "tight", "fight", "sight"}
        found = expected_rhymes & words
        assert len(found) >= 2, f"Should find common rhymes, found: {found}"

    def test_phonetic_mode_missing_word_error(self, test_lexicon):
        """Phonetic mode with non-dictionary word should raise ValueError."""
        with pytest.raises(ValueError, match="Phonetic representation not found"):
            find_equidistant("qwerty", "zxcvb", mode="phono", lexicon_data=test_lexicon)

    def test_orthographic_vs_phonetic_different(self, test_lexicon):
        """Orthographic and phonetic modes should give different results."""
        orth_hits = find_equidistant("light", "night", mode="orth", lexicon_data=test_lexicon)
        phono_hits = find_equidistant("light", "night", mode="phono", lexicon_data=test_lexicon)

        orth_words = {h.word for h in orth_hits}
        phono_words = {h.word for h in phono_hits}

        # Sets should be different (phonetic captures sound, orth captures spelling)
        assert orth_words != phono_words


class TestFilters:
    """Tests for filtering by frequency, POS, and syllables."""

    def test_min_zipf_filter(self, test_lexicon):
        """Higher min_zipf should return fewer, more common words."""
        hits_low = find_equidistant("cat", "dog", min_zipf=1.0, lexicon_data=test_lexicon)
        hits_high = find_equidistant("cat", "dog", min_zipf=5.0, lexicon_data=test_lexicon)

        # Higher threshold should filter out rare words
        assert len(hits_high) <= len(hits_low)

        # All high-threshold results should have zipf >= 5.0
        for hit in hits_high:
            assert hit.zipf_frequency >= 5.0

    def test_pos_filter_noun(self, test_lexicon):
        """POS filter should only return specified part of speech."""
        hits = find_equidistant(
            "cat", "dog", pos_filter="NOUN", window=1, lexicon_data=test_lexicon
        )

        # All results should be nouns
        for hit in hits:
            assert hit.pos == "NOUN", f"{hit.word} should be a NOUN, got {hit.pos}"

    def test_syllable_filter_exact(self, test_lexicon):
        """Syllable filter with exact count should only return matching words."""
        hits = find_equidistant(
            "cat", "dog", syllable_filter=(1, 1), window=1, lexicon_data=test_lexicon
        )

        # All results should have exactly 1 syllable
        for hit in hits:
            assert hit.syllables == 1, f"{hit.word} should have 1 syllable, got {hit.syllables}"

    def test_syllable_filter_range(self, test_lexicon):
        """Syllable filter with range should only return words in range."""
        hits = find_equidistant(
            "cat", "dog", syllable_filter=(2, 3), window=1, lexicon_data=test_lexicon
        )

        # All results should have 2-3 syllables
        for hit in hits:
            assert 2 <= hit.syllables <= 3, (
                f"{hit.word} should have 2-3 syllables, got {hit.syllables}"
            )

    def test_combined_filters(self, test_lexicon):
        """Multiple filters should all be applied."""
        hits = find_equidistant(
            "stone",
            "storm",
            min_zipf=4.0,
            pos_filter="NOUN",
            syllable_filter=(1, 1),
            lexicon_data=test_lexicon,
        )

        # All filters should be applied
        for hit in hits:
            assert hit.zipf_frequency >= 4.0
            assert hit.pos == "NOUN"
            assert hit.syllables == 1


class TestScoring:
    """Tests for craft-aware scoring and ranking."""

    def test_results_are_sorted_by_score(self, test_lexicon):
        """Results should be sorted by score (descending)."""
        hits = find_equidistant("cat", "dog", lexicon_data=test_lexicon)

        if len(hits) < 2:
            pytest.skip("Not enough results to test sorting")

        # Verify descending score order
        for i in range(len(hits) - 1):
            assert hits[i].score >= hits[i + 1].score, (
                f"Results should be sorted by score: {hits[i].score} >= {hits[i + 1].score}"
            )

    def test_exact_distance_scores_higher(self, test_lexicon):
        """Words at exact target distance should score higher than those in window."""
        hits = find_equidistant("cat", "dog", window=1, lexicon_data=test_lexicon)

        if len(hits) < 5:
            pytest.skip("Not enough results to test scoring")

        target_d = hits[0].target_distance

        # Get scores for exact matches vs window matches
        exact_scores = [h.score for h in hits if h.dist_a == target_d and h.dist_b == target_d]
        window_scores = [h.score for h in hits if h.dist_a != target_d or h.dist_b != target_d]

        if exact_scores and window_scores:
            # On average, exact matches should score higher
            avg_exact = sum(exact_scores) / len(exact_scores)
            avg_window = sum(window_scores) / len(window_scores)
            assert avg_exact > avg_window


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_words(self, test_lexicon):
        """Single-character words should work."""
        hits = find_equidistant("a", "i", lexicon_data=test_lexicon)
        # May or may not find results, but should not crash
        assert isinstance(hits, list)

    def test_very_long_words(self, test_lexicon):
        """Long words should work (if they exist in lexicon)."""
        # Test with reasonably long common words
        hits = find_equidistant("understanding", "comprehensive", lexicon_data=test_lexicon)
        assert isinstance(hits, list)

    def test_case_insensitive(self, test_lexicon):
        """Search should be case-insensitive."""
        hits_lower = find_equidistant("cat", "dog", lexicon_data=test_lexicon)
        hits_upper = find_equidistant("CAT", "DOG", lexicon_data=test_lexicon)
        hits_mixed = find_equidistant("Cat", "Dog", lexicon_data=test_lexicon)

        # Should all return same results (ignoring order)
        words_lower = {h.word for h in hits_lower}
        words_upper = {h.word for h in hits_upper}
        words_mixed = {h.word for h in hits_mixed}

        assert words_lower == words_upper == words_mixed

    def test_anchor_words_excluded_from_results(self, test_lexicon):
        """Anchor words themselves should not appear in results."""
        hits = find_equidistant("cat", "dog", window=1, lexicon_data=test_lexicon)

        words = {h.word for h in hits}
        assert "cat" not in words
        assert "dog" not in words


class TestSyllableRangeParser:
    """Tests for the CLI syllable range parser."""

    def test_parse_exact_count(self):
        """Parse exact syllable count like '2'."""
        assert parse_syllable_range("2") == (2, 2)
        assert parse_syllable_range("5") == (5, 5)

    def test_parse_range(self):
        """Parse range like '1..3'."""
        assert parse_syllable_range("1..3") == (1, 3)
        assert parse_syllable_range("2..5") == (2, 5)

    def test_parse_open_range_min(self):
        """Parse open-ended range like '2..' (2 or more)."""
        assert parse_syllable_range("2..") == (2, 999)

    def test_parse_open_range_max(self):
        """Parse open-ended range like '..5' (up to 5)."""
        assert parse_syllable_range("..5") == (0, 5)

    def test_parse_invalid_format_raises_error(self):
        """Invalid formats should raise ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_syllable_range("1..3..5")  # Too many dots

        with pytest.raises(argparse.ArgumentTypeError):
            parse_syllable_range("abc")  # Not a number

        with pytest.raises(argparse.ArgumentTypeError):
            parse_syllable_range("a..b")  # Non-numeric range

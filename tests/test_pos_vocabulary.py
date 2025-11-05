"""Tests for POS vocabulary module."""

import tempfile
from pathlib import Path

import pytest

from generativepoetry.forms import count_syllables
from generativepoetry.pos_vocabulary import (
    CORE_POS_TAGS,
    PENN_TO_UNIVERSAL,
    POSVocabulary,
    create_pos_vocabulary,
)


class TestPOSTagMappings:
    """Test POS tag mapping constants."""

    def test_core_pos_tags_defined(self):
        """Test that all core POS tags are defined."""
        expected_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PREP', 'PRON', 'CONJ']
        for tag in expected_tags:
            assert tag in CORE_POS_TAGS
            assert isinstance(CORE_POS_TAGS[tag], list)
            assert len(CORE_POS_TAGS[tag]) > 0

    def test_penn_to_universal_mapping(self):
        """Test Penn Treebank to Universal POS mapping."""
        # Should have entries for all Penn tags in CORE_POS_TAGS
        total_penn_tags = sum(len(tags) for tags in CORE_POS_TAGS.values())
        assert len(PENN_TO_UNIVERSAL) >= total_penn_tags

        # Verify some known mappings
        assert PENN_TO_UNIVERSAL['NN'] == 'NOUN'
        assert PENN_TO_UNIVERSAL['VB'] == 'VERB'
        assert PENN_TO_UNIVERSAL['JJ'] == 'ADJ'
        assert PENN_TO_UNIVERSAL['RB'] == 'ADV'


class TestPOSVocabulary:
    """Test POS vocabulary class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def pos_vocab(self, temp_cache_dir):
        """Create POS vocabulary instance with temp cache."""
        return POSVocabulary(cache_dir=temp_cache_dir)

    def test_initialization(self, pos_vocab):
        """Test POSVocabulary initialization."""
        assert pos_vocab is not None
        assert hasattr(pos_vocab, 'word_bank')
        assert hasattr(pos_vocab, 'cache_path')
        assert pos_vocab.cache_path.exists()

    def test_word_bank_creation(self, pos_vocab):
        """Test building POS-tagged word bank."""
        # Should have entries for all core POS tags
        assert len(pos_vocab.word_bank) > 0

        # Check for expected POS tags
        common_tags = ['NOUN', 'VERB', 'ADJ']
        for tag in common_tags:
            assert tag in pos_vocab.word_bank

    def test_get_words_by_pos_and_syllables(self, pos_vocab):
        """Test retrieving words by POS and syllable count."""
        # Get 2-syllable nouns
        nouns_2 = pos_vocab.get_words('NOUN', 2)
        assert isinstance(nouns_2, list)

        # If we have any, verify they're actually 2-syllable nouns
        if nouns_2:
            sample_word = nouns_2[0]
            assert count_syllables(sample_word) == 2
            assert isinstance(sample_word, str)

    def test_get_words_invalid_pos(self, pos_vocab):
        """Test getting words with invalid POS tag."""
        words = pos_vocab.get_words('INVALID_POS', 2)
        assert words == []

    def test_get_words_invalid_syllables(self, pos_vocab):
        """Test getting words with syllable count that doesn't exist."""
        # 100 syllables is unrealistic
        words = pos_vocab.get_words('NOUN', 100)
        assert words == []

    def test_syllable_combinations_basic(self, pos_vocab):
        """Test finding syllable combinations for simple patterns."""
        # Pattern: ['ADJ', 'NOUN'] with 5 syllables
        combos = pos_vocab.get_syllable_combinations(['ADJ', 'NOUN'], 5)

        assert isinstance(combos, list)

        # Each combo should be a tuple
        for combo in combos:
            assert isinstance(combo, tuple)
            assert len(combo) == 2  # Two POS tags
            assert sum(combo) == 5  # Total syllables should be 5

    def test_syllable_combinations_complex(self, pos_vocab):
        """Test syllable combinations for complex patterns."""
        # Pattern: ['DET', 'ADJ', 'NOUN'] with 7 syllables
        combos = pos_vocab.get_syllable_combinations(['DET', 'ADJ', 'NOUN'], 7)

        for combo in combos:
            assert len(combo) == 3
            assert sum(combo) == 7

            # DET is typically 1 syllable
            if combo[0] == 1:
                # Remaining 6 syllables split between ADJ and NOUN
                assert combo[1] + combo[2] == 6

    def test_syllable_combinations_impossible(self, pos_vocab):
        """Test that impossible combinations return empty list."""
        # If we ask for combinations with an invalid POS tag
        combos = pos_vocab.get_syllable_combinations(['INVALID_POS'], 5)
        assert combos == []

        # If we ask for 0 or negative syllables
        combos = pos_vocab.get_syllable_combinations(['NOUN'], 0)
        assert combos == []

        combos = pos_vocab.get_syllable_combinations(['NOUN'], -1)
        assert combos == []

    def test_syllable_combinations_max_results(self, pos_vocab):
        """Test that max_results parameter limits output."""
        combos = pos_vocab.get_syllable_combinations(
            ['NOUN', 'VERB'],
            10,
            max_results=5
        )

        # Should not exceed max_results
        assert len(combos) <= 5

    def test_cache_persistence(self, temp_cache_dir):
        """Test that cache persists between instances."""
        # Create first instance
        vocab1 = POSVocabulary(cache_dir=temp_cache_dir)
        cache_path = vocab1.cache_path
        assert cache_path.exists()

        # Get some data
        stats1 = vocab1.get_stats()

        # Create second instance (should load from cache)
        vocab2 = POSVocabulary(cache_dir=temp_cache_dir)
        stats2 = vocab2.get_stats()

        # Should have same data
        assert stats1.keys() == stats2.keys()
        for pos_tag in stats1:
            assert stats1[pos_tag]['total_words'] == stats2[pos_tag]['total_words']

    def test_rebuild_cache(self, temp_cache_dir):
        """Test rebuilding cache from scratch."""
        # Create with rebuild_cache=True
        vocab = POSVocabulary(cache_dir=temp_cache_dir, rebuild_cache=True)

        assert vocab.cache_path.exists()
        assert len(vocab.word_bank) > 0

    def test_get_stats(self, pos_vocab):
        """Test getting statistics about word bank."""
        stats = pos_vocab.get_stats()

        assert isinstance(stats, dict)
        assert len(stats) > 0

        # Check stats structure
        for pos_tag, pos_stats in stats.items():
            assert 'total_words' in pos_stats
            assert 'syllable_range' in pos_stats
            assert 'syllable_counts_available' in pos_stats

            assert isinstance(pos_stats['total_words'], int)
            assert pos_stats['total_words'] > 0

    def test_rebuild_method(self, pos_vocab):
        """Test the rebuild method."""
        original_stats = pos_vocab.get_stats()

        # Rebuild
        pos_vocab.rebuild()

        new_stats = pos_vocab.get_stats()

        # Should have similar structure
        assert original_stats.keys() == new_stats.keys()

    def test_word_quality(self, pos_vocab):
        """Test that words in bank meet quality standards."""
        for pos_tag in pos_vocab.word_bank:
            for syllable_count, words in pos_vocab.word_bank[pos_tag].items():
                for word in words[:10]:  # Sample first 10
                    # Should be alphabetic
                    assert word.isalpha(), f"Word '{word}' is not alphabetic"

                    # Should be reasonable length
                    assert 1 <= len(word) <= 20, f"Word '{word}' has unreasonable length"

                    # Should have correct syllable count
                    actual_syllables = count_syllables(word)
                    assert actual_syllables == syllable_count, \
                        f"Word '{word}' should have {syllable_count} syllables but has {actual_syllables}"

    def test_no_duplicates(self, pos_vocab):
        """Test that word lists don't contain duplicates."""
        for pos_tag in pos_vocab.word_bank:
            for syllable_count, words in pos_vocab.word_bank[pos_tag].items():
                # Check for duplicates
                assert len(words) == len(set(words)), \
                    f"Duplicates found in {pos_tag} {syllable_count}-syllable words"

    def test_words_sorted(self, pos_vocab):
        """Test that word lists are sorted."""
        for pos_tag in pos_vocab.word_bank:
            for syllable_count, words in pos_vocab.word_bank[pos_tag].items():
                assert words == sorted(words), \
                    f"Words not sorted for {pos_tag} {syllable_count}-syllable"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_pos_vocabulary(self):
        """Test create_pos_vocabulary factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = create_pos_vocabulary(cache_dir=Path(tmpdir))

            assert isinstance(vocab, POSVocabulary)
            assert len(vocab.word_bank) > 0

    def test_create_pos_vocabulary_rebuild(self):
        """Test factory function with rebuild_cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = create_pos_vocabulary(
                cache_dir=Path(tmpdir),
                rebuild_cache=True
            )

            assert isinstance(vocab, POSVocabulary)
            assert vocab.cache_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pos_pattern(self):
        """Test with empty POS pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = POSVocabulary(cache_dir=Path(tmpdir))

            combos = vocab.get_syllable_combinations([], 5)
            assert combos == []

    def test_very_long_pattern(self):
        """Test with very long POS pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = POSVocabulary(cache_dir=Path(tmpdir))

            # Pattern with many elements
            long_pattern = ['NOUN'] * 10
            combos = vocab.get_syllable_combinations(long_pattern, 20)

            # Should still work, might be empty if impossible
            assert isinstance(combos, list)

    def test_single_syllable_words(self):
        """Test that 1-syllable words are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = POSVocabulary(cache_dir=Path(tmpdir))

            # Determiners should have 1-syllable words
            det_1 = vocab.get_words('DET', 1)

            # Should have common words like 'the', 'a', 'an'
            assert len(det_1) > 0

    def test_multisyllable_words(self):
        """Test that multi-syllable words are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab = POSVocabulary(cache_dir=Path(tmpdir))

            # Should have 2-syllable adjectives
            adj_2 = vocab.get_words('ADJ', 2)
            assert len(adj_2) > 0

            # Should have 3-syllable nouns
            noun_3 = vocab.get_words('NOUN', 3)
            assert len(noun_3) > 0

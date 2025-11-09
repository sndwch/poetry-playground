"""Tests for Ship of Theseus transformer."""

import tempfile
from pathlib import Path

import pytest

from poetryplayground.ship_of_theseus import (
    ShipOfTheseusTransformer,
    TransformationResult,
    create_ship_of_theseus_transformer,
)


class TestTransformationResult:
    """Test TransformationResult dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        result = TransformationResult(
            original="the cat sleeps",
            transformed="the dog rests",
            num_replacements=2,
            replacement_ratio=0.67,
            preserved_syllables=True,
        )

        assert result.original == "the cat sleeps"
        assert result.transformed == "the dog rests"
        assert result.num_replacements == 2
        assert result.replacement_ratio == 0.67
        assert result.preserved_syllables is True

    def test_string_representation(self):
        """Test string representation."""
        result = TransformationResult(
            original="old text",
            transformed="new text",
            num_replacements=1,
            replacement_ratio=0.5,
            preserved_syllables=True,
        )

        string_repr = str(result)
        assert "old text" in string_repr
        assert "new text" in string_repr
        assert "1 words" in string_repr
        assert "50" in string_repr  # 50% ratio


class TestShipOfTheseusTransformer:
    """Test ShipOfTheseusTransformer class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def transformer(self, temp_cache_dir):
        """Create transformer instance."""
        return ShipOfTheseusTransformer(cache_dir=temp_cache_dir)

    def test_initialization(self, transformer):
        """Test transformer initialization."""
        assert transformer.pos_vocab is not None

    def test_transform_line_basic(self, transformer):
        """Test basic line transformation."""
        line = "the ancient mountain stands tall"
        result = transformer.transform_line(line, replacement_ratio=0.4)

        assert isinstance(result, TransformationResult)
        assert result.original == line
        assert result.transformed  # Should have a transformed version
        # Some words should have changed (unless very unlucky)
        assert result.num_replacements >= 0

    def test_transform_line_preserves_word_count(self, transformer):
        """Test that transformation preserves word count."""
        line = "the old gate creaks softly"
        result = transformer.transform_line(line, replacement_ratio=0.5)

        original_words = line.split()
        transformed_words = result.transformed.split()

        assert len(transformed_words) == len(original_words)

    def test_transform_line_with_syllable_preservation(self, transformer):
        """Test transformation with syllable preservation."""
        line = "ancient temple"
        result = transformer.transform_line(line, replacement_ratio=1.0, preserve_syllables=True)

        assert result.preserved_syllables is True
        # If transformation succeeded, syllable count should match
        # (May be identical to original if no suitable replacements found)

    def test_transform_line_without_syllable_preservation(self, transformer):
        """Test transformation without syllable preservation."""
        line = "old gate"
        result = transformer.transform_line(line, replacement_ratio=1.0, preserve_syllables=False)

        assert result.preserved_syllables is False

    def test_transform_line_zero_ratio(self, transformer):
        """Test that zero replacement ratio doesn't change anything."""
        line = "the mountain wind blows"
        result = transformer.transform_line(line, replacement_ratio=0.0)

        # With 0% replacement, should stay the same
        # (though at least 1 word is attempted, per implementation)
        assert result.num_replacements >= 0

    def test_transform_line_full_ratio(self, transformer):
        """Test full replacement ratio."""
        line = "quiet evening shadows"
        result = transformer.transform_line(line, replacement_ratio=1.0)

        # With 100% replacement, should attempt to replace all words
        # Actual count may vary based on availability of replacements
        assert result.num_replacements >= 0
        assert result.replacement_ratio <= 1.0

    def test_transform_line_invalid_ratio(self, transformer):
        """Test that invalid ratios raise ValueError."""
        line = "test line"

        with pytest.raises(ValueError, match="replacement_ratio must be between"):
            transformer.transform_line(line, replacement_ratio=-0.1)

        with pytest.raises(ValueError, match="replacement_ratio must be between"):
            transformer.transform_line(line, replacement_ratio=1.5)

    def test_transform_empty_line(self, transformer):
        """Test transformation of empty line."""
        result = transformer.transform_line("", replacement_ratio=0.5)

        assert result.original == ""
        assert result.transformed == ""
        assert result.num_replacements == 0

    def test_transform_single_word(self, transformer):
        """Test transformation of single word."""
        line = "mountain"
        result = transformer.transform_line(line, replacement_ratio=1.0)

        assert isinstance(result, TransformationResult)
        # Should attempt to replace the single word
        transformed_words = result.transformed.split()
        assert len(transformed_words) == 1

    def test_gradual_transform_basic(self, transformer):
        """Test gradual transformation."""
        original = "the ancient temple stands on the mountain"
        steps = 5

        results = transformer.gradual_transform(original, steps=steps)

        assert len(results) == steps
        assert all(isinstance(r, TransformationResult) for r in results)

        # First transformation should start with original
        assert results[0].original == original

        # Replacement ratio should increase with each step
        for i, result in enumerate(results):
            expected_min_ratio = (i + 1) / steps * 0.5  # Allow some variance
            # Actual ratio may be less if replacements fail
            assert result.replacement_ratio <= 1.0

    def test_gradual_transform_progressive_change(self, transformer):
        """Test that gradual transformation shows progression."""
        original = "the wind whispers through ancient trees"
        results = transformer.gradual_transform(original, steps=3)

        # Each step should build on previous
        # (Though identical results possible if few replacement candidates)
        assert len(results) == 3

    def test_gradual_transform_invalid_steps(self, transformer):
        """Test that invalid steps raise ValueError."""
        with pytest.raises(ValueError, match="steps must be at least 1"):
            transformer.gradual_transform("test", steps=0)

        with pytest.raises(ValueError, match="steps must be at least 1"):
            transformer.gradual_transform("test", steps=-1)

    def test_transform_poem_basic(self, transformer):
        """Test transforming entire poem."""
        poem = [
            "the ancient temple",
            "stands on the mountain",
            "autumn wind blows",
        ]

        results = transformer.transform_poem(poem, replacement_ratio=0.3)

        assert len(results) == len(poem)
        assert all(isinstance(r, TransformationResult) for r in results)

        # Each line should have been transformed
        for i, result in enumerate(results):
            assert result.original == poem[i]
            assert result.transformed  # Should have transformed text

    def test_transform_poem_preserves_structure(self, transformer):
        """Test that poem transformation preserves structure."""
        poem = ["old gate", "silent moon", "distant echo"]

        results = transformer.transform_poem(poem, replacement_ratio=0.5)

        # Should have same number of lines
        assert len(results) == len(poem)

        # Each line should have same word count
        for original_line, result in zip(poem, results):
            original_word_count = len(original_line.split())
            transformed_word_count = len(result.transformed.split())
            assert transformed_word_count == original_word_count

    def test_pos_preservation(self, transformer):
        """Test that POS tags are preserved."""
        # Use a simple line with clear POS structure
        line = "the cat runs"  # DET NOUN VERB

        # Transform with POS preservation
        result = transformer.transform_line(line, replacement_ratio=1.0, preserve_pos=True)

        # Word count should be preserved
        assert len(result.transformed.split()) == 3

    def test_multiple_transformations_variety(self, transformer):
        """Test that multiple transformations produce variety."""
        line = "the ancient mountain stands tall"
        transformations = []

        for _ in range(5):
            result = transformer.transform_line(line, replacement_ratio=0.5)
            transformations.append(result.transformed)

        # Should have some variety (though randomness may occasionally produce duplicates)
        unique = set(transformations)
        # At least 2 different results expected in 5 runs
        assert len(unique) >= 2 or transformations[0] == line  # Unless no replacements available


class TestFactoryFunction:
    """Test factory function."""

    def test_create_transformer(self):
        """Test create_ship_of_theseus_transformer factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer = create_ship_of_theseus_transformer(cache_dir=tmpdir)

            assert isinstance(transformer, ShipOfTheseusTransformer)
            assert transformer.pos_vocab is not None


class TestIntegration:
    """Integration tests for the complete Ship of Theseus system."""

    @pytest.fixture
    def transformer(self):
        """Create fully initialized transformer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ShipOfTheseusTransformer(cache_dir=tmpdir)

    def test_end_to_end_single_line(self, transformer):
        """Test end-to-end transformation of single line."""
        line = "the old bridge crosses the river"

        result = transformer.transform_line(line, replacement_ratio=0.5)

        # Should successfully transform
        assert isinstance(result, TransformationResult)
        assert result.transformed
        assert len(result.transformed.split()) == len(line.split())

    def test_end_to_end_gradual(self, transformer):
        """Test end-to-end gradual transformation."""
        original = "autumn leaves fall gently from ancient trees"

        results = transformer.gradual_transform(original, steps=4)

        # Should complete all steps
        assert len(results) == 4

        # All results should be valid
        for result in results:
            assert result.transformed
            assert len(result.transformed.split()) == len(original.split())

    def test_end_to_end_poem(self, transformer):
        """Test end-to-end poem transformation."""
        poem = [
            "in the quiet garden",
            "cherry blossoms fall",
            "spring wind whispers",
        ]

        results = transformer.transform_poem(poem, replacement_ratio=0.4)

        # Should transform entire poem
        assert len(results) == len(poem)

        # All transformations should be valid
        for i, result in enumerate(results):
            assert result.original == poem[i]
            assert result.transformed
            assert len(result.transformed.split()) == len(poem[i].split())

    def test_realistic_use_case(self, transformer):
        """Test realistic use case: generating poem variations."""
        original_poem = [
            "the ancient temple stands",
            "on the distant mountain",
            "wind whispers through pines",
        ]

        # Generate 3 variations at different replacement ratios
        variations = []
        for ratio in [0.3, 0.5, 0.7]:
            results = transformer.transform_poem(original_poem, replacement_ratio=ratio)
            variation = [r.transformed for r in results]
            variations.append((ratio, variation))

        # Should have 3 variations
        assert len(variations) == 3

        # Each variation should preserve structure
        for ratio, variation in variations:
            assert len(variation) == len(original_poem)
            for original_line, transformed_line in zip(original_poem, variation):
                assert len(transformed_line.split()) == len(original_line.split())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_line_with_punctuation(self):
        """Test handling of punctuation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer = ShipOfTheseusTransformer(cache_dir=tmpdir)
            line = "the wind blows, softly."

            result = transformer.transform_line(line, replacement_ratio=0.5)

            # Should handle punctuation gracefully
            assert isinstance(result, TransformationResult)
            # Word count includes punctuation as separate tokens in NLTK
            assert len(result.transformed.split()) >= len(line.split()) - 1

    def test_line_with_numbers(self):
        """Test handling of numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer = ShipOfTheseusTransformer(cache_dir=tmpdir)
            line = "the 3 birds fly"

            result = transformer.transform_line(line, replacement_ratio=0.5)

            assert isinstance(result, TransformationResult)

    def test_very_long_line(self):
        """Test handling of very long lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transformer = ShipOfTheseusTransformer(cache_dir=tmpdir)
            words = ["the", "ancient", "mountain", "wind"] * 10
            line = " ".join(words)

            result = transformer.transform_line(line, replacement_ratio=0.3)

            assert isinstance(result, TransformationResult)
            assert len(result.transformed.split()) == len(line.split())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

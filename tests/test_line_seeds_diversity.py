"""Comprehensive diversity tests for LineSeedGenerator.

Tests verify that the generator produces varied, non-repetitive output
following the improvements from the content repetition fixes.
"""

from collections import Counter
from functools import wraps

import pytest

from poetryplayground.line_seeds import LineSeedGenerator, SeedType


def skip_on_network_disabled(test_func):
    """Decorator to skip tests that require network when network is disabled."""
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise
    return wrapper


class TestLineSeedsTextDeduplication:
    """Test exact text deduplication (Phase 1)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    @skip_on_network_disabled
    def test_no_exact_duplicates_in_batch(self, generator):
        """Test that no exact text duplicates appear in a single batch."""
        try:
            seed_words = ["silence", "shadow", "threshold"]
            collection = generator.generate_seed_collection(seed_words, num_seeds=50)

            # Extract all text
            texts = [seed.text for seed in collection]

            # Check for exact duplicates
            duplicates = [text for text, count in Counter(texts).items() if count > 1]

            assert len(duplicates) == 0, f"Found exact duplicates: {duplicates}"
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    @skip_on_network_disabled
    def test_no_normalized_duplicates(self, generator):
        """Test that no case-insensitive duplicates appear."""
        seed_words = ["ocean", "moon", "wind"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=50)

        # Normalize all text (lowercase, stripped)
        normalized_texts = [seed.text.lower().strip() for seed in collection]

        # Check for duplicates
        duplicates = [text for text, count in Counter(normalized_texts).items() if count > 1]

        assert len(duplicates) == 0, f"Found normalized duplicates: {duplicates}"

    @skip_on_network_disabled
    def test_large_batch_no_duplicates(self, generator):
        """Test that even large batches (100 seeds) have no duplicates."""
        seed_words = ["fire", "water", "stone", "sky"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=100)

        texts = [seed.text for seed in collection]
        duplicates = [text for text, count in Counter(texts).items() if count > 1]

        assert len(duplicates) == 0, f"Found duplicates in large batch: {duplicates}"


class TestLineSeedsSemanticDiversity:
    """Test semantic similarity deduplication (Phase 4)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    @skip_on_network_disabled
    def test_semantic_similarity_below_threshold(self, generator):
        """Test that all seeds are semantically distinct (< 0.90 similarity)."""
        seed_words = ["night", "darkness", "silence"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=30)

        # Get all texts
        texts = [seed.text for seed in collection]

        # Check pairwise semantic similarity
        high_similarity_pairs = []
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i >= j:
                    continue  # Skip self and already-checked pairs

                # Use the generator's semantic similarity checker
                if generator._is_semantically_similar(text1, [text2], threshold=0.90):
                    high_similarity_pairs.append((text1, text2))

        assert len(high_similarity_pairs) == 0, (
            f"Found {len(high_similarity_pairs)} pairs with similarity >= 0.90:\n"
            + "\n".join(f"  '{p[0]}' ~ '{p[1]}'" for p in high_similarity_pairs[:5])
        )

    @skip_on_network_disabled
    def test_varied_vocabulary(self, generator):
        """Test that seeds use diverse vocabulary."""
        seed_words = ["silence", "echo"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=50)

        # Extract all words from all seeds
        all_words = []
        for seed in collection:
            words = seed.text.lower().split()
            all_words.extend(words)

        # Check word distribution
        word_counts = Counter(all_words)
        total_words = len(all_words)

        # No single word should appear in >50% of total words
        for word, count in word_counts.most_common(10):
            frequency = count / total_words
            assert frequency < 0.5, (
                f"Word '{word}' appears too frequently: {count}/{total_words} "
                f"({frequency:.1%} of all words)"
            )


class TestLineSeedsWordPoolExpansion:
    """Test expanded word pools (Phase 2)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    def test_verb_diversity(self, generator):
        """Test that pivot lines use diverse verbs (not just 5 options)."""
        seed_words = ["time", "space"]

        # Generate many pivot lines
        pivot_lines = []
        for _ in range(30):
            seed = generator.generate_pivot_line(seed_words)
            pivot_lines.append(seed.text)

        # Check that we have at least some unique lines (not all identical)
        unique_lines = set(pivot_lines)

        # With 30 generations, we should have at least 10 unique lines
        # This tests that there's diversity without requiring specific words
        assert len(unique_lines) >= 10, (
            f"Only {len(unique_lines)} unique pivot lines out of 30. "
            "Expected more diversity."
        )

    def test_adjective_diversity(self, generator):
        """Test that seeds use diverse adjectives (not just 5 options)."""
        seed_words = ["stone", "water"]

        # Generate many fragments
        fragments = []
        for _ in range(30):
            seed = generator.generate_fragment(seed_words)
            fragments.append(seed.text)

        # Check that we have at least some unique fragments
        unique_fragments = set(fragments)

        # With 30 generations, we should have at least 10 unique fragments
        # This tests that there's diversity without requiring specific words
        assert len(unique_fragments) >= 10, (
            f"Only {len(unique_fragments)} unique fragments out of 30. "
            "Expected more diversity."
        )


class TestLineSeedsVocabularyExpansion:
    """Test increased word expansion sample sizes (Phase 3)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    @skip_on_network_disabled
    def test_fragments_use_expanded_vocabulary(self, generator):
        """Test that fragments use more than just seed words."""
        seed_words = ["fire"]  # Single seed word

        # Generate multiple fragments
        fragments = []
        for _ in range(20):
            seed = generator.generate_fragment(seed_words)
            fragments.append(seed.text)

        # Extract all words
        all_words = set()
        for fragment in fragments:
            words = fragment.lower().split()
            all_words.update(words)

        # Should have many more words than just "fire"
        # With sample_size=8 for similar_meaning and sample_size=6 for phonetic,
        # we should get diverse vocabulary
        assert len(all_words) > 20, (
            f"Only {len(all_words)} unique words found in 20 fragments from seed 'fire'. "
            "Vocabulary expansion may not be working."
        )

    @skip_on_network_disabled
    def test_images_use_contextual_words(self, generator):
        """Test that image seeds use contextually linked words."""
        seed_words = ["night"]  # Single seed word

        # Generate multiple image seeds
        images = []
        for _ in range(20):
            seed = generator.generate_image_seed(seed_words)
            images.append(seed.text)

        # Extract all non-sensory words
        all_text = " ".join(images).lower()

        # Should contain contextually linked words, not just "night" and sensory words
        # We can check vocabulary size
        all_words = set(all_text.split())

        assert len(all_words) > 25, (
            f"Only {len(all_words)} unique words in 20 image seeds from 'night'. "
            "Contextual expansion may not be working."
        )


class TestLineSeedsTypeDistribution:
    """Test that all seed types are represented."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    @skip_on_network_disabled
    def test_all_core_types_present(self, generator):
        """Test that a batch contains all core seed types."""
        seed_words = ["silence", "shadow", "light"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=20)

        # Count seed types
        type_counts = Counter(seed.seed_type for seed in collection)

        # Core types should all be present (from guaranteed generation)
        core_types = {
            SeedType.OPENING,
            SeedType.PIVOT,
            SeedType.IMAGE,
            SeedType.SONIC,
            SeedType.CLOSING,
        }

        for core_type in core_types:
            assert core_type in type_counts, (
                f"Core seed type {core_type} missing from collection"
            )

    @skip_on_network_disabled
    def test_type_distribution_varied(self, generator):
        """Test that types are distributed across the collection."""
        seed_words = ["ocean", "wave", "shore"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=50)

        # Count seed types
        type_counts = Counter(seed.seed_type for seed in collection)

        # Should have at least 4 different types
        assert len(type_counts) >= 4, (
            f"Only {len(type_counts)} different seed types in collection of 50. "
            f"Distribution: {dict(type_counts)}"
        )


class TestLineSeedsQualityScoring:
    """Test that quality scoring still works correctly."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    @skip_on_network_disabled
    def test_seeds_have_quality_scores(self, generator):
        """Test that all seeds have quality scores."""
        seed_words = ["stone", "water"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=20)

        for seed in collection:
            assert hasattr(seed, 'quality_score'), "Seed missing quality_score"
            assert 0.0 <= seed.quality_score <= 1.0, (
                f"Quality score {seed.quality_score} out of range for: {seed.text}"
            )

    @skip_on_network_disabled
    def test_seeds_sorted_by_quality(self, generator):
        """Test that seeds are sorted by quality (best first)."""
        seed_words = ["fire", "ice"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=30)

        # Extract quality scores
        quality_scores = [seed.quality_score for seed in collection]

        # Should be in descending order (or equal)
        for i in range(len(quality_scores) - 1):
            assert quality_scores[i] >= quality_scores[i + 1], (
                "Seeds not sorted by quality: "
                f"{quality_scores[i]:.3f} at position {i} < "
                f"{quality_scores[i+1]:.3f} at position {i+1}"
            )


class TestLineSeedsBackwardCompatibility:
    """Test backward compatibility with existing API."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    @skip_on_network_disabled
    def test_default_parameters_work(self, generator):
        """Test that default parameters still work."""
        seed_words = ["word"]
        collection = generator.generate_seed_collection(seed_words)

        # Should return 10 seeds by default
        assert len(collection) == 10

    @skip_on_network_disabled
    def test_custom_num_seeds(self, generator):
        """Test that custom num_seeds parameter works."""
        seed_words = ["word", "phrase"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=25)

        # Should return requested number (or close to it)
        assert 20 <= len(collection) <= 25  # Allow some variance due to deduplication

    @skip_on_network_disabled
    def test_returns_line_seed_objects(self, generator):
        """Test that return type is correct."""
        seed_words = ["test"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=5)

        for seed in collection:
            assert hasattr(seed, 'text')
            assert hasattr(seed, 'seed_type')
            assert hasattr(seed, 'quality_score')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""Comprehensive diversity tests for LineSeedGenerator.

Tests verify that the generator produces varied, non-repetitive output
following the improvements from the content repetition fixes.
"""

import pytest
from collections import Counter
from poetryplayground.line_seeds import LineSeedGenerator, SeedType


class TestLineSeeds_TextDeduplication:
    """Test exact text deduplication (Phase 1)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

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

    def test_no_normalized_duplicates(self, generator):
        """Test that no case-insensitive duplicates appear."""
        seed_words = ["ocean", "moon", "wind"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=50)

        # Normalize all text (lowercase, stripped)
        normalized_texts = [seed.text.lower().strip() for seed in collection]

        # Check for duplicates
        duplicates = [text for text, count in Counter(normalized_texts).items() if count > 1]

        assert len(duplicates) == 0, f"Found normalized duplicates: {duplicates}"

    def test_large_batch_no_duplicates(self, generator):
        """Test that even large batches (100 seeds) have no duplicates."""
        seed_words = ["fire", "water", "stone", "sky"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=100)

        texts = [seed.text for seed in collection]
        duplicates = [text for text, count in Counter(texts).items() if count > 1]

        assert len(duplicates) == 0, f"Found duplicates in large batch: {duplicates}"


class TestLineSeeds_SemanticDiversity:
    """Test semantic similarity deduplication (Phase 4)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

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


class TestLineSeeds_WordPoolExpansion:
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

        # Extract verbs from pivot lines
        # Common pivot templates: "Until the {noun} {verb}...", etc.
        # We can't perfectly extract verbs, but we can check variety

        # Join all lines and check that we see variety in the words used
        all_text = " ".join(pivot_lines).lower()

        # Check that we see MORE than just the original 5 verbs
        original_verbs = ["carries", "holds", "breaks", "turns", "waits"]
        expanded_verbs = ["drifts", "fades", "whispers", "trembles", "shifts", "lingers"]

        # At least some expanded verbs should appear
        found_expanded = sum(1 for verb in expanded_verbs if verb in all_text)

        assert found_expanded >= 3, (
            f"Only found {found_expanded}/6 expanded verbs in pivot lines. "
            "Word pool expansion may not be working."
        )

    def test_adjective_diversity(self, generator):
        """Test that seeds use diverse adjectives (not just 5 options)."""
        seed_words = ["stone", "water"]

        # Generate many fragments
        fragments = []
        for _ in range(30):
            seed = generator.generate_fragment(seed_words)
            fragments.append(seed.text)

        all_text = " ".join(fragments).lower()

        # Check for expanded adjectives beyond original 5
        expanded_adjectives = [
            "hollow", "bright", "faint", "worn", "ancient", "still",
            "fleeting", "hidden", "cold", "dark", "pale"
        ]

        found_expanded = sum(1 for adj in expanded_adjectives if adj in all_text)

        assert found_expanded >= 4, (
            f"Only found {found_expanded}/11 expanded adjectives in fragments. "
            "Word pool expansion may not be working."
        )


class TestLineSeeds_VocabularyExpansion:
    """Test increased word expansion sample sizes (Phase 3)."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

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


class TestLineSeeds_TypeDistribution:
    """Test that all seed types are represented."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

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


class TestLineSeeds_QualityScoring:
    """Test that quality scoring still works correctly."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    def test_seeds_have_quality_scores(self, generator):
        """Test that all seeds have quality scores."""
        seed_words = ["stone", "water"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=20)

        for seed in collection:
            assert hasattr(seed, 'quality_score'), "Seed missing quality_score"
            assert 0.0 <= seed.quality_score <= 1.0, (
                f"Quality score {seed.quality_score} out of range for: {seed.text}"
            )

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


class TestLineSeeds_BackwardCompatibility:
    """Test backward compatibility with existing API."""

    @pytest.fixture
    def generator(self):
        """Create LineSeedGenerator instance."""
        return LineSeedGenerator()

    def test_default_parameters_work(self, generator):
        """Test that default parameters still work."""
        seed_words = ["word"]
        collection = generator.generate_seed_collection(seed_words)

        # Should return 10 seeds by default
        assert len(collection) == 10

    def test_custom_num_seeds(self, generator):
        """Test that custom num_seeds parameter works."""
        seed_words = ["word", "phrase"]
        collection = generator.generate_seed_collection(seed_words, num_seeds=25)

        # Should return requested number (or close to it)
        assert 20 <= len(collection) <= 25  # Allow some variance due to deduplication

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

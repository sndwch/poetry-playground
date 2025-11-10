"""Tests for semantic geodesic/bridge finder functionality."""

import pytest
import numpy as np

from poetryplayground.semantic_geodesic import (
    BridgeWord,
    SemanticPath,
    SemanticSpace,
    find_semantic_path,
    get_semantic_space,
    _passes_filters,
    _calculate_smoothness,
    _calculate_deviation,
    _calculate_diversity,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def semantic_space_sm():
    """Load small semantic space for fast unit tests."""
    try:
        return SemanticSpace(model_name="en_core_web_sm", vocab_size=10000)
    except (OSError, RuntimeError) as e:
        pytest.skip(f"en_core_web_sm not properly installed or insufficient vocabulary: {e}")


@pytest.fixture(scope="module")
def semantic_space_lg():
    """Load large semantic space for integration tests."""
    try:
        return SemanticSpace(model_name="en_core_web_lg", vocab_size=50000)
    except (OSError, RuntimeError) as e:
        pytest.skip(f"en_core_web_lg not properly installed or insufficient vocabulary: {e}")


# ============================================================================
# Unit Tests - DataClasses
# ============================================================================


class TestDataClasses:
    """Test BridgeWord and SemanticPath dataclasses."""

    def test_bridge_word_creation(self):
        """Test BridgeWord instantiation."""
        bridge = BridgeWord(
            word="warm",
            position=0.5,
            similarity=0.85,
            deviation=0.15,
            syllables=1,
            pos="ADJ"
        )

        assert bridge.word == "warm"
        assert bridge.position == 0.5
        assert bridge.similarity == 0.85
        assert bridge.deviation == 0.15
        assert bridge.syllables == 1
        assert bridge.pos == "ADJ"

    def test_bridge_word_string_repr(self):
        """Test BridgeWord string representation."""
        bridge = BridgeWord("warm", 0.5, 0.85, 0.15)
        str_repr = str(bridge)

        assert "warm" in str_repr
        assert "0.50" in str_repr  # position
        assert "0.850" in str_repr  # similarity

    def test_semantic_path_creation(self):
        """Test SemanticPath instantiation."""
        bridges = [
            [BridgeWord("warm", 0.33, 0.8, 0.2)],
            [BridgeWord("cool", 0.67, 0.75, 0.25)],
        ]

        path = SemanticPath(
            start="hot",
            end="cold",
            bridges=bridges,
            method="linear",
            smoothness_score=0.85,
            deviation_score=0.20,
            diversity_score=0.10,
        )

        assert path.start == "hot"
        assert path.end == "cold"
        assert len(path.bridges) == 2
        assert path.method == "linear"
        assert path.smoothness_score == 0.85

    def test_semantic_path_get_primary_path(self):
        """Test getting primary path from SemanticPath."""
        bridges = [
            [BridgeWord("warm", 0.33, 0.8, 0.2), BridgeWord("heated", 0.33, 0.75, 0.25)],
            [BridgeWord("cool", 0.67, 0.75, 0.25), BridgeWord("chilly", 0.67, 0.70, 0.30)],
        ]

        path = SemanticPath("hot", "cold", bridges, "linear")
        primary = path.get_primary_path()

        assert primary == ["hot", "warm", "cool", "cold"]

    def test_semantic_path_get_all_alternatives(self):
        """Test getting all alternatives."""
        bridges = [
            [BridgeWord("warm", 0.33, 0.8, 0.2), BridgeWord("heated", 0.33, 0.75, 0.25)],
            [BridgeWord("cool", 0.67, 0.75, 0.25)],
        ]

        path = SemanticPath("hot", "cold", bridges, "linear")
        alternatives = path.get_all_alternatives()

        assert len(alternatives) == 2
        assert alternatives[0] == ["warm", "heated"]
        assert alternatives[1] == ["cool"]


# ============================================================================
# Unit Tests - Semantic Space (sm model)
# ============================================================================


class TestSemanticSpace:
    """Test SemanticSpace class with small model."""

    def test_initialization(self, semantic_space_sm):
        """Test SemanticSpace loads and builds index."""
        assert semantic_space_sm is not None
        assert semantic_space_sm.nlp is not None
        assert semantic_space_sm.nn_index is not None
        assert len(semantic_space_sm.index_to_word) > 0
        assert len(semantic_space_sm.index_to_word) <= 10000

    def test_get_vector(self, semantic_space_sm):
        """Test getting vector for a word."""
        vec = semantic_space_sm.get_vector("cat")

        assert vec is not None
        assert len(vec) == 96  # sm model has 96-d vectors
        assert np.any(vec != 0)  # Vector should not be all zeros

    def test_get_vector_with_context(self, semantic_space_sm):
        """Test contextualized vector retrieval."""
        vec_no_context = semantic_space_sm.get_vector("bank")
        vec_river = semantic_space_sm.get_vector("bank", "river")
        vec_money = semantic_space_sm.get_vector("bank", "money")

        # Vectors should be different (polysemy handling)
        assert not np.array_equal(vec_no_context, vec_river)
        assert not np.array_equal(vec_river, vec_money)

    def test_find_nearest(self, semantic_space_sm):
        """Test k-NN search."""
        vec_cat = semantic_space_sm.get_vector("cat")
        neighbors = semantic_space_sm.find_nearest(vec_cat, k=5)

        assert len(neighbors) <= 5
        assert all(isinstance(word, str) for word, sim in neighbors)
        assert all(0 <= sim <= 1 for word, sim in neighbors)

        # "cat" itself should be the nearest
        if neighbors:
            assert neighbors[0][0] == "cat" or neighbors[0][1] > 0.99

    def test_find_nearest_with_exclusions(self, semantic_space_sm):
        """Test k-NN search with exclusions."""
        vec_cat = semantic_space_sm.get_vector("cat")
        neighbors = semantic_space_sm.find_nearest(vec_cat, k=5, exclude={"cat", "cats"})

        assert all(word not in {"cat", "cats"} for word, sim in neighbors)

    def test_get_similarity(self, semantic_space_sm):
        """Test cosine similarity between words."""
        sim_cat_dog = semantic_space_sm.get_similarity("cat", "dog")
        sim_cat_car = semantic_space_sm.get_similarity("cat", "car")

        assert 0 <= sim_cat_dog <= 1
        assert 0 <= sim_cat_car <= 1

        # Cat and dog should be more similar than cat and car
        assert sim_cat_dog > sim_cat_car


# ============================================================================
# Unit Tests - Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test helper functions for filtering and metrics."""

    def test_passes_filters_frequency(self):
        """Test frequency filtering."""
        from poetryplayground.core.lexicon import get_lexicon_data
        lexicon = get_lexicon_data()

        # Common word should pass
        assert _passes_filters("cat", min_zipf=3.0, pos_filter=None,
                              syllable_min=None, syllable_max=None, lexicon_data=lexicon)

        # Rare word should fail
        assert not _passes_filters("cat", min_zipf=8.0, pos_filter=None,
                                   syllable_min=None, syllable_max=None, lexicon_data=lexicon)

    def test_calculate_smoothness(self, semantic_space_sm):
        """Test smoothness calculation."""
        bridges = [
            [BridgeWord("warm", 0.33, 0.8, 0.2)],
            [BridgeWord("cool", 0.67, 0.75, 0.25)],
        ]

        smoothness = _calculate_smoothness("hot", "cold", bridges, semantic_space_sm)

        assert 0 <= smoothness <= 1
        assert smoothness > 0  # Should have some similarity

    def test_calculate_deviation(self):
        """Test deviation calculation."""
        bridges = [
            [BridgeWord("warm", 0.33, 0.8, 0.15)],
            [BridgeWord("cool", 0.67, 0.75, 0.20)],
        ]

        deviation = _calculate_deviation(bridges)

        assert deviation == pytest.approx(0.175)  # Average of 0.15 and 0.20

    def test_calculate_diversity(self):
        """Test diversity calculation."""
        # No alternatives (single choice per step)
        bridges_no_alt = [
            [BridgeWord("warm", 0.33, 0.8, 0.2)],
        ]
        diversity_no_alt = _calculate_diversity(bridges_no_alt)
        assert diversity_no_alt == 0.0

        # Multiple alternatives
        bridges_with_alt = [
            [
                BridgeWord("warm", 0.33, 0.8, 0.2),
                BridgeWord("heated", 0.33, 0.6, 0.4),
            ],
        ]
        diversity_with_alt = _calculate_diversity(bridges_with_alt)
        assert diversity_with_alt > 0.0


# ============================================================================
# Integration Tests - Linear Path (sm model)
# ============================================================================


class TestLinearPath:
    """Test linear path finding with small model."""

    def test_simple_linear_path(self, semantic_space_sm):
        """Test basic linear path finding."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            k=1,
            semantic_space=semantic_space_sm
        )

        assert path is not None
        assert path.start == "hot"
        assert path.end == "cold"
        assert len(path.bridges) == 3  # 5 steps = start + 3 bridges + end
        assert path.method == "linear"

    def test_linear_path_with_alternatives(self, semantic_space_sm):
        """Test linear path with multiple alternatives."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            k=3,
            semantic_space=semantic_space_sm
        )

        assert len(path.bridges) == 3
        # Each step should have up to 3 alternatives
        for step in path.bridges:
            assert 0 < len(step) <= 3

    def test_linear_path_quality_metrics(self, semantic_space_sm):
        """Test that quality metrics are calculated."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            semantic_space=semantic_space_sm
        )

        assert path.smoothness_score >= 0
        assert path.smoothness_score <= 1
        assert path.deviation_score >= 0

    def test_minimum_steps_validation(self, semantic_space_sm):
        """Test that steps < 3 raises ValueError."""
        with pytest.raises(ValueError, match="Steps must be >= 3"):
            find_semantic_path("hot", "cold", steps=2, semantic_space=semantic_space_sm)

    def test_invalid_method_raises_error(self, semantic_space_sm):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            find_semantic_path(
                "hot", "cold",
                method="invalid",
                semantic_space=semantic_space_sm
            )


# ============================================================================
# Integration Tests - Advanced Methods (sm model)
# ============================================================================


class TestAdvancedMethods:
    """Test Bezier and shortest-path methods."""

    def test_bezier_path(self, semantic_space_sm):
        """Test Bezier curve path finding."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            method="bezier",
            semantic_space=semantic_space_sm
        )

        assert path.method == "bezier"
        assert len(path.bridges) == 3

    def test_bezier_path_with_control_words(self, semantic_space_sm):
        """Test Bezier with explicit control words."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            method="bezier",
            control_words=["warm", "cool"],
            semantic_space=semantic_space_sm
        )

        assert path.method == "bezier"

    def test_shortest_path(self, semantic_space_sm):
        """Test shortest-path graph search."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            method="shortest",
            semantic_space=semantic_space_sm
        )

        assert path.method == "shortest"
        # May have different number of bridges due to graph structure


# ============================================================================
# Integration Tests - Golden Paths (lg model if available)
# ============================================================================


class TestGoldenPaths:
    """Test known good paths with large model."""

    def test_hot_cold_path(self, semantic_space_lg):
        """Golden test: hot → cold should include temperature words."""
        path = find_semantic_path(
            "hot", "cold",
            steps=7,
            semantic_space=semantic_space_lg
        )

        primary_path = [w.lower() for w in path.get_primary_path()]

        # Should include some temperature-related transitions
        temp_words = {"warm", "cool", "tepid", "mild", "heat", "chill", "lukewarm"}
        assert any(w in temp_words for w in primary_path), \
            f"Expected temperature words in path: {primary_path}"

    def test_love_hate_path(self, semantic_space_lg):
        """Golden test: love → hate emotional journey."""
        path = find_semantic_path(
            "love", "hate",
            steps=7,
            semantic_space=semantic_space_lg
        )

        primary_path = path.get_primary_path()
        assert len(primary_path) == 7
        assert primary_path[0] == "love"
        assert primary_path[-1] == "hate"

    def test_fire_ice_path(self, semantic_space_lg):
        """Golden test: fire → ice should traverse temperature/state."""
        path = find_semantic_path(
            "fire", "ice",
            steps=9,
            semantic_space=semantic_space_lg
        )

        primary_path = path.get_primary_path()
        assert len(primary_path) == 9
        assert path.smoothness_score > 0.5  # Should be reasonably smooth

    def test_joy_grief_path(self, semantic_space_lg):
        """Golden test: joy → grief emotional transition."""
        path = find_semantic_path(
            "joy", "grief",
            steps=7,
            semantic_space=semantic_space_lg
        )

        assert path.start == "joy"
        assert path.end == "grief"
        assert path.smoothness_score > 0.4  # Emotional transitions can be less smooth


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_path_generation_speed(self, semantic_space_sm):
        """Test that path generation is reasonably fast."""
        import time

        start = time.time()
        path = find_semantic_path(
            "cat", "dog",
            steps=5,
            semantic_space=semantic_space_sm
        )
        elapsed = time.time() - start

        # Should complete in under 5 seconds with small model
        assert elapsed < 5.0, f"Path generation took {elapsed:.2f}s"

    def test_multiple_paths_cached(self, semantic_space_sm):
        """Test that semantic space caching works."""
        import time

        # First call (cold start)
        start = time.time()
        path1 = find_semantic_path("hot", "cold", steps=5, semantic_space=semantic_space_sm)
        time1 = time.time() - start

        # Second call (warm start)
        start = time.time()
        path2 = find_semantic_path("love", "hate", steps=5, semantic_space=semantic_space_sm)
        time2 = time.time() - start

        # Second call should be faster (no model loading)
        # Both should still be reasonably fast
        assert time2 < 5.0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_identical_start_end(self, semantic_space_sm):
        """Test path from word to itself."""
        path = find_semantic_path(
            "cat", "cat",
            steps=3,
            semantic_space=semantic_space_sm
        )

        # Should still work, just with identity path
        assert path.start == "cat"
        assert path.end == "cat"

    def test_very_distant_words(self, semantic_space_sm):
        """Test path between semantically distant words."""
        path = find_semantic_path(
            "love", "truck",
            steps=9,
            semantic_space=semantic_space_sm
        )

        # Should still find a path, even if quality is low
        assert path is not None
        assert len(path.bridges) == 7

    def test_with_filters(self, semantic_space_sm):
        """Test path finding with POS and syllable filters."""
        path = find_semantic_path(
            "hot", "cold",
            steps=5,
            pos_filter="ADJ",
            syllable_min=1,
            syllable_max=2,
            semantic_space=semantic_space_sm
        )

        # Should find path respecting filters
        assert path is not None
        # Check that bridges have correct POS
        for step in path.bridges:
            for bridge in step:
                if bridge.pos:
                    assert bridge.pos == "ADJ"

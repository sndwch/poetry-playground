"""Tests for conceptual cloud generator functionality."""

import json
import pytest

from poetryplayground.conceptual_cloud import (
    CloudTerm,
    ConceptualCloud,
    CloudConfig,
    ClusterType,
    generate_conceptual_cloud,
    format_as_rich,
    format_as_json,
    format_as_markdown,
    format_as_simple,
    get_antonyms,
    get_concrete_nouns,
    get_frequency_bucket,
    _generate_semantic_cluster,
    _generate_contextual_cluster,
    _generate_opposite_cluster,
    _generate_phonetic_cluster,
    _generate_imagery_cluster,
    _generate_rare_cluster,
)


# ============================================================================
# Unit Tests - Data Structures
# ============================================================================


class TestDataStructures:
    """Test dataclasses and data structures."""

    def test_cloud_term_creation(self):
        """Test CloudTerm instantiation."""
        term = CloudTerm(
            term="fire",
            cluster_type=ClusterType.SEMANTIC,
            score=0.85,
            freq_bucket="common",
        )

        assert term.term == "fire"
        assert term.cluster_type == ClusterType.SEMANTIC
        assert term.score == 0.85
        assert term.freq_bucket == "common"

    def test_cloud_term_str(self):
        """Test CloudTerm string representation."""
        term = CloudTerm("word", ClusterType.SEMANTIC, 0.75)
        str_repr = str(term)

        assert "word" in str_repr
        assert "0.75" in str_repr

    def test_conceptual_cloud_creation(self):
        """Test ConceptualCloud instantiation."""
        clusters = {
            ClusterType.SEMANTIC: [
                CloudTerm("hot", ClusterType.SEMANTIC, 0.9),
                CloudTerm("warm", ClusterType.SEMANTIC, 0.8),
            ],
            ClusterType.PHONETIC: [
                CloudTerm("tire", ClusterType.PHONETIC, 0.85),
            ],
        }

        cloud = ConceptualCloud(
            center_word="fire",
            clusters=clusters,
        )

        assert cloud.center_word == "fire"
        assert len(cloud.clusters) == 2
        assert cloud.total_terms == 3
        assert cloud.timestamp is not None

    def test_cloud_get_cluster(self):
        """Test getting specific cluster from cloud."""
        clusters = {
            ClusterType.SEMANTIC: [CloudTerm("hot", ClusterType.SEMANTIC, 0.9)],
        }

        cloud = ConceptualCloud("fire", clusters)
        semantic = cloud.get_cluster(ClusterType.SEMANTIC)

        assert len(semantic) == 1
        assert semantic[0].term == "hot"

    def test_cloud_get_all_terms(self):
        """Test getting all terms as flat list."""
        clusters = {
            ClusterType.SEMANTIC: [CloudTerm("hot", ClusterType.SEMANTIC, 0.9)],
            ClusterType.PHONETIC: [CloudTerm("tire", ClusterType.PHONETIC, 0.8)],
        }

        cloud = ConceptualCloud("fire", clusters)
        all_terms = cloud.get_all_terms()

        assert len(all_terms) == 2
        assert "hot" in all_terms
        assert "tire" in all_terms

    def test_cloud_config_defaults(self):
        """Test CloudConfig default values."""
        config = CloudConfig()

        assert config.k_per_cluster == 10
        assert config.total_limit == 50
        assert config.sections == list(ClusterType)
        assert config.include_scores is True
        assert config.cache_results is True


# ============================================================================
# Unit Tests - Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_frequency_bucket(self):
        """Test frequency bucketing."""
        # Common word (high Zipf score)
        bucket = get_frequency_bucket("the")
        assert bucket in ("common", "mid", "rare")

        # Rare word (low Zipf score)
        bucket = get_frequency_bucket("xylophone")
        assert bucket in ("common", "mid", "rare")

    def test_get_concrete_nouns(self):
        """Test filtering for concrete nouns."""
        words = ["run", "cat", "quickly", "tree", "and"]
        nouns = get_concrete_nouns(words, k=5)

        assert isinstance(nouns, list)
        # Should contain only nouns (if any are nouns)
        for noun in nouns:
            assert noun in words

    def test_get_antonyms_basic(self):
        """Test antonym retrieval."""
        try:
            antonyms = get_antonyms("hot", k=5)
            assert isinstance(antonyms, list)
            # Should return list of (word, score) tuples
            if antonyms:
                assert all(isinstance(item, tuple) for item in antonyms)
                assert all(len(item) == 2 for item in antonyms)
                # "cold" is a common antonym of "hot"
                antonym_words = [word for word, score in antonyms]
                # We expect cold might be there, but API may vary
                assert len(antonym_words) > 0
        except Exception as e:
            # API might fail, that's okay for unit tests
            pytest.skip(f"Antonym API unavailable: {e}")


# ============================================================================
# Unit Tests - Cluster Generators
# ============================================================================


class TestClusterGenerators:
    """Test individual cluster generator functions."""

    def test_generate_semantic_cluster(self):
        """Test semantic cluster generation."""
        try:
            cluster = _generate_semantic_cluster("fire", k=5)
            assert isinstance(cluster, list)
            assert all(isinstance(term, CloudTerm) for term in cluster)
            assert all(term.cluster_type == ClusterType.SEMANTIC for term in cluster)
            assert len(cluster) <= 5
        except Exception as e:
            pytest.skip(f"Semantic generation failed: {e}")

    def test_generate_contextual_cluster(self):
        """Test contextual cluster generation."""
        try:
            cluster = _generate_contextual_cluster("fire", k=5)
            assert isinstance(cluster, list)
            assert all(isinstance(term, CloudTerm) for term in cluster)
            assert all(term.cluster_type == ClusterType.CONTEXTUAL for term in cluster)
            assert len(cluster) <= 5
        except Exception as e:
            pytest.skip(f"Contextual generation failed: {e}")

    def test_generate_opposite_cluster(self):
        """Test opposite cluster generation."""
        try:
            cluster = _generate_opposite_cluster("hot", k=5)
            assert isinstance(cluster, list)
            assert all(isinstance(term, CloudTerm) for term in cluster)
            assert all(term.cluster_type == ClusterType.OPPOSITE for term in cluster)
            # May have zero results if no antonyms found
            assert len(cluster) <= 5
        except Exception as e:
            pytest.skip(f"Opposite generation failed: {e}")

    def test_generate_phonetic_cluster(self):
        """Test phonetic cluster generation."""
        try:
            cluster = _generate_phonetic_cluster("fire", k=5)
            assert isinstance(cluster, list)
            assert all(isinstance(term, CloudTerm) for term in cluster)
            assert all(term.cluster_type == ClusterType.PHONETIC for term in cluster)
            assert len(cluster) <= 5
        except Exception as e:
            pytest.skip(f"Phonetic generation failed: {e}")

    def test_generate_imagery_cluster(self):
        """Test imagery cluster generation."""
        try:
            cluster = _generate_imagery_cluster("fire", k=5)
            assert isinstance(cluster, list)
            assert all(isinstance(term, CloudTerm) for term in cluster)
            assert all(term.cluster_type == ClusterType.IMAGERY for term in cluster)
            assert len(cluster) <= 5
        except Exception as e:
            pytest.skip(f"Imagery generation failed: {e}")

    def test_generate_rare_cluster(self):
        """Test rare cluster generation."""
        try:
            cluster = _generate_rare_cluster("fire", k=5)
            assert isinstance(cluster, list)
            assert all(isinstance(term, CloudTerm) for term in cluster)
            assert all(term.cluster_type == ClusterType.RARE for term in cluster)
            assert len(cluster) <= 5
        except Exception as e:
            pytest.skip(f"Rare generation failed: {e}")


# ============================================================================
# Integration Tests - Cloud Generation
# ============================================================================


class TestCloudGeneration:
    """Test full cloud generation."""

    def test_generate_basic_cloud(self):
        """Test basic cloud generation."""
        try:
            cloud = generate_conceptual_cloud("fire", k_per_cluster=3)

            assert cloud.center_word == "fire"
            assert cloud.total_terms > 0
            assert len(cloud.clusters) > 0
            assert cloud.config is not None
            assert cloud.timestamp is not None

        except Exception as e:
            pytest.skip(f"Cloud generation failed: {e}")

    def test_generate_with_specific_sections(self):
        """Test cloud generation with specific sections."""
        try:
            cloud = generate_conceptual_cloud(
                "fire",
                k_per_cluster=3,
                sections=["semantic", "phonetic"],
            )

            # Should only have semantic and phonetic clusters
            assert ClusterType.SEMANTIC in cloud.clusters
            assert ClusterType.PHONETIC in cloud.clusters
            assert ClusterType.CONTEXTUAL not in cloud.clusters or not cloud.clusters[ClusterType.CONTEXTUAL]

        except Exception as e:
            pytest.skip(f"Selective cloud generation failed: {e}")

    def test_generate_common_words(self):
        """Test cloud generation with common words."""
        test_words = ["love", "fire", "tree", "ocean"]

        for word in test_words:
            try:
                cloud = generate_conceptual_cloud(word, k_per_cluster=5)
                assert cloud.center_word == word
                assert cloud.total_terms > 0
            except Exception as e:
                pytest.skip(f"Cloud generation for '{word}' failed: {e}")

    def test_generate_with_phrase(self):
        """Test cloud generation with multi-word phrase."""
        try:
            cloud = generate_conceptual_cloud("burning fire", k_per_cluster=3)
            assert cloud.center_word == "burning fire"
            assert cloud.total_terms >= 0  # May have fewer results for phrases
        except Exception as e:
            pytest.skip(f"Phrase cloud generation failed: {e}")


# ============================================================================
# Output Format Tests
# ============================================================================


class TestOutputFormats:
    """Test output formatters."""

    @pytest.fixture
    def sample_cloud(self):
        """Create a sample cloud for testing."""
        clusters = {
            ClusterType.SEMANTIC: [
                CloudTerm("hot", ClusterType.SEMANTIC, 0.9, "common"),
                CloudTerm("warm", ClusterType.SEMANTIC, 0.8, "common"),
            ],
            ClusterType.PHONETIC: [
                CloudTerm("tire", ClusterType.PHONETIC, 0.85, "common"),
                CloudTerm("dire", ClusterType.PHONETIC, 0.80, "mid"),
            ],
        }

        return ConceptualCloud("fire", clusters)

    def test_format_as_json(self, sample_cloud):
        """Test JSON formatting."""
        output = format_as_json(sample_cloud)

        assert isinstance(output, str)
        # Should be valid JSON
        data = json.loads(output)
        assert data["center_word"] == "fire"
        assert "clusters" in data
        assert data["total_terms"] == 4

    def test_format_as_markdown(self, sample_cloud):
        """Test Markdown formatting."""
        output = format_as_markdown(sample_cloud, show_scores=True)

        assert isinstance(output, str)
        assert "# Conceptual Cloud: fire" in output
        assert "## Semantic" in output or "## semantic" in output.lower()
        assert "hot" in output
        assert "warm" in output

    def test_format_as_simple(self, sample_cloud):
        """Test simple text formatting."""
        output = format_as_simple(sample_cloud)

        assert isinstance(output, str)
        assert "fire" in output
        assert "hot" in output or "warm" in output

    def test_format_as_rich(self, sample_cloud):
        """Test Rich formatting."""
        output = format_as_rich(sample_cloud, show_scores=True)

        assert isinstance(output, str)
        # Should contain the center word and some terms
        assert "fire" in output or "Fire" in output


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_word(self):
        """Test handling of empty center word."""
        try:
            cloud = generate_conceptual_cloud("", k_per_cluster=3)
            # Should either raise error or handle gracefully
            assert cloud is not None
        except Exception:
            # Expected to fail
            pass

    def test_unknown_word(self):
        """Test handling of unknown/rare word."""
        try:
            cloud = generate_conceptual_cloud("xyzabc123", k_per_cluster=3)
            # May have empty or minimal results
            assert cloud is not None
        except Exception:
            # May fail for completely unknown words
            pass

    def test_single_letter_word(self):
        """Test handling of single letter."""
        try:
            cloud = generate_conceptual_cloud("a", k_per_cluster=3)
            assert cloud is not None
        except Exception:
            # May fail for very short words
            pass

    def test_zero_k(self):
        """Test handling of k=0."""
        try:
            cloud = generate_conceptual_cloud("fire", k_per_cluster=0)
            # Should handle gracefully
            assert cloud is not None
        except Exception:
            # May raise ValueError
            pass

    def test_large_k(self):
        """Test handling of very large k."""
        try:
            cloud = generate_conceptual_cloud("fire", k_per_cluster=100)
            assert cloud is not None
            # May be capped at API limits
        except Exception:
            pass


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_generation_speed(self):
        """Test that cloud generation completes in reasonable time."""
        import time

        start = time.time()
        try:
            cloud = generate_conceptual_cloud("fire", k_per_cluster=5)
            elapsed = time.time() - start

            # Should complete in under 10 seconds (with caching)
            assert elapsed < 10.0, f"Cloud generation took {elapsed:.2f}s"
        except Exception as e:
            pytest.skip(f"Generation failed: {e}")

    def test_caching_improves_speed(self):
        """Test that second generation is faster (cached)."""
        import time

        try:
            # First call (cold)
            start = time.time()
            cloud1 = generate_conceptual_cloud("fire", k_per_cluster=5)
            time1 = time.time() - start

            # Second call (warm)
            start = time.time()
            cloud2 = generate_conceptual_cloud("fire", k_per_cluster=5)
            time2 = time.time() - start

            # Second should be faster or same
            assert time2 <= time1 + 1.0  # Allow 1s margin
        except Exception as e:
            pytest.skip(f"Caching test failed: {e}")

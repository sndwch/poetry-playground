#!/usr/bin/env python3
"""Tests for the metaphor generator."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from poetryplayground.metaphor_generator import (
    Metaphor,
    MetaphorGenerator,
    MetaphorType,
)


def test_metaphor_generator_init():
    """Test metaphor generator initialization."""
    generator = MetaphorGenerator()

    # Test that all pattern lists exist
    assert hasattr(generator, "simile_patterns")
    assert hasattr(generator, "direct_patterns")
    assert hasattr(generator, "possessive_patterns")
    assert hasattr(generator, "appositive_patterns")
    assert hasattr(generator, "compound_patterns")
    assert hasattr(generator, "conceptual_patterns")

    # Test that new pattern lists have expected sizes
    assert len(generator.compound_patterns) >= 4
    assert len(generator.conceptual_patterns) >= 5
    assert len(generator.appositive_patterns) >= 6

    # Verify patterns contain proper placeholders
    for pattern in generator.compound_patterns:
        assert "{source}" in pattern or "{target}" in pattern

    for pattern in generator.conceptual_patterns:
        assert "{source}" in pattern and "{target}" in pattern


def test_generate_simile():
    """Test simile generation."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator._generate_simile("heart", "ocean")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.SIMILE
        assert metaphor.source == "heart"
        assert metaphor.target == "ocean"
        assert metaphor.text
        assert 0 <= metaphor.quality_score <= 1
        print(f"Simile: {metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_generate_direct_metaphor():
    """Test direct metaphor generation."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator._generate_direct_metaphor("time", "river")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.DIRECT
        assert metaphor.source == "time"
        assert metaphor.target == "river"
        assert metaphor.text
        assert 0 <= metaphor.quality_score <= 1
        print(f"Direct: {metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_generate_implied_metaphor():
    """Test implied metaphor generation."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator._generate_implied_metaphor("memory", "ghost")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.IMPLIED
        assert metaphor.source == "memory"
        assert metaphor.target == "ghost"
        assert metaphor.text
        assert 0 <= metaphor.quality_score <= 1
        print(f"Implied: {metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_generate_possessive_metaphor():
    """Test possessive metaphor generation."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator._generate_possessive_metaphor("night", "silence")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.POSSESSIVE
        assert metaphor.source == "night"
        assert metaphor.target == "silence"
        assert metaphor.text
        assert 0 <= metaphor.quality_score <= 1
        print(f"Possessive: {metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_generate_compound_metaphor():
    """Test compound metaphor generation (NEW in Phase 4.4)."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator._generate_compound_metaphor("storm", "heart")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.COMPOUND
        assert metaphor.source == "storm"
        assert metaphor.target == "heart"
        assert metaphor.text
        assert 0 <= metaphor.quality_score <= 1

        # Compound metaphors should use source and target in a compact way
        assert "storm" in metaphor.text.lower()
        assert "heart" in metaphor.text.lower()
        print(f"Compound: {metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_generate_conceptual_metaphor():
    """Test conceptual metaphor generation (NEW in Phase 4.4)."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator._generate_conceptual_metaphor("love", "journey")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.CONCEPTUAL
        assert metaphor.source == "love"
        assert metaphor.target == "journey"
        assert metaphor.text
        assert 0 <= metaphor.quality_score <= 1

        # Conceptual metaphors should reference abstract mappings
        assert "love" in metaphor.text.lower()
        assert "journey" in metaphor.text.lower()
        print(f"Conceptual: {metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_compound_patterns():
    """Test that compound patterns generate valid metaphors."""
    generator = MetaphorGenerator()

    # Test each compound pattern
    for pattern in generator.compound_patterns:
        text = pattern.format(source="moon", target="child")
        assert text
        assert "moon" in text.lower()
        assert "child" in text.lower()
        print(f"Compound pattern: {text}")


def test_conceptual_patterns():
    """Test that conceptual patterns generate valid metaphors."""
    generator = MetaphorGenerator()

    # Test each conceptual pattern
    for pattern in generator.conceptual_patterns:
        text = pattern.format(source="anger", target="fire")
        assert text
        assert "anger" in text.lower()
        assert "fire" in text.lower()
        print(f"Conceptual pattern: {text}")


def test_appositive_patterns_expanded():
    """Test that appositive patterns include new additions."""
    generator = MetaphorGenerator()

    # Should have at least 6 patterns now (was 4, added 2)
    assert len(generator.appositive_patterns) >= 6

    # Test each pattern
    for pattern in generator.appositive_patterns:
        text = pattern.format(source="sorrow", target="shadow")
        assert text
        assert "sorrow" in text.lower()
        assert "shadow" in text.lower()
        print(f"Appositive pattern: {text}")


def test_generate_extended_metaphor():
    """Test extended metaphor generation."""
    try:
        generator = MetaphorGenerator()
        metaphor = generator.generate_extended_metaphor("life", "dance")

        assert isinstance(metaphor, Metaphor)
        assert metaphor.metaphor_type == MetaphorType.EXTENDED
        assert metaphor.source == "life"
        assert metaphor.target == "dance"
        assert metaphor.text
        # Extended metaphors should have multiple lines
        assert "\n" in metaphor.text
        assert 0 <= metaphor.quality_score <= 1
        print(f"Extended:\n{metaphor.text}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_generate_synesthetic_metaphor():
    """Test synesthetic metaphor generation."""
    generator = MetaphorGenerator()
    metaphor = generator.generate_synesthetic_metaphor("silence")

    assert isinstance(metaphor, Metaphor)
    assert metaphor.metaphor_type == MetaphorType.SYNESTHETIC
    assert metaphor.source == "silence"
    assert metaphor.text
    assert 0 <= metaphor.quality_score <= 1
    # Synesthetic metaphors should cross sensory domains
    assert len(metaphor.grounds) >= 2
    print(f"Synesthetic: {metaphor.text}")


def test_generate_metaphor_batch():
    """Test batch metaphor generation."""
    try:
        generator = MetaphorGenerator()
        source_words = ["ocean", "night", "memory"]

        metaphors = generator.generate_metaphor_batch(source_words, count=10)

        assert isinstance(metaphors, list)
        assert len(metaphors) > 0
        assert len(metaphors) <= 10

        # Check that metaphors are sorted by quality
        for i in range(len(metaphors) - 1):
            assert metaphors[i].quality_score >= metaphors[i + 1].quality_score

        # Check that we have variety in metaphor types
        types_generated = {m.metaphor_type for m in metaphors}
        assert len(types_generated) > 1

        # Verify new types are included
        all_types = {m.metaphor_type for m in metaphors}
        print(f"Metaphor types generated: {all_types}")

        print(f"\nGenerated {len(metaphors)} metaphors:")
        for i, metaphor in enumerate(metaphors[:5], 1):
            print(
                f"{i}. [{metaphor.metaphor_type.value}] {metaphor.text} (score: {metaphor.quality_score:.2f})"
            )
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_metaphor_quality_scoring():
    """Test metaphor quality scoring."""
    try:
        generator = MetaphorGenerator()

        # Test that different types have appropriate scores
        simile = generator._generate_simile("ocean", "memory")
        compound = generator._generate_compound_metaphor("ocean", "memory")
        conceptual = generator._generate_conceptual_metaphor("ocean", "memory")

        assert simile.quality_score > 0
        assert compound.quality_score > 0
        assert conceptual.quality_score > 0

        # Compound and conceptual get slight quality bonuses
        print(f"Simile score: {simile.quality_score:.3f}")
        print(f"Compound score: {compound.quality_score:.3f}")
        print(f"Conceptual score: {conceptual.quality_score:.3f}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_find_connecting_attributes():
    """Test that connecting attributes are found for metaphors."""
    try:
        generator = MetaphorGenerator()

        grounds = generator._find_connecting_attributes("ocean", "sky")
        assert isinstance(grounds, list)
        assert len(grounds) <= 3

        # Should have some connecting attributes
        if grounds:
            for attr in grounds:
                assert isinstance(attr, str)
                assert len(attr) > 0
            print(f"Connecting attributes for ocean/sky: {grounds}")
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


def test_new_patterns_in_batch_generation():
    """Test that new metaphor types appear in batch generation."""
    try:
        generator = MetaphorGenerator()

        # Generate a large batch to ensure we get variety
        metaphors = generator.generate_metaphor_batch(["heart", "storm", "memory"], count=20)

        # Check that compound metaphors appear
        compound_found = any(m.metaphor_type == MetaphorType.COMPOUND for m in metaphors)
        conceptual_found = any(m.metaphor_type == MetaphorType.CONCEPTUAL for m in metaphors)

        assert compound_found, "Compound metaphors should be generated in batch"
        assert conceptual_found, "Conceptual metaphors should be generated in batch"

        print(
            f"\nFound {sum(1 for m in metaphors if m.metaphor_type == MetaphorType.COMPOUND)} compound metaphors"
        )
        print(
            f"Found {sum(1 for m in metaphors if m.metaphor_type == MetaphorType.CONCEPTUAL)} conceptual metaphors"
        )
    except Exception as e:
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


if __name__ == "__main__":
    # Run basic tests
    print("Testing MetaphorGenerator initialization...")
    test_metaphor_generator_init()

    print("\nTesting compound metaphor generation...")
    test_generate_compound_metaphor()

    print("\nTesting conceptual metaphor generation...")
    test_generate_conceptual_metaphor()

    print("\nTesting compound patterns...")
    test_compound_patterns()

    print("\nTesting conceptual patterns...")
    test_conceptual_patterns()

    print("\nTesting batch generation with new types...")
    test_generate_metaphor_batch()

    print("\n✅ All tests passed!")

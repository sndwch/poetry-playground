#!/usr/bin/env python3
"""Tests for the line seeds generator."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generativepoetry.line_seeds import (
    LineSeedGenerator,
    LineSeed,
    SeedType,
    GenerationStrategy
)


def test_line_seed_generator():
    """Test basic line seed generation."""
    generator = LineSeedGenerator()
    seed_words = ['ocean', 'memory', 'light']

    # Test opening line generation
    opening = generator.generate_opening_line(seed_words)
    assert isinstance(opening, LineSeed)
    assert opening.seed_type == SeedType.OPENING
    assert opening.text
    assert 0 <= opening.quality_score <= 1
    assert 0 <= opening.momentum <= 1
    assert 0 <= opening.openness <= 1
    print(f"Opening line: {opening.text}")

    # Test fragment generation
    fragment = generator.generate_fragment(seed_words)
    assert isinstance(fragment, LineSeed)
    assert fragment.seed_type == SeedType.FRAGMENT
    assert fragment.text
    assert '...' in fragment.text
    print(f"Fragment: {fragment.text}")

    # Test image seed generation
    image = generator.generate_image_seed(seed_words)
    assert isinstance(image, LineSeed)
    assert image.seed_type == SeedType.IMAGE
    assert image.text
    print(f"Image seed: {image.text}")

    # Test pivot line generation
    pivot = generator.generate_pivot_line(seed_words)
    assert isinstance(pivot, LineSeed)
    assert pivot.seed_type == SeedType.PIVOT
    assert pivot.text
    print(f"Pivot line: {pivot.text}")

    # Test sonic pattern generation
    sonic = generator.generate_sonic_pattern(seed_words)
    assert isinstance(sonic, LineSeed)
    assert sonic.seed_type == SeedType.SONIC
    assert sonic.text
    print(f"Sonic pattern: {sonic.text}")

    # Test ending approach
    ending = generator.generate_ending_approach(seed_words, opening.text)
    assert isinstance(ending, LineSeed)
    assert ending.seed_type == SeedType.CLOSING
    assert ending.text
    print(f"Ending approach: {ending.text}")
    if ending.notes:
        print(f"  Notes: {ending.notes}")


def test_seed_collection():
    """Test generating a collection of seeds."""
    generator = LineSeedGenerator()
    seed_words = ['night', 'stars', 'silence']

    seeds = generator.generate_seed_collection(seed_words, num_seeds=10)
    assert len(seeds) <= 10
    assert all(isinstance(s, LineSeed) for s in seeds)

    # Check variety of types
    types = set(s.seed_type for s in seeds)
    assert len(types) >= 3  # Should have at least 3 different types

    print("\nSeed Collection:")
    print("-" * 40)
    for seed in seeds[:5]:  # Print first 5
        print(f"[{seed.seed_type.value}] {seed.text}")
        print(f"  Quality: {seed.quality_score:.2f}, Momentum: {seed.momentum:.2f}")


def test_different_moods():
    """Test generation with different seed words suggesting different moods."""
    generator = LineSeedGenerator()

    # Test with dark/melancholic words
    dark_words = ['shadow', 'grief', 'hollow']
    dark_opening = generator.generate_opening_line(dark_words)
    print(f"\nDark opening: {dark_opening.text}")

    # Test with bright/hopeful words
    bright_words = ['dawn', 'bloom', 'singing']
    bright_opening = generator.generate_opening_line(bright_words)
    print(f"Bright opening: {bright_opening.text}")

    # Test with abstract words
    abstract_words = ['time', 'thought', 'becoming']
    abstract_opening = generator.generate_opening_line(abstract_words)
    print(f"Abstract opening: {abstract_opening.text}")


def test_quality_evaluation():
    """Test that quality evaluation produces reasonable scores."""
    generator = LineSeedGenerator()

    # Generate multiple seeds and check quality distribution
    seed_words = ['rain', 'window', 'waiting']
    seeds = generator.generate_seed_collection(seed_words, num_seeds=20)

    qualities = [s.quality_score for s in seeds]
    avg_quality = sum(qualities) / len(qualities)

    print(f"\nQuality scores - Min: {min(qualities):.2f}, "
          f"Max: {max(qualities):.2f}, Avg: {avg_quality:.2f}")

    # Should have some variation in quality
    assert min(qualities) < max(qualities)
    # Average should be reasonable
    assert 0.3 < avg_quality < 0.8


def demonstrate_interactive_session():
    """Demonstrate how a poet might use this interactively."""
    generator = LineSeedGenerator()
    seed_words = ['metamorphosis', 'glass', 'breaking']

    print("\n" + "="*50)
    print("INTERACTIVE POETRY IDEATION SESSION")
    print(f"Seed words: {', '.join(seed_words)}")
    print("="*50 + "\n")

    # Generate opening
    opening = generator.generate_opening_line(seed_words)
    print(f"Start with this opening:\n  {opening.text}\n")

    # Generate some fragments to develop
    print("Develop with these fragments:")
    for _ in range(3):
        fragment = generator.generate_fragment(seed_words, position='middle')
        print(f"  {fragment.text}")

    # Add an image
    print("\nAdd this image somewhere:")
    image = generator.generate_image_seed(seed_words)
    print(f"  {image.text}")

    # Suggest a pivot
    print("\nIf you need to change direction:")
    pivot = generator.generate_pivot_line(seed_words)
    print(f"  {pivot.text}")

    # Offer ending options
    print("\nPossible endings:")
    for _ in range(2):
        ending = generator.generate_ending_approach(seed_words, opening.text)
        print(f"  {ending.text}")
        if ending.notes:
            print(f"    ({ending.notes})")


if __name__ == "__main__":
    print("Testing Line Seed Generator...")
    print("="*50)

    test_line_seed_generator()
    print("\n" + "="*50)

    test_seed_collection()
    print("\n" + "="*50)

    test_different_moods()
    print("\n" + "="*50)

    test_quality_evaluation()
    print("\n" + "="*50)

    demonstrate_interactive_session()

    print("\n✅ All tests passed!")
#!/usr/bin/env python3
"""Demo script for line seeds feature."""

# Suppress the pronouncing warning
from generativepoetry.line_seeds import LineSeedGenerator
from generativepoetry.pronouncing_patch import *


def demo():
    """Run a demo of the line seeds generator."""
    generator = LineSeedGenerator()

    # Example seed words
    seed_words = ["ocean", "memory", "dissolve"]

    print("\n" + "=" * 60)
    print("POETRY LINE SEEDS DEMO")
    print(f"Generating seeds from: {', '.join(seed_words)}")
    print("=" * 60 + "\n")

    # Generate and display seeds
    seeds = generator.generate_seed_collection(seed_words, num_seeds=15)

    # Group by type
    from generativepoetry.line_seeds import SeedType

    by_type = {}
    for seed in seeds:
        if seed.seed_type not in by_type:
            by_type[seed.seed_type] = []
        by_type[seed.seed_type].append(seed)

    # Display organized results
    if SeedType.OPENING in by_type:
        print("OPENING LINES (to start your poem):")
        print("-" * 40)
        for seed in by_type[SeedType.OPENING][:3]:
            print(f"  {seed.text}")
        print()

    if SeedType.FRAGMENT in by_type or SeedType.PIVOT in by_type:
        print("FRAGMENTS & PIVOTS (to develop ideas):")
        print("-" * 40)
        fragments = by_type.get(SeedType.FRAGMENT, []) + by_type.get(SeedType.PIVOT, [])
        for seed in fragments[:4]:
            print(f"  {seed.text}")
        print()

    if SeedType.IMAGE in by_type:
        print("IMAGES (concrete sensory details):")
        print("-" * 40)
        for seed in by_type[SeedType.IMAGE][:3]:
            print(f"  {seed.text}")
        print()

    if SeedType.SONIC in by_type:
        print("SOUND PATTERNS (for rhythm and music):")
        print("-" * 40)
        for seed in by_type[SeedType.SONIC][:2]:
            print(f"  {seed.text}")
        print()

    if SeedType.CLOSING in by_type:
        print("ENDINGS (ways to conclude):")
        print("-" * 40)
        for seed in by_type[SeedType.CLOSING][:2]:
            print(f"  {seed.text}")
            if seed.notes:
                print(f"    → {seed.notes}")
        print()

    print("=" * 60)
    print("💡 Use these seeds as starting points for your own poetry!")
    print("=" * 60)


if __name__ == "__main__":
    demo()

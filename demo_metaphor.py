#!/usr/bin/env python3
"""Demo script for the metaphor generator."""

# Suppress warnings
from generativepoetry.pronouncing_patch import *
from generativepoetry.metaphor_generator import MetaphorGenerator, MetaphorType

def demo():
    """Run a demo of the metaphor generator."""
    generator = MetaphorGenerator()

    # Example seed words
    seed_words = ['memory', 'time', 'silence']

    print("\n" + "="*70)
    print("METAPHOR GENERATOR DEMO")
    print(f"Creating metaphors for: {', '.join(seed_words)}")
    print("="*70 + "\n")

    # Generate batch of metaphors
    metaphors = generator.generate_metaphor_batch(seed_words, count=20)

    # Group by type for display
    by_type = {}
    for metaphor in metaphors:
        if metaphor.metaphor_type not in by_type:
            by_type[metaphor.metaphor_type] = []
        by_type[metaphor.metaphor_type].append(metaphor)

    # Display different types
    if MetaphorType.SIMILE in by_type:
        print("SIMILES (Comparisons using 'like' or 'as'):")
        print("-" * 50)
        for m in by_type[MetaphorType.SIMILE][:5]:
            print(f"  • {m.text}")
            if m.grounds:
                print(f"    → Connecting: {', '.join(m.grounds[:2])}")
        print()

    if MetaphorType.DIRECT in by_type:
        print("DIRECT METAPHORS (X is Y):")
        print("-" * 50)
        for m in by_type[MetaphorType.DIRECT][:5]:
            print(f"  • {m.text}")
        print()

    if MetaphorType.IMPLIED in by_type:
        print("IMPLIED METAPHORS (Using associated verbs):")
        print("-" * 50)
        for m in by_type[MetaphorType.IMPLIED][:5]:
            print(f"  • {m.text}")
        print()

    if MetaphorType.POSSESSIVE in by_type:
        print("POSSESSIVE METAPHORS (Y's X):")
        print("-" * 50)
        for m in by_type[MetaphorType.POSSESSIVE][:4]:
            print(f"  • {m.text}")
        print()

    # Generate an extended metaphor
    print("EXTENDED METAPHOR (Multi-line development):")
    print("-" * 50)
    targets = generator._find_target_domains('memory')
    if targets:
        extended = generator.generate_extended_metaphor('memory', targets[0])
        for line in extended.text.split('\n'):
            print(f"  {line}")
    print()

    # Generate synesthetic metaphors
    print("SYNESTHETIC METAPHORS (Cross-sensory):")
    print("-" * 50)
    for word in seed_words:
        synesthetic = generator.generate_synesthetic_metaphor(word)
        if synesthetic:
            print(f"  • {synesthetic.text}")
    print()

    # Try to extract patterns from Gutenberg
    print("MINING CLASSIC LITERATURE:")
    print("-" * 50)
    print("Extracting metaphorical patterns from Project Gutenberg...")
    patterns = generator.extract_metaphor_patterns()
    if patterns:
        print(f"Found {len(patterns)} metaphorical patterns")
        for source, target, context in patterns[:3]:
            print(f"  • \"{source}\" compared to \"{target}\"")
            if len(context) > 100:
                context = context[:100] + "..."
            print(f"    Context: {context}")
    else:
        print("  (Gutenberg access may be limited)")
    print()

    # Show quality scores
    print("TOP METAPHORS BY QUALITY:")
    print("-" * 50)
    top_metaphors = sorted(metaphors, key=lambda m: m.quality_score, reverse=True)[:5]
    for i, m in enumerate(top_metaphors, 1):
        print(f"  {i}. {m.text}")
        print(f"     Quality: {m.quality_score:.2f}, Type: {m.metaphor_type.value}")

    print("\n" + "="*70)
    print("💡 Use these metaphors as inspiration for your poetry!")
    print("="*70)


if __name__ == "__main__":
    demo()
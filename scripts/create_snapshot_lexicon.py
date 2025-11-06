#!/usr/bin/env python3
"""Creates a deterministic snapshot lexicon for testing.

This script generates a fixed word list from wordfreq's top 10,000 English words.
The snapshot is used in tests to ensure deterministic, reproducible results across
test runs and CI environments.

Usage:
    python scripts/create_snapshot_lexicon.py
"""

from pathlib import Path

from wordfreq import top_n_list


def main():
    """Generate snapshot lexicon and write to test data directory."""
    # Get top 10k words (balances coverage with performance)
    print("Fetching top 10,000 English words from wordfreq...")
    words = top_n_list("en", 10000)

    # Determine output path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "tests" / "data" / "snapshot_lexicon.txt"

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write words to file
    print(f"Writing {len(words)} words to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))

    print(f"✓ Successfully created snapshot lexicon with {len(words)} words")
    print(f"  Location: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    print("\nThis file should be committed to git for deterministic testing.")


if __name__ == "__main__":
    main()

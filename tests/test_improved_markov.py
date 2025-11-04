#!/usr/bin/env python3
"""Test improved Markov poem generation."""

import sys

sys.stderr = open("/dev/null", "w")  # Suppress warnings

import pytest
from generativepoetry.poemgen import PoemGenerator


def test_markov():
    try:
        pg = PoemGenerator()
        poem = pg.poem_from_markov(["future", "hope"], num_lines=6, min_line_words=4, max_line_words=7)

        print("\nGenerated Poem:")
        print("-" * 40)
        if hasattr(poem, "lines"):
            for line in poem.lines:
                print(line)
        else:
            print(poem)
        print("-" * 40)
    except Exception as e:
        # Skip if network calls are disabled in tests
        if "Network calls disabled" in str(e):
            pytest.skip("Network calls disabled in test environment")
        raise


if __name__ == "__main__":
    test_markov()

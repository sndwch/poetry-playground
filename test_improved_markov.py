#!/usr/bin/env python3
"""Test improved Markov poem generation."""

import sys
sys.stderr = open('/dev/null', 'w')  # Suppress warnings

from generativepoetry.poemgen import PoemGenerator

def test_markov():
    pg = PoemGenerator()
    poem = pg.poem_from_markov(['future', 'hope'], num_lines=6, min_line_words=4, max_line_words=7)

    print("\nGenerated Poem:")
    print("-" * 40)
    if hasattr(poem, 'lines'):
        for line in poem.lines:
            print(line)
    else:
        print(poem)
    print("-" * 40)

if __name__ == "__main__":
    test_markov()
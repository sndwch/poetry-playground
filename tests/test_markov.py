#!/usr/bin/env python3
"""Test Markov poem generation directly."""

from poetryplayground.poemgen import PoemGenerator


def test_markov():
    pg = PoemGenerator()
    try:
        poem = pg.poem_from_markov(
            ["test", "word"], num_lines=3, min_line_words=3, max_line_words=5
        )
        print("Generated poem:")
        print(poem)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_markov()

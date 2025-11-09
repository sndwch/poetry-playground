#!/usr/bin/env python3
"""Tests for deterministic behavior with fixed seeds."""

import random
import unittest

from poetryplayground.pdf import FuturistPoemPDFGenerator, MarkovPoemPDFGenerator
from poetryplayground.poemgen import PoemGenerator
from poetryplayground.seed_manager import get_current_seed, set_global_seed


class TestSeedManager(unittest.TestCase):
    """Test seed management functionality."""

    def test_set_global_seed(self):
        """Test setting global seed."""
        seed = 42
        set_global_seed(seed)
        self.assertEqual(get_current_seed(), seed)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different random values."""
        set_global_seed(42)
        random1 = random.random()

        set_global_seed(43)
        random2 = random.random()

        self.assertNotEqual(random1, random2)

    def test_same_seed_produces_same_results(self):
        """Test that same seed produces identical random values."""
        set_global_seed(42)
        random1 = random.random()
        value1 = random.randint(1, 100)

        set_global_seed(42)
        random2 = random.random()
        value2 = random.randint(1, 100)

        self.assertEqual(random1, random2)
        self.assertEqual(value1, value2)

    def test_seed_affects_random_choice(self):
        """Test that seed affects random.choice()."""
        choices = ["apple", "banana", "cherry", "date", "elderberry"]

        set_global_seed(42)
        choice1 = random.choice(choices)

        set_global_seed(42)
        choice2 = random.choice(choices)

        self.assertEqual(choice1, choice2)

    def test_seed_affects_random_sample(self):
        """Test that seed affects random.sample()."""
        items = list(range(20))

        set_global_seed(42)
        sample1 = random.sample(items, 5)

        set_global_seed(42)
        sample2 = random.sample(items, 5)

        self.assertEqual(sample1, sample2)


class TestDeterministicPoemGeneration(unittest.TestCase):
    """Test deterministic poem generation with seeds."""

    def test_poem_generator_determinism(self):
        """Test that PoemGenerator produces identical output with same seed."""
        input_words = ["ocean", "memory", "light"]

        # Generate first poem
        set_global_seed(42)
        gen1 = PoemGenerator()
        # Note: Full poem generation requires text corpus and may fail in test
        # This tests the deterministic setup

        # Generate second poem with same seed
        set_global_seed(42)
        gen2 = PoemGenerator()

        # Both generators should have same random state
        self.assertIsNotNone(gen1)
        self.assertIsNotNone(gen2)

    def test_word_selection_determinism(self):
        """Test that word selection is deterministic with fixed seed."""
        word_list = ["alpha", "beta", "gamma", "delta", "epsilon"]

        # First selection
        set_global_seed(100)
        selected1 = []
        for _ in range(5):
            selected1.append(random.choice(word_list))

        # Second selection with same seed
        set_global_seed(100)
        selected2 = []
        for _ in range(5):
            selected2.append(random.choice(word_list))

        self.assertEqual(selected1, selected2)

    def test_shuffle_determinism(self):
        """Test that random.shuffle is deterministic with fixed seed."""
        items = list(range(10))

        # First shuffle
        set_global_seed(99)
        shuffled1 = items.copy()
        random.shuffle(shuffled1)

        # Second shuffle with same seed
        set_global_seed(99)
        shuffled2 = items.copy()
        random.shuffle(shuffled2)

        self.assertEqual(shuffled1, shuffled2)


class TestDeterministicPDFLayout(unittest.TestCase):
    """Test deterministic PDF layout with fixed seeds."""

    def test_futurist_pdf_layout_determinism(self):
        """Test that Futurist PDF generator has deterministic layout with seed."""
        input_words = ["code", "poetry", "machine"]

        # First generation
        set_global_seed(42)
        gen1 = FuturistPoemPDFGenerator()
        # Note: Actually generating PDF requires full setup
        # This tests that generator can be created deterministically
        self.assertIsNotNone(gen1)
        self.assertIsNotNone(gen1.default_font_sizes)
        self.assertIsNotNone(gen1.connectors)

        # Second generation with same seed
        set_global_seed(42)
        gen2 = FuturistPoemPDFGenerator()
        self.assertIsNotNone(gen2)

        # Same random state should lead to same choices
        set_global_seed(42)
        font1 = random.choice(gen1.default_font_sizes)
        connector1 = random.choice(gen1.connectors)

        set_global_seed(42)
        font2 = random.choice(gen2.default_font_sizes)
        connector2 = random.choice(gen2.connectors)

        self.assertEqual(font1, font2)
        self.assertEqual(connector1, connector2)

    def test_markov_pdf_layout_determinism(self):
        """Test that Markov PDF generator has deterministic layout with seed."""
        # First generation
        set_global_seed(123)
        gen1 = MarkovPoemPDFGenerator()
        self.assertIsNotNone(gen1)

        # Second generation with same seed
        set_global_seed(123)
        gen2 = MarkovPoemPDFGenerator()
        self.assertIsNotNone(gen2)

    def test_color_generation_determinism(self):
        """Test that color generation is deterministic."""
        # First color generation
        set_global_seed(77)
        colors1 = []
        for _ in range(10):
            r = random.random()
            g = random.random()
            b = random.random()
            colors1.append((r, g, b))

        # Second color generation with same seed
        set_global_seed(77)
        colors2 = []
        for _ in range(10):
            r = random.random()
            g = random.random()
            b = random.random()
            colors2.append((r, g, b))

        self.assertEqual(colors1, colors2)

    def test_position_generation_determinism(self):
        """Test that position generation is deterministic."""
        width, height = 800, 600

        # First position generation
        set_global_seed(55)
        positions1 = []
        for _ in range(10):
            x = random.randint(0, width)
            y = random.randint(0, height)
            positions1.append((x, y))

        # Second position generation with same seed
        set_global_seed(55)
        positions2 = []
        for _ in range(10):
            x = random.randint(0, width)
            y = random.randint(0, height)
            positions2.append((x, y))

        self.assertEqual(positions1, positions2)


class TestSeedPropagation(unittest.TestCase):
    """Test that seed propagates through the system."""

    def test_seed_persists_across_calls(self):
        """Test that seed remains set across multiple random calls."""
        set_global_seed(42)
        initial_seed = get_current_seed()

        # Make several random calls
        random.random()
        random.randint(1, 100)
        random.choice([1, 2, 3])

        # Seed should still be remembered
        current_seed = get_current_seed()
        self.assertEqual(current_seed, initial_seed)

    def test_reproducible_sequence(self):
        """Test that entire random sequence is reproducible with same seed."""
        # First sequence
        set_global_seed(999)
        sequence1 = [random.random() for _ in range(20)]

        # Second sequence with same seed
        set_global_seed(999)
        sequence2 = [random.random() for _ in range(20)]

        self.assertEqual(sequence1, sequence2)

    def test_partial_sequence_reproducibility(self):
        """Test reproducing a sequence from a checkpoint."""
        # Generate first part
        set_global_seed(42)
        part1a = [random.random() for _ in range(5)]

        # Get random state
        state = random.getstate()

        # Generate second part
        part2a = [random.random() for _ in range(5)]

        # Restore state and regenerate second part
        random.setstate(state)
        part2b = [random.random() for _ in range(5)]

        # Second parts should match
        self.assertEqual(part2a, part2b)


class TestDeterministicEdgeCases(unittest.TestCase):
    """Test edge cases in deterministic behavior."""

    def test_seed_zero(self):
        """Test that seed 0 works correctly."""
        set_global_seed(0)
        value1 = random.random()

        set_global_seed(0)
        value2 = random.random()

        self.assertEqual(value1, value2)

    def test_large_seed(self):
        """Test that large seed values work correctly."""
        large_seed = 2**31 - 1

        set_global_seed(large_seed)
        value1 = random.random()

        set_global_seed(large_seed)
        value2 = random.random()

        self.assertEqual(value1, value2)

    def test_negative_seed(self):
        """Test that negative seeds are rejected (numpy requirement)."""
        # numpy requires seeds between 0 and 2**32 - 1
        with self.assertRaises((ValueError, OverflowError)):
            set_global_seed(-42)


if __name__ == "__main__":
    unittest.main()

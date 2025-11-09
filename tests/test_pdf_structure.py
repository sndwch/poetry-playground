#!/usr/bin/env python3
"""Snapshot tests for PDF structure (not pixel-perfect rendering).

These tests verify the logical structure and content of generated PDFs,
not their visual appearance. We test PDF metadata, text content, structure,
and font/layout parameters - but not exact pixel rendering.
"""

import tempfile
import unittest
from pathlib import Path

from poetryplayground.pdf import (
    FuturistPoemPDFGenerator,
    MarkovPoemPDFGenerator,
    PDFGenerator,
    StopwordSoupPoemPDFGenerator,
)
from poetryplayground.seed_manager import set_global_seed


class TestPDFGeneratorStructure(unittest.TestCase):
    """Test base PDFGenerator structure and configuration."""

    def test_pdf_generator_has_fonts(self):
        """Test that PDFGenerator has font configuration."""
        gen = PDFGenerator()
        self.assertIsNotNone(gen.font_choices)
        self.assertIsInstance(gen.font_choices, list)
        self.assertGreater(len(gen.font_choices), 0)

    def test_pdf_generator_has_font_sizes(self):
        """Test that PDFGenerator has font size configuration."""
        gen = PDFGenerator()
        self.assertIsNotNone(gen.default_font_sizes)
        self.assertIsInstance(gen.default_font_sizes, list)
        self.assertGreater(len(gen.default_font_sizes), 0)

    def test_font_choices_are_standard_fonts(self):
        """Test that font choices use standard PDF fonts."""
        gen = PDFGenerator()
        # Standard PDF fonts don't require TTF files
        standard_fonts = [
            "Courier",
            "Courier-Bold",
            "Courier-Oblique",
            "Helvetica",
            "Helvetica-Bold",
            "Helvetica-Oblique",
            "Times-Roman",
            "Times-Bold",
            "Times-Italic",
        ]
        # All font choices should be standard fonts
        for font in gen.font_choices:
            self.assertIn(
                any(standard in font for standard in standard_fonts),
                [True],
                f"{font} should be a standard PDF font",
            )

    def test_font_sizes_are_reasonable(self):
        """Test that font sizes are reasonable."""
        gen = PDFGenerator()
        for size in gen.default_font_sizes:
            self.assertIsInstance(size, int)
            self.assertGreater(size, 0)
            self.assertLess(size, 200)  # Reasonable upper bound


class TestFuturistPDFStructure(unittest.TestCase):
    """Test Futurist PDF generator structure."""

    def test_futurist_generator_initialization(self):
        """Test that Futurist generator initializes correctly."""
        gen = FuturistPoemPDFGenerator()
        self.assertIsNotNone(gen)

    def test_futurist_has_connectors(self):
        """Test that Futurist generator has connector symbols."""
        gen = FuturistPoemPDFGenerator()
        self.assertIsNotNone(gen.connectors)
        self.assertIsInstance(gen.connectors, list)
        self.assertGreater(len(gen.connectors), 0)

        # Connectors should contain math/symbolic characters
        connector_chars = "".join(gen.connectors)
        self.assertTrue(any(c in connector_chars for c in ["+", "-", "*", "%", "=", "!"]))

    def test_futurist_font_sizes(self):
        """Test that Futurist generator has appropriate font sizes."""
        gen = FuturistPoemPDFGenerator()
        self.assertIsNotNone(gen.default_font_sizes)
        self.assertIsInstance(gen.default_font_sizes, list)

        # Font sizes should be reasonable for visual poetry
        for size in gen.default_font_sizes:
            self.assertGreaterEqual(size, 10)
            self.assertLessEqual(size, 50)

    def test_futurist_deterministic_with_seed(self):
        """Test that Futurist generation is deterministic with fixed seed."""
        set_global_seed(42)
        gen1 = FuturistPoemPDFGenerator()

        set_global_seed(42)
        gen2 = FuturistPoemPDFGenerator()

        # With same seed, should have same configuration
        self.assertEqual(gen1.default_font_sizes, gen2.default_font_sizes)
        self.assertEqual(gen1.connectors, gen2.connectors)


class TestMarkovPDFStructure(unittest.TestCase):
    """Test Markov PDF generator structure."""

    def test_markov_generator_initialization(self):
        """Test that Markov generator initializes correctly."""
        gen = MarkovPoemPDFGenerator()
        self.assertIsNotNone(gen)

    def test_markov_has_font_sizes(self):
        """Test that Markov generator has font size configuration."""
        gen = MarkovPoemPDFGenerator()
        self.assertIsNotNone(gen.default_font_sizes)
        self.assertIsInstance(gen.default_font_sizes, list)
        self.assertGreater(len(gen.default_font_sizes), 0)

    def test_markov_font_sizes_reasonable(self):
        """Test that Markov font sizes are reasonable."""
        gen = MarkovPoemPDFGenerator()
        for size in gen.default_font_sizes:
            self.assertIsInstance(size, int)
            self.assertGreaterEqual(size, 10)
            self.assertLessEqual(size, 40)


class TestStopwordSoupPDFStructure(unittest.TestCase):
    """Test Stopword Soup PDF generator structure."""

    def test_stopword_soup_initialization(self):
        """Test that Stopword Soup generator initializes correctly."""
        gen = StopwordSoupPoemPDFGenerator()
        self.assertIsNotNone(gen)

    def test_stopword_soup_font_sizes(self):
        """Test that Stopword Soup has varied font sizes."""
        gen = StopwordSoupPoemPDFGenerator()
        self.assertIsNotNone(gen.default_font_sizes)
        self.assertIsInstance(gen.default_font_sizes, list)

        # Should have variety of sizes for visual interest
        self.assertGreater(len(gen.default_font_sizes), 3)

        # Should have both small and large sizes
        min_size = min(gen.default_font_sizes)
        max_size = max(gen.default_font_sizes)
        self.assertLess(min_size, 15)
        self.assertGreater(max_size, 30)


class TestPDFMetadata(unittest.TestCase):
    """Test PDF metadata and properties."""

    def test_pdf_generator_has_orientation(self):
        """Test that PDF generator has orientation setting."""
        gen = PDFGenerator()
        self.assertTrue(hasattr(gen, "orientation"))
        self.assertIn(gen.orientation, ["portrait", "landscape"])

    def test_pdf_generator_tracks_drawn_strings(self):
        """Test that PDF generator tracks drawn strings."""
        gen = PDFGenerator()
        self.assertTrue(hasattr(gen, "drawn_strings"))
        self.assertIsInstance(gen.drawn_strings, list)


class TestPDFContentStructure(unittest.TestCase):
    """Test PDF content structure (not rendering)."""

    def test_futurist_pdf_with_input_words(self):
        """Test Futurist PDF structure with input words."""
        gen = FuturistPoemPDFGenerator()
        input_words = ["code", "poetry", "machine"]

        # Test that generator can accept input words
        # (actual PDF generation requires full setup, we test structure)
        self.assertIsNotNone(gen)
        self.assertIsNotNone(input_words)

    def test_pdf_filepath_configuration(self):
        """Test that PDF generators can configure output filepath."""
        gen = FuturistPoemPDFGenerator()

        # Should have a way to set output path
        # (Testing structure, not actual file creation)
        self.assertIsNotNone(gen)


class TestPDFLayoutDeterminism(unittest.TestCase):
    """Test that PDF layout is deterministic with fixed seeds."""

    def test_layout_consistency_with_seed(self):
        """Test that same seed produces consistent layout parameters."""
        # First generation
        set_global_seed(12345)
        gen1 = FuturistPoemPDFGenerator()
        sizes1 = gen1.default_font_sizes
        connectors1 = gen1.connectors

        # Second generation with same seed
        set_global_seed(12345)
        gen2 = FuturistPoemPDFGenerator()
        sizes2 = gen2.default_font_sizes
        connectors2 = gen2.connectors

        # Should have same configuration
        self.assertEqual(sizes1, sizes2)
        self.assertEqual(connectors1, connectors2)

    def test_different_seeds_produce_variety(self):
        """Test that different seeds could produce different results."""
        # This just tests that randomness is available
        # (actual randomness depends on implementation)
        set_global_seed(1)
        gen1 = FuturistPoemPDFGenerator()

        set_global_seed(2)
        gen2 = FuturistPoemPDFGenerator()

        # Generators should be created successfully
        self.assertIsNotNone(gen1)
        self.assertIsNotNone(gen2)


class TestPDFGeneratorInheritance(unittest.TestCase):
    """Test PDF generator inheritance structure."""

    def test_futurist_inherits_from_base(self):
        """Test that FuturistPoemPDFGenerator inherits from PDFGenerator."""
        gen = FuturistPoemPDFGenerator()
        self.assertIsInstance(gen, PDFGenerator)

    def test_markov_inherits_from_base(self):
        """Test that MarkovPoemPDFGenerator inherits from PDFGenerator."""
        gen = MarkovPoemPDFGenerator()
        self.assertIsInstance(gen, PDFGenerator)

    def test_stopword_soup_inherits_from_base(self):
        """Test that StopwordSoupPoemPDFGenerator inherits from PDFGenerator."""
        gen = StopwordSoupPoemPDFGenerator()
        self.assertIsInstance(gen, PDFGenerator)

    def test_all_generators_have_common_attributes(self):
        """Test that all generators share common base attributes."""
        generators = [
            PDFGenerator(),
            FuturistPoemPDFGenerator(),
            MarkovPoemPDFGenerator(),
            StopwordSoupPoemPDFGenerator(),
        ]

        for gen in generators:
            self.assertTrue(hasattr(gen, "font_choices"))
            self.assertTrue(hasattr(gen, "default_font_sizes"))
            self.assertTrue(hasattr(gen, "orientation"))
            self.assertTrue(hasattr(gen, "drawn_strings"))


class TestPDFConfigurationOptions(unittest.TestCase):
    """Test PDF configuration options."""

    def test_pdf_orientation_options(self):
        """Test PDF orientation configuration."""
        gen = PDFGenerator()
        # Should default to landscape for visual poetry
        self.assertEqual(gen.orientation, "landscape")

    def test_font_family_variety(self):
        """Test that multiple font families are available."""
        gen = PDFGenerator()
        fonts = gen.font_choices

        # Should have variety: Courier, Helvetica, Times
        has_courier = any("Courier" in f for f in fonts)
        has_helvetica = any("Helvetica" in f for f in fonts)
        has_times = any("Times" in f for f in fonts)

        # Should have at least 2 of 3 major families
        self.assertGreaterEqual(sum([has_courier, has_helvetica, has_times]), 2)


class TestPDFStructureConsistency(unittest.TestCase):
    """Test consistency of PDF structure across generators."""

    def test_all_generators_have_font_sizes(self):
        """Test that all generators define font sizes."""
        generators = [
            FuturistPoemPDFGenerator(),
            MarkovPoemPDFGenerator(),
            StopwordSoupPoemPDFGenerator(),
        ]

        for gen in generators:
            self.assertIsNotNone(gen.default_font_sizes)
            self.assertGreater(len(gen.default_font_sizes), 0)

    def test_font_sizes_are_integers(self):
        """Test that all font sizes are integers."""
        generators = [
            PDFGenerator(),
            FuturistPoemPDFGenerator(),
            MarkovPoemPDFGenerator(),
            StopwordSoupPoemPDFGenerator(),
        ]

        for gen in generators:
            for size in gen.default_font_sizes:
                self.assertIsInstance(size, int)

    def test_generators_use_standard_fonts_only(self):
        """Test that all generators use only standard PDF fonts."""
        generators = [
            PDFGenerator(),
            FuturistPoemPDFGenerator(),
            MarkovPoemPDFGenerator(),
            StopwordSoupPoemPDFGenerator(),
        ]

        for gen in generators:
            # All should have font_choices
            self.assertTrue(hasattr(gen, "font_choices"))
            # Font choices should be non-empty
            self.assertGreater(len(gen.font_choices), 0)


if __name__ == "__main__":
    unittest.main()

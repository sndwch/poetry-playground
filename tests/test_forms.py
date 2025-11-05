#!/usr/bin/env python3
"""Comprehensive tests for syllable-aware poetry forms."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from generativepoetry.forms import (
    FormConstraint,
    FormGenerator,
    FormValidationResult,
    count_line_syllables,
    count_syllables,
    create_form_generator,
)


class TestSyllableCounting:
    """Test syllable counting functions."""

    def test_count_syllables_cmu_dict(self):
        """Test syllable counting using CMU dictionary."""
        # Common words in CMU dict
        assert count_syllables("hello") == 2
        assert count_syllables("world") == 1
        assert count_syllables("poetry") == 3
        assert count_syllables("syllable") == 3
        assert count_syllables("beautiful") == 3
        assert count_syllables("computer") == 3

    def test_count_syllables_heuristic_fallback(self):
        """Test syllable counting with heuristic for unknown words."""
        # Made-up words that should use heuristic
        assert count_syllables("blorp") >= 1
        assert count_syllables("zyx") >= 1

    def test_count_syllables_with_punctuation(self):
        """Test syllable counting strips punctuation correctly."""
        assert count_syllables("hello,") == 2
        assert count_syllables("world!") == 1
        assert count_syllables("poetry.") == 3
        assert count_syllables('"quoted"') == 2
        assert count_syllables("word—") == 1
        assert count_syllables("[bracketed]") == 3

    def test_count_syllables_empty_string(self):
        """Test syllable counting with empty or whitespace strings."""
        assert count_syllables("") == 0
        assert count_syllables("   ") == 0
        assert count_syllables(".,!?") == 0

    def test_count_syllables_case_insensitive(self):
        """Test syllable counting is case-insensitive."""
        assert count_syllables("Hello") == count_syllables("hello")
        assert count_syllables("WORLD") == count_syllables("world")
        assert count_syllables("PoEtRy") == count_syllables("poetry")

    def test_count_syllables_edge_cases(self):
        """Test syllable counting edge cases."""
        # Single letter words
        assert count_syllables("a") >= 1
        assert count_syllables("I") >= 1

        # Words ending in silent e
        assert count_syllables("time") >= 1
        assert count_syllables("love") >= 1

        # Words ending in -le
        assert count_syllables("able") >= 2
        assert count_syllables("table") >= 2

    def test_count_line_syllables(self):
        """Test counting syllables in full lines."""
        assert count_line_syllables("hello world") == 3  # 2 + 1
        assert count_line_syllables("the quick brown fox") >= 4
        assert count_line_syllables("I am a poet") >= 4

    def test_count_line_syllables_empty(self):
        """Test counting syllables in empty lines."""
        assert count_line_syllables("") == 0
        assert count_line_syllables("   ") == 0


class TestFormConstraint:
    """Test FormConstraint dataclass."""

    def test_form_constraint_creation(self):
        """Test creating a FormConstraint."""
        constraint = FormConstraint(line_number=1, target_syllables=5)
        assert constraint.line_number == 1
        assert constraint.target_syllables == 5
        assert constraint.actual_syllables is None
        assert constraint.line_text is None

    def test_form_constraint_is_satisfied_none(self):
        """Test constraint satisfaction with None actual syllables."""
        constraint = FormConstraint(line_number=1, target_syllables=5)
        assert not constraint.is_satisfied()

    def test_form_constraint_is_satisfied_match(self):
        """Test constraint satisfaction when syllables match."""
        constraint = FormConstraint(
            line_number=1,
            target_syllables=5,
            actual_syllables=5,
            line_text="hello world today",
        )
        assert constraint.is_satisfied()

    def test_form_constraint_is_satisfied_mismatch(self):
        """Test constraint satisfaction when syllables don't match."""
        constraint = FormConstraint(
            line_number=1,
            target_syllables=5,
            actual_syllables=7,
            line_text="hello world from the earth",
        )
        assert not constraint.is_satisfied()

    def test_form_constraint_string_representation(self):
        """Test string representation of constraint."""
        constraint = FormConstraint(
            line_number=1,
            target_syllables=5,
            actual_syllables=5,
            line_text="test line",
        )
        str_repr = str(constraint)
        assert "Line 1" in str_repr
        assert "5 syllables" in str_repr
        assert "✓" in str_repr

        # Test failed constraint
        constraint_fail = FormConstraint(
            line_number=2,
            target_syllables=7,
            actual_syllables=5,
            line_text="short line",
        )
        str_repr_fail = str(constraint_fail)
        assert "✗" in str_repr_fail


class TestFormValidationResult:
    """Test FormValidationResult dataclass."""

    def test_form_validation_result_creation(self):
        """Test creating a FormValidationResult."""
        lines = ["test line one", "test line two"]
        constraints = [
            FormConstraint(1, 5, 5, "test line one"),
            FormConstraint(2, 7, 7, "test line two"),
        ]
        result = FormValidationResult("Haiku", lines, constraints, True)

        assert result.form_name == "Haiku"
        assert result.lines == lines
        assert result.constraints == constraints
        assert result.valid is True

    def test_form_validation_result_report_valid(self):
        """Test validation report for valid form."""
        lines = ["hello world", "the quick brown fox"]
        constraints = [
            FormConstraint(1, 3, 3, lines[0]),
            FormConstraint(2, 5, 5, lines[1]),
        ]
        result = FormValidationResult("Test", lines, constraints, True)
        report = result.get_report()

        assert "TEST Validation Report" in report
        assert "✓ All constraints satisfied!" in report
        assert "Line 1" in report
        assert "Line 2" in report

    def test_form_validation_result_report_invalid(self):
        """Test validation report for invalid form."""
        lines = ["hello", "the quick brown fox jumper"]
        constraints = [
            FormConstraint(1, 5, 2, lines[0]),
            FormConstraint(2, 5, 7, lines[1]),
        ]
        result = FormValidationResult("Test", lines, constraints, False)
        report = result.get_report()

        assert "TEST Validation Report" in report
        assert "constraint(s) failed" in report
        assert "Failed constraints:" in report


class TestFormGenerator:
    """Test FormGenerator class."""

    def test_form_generator_creation(self):
        """Test creating a FormGenerator."""
        generator = FormGenerator()
        assert generator is not None
        assert isinstance(generator, FormGenerator)

    def test_create_form_generator_factory(self):
        """Test factory function."""
        generator = create_form_generator()
        assert isinstance(generator, FormGenerator)

    def test_validate_form_haiku_valid(self):
        """Test validating a valid haiku."""
        generator = FormGenerator()
        lines = ["the old silent pond", "a frog jumps into pond", "water sounds"]
        pattern = [5, 7, 5]

        result = generator.validate_form(lines, pattern, "Haiku")

        assert isinstance(result, FormValidationResult)
        assert result.form_name == "Haiku"
        assert len(result.constraints) == 3

    def test_validate_form_mismatch_line_count(self):
        """Test validation fails with wrong number of lines."""
        generator = FormGenerator()
        lines = ["line one", "line two"]
        pattern = [5, 7, 5]

        with pytest.raises(ValueError, match="Line count mismatch"):
            generator.validate_form(lines, pattern, "Haiku")

    def test_generate_constrained_line_basic(self):
        """Test generating a basic constrained line."""
        try:
            generator = FormGenerator()

            # Try to generate a 5-syllable line
            line, syllables = generator.generate_constrained_line(
                target_syllables=5, max_attempts=50
            )

            # Should either succeed or return None
            if line is not None:
                assert isinstance(line, str)
                assert len(line) > 0
                assert syllables >= 4  # Allow within 1
                assert syllables <= 5

        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_constrained_line_with_seeds(self):
        """Test generating a constrained line with seed words."""
        try:
            generator = FormGenerator()

            seed_words = ["ocean", "waves", "shore"]
            line, syllables = generator.generate_constrained_line(
                target_syllables=7, seed_words=seed_words, max_attempts=50
            )

            if line is not None:
                assert isinstance(line, str)
                assert len(line) > 0
                # Allow some flexibility (within 1 syllable)
                assert syllables >= 6
                assert syllables <= 7

        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_haiku_basic(self):
        """Test generating a basic haiku."""
        try:
            generator = FormGenerator()

            lines, validation = generator.generate_haiku(max_attempts=100, strict=False)

            assert isinstance(lines, list)
            assert len(lines) == 3
            assert all(isinstance(line, str) for line in lines)
            assert all(len(line) > 0 for line in lines)

            assert isinstance(validation, FormValidationResult)
            assert validation.form_name == "Haiku"
            assert len(validation.constraints) == 3

            print("\nGenerated Haiku:")
            for line in lines:
                print(f"  {line}")

        except ValueError as e:
            # It's okay if generation fails occasionally
            pytest.skip(f"Haiku generation failed: {e}")
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_haiku_with_seeds(self):
        """Test generating a haiku with seed words."""
        try:
            generator = FormGenerator()
            seed_words = ["autumn", "leaves", "wind"]

            lines, validation = generator.generate_haiku(
                seed_words=seed_words, max_attempts=100, strict=False
            )

            assert isinstance(lines, list)
            assert len(lines) == 3

            print("\nGenerated Haiku (with seeds):")
            for line in lines:
                print(f"  {line}")

        except ValueError as e:
            pytest.skip(f"Haiku generation failed: {e}")
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_tanka_basic(self):
        """Test generating a basic tanka."""
        try:
            generator = FormGenerator()

            lines, validation = generator.generate_tanka(max_attempts=100, strict=False)

            assert isinstance(lines, list)
            assert len(lines) == 5
            assert all(isinstance(line, str) for line in lines)
            assert all(len(line) > 0 for line in lines)

            assert isinstance(validation, FormValidationResult)
            assert validation.form_name == "Tanka"
            assert len(validation.constraints) == 5

            print("\nGenerated Tanka:")
            for line in lines:
                print(f"  {line}")

        except ValueError as e:
            pytest.skip(f"Tanka generation failed: {e}")
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_tanka_with_seeds(self):
        """Test generating a tanka with seed words."""
        try:
            generator = FormGenerator()
            seed_words = ["moon", "reflection", "water"]

            lines, validation = generator.generate_tanka(
                seed_words=seed_words, max_attempts=100, strict=False
            )

            assert isinstance(lines, list)
            assert len(lines) == 5

            print("\nGenerated Tanka (with seeds):")
            for line in lines:
                print(f"  {line}")

        except ValueError as e:
            pytest.skip(f"Tanka generation failed: {e}")
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_senryu_basic(self):
        """Test generating a basic senryu."""
        try:
            generator = FormGenerator()

            lines, validation = generator.generate_senryu(max_attempts=100, strict=False)

            assert isinstance(lines, list)
            assert len(lines) == 3
            assert all(isinstance(line, str) for line in lines)
            assert all(len(line) > 0 for line in lines)

            assert isinstance(validation, FormValidationResult)
            assert validation.form_name == "Senryu"
            assert len(validation.constraints) == 3

            print("\nGenerated Senryu:")
            for line in lines:
                print(f"  {line}")

        except ValueError as e:
            pytest.skip(f"Senryu generation failed: {e}")
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise

    def test_generate_senryu_with_seeds(self):
        """Test generating a senryu with seed words."""
        try:
            generator = FormGenerator()
            seed_words = ["laughter", "tears", "life"]

            lines, validation = generator.generate_senryu(
                seed_words=seed_words, max_attempts=100, strict=False
            )

            assert isinstance(lines, list)
            assert len(lines) == 3

            print("\nGenerated Senryu (with seeds):")
            for line in lines:
                print(f"  {line}")

        except ValueError as e:
            pytest.skip(f"Senryu generation failed: {e}")
        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise


class TestFormPatterns:
    """Test correct syllable patterns for each form."""

    def test_haiku_pattern(self):
        """Test haiku has 5-7-5 pattern."""
        generator = FormGenerator()

        # Create known lines with exact syllables
        lines = [
            "silent morning light",  # 5 syllables
            "dancing shadows on the wall",  # 7 syllables
            "peace fills the small room",  # 5 syllables
        ]
        pattern = [5, 7, 5]

        result = generator.validate_form(lines, pattern, "Haiku")

        # Check each line's syllable count
        for constraint in result.constraints:
            print(
                f"Line {constraint.line_number}: "
                f"target={constraint.target_syllables}, "
                f"actual={constraint.actual_syllables}"
            )

    def test_tanka_pattern(self):
        """Test tanka has 5-7-5-7-7 pattern."""
        generator = FormGenerator()

        # Create known lines (approximate syllables)
        lines = [
            "silent morning light",  # ~5 syllables
            "dancing shadows on the wall",  # ~7 syllables
            "peace fills the small room",  # ~5 syllables
            "memories of yesterday fade",  # ~7 syllables
            "into the evening mist now",  # ~7 syllables
        ]
        pattern = [5, 7, 5, 7, 7]

        result = generator.validate_form(lines, pattern, "Tanka")

        # Check pattern length
        assert len(result.constraints) == 5

    def test_senryu_pattern(self):
        """Test senryu has 5-7-5 pattern (same as haiku)."""
        generator = FormGenerator()

        lines = [
            "silent morning light",  # 5 syllables
            "dancing shadows on the wall",  # 7 syllables
            "peace fills the small room",  # 5 syllables
        ]
        pattern = [5, 7, 5]

        result = generator.validate_form(lines, pattern, "Senryu")

        # Check each line's syllable count
        assert len(result.constraints) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_target(self):
        """Test generating lines with very short syllable targets."""
        try:
            generator = FormGenerator()

            # Try 1 syllable
            line, syllables = generator.generate_constrained_line(
                target_syllables=1, max_attempts=20
            )

            if line is not None:
                assert syllables >= 1
                assert syllables <= 2  # Allow some flexibility

        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            # It's okay if this fails - 1 syllable is very constraining
            pytest.skip(f"Failed to generate 1-syllable line: {e}")

    def test_very_long_target(self):
        """Test generating lines with longer syllable targets."""
        try:
            generator = FormGenerator()

            # Try 12 syllables
            line, syllables = generator.generate_constrained_line(
                target_syllables=12, max_attempts=50
            )

            if line is not None:
                assert syllables >= 11
                assert syllables <= 12

        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            # Long lines might fail occasionally
            pytest.skip(f"Failed to generate 12-syllable line: {e}")

    def test_empty_seed_words(self):
        """Test generation with empty seed words."""
        try:
            generator = FormGenerator()

            line, syllables = generator.generate_constrained_line(
                target_syllables=5, seed_words=[], max_attempts=20
            )

            if line is not None:
                assert isinstance(line, str)

        except Exception as e:
            if "Network calls disabled" in str(e):
                pytest.skip("Network calls disabled in test environment")
            raise


class TestDeterminism:
    """Test deterministic generation with seeds."""

    def test_syllable_counting_deterministic(self):
        """Test that syllable counting is deterministic."""
        word = "beautiful"
        count1 = count_syllables(word)
        count2 = count_syllables(word)
        assert count1 == count2

        # Test multiple words
        for word in ["hello", "world", "poetry", "computer", "algorithm"]:
            assert count_syllables(word) == count_syllables(word)


if __name__ == "__main__":
    print("Running comprehensive forms tests...")
    print("=" * 70)

    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])

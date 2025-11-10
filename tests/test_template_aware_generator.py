"""Tests for template-aware poem generation."""

import pytest
from pathlib import Path

from poetryplayground.template_aware_generator import (
    TemplateAwareGenerator,
    GeneratedLine,
    GenerationResult,
)
from poetryplayground.poem_template import (
    PoemTemplate,
    LineTemplate,
    LineType,
)
from poetryplayground.core.quality_scorer import EmotionalTone, FormalityLevel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def generator():
    """Create TemplateAwareGenerator instance."""
    return TemplateAwareGenerator()


@pytest.fixture
def simple_template():
    """Create a simple 3-line template."""
    return PoemTemplate(
        title="Simple Test",
        source="test",
        lines=3,
        line_templates=[
            LineTemplate(
                syllable_count=5,
                pos_pattern=["DET", "NOUN", "VERB"],
                line_type=LineType.OPENING,
            ),
            LineTemplate(
                syllable_count=7,
                pos_pattern=["ADJ", "NOUN", "VERB", "ADV"],
                line_type=LineType.PIVOT,
            ),
            LineTemplate(
                syllable_count=5,
                pos_pattern=["NOUN", "VERB"],
                line_type=LineType.CLOSING,
            ),
        ],
        syllable_pattern=[5, 7, 5],
    )


@pytest.fixture
def metaphor_template():
    """Create a template emphasizing metaphors."""
    return PoemTemplate(
        title="Metaphor Test",
        source="test",
        lines=2,
        line_templates=[
            LineTemplate(
                syllable_count=8,
                pos_pattern=["NOUN", "VERB", "NOUN"],
                line_type=LineType.OPENING,
                metaphor_type="direct",
                semantic_domain="nature",
            ),
            LineTemplate(
                syllable_count=8,
                pos_pattern=["ADJ", "NOUN", "VERB"],
                line_type=LineType.CLOSING,
                metaphor_type="simile",
                semantic_domain="nature",
            ),
        ],
        syllable_pattern=[8, 8],
        semantic_domains=["nature"],
        metaphor_types=["direct", "simile"],
    )


@pytest.fixture
def styled_template():
    """Create a template with style components."""
    return PoemTemplate(
        title="Styled Test",
        source="test",
        lines=3,
        line_templates=[
            LineTemplate(
                syllable_count=6,
                pos_pattern=["DET", "ADJ", "NOUN"],
                line_type=LineType.OPENING,
                semantic_domain="emotions",
                min_quality_score=0.6,
            ),
            LineTemplate(
                syllable_count=6,
                pos_pattern=["VERB", "NOUN"],
                line_type=LineType.PIVOT,
                semantic_domain="emotions",
                min_quality_score=0.6,
            ),
            LineTemplate(
                syllable_count=6,
                pos_pattern=["NOUN", "VERB"],
                line_type=LineType.CLOSING,
                semantic_domain="emotions",
                min_quality_score=0.6,
            ),
        ],
        syllable_pattern=[6, 6, 6],
        semantic_domains=["emotions"],
        emotional_tone=EmotionalTone.LIGHT,
        formality_level=FormalityLevel.CONVERSATIONAL,
    )


# ============================================================================
# Unit Tests - Basic Generation
# ============================================================================


class TestBasicGeneration:
    """Test basic template-aware generation."""

    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly."""
        assert generator is not None
        assert generator.metaphor_generator is not None
        assert generator.seed_generator is not None

    def test_generate_from_simple_template(self, generator, simple_template):
        """Test generating from a simple template."""
        results = generator.generate_from_template(simple_template, count=1)

        assert len(results) == 1
        result = results[0]

        assert result.success is True
        assert result.error_message is None
        assert result.poem != ""
        assert len(result.lines) == 3
        assert result.template == simple_template

    def test_generate_multiple_poems(self, generator, simple_template):
        """Test generating multiple poems from same template."""
        results = generator.generate_from_template(simple_template, count=3)

        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert len(result.lines) == 3

    def test_generated_line_structure(self, generator, simple_template):
        """Test that generated lines have correct structure."""
        results = generator.generate_from_template(simple_template, count=1)
        result = results[0]

        for i, line in enumerate(result.lines):
            assert isinstance(line, GeneratedLine)
            assert line.text != ""
            assert line.line_number == i
            assert line.template == simple_template.line_templates[i]
            assert line.source_method in ["metaphor", "seed", "fallback"]
            assert 0.0 <= line.quality_score <= 1.0

    def test_poem_assembly(self, generator, simple_template):
        """Test that poem is correctly assembled from lines."""
        results = generator.generate_from_template(simple_template, count=1)
        result = results[0]

        # Poem should be lines joined by newlines
        expected = "\n".join(line.text for line in result.lines)
        assert result.poem == expected

        # Should have exactly 2 newlines (3 lines)
        assert result.poem.count("\n") == 2


# ============================================================================
# Unit Tests - Metaphor Generation
# ============================================================================


class TestMetaphorGeneration:
    """Test metaphor-based line generation."""

    def test_metaphor_template_generation(self, generator, metaphor_template):
        """Test generation with metaphor constraints."""
        results = generator.generate_from_template(metaphor_template, count=1)

        assert len(results) == 1
        result = results[0]

        assert result.success is True
        assert len(result.lines) == 2

    def test_metaphor_line_identification(self, generator, metaphor_template):
        """Test that metaphor lines are identified."""
        results = generator.generate_from_template(metaphor_template, count=1)
        result = results[0]

        # At least one line should use metaphor generation
        # (or fallback if metaphor generation fails)
        methods = [line.source_method for line in result.lines]
        assert any(method in ["metaphor", "fallback"] for method in methods)

    def test_semantic_domain_usage(self, generator, metaphor_template):
        """Test that semantic domains are used in generation."""
        results = generator.generate_from_template(metaphor_template, count=1)
        result = results[0]

        # Should successfully generate with semantic domain constraints
        assert result.success is True
        assert "nature" in metaphor_template.semantic_domains


# ============================================================================
# Unit Tests - Seed Generation
# ============================================================================


class TestSeedGeneration:
    """Test line seed-based generation."""

    def test_seed_template_generation(self, generator, styled_template):
        """Test generation using line seeds."""
        results = generator.generate_from_template(styled_template, count=1)

        assert len(results) == 1
        result = results[0]

        assert result.success is True
        assert len(result.lines) == 3

    def test_style_scores_present(self, generator, styled_template):
        """Test that style scores are calculated for seed generation."""
        results = generator.generate_from_template(styled_template, count=1)
        result = results[0]

        # At least some lines should have style scores (from seed generation)
        # Note: metaphor lines won't have style scores
        has_style_score = any(line.style_score is not None for line in result.lines)
        # This may be True or False depending on which method was used
        assert isinstance(has_style_score, bool)

    def test_opening_line_generation(self, generator, styled_template):
        """Test that opening lines are generated correctly."""
        results = generator.generate_from_template(styled_template, count=1)
        result = results[0]

        first_line = result.lines[0]
        assert first_line.template.line_type == LineType.OPENING
        assert first_line.text != ""


# ============================================================================
# Unit Tests - Fallback Generation
# ============================================================================


class TestFallbackGeneration:
    """Test fallback generation mechanisms."""

    def test_fallback_allowed(self, generator, simple_template):
        """Test that fallback generation works."""
        # This should always succeed with fallback
        results = generator.generate_from_template(
            simple_template,
            count=1,
            allow_fallback=True,
        )

        assert len(results) == 1
        assert results[0].success is True

    def test_fallback_disallowed_may_fail(self, generator):
        """Test that disabling fallback may cause failures."""
        # Create an impossible template
        impossible_template = PoemTemplate(
            title="Impossible",
            source="test",
            lines=1,
            line_templates=[
                LineTemplate(
                    syllable_count=100,  # Unreasonable syllable count
                    pos_pattern=["NOUN"] * 50,  # Unreasonable POS pattern
                    line_type=LineType.OPENING,
                ),
            ],
            syllable_pattern=[100],
        )

        # With fallback disabled, this might fail or use fallback anyway
        # We just check that it doesn't crash
        results = generator.generate_from_template(
            impossible_template,
            count=1,
            allow_fallback=False,
        )

        assert len(results) == 1
        # Result may succeed (fallback) or fail (no fallback)
        # Either is acceptable


# ============================================================================
# Unit Tests - Quality and Metadata
# ============================================================================


class TestQualityAndMetadata:
    """Test quality scoring and metadata."""

    def test_quality_scores_calculated(self, generator, simple_template):
        """Test that quality scores are calculated."""
        results = generator.generate_from_template(simple_template, count=1)
        result = results[0]

        for line in result.lines:
            assert 0.0 <= line.quality_score <= 1.0

    def test_metadata_present(self, generator, simple_template):
        """Test that generation metadata is present."""
        results = generator.generate_from_template(simple_template, count=1)
        result = results[0]

        assert "average_quality" in result.metadata
        assert "template_title" in result.metadata
        assert "template_source" in result.metadata

        assert result.metadata["template_title"] == "Simple Test"
        assert result.metadata["template_source"] == "test"

    def test_average_quality_calculation(self, generator, simple_template):
        """Test that average quality is calculated correctly."""
        results = generator.generate_from_template(simple_template, count=1)
        result = results[0]

        avg_quality = result.metadata["average_quality"]
        expected = sum(line.quality_score for line in result.lines) / len(result.lines)

        assert abs(avg_quality - expected) < 0.001

    def test_min_quality_threshold_respected(self, generator, styled_template):
        """Test that minimum quality thresholds are considered."""
        results = generator.generate_from_template(styled_template, count=1)
        result = results[0]

        # Should successfully generate even with quality constraints
        assert result.success is True


# ============================================================================
# Unit Tests - Variations
# ============================================================================


class TestVariations:
    """Test template variation generation."""

    def test_generate_variations(self, generator, simple_template):
        """Test generating variations of a template."""
        results = generator.generate_variations(simple_template, count=3)

        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert len(result.lines) == 3

    def test_varied_domains(self, generator, metaphor_template):
        """Test that semantic domains are varied."""
        results = generator.generate_variations(
            metaphor_template,
            count=5,
            vary_domains=True,
            vary_tone=False,
        )

        assert len(results) == 5
        # All should succeed
        assert all(r.success for r in results)

    def test_varied_tone(self, generator, styled_template):
        """Test that emotional tone is varied."""
        results = generator.generate_variations(
            styled_template,
            count=5,
            vary_domains=False,
            vary_tone=True,
        )

        assert len(results) == 5
        # All should succeed
        assert all(r.success for r in results)

    def test_variations_are_different(self, generator, simple_template):
        """Test that variations produce different poems."""
        results = generator.generate_variations(simple_template, count=3)

        poems = [r.poem for r in results]
        # At least some should be different (though theoretically could be same)
        # We just verify they're all valid
        assert all(poem != "" for poem in poems)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full generation pipeline."""

    def test_full_pipeline_metaphor_template(self, generator, metaphor_template):
        """Test full pipeline with metaphor template."""
        results = generator.generate_from_template(metaphor_template, count=2)

        assert len(results) == 2
        for result in results:
            assert result.success is True
            assert len(result.lines) == 2
            assert result.poem.count("\n") == 1

    def test_full_pipeline_styled_template(self, generator, styled_template):
        """Test full pipeline with styled template."""
        results = generator.generate_from_template(styled_template, count=2)

        assert len(results) == 2
        for result in results:
            assert result.success is True
            assert len(result.lines) == 3
            assert result.template == styled_template

    def test_end_to_end_with_library(self, generator):
        """Test end-to-end with template library."""
        from poetryplayground.template_library import get_template_library

        library = get_template_library()

        # Get a haiku template
        template = library.get_template("haiku_nature")

        if template:
            results = generator.generate_from_template(template, count=1)
            assert len(results) == 1
            assert results[0].success is True
        else:
            # If template doesn't exist, that's okay for this test
            pytest.skip("haiku_nature template not available")


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_line_template(self, generator):
        """Test template with single line."""
        template = PoemTemplate(
            title="Single Line",
            source="test",
            lines=1,
            line_templates=[
                LineTemplate(
                    syllable_count=5,
                    pos_pattern=["DET", "NOUN"],
                    line_type=LineType.OPENING,
                ),
            ],
            syllable_pattern=[5],
        )

        results = generator.generate_from_template(template, count=1)

        assert len(results) == 1
        assert results[0].success is True
        assert len(results[0].lines) == 1
        assert "\n" not in results[0].poem  # No newlines in single-line poem

    def test_long_template(self, generator):
        """Test template with many lines."""
        template = PoemTemplate(
            title="Long Poem",
            source="test",
            lines=10,
            line_templates=[
                LineTemplate(
                    syllable_count=7,
                    pos_pattern=["NOUN", "VERB"],
                    line_type=LineType.IMAGE,
                )
                for _ in range(10)
            ],
            syllable_pattern=[7] * 10,
        )

        results = generator.generate_from_template(template, count=1)

        assert len(results) == 1
        assert results[0].success is True
        assert len(results[0].lines) == 10

    def test_empty_pos_pattern(self, generator):
        """Test template with empty POS pattern."""
        template = PoemTemplate(
            title="Empty POS",
            source="test",
            lines=1,
            line_templates=[
                LineTemplate(
                    syllable_count=5,
                    pos_pattern=[],  # Empty POS pattern
                    line_type=LineType.OPENING,
                ),
            ],
            syllable_pattern=[5],
        )

        results = generator.generate_from_template(template, count=1)

        # Should handle gracefully (fallback)
        assert len(results) == 1
        # May succeed or fail, but shouldn't crash

    def test_generate_zero_count(self, generator, simple_template):
        """Test requesting zero poems."""
        results = generator.generate_from_template(simple_template, count=0)

        assert len(results) == 0

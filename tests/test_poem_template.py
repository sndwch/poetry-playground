"""Tests for poem template system."""

import pytest

from poetryplayground.poem_template import (
    LineTemplate,
    LineType,
    PoemTemplate,
    create_haiku_template,
    create_tanka_template,
)
from poetryplayground.core.quality_scorer import EmotionalTone, FormalityLevel


# ============================================================================
# Unit Tests - LineTemplate
# ============================================================================


class TestLineTemplate:
    """Test LineTemplate dataclass."""

    def test_line_template_creation(self):
        """Test basic LineTemplate instantiation."""
        template = LineTemplate(
            syllable_count=5,
            pos_pattern=["DET", "NOUN", "VERB"],
            line_type=LineType.OPENING,
        )

        assert template.syllable_count == 5
        assert template.pos_pattern == ["DET", "NOUN", "VERB"]
        assert template.line_type == LineType.OPENING
        assert template.concreteness_target == 0.5  # Default
        assert template.min_quality_score == 0.6  # Default

    def test_line_template_with_metaphor(self):
        """Test LineTemplate with metaphor type."""
        template = LineTemplate(
            syllable_count=7,
            pos_pattern=["NOUN", "VERB", "DET", "NOUN"],
            line_type=LineType.IMAGE,
            metaphor_type="SIMILE",
            semantic_domain="nature",
        )

        assert template.metaphor_type == "SIMILE"
        assert template.semantic_domain == "nature"

    def test_line_template_serialization(self):
        """Test to_dict and from_dict."""
        original = LineTemplate(
            syllable_count=5,
            pos_pattern=["ADJ", "NOUN"],
            line_type=LineType.PIVOT,
            metaphor_type="DIRECT",
            semantic_domain="emotion",
            concreteness_target=0.7,
            min_quality_score=0.8,
        )

        # Serialize
        data = original.to_dict()
        assert data["syllable_count"] == 5
        assert data["pos_pattern"] == ["ADJ", "NOUN"]
        assert data["line_type"] == "pivot"
        assert data["metaphor_type"] == "DIRECT"

        # Deserialize
        restored = LineTemplate.from_dict(data)
        assert restored.syllable_count == original.syllable_count
        assert restored.pos_pattern == original.pos_pattern
        assert restored.line_type == original.line_type
        assert restored.metaphor_type == original.metaphor_type
        assert restored.semantic_domain == original.semantic_domain
        assert restored.concreteness_target == original.concreteness_target
        assert restored.min_quality_score == original.min_quality_score


# ============================================================================
# Unit Tests - PoemTemplate
# ============================================================================


class TestPoemTemplate:
    """Test PoemTemplate dataclass."""

    def test_minimal_template_creation(self):
        """Test creating minimal valid template."""
        template = PoemTemplate(
            title="Test Template",
            lines=3,
            syllable_pattern=[5, 7, 5],
        )

        assert template.title == "Test Template"
        assert template.lines == 3
        assert template.syllable_pattern == [5, 7, 5]
        assert template.source == "user-provided"  # Default
        assert template.author == "anonymous"  # Default

    def test_full_template_creation(self):
        """Test creating template with all fields."""
        line_templates = [
            LineTemplate(5, ["DET", "NOUN", "VERB"], LineType.OPENING),
            LineTemplate(7, ["ADJ", "NOUN", "VERB", "ADV"], LineType.IMAGE),
            LineTemplate(5, ["DET", "NOUN", "VERB"], LineType.CLOSING),
        ]

        template = PoemTemplate(
            title="Full Haiku",
            source="test",
            author="tester",
            lines=3,
            line_templates=line_templates,
            syllable_pattern=[5, 7, 5],
            semantic_domains=["nature", "season"],
            metaphor_types=["SIMILE", "DIRECT"],
            emotional_tone=EmotionalTone.LIGHT,
            formality_level=FormalityLevel.FORMAL,
            concreteness_ratio=0.8,
            min_quality_score=0.7,
            notes="Test template",
        )

        assert len(template.line_templates) == 3
        assert template.semantic_domains == ["nature", "season"]
        assert template.metaphor_types == ["SIMILE", "DIRECT"]
        assert template.emotional_tone == EmotionalTone.LIGHT
        assert template.formality_level == FormalityLevel.FORMAL
        assert template.concreteness_ratio == 0.8
        assert template.min_quality_score == 0.7
        assert template.notes == "Test template"

    def test_template_validation_line_count(self):
        """Test validation catches line count mismatch."""
        with pytest.raises(ValueError, match="must match line count"):
            PoemTemplate(
                title="Bad Template",
                lines=3,
                syllable_pattern=[5, 7],  # Only 2 entries for 3 lines
            )

    def test_template_validation_line_templates_count(self):
        """Test validation catches line templates count mismatch."""
        line_templates = [
            LineTemplate(5, ["DET", "NOUN", "VERB"], LineType.OPENING),
            LineTemplate(7, ["ADJ", "NOUN", "VERB"], LineType.IMAGE),
        ]

        with pytest.raises(ValueError, match="must match line count"):
            PoemTemplate(
                title="Bad Template",
                lines=3,
                line_templates=line_templates,  # Only 2 templates for 3 lines
            )

    def test_template_validation_quality_score_range(self):
        """Test validation catches invalid quality scores."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            PoemTemplate(
                title="Bad Template",
                lines=1,
                min_quality_score=1.5,  # Invalid
            )

    def test_template_validation_concreteness_range(self):
        """Test validation catches invalid concreteness ratio."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            PoemTemplate(
                title="Bad Template",
                lines=1,
                concreteness_ratio=-0.1,  # Invalid
            )

    def test_template_validation_empty_title(self):
        """Test validation catches empty title."""
        with pytest.raises(ValueError, match="must have a title"):
            PoemTemplate(
                title="",
                lines=1,
            )

    def test_template_validation_zero_lines(self):
        """Test validation catches zero lines."""
        with pytest.raises(ValueError, match="at least 1 line"):
            PoemTemplate(
                title="Bad Template",
                lines=0,
            )

    def test_template_serialization(self):
        """Test to_dict and from_dict with full template."""
        original = PoemTemplate(
            title="Test Haiku",
            source="manual",
            author="testbot",
            lines=3,
            syllable_pattern=[5, 7, 5],
            line_templates=[
                LineTemplate(5, ["DET", "NOUN", "VERB"], LineType.OPENING),
                LineTemplate(7, ["ADJ", "NOUN", "VERB", "ADV"], LineType.IMAGE),
                LineTemplate(5, ["DET", "NOUN", "VERB"], LineType.CLOSING),
            ],
            semantic_domains=["nature"],
            metaphor_types=["SIMILE"],
            emotional_tone=EmotionalTone.NEUTRAL,
            formality_level=FormalityLevel.CONVERSATIONAL,
            concreteness_ratio=0.6,
            min_quality_score=0.65,
            style_components={"avg_line_length": 15, "pos_diversity": 0.8},
            notes="Test notes",
        )

        # Serialize
        data = original.to_dict()
        assert data["title"] == "Test Haiku"
        assert data["lines"] == 3
        assert data["syllable_pattern"] == [5, 7, 5]
        assert len(data["line_templates"]) == 3
        assert data["emotional_tone"] == "neutral"
        assert data["formality_level"] == "conversational"

        # Deserialize
        restored = PoemTemplate.from_dict(data)
        assert restored.title == original.title
        assert restored.source == original.source
        assert restored.author == original.author
        assert restored.lines == original.lines
        assert restored.syllable_pattern == original.syllable_pattern
        assert len(restored.line_templates) == len(original.line_templates)
        assert restored.semantic_domains == original.semantic_domains
        assert restored.metaphor_types == original.metaphor_types
        assert restored.emotional_tone == original.emotional_tone
        assert restored.formality_level == original.formality_level
        assert restored.concreteness_ratio == original.concreteness_ratio
        assert restored.min_quality_score == original.min_quality_score
        assert restored.style_components == original.style_components
        assert restored.notes == original.notes

    def test_get_line_template(self):
        """Test getting specific line template."""
        line_templates = [
            LineTemplate(5, ["DET", "NOUN"], LineType.OPENING),
            LineTemplate(7, ["ADJ", "NOUN", "VERB"], LineType.IMAGE),
        ]

        template = PoemTemplate(
            title="Test",
            lines=2,
            line_templates=line_templates,
        )

        # Valid indices
        line0 = template.get_line_template(0)
        assert line0 is not None
        assert line0.syllable_count == 5

        line1 = template.get_line_template(1)
        assert line1 is not None
        assert line1.syllable_count == 7

        # Invalid index
        assert template.get_line_template(2) is None
        assert template.get_line_template(-1) is None

    def test_get_total_syllables(self):
        """Test calculating total syllables."""
        template = PoemTemplate(
            title="Test",
            lines=3,
            syllable_pattern=[5, 7, 5],
        )

        assert template.get_total_syllables() == 17

    def test_get_total_syllables_empty(self):
        """Test total syllables with no pattern."""
        template = PoemTemplate(
            title="Test",
            lines=1,
        )

        assert template.get_total_syllables() == 0

    def test_matches_structure(self):
        """Test structure matching."""
        template = PoemTemplate(
            title="Test",
            lines=3,
            syllable_pattern=[5, 7, 5],
        )

        assert template.matches_structure([5, 7, 5]) is True
        assert template.matches_structure([5, 7, 4]) is False
        assert template.matches_structure([5, 7]) is False
        assert template.matches_structure([5, 7, 5, 7]) is False

    def test_string_representation(self):
        """Test __str__ and __repr__."""
        template = PoemTemplate(
            title="Test Haiku",
            lines=3,
            syllable_pattern=[5, 7, 5],
            emotional_tone=EmotionalTone.DARK,
        )

        str_repr = str(template)
        assert "Test Haiku" in str_repr
        assert "3 lines" in str_repr
        assert "17 syllables" in str_repr
        assert "dark" in str_repr

        repr_str = repr(template)
        assert "PoemTemplate" in repr_str
        assert "title='Test Haiku'" in repr_str
        assert "lines=3" in repr_str


# ============================================================================
# Integration Tests - Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Test convenience template creation functions."""

    def test_create_haiku_template(self):
        """Test haiku template creation."""
        template = create_haiku_template()

        assert template.title == "Haiku Template"
        assert template.source == "traditional"
        assert template.lines == 3
        assert template.syllable_pattern == [5, 7, 5]
        assert len(template.line_templates) == 3
        assert template.semantic_domains == ["nature"]
        assert template.formality_level == FormalityLevel.FORMAL
        assert template.concreteness_ratio == 0.7

    def test_create_haiku_template_custom(self):
        """Test haiku template with custom parameters."""
        template = create_haiku_template(
            semantic_domains=["urban", "technology"],
            emotional_tone=EmotionalTone.DARK,
        )

        assert template.semantic_domains == ["urban", "technology"]
        assert template.emotional_tone == EmotionalTone.DARK
        assert template.syllable_pattern == [5, 7, 5]  # Structure unchanged

    def test_create_tanka_template(self):
        """Test tanka template creation."""
        template = create_tanka_template()

        assert template.title == "Tanka Template"
        assert template.source == "traditional"
        assert template.lines == 5
        assert template.syllable_pattern == [5, 7, 5, 7, 7]
        assert len(template.line_templates) == 5
        assert template.semantic_domains == ["nature", "emotion"]
        assert template.formality_level == FormalityLevel.FORMAL
        assert template.concreteness_ratio == 0.6

    def test_create_tanka_template_custom(self):
        """Test tanka template with custom parameters."""
        template = create_tanka_template(
            semantic_domains=["love", "loss"],
            emotional_tone=EmotionalTone.MIXED,
        )

        assert template.semantic_domains == ["love", "loss"]
        assert template.emotional_tone == EmotionalTone.MIXED
        assert template.syllable_pattern == [5, 7, 5, 7, 7]  # Structure unchanged


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_template_with_no_line_templates(self):
        """Test template with syllable pattern but no line templates."""
        template = PoemTemplate(
            title="Minimal",
            lines=2,
            syllable_pattern=[5, 7],
        )

        # Should be valid - line_templates are optional
        assert template.lines == 2
        assert len(template.line_templates) == 0

    def test_template_with_empty_lists(self):
        """Test template with empty semantic domains and metaphor types."""
        template = PoemTemplate(
            title="Empty",
            lines=1,
            semantic_domains=[],
            metaphor_types=[],
        )

        assert template.semantic_domains == []
        assert template.metaphor_types == []

    def test_deserialization_with_missing_optional_fields(self):
        """Test deserializing with minimal data."""
        data = {
            "title": "Minimal",
            "lines": 1,
        }

        template = PoemTemplate.from_dict(data)
        assert template.title == "Minimal"
        assert template.lines == 1
        assert template.source == "user-provided"  # Default
        assert template.author == "anonymous"  # Default
        assert template.emotional_tone == EmotionalTone.NEUTRAL  # Default

    def test_line_template_validation_in_poem_template(self):
        """Test that invalid line templates are caught during validation."""
        line_templates = [
            LineTemplate(0, ["NOUN"], LineType.OPENING),  # Invalid: 0 syllables
        ]

        with pytest.raises(ValueError, match="must have at least 1 syllable"):
            PoemTemplate(
                title="Bad",
                lines=1,
                line_templates=line_templates,
            )

    def test_line_template_with_empty_pos_pattern(self):
        """Test that empty POS pattern is caught."""
        line_templates = [
            LineTemplate(5, [], LineType.OPENING),  # Invalid: empty POS pattern
        ]

        with pytest.raises(ValueError, match="must have a POS pattern"):
            PoemTemplate(
                title="Bad",
                lines=1,
                line_templates=line_templates,
            )

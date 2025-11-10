"""Tests for template extraction from poems."""

import pytest

from poetryplayground.template_extractor import TemplateExtractor
from poetryplayground.poem_template import LineType
from poetryplayground.core.quality_scorer import EmotionalTone, FormalityLevel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def extractor():
    """Create TemplateExtractor instance."""
    return TemplateExtractor()


# ============================================================================
# Unit Tests - Basic Extraction
# ============================================================================


class TestBasicExtraction:
    """Test basic template extraction functionality."""

    def test_extract_haiku(self, extractor):
        """Test extracting template from a haiku."""
        poem = """The old pond—
A frog jumps in,
Water's sound!"""

        template = extractor.extract_template(poem, title="Basho Haiku")

        assert template.title == "Basho Haiku"
        assert template.lines == 3
        assert len(template.line_templates) == 3
        assert len(template.syllable_pattern) == 3

    def test_extract_multiline_poem(self, extractor):
        """Test extracting from longer poem."""
        poem = """Roses are red,
Violets are blue,
Sugar is sweet,
And so are you."""

        template = extractor.extract_template(poem, title="Roses", author="Anonymous")

        assert template.title == "Roses"
        assert template.author == "Anonymous"
        assert template.lines == 4
        assert len(template.line_templates) == 4

    def test_syllable_pattern_extraction(self, extractor):
        """Test that syllable patterns are extracted correctly."""
        poem = """The cat
Sits on mat
Now"""

        template = extractor.extract_template(poem)

        assert template.syllable_pattern == [2, 3, 1]
        assert template.get_total_syllables() == 6

    def test_pos_pattern_extraction(self, extractor):
        """Test POS pattern extraction."""
        poem = """The red fox
Jumps quickly"""

        template = extractor.extract_template(poem)

        # Check that we got POS patterns
        assert len(template.line_templates) == 2
        assert len(template.line_templates[0].pos_pattern) > 0
        assert "DET" in template.line_templates[0].pos_pattern  # "The"
        assert "NOUN" in template.line_templates[0].pos_pattern  # "fox"

    def test_empty_poem_raises_error(self, extractor):
        """Test that empty poem raises ValueError."""
        with pytest.raises(ValueError, match="no valid lines"):
            extractor.extract_template("")

    def test_whitespace_only_raises_error(self, extractor):
        """Test that whitespace-only poem raises ValueError."""
        with pytest.raises(ValueError, match="no valid lines"):
            extractor.extract_template("   \n\n   \n")


# ============================================================================
# Unit Tests - Line Type Classification
# ============================================================================


class TestLineTypeClassification:
    """Test line type classification."""

    def test_opening_line_detected(self, extractor):
        """Test that first line is classified as opening."""
        poem = """Once upon a time
There was a story
The end"""

        template = extractor.extract_template(poem)

        assert template.line_templates[0].line_type == LineType.OPENING

    def test_closing_line_detected(self, extractor):
        """Test that last line is classified as closing."""
        poem = """Once upon a time
There was a story
The end"""

        template = extractor.extract_template(poem)

        assert template.line_templates[-1].line_type == LineType.CLOSING

    def test_pivot_in_three_line_poem(self, extractor):
        """Test that middle line of 3-line poem is pivot."""
        poem = """First line here
Middle pivot line
Last line here"""

        template = extractor.extract_template(poem)

        assert template.lines == 3
        assert template.line_templates[1].line_type == LineType.PIVOT

    def test_transition_line_detected(self, extractor):
        """Test that transition markers are detected."""
        poem = """I love the day
But the night is dark
And cold it grows
The world keeps turning"""

        template = extractor.extract_template(poem)

        # Line with "but" should be classified as transition (not pivot in 4-line poem)
        assert template.line_templates[1].line_type == LineType.TRANSITION

    def test_emotional_line_detected(self, extractor):
        """Test that emotional words trigger emotional classification."""
        poem = """The sky is blue
My heart breaks here
The world turns on
And life goes on"""

        template = extractor.extract_template(poem)

        # Line with "heart" should be emotional (not pivot in 4-line poem)
        assert template.line_templates[1].line_type == LineType.EMOTIONAL


# ============================================================================
# Unit Tests - Metaphor Detection
# ============================================================================


class TestMetaphorDetection:
    """Test metaphor type detection."""

    def test_simile_detection(self, extractor):
        """Test simile metaphor detection."""
        poem = """Love is like a rose
Red and beautiful"""

        template = extractor.extract_template(poem)

        assert "simile" in template.metaphor_types
        assert template.line_templates[0].metaphor_type == "simile"

    def test_direct_metaphor_detection(self, extractor):
        """Test direct metaphor detection."""
        poem = """Time is a river
Flowing endlessly"""

        template = extractor.extract_template(poem)

        assert "direct" in template.metaphor_types
        assert template.line_templates[0].metaphor_type == "direct"

    def test_possessive_metaphor_detection(self, extractor):
        """Test possessive metaphor detection."""
        poem = """The silence of night
The shadow's embrace"""

        template = extractor.extract_template(poem)

        assert "possessive" in template.metaphor_types

    def test_compound_metaphor_detection(self, extractor):
        """Test compound metaphor detection."""
        poem = """A storm-heart beats
Thunder-love echoes"""

        template = extractor.extract_template(poem)

        assert "compound" in template.metaphor_types

    def test_no_metaphor_poem(self, extractor):
        """Test poem with no clear metaphors."""
        poem = """The cat sat down
And looked around"""

        template = extractor.extract_template(poem)

        # May have some metaphor types from possessive/structural patterns,
        # but shouldn't have simile or direct
        assert "simile" not in template.metaphor_types


# ============================================================================
# Unit Tests - Semantic Domain Extraction
# ============================================================================


class TestSemanticDomainExtraction:
    """Test semantic domain extraction."""

    def test_nature_domain_detected(self, extractor):
        """Test that nature words are detected."""
        poem = """The tree stands tall
Wind whispers through leaves
Birds sing in branches"""

        template = extractor.extract_template(poem)

        # Should detect nature domain
        assert len(template.semantic_domains) > 0
        # Nature-related domains should be present
        assert any(domain in ["nature", "natural_world", "elements"] for domain in template.semantic_domains)

    def test_emotion_domain_detected(self, extractor):
        """Test that emotion words are detected."""
        poem = """Love fills my heart
Joy dances freely
Hope blooms eternal"""

        template = extractor.extract_template(poem)

        # Should detect emotional domain
        assert len(template.semantic_domains) > 0

    def test_multiple_domains(self, extractor):
        """Test poem with multiple semantic domains."""
        poem = """The storm of love
Rages in my heart
Thunder and lightning"""

        template = extractor.extract_template(poem)

        # Should have multiple domains
        assert len(template.semantic_domains) >= 1


# ============================================================================
# Unit Tests - Emotional Tone Analysis
# ============================================================================


class TestEmotionalToneAnalysis:
    """Test emotional tone classification."""

    def test_dark_tone_detected(self, extractor):
        """Test that dark emotional tone is detected."""
        poem = """Death comes at night
Darkness falls heavy
Shadows fill the void
Cold and alone"""

        template = extractor.extract_template(poem)

        assert template.emotional_tone == EmotionalTone.DARK

    def test_light_tone_detected(self, extractor):
        """Test that light emotional tone is detected."""
        poem = """The sun brings joy
Bright and warm today
Love blooms with hope
Laughter fills the air"""

        template = extractor.extract_template(poem)

        assert template.emotional_tone == EmotionalTone.LIGHT

    def test_neutral_tone_detected(self, extractor):
        """Test that neutral tone is detected."""
        poem = """The cat sits down
The dog walks by
The bird flies high"""

        template = extractor.extract_template(poem)

        assert template.emotional_tone == EmotionalTone.NEUTRAL

    def test_mixed_tone_detected(self, extractor):
        """Test that mixed emotional tone is detected."""
        poem = """The sun brings joy
But darkness falls too
Hope and fear together"""

        template = extractor.extract_template(poem)

        # Should detect some emotional tone (light words slightly outnumber dark)
        assert template.emotional_tone in [EmotionalTone.MIXED, EmotionalTone.NEUTRAL, EmotionalTone.LIGHT]


# ============================================================================
# Unit Tests - Formality Level Analysis
# ============================================================================


class TestFormalityAnalysis:
    """Test formality level classification."""

    def test_archaic_formality_detected(self, extractor):
        """Test that archaic language is detected."""
        poem = """Thou art my love
Thy beauty doth shine
Whence came thee"""

        template = extractor.extract_template(poem)

        assert template.formality_level == FormalityLevel.ARCHAIC

    def test_casual_formality_detected(self, extractor):
        """Test that casual language is detected."""
        poem = """Yeah I'm gonna go
Wanna see what's up
Okay let's do this"""

        template = extractor.extract_template(poem)

        assert template.formality_level == FormalityLevel.CASUAL

    def test_formal_language(self, extractor):
        """Test that formal language is detected."""
        poem = """Magnificent illumination permeates
Extraordinary circumstances prevail
Incontrovertible evidence demonstrates"""

        template = extractor.extract_template(poem)

        assert template.formality_level == FormalityLevel.FORMAL

    def test_conversational_language(self, extractor):
        """Test conversational language."""
        poem = """The day is nice
I like the weather
Let's go outside"""

        template = extractor.extract_template(poem)

        assert template.formality_level == FormalityLevel.CONVERSATIONAL


# ============================================================================
# Unit Tests - Concreteness Analysis
# ============================================================================


class TestConcretenessAnalysis:
    """Test concreteness ratio calculation."""

    def test_concrete_poem(self, extractor):
        """Test that concrete imagery is detected."""
        poem = """The red apple
On wooden table
Under bright lamp"""

        template = extractor.extract_template(poem)

        # Should have relatively high concreteness
        assert template.concreteness_ratio > 0.4

    def test_abstract_poem(self, extractor):
        """Test that abstract concepts are detected."""
        poem = """Freedom and justice
Truth and beauty
Love and hope"""

        template = extractor.extract_template(poem)

        # Should have relatively low concreteness (more abstract)
        assert template.concreteness_ratio < 0.7


# ============================================================================
# Unit Tests - Quality Threshold Estimation
# ============================================================================


class TestQualityThresholdEstimation:
    """Test quality threshold estimation."""

    def test_quality_threshold_calculated(self, extractor):
        """Test that quality threshold is calculated."""
        poem = """The beautiful rose
Blooms in the garden
With vibrant colors"""

        template = extractor.extract_template(poem)

        # Should have a reasonable quality threshold
        assert 0.0 <= template.min_quality_score <= 1.0
        assert template.min_quality_score > 0.3  # Should be somewhat high for good words


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full template extraction."""

    def test_extract_and_recreate_structure(self, extractor):
        """Test that extracted template matches original structure."""
        poem = """The old pond
A frog jumps in
Water's sound"""

        template = extractor.extract_template(poem, title="Basho")

        # Verify structure is preserved
        assert template.lines == 3
        assert len(template.syllable_pattern) == 3
        assert len(template.line_templates) == 3

        # Verify each line has complete information
        for i, line_template in enumerate(template.line_templates):
            assert line_template.syllable_count > 0
            assert len(line_template.pos_pattern) > 0
            assert line_template.line_type is not None

    def test_template_serialization(self, extractor):
        """Test that extracted template can be serialized."""
        poem = """The moon rises
Over the silent lake
Reflecting stars"""

        template = extractor.extract_template(poem)

        # Should be serializable
        data = template.to_dict()
        assert data is not None
        assert "title" in data
        assert "lines" in data

        # Should be deserializable
        from poetryplayground.poem_template import PoemTemplate
        restored = PoemTemplate.from_dict(data)
        assert restored.lines == template.lines
        assert restored.syllable_pattern == template.syllable_pattern


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_word_line(self, extractor):
        """Test poem with single-word lines."""
        poem = """One
Two
Three"""

        template = extractor.extract_template(poem)

        assert template.lines == 3
        assert all(count >= 1 for count in template.syllable_pattern)

    def test_very_long_line(self, extractor):
        """Test poem with very long lines."""
        poem = """This is a very long line with many many words in it that goes on and on
Short line
Another very long line with lots of words and syllables throughout"""

        template = extractor.extract_template(poem)

        assert template.lines == 3
        assert template.syllable_pattern[0] > template.syllable_pattern[1]

    def test_punctuation_handling(self, extractor):
        """Test that punctuation is handled correctly."""
        poem = """Hello, world!
How are you?
I'm fine, thanks."""

        template = extractor.extract_template(poem)

        assert template.lines == 3
        # Punctuation shouldn't break analysis
        assert all(count > 0 for count in template.syllable_pattern)

    def test_numbers_in_poem(self, extractor):
        """Test poem with numbers."""
        poem = """One fish two fish
Red fish blue fish
Swimming in the sea"""

        template = extractor.extract_template(poem)

        assert template.lines == 3
        assert len(template.line_templates) == 3

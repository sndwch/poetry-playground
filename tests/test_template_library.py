"""Tests for template library system."""

import json
import pytest
from pathlib import Path

from poetryplayground.template_library import TemplateLibrary, get_template_library
from poetryplayground.poem_template import (
    PoemTemplate,
    LineTemplate,
    LineType,
    create_haiku_template,
)
from poetryplayground.core.quality_scorer import EmotionalTone, FormalityLevel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_templates_dir(tmp_path):
    """Create temporary directory for templates."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    return templates_dir


@pytest.fixture
def library(temp_templates_dir):
    """Create TemplateLibrary with temporary directory."""
    return TemplateLibrary(templates_dir=temp_templates_dir)


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return PoemTemplate(
        title="Test Template",
        source="test",
        author="pytest",
        lines=2,
        syllable_pattern=[5, 7],
        line_templates=[
            LineTemplate(5, ["DET", "NOUN", "VERB"], LineType.OPENING),
            LineTemplate(7, ["ADJ", "NOUN", "VERB", "ADV"], LineType.CLOSING),
        ],
        semantic_domains=["test", "sample"],
        metaphor_types=["SIMILE"],
        emotional_tone=EmotionalTone.NEUTRAL,
        min_quality_score=0.7,
    )


# ============================================================================
# Unit Tests - Initialization
# ============================================================================


class TestInitialization:
    """Test TemplateLibrary initialization."""

    def test_library_initialization(self, temp_templates_dir):
        """Test basic library initialization."""
        library = TemplateLibrary(templates_dir=temp_templates_dir)

        assert library.templates_dir == temp_templates_dir
        assert isinstance(library.templates, dict)
        assert len(library.templates) > 0  # Should have built-in templates

    def test_builtin_templates_loaded(self, library):
        """Test that built-in templates are loaded."""
        # Check for built-in templates
        assert "haiku" in library
        assert "haiku_dark" in library
        assert "haiku_nature" in library
        assert "tanka" in library
        assert "tanka_emotional" in library

        # Verify they're valid PoemTemplate objects
        haiku = library.get_template("haiku")
        assert isinstance(haiku, PoemTemplate)
        assert haiku.syllable_pattern == [5, 7, 5]

    def test_templates_directory_created(self, tmp_path):
        """Test that templates directory is created if it doesn't exist."""
        new_dir = tmp_path / "nonexistent" / "templates"
        assert not new_dir.exists()

        library = TemplateLibrary(templates_dir=new_dir)
        assert new_dir.exists()
        assert library.templates_dir == new_dir


# ============================================================================
# Unit Tests - Template Management
# ============================================================================


class TestTemplateManagement:
    """Test adding, removing, and retrieving templates."""

    def test_add_template_with_explicit_name(self, library, sample_template):
        """Test adding template with explicit name."""
        name = library.add_template(sample_template, name="my_test")

        assert name == "my_test"
        assert "my_test" in library
        assert library.get_template("my_test") == sample_template

    def test_add_template_auto_name(self, library, sample_template):
        """Test adding template with auto-generated name."""
        name = library.add_template(sample_template)

        # Should use sanitized title
        assert name == "test_template"
        assert name in library

    def test_add_template_name_sanitization(self, library):
        """Test that template names are properly sanitized."""
        template = PoemTemplate(
            title="Test! @#$ Template (Cool)",
            lines=1,
        )

        name = library.add_template(template)
        # Should replace special characters with underscores
        assert name == "test______template__cool_"

    def test_get_template_existing(self, library):
        """Test getting existing template."""
        haiku = library.get_template("haiku")
        assert haiku is not None
        assert isinstance(haiku, PoemTemplate)

    def test_get_template_nonexistent(self, library):
        """Test getting non-existent template returns None."""
        result = library.get_template("nonexistent")
        assert result is None

    def test_remove_template_existing(self, library, sample_template):
        """Test removing existing template."""
        library.add_template(sample_template, name="to_remove")
        assert "to_remove" in library

        result = library.remove_template("to_remove")
        assert result is True
        assert "to_remove" not in library

    def test_remove_template_nonexistent(self, library):
        """Test removing non-existent template."""
        result = library.remove_template("nonexistent")
        assert result is False

    def test_list_templates(self, library, sample_template):
        """Test listing all templates."""
        library.add_template(sample_template, name="test1")

        templates = library.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "test1" in templates
        assert "haiku" in templates

        # Should be sorted
        assert templates == sorted(templates)

    def test_contains_operator(self, library, sample_template):
        """Test 'in' operator for checking template existence."""
        library.add_template(sample_template, name="check_me")

        assert "check_me" in library
        assert "nonexistent" not in library

    def test_len_operator(self, library, sample_template):
        """Test len() operator."""
        initial_count = len(library)

        library.add_template(sample_template, name="new")
        assert len(library) == initial_count + 1

        library.remove_template("new")
        assert len(library) == initial_count


# ============================================================================
# Unit Tests - Persistence
# ============================================================================


class TestPersistence:
    """Test saving and loading templates to/from disk."""

    def test_save_template_new_file(self, library, sample_template, temp_templates_dir):
        """Test saving template to new file."""
        library.add_template(sample_template, name="save_test")

        result = library.save_template("save_test")
        assert result is True

        # Check file was created
        template_file = temp_templates_dir / "save_test.json"
        assert template_file.exists()

        # Verify content
        with open(template_file) as f:
            data = json.load(f)
            assert data["title"] == "Test Template"
            assert data["lines"] == 2

    def test_save_template_overwrite_protection(self, library, sample_template, temp_templates_dir):
        """Test that save without overwrite=True prevents overwriting."""
        library.add_template(sample_template, name="protect")
        library.save_template("protect")

        # Try to save again without overwrite
        with pytest.raises(FileExistsError, match="already exists"):
            library.save_template("protect", overwrite=False)

    def test_save_template_with_overwrite(self, library, sample_template, temp_templates_dir):
        """Test saving with overwrite=True."""
        library.add_template(sample_template, name="overwrite")
        library.save_template("overwrite")

        # Modify template
        sample_template.notes = "Modified"
        library.add_template(sample_template, name="overwrite")

        # Save with overwrite
        result = library.save_template("overwrite", overwrite=True)
        assert result is True

        # Verify modification was saved
        with open(temp_templates_dir / "overwrite.json") as f:
            data = json.load(f)
            assert data["notes"] == "Modified"

    def test_save_template_nonexistent(self, library):
        """Test saving non-existent template."""
        result = library.save_template("nonexistent")
        assert result is False

    def test_load_template_from_file(self, library, sample_template, temp_templates_dir):
        """Test loading template from external file."""
        # Create a JSON file manually
        template_file = temp_templates_dir / "external.json"
        with open(template_file, "w") as f:
            json.dump(sample_template.to_dict(), f)

        # Load it
        name = library.load_template(template_file)
        assert name == "external"
        assert "external" in library

        loaded = library.get_template("external")
        assert loaded.title == sample_template.title

    def test_load_template_invalid_json(self, library, temp_templates_dir):
        """Test loading template from invalid JSON file."""
        invalid_file = temp_templates_dir / "invalid.json"
        with open(invalid_file, "w") as f:
            f.write("not valid json{")

        name = library.load_template(invalid_file)
        assert name is None

    def test_load_user_templates_on_init(self, temp_templates_dir, sample_template):
        """Test that user templates are loaded on initialization."""
        # Save a template first
        template_file = temp_templates_dir / "user_template.json"
        with open(template_file, "w") as f:
            json.dump(sample_template.to_dict(), f)

        # Create new library (should load the template)
        library = TemplateLibrary(templates_dir=temp_templates_dir)

        assert "user_template" in library
        loaded = library.get_template("user_template")
        assert loaded.title == sample_template.title

    def test_export_template(self, library, sample_template, tmp_path):
        """Test exporting template to external location."""
        library.add_template(sample_template, name="export_me")

        export_path = tmp_path / "exported.json"
        result = library.export_template("export_me", export_path)

        assert result is True
        assert export_path.exists()

        # Verify content
        with open(export_path) as f:
            data = json.load(f)
            assert data["title"] == "Test Template"

    def test_export_template_nonexistent(self, library, tmp_path):
        """Test exporting non-existent template."""
        export_path = tmp_path / "nonexistent.json"
        result = library.export_template("nonexistent", export_path)
        assert result is False


# ============================================================================
# Unit Tests - Search and Filtering
# ============================================================================


class TestSearchAndFiltering:
    """Test searching and filtering templates."""

    def test_search_by_lines(self, library):
        """Test searching templates by line count."""
        results = library.search_templates(lines=3)

        assert "haiku" in results
        assert "haiku_dark" in results
        assert "tanka" not in results  # Tanka has 5 lines

    def test_search_by_syllable_pattern(self, library):
        """Test searching by exact syllable pattern."""
        results = library.search_templates(syllable_pattern=[5, 7, 5])

        assert "haiku" in results
        assert "haiku_dark" in results
        assert "haiku_nature" in results
        assert "tanka" not in results

    def test_search_by_semantic_domain(self, library):
        """Test searching by semantic domain."""
        results = library.search_templates(semantic_domain="nature")

        assert "haiku" in results
        assert "haiku_nature" in results
        assert "tanka" in results

    def test_search_by_emotional_tone(self, library):
        """Test searching by emotional tone."""
        results = library.search_templates(emotional_tone=EmotionalTone.DARK)

        assert "haiku_dark" in results
        assert "haiku" not in results  # Neutral tone

    def test_search_multiple_criteria(self, library):
        """Test searching with multiple criteria."""
        results = library.search_templates(
            lines=3, emotional_tone=EmotionalTone.NEUTRAL
        )

        assert "haiku" in results
        assert "haiku_nature" in results
        assert "haiku_dark" not in results  # Dark tone
        assert "tanka" not in results  # 5 lines

    def test_search_no_matches(self, library):
        """Test search with no matches."""
        results = library.search_templates(syllable_pattern=[99, 99, 99])
        assert results == []

    def test_get_templates_by_form(self, library):
        """Test getting templates by form type."""
        haiku_templates = library.get_templates_by_form("haiku")

        assert "haiku" in haiku_templates
        assert "haiku_dark" in haiku_templates
        assert "haiku_nature" in haiku_templates
        assert "tanka" not in haiku_templates

    def test_get_templates_by_form_case_insensitive(self, library):
        """Test form search is case-insensitive."""
        results1 = library.get_templates_by_form("HAIKU")
        results2 = library.get_templates_by_form("haiku")
        results3 = library.get_templates_by_form("Haiku")

        assert results1 == results2 == results3


# ============================================================================
# Unit Tests - Template Info
# ============================================================================


class TestTemplateInfo:
    """Test getting template information."""

    def test_get_template_info_existing(self, library):
        """Test getting info for existing template."""
        info = library.get_template_info("haiku")

        assert info is not None
        assert info["name"] == "haiku"
        assert info["title"] == "Haiku Template"
        assert info["lines"] == 3
        assert info["total_syllables"] == 17
        assert info["syllable_pattern"] == [5, 7, 5]
        assert "nature" in info["semantic_domains"]
        assert info["emotional_tone"] == "neutral"
        assert info["formality_level"] == "formal"

    def test_get_template_info_nonexistent(self, library):
        """Test getting info for non-existent template."""
        info = library.get_template_info("nonexistent")
        assert info is None

    def test_get_template_info_has_line_templates(self, library):
        """Test has_line_templates flag."""
        info = library.get_template_info("haiku")
        assert info["has_line_templates"] is True


# ============================================================================
# Unit Tests - String Representation
# ============================================================================


class TestStringRepresentation:
    """Test string representations."""

    def test_str_representation(self, library):
        """Test __str__ method."""
        str_repr = str(library)
        assert "TemplateLibrary" in str_repr
        assert "templates" in str_repr

    def test_repr_representation(self, library):
        """Test __repr__ method."""
        repr_str = repr(library)
        assert "TemplateLibrary" in repr_str
        assert "templates_dir" in repr_str
        assert "count" in repr_str


# ============================================================================
# Integration Tests - Singleton Pattern
# ============================================================================


class TestSingletonPattern:
    """Test global singleton instance."""

    def test_get_template_library_singleton(self):
        """Test that get_template_library returns singleton."""
        # Note: This test might be affected by other tests
        # In real usage, the singleton persists across calls
        lib1 = get_template_library()
        lib2 = get_template_library()

        # Should be the same instance
        assert lib1 is lib2

    def test_get_template_library_with_custom_dir(self, tmp_path):
        """Test that providing custom dir creates new instance."""
        custom_dir = tmp_path / "custom"
        lib = get_template_library(templates_dir=custom_dir)

        assert lib.templates_dir == custom_dir


# ============================================================================
# Integration Tests - Full Workflow
# ============================================================================


class TestFullWorkflow:
    """Test complete workflow scenarios."""

    def test_create_save_load_workflow(self, temp_templates_dir):
        """Test creating, saving, and loading template."""
        # Create library and template
        library1 = TemplateLibrary(templates_dir=temp_templates_dir)
        template = create_haiku_template(
            semantic_domains=["urban", "city"],
            emotional_tone=EmotionalTone.DARK,
        )
        template.title = "Urban Haiku"

        # Add and save
        name = library1.add_template(template, name="urban_haiku")
        library1.save_template(name)

        # Create new library instance (simulates restart)
        library2 = TemplateLibrary(templates_dir=temp_templates_dir)

        # Template should be loaded automatically
        assert "urban_haiku" in library2
        loaded = library2.get_template("urban_haiku")
        assert loaded.title == "Urban Haiku"
        assert "urban" in loaded.semantic_domains

    def test_search_and_use_workflow(self, library):
        """Test searching for and using templates."""
        # Search for nature-themed templates
        nature_templates = library.search_templates(semantic_domain="nature")
        assert len(nature_templates) > 0

        # Get one and verify it's usable
        template = library.get_template(nature_templates[0])
        assert template is not None
        assert template.lines > 0
        assert len(template.syllable_pattern) == template.lines

    def test_export_import_workflow(self, library, tmp_path):
        """Test exporting and importing template."""
        # Export built-in haiku
        export_path = tmp_path / "exported_haiku.json"
        library.export_template("haiku", export_path)

        # Import it with new name
        library.load_template(export_path)

        # Should now have it under filename
        assert "exported_haiku" in library

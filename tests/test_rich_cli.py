"""Tests for Rich CLI enhancements."""

import pytest


def test_list_procedures(capture_rich_output):
    """Test that --list-procedures displays a Rich table."""
    from generativepoetry.cli import _list_procedures

    # Call the function with our captured console
    _list_procedures()

    # Get the output
    output = capture_rich_output.export_text()

    # Verify table structure and content
    assert "Available Generation Procedures" in output
    assert "Futurist Poem" in output
    assert "Haiku Generator" in output
    assert "Metaphor Generator" in output
    assert "Visual Poetry" in output
    assert "Ideation" in output
    assert "Forms" in output

    # Verify all 16 procedures are listed
    for i in range(1, 17):
        assert str(i) in output


def test_list_fonts(capture_rich_output):
    """Test that --list-fonts displays fonts in Rich panels/columns."""
    from generativepoetry.cli import _list_fonts

    # Call the function with our captured console
    _list_fonts()

    # Get the output
    output = capture_rich_output.export_text()

    # Verify panel structure and content
    assert "Standard PostScript Fonts" in output

    # Verify standard fonts are listed
    assert "Courier" in output
    assert "Helvetica" in output
    assert "Times-Roman" in output

    # Verify note is displayed
    assert "reportlab.pdfbase.pdfmetrics" in output


def test_display_poem_output(capture_rich_output):
    """Test the display_poem_output helper function."""
    from generativepoetry.rich_output import display_poem_output

    poem_lines = [
        "ancient pond",
        "a frog jumps in—",
        "the sound of water",
    ]

    metadata = {"seed": "42", "form": "haiku", "syllables": "5-7-5"}

    display_poem_output(poem_lines, title="Test Haiku", metadata=metadata)

    output = capture_rich_output.export_text()

    # Verify poem content
    assert "ancient pond" in output
    assert "a frog jumps in" in output
    assert "the sound of water" in output

    # Verify title
    assert "Test Haiku" in output

    # Verify metadata
    assert "Seed: 42" in output
    assert "Form: haiku" in output
    assert "Syllables: 5-7-5" in output


def test_display_poem_output_no_metadata(capture_rich_output):
    """Test display_poem_output without metadata."""
    from generativepoetry.rich_output import display_poem_output

    poem_lines = ["line one", "line two", "line three"]

    display_poem_output(poem_lines, title="Simple Poem")

    output = capture_rich_output.export_text()

    # Verify poem content
    assert "line one" in output
    assert "line two" in output
    assert "line three" in output

    # Verify title
    assert "Simple Poem" in output


def test_display_error(capture_rich_output):
    """Test the display_error helper function."""
    from generativepoetry.rich_output import display_error

    display_error("Something went wrong", details="Check your input parameters")

    output = capture_rich_output.export_text()

    assert "Error" in output
    assert "Something went wrong" in output
    assert "Check your input parameters" in output


def test_display_success(capture_rich_output):
    """Test the display_success helper function."""
    from generativepoetry.rich_output import display_success

    display_success("Operation completed successfully!")

    output = capture_rich_output.export_text()

    assert "Success" in output
    assert "Operation completed successfully!" in output


def test_rich_console_theme():
    """Test that the global console has the custom theme."""
    from generativepoetry.rich_console import console, custom_theme

    # Verify console exists
    assert console is not None

    # Verify theme colors are defined
    assert "info" in custom_theme.styles
    assert "error" in custom_theme.styles
    assert "success" in custom_theme.styles


def test_status_spinner_does_not_crash(capture_rich_output):
    """Test that status spinners work without crashing."""
    # This tests that our Rich integration doesn't break
    # We can't easily test the actual spinner animation in tests

    with capture_rich_output.status("Testing..."):
        capture_rich_output.log("Step 1")
        capture_rich_output.log("Step 2")

    # Just verify it didn't crash
    output = capture_rich_output.export_text()
    assert "Step 1" in output
    assert "Step 2" in output

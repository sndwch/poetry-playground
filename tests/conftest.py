"""Pytest configuration and fixtures."""

import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_responses():
    """Mock API responses to avoid hitting real APIs in tests."""
    responses = {
        "rhyme": ["time", "chime", "lime"],
        "similar_sounding": ["test", "best", "rest"],
        "similar_meaning": ["exam", "quiz", "trial"],
        "frequently_following": ["the", "and", "of"],
    }
    return responses


@pytest.fixture(autouse=True)
def disable_network_calls():
    """Disable network calls in tests by default."""
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("Network calls disabled in tests")
        yield mock_get


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a sample text for testing.
    It has multiple sentences and paragraphs.

    This is the second paragraph.
    It also has multiple sentences.
    """


@pytest.fixture
def rich_console():
    """Console that captures output for testing.

    This console:
    - Writes to a StringIO buffer (can be read back)
    - Disables ANSI codes for easier testing
    - Sets fixed width for consistent output
    - Records output for export_text()
    """
    string_io = StringIO()
    return Console(
        file=string_io,
        force_terminal=False,  # Disable ANSI codes
        width=80,  # Fixed width for consistent output
        legacy_windows=False,
        record=True,  # Enable recording for export
    )


@pytest.fixture
def capture_rich_output(rich_console, monkeypatch):
    """Capture Rich output by temporarily replacing the global console.

    This fixture allows testing functions that use the global console
    from generativepoetry.rich_console.

    Returns the captured Console instance.
    """
    # Import modules first so they're loaded
    import generativepoetry.rich_console
    import generativepoetry.rich_output
    import generativepoetry.cli

    # Temporarily replace the global console in all modules that import it
    monkeypatch.setattr(generativepoetry.rich_console, "console", rich_console)
    monkeypatch.setattr(generativepoetry.rich_output, "console", rich_console)
    monkeypatch.setattr(generativepoetry.cli, "console", rich_console)

    return rich_console

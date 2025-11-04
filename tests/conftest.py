"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_responses():
    """Mock API responses to avoid hitting real APIs in tests."""
    responses = {
        'rhyme': ['time', 'chime', 'lime'],
        'similar_sounding': ['test', 'best', 'rest'],
        'similar_meaning': ['exam', 'quiz', 'trial'],
        'frequently_following': ['the', 'and', 'of'],
    }
    return responses


@pytest.fixture(autouse=True)
def disable_network_calls():
    """Disable network calls in tests by default."""
    with patch('requests.get') as mock_get:
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
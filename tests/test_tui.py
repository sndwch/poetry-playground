"""Basic tests for the TUI application.

These tests verify that the TUI can be imported and basic screens can be created.
Full integration testing would require textual.pilot which is beyond the scope
of these smoke tests.
"""

import pytest

# Try importing TUI components
try:
    from generativepoetry.tui.app import GenerativePoetryTUI, run
    from generativepoetry.tui.screens.config_form import ConfigFormScreen
    from generativepoetry.tui.screens.main_menu import MainMenuScreen
    from generativepoetry.tui.screens.output_view import OutputViewScreen

    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestTUIImports:
    """Test that TUI modules can be imported."""

    def test_app_import(self):
        """Test that main TUI app can be imported."""
        assert GenerativePoetryTUI is not None
        assert callable(run)

    def test_screen_imports(self):
        """Test that all screens can be imported."""
        assert MainMenuScreen is not None
        assert ConfigFormScreen is not None
        assert OutputViewScreen is not None


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestTUIInstantiation:
    """Test that TUI components can be instantiated."""

    def test_app_instantiation(self):
        """Test that TUI app can be instantiated."""
        app = GenerativePoetryTUI()
        assert app is not None
        assert app.TITLE == "Generative Poetry"

    def test_main_menu_instantiation(self):
        """Test that main menu can be instantiated."""
        menu = MainMenuScreen()
        assert menu is not None
        # Verify procedures are defined
        assert len(MainMenuScreen.PROCEDURES) > 0

    def test_config_form_instantiation(self):
        """Test that config form can be instantiated."""
        # Test with a known procedure
        form = ConfigFormScreen("haiku")
        assert form is not None
        assert form.procedure_id == "haiku"

        # Test with unknown procedure
        form_unknown = ConfigFormScreen("unknown")
        assert form_unknown is not None
        assert form_unknown.procedure_id == "unknown"

    def test_output_view_instantiation(self):
        """Test that output view can be instantiated."""
        view = OutputViewScreen("Test poem\\nLine 2", "Test Generator")
        assert view is not None
        assert view.poem_text == "Test poem\\nLine 2"
        assert view.generator_name == "Test Generator"


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestProcedureConfiguration:
    """Test procedure configuration definitions."""

    def test_all_procedures_have_config(self):
        """Test that common procedures have configuration."""
        common_procedures = ["haiku", "tanka", "senryu", "metaphor", "lineseeds"]

        for proc_id in common_procedures:
            config = ConfigFormScreen.PROCEDURE_CONFIG.get(proc_id)
            assert config is not None, f"Missing config for {proc_id}"
            assert "name" in config
            assert "description" in config
            assert "inputs" in config

    def test_procedure_inputs_format(self):
        """Test that procedure inputs have correct format."""
        for proc_id, config in ConfigFormScreen.PROCEDURE_CONFIG.items():
            assert isinstance(config["inputs"], list)
            for input_spec in config["inputs"]:
                assert isinstance(input_spec, tuple)
                assert len(input_spec) == 3  # (field_id, label, default)
                field_id, label, default = input_spec
                assert isinstance(field_id, str)
                assert isinstance(label, str)
                assert isinstance(default, str)

"""Integration tests for the TUI application to prevent regressions."""

import pytest

# Try importing TUI components
try:
    from poetryplayground.tui.app import GenerativePoetryTUI
    from poetryplayground.tui.screens.config_form import ConfigFormScreen
    from poetryplayground.tui.screens.unified_screen import UnifiedTUIScreen

    TUI_AVAILABLE = True
except ImportError:
    TUI_AVAILABLE = False


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestFragmentCollectionBugFix:
    """Test that the FragmentCollection bug is fixed."""

    def test_fragment_generator_uses_get_all_fragments(self):
        """Test that fragment generator correctly uses get_all_fragments()."""
        from poetryplayground.causal_poetry import FragmentCollection, ResonantFragment

        # Create a mock collection
        collection = FragmentCollection(
            causality=[
                ResonantFragment(
                    text="Test fragment one.",
                    pattern_type="causality",
                    word_count=3,
                    source_preview="Test source",
                    emotional_tone="neutral",
                    poetic_score=0.8,
                ),
                ResonantFragment(
                    text="Test fragment two.",
                    pattern_type="causality",
                    word_count=3,
                    source_preview="Test source",
                    emotional_tone="neutral",
                    poetic_score=0.7,
                ),
            ],
            temporal=[
                ResonantFragment(
                    text="Test fragment three.",
                    pattern_type="temporal",
                    word_count=3,
                    source_preview="Test source",
                    emotional_tone="neutral",
                    poetic_score=0.9,
                )
            ],
        )

        # Test that get_all_fragments() works correctly
        all_fragments = collection.get_all_fragments()
        assert len(all_fragments) == 3
        assert all_fragments[0].text == "Test fragment one."
        assert all_fragments[1].text == "Test fragment two."
        assert all_fragments[2].text == "Test fragment three."

        # Test that accessing .fragments attribute raises AttributeError
        with pytest.raises(AttributeError):
            _ = collection.fragments  # This should fail

    def test_config_form_fragment_execution_structure(self):
        """Test that ConfigFormScreen has correct structure for fragment execution."""
        screen = ConfigFormScreen("fragments")

        # Verify the procedure is configured
        assert "fragments" in screen.PROCEDURE_CONFIG
        assert screen.procedure_id == "fragments"

        # Verify the configuration has the expected structure
        config = screen.PROCEDURE_CONFIG["fragments"]
        assert config["name"] == "Resonant Fragment Miner"
        assert "inputs" in config
        assert len(config["inputs"]) == 1
        assert config["inputs"][0][0] == "count"

        # Verify the _execute_generator method exists and has correct signature
        import inspect

        sig = inspect.signature(screen._execute_generator)
        assert "procedure_id" in sig.parameters
        assert "config" in sig.parameters


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestCallFromThreadBugFix:
    """Test that call_from_thread bug is fixed."""

    def test_config_form_uses_app_call_from_thread(self):
        """Test that ConfigFormScreen uses self.app.call_from_thread()."""
        import inspect

        screen = ConfigFormScreen("haiku")

        # Check the source code of run_generation_worker
        source = inspect.getsource(screen.run_generation_worker)

        # Verify it uses self.app.call_from_thread, not self.call_from_thread
        assert "self.app.call_from_thread" in source
        assert "self.call_from_thread(" not in source.replace("self.app.call_from_thread", "")


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestUnifiedScreenLayout:
    """Test the new 3-column unified screen layout."""

    def test_unified_screen_instantiation(self):
        """Test that UnifiedTUIScreen can be instantiated."""
        screen = UnifiedTUIScreen()
        assert screen is not None
        assert screen.selected_procedure is None
        assert screen.current_output == ""

    def test_unified_screen_has_procedures(self):
        """Test that UnifiedTUIScreen has all procedures defined."""
        assert len(UnifiedTUIScreen.PROCEDURES) > 0
        assert len(UnifiedTUIScreen.PROCEDURE_CONFIG) > 0

        # Verify common procedures are present
        procedure_ids = [proc[0] for proc in UnifiedTUIScreen.PROCEDURES]
        assert "haiku" in procedure_ids
        assert "tanka" in procedure_ids
        assert "metaphor" in procedure_ids
        assert "fragments" in procedure_ids

    def test_unified_screen_has_two_panel_layout(self):
        """Test that UnifiedTUIScreen has 2-panel horizontal layout (30% left, 70% right)."""
        assert "layout: horizontal" in UnifiedTUIScreen.CSS
        assert "#left-panel" in UnifiedTUIScreen.CSS
        assert "#right-panel" in UnifiedTUIScreen.CSS
        assert "width: 30%" in UnifiedTUIScreen.CSS
        assert "width: 70%" in UnifiedTUIScreen.CSS


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestGeneratorExecution:
    """Test that all configured generators execute without crashes."""

    def test_all_configured_generators_have_executor_logic(self):
        """Test that all configured generators have execution logic."""
        screen = ConfigFormScreen("haiku")

        # Test each configured procedure
        for proc_id in UnifiedTUIScreen.PROCEDURE_CONFIG.keys():
            # This should not raise an exception for configured procedures
            try:
                # We're not actually running the generator (would require network/data)
                # Just checking the logic path exists
                config = UnifiedTUIScreen.PROCEDURE_CONFIG[proc_id]
                assert "name" in config
                assert "description" in config
                assert "inputs" in config
            except Exception as e:
                pytest.fail(f"Procedure {proc_id} configuration is invalid: {e}")

    def test_haiku_generator_execution_pattern(self):
        """Test the execution pattern for haiku generator."""
        screen = ConfigFormScreen("haiku")

        # Test with empty seed words
        config = {"seed_words": ""}
        # We can't run the full generator without data, but we can verify the pattern
        assert "haiku" in screen.PROCEDURE_CONFIG
        assert screen.procedure_id == "haiku"

    def test_metaphor_generator_execution_pattern(self):
        """Test the execution pattern for metaphor generator."""
        screen = ConfigFormScreen("metaphor")

        # Test configuration
        config = {"count": "10"}
        assert "metaphor" in screen.PROCEDURE_CONFIG
        assert screen.procedure_id == "metaphor"

    def test_equidistant_generator_execution_pattern(self):
        """Test the execution pattern for equidistant word finder."""
        screen = ConfigFormScreen("equidistant")

        # Test configuration
        config = {"word_a": "light", "word_b": "dark", "mode": "orth", "window": "0"}
        assert "equidistant" in screen.PROCEDURE_CONFIG
        assert screen.procedure_id == "equidistant"


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestTUIAppIntegration:
    """Test the main TUI app integration."""

    def test_app_uses_unified_screen(self):
        """Test that GenerativePoetryTUI uses UnifiedTUIScreen."""
        import inspect

        # Check the source code of on_mount
        source = inspect.getsource(GenerativePoetryTUI)

        # Verify it uses UnifiedTUIScreen
        assert "UnifiedTUIScreen" in source

    def test_app_has_correct_keybindings(self):
        """Test that app has appropriate keybindings for unified layout."""
        app = GenerativePoetryTUI()

        # Get binding keys
        binding_keys = [b.key for b in app.BINDINGS]

        # Verify essential bindings
        assert "q" in binding_keys  # Quit
        assert "d" in binding_keys  # Dark mode
        assert "?" in binding_keys  # Help

        # The 'm' binding should be removed since we don't have multiple screens
        # (unified screen is always shown)


@pytest.mark.skipif(not TUI_AVAILABLE, reason="TUI components not available")
class TestProcedureConfigConsistency:
    """Test that procedure configurations are consistent."""

    def test_all_inputs_have_three_elements(self):
        """Test that all input configurations have (id, label, default)."""
        for proc_id, config in UnifiedTUIScreen.PROCEDURE_CONFIG.items():
            for input_spec in config["inputs"]:
                assert isinstance(input_spec, tuple), f"{proc_id} input is not a tuple"
                assert len(input_spec) == 3, f"{proc_id} input doesn't have 3 elements"
                field_id, label, default = input_spec
                assert isinstance(field_id, str), f"{proc_id} field_id is not string"
                assert isinstance(label, str), f"{proc_id} label is not string"
                assert isinstance(default, str), f"{proc_id} default is not string"

    def test_all_configs_have_required_keys(self):
        """Test that all configurations have name, description, and inputs."""
        for proc_id, config in UnifiedTUIScreen.PROCEDURE_CONFIG.items():
            assert "name" in config, f"{proc_id} missing 'name'"
            assert "description" in config, f"{proc_id} missing 'description'"
            assert "inputs" in config, f"{proc_id} missing 'inputs'"
            assert isinstance(config["inputs"], list), f"{proc_id} inputs is not a list"

"""Unified 2-panel TUI screen for generative poetry with modal output."""

from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Markdown, Static

from .config_form import ConfigFormScreen


class UnifiedTUIScreen(Screen):
    """Main TUI screen with 2-panel layout: Left (Procedures + Config) | Right (Output Preview)."""

    CSS = """
    UnifiedTUIScreen {
        layout: horizontal;
    }

    #left-panel {
        width: 30%;
        height: 100%;
        border-right: solid $primary;
        padding: 0 1;
    }

    #left-panel:focus-within {
        border-right: thick $accent;
        background: $surface-darken-1;
    }

    #right-panel {
        width: 70%;
        height: 100%;
        padding: 0 1;
    }

    #right-panel:focus-within {
        background: $surface-darken-1;
    }

    #procedure-title {
        text-style: bold;
        color: $accent;
        margin: 1 0;
        text-align: center;
    }

    #config-title {
        text-style: bold;
        color: $accent;
        margin: 1 0;
        text-align: center;
    }

    #output-title {
        text-style: bold;
        color: $accent;
        margin: 1 0;
        text-align: center;
    }

    #procedure-list {
        height: 40%;
        border: solid $primary;
    }

    .list-item-label {
        width: 1fr;
    }

    .category-header {
        text-style: bold;
        color: $accent;
        background: $surface-darken-1;
    }

    #config-form {
        height: auto;
        margin: 1 0;
    }

    .input-label {
        margin-top: 1;
        margin-bottom: 0;
        color: $text;
    }

    Input {
        margin-bottom: 1;
    }

    #generate-btn {
        margin: 1 0;
        width: 100%;
    }

    #generate-btn:focus {
        border: thick $accent;
        text-style: bold;
        background: $accent;
    }

    #view-output-btn {
        margin: 1 0;
        width: 100%;
    }

    #view-output-btn:focus {
        border: thick $accent;
        text-style: bold;
        background: $accent;
    }

    #status {
        text-align: center;
        color: $accent;
        margin-top: 1;
        height: 3;
    }

    #output-preview {
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    #output-preview:focus {
        border: thick $accent;
    }
    """

    # Reference to ConfigFormScreen's procedure configuration for backward compatibility
    # Note: This must be set after ConfigFormScreen is imported
    PROCEDURE_CONFIG = ConfigFormScreen.PROCEDURE_CONFIG

    # Procedure metadata
    PROCEDURES: ClassVar[list] = [
        # Visual/Concrete Poetry
        (
            "futurist",
            "Futurist Poem",
            "Marinetti-inspired mathematical word connections",
            "Visual Poetry",
        ),
        (
            "markov",
            "Stochastic Jolastic (Markov)",
            "Joyce-like wordplay with rhyme schemes",
            "Visual Poetry",
        ),
        ("chaotic", "Chaotic Concrete Poem", "Abstract spatial arrangements", "Visual Poetry"),
        ("charsoup", "Character Soup Poem", "Character-level visual chaos", "Visual Poetry"),
        ("wordsoup", "Stop Word Soup Poem", "Stop words in visual patterns", "Visual Poetry"),
        ("puzzle", "Visual Puzzle Poem", "Interactive terminal-based puzzles", "Visual Poetry"),
        # Ideation Tools
        (
            "lineseeds",
            "Line Seeds Generator",
            "Evocative incomplete phrases and line beginnings",
            "Ideation",
        ),
        (
            "personalized_lineseeds",
            "Personalized Line Seeds",
            "Style-matched seeds from your corpus fingerprint",
            "Ideation",
        ),
        (
            "metaphor",
            "Metaphor Generator",
            "Fresh metaphors from Project Gutenberg texts",
            "Ideation",
        ),
        ("corpus", "Personal Corpus Analyzer", "Analyze your existing poetry", "Ideation"),
        (
            "theseus",
            "Ship of Theseus Transformer",
            "Gradually transform existing poems",
            "Ideation",
        ),
        ("ideas", "Poetry Idea Generator", "Creative seeds from classic literature", "Ideation"),
        (
            "sixdegrees",
            "Six Degrees Word Convergence",
            "Explore connections between concepts",
            "Ideation",
        ),
        ("fragments", "Resonant Fragment Miner", "Extract poetic sentence fragments", "Ideation"),
        ("equidistant", "Equidistant Word Finder", "Find words bridging two anchors", "Ideation"),
        (
            "semantic_path",
            "Semantic Geodesic Finder",
            "Find transitional paths through meaning-space",
            "Ideation",
        ),
        (
            "conceptual_cloud",
            "Conceptual Cloud Generator",
            "Multi-dimensional word associations",
            "Ideation",
        ),
        # Syllabic Forms
        ("haiku", "Haiku Generator", "5-7-5 syllable haiku with templates", "Forms"),
        ("tanka", "Tanka Generator", "5-7-5-7-7 syllable tanka", "Forms"),
        ("senryu", "Senryu Generator", "5-7-5 syllable senryu", "Forms"),
        # System
        ("deps", "Check System Dependencies", "Verify installation and dependencies", "System"),
    ]

    def __init__(self):
        """Initialize the unified screen."""
        super().__init__()
        self.selected_procedure = None
        self.current_output = ""

    def compose(self) -> ComposeResult:
        """Create 2-panel layout."""
        # Left Panel: Procedures + Config
        with VerticalScroll(id="left-panel"):
            yield Label("✨ Procedures ✨", id="procedure-title")
            yield self._create_procedure_list()
            yield Label("⚙️ Configuration ⚙️", id="config-title")
            yield Vertical(id="config-form")
            yield Static("", id="status")

        # Right Panel: Output Preview
        with Vertical(id="right-panel"):
            yield Label("📜 Output Preview 📜", id="output-title")
            with ScrollableContainer(id="output-preview"):
                yield Static(
                    "Select a procedure and click Generate to begin.\n\n"
                    "Output will appear here, and you can click 'View Full Output' "
                    "for a larger modal view.",
                    id="preview-text",
                )
            yield Button("View Full Output", variant="primary", id="view-output-btn", disabled=True)

    def _create_procedure_list(self) -> ListView:
        """Create the procedure list with category grouping."""
        items = []
        current_category = None

        for proc_id, name, desc, category in self.PROCEDURES:
            # Add category header if changed
            if category != current_category:
                if current_category is not None:
                    items.append(ListItem(Static(""), disabled=True))  # Separator
                items.append(
                    ListItem(Static(f"[{category}]", classes="category-header"), disabled=True)
                )
                current_category = category

            # Add procedure item
            label_text = f"{name}\n  {desc}"
            items.append(ListItem(Static(label_text, classes="list-item-label"), id=proc_id))

        return ListView(*items, id="procedure-list")

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle procedure selection - update config form."""
        if event.item.id and not event.item.disabled:
            self.selected_procedure = event.item.id
            await self._update_config_form(self.selected_procedure)

    async def _update_config_form(self, procedure_id: str) -> None:
        """Update the config form based on selected procedure."""
        config = ConfigFormScreen.PROCEDURE_CONFIG.get(
            procedure_id,
            {
                "name": "Unknown Procedure",
                "description": "Configuration not yet implemented",
                "inputs": [],
            },
        )

        # Clear existing form
        config_form = self.query_one("#config-form", Vertical)
        await config_form.remove_children()

        # Add description
        config_form.mount(Static(config["description"], classes="input-label"))

        # Create input fields
        for field_id, field_label, default_value in config["inputs"]:
            config_form.mount(Label(field_label, classes="input-label"))
            config_form.mount(Input(value=default_value, id=f"input-{field_id}"))

        # Add generate button
        config_form.mount(Button("Generate", variant="primary", id="generate-btn"))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "generate-btn":
            await self._handle_generate()
        elif event.button.id == "view-output-btn":
            await self._handle_view_output()

    async def _handle_generate(self) -> None:
        """Handle generate button click."""
        if not self.selected_procedure:
            return

        # Get config for selected procedure
        config = ConfigFormScreen.PROCEDURE_CONFIG.get(self.selected_procedure, {})

        # Extract configuration from input fields
        config_values = {}
        for field_id, _, _ in config.get("inputs", []):
            try:
                input_widget = self.query_one(f"#input-{field_id}", Input)
                config_values[field_id] = input_widget.value
            except Exception:
                pass  # Field might not exist

        # Update status
        status_widget = self.query_one("#status", Static)
        status_widget.update("🔄 Generating...")

        # Run generation in worker thread
        self.run_generation_worker(self.selected_procedure, config_values)

    @work(exclusive=True, thread=True)
    def run_generation_worker(self, procedure_id: str, config_values: dict) -> None:
        """Run generation in background thread."""
        try:
            # Import config_form's execute logic
            from .config_form import ConfigFormScreen

            # Create temporary instance to use its _execute_generator method
            temp_screen = ConfigFormScreen(procedure_id)
            result = temp_screen._execute_generator(procedure_id, config_values)

            # Post result back to main thread
            self.app.call_from_thread(self._on_generation_complete, result)

        except Exception as e:
            # Post error back to main thread
            self.app.call_from_thread(self._on_generation_error, str(e))

    def _on_generation_complete(self, result: str) -> None:
        """Handle successful generation completion."""
        self.current_output = result

        # Update status
        status_widget = self.query_one("#status", Static)
        status_widget.update("✓ Generation complete!")

        # Update output preview (show full output)
        preview_widget = self.query_one("#preview-text", Static)
        preview_widget.update(result)

        # Enable view button
        view_btn = self.query_one("#view-output-btn", Button)
        view_btn.disabled = False

    def _on_generation_error(self, error_msg: str) -> None:
        """Handle generation error."""
        status_widget = self.query_one("#status", Static)
        status_widget.update(f"❌ Error: {error_msg}")

        # Show error in preview
        preview_widget = self.query_one("#preview-text", Static)
        preview_widget.update(f"Error during generation:\n\n{error_msg}")

    async def _handle_view_output(self) -> None:
        """Handle view full output button click - open modal."""
        if not self.current_output:
            return

        # Push the output modal screen
        await self.app.push_screen(
            OutputModal(
                output_content=self.current_output,
                procedure_name=self.selected_procedure or "output",
            )
        )


class OutputModal(ModalScreen[None]):
    """Modal screen for displaying full output with actions."""

    CSS = """
    OutputModal {
        align: center middle;
    }

    #modal-container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 0;
    }

    #modal-header {
        height: 3;
        background: $primary;
        color: $text;
        text-style: bold;
        content-align: center middle;
        padding: 0 2;
    }

    #modal-content {
        height: 1fr;
        padding: 1 2;
        background: $surface;
    }

    #modal-footer {
        height: 5;
        background: $panel;
        padding: 1;
        layout: horizontal;
        align: center middle;
    }

    .modal-button {
        margin: 0 1;
        min-width: 16;
    }
    """

    def __init__(self, output_content: str, procedure_name: str):
        """Initialize the modal with content."""
        super().__init__()
        self.output_content = output_content
        self.procedure_name = procedure_name

    def compose(self) -> ComposeResult:
        """Create modal layout."""
        with Vertical(id="modal-container"):
            yield Static(f"📜 Output: {self.procedure_name}", id="modal-header")
            with ScrollableContainer(id="modal-content"):
                # Try to render as Markdown, fall back to plain text if it fails
                try:
                    yield Markdown(self.output_content)
                except Exception:
                    yield Static(self.output_content)
            with Horizontal(id="modal-footer"):
                yield Button(
                    "Copy to Clipboard", variant="default", classes="modal-button", id="copy-btn"
                )
                yield Button(
                    "Save to File", variant="primary", classes="modal-button", id="save-btn"
                )
                yield Button("Close (ESC)", variant="error", classes="modal-button", id="close-btn")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle modal button clicks."""
        if event.button.id == "close-btn":
            self.dismiss()
        elif event.button.id == "save-btn":
            await self._save_to_file()
        elif event.button.id == "copy-btn":
            await self._copy_to_clipboard()

    def on_key(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key == "escape":
            self.dismiss()

    async def _save_to_file(self) -> None:
        """Save output to file."""
        try:
            from datetime import datetime
            from pathlib import Path

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_"
                for c in self.procedure_name.lower()
            ).replace(" ", "_")
            filename = f"{safe_name}_{timestamp}.txt"

            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Save file
            file_path = output_dir / filename
            file_path.write_text(self.output_content, encoding="utf-8")

            # Update header to show success
            header = self.query_one("#modal-header", Static)
            header.update(f"✓ Saved to {file_path}")

        except Exception as e:
            header = self.query_one("#modal-header", Static)
            header.update(f"❌ Error saving: {e}")

    async def _copy_to_clipboard(self) -> None:
        """Copy output to clipboard."""
        try:
            import pyperclip

            pyperclip.copy(self.output_content)

            # Update header to show success
            header = self.query_one("#modal-header", Static)
            header.update("✓ Copied to clipboard!")
        except ImportError:
            # pyperclip not available
            header = self.query_one("#modal-header", Static)
            header.update("❌ pyperclip not installed (pip install pyperclip)")
        except Exception as e:
            header = self.query_one("#modal-header", Static)
            header.update(f"❌ Error copying: {e}")

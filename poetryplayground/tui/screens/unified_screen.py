"""Unified 3-column TUI screen for generative poetry."""

from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static, TextArea


class UnifiedTUIScreen(Screen):
    """Main TUI screen with 3-column layout: List | Config | Output."""

    CSS = """
    UnifiedTUIScreen {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 2fr;
    }

    #procedure-column {
        width: 100%;
        height: 100%;
        border-right: solid $primary;
        padding: 0 1;
    }

    #procedure-column:focus-within {
        border-right: thick $accent;
        background: $surface-darken-1;
    }

    #config-column {
        width: 100%;
        height: 100%;
        border-right: solid $primary;
        padding: 0 1;
    }

    #config-column:focus-within {
        border-right: thick $accent;
        background: $surface-darken-1;
    }

    #output-column {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    #output-column:focus-within {
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
        height: 100%;
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

    #status {
        text-align: center;
        color: $accent;
        margin-top: 1;
        height: 3;
    }

    #output-display {
        height: 100%;
        border: solid $primary;
    }

    #output-display:focus {
        border: thick $accent;
    }

    #save-btn {
        margin: 1 0;
        width: 100%;
    }

    #save-btn:focus {
        border: thick $accent;
        text-style: bold;
        background: $accent;
    }
    """

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

    # Configuration for each procedure
    PROCEDURE_CONFIG: ClassVar[dict] = {
        "haiku": {
            "name": "Haiku Generator",
            "description": "Generate 5-7-5 syllable haiku with grammatical templates",
            "inputs": [("seed_words", "Seed Words (optional, comma-separated)", "")],
        },
        "tanka": {
            "name": "Tanka Generator",
            "description": "Generate 5-7-5-7-7 syllable tanka",
            "inputs": [("seed_words", "Seed Words (optional, comma-separated)", "")],
        },
        "senryu": {
            "name": "Senryu Generator",
            "description": "Generate 5-7-5 syllable senryu focused on human nature",
            "inputs": [("seed_words", "Seed Words (optional, comma-separated)", "")],
        },
        "metaphor": {
            "name": "Metaphor Generator",
            "description": "Generate fresh metaphors from Project Gutenberg texts",
            "inputs": [("count", "Number of metaphors", "10")],
        },
        "lineseeds": {
            "name": "Line Seeds Generator",
            "description": "Generate evocative incomplete phrases and line beginnings",
            "inputs": [
                ("seed_words", "Seed Words (space or comma-separated)", ""),
                ("count", "Number of seeds", "10"),
            ],
        },
        "ideas": {
            "name": "Poetry Idea Generator",
            "description": "Mine creative seeds from classic literature",
            "inputs": [("count", "Number of ideas", "10")],
        },
        "fragments": {
            "name": "Resonant Fragment Miner",
            "description": "Extract poetic sentence fragments from literature",
            "inputs": [("count", "Number of fragments", "50")],
        },
        "equidistant": {
            "name": "Equidistant Word Finder",
            "description": "Find words equidistant from two anchor words",
            "inputs": [
                ("word_a", "First anchor word", ""),
                ("word_b", "Second anchor word", ""),
                ("mode", "Mode (orth/phono)", "orth"),
                ("window", "Window (distance tolerance)", "0"),
            ],
        },
        "semantic_path": {
            "name": "Semantic Geodesic Finder",
            "description": "Find semantic paths between words through meaning-space",
            "inputs": [
                ("start_word", "Start word", ""),
                ("end_word", "End word", ""),
                ("steps", "Number of steps", "5"),
                ("alternatives", "Alternatives per step", "3"),
                ("method", "Method (linear/bezier/shortest)", "linear"),
            ],
        },
        "conceptual_cloud": {
            "name": "Conceptual Cloud Generator",
            "description": "Generate multi-dimensional word associations (poet's radar)",
            "inputs": [
                ("center_word", "Center word or phrase", ""),
                ("k_per_cluster", "Words per cluster", "10"),
                (
                    "sections",
                    "Sections (all or comma-separated: semantic,contextual,opposite,phonetic,imagery,rare)",
                    "all",
                ),
                ("output_format", "Format (rich/json/markdown/simple)", "rich"),
            ],
        },
    }

    def __init__(self):
        """Initialize the unified screen."""
        super().__init__()
        self.selected_procedure = None
        self.current_output = ""

    def compose(self) -> ComposeResult:
        """Create 3-column layout."""
        # Column 1: Procedure List
        with Vertical(id="procedure-column"):
            yield Label("✨ Procedures ✨", id="procedure-title")
            yield self._create_procedure_list()

        # Column 2: Config Form
        with Vertical(id="config-column"):
            yield Label("⚙️ Configuration ⚙️", id="config-title")
            yield Vertical(id="config-form")
            yield Static("", id="status")

        # Column 3: Output Display
        with Vertical(id="output-column"):
            yield Label("📜 Output 📜", id="output-title")
            yield TextArea(
                "Select a procedure and click Generate to begin.",
                read_only=True,
                id="output-display",
            )
            yield Button("Save to File", variant="primary", id="save-btn", disabled=True)

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
        config = self.PROCEDURE_CONFIG.get(
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
        elif event.button.id == "save-btn":
            await self._handle_save()

    async def _handle_generate(self) -> None:
        """Handle generate button click."""
        if not self.selected_procedure:
            return

        # Get config for selected procedure
        config = self.PROCEDURE_CONFIG.get(self.selected_procedure, {})

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

        # Update output display
        output_display = self.query_one("#output-display", TextArea)
        output_display.load_text(result)

        # Enable save button
        save_btn = self.query_one("#save-btn", Button)
        save_btn.disabled = False

    def _on_generation_error(self, error_msg: str) -> None:
        """Handle generation error."""
        status_widget = self.query_one("#status", Static)
        status_widget.update(f"❌ Error: {error_msg}")

        # Show error in output
        output_display = self.query_one("#output-display", TextArea)
        output_display.load_text(f"Error during generation:\n\n{error_msg}")

    async def _handle_save(self) -> None:
        """Handle save button click."""
        if not self.current_output:
            return

        try:
            from datetime import datetime
            from pathlib import Path

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            procedure_name = self.selected_procedure or "output"
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in procedure_name.lower()
            ).replace(" ", "_")
            filename = f"{safe_name}_{timestamp}.txt"

            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Save file
            file_path = output_dir / filename
            file_path.write_text(self.current_output, encoding="utf-8")

            # Update status
            status_widget = self.query_one("#status", Static)
            status_widget.update(f"✓ Saved to {file_path}")

        except Exception as e:
            status_widget = self.query_one("#status", Static)
            status_widget.update(f"❌ Error saving: {e}")

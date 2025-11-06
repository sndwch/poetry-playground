"""Configuration form screen for generation procedures."""

from typing import ClassVar

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Static


class ConfigFormScreen(Screen):
    """Configuration and execution screen for a specific generation procedure."""

    CSS = """
    ConfigFormScreen {
        align: center middle;
    }

    #config-container {
        width: 80;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1 2;
    }

    #title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1 0;
    }

    #description {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #form-content {
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

    #button-row {
        align: center middle;
        margin-top: 1;
    }

    #generate-btn {
        margin: 0 1;
    }

    #back-btn {
        margin: 0 1;
    }

    #status {
        text-align: center;
        color: $accent;
        margin-top: 1;
        height: 3;
    }
    """

    # Procedure metadata and configuration
    PROCEDURE_CONFIG: ClassVar[dict] = {
        "haiku": {
            "name": "Haiku Generator",
            "description": "Generate 5-7-5 syllable haiku with grammatical templates",
            "inputs": [
                ("seed_words", "Seed Words (optional, comma-separated)", ""),
            ],
        },
        "tanka": {
            "name": "Tanka Generator",
            "description": "Generate 5-7-5-7-7 syllable tanka",
            "inputs": [
                ("seed_words", "Seed Words (optional, comma-separated)", ""),
            ],
        },
        "senryu": {
            "name": "Senryu Generator",
            "description": "Generate 5-7-5 syllable senryu focused on human nature",
            "inputs": [
                ("seed_words", "Seed Words (optional, comma-separated)", ""),
            ],
        },
        "metaphor": {
            "name": "Metaphor Generator",
            "description": "Generate fresh metaphors from Project Gutenberg texts",
            "inputs": [
                ("count", "Number of metaphors", "10"),
            ],
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
            "inputs": [
                ("count", "Number of ideas", "10"),
            ],
        },
        "fragments": {
            "name": "Resonant Fragment Miner",
            "description": "Extract poetic sentence fragments from literature",
            "inputs": [
                ("count", "Number of fragments", "50"),
            ],
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
    }

    def __init__(self, procedure_id: str):
        """Initialize config form for a specific procedure.

        Args:
            procedure_id: ID of the generation procedure to configure
        """
        super().__init__()
        self.procedure_id = procedure_id
        self.config = self.PROCEDURE_CONFIG.get(
            procedure_id,
            {
                "name": "Unknown Procedure",
                "description": "Configuration not yet implemented",
                "inputs": [],
            },
        )

    def compose(self) -> ComposeResult:
        """Create form widgets."""
        with Container(id="config-container"):
            yield Label(self.config["name"], id="title")
            yield Label(self.config["description"], id="description")

            with Vertical(id="form-content"):
                # Create input fields based on configuration
                for field_id, field_label, default_value in self.config["inputs"]:
                    yield Label(field_label, classes="input-label")
                    yield Input(value=default_value, id=f"input-{field_id}")

            with Horizontal(id="button-row"):
                yield Button("Generate", variant="primary", id="generate-btn")
                yield Button("Back to Menu", variant="default", id="back-btn")

            yield Static("", id="status")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "generate-btn":
            await self._handle_generate()
        elif event.button.id == "back-btn":
            await self._handle_back()

    async def _handle_generate(self) -> None:
        """Handle generate button click."""
        # Extract configuration from input fields
        config_values = {}
        for field_id, _, _ in self.config["inputs"]:
            input_widget = self.query_one(f"#input-{field_id}", Input)
            config_values[field_id] = input_widget.value

        # Update status
        status_widget = self.query_one("#status", Static)
        status_widget.update("🔄 Generating...")

        # Run generation in worker thread
        self.run_generation_worker(self.procedure_id, config_values)

    @work(exclusive=True, thread=True)
    def run_generation_worker(self, procedure_id: str, config_values: dict) -> None:
        """Run generation in background thread.

        Args:
            procedure_id: ID of the procedure to run
            config_values: Configuration values from form
        """
        try:
            # Import generators here to avoid loading everything on startup
            result = self._execute_generator(procedure_id, config_values)

            # Post result back to main thread
            self.app.call_from_thread(self._on_generation_complete, result)

        except Exception as e:
            # Post error back to main thread
            self.app.call_from_thread(self._on_generation_error, str(e))

    def _execute_generator(self, procedure_id: str, config: dict) -> str:
        """Execute the appropriate generator based on procedure ID.

        Args:
            procedure_id: ID of the procedure to run
            config: Configuration dictionary

        Returns:
            Generated text output

        Raises:
            ValueError: If procedure ID is not recognized
        """
        if procedure_id in ["haiku", "tanka", "senryu"]:
            from generativepoetry.forms import FormGenerator

            generator = FormGenerator()
            seed_words = None
            if config.get("seed_words"):
                seed_words = [
                    w.strip() for w in config["seed_words"].replace(",", " ").split() if w.strip()
                ]

            if procedure_id == "haiku":
                lines, validation = generator.generate_haiku(seed_words=seed_words, strict=True)
            elif procedure_id == "tanka":
                lines, validation = generator.generate_tanka(seed_words=seed_words, strict=True)
            else:  # senryu
                lines, validation = generator.generate_senryu(seed_words=seed_words, strict=True)

            result = "\n".join(lines)
            if validation.valid:
                result += f"\n\n✓ Valid {procedure_id} ({validation.syllable_pattern})"
            else:
                result += "\n\n" + validation.get_report()

            return result

        elif procedure_id == "metaphor":
            from generativepoetry.metaphor_generator import MetaphorGenerator

            count = int(config.get("count", 10))
            generator = MetaphorGenerator()
            metaphors = generator.generate_metaphors(target_count=count, num_texts=5)

            return "\n\n".join(f"{i + 1}. {m}" for i, m in enumerate(metaphors[:count]))

        elif procedure_id == "lineseeds":
            from generativepoetry.line_seeds import LineSeedGenerator

            seed_words_str = config.get("seed_words", "")
            seed_words = (
                [w.strip() for w in seed_words_str.replace(",", " ").split() if w.strip()]
                if seed_words_str
                else None
            )
            count = int(config.get("count", 10))

            generator = LineSeedGenerator()
            seeds = generator.generate_line_seeds(seed_words=seed_words, count=count)

            return "\n\n".join(f"{i + 1}. {seed}" for i, seed in enumerate(seeds))

        elif procedure_id == "ideas":
            from generativepoetry.idea_generator import IdeaGenerator

            count = int(config.get("count", 10))
            generator = IdeaGenerator()
            ideas = generator.mine_poetry_ideas(target_count=count, num_texts=5)

            result_lines = []
            for i, idea in enumerate(ideas[:count], 1):
                result_lines.append(f"{i}. [{idea.category}] {idea.text}")
                if idea.source_info:
                    result_lines.append(f"   Source: {idea.source_info}")
                result_lines.append("")

            return "\n".join(result_lines)

        elif procedure_id == "fragments":
            from generativepoetry.causal_poetry import ResonantFragmentMiner

            count = int(config.get("count", 50))
            miner = ResonantFragmentMiner()
            collection = miner.mine_fragments(target_count=count, num_texts=5)

            result_lines = []
            for i, frag in enumerate(collection.get_all_fragments()[:count], 1):
                result_lines.append(f"{i}. {frag.text}")
                result_lines.append(
                    f"   Pattern: {frag.pattern_type} | Score: {frag.poetic_score:.2f}"
                )
                result_lines.append("")

            return "\n".join(result_lines)

        elif procedure_id == "equidistant":
            from generativepoetry.finders import find_equidistant

            word_a = config.get("word_a", "").strip()
            word_b = config.get("word_b", "").strip()

            if not word_a or not word_b:
                return "Error: Both anchor words are required"

            mode = config.get("mode", "orth")
            if mode not in ["orth", "phono"]:
                mode = "orth"

            try:
                window = int(config.get("window", 0))
            except ValueError:
                window = 0

            hits = find_equidistant(a=word_a, b=word_b, mode=mode, window=window)

            if not hits:
                return f"No equidistant words found for '{word_a}' and '{word_b}'"

            result_lines = [f"Found {len(hits)} equidistant words:\n"]
            for i, hit in enumerate(hits[:20], 1):
                syl_str = f"{hit.syllables}s" if hit.syllables else "?"
                pos_str = hit.pos if hit.pos else "?"
                result_lines.append(
                    f"{i:2d}. {hit.word:15s} (d={hit.dist_a}/{hit.dist_b}, "
                    f"{syl_str}, {pos_str}, score={hit.score:.2f})"
                )

            if len(hits) > 20:
                result_lines.append(f"\n... and {len(hits) - 20} more results")

            return "\n".join(result_lines)

        else:
            return f"Generator for '{procedure_id}' not yet implemented in TUI.\n\nUse the CLI interface for now:\n  generative-poetry-cli"

    def _on_generation_complete(self, result: str) -> None:
        """Handle successful generation completion (main thread).

        Args:
            result: Generated text output
        """
        # Update status
        status_widget = self.query_one("#status", Static)
        status_widget.update("✓ Generation complete!")

        # Show output screen
        from generativepoetry.tui.screens.output_view import OutputViewScreen

        self.app.push_screen(OutputViewScreen(result, self.config["name"]))

    def _on_generation_error(self, error_msg: str) -> None:
        """Handle generation error (main thread).

        Args:
            error_msg: Error message to display
        """
        status_widget = self.query_one("#status", Static)
        status_widget.update(f"❌ Error: {error_msg}")

    async def _handle_back(self) -> None:
        """Handle back button click."""
        self.app.pop_screen()

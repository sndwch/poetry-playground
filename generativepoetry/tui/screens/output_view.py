"""Output view screen for displaying generated poetry."""

from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, Static, TextArea


class OutputViewScreen(Screen):
    """Display generated poem with save/copy options."""

    CSS = """
    OutputViewScreen {
        align: center middle;
    }

    #output-container {
        width: 90;
        height: auto;
        max-height: 90%;
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

    #poem-display {
        height: 1fr;
        min-height: 15;
        max-height: 30;
        border: solid $primary-lighten-1;
        margin: 1 0;
        background: $surface-darken-1;
    }

    #button-row {
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }

    #status-message {
        text-align: center;
        color: $success;
        margin-top: 1;
        height: 1;
    }
    """

    def __init__(self, poem_text: str, generator_name: str):
        """Initialize output view.

        Args:
            poem_text: Generated poem or text to display
            generator_name: Name of the generator that produced this output
        """
        super().__init__()
        self.poem_text = poem_text
        self.generator_name = generator_name

    def compose(self) -> ComposeResult:
        """Create output display widgets."""
        with Container(id="output-container"):
            yield Static(f"✨ {self.generator_name} ✨", id="title")
            yield TextArea(self.poem_text, read_only=True, id="poem-display")

            with Horizontal(id="button-row"):
                yield Button("Save to File", variant="primary", id="save-btn")
                yield Button("Generate Another", variant="default", id="again-btn")
                yield Button("Back to Menu", variant="default", id="menu-btn")

            yield Static("", id="status-message")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "save-btn":
            await self._save_to_file()

        elif event.button.id == "again-btn":
            # Go back to config screen (previous screen)
            self.app.pop_screen()

        elif event.button.id == "menu-btn":
            # Go back to main menu (pop both output and config screens)
            self.app.pop_screen()  # Pop output screen
            self.app.pop_screen()  # Pop config screen

    async def _save_to_file(self) -> None:
        """Save poem to file with timestamp."""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize generator name for filename
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_"
                for c in self.generator_name.lower()
            ).replace(" ", "_")
            filename = f"{safe_name}_{timestamp}.txt"

            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Write file
            file_path = output_dir / filename
            file_path.write_text(self.poem_text, encoding="utf-8")

            # Update status
            status_widget = self.query_one("#status-message", Static)
            status_widget.update(f"✓ Saved to {file_path}")

        except Exception as e:
            status_widget = self.query_one("#status-message", Static)
            status_widget.update(f"❌ Error saving: {e}")

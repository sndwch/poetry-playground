"""Main Textual TUI application for generative poetry.

This module provides a full-featured terminal user interface for all poetry
generation procedures in the poetryplayground library.
"""

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from poetryplayground.tui.screens.unified_screen import UnifiedTUIScreen


class GenerativePoetryTUI(App):
    """A full-featured TUI for generative poetry.

    This application provides an interactive terminal interface for all
    generation procedures, with a 3-column layout showing procedures,
    configuration, and output simultaneously.
    """

    CSS_PATH = "tui.tcss"
    TITLE = "Generative Poetry"

    BINDINGS: ClassVar[list] = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Dark Mode", show=True),
        Binding("?", "help", "Help", show=True),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        """Show unified TUI on startup."""
        self.push_screen(UnifiedTUIScreen())

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark


def run():
    """Entry point for the TUI application."""
    app = GenerativePoetryTUI()
    app.run()


if __name__ == "__main__":
    run()

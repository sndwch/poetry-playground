"""Main Textual TUI application for generative poetry.

This module provides a full-featured terminal user interface for all poetry
generation procedures in the generativepoetry library.
"""

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from generativepoetry.tui.screens.main_menu import MainMenuScreen


class GenerativePoetryTUI(App):
    """A full-featured TUI for generative poetry.

    This application provides an interactive terminal interface for all
    generation procedures, with dynamic configuration forms and output display.
    """

    CSS_PATH = "tui.tcss"
    TITLE = "Generative Poetry"

    BINDINGS: ClassVar[list] = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_dark", "Dark Mode", show=True),
        Binding("m", "main_menu", "Main Menu", show=True),
        Binding("?", "help", "Help", show=True),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        """Show main menu on startup."""
        self.push_screen(MainMenuScreen())

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.dark = not self.dark

    def action_main_menu(self) -> None:
        """Return to main menu."""
        # Pop all screens except main menu
        while len(self.screen_stack) > 1:
            self.pop_screen()


def run():
    """Entry point for the TUI application."""
    app = GenerativePoetryTUI()
    app.run()


if __name__ == "__main__":
    run()

"""Main menu screen for selecting generation procedures."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from poetryplayground.tui.procedures import PROCEDURES


class MainMenuScreen(Screen):
    """Main procedure selection screen with all available generators.

    Procedures are imported from poetryplayground.tui.procedures to maintain
    a single source of truth across all TUI screens.
    """

    # Use shared procedures list (imported at module level)
    PROCEDURES = PROCEDURES

    CSS = """
    MainMenuScreen {
        align: center middle;
    }

    #menu-container {
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

    #subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }

    #procedure-list {
        height: auto;
        max-height: 30;
        border: solid $primary;
        margin: 1 0;
    }

    .list-item-label {
        width: 1fr;
    }

    .category-header {
        text-style: bold;
        color: $accent;
        background: $surface-darken-1;
    }
    """

    def compose(self) -> ComposeResult:
        """Create menu widgets."""
        with Container(id="menu-container"):
            yield Label("✨ Generative Poetry ✨", id="title")
            yield Label("Select a procedure to begin", id="subtitle")
            yield self._create_procedure_list()

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
        """Handle procedure selection."""
        if event.item.id and not event.item.disabled:
            proc_id = event.item.id

            # Import here to avoid circular dependencies
            from poetryplayground.tui.screens.config_form import ConfigFormScreen

            await self.app.push_screen(ConfigFormScreen(proc_id))

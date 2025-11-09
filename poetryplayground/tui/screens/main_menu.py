"""Main menu screen for selecting generation procedures."""

from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static


class MainMenuScreen(Screen):
    """Main procedure selection screen with all available generators."""

    # Define all available procedures with their metadata
    # Format: (id, name, description, category)
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
        # Syllabic Forms
        ("haiku", "Haiku Generator", "5-7-5 syllable haiku with templates", "Forms"),
        ("tanka", "Tanka Generator", "5-7-5-7-7 syllable tanka", "Forms"),
        ("senryu", "Senryu Generator", "5-7-5 syllable senryu", "Forms"),
        # System
        ("deps", "Check System Dependencies", "Verify installation and dependencies", "System"),
    ]

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

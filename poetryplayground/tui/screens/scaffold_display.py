"""Hierarchical display screen for poem scaffolds.

This screen uses a Textual Tree widget to display PoemScaffold objects
in a clean, navigable hierarchical format. It shows:
- The overall thematic path
- Each stanza's building blocks (palette, ideas, metaphor, seeds)
"""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, Tree

from poetryplayground.poem_scaffold import PoemScaffold


class ScaffoldDisplayScreen(Screen):
    """Display a PoemScaffold in hierarchical tree format.

    This is a dedicated display screen that does NOT use the standard
    OutputScreen. It provides an interactive tree view of the scaffold
    with collapsible sections for each stanza's components.
    """

    CSS = """
    ScaffoldDisplayScreen {
        align: center middle;
    }

    #scaffold-container {
        width: 90;
        height: auto;
        max-height: 95%;
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

    #scaffold-tree {
        height: auto;
        max-height: 70;
        border: solid $primary;
        margin: 1 0;
        padding: 1;
    }

    #button-row {
        align: center middle;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }

    .tree-label {
        color: $text;
    }

    .section-header {
        color: $accent;
        text-style: bold;
    }

    .theme-word {
        color: $warning;
        text-style: bold;
    }

    .quality-score {
        color: $success;
    }
    """

    def __init__(self, scaffold: PoemScaffold):
        """Initialize the display with a PoemScaffold.

        Args:
            scaffold: The PoemScaffold to display
        """
        super().__init__()
        self.scaffold = scaffold

    def compose(self) -> ComposeResult:
        """Create the display widgets."""
        with Container(id="scaffold-container"):
            # Title showing the thematic arc
            yield Label(
                f'✨ POEM SCAFFOLD: "{self.scaffold.start_concept}" → "{self.scaffold.end_concept}" ✨',
                id="title",
            )

            # Thematic path
            path_str = " → ".join(self.scaffold.thematic_path)
            yield Label(f"Thematic Path: [{path_str}]", id="subtitle")

            # Tree widget for hierarchical display
            yield self._create_scaffold_tree()

            # Navigation buttons
            with Vertical(id="button-row"):
                yield Button("Back to Menu", id="back-btn", variant="primary")

    def _create_scaffold_tree(self) -> Tree:
        """Create a hierarchical tree view of the scaffold.

        Returns:
            Tree widget populated with scaffold data
        """
        tree: Tree[str] = Tree("Poem Scaffold", id="scaffold-tree")
        tree.root.expand()

        # Add each stanza as a top-level node
        for stanza in self.scaffold.stanzas:
            stanza_label = f'STANZA {stanza.stanza_number}: "{stanza.key_theme_word.upper()}"'
            stanza_node = tree.root.add(stanza_label, expand=True)

            # Key Theme Word (already shown in stanza label, but add as first item)
            stanza_node.add(f"[Key Theme Word]: {stanza.key_theme_word}")

            # Vocabulary Palette
            palette_node = stanza_node.add("[Vocabulary Palette]:", expand=True)
            for cluster_type, terms in stanza.conceptual_palette.items():
                if terms:
                    # Format: "• Imagery: [word1, word2, word3, ...]"
                    word_list = ", ".join([term.term for term in terms[:8]])  # Limit display
                    if len(terms) > 8:
                        word_list += f", ... ({len(terms) - 8} more)"
                    palette_node.add(f"  • {cluster_type.capitalize()}: [{word_list}]")

            # Lateral Ideas
            if stanza.lateral_ideas:
                ideas_node = stanza_node.add("[Lateral Ideas]:", expand=True)
                for idea in stanza.lateral_ideas[:5]:  # Show top 5
                    # Format: "• word: (definition snippet...)"
                    definition_snippet = (
                        idea.definition[:60] + "..."
                        if len(idea.definition) > 60
                        else idea.definition
                    )
                    ideas_node.add(f"  • {idea.word}: ({definition_snippet})")

            # Key Metaphor
            metaphor_node = stanza_node.add("[Key Metaphor]:", expand=True)
            if stanza.key_metaphor:
                quality_str = f"{stanza.key_metaphor.quality_score:.2f}"
                metaphor_node.add(f'  • "{stanza.key_metaphor.text}" (Quality: {quality_str})')
            else:
                metaphor_node.add("  • (No metaphor generated)")

            # Starter Lines
            seeds_node = stanza_node.add("[Starter Lines]:", expand=True)
            if stanza.line_seeds:
                for seed in stanza.line_seeds[:5]:  # Show top 5
                    seeds_node.add(f'  • "{seed.text}"')
            else:
                seeds_node.add("  • (No starter lines generated)")

        return tree

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: The button press event
        """
        if event.button.id == "back-btn":
            # Return to main menu
            self.app.pop_screen()

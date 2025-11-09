"""Global Rich console with custom theme for generative poetry CLI.

This module provides a shared Console instance used throughout the application
for consistent styling and output formatting.
"""

from rich.console import Console
from rich.theme import Theme

# Custom theme matching the project's aesthetic
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "accent": "magenta",
        "muted": "dim white",
        "title": "bold magenta",
        "category": "yellow",
        "procedure": "bold cyan",
    }
)

# Global console instance - use this throughout the application
console = Console(theme=custom_theme)

"""Rich output formatting helpers for generative poetry CLI.

This module provides reusable functions for displaying poems and other output
using Rich's Panel and other components for professional presentation.
"""

from datetime import datetime
from typing import Dict, List, Optional

from rich.panel import Panel

from poetryplayground.rich_console import console


def display_poem_output(
    poem_lines: List[str],
    title: str = "Poem Generated",
    metadata: Optional[Dict[str, str]] = None,
    border_style: str = "blue",
) -> None:
    """Display a poem in a Rich Panel with optional metadata.

    Args:
        poem_lines: List of poem lines to display
        title: Title for the panel (default: "Poem Generated")
        metadata: Optional dictionary of metadata to display in subtitle
        border_style: Border color/style (default: "blue")

    Example:
        >>> display_poem_output(
        ...     ["line one", "line two"],
        ...     title="Haiku",
        ...     metadata={"seed": "42", "form": "haiku"}
        ... )
    """
    poem_text = "\n".join(poem_lines)

    # Build title with Rich markup
    formatted_title = f"[bold magenta]{title}[/bold magenta]"

    # Build subtitle with metadata if provided
    subtitle = None
    if metadata:
        footer_parts = []

        # Add common metadata fields in consistent order
        if "seed" in metadata:
            footer_parts.append(f"[cyan]Seed:[/cyan] {metadata['seed']}")

        if "form" in metadata:
            footer_parts.append(f"[cyan]Form:[/cyan] {metadata['form']}")

        if "template" in metadata:
            footer_parts.append(f"[cyan]Template:[/cyan] {metadata['template']}")

        if "syllables" in metadata:
            footer_parts.append(f"[cyan]Syllables:[/cyan] {metadata['syllables']}")

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        footer_parts.append(f"[cyan]Generated:[/cyan] {timestamp}")

        # Add any custom metadata
        for key, value in metadata.items():
            if key not in ["seed", "form", "template", "syllables"]:
                footer_parts.append(f"[cyan]{key.title()}:[/cyan] {value}")

        subtitle = f"[dim]{' | '.join(footer_parts)}[/dim]" if footer_parts else None

    # Create and print panel
    panel = Panel(
        poem_text,
        title=formatted_title,
        subtitle=subtitle,
        border_style=border_style,
        padding=(1, 2),
    )

    console.print(panel)


def display_error(message: str, details: Optional[str] = None) -> None:
    """Display an error message in a Rich Panel.

    Args:
        message: Main error message
        details: Optional detailed error information
    """
    content = f"[bold red]Error:[/bold red] {message}"
    if details:
        content += f"\n\n[dim]{details}[/dim]"

    panel = Panel(content, border_style="red", title="[bold red]Error[/bold red]")

    console.print(panel)


def display_success(message: str) -> None:
    """Display a success message in a Rich Panel.

    Args:
        message: Success message to display
    """
    panel = Panel(
        f"[bold green]{message}[/bold green]",
        border_style="green",
        title="[bold green]Success[/bold green]",
    )

    console.print(panel)


def display_info(message: str, title: str = "Info") -> None:
    """Display an informational message in a Rich Panel.

    Args:
        message: Info message to display
        title: Panel title (default: "Info")
    """
    panel = Panel(message, border_style="cyan", title=f"[bold cyan]{title}[/bold cyan]")

    console.print(panel)

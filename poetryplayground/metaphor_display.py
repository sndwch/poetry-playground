"""Display formatter for metaphor generator output.

This module provides centralized formatting for metaphor display across CLI, TUI,
and export contexts. It transforms Metaphor objects into clean, structured output
that's immediately usable for poets.
"""

import json
import re
from typing import Dict, List

from poetryplayground.metaphor_generator import Metaphor, MetaphorType


def _strip_ansi_codes(text: str) -> str:
    """Strip ANSI color codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def _truncate_context(context: str, max_length: int = 40) -> str:
    """Truncate context sentence to max length with ellipsis."""
    if len(context) <= max_length:
        return context
    return context[:max_length] + "..."


def _quality_bar(score: float, length: int = 10) -> str:
    """Create visual quality bar using block characters.

    Args:
        score: Quality score 0.0-1.0
        length: Number of blocks in full bar

    Returns:
        String like "████████  " for score 0.8
    """
    filled = int(score * length)
    return "█" * filled + " " * (length - filled)


def _quality_stars(score: float) -> str:
    """Convert quality score to star rating.

    Args:
        score: Quality score 0.0-1.0

    Returns:
        String like "★★★★☆" for score 0.85
    """
    filled_stars = int(score * 5 + 0.5)  # Round to nearest
    empty_stars = 5 - filled_stars
    return "★" * filled_stars + "☆" * empty_stars


def format_metaphors(
    metaphors: List[Metaphor],
    mode: str = "compact",
    max_per_category: int = 5,
    show_context: bool = False,
) -> str:
    """Format metaphors for display.

    Args:
        metaphors: List of Metaphor objects to format
        mode: Output mode - "compact", "detailed", "json", "markdown", "simple"
        max_per_category: Maximum metaphors to show per type (for compact/detailed)
        show_context: Whether to show context sentences (for detailed mode)

    Returns:
        Formatted string ready for display
    """
    if not metaphors:
        return "No metaphors found."

    if mode == "compact":
        return _format_compact(metaphors, max_per_category)
    elif mode == "detailed":
        return _format_detailed(metaphors, max_per_category, show_context)
    elif mode == "json":
        return _export_json(metaphors)
    elif mode == "markdown":
        return _export_markdown(metaphors)
    elif mode == "simple":
        return _export_simple(metaphors)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use compact, detailed, json, markdown, or simple")


def _format_compact(metaphors: List[Metaphor], max_per_category: int) -> str:
    """Format metaphors in compact TUI-optimized mode with box drawing.

    Groups by type, sorts by quality, shows bars. Width: 60 chars.
    """
    # Group by type
    by_type: Dict[MetaphorType, List[Metaphor]] = {}
    for m in metaphors:
        if m.metaphor_type not in by_type:
            by_type[m.metaphor_type] = []
        by_type[m.metaphor_type].append(m)

    # Sort each group by quality
    for type_list in by_type.values():
        type_list.sort(key=lambda m: m.quality_score, reverse=True)

    # Build output
    lines = []
    lines.append("┌─── METAPHORS FROM LITERATURE ─────────────────────┐")
    lines.append("│                                                    │")

    # Define display order and labels
    type_labels = {
        MetaphorType.SIMILE: "SIMILES (like/as)",
        MetaphorType.DIRECT: "DIRECT (is/are)",
        MetaphorType.POSSESSIVE: "POSSESSIVE (of)",
        MetaphorType.APPOSITIVE: "APPOSITIVE",
        MetaphorType.IMPLIED: "IMPLIED",
    }

    # Display each type that has metaphors
    for metaphor_type, label in type_labels.items():
        if metaphor_type in by_type:
            items = by_type[metaphor_type][:max_per_category]
            lines.extend(_format_compact_section(label, items))

    # Add summary
    total = len(metaphors)
    avg_quality = sum(m.quality_score for m in metaphors) / total if total > 0 else 0
    lines.append("│                                                    │")
    lines.append(f"│ Summary: {total} metaphors | Avg quality: {avg_quality:.2f}         │")
    lines.append("└────────────────────────────────────────────────────┘")

    return "\n".join(lines)


def _format_compact_section(title: str, items: List[Metaphor]) -> List[str]:
    """Format a section of metaphors for compact mode."""
    lines = []
    lines.append(f"│ {title:<35} Quality        │")
    lines.append(f"│ {'─' * 35} {'─' * 14} │")

    for metaphor in items:
        # Create quality bar
        bar = _quality_bar(metaphor.quality_score, length=10)

        # Format metaphor line (source → target)
        metaphor_text = f"{metaphor.source} → {metaphor.target}"
        # Truncate if too long
        if len(metaphor_text) > 30:
            metaphor_text = metaphor_text[:27] + "..."

        lines.append(f"│ {metaphor_text:<35} {metaphor.quality_score:.2f} {bar}│")

        # Add truncated context
        context = _truncate_context(metaphor.source_text or "", 40)
        lines.append(f'│   "{context}"                  │'[:54])  # Ensure fits
        lines.append("│                                                    │")

    return lines


def _format_detailed(metaphors: List[Metaphor], max_per_category: int, show_context: bool) -> str:
    """Format metaphors in detailed CLI mode with quality tiers."""
    lines = []
    lines.append("=" * 60)
    lines.append("METAPHORS FROM LITERATURE")
    lines.append("=" * 60)
    lines.append("")

    # Group by quality tier
    excellent = [m for m in metaphors if m.quality_score >= 0.8]
    good = [m for m in metaphors if 0.6 <= m.quality_score < 0.8]
    fair = [m for m in metaphors if 0.5 <= m.quality_score < 0.6]

    # Display each tier
    if excellent:
        lines.append("EXCELLENT (0.8+)")
        lines.append("-" * 60)
        for m in excellent[:max_per_category]:
            stars = _quality_stars(m.quality_score)
            lines.append(f"• {m.source} → {m.target} ({stars} {m.quality_score:.2f})")
            lines.append(f"  Type: {m.metaphor_type.value}")
            if show_context and m.source_text:
                lines.append(f'  Context: "{_truncate_context(m.source_text, 70)}"')
            lines.append("")

    if good:
        lines.append("GOOD (0.6-0.8)")
        lines.append("-" * 60)
        for m in good[:max_per_category]:
            stars = _quality_stars(m.quality_score)
            lines.append(f"• {m.source} → {m.target} ({stars} {m.quality_score:.2f})")
            lines.append(f"  Type: {m.metaphor_type.value}")
            if show_context and m.source_text:
                lines.append(f'  Context: "{_truncate_context(m.source_text, 70)}"')
            lines.append("")

    if fair:
        lines.append("FAIR (0.5-0.6)")
        lines.append("-" * 60)
        for m in fair[:max_per_category]:
            stars = _quality_stars(m.quality_score)
            lines.append(f"• {m.source} → {m.target} ({stars} {m.quality_score:.2f})")
            lines.append(f"  Type: {m.metaphor_type.value}")
            if show_context and m.source_text:
                lines.append(f'  Context: "{_truncate_context(m.source_text, 70)}"')
            lines.append("")

    # Summary
    total = len(metaphors)
    avg_quality = sum(m.quality_score for m in metaphors) / total if total > 0 else 0
    lines.append("=" * 60)
    lines.append(f"Total: {total} metaphors | Average quality: {avg_quality:.2f}")
    lines.append("=" * 60)

    return "\n".join(lines)


def _export_json(metaphors: List[Metaphor]) -> str:
    """Export metaphors as JSON."""
    data = {
        "metaphors": [
            {
                "source": m.source,
                "target": m.target,
                "text": m.text,
                "type": m.metaphor_type.value,
                "quality": m.quality_score,
                "context": m.source_text or "",
                "grounds": m.grounds or [],
            }
            for m in metaphors
        ],
        "total": len(metaphors),
        "avg_quality": sum(m.quality_score for m in metaphors) / len(metaphors) if metaphors else 0,
    }
    return json.dumps(data, indent=2)


def _export_markdown(metaphors: List[Metaphor]) -> str:
    """Export metaphors as markdown table."""
    lines = ["# Extracted Metaphors\n"]
    lines.append("| Source | Target | Type | Quality | Context |")
    lines.append("|--------|--------|------|---------|---------|")

    for m in metaphors:
        context = _truncate_context(m.source_text or "", 50)
        stars = _quality_stars(m.quality_score)
        lines.append(
            f"| {m.source} | {m.target} | {m.metaphor_type.value} | {stars} {m.quality_score:.2f} | {context} |"
        )

    # Add summary
    total = len(metaphors)
    avg_quality = sum(m.quality_score for m in metaphors) / total if total > 0 else 0
    lines.append("")
    lines.append(f"**Total:** {total} metaphors | **Average Quality:** {avg_quality:.2f}")

    return "\n".join(lines)


def _export_simple(metaphors: List[Metaphor]) -> str:
    """Export as simple source→target list, sorted by quality."""
    sorted_metaphors = sorted(metaphors, key=lambda m: m.quality_score, reverse=True)
    lines = []
    for m in sorted_metaphors:
        lines.append(f"{m.source} → {m.target} ({m.quality_score:.2f})")
    return "\n".join(lines)

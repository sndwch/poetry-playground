"""Poem template system for structure-based generation.

This module provides the PoemTemplate dataclass and related utilities for
extracting, storing, and applying structural patterns from existing poems.

A template captures:
- Line-by-line POS patterns and syllable structure
- Metaphor types and positions
- Semantic domain mappings
- Emotional tone and register
- Line types (opening, pivot, image, closing)
- Quality thresholds

Example:
    >>> from poetryplayground.poem_template import PoemTemplate
    >>> template = PoemTemplate(
    ...     title="Haiku Template",
    ...     lines=3,
    ...     syllable_pattern=[5, 7, 5],
    ...     pos_patterns=[
    ...         ['DET', 'NOUN', 'VERB'],
    ...         ['ADJ', 'NOUN', 'VERB', 'ADV'],
    ...         ['DET', 'NOUN', 'VERB']
    ...     ]
    ... )
    >>> data = template.to_dict()
    >>> restored = PoemTemplate.from_dict(data)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from poetryplayground.core.quality_scorer import EmotionalTone, FormalityLevel


class LineType(Enum):
    """Types of lines in a poem's structure."""

    OPENING = "opening"  # First line or opening stance
    PIVOT = "pivot"  # Turning point or volta
    IMAGE = "image"  # Concrete imagery or description
    EMOTIONAL = "emotional"  # Emotional expression
    SONIC = "sonic"  # Sound-focused (alliteration, etc.)
    CLOSING = "closing"  # Concluding line
    FRAGMENT = "fragment"  # Incomplete thought
    TRANSITION = "transition"  # Bridge between ideas


@dataclass
class LineTemplate:
    """Template for a single line in a poem.

    Captures structural and semantic information for one line.

    Attributes:
        syllable_count: Number of syllables in the line
        pos_pattern: Part-of-speech sequence (e.g., ['DET', 'NOUN', 'VERB'])
        line_type: Functional role of this line
        metaphor_type: Optional metaphor type if line contains metaphor
        semantic_domain: Optional semantic domain (e.g., "nature", "emotion")
        concreteness_target: Target concreteness ratio (0=abstract, 1=concrete)
        min_quality_score: Minimum acceptable quality score for words
    """

    syllable_count: int
    pos_pattern: List[str]
    line_type: LineType = LineType.IMAGE
    metaphor_type: Optional[str] = None
    semantic_domain: Optional[str] = None
    concreteness_target: float = 0.5
    min_quality_score: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "syllable_count": self.syllable_count,
            "pos_pattern": self.pos_pattern,
            "line_type": self.line_type.value,
            "metaphor_type": self.metaphor_type,
            "semantic_domain": self.semantic_domain,
            "concreteness_target": self.concreteness_target,
            "min_quality_score": self.min_quality_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineTemplate":
        """Create from dictionary."""
        return cls(
            syllable_count=data["syllable_count"],
            pos_pattern=data["pos_pattern"],
            line_type=LineType(data.get("line_type", "image")),
            metaphor_type=data.get("metaphor_type"),
            semantic_domain=data.get("semantic_domain"),
            concreteness_target=data.get("concreteness_target", 0.5),
            min_quality_score=data.get("min_quality_score", 0.6),
        )


@dataclass
class PoemTemplate:
    """Template for poem structure and constraints.

    A PoemTemplate captures the structural and semantic scaffolding of a poem,
    allowing similar poems to be generated while maintaining the original's
    essential characteristics.

    Attributes:
        title: Descriptive name for this template
        source: Origin of the template (e.g., "Shakespeare Sonnet 18")
        author: Original author or "generated"
        lines: Number of lines in the poem
        line_templates: Template for each line
        syllable_pattern: Total syllables per line
        semantic_domains: Semantic domains to draw from
        metaphor_types: Types of metaphors to include
        emotional_tone: Desired emotional register
        formality_level: Desired formality level
        concreteness_ratio: Overall abstract vs concrete balance (0-1)
        min_quality_score: Minimum quality threshold for all content
        style_components: Optional stylistic fingerprint
        notes: Optional generation notes or constraints
    """

    # Metadata
    title: str
    source: str = "user-provided"
    author: str = "anonymous"

    # Structure
    lines: int = 3
    line_templates: List[LineTemplate] = field(default_factory=list)
    syllable_pattern: List[int] = field(default_factory=list)

    # Semantics
    semantic_domains: List[str] = field(default_factory=list)
    metaphor_types: List[str] = field(default_factory=list)
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    formality_level: FormalityLevel = FormalityLevel.CONVERSATIONAL

    # Quality and Style
    concreteness_ratio: float = 0.5  # 0=abstract, 1=concrete
    min_quality_score: float = 0.6
    style_components: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate template after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate template consistency.

        Raises:
            ValueError: If template has invalid or inconsistent data
        """
        # Basic validation
        if self.lines < 1:
            raise ValueError("Template must have at least 1 line")

        if not self.title:
            raise ValueError("Template must have a title")

        # Validate syllable pattern length matches lines
        if self.syllable_pattern and len(self.syllable_pattern) != self.lines:
            raise ValueError(
                f"Syllable pattern length ({len(self.syllable_pattern)}) "
                f"must match line count ({self.lines})"
            )

        # Validate line templates length matches lines
        if self.line_templates and len(self.line_templates) != self.lines:
            raise ValueError(
                f"Line templates count ({len(self.line_templates)}) "
                f"must match line count ({self.lines})"
            )

        # Validate quality scores are in valid range
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be between 0.0 and 1.0")

        if not 0.0 <= self.concreteness_ratio <= 1.0:
            raise ValueError("concreteness_ratio must be between 0.0 and 1.0")

        # Validate each line template
        for i, line_template in enumerate(self.line_templates):
            if line_template.syllable_count < 1:
                raise ValueError(f"Line {i + 1} must have at least 1 syllable")

            if not line_template.pos_pattern:
                raise ValueError(f"Line {i + 1} must have a POS pattern")

            if not 0.0 <= line_template.concreteness_target <= 1.0:
                raise ValueError(f"Line {i + 1} concreteness_target must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            # Metadata
            "title": self.title,
            "source": self.source,
            "author": self.author,
            # Structure
            "lines": self.lines,
            "line_templates": [lt.to_dict() for lt in self.line_templates],
            "syllable_pattern": self.syllable_pattern,
            # Semantics
            "semantic_domains": self.semantic_domains,
            "metaphor_types": self.metaphor_types,
            "emotional_tone": self.emotional_tone.value,
            "formality_level": self.formality_level.value,
            # Quality and Style
            "concreteness_ratio": self.concreteness_ratio,
            "min_quality_score": self.min_quality_score,
            "style_components": self.style_components,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoemTemplate":
        """Create template from dictionary.

        Args:
            data: Dictionary containing template data

        Returns:
            PoemTemplate instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Parse line templates if present
        line_templates = []
        if "line_templates" in data:
            line_templates = [LineTemplate.from_dict(lt) for lt in data["line_templates"]]

        return cls(
            # Metadata
            title=data["title"],
            source=data.get("source", "user-provided"),
            author=data.get("author", "anonymous"),
            # Structure
            lines=data.get("lines", 3),
            line_templates=line_templates,
            syllable_pattern=data.get("syllable_pattern", []),
            # Semantics
            semantic_domains=data.get("semantic_domains", []),
            metaphor_types=data.get("metaphor_types", []),
            emotional_tone=EmotionalTone(data.get("emotional_tone", "neutral")),
            formality_level=FormalityLevel(data.get("formality_level", "conversational")),
            # Quality and Style
            concreteness_ratio=data.get("concreteness_ratio", 0.5),
            min_quality_score=data.get("min_quality_score", 0.6),
            style_components=data.get("style_components"),
            notes=data.get("notes"),
        )

    def get_line_template(self, line_index: int) -> Optional[LineTemplate]:
        """Get template for a specific line.

        Args:
            line_index: Zero-based line index

        Returns:
            LineTemplate for the line, or None if not available
        """
        if 0 <= line_index < len(self.line_templates):
            return self.line_templates[line_index]
        return None

    def get_total_syllables(self) -> int:
        """Calculate total syllables in the poem.

        Returns:
            Sum of all syllables across all lines
        """
        return sum(self.syllable_pattern) if self.syllable_pattern else 0

    def matches_structure(self, syllables: List[int]) -> bool:
        """Check if syllable pattern matches this template.

        Args:
            syllables: List of syllable counts to check

        Returns:
            True if pattern matches template's syllable_pattern
        """
        return syllables == self.syllable_pattern

    def __str__(self) -> str:
        """Human-readable template summary."""
        return (
            f"PoemTemplate('{self.title}', {self.lines} lines, "
            f"{self.get_total_syllables()} syllables, "
            f"{self.emotional_tone.value} tone)"
        )

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"PoemTemplate(title={self.title!r}, lines={self.lines}, "
            f"syllable_pattern={self.syllable_pattern}, "
            f"semantic_domains={self.semantic_domains})"
        )


# Convenience functions for common template patterns


def create_haiku_template(
    semantic_domains: Optional[List[str]] = None,
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL,
) -> PoemTemplate:
    """Create a standard haiku template (5-7-5 syllables).

    Args:
        semantic_domains: Optional semantic domains to constrain vocabulary
        emotional_tone: Emotional register for the haiku

    Returns:
        PoemTemplate configured for haiku structure
    """
    return PoemTemplate(
        title="Haiku Template",
        source="traditional",
        lines=3,
        syllable_pattern=[5, 7, 5],
        line_templates=[
            LineTemplate(
                syllable_count=5,
                pos_pattern=["DET", "NOUN", "VERB"],
                line_type=LineType.OPENING,
            ),
            LineTemplate(
                syllable_count=7,
                pos_pattern=["ADJ", "NOUN", "VERB", "ADV"],
                line_type=LineType.IMAGE,
            ),
            LineTemplate(
                syllable_count=5,
                pos_pattern=["DET", "NOUN", "VERB"],
                line_type=LineType.CLOSING,
            ),
        ],
        semantic_domains=semantic_domains or ["nature"],
        emotional_tone=emotional_tone,
        formality_level=FormalityLevel.FORMAL,
        concreteness_ratio=0.7,  # Haiku tends toward concrete imagery
    )


def create_tanka_template(
    semantic_domains: Optional[List[str]] = None,
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL,
) -> PoemTemplate:
    """Create a standard tanka template (5-7-5-7-7 syllables).

    Args:
        semantic_domains: Optional semantic domains to constrain vocabulary
        emotional_tone: Emotional register for the tanka

    Returns:
        PoemTemplate configured for tanka structure
    """
    return PoemTemplate(
        title="Tanka Template",
        source="traditional",
        lines=5,
        syllable_pattern=[5, 7, 5, 7, 7],
        line_templates=[
            LineTemplate(
                syllable_count=5,
                pos_pattern=["DET", "NOUN", "VERB"],
                line_type=LineType.OPENING,
            ),
            LineTemplate(
                syllable_count=7,
                pos_pattern=["ADJ", "NOUN", "VERB", "ADV"],
                line_type=LineType.IMAGE,
            ),
            LineTemplate(
                syllable_count=5,
                pos_pattern=["DET", "NOUN", "VERB"],
                line_type=LineType.PIVOT,
            ),
            LineTemplate(
                syllable_count=7,
                pos_pattern=["ADJ", "NOUN", "VERB", "ADV"],
                line_type=LineType.EMOTIONAL,
            ),
            LineTemplate(
                syllable_count=7,
                pos_pattern=["DET", "NOUN", "VERB", "ADV"],
                line_type=LineType.CLOSING,
            ),
        ],
        semantic_domains=semantic_domains or ["nature", "emotion"],
        emotional_tone=emotional_tone,
        formality_level=FormalityLevel.FORMAL,
        concreteness_ratio=0.6,
    )

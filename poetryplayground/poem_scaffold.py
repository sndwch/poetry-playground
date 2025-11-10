"""Data structures for poem scaffolds.

This module defines the data structures used by the GeneratePoemScaffold strategy
to represent the thematic structure and creative materials for multi-stanza poems.

A poem scaffold is not a complete poem, but rather a structured collection of
"building blocks" that a poet can use to compose a thematically-linked poem.

Example:
    >>> from poetryplayground.poem_scaffold import PoemScaffold, StanzaScaffold
    >>> # Create a scaffold for a poem about rust → forgiveness
    >>> scaffold = PoemScaffold(
    ...     start_concept="rust",
    ...     end_concept="forgiveness",
    ...     thematic_path=["rust", "memory", "forgiveness"],
    ...     stanzas=[...]
    ... )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from poetryplayground.conceptual_cloud import CloudTerm
from poetryplayground.definitional_finder import DefinitionalResult
from poetryplayground.line_seeds import LineSeed
from poetryplayground.metaphor_generator import Metaphor


@dataclass
class StanzaScaffold:
    """The complete thematic building blocks for a single stanza.

    This dataclass contains all the creative materials needed to compose
    a single stanza, organized into four categories:
    - Vocabulary palette: Words clustered by semantic type
    - Lateral ideas: Alternative conceptual angles from definitions
    - Key metaphor: A central generative image
    - Starter lines: Opening phrases to begin composition

    Attributes:
        stanza_number: 1-indexed stanza number in the poem
        key_theme_word: The backbone word for this stanza from the semantic path
        conceptual_palette: Words organized by cluster type (e.g., "imagery", "contextual")
        lateral_ideas: Words from definition search providing alternative angles
        key_metaphor: A central metaphor for this stanza (may be None if generation failed)
        line_seeds: Opening phrases/fragments to inspire line beginnings

    Example:
        >>> stanza = StanzaScaffold(
        ...     stanza_number=1,
        ...     key_theme_word="rust",
        ...     conceptual_palette={
        ...         "imagery": [CloudTerm(term="iron", quality=0.85, ...)],
        ...         "contextual": [CloudTerm(term="decay", quality=0.79, ...)]
        ...     },
        ...     lateral_ideas=[DefinitionalResult(word="corrosion", ...)],
        ...     key_metaphor=Metaphor(source="rust", target="memory", ...),
        ...     line_seeds=[LineSeed(text="The iron water waits...", ...)]
        ... )
    """

    stanza_number: int
    key_theme_word: str

    # The "Vocabulary Palette" from ConceptualCloud
    # Keys are cluster types like "imagery", "contextual", "rare", etc.
    conceptual_palette: Dict[str, List[CloudTerm]] = field(default_factory=dict)

    # "Strange Orbit" ideas from DefinitionalFinder
    lateral_ideas: List[DefinitionalResult] = field(default_factory=list)

    # "Generative Images" from MetaphorGenerator
    key_metaphor: Optional[Metaphor] = None

    # "Starter Lines" from LineSeedGenerator (Personalized or Generic)
    line_seeds: List[LineSeed] = field(default_factory=list)


@dataclass
class PoemScaffold:
    """The full, multi-stanza thematic structure for a new poem.

    This dataclass represents the complete scaffold for a poem with a thematic
    arc from start_concept to end_concept, broken down into discrete stanzas.
    Each stanza has its own creative materials provided by the orchestrated
    generation tools.

    This scaffold is designed to be presented to the poet as a hierarchical
    display in the TUI, showing the overall thematic path and the specific
    building blocks for each stanza.

    Attributes:
        start_concept: The thematic starting point of the poem
        end_concept: The thematic destination of the poem
        thematic_path: The full semantic path (list of theme words, one per stanza)
        stanzas: A list of scaffolds, one for each stanza

    Example:
        >>> scaffold = PoemScaffold(
        ...     start_concept="rust",
        ...     end_concept="forgiveness",
        ...     thematic_path=["rust", "memory", "forgiveness"],
        ...     stanzas=[
        ...         StanzaScaffold(stanza_number=1, key_theme_word="rust", ...),
        ...         StanzaScaffold(stanza_number=2, key_theme_word="memory", ...),
        ...         StanzaScaffold(stanza_number=3, key_theme_word="forgiveness", ...)
        ...     ]
        ... )
    """

    start_concept: str
    end_concept: str
    thematic_path: List[str]

    # A list of scaffolds, one for each stanza
    stanzas: List[StanzaScaffold] = field(default_factory=list)

    def __post_init__(self):
        """Validate the scaffold structure."""
        # Ensure thematic path length matches number of stanzas
        if len(self.stanzas) > 0 and len(self.thematic_path) != len(self.stanzas):
            raise ValueError(
                f"Thematic path length ({len(self.thematic_path)}) "
                f"must match number of stanzas ({len(self.stanzas)})"
            )

        # Validate stanza numbers are sequential
        for i, stanza in enumerate(self.stanzas, 1):
            if stanza.stanza_number != i:
                raise ValueError(
                    f"Stanza numbers must be sequential starting from 1. "
                    f"Expected stanza_number={i}, got {stanza.stanza_number}"
                )

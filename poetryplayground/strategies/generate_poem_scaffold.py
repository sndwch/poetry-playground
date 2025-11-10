"""Strategy for generating multi-stanza poem scaffolds with thematic arcs.

This orchestration strategy creates a complete structural framework for a poem
by combining semantic pathfinding, conceptual clustering, metaphor generation,
and line seed creation. It does not write the poem itself, but provides the
poet with high-quality creative materials organized by stanza.

Example:
    >>> from poetryplayground.strategies.generate_poem_scaffold import GeneratePoemScaffoldStrategy
    >>> strategy = GeneratePoemScaffoldStrategy()
    >>> scaffold = strategy.run(
    ...     start_concept="rust",
    ...     end_concept="forgiveness",
    ...     num_stanzas=3
    ... )
    >>> print(scaffold.thematic_path)
    ['rust', 'memory', 'forgiveness']
"""

import logging
from pathlib import Path
from typing import Optional

from poetryplayground.conceptual_cloud import (
    ClusterType,
    generate_conceptual_cloud,
)
from poetryplayground.corpus_analyzer import PersonalCorpusAnalyzer
from poetryplayground.definitional_finder import find_words_by_definition
from poetryplayground.line_seeds import LineSeedGenerator
from poetryplayground.logger import logger
from poetryplayground.metaphor_generator import MetaphorGenerator
from poetryplayground.personalized_seeds import PersonalizedLineSeedGenerator
from poetryplayground.poem_scaffold import PoemScaffold, StanzaScaffold
from poetryplayground.semantic_geodesic import find_semantic_path

# Set up logger for this module
module_logger = logging.getLogger(__name__)


class GeneratePoemScaffoldStrategy:
    """Orchestrate multiple generators to create a multi-stanza poem scaffold.

    This strategy implements the "No-Shortcuts" orchestration philosophy:
    it uses existing, debugged, quality-aware generators and combines them
    intelligently to create a complete thematic structure for a poem.

    The scaffold is organized as:
    - A semantic path from start_concept to end_concept (the "backbone")
    - For each step in the path (each stanza):
      - Vocabulary palette (from ConceptualCloud)
      - Lateral ideas (from DefinitionalFinder)
      - Central metaphor (from MetaphorGenerator)
      - Starter lines (from LineSeedGenerator or PersonalizedLineSeedGenerator)

    Attributes:
        metaphor_gen: MetaphorGenerator instance for creating metaphors
        seed_generator: LineSeedGenerator or PersonalizedLineSeedGenerator for starter lines
    """

    def __init__(self):
        """Initialize the strategy with all needed generators."""
        # These generators are instantiated on-demand to avoid unnecessary setup
        self.metaphor_gen: Optional[MetaphorGenerator] = None
        self.seed_generator: Optional[LineSeedGenerator] = None

    def run(
        self,
        start_concept: str,
        end_concept: str,
        num_stanzas: int = 3,
        fingerprint_path: Optional[str] = None,
    ) -> PoemScaffold:
        """Generate a complete poem scaffold with thematic arc.

        This is the main entry point for the strategy. It orchestrates all
        the sub-generators to create a structured, multi-stanza poem framework.

        Args:
            start_concept: The thematic starting point (e.g., "rust")
            end_concept: The thematic destination (e.g., "forgiveness")
            num_stanzas: Number of stanzas (minimum 2, default 3)
            fingerprint_path: Optional path to saved StyleFingerprint for personalized line seeds

        Returns:
            A complete PoemScaffold with all creative materials

        Raises:
            ValueError: If num_stanzas < 2 or if start_concept == end_concept
            FileNotFoundError: If fingerprint_path is provided but doesn't exist

        Example:
            >>> strategy = GeneratePoemScaffoldStrategy()
            >>> scaffold = strategy.run("silence", "thunder", num_stanzas=3)
            >>> print(scaffold.thematic_path)
            ['silence', 'resonance', 'thunder']
        """
        # Phase 1: Validation & Setup
        logger.info(
            f"GeneratePoemScaffold: '{start_concept}' → '{end_concept}' ({num_stanzas} stanzas)"
        )

        if num_stanzas < 2:
            raise ValueError(f"num_stanzas must be >= 2, got {num_stanzas}")

        if start_concept.lower() == end_concept.lower():
            raise ValueError("start_concept and end_concept must be different")

        # Instantiate generators
        self._setup_generators(fingerprint_path)

        # Phase 2: Generate Thematic Backbone
        logger.info("Phase 2: Generating thematic backbone via semantic path...")
        path_obj = find_semantic_path(
            start_word=start_concept,
            end_word=end_concept,
            steps=num_stanzas,
            method="bezier",  # Bezier method for curved, creative paths
        )

        # Extract the primary word from each step
        thematic_path = [step.bridges[0].word for step in path_obj.steps]
        logger.info(f"Thematic path: {' → '.join(thematic_path)}")

        # Phase 3: Iterate and Build Each StanzaScaffold
        logger.info("Phase 3: Building stanza scaffolds...")
        stanzas = []

        for i, theme_word in enumerate(thematic_path, 1):
            logger.info(f"  Building stanza {i}/{num_stanzas}: '{theme_word}'")
            stanza = self._build_stanza_scaffold(i, theme_word)
            stanzas.append(stanza)

        # Phase 4: Assemble Final PoemScaffold
        scaffold = PoemScaffold(
            start_concept=start_concept,
            end_concept=end_concept,
            thematic_path=thematic_path,
            stanzas=stanzas,
        )

        logger.info(f"GeneratePoemScaffold complete: {len(scaffold.stanzas)} stanzas generated")
        return scaffold

    def _setup_generators(self, fingerprint_path: Optional[str]):
        """Initialize all needed generators.

        Args:
            fingerprint_path: Optional path to StyleFingerprint for personalized seeds
        """
        # Metaphor generator
        if self.metaphor_gen is None:
            logger.debug("Initializing MetaphorGenerator...")
            self.metaphor_gen = MetaphorGenerator()

        # Line seed generator (personalized or generic)
        if fingerprint_path:
            logger.debug(f"Loading StyleFingerprint from {fingerprint_path}...")
            fingerprint_path_obj = Path(fingerprint_path)

            if not fingerprint_path_obj.exists():
                raise FileNotFoundError(f"StyleFingerprint not found at {fingerprint_path}")

            # Load the fingerprint and create personalized generator
            analyzer = PersonalCorpusAnalyzer()
            fingerprint = analyzer.load_fingerprint(fingerprint_path)
            self.seed_generator = PersonalizedLineSeedGenerator(fingerprint, strictness=0.7)
            logger.info("Using PersonalizedLineSeedGenerator with loaded fingerprint")
        else:
            logger.debug("Using generic LineSeedGenerator...")
            self.seed_generator = LineSeedGenerator()

    def _build_stanza_scaffold(self, stanza_number: int, theme_word: str) -> StanzaScaffold:
        """Build a complete scaffold for a single stanza.

        This method orchestrates all the sub-generation steps for one stanza,
        following the algorithm from POEM_SCAFFOLD_STRATEGY.MD.

        Args:
            stanza_number: 1-indexed stanza number
            theme_word: The backbone word for this stanza

        Returns:
            A complete StanzaScaffold with all creative materials
        """
        # Step 3a: Generate Vocabulary Palette (Conceptual Cloud)
        logger.debug(f"    3a: Generating vocabulary palette for '{theme_word}'...")
        stanza_palette = generate_conceptual_cloud(
            center_word=theme_word,
            k_per_cluster=5,
            sections=["imagery", "contextual", "rare"],
        )

        # Convert ConceptualCloud to the format expected by StanzaScaffold
        conceptual_palette = {
            cluster_type: stanza_palette.get_cluster(ClusterType[cluster_type.upper()])
            for cluster_type in ["imagery", "contextual", "rare"]
        }

        # Step 3b: Generate Lateral Ideas (Definitional Finder)
        logger.debug(f"    3b: Generating lateral ideas for '{theme_word}'...")
        lateral_ideas = find_words_by_definition(
            theme_word, pos_filter="n", limit=3, min_quality=0.6
        )

        # Step 3c: Generate Central Image (Metaphor Generator)
        logger.debug(f"    3c: Generating key metaphor for '{theme_word}'...")
        # Get a target word from the imagery cluster for the metaphor
        imagery_words = conceptual_palette.get("imagery", [])
        target_word = imagery_words[0].term if imagery_words else theme_word

        key_metaphor = self.metaphor_gen.generate_metaphor_from_pair(
            source=theme_word, target=target_word
        )

        # Step 3d: Generate Starter Lines (LineSeed Generator)
        logger.debug(f"    3d: Generating starter lines for '{theme_word}'...")
        # Build seed words from theme word + palette
        palette_list = [term.term for term in stanza_palette.get_all_terms()]
        seed_words = [theme_word, *palette_list[:10]]  # Limit to avoid overwhelming API

        line_seeds = self.seed_generator.generate_seed_collection(
            seed_words=seed_words, num_seeds=3
        ).seeds

        # Assemble the stanza scaffold
        stanza = StanzaScaffold(
            stanza_number=stanza_number,
            key_theme_word=theme_word,
            conceptual_palette=conceptual_palette,
            lateral_ideas=lateral_ideas,
            key_metaphor=key_metaphor,
            line_seeds=line_seeds,
        )

        logger.debug(
            f"    Stanza {stanza_number} complete: "
            f"{len(lateral_ideas)} lateral ideas, "
            f"{len(line_seeds)} starter lines, "
            f"metaphor={'present' if key_metaphor else 'absent'}"
        )

        return stanza

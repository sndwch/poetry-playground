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
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
from poetryplayground.strategy_engine import BaseStrategy, StrategyResult

# Set up logger for this module
module_logger = logging.getLogger(__name__)


class GeneratePoemScaffoldStrategy(BaseStrategy):
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

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters for this strategy.

        Required params:
        - start_concept (str): The thematic starting point
        - end_concept (str): The thematic destination

        Optional params:
        - num_stanzas (int): Number of stanzas (minimum 2, default 3)
        - fingerprint_path (str): Path to saved StyleFingerprint

        Args:
            params: Dictionary of parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required params
        if "start_concept" not in params:
            return False, "Missing required parameter: start_concept"

        if "end_concept" not in params:
            return False, "Missing required parameter: end_concept"

        # Validate types
        if not isinstance(params["start_concept"], str) or not params["start_concept"].strip():
            return False, "start_concept must be a non-empty string"

        if not isinstance(params["end_concept"], str) or not params["end_concept"].strip():
            return False, "end_concept must be a non-empty string"

        # Check that concepts are different
        if params["start_concept"].strip().lower() == params["end_concept"].strip().lower():
            return False, "start_concept and end_concept must be different"

        # Validate optional params
        if "num_stanzas" in params and (
            not isinstance(params["num_stanzas"], int) or params["num_stanzas"] < 2
        ):
            return False, "num_stanzas must be an integer >= 2"

        fingerprint_path = params.get("fingerprint_path")
        if fingerprint_path:
            if not isinstance(fingerprint_path, str):
                return False, "fingerprint_path must be a string"
            fingerprint_path_obj = Path(fingerprint_path)
            if not fingerprint_path_obj.exists():
                return False, f"StyleFingerprint not found at {fingerprint_path}"

        return True, ""

    def run(self, params: Dict[str, Any]) -> StrategyResult:
        """Generate a complete poem scaffold with thematic arc.

        This is the main entry point for the strategy. It orchestrates all
        the sub-generators to create a structured, multi-stanza poem framework.

        Args:
            params: Validated parameters containing:
                - start_concept (str): The thematic starting point
                - end_concept (str): The thematic destination
                - num_stanzas (int, optional): Number of stanzas (default 3)
                - fingerprint_path (str, optional): Path to StyleFingerprint

        Returns:
            StrategyResult containing the generated PoemScaffold

        Example:
            >>> strategy = GeneratePoemScaffoldStrategy()
            >>> result = strategy.run({
            ...     'start_concept': 'silence',
            ...     'end_concept': 'thunder',
            ...     'num_stanzas': 3
            ... })
            >>> print(result.metadata['thematic_path'])
            ['silence', 'resonance', 'thunder']
        """
        start_time = time.time()

        # Extract params with defaults
        start_concept = params["start_concept"].strip()
        end_concept = params["end_concept"].strip()
        num_stanzas = params.get("num_stanzas", 3)
        fingerprint_path = params.get("fingerprint_path")

        # Phase 1: Validation & Setup
        logger.info(
            f"GeneratePoemScaffold: '{start_concept}' → '{end_concept}' ({num_stanzas} stanzas)"
        )

        # Instantiate generators
        self._setup_generators(fingerprint_path)

        # Phase 2: Generate Thematic Backbone
        logger.info("Phase 2: Generating thematic backbone via semantic path...")
        path_obj = find_semantic_path(
            start=start_concept,
            end=end_concept,
            steps=num_stanzas,
            method="bezier",  # Bezier method for curved, creative paths
        )

        # Extract the primary path (start + bridges + end)
        thematic_path = path_obj.get_primary_path()
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

        execution_time = time.time() - start_time

        logger.info(
            f"GeneratePoemScaffold complete: {len(scaffold.stanzas)} stanzas generated "
            f"in {execution_time:.2f}s"
        )

        # Track which generators were used
        generators_used = [
            "SemanticPath",
            "ConceptualCloud",
            "DefinitionalFinder",
            "MetaphorGenerator",
        ]
        if fingerprint_path:
            generators_used.append("PersonalizedLineSeedGenerator")
        else:
            generators_used.append("LineSeedGenerator")

        # Return StrategyResult with the scaffold as custom data
        return StrategyResult(
            strategy_name="generate_poem_scaffold",
            building_blocks=[],  # Scaffolds don't use building blocks
            execution_time=execution_time,
            generators_used=generators_used,
            metadata={
                "start_concept": start_concept,
                "end_concept": end_concept,
                "num_stanzas": num_stanzas,
                "thematic_path": thematic_path,
                "scaffold": scaffold,  # Include the actual scaffold
            },
            params=params,
        )

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
        palette_list = stanza_palette.get_all_terms()  # Already returns List[str]
        seed_words = [theme_word, *palette_list[:10]]  # Limit to avoid overwhelming API

        line_seeds = self.seed_generator.generate_seed_collection(
            seed_words=seed_words, num_seeds=3
        )  # Already returns List[LineSeed]

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

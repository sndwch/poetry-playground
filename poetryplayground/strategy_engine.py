"""Strategy Engine: Multi-generator orchestration system.

This module provides the infrastructure for executing "creative recipes" that
orchestrate multiple underlying generators (ConceptualCloud, LineSeedGenerator,
MetaphorGenerator, etc.) to fulfill high-level creative briefs.

The Strategy Engine:
1. Takes a creative brief (e.g., "Bridge 'rust' and 'forgiveness'")
2. Orchestrates multiple generators in parallel/sequential as needed
3. Normalizes disparate results into unified PoeticBuildingBlock format
4. Ranks results by quality score across all types
5. Returns a single, unified StrategyResult

Example:
    >>> engine = StrategyEngine()
    >>> result = engine.execute("bridge_two_concepts", {
    ...     'start_word': 'rust',
    ...     'end_word': 'forgiveness',
    ...     'seed_words': ['severe', 'quiet']
    ... })
    >>> for block in result.building_blocks[:5]:
    ...     print(f"{block.text} ({block.quality_score:.2f})")
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type


@dataclass
class PoeticBuildingBlock:
    """Universal container for cross-generator results.

    This dataclass normalizes output from different generators (LineSeed,
    CloudTerm, Metaphor, etc.) into a unified format that can be ranked
    by quality score and displayed consistently.

    Attributes:
        text: The actual poetic content ("ash", "But rust never forgives...")
        source_method: Generator that produced this (e.g., "ConceptualCloud")
        block_type: Category of content ("BridgeWord", "Imagery", "Metaphor", "LineSeed")
        quality_score: Universal quality score (0.0-1.0)
        metadata: Additional context (freq_bucket, POS, concreteness, etc.)
        original_object: The original LineSeed/CloudTerm/etc. object for rich display

    Example:
        >>> block = PoeticBuildingBlock(
        ...     text="ash",
        ...     source_method="ConceptualCloud",
        ...     block_type="Imagery",
        ...     quality_score=0.85,
        ...     metadata={"cluster_type": "IMAGERY", "concreteness": 0.96},
        ...     original_object=cloud_term
        ... )
    """

    text: str
    source_method: str
    block_type: str
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_object: Any = None


@dataclass
class StrategyResult:
    """Output from strategy execution.

    Contains the unified, ranked list of building blocks plus metadata about
    the execution (timing, generators used, etc.).

    Attributes:
        strategy_name: Name of the recipe that was run
        building_blocks: Ranked list of PoeticBuildingBlock objects
        execution_time: Time taken to execute (seconds)
        generators_used: List of generator names that were called
        metadata: Additional strategy-specific information
        params: Original parameters passed to the strategy

    Example:
        >>> result = StrategyResult(
        ...     strategy_name="bridge_two_concepts",
        ...     building_blocks=[...],  # 30 ranked blocks
        ...     execution_time=2.3,
        ...     generators_used=["SemanticPath", "ConceptualCloud", "LineSeedGenerator"],
        ...     metadata={"start_word": "rust", "end_word": "forgiveness"}
        ... )
    """

    strategy_name: str
    building_blocks: List[PoeticBuildingBlock]
    execution_time: float
    generators_used: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all strategy implementations.

    Strategies are "creative recipes" that orchestrate multiple generators
    to fulfill a high-level creative brief. Each strategy defines:
    1. Required parameters (via validate_params)
    2. Orchestration logic (via run)
    3. Normalization logic (via normalize_results)

    Subclasses should implement parallel/sequential execution as appropriate
    for their specific generator dependencies.
    """

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate strategy-specific parameters.

        Args:
            params: Dictionary of parameters for this strategy

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if params are valid
            - error_message: Empty string if valid, else error description

        Example:
            >>> strategy = BridgeTwoConceptsStrategy()
            >>> valid, error = strategy.validate_params({
            ...     'start_word': 'rust',
            ...     'end_word': 'forgiveness'
            ... })
            >>> assert valid
        """
        pass

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> StrategyResult:
        """Execute the strategy orchestration.

        This is the main entry point for the strategy. It should:
        1. Run generators (parallel/sequential as needed)
        2. Normalize results to PoeticBuildingBlock format
        3. Rank by quality score
        4. Return StrategyResult

        Args:
            params: Dictionary of validated parameters

        Returns:
            StrategyResult containing ranked building blocks

        Example:
            >>> strategy = BridgeTwoConceptsStrategy()
            >>> result = strategy.run({
            ...     'start_word': 'rust',
            ...     'end_word': 'forgiveness',
            ...     'seed_words': ['severe']
            ... })
            >>> assert len(result.building_blocks) > 0
        """
        pass

    def normalize_results(self, raw_results: Dict[str, Any]) -> List[PoeticBuildingBlock]:
        """Convert disparate generator outputs to PoeticBuildingBlock format.

        This method handles the complexity of converting different output types
        (LineSeed, CloudTerm, Metaphor, BridgeWord, etc.) into a unified format.

        Args:
            raw_results: Dictionary mapping generator names to their outputs
                Example: {
                    'semantic_path': [BridgeWord(...), ...],
                    'cloud_start': [CloudTerm(...), ...],
                    'metaphors': [Metaphor(...), ...]
                }

        Returns:
            List of PoeticBuildingBlock objects

        Example:
            >>> blocks = strategy.normalize_results({
            ...     'line_seeds': [LineSeed(text="...", quality_score=0.87, ...)],
            ...     'cloud_terms': [CloudTerm(term="ash", score=0.85, ...)]
            ... })
            >>> assert all(isinstance(b, PoeticBuildingBlock) for b in blocks)
        """
        blocks = []

        for generator_name, results in raw_results.items():
            if not results:
                continue

            # Handle list results
            if isinstance(results, list):
                for item in results:
                    block = self._normalize_single_item(item, generator_name)
                    if block:
                        blocks.append(block)
            # Handle single result
            else:
                block = self._normalize_single_item(results, generator_name)
                if block:
                    blocks.append(block)

        return blocks

    def _normalize_single_item(self, item: Any, generator_name: str) -> PoeticBuildingBlock | None:
        """Normalize a single item to PoeticBuildingBlock.

        Args:
            item: A LineSeed, CloudTerm, Metaphor, or other generator output
            generator_name: Name of the generator that produced this item

        Returns:
            PoeticBuildingBlock or None if item type is unknown
        """
        # Import here to avoid circular dependencies
        from poetryplayground.conceptual_cloud import CloudTerm
        from poetryplayground.line_seeds import LineSeed
        from poetryplayground.metaphor_generator import Metaphor

        # Handle CloudTerm
        if isinstance(item, CloudTerm):
            return PoeticBuildingBlock(
                text=item.term,
                source_method="ConceptualCloud",
                block_type=item.cluster_type.value
                if hasattr(item, "cluster_type")
                else "CloudTerm",
                quality_score=item.score,
                metadata={
                    "freq_bucket": item.freq_bucket,
                    "cluster_type": (
                        item.cluster_type.value if hasattr(item, "cluster_type") else "unknown"
                    ),
                    **item.metadata,
                },
                original_object=item,
            )

        # Handle LineSeed
        elif isinstance(item, LineSeed):
            return PoeticBuildingBlock(
                text=item.text,
                source_method="LineSeedGenerator",
                block_type=f"LineSeed-{item.seed_type.value}"
                if hasattr(item, "seed_type")
                else "LineSeed",
                quality_score=item.quality_score,
                metadata={
                    "seed_type": item.seed_type.value if hasattr(item, "seed_type") else "unknown",
                    "strategy": (
                        item.strategy.value if hasattr(item, "strategy") and item.strategy else None
                    ),
                    "momentum": item.momentum,
                    "openness": item.openness,
                    "notes": item.notes,
                },
                original_object=item,
            )

        # Handle Metaphor
        elif isinstance(item, Metaphor):
            return PoeticBuildingBlock(
                text=item.text,
                source_method="MetaphorGenerator",
                block_type="Metaphor",
                quality_score=item.quality_score,
                metadata={
                    "source": item.source,
                    "target": item.target,
                    "metaphor_type": item.metaphor_type.value,
                    "grounds": item.grounds or [],
                    "source_text": item.source_text,
                },
                original_object=item,
            )

        # Handle BridgeWord (from semantic_geodesic)
        elif hasattr(item, "word") and hasattr(item, "similarity"):
            return PoeticBuildingBlock(
                text=item.word,
                source_method="SemanticPath",
                block_type="BridgeWord",
                quality_score=item.similarity if hasattr(item, "similarity") else 0.5,
                metadata={
                    "similarity": item.similarity if hasattr(item, "similarity") else None,
                    "step": item.step if hasattr(item, "step") else None,
                },
                original_object=item,
            )

        # Unknown type - try to extract basic info
        elif hasattr(item, "text") or hasattr(item, "term") or hasattr(item, "word"):
            text = (
                getattr(item, "text", None)
                or getattr(item, "term", None)
                or getattr(item, "word", "")
            )
            score = getattr(item, "quality_score", None) or getattr(item, "score", 0.5)

            return PoeticBuildingBlock(
                text=str(text),
                source_method=generator_name,
                block_type="Unknown",
                quality_score=float(score),
                metadata={"original_type": type(item).__name__},
                original_object=item,
            )

        return None


class StrategyEngine:
    """Main controller for the Strategy Engine orchestration system.

    The StrategyEngine maintains a registry of available strategies and
    dispatches execution to the appropriate strategy class based on name.

    This implements a plugin-style architecture where new strategies can
    be registered at runtime.

    Example:
        >>> engine = StrategyEngine()
        >>> engine.register_strategy("bridge_two_concepts", BridgeTwoConceptsStrategy)
        >>> result = engine.execute("bridge_two_concepts", {
        ...     'start_word': 'rust',
        ...     'end_word': 'forgiveness'
        ... })
    """

    def __init__(self):
        """Initialize the Strategy Engine with empty strategy registry."""
        self.strategies: Dict[str, Type[BaseStrategy]] = {}

    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a strategy class for later execution.

        Args:
            name: Unique identifier for this strategy
            strategy_class: Class implementing BaseStrategy

        Raises:
            ValueError: If strategy_class doesn't inherit from BaseStrategy

        Example:
            >>> engine = StrategyEngine()
            >>> engine.register_strategy("my_strategy", MyStrategyClass)
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class.__name__} must inherit from BaseStrategy")

        self.strategies[name] = strategy_class

    def execute(self, strategy_name: str, params: Dict[str, Any]) -> StrategyResult:
        """Execute a registered strategy with the given parameters.

        This is the main entry point for the Strategy Engine. It:
        1. Looks up the strategy by name
        2. Validates parameters
        3. Instantiates the strategy
        4. Executes the strategy
        5. Returns results

        Args:
            strategy_name: Name of the registered strategy to run
            params: Dictionary of parameters for the strategy

        Returns:
            StrategyResult containing ranked building blocks

        Raises:
            ValueError: If strategy not registered or params invalid

        Example:
            >>> engine = StrategyEngine()
            >>> result = engine.execute("bridge_two_concepts", {
            ...     'start_word': 'rust',
            ...     'end_word': 'forgiveness'
            ... })
            >>> print(f"Generated {len(result.building_blocks)} building blocks")
        """
        # Check if strategy exists
        if strategy_name not in self.strategies:
            available = ", ".join(self.strategies.keys())
            raise ValueError(
                f"Strategy '{strategy_name}' not registered. "
                f"Available strategies: {available if available else 'none'}"
            )

        # Get strategy class and instantiate
        strategy_class = self.strategies[strategy_name]
        strategy = strategy_class()

        # Validate parameters
        is_valid, error_message = strategy.validate_params(params)
        if not is_valid:
            raise ValueError(f"Invalid parameters for strategy '{strategy_name}': {error_message}")

        # Execute strategy
        start_time = time.time()
        result = strategy.run(params)
        execution_time = time.time() - start_time

        # Update result with execution time if not already set
        if result.execution_time == 0.0:
            result.execution_time = execution_time

        return result

    def list_strategies(self) -> List[str]:
        """Get list of all registered strategy names.

        Returns:
            List of strategy names

        Example:
            >>> engine = StrategyEngine()
            >>> engine.register_strategy("bridge", BridgeStrategy)
            >>> engine.register_strategy("explore", ExploreStrategy)
            >>> print(engine.list_strategies())
            ['bridge', 'explore']
        """
        return list(self.strategies.keys())


# Global singleton instance
_engine_instance: StrategyEngine | None = None


def get_strategy_engine() -> StrategyEngine:
    """Get the global StrategyEngine singleton instance.

    This ensures all parts of the codebase share the same strategy registry.

    Returns:
        StrategyEngine singleton instance

    Example:
        >>> engine = get_strategy_engine()
        >>> result = engine.execute("bridge_two_concepts", {...})
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = StrategyEngine()
    return _engine_instance

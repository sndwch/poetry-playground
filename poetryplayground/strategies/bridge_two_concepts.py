"""Bridge Two Concepts Strategy: Semantic pathfinding and multi-generator orchestration.

This strategy takes two words (start and end) and bridges them using:
1. Semantic paths (transitional vocabulary)
2. Conceptual clouds (imagery and rare words for both concepts)
3. Metaphors (direct creative connections)
4. Line seeds (opening and pivot lines using discovered vocabulary)

The strategy uses hybrid execution:
- PARALLEL: Independent generators run concurrently
- SEQUENTIAL: Dependent generators run after parallel batch completes

Example:
    >>> strategy = BridgeTwoConceptsStrategy()
    >>> result = strategy.run({
    ...     'start_word': 'rust',
    ...     'end_word': 'forgiveness',
    ...     'seed_words': ['severe', 'quiet']
    ... })
    >>> print(f"{len(result.building_blocks)} building blocks generated")
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from poetryplayground.conceptual_cloud import ClusterType, generate_conceptual_cloud
from poetryplayground.line_seeds import LineSeedGenerator
from poetryplayground.metaphor_generator import MetaphorGenerator
from poetryplayground.semantic_geodesic import find_semantic_path
from poetryplayground.strategy_engine import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)


class BridgeTwoConceptsStrategy(BaseStrategy):
    """Strategy for bridging two concepts using multiple generators.

    This strategy orchestrates 6 generators in two phases:

    **Phase 1 - Parallel Batch (Independent generators)**:
    - Semantic path finding (start → end)
    - Conceptual cloud for start word
    - Conceptual cloud for end word
    - Metaphor generation (start → end)

    **Phase 2 - Sequential Batch (Dependent on Phase 1)**:
    - Opening line (using start + bridge words)
    - Pivot line (using all discovered vocabulary)

    The hybrid execution model respects dependencies while maximizing parallelism.

    Attributes:
        max_workers: Number of threads for parallel execution (default: 4)
        min_quality_threshold: Minimum quality score for results (default: 0.5)
    """

    def __init__(self, max_workers: int = 4, min_quality_threshold: float = 0.5):
        """Initialize the Bridge Two Concepts strategy.

        Args:
            max_workers: Maximum number of parallel threads for Phase 1
            min_quality_threshold: Minimum quality score to include in results
        """
        self.max_workers = max_workers
        self.min_quality_threshold = min_quality_threshold

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters for this strategy.

        Required params:
        - start_word (str): The starting concept
        - end_word (str): The ending concept

        Optional params:
        - seed_words (List[str]): Additional tone/context words
        - num_steps (int): Steps in semantic path (default: 5)
        - k_per_cluster (int): Terms per cloud cluster (default: 5)

        Args:
            params: Dictionary of parameters

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> strategy = BridgeTwoConceptsStrategy()
            >>> valid, error = strategy.validate_params({'start_word': 'rust'})
            >>> assert not valid  # Missing end_word
            >>> valid, error = strategy.validate_params({
            ...     'start_word': 'rust',
            ...     'end_word': 'forgiveness'
            ... })
            >>> assert valid
        """
        # Check required params
        if "start_word" not in params:
            return False, "Missing required parameter: start_word"

        if "end_word" not in params:
            return False, "Missing required parameter: end_word"

        # Validate types
        if not isinstance(params["start_word"], str) or not params["start_word"].strip():
            return False, "start_word must be a non-empty string"

        if not isinstance(params["end_word"], str) or not params["end_word"].strip():
            return False, "end_word must be a non-empty string"

        # Validate optional params if provided
        if "seed_words" in params:
            if not isinstance(params["seed_words"], list):
                return False, "seed_words must be a list"
            if not all(isinstance(w, str) for w in params["seed_words"]):
                return False, "All seed_words must be strings"

        if "num_steps" in params and (
            not isinstance(params["num_steps"], int) or params["num_steps"] < 3
        ):
            return False, "num_steps must be an integer >= 3"

        if "k_per_cluster" in params and (
            not isinstance(params["k_per_cluster"], int) or params["k_per_cluster"] < 1
        ):
            return False, "k_per_cluster must be an integer >= 1"

        return True, ""

    def run(self, params: Dict[str, Any]) -> StrategyResult:
        """Execute the Bridge Two Concepts strategy.

        Orchestration flow:
        1. Phase 1: Run independent generators in parallel
        2. Phase 2: Run dependent generators sequentially
        3. Normalize all results to PoeticBuildingBlock
        4. Rank by quality score
        5. Return StrategyResult

        Args:
            params: Validated parameters

        Returns:
            StrategyResult with ranked building blocks

        Example:
            >>> strategy = BridgeTwoConceptsStrategy()
            >>> result = strategy.run({
            ...     'start_word': 'rust',
            ...     'end_word': 'forgiveness',
            ...     'seed_words': ['severe']
            ... })
            >>> assert result.strategy_name == "bridge_two_concepts"
            >>> assert len(result.building_blocks) > 0
        """
        import time

        start_time = time.time()

        # Extract params with defaults
        start_word = params["start_word"].strip()
        end_word = params["end_word"].strip()
        seed_words = params.get("seed_words", [])
        num_steps = params.get("num_steps", 5)
        k_per_cluster = params.get("k_per_cluster", 5)

        logger.info(
            f"BridgeTwoConceptsStrategy: Bridging '{start_word}' → '{end_word}' "
            f"(seed_words={seed_words})"
        )

        # Phase 1: Parallel execution
        parallel_results = self._run_parallel_batch(start_word, end_word, num_steps, k_per_cluster)

        # Phase 2: Sequential execution (uses results from Phase 1)
        sequential_results = self._run_sequential_batch(
            start_word, end_word, seed_words, parallel_results
        )

        # Combine all results
        all_results = {**parallel_results, **sequential_results}

        # Normalize to PoeticBuildingBlock
        building_blocks = self.normalize_results(all_results)

        # Filter by quality threshold
        building_blocks = [
            b for b in building_blocks if b.quality_score >= self.min_quality_threshold
        ]

        # Rank by quality score (descending)
        building_blocks.sort(key=lambda b: b.quality_score, reverse=True)

        execution_time = time.time() - start_time

        # Track which generators were used
        generators_used = []
        if parallel_results.get("semantic_path"):
            generators_used.append("SemanticPath")
        if parallel_results.get("cloud_start"):
            generators_used.append("ConceptualCloud (start)")
        if parallel_results.get("cloud_end"):
            generators_used.append("ConceptualCloud (end)")
        if parallel_results.get("metaphors"):
            generators_used.append("MetaphorGenerator")
        if sequential_results.get("opening_line"):
            generators_used.append("LineSeedGenerator (opening)")
        if sequential_results.get("pivot_line"):
            generators_used.append("LineSeedGenerator (pivot)")

        logger.info(
            f"BridgeTwoConceptsStrategy: Generated {len(building_blocks)} blocks "
            f"in {execution_time:.2f}s"
        )

        return StrategyResult(
            strategy_name="bridge_two_concepts",
            building_blocks=building_blocks,
            execution_time=execution_time,
            generators_used=generators_used,
            metadata={
                "start_word": start_word,
                "end_word": end_word,
                "seed_words": seed_words,
                "num_steps": num_steps,
                "k_per_cluster": k_per_cluster,
                "min_quality_threshold": self.min_quality_threshold,
            },
            params=params,
        )

    def _run_parallel_batch(
        self, start_word: str, end_word: str, num_steps: int, k_per_cluster: int
    ) -> Dict[str, Any]:
        """Run independent generators in parallel using ThreadPoolExecutor.

        This phase runs 4 generators concurrently:
        1. Semantic path (start → end)
        2. Conceptual cloud for start word
        3. Conceptual cloud for end word
        4. Metaphor generation

        Args:
            start_word: Starting concept
            end_word: Ending concept
            num_steps: Steps in semantic path
            k_per_cluster: Terms per cloud cluster

        Returns:
            Dictionary mapping generator names to their outputs

        Example:
            >>> results = strategy._run_parallel_batch("rust", "forgiveness", 5, 5)
            >>> assert "semantic_path" in results
            >>> assert "cloud_start" in results
        """
        results = {}

        # Define tasks for parallel execution
        tasks = {
            "semantic_path": (self._get_semantic_path, (start_word, end_word, num_steps)),
            "cloud_start": (self._get_cloud, (start_word, k_per_cluster)),
            "cloud_end": (self._get_cloud, (end_word, k_per_cluster)),
            "metaphors": (self._get_metaphors, (start_word, end_word)),
        }

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(func, *args): name for name, (func, args) in tasks.items()}

            # Collect results as they complete
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    results[task_name] = result
                    logger.debug(f"Parallel task '{task_name}' completed")
                except Exception as e:
                    logger.error(f"Parallel task '{task_name}' failed: {e}")
                    results[task_name] = []  # Empty result on error

        return results

    def _run_sequential_batch(
        self,
        start_word: str,
        end_word: str,
        seed_words: List[str],
        parallel_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run dependent generators sequentially after parallel batch.

        This phase generates line seeds using vocabulary discovered in Phase 1:
        1. Opening line (using start word + bridge words)
        2. Pivot line (using all discovered terms)

        Args:
            start_word: Starting concept
            end_word: Ending concept
            seed_words: User-provided tone words
            parallel_results: Results from parallel batch (contains bridge words, cloud terms)

        Returns:
            Dictionary mapping generator names to their outputs

        Example:
            >>> parallel = {'semantic_path': [...], 'cloud_start': [...]}
            >>> results = strategy._run_sequential_batch("rust", "forgiveness", [], parallel)
            >>> assert "opening_line" in results
        """
        results = {}

        # Extract vocabulary from parallel results
        bridge_words = []
        if "semantic_path" in parallel_results:
            path_result = parallel_results["semantic_path"]
            if hasattr(path_result, "bridges"):
                # SemanticPath returns object with bridges attribute
                for bridge_list in path_result.bridges:
                    if bridge_list:
                        bridge_words.append(bridge_list[0].word)  # Take first alternative
            elif isinstance(path_result, list):
                # Or it might return list directly
                bridge_words.extend([str(item) for item in path_result[:3]])

        imagery_words = []
        for cloud_key in ["cloud_start", "cloud_end"]:
            if cloud_key in parallel_results:
                cloud = parallel_results[cloud_key]
                if hasattr(cloud, "clusters"):
                    # Get imagery terms
                    imagery_cluster = cloud.clusters.get(ClusterType.IMAGERY, [])
                    imagery_words.extend([term.term for term in imagery_cluster[:3]])

        # Generate opening line using start word + bridge words
        try:
            opening_seed_words = [start_word, *bridge_words[:2], *seed_words]
            opening = LineSeedGenerator().generate_opening_line(opening_seed_words)
            results["opening_line"] = [opening] if opening else []
            logger.debug("Sequential task 'opening_line' completed")
        except Exception as e:
            logger.error(f"Sequential task 'opening_line' failed: {e}")
            results["opening_line"] = []

        # Generate pivot line using all discovered vocabulary
        try:
            pivot_seed_words = [
                start_word,
                end_word,
                *bridge_words[:3],
                *imagery_words[:3],
                *seed_words,
            ]
            pivot = LineSeedGenerator().generate_pivot_line(pivot_seed_words)
            results["pivot_line"] = [pivot] if pivot else []
            logger.debug("Sequential task 'pivot_line' completed")
        except Exception as e:
            logger.error(f"Sequential task 'pivot_line' failed: {e}")
            results["pivot_line"] = []

        return results

    # -------------------------------------------------------------------------
    # Generator Wrapper Methods (Called by parallel/sequential batches)
    # -------------------------------------------------------------------------

    def _get_semantic_path(self, start_word: str, end_word: str, num_steps: int):
        """Get semantic path from start to end word.

        Args:
            start_word: Starting word
            end_word: Ending word
            num_steps: Number of intermediate steps

        Returns:
            SemanticPath object with bridge words
        """
        try:
            return find_semantic_path(
                start_word=start_word,
                end_word=end_word,
                steps=num_steps,
                method="linear",  # Fast and predictable
            )
        except Exception as e:
            logger.error(f"find_semantic_path failed: {e}")
            return None

    def _get_cloud(self, center_word: str, k_per_cluster: int):
        """Get conceptual cloud for a word.

        Args:
            center_word: Center word for the cloud
            k_per_cluster: Number of terms per cluster

        Returns:
            ConceptualCloud object with clusters
        """
        try:
            return generate_conceptual_cloud(
                center_word=center_word,
                k_per_cluster=k_per_cluster,
                total_limit=50,  # Reasonable limit
                sections=["imagery", "rare"],  # Focus on useful types
                include_scores=True,
                min_score=0.5,  # Quality filter
            )
        except Exception as e:
            logger.error(f"generate_conceptual_cloud failed for '{center_word}': {e}")
            return None

    def _get_metaphors(self, start_word: str, end_word: str):
        """Generate metaphors connecting start and end words.

        Args:
            start_word: Source of metaphor
            end_word: Target of metaphor

        Returns:
            List of Metaphor objects
        """
        try:
            generator = MetaphorGenerator()
            # Try to generate metaphors with these words
            metaphors = generator.generate_metaphor_batch([start_word, end_word], count=3)
            return metaphors
        except Exception as e:
            logger.error(f"MetaphorGenerator failed: {e}")
            return []

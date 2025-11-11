"""Strategy for extracting contextual samples from Project Gutenberg literature.

This strategy wraps the ContextualSampler to provide a formal strategy interface
that can be executed by the StrategyEngine and integrated with TUI/CLI.

Example:
    >>> from poetryplayground.strategies.contextual_sampler import ContextualSamplerStrategy
    >>> from poetryplayground.strategy_engine import get_strategy_engine
    >>>
    >>> engine = get_strategy_engine()
    >>> engine.register_strategy("contextual_sampler", ContextualSamplerStrategy)
    >>>
    >>> result = engine.execute("contextual_sampler", {
    ...     "search_term": "tomorrow",
    ...     "sample_count": 10,
    ...     "context_window": 200,
    ...     "min_quality": 0.5,
    ...     "transform_ratio": 0.0
    ... })
    >>> for block in result.building_blocks:
    ...     print(f"{block.quality_score:.2f}: {block.text}")
"""

import logging
import time
from typing import Any, Dict, List, Optional

from poetryplayground.contextual_sampler import ContextSample, ContextualSampler
from poetryplayground.core.document_library import document_library
from poetryplayground.core.quality_scorer import get_quality_scorer
from poetryplayground.logger import logger
from poetryplayground.ship_of_theseus import ShipOfTheseusTransformer
from poetryplayground.strategy_engine import BaseStrategy, PoeticBuildingBlock, StrategyResult

module_logger = logging.getLogger(__name__)


class ContextualSamplerStrategy(BaseStrategy):
    """Strategy for extracting context-aware samples from classic literature.

    This strategy searches for specific words or phrases in Project Gutenberg texts
    and extracts the surrounding context (sentences before and after) as creative prompts.
    Optionally applies Ship of Theseus transformation to mutate the samples.

    Parameters:
        search_term (str): Word or phrase to search for (required)
        sample_count (int): Number of samples to retrieve (1-100, default 20)
        context_window (int): Maximum characters of context (50-1000, default 200)
        min_quality (float): Minimum quality score (0.0-1.0, default 0.5)
        transform_ratio (float): Transformation intensity (0.0-1.0, default 0.0 = none)
        locc_codes (List[str]): Optional genre filters (PZ=Fiction, PR=English Lit, PS=American Lit)
    """

    def __init__(self):
        """Initialize the strategy."""
        self.sampler: Optional[ContextualSampler] = None
        self.transformer: Optional[ShipOfTheseusTransformer] = None

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters for this strategy.

        Args:
            params: Dictionary of parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameter
        if "search_term" not in params:
            return False, "Missing required parameter: search_term"

        search_term = params["search_term"]
        if not isinstance(search_term, str) or not search_term.strip():
            return False, "search_term must be a non-empty string"

        # Validate optional parameters
        sample_count = params.get("sample_count", 20)
        if not isinstance(sample_count, int) or sample_count < 1 or sample_count > 100:
            return False, "sample_count must be an integer between 1 and 100"

        context_window = params.get("context_window", 200)
        if not isinstance(context_window, int) or context_window < 50 or context_window > 1000:
            return False, "context_window must be an integer between 50 and 1000"

        min_quality = params.get("min_quality", 0.5)
        if not isinstance(min_quality, (int, float)) or min_quality < 0.0 or min_quality > 1.0:
            return False, "min_quality must be a number between 0.0 and 1.0"

        transform_ratio = params.get("transform_ratio", 0.0)
        if (
            not isinstance(transform_ratio, (int, float))
            or transform_ratio < 0.0
            or transform_ratio > 1.0
        ):
            return False, "transform_ratio must be a number between 0.0 and 1.0"

        # Validate locc_codes if provided
        locc_codes = params.get("locc_codes")
        if locc_codes is not None:
            if not isinstance(locc_codes, list):
                return False, "locc_codes must be a list of strings"
            valid_codes = {"PZ", "PR", "PS", "PT", "PQ"}  # Common literature codes
            for code in locc_codes:
                if not isinstance(code, str) or code.upper() not in valid_codes:
                    return (
                        False,
                        f"Invalid locc_code: {code}. Valid codes: {', '.join(sorted(valid_codes))}",
                    )

        return True, ""

    def run(self, params: Dict[str, Any]) -> StrategyResult:
        """Execute the contextual sampling strategy.

        Args:
            params: Validated parameters

        Returns:
            StrategyResult containing sampled building blocks

        Raises:
            ValueError: If parameters are invalid
        """
        start_time = time.time()

        # Extract parameters
        search_term = params["search_term"].strip()
        sample_count = params.get("sample_count", 20)
        context_window = params.get("context_window", 200)
        min_quality = params.get("min_quality", 0.5)
        transform_ratio = params.get("transform_ratio", 0.0)
        locc_codes = params.get("locc_codes")

        logger.info(
            f"ContextualSamplerStrategy: '{search_term}' "
            f"(samples={sample_count}, window={context_window}, quality>={min_quality:.2f})"
        )

        # Initialize components
        if self.sampler is None:
            quality_scorer = get_quality_scorer()
            self.sampler = ContextualSampler(
                document_library=document_library,
                quality_scorer=quality_scorer,
                transformer=None,  # We'll handle transformation separately
            )

        # Execute sampling
        try:
            samples = self.sampler.sample(
                search_term=search_term,
                sample_count=sample_count,
                context_window=context_window,
                min_quality=min_quality,
                locc_codes=locc_codes,
            )
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise

        logger.info(f"Retrieved {len(samples)} samples")

        # Convert samples to building blocks
        building_blocks = []
        generators_used = ["DocumentLibrary", "ContextualSampler"]

        for sample in samples:
            # Create original building block
            block = PoeticBuildingBlock(
                text=sample.text,
                source_method="ContextualSampler",
                block_type="ContextSample",
                quality_score=sample.quality_score,
                metadata={
                    "source_title": sample.source_title,
                    "source_author": sample.source_author,
                    "search_term": sample.search_term,
                    "match_position": sample.match_position,
                    "before_context": sample.before_context,
                    "after_context": sample.after_context,
                    "was_transformed": False,
                },
                original_object=sample,
            )
            building_blocks.append(block)

        # Apply transformation if requested
        if transform_ratio > 0.0:
            logger.info(f"Applying Ship of Theseus transformation (ratio={transform_ratio:.2f})...")
            transformed_blocks = self._apply_transformation(samples, transform_ratio)
            building_blocks.extend(transformed_blocks)
            generators_used.append("ShipOfTheseusTransformer")

        # Sort all blocks by quality (descending)
        building_blocks.sort(key=lambda b: b.quality_score, reverse=True)

        execution_time = time.time() - start_time

        logger.info(
            f"ContextualSamplerStrategy complete: {len(building_blocks)} blocks generated "
            f"in {execution_time:.2f}s"
        )

        # Return strategy result
        return StrategyResult(
            strategy_name="contextual_sampler",
            building_blocks=building_blocks,
            execution_time=execution_time,
            generators_used=generators_used,
            metadata={
                "search_term": search_term,
                "sample_count": sample_count,
                "context_window": context_window,
                "min_quality": min_quality,
                "transform_ratio": transform_ratio,
                "locc_codes": locc_codes,
                "original_samples_count": len(samples),
                "transformed_samples_count": len(building_blocks) - len(samples)
                if transform_ratio > 0.0
                else 0,
            },
            params=params,
        )

    def _apply_transformation(
        self, samples: List[ContextSample], transform_ratio: float
    ) -> List[PoeticBuildingBlock]:
        """Apply Ship of Theseus transformation to samples.

        Args:
            samples: List of original samples
            transform_ratio: Transformation intensity (0.0-1.0)

        Returns:
            List of transformed building blocks
        """
        # Initialize transformer if needed
        if self.transformer is None:
            self.transformer = ShipOfTheseusTransformer()

        transformed_blocks = []

        for sample in samples:
            try:
                # Transform the combined context text
                result = self.transformer.transform_line(
                    line=sample.text,
                    replacement_ratio=transform_ratio,
                    preserve_pos=True,
                    preserve_syllables=True,
                )

                # Calculate quality score for transformed text
                # Penalize slightly for transformation (less "authentic")
                quality_score = sample.quality_score * 0.9

                # Create building block for transformed version
                block = PoeticBuildingBlock(
                    text=result.transformed,
                    source_method="ShipOfTheseusTransformer",
                    block_type="TransformedContextSample",
                    quality_score=quality_score,
                    metadata={
                        "source_title": sample.source_title,
                        "source_author": sample.source_author,
                        "search_term": sample.search_term,
                        "original_text": sample.text,
                        "was_transformed": True,
                        "num_replacements": result.num_replacements,
                        "replacement_ratio": result.replacement_ratio,
                    },
                    original_object=result,
                )
                transformed_blocks.append(block)

            except Exception as e:
                logger.warning(f"Failed to transform sample from {sample.source_title}: {e}")
                continue

        logger.info(f"Successfully transformed {len(transformed_blocks)} samples")

        return transformed_blocks

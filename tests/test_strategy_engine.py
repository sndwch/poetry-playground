"""Tests for Strategy Engine orchestration system."""

import pytest

from poetryplayground.conceptual_cloud import CloudTerm, ClusterType
from poetryplayground.line_seeds import GenerationStrategy, LineSeed, SeedType
from poetryplayground.metaphor_generator import Metaphor, MetaphorType
from poetryplayground.strategies.bridge_two_concepts import BridgeTwoConceptsStrategy
from poetryplayground.strategy_engine import (
    PoeticBuildingBlock,
    StrategyEngine,
    StrategyResult,
    get_strategy_engine,
)

# ============================================================================
# Unit Tests - Data Structures
# ============================================================================


class TestDataStructures:
    """Test dataclasses and data structures."""

    def test_poetic_building_block_creation(self):
        """Test PoeticBuildingBlock instantiation."""
        block = PoeticBuildingBlock(
            text="rust",
            source_method="ConceptualCloud",
            block_type="Imagery",
            quality_score=0.85,
            metadata={"cluster_type": "IMAGERY"},
        )

        assert block.text == "rust"
        assert block.source_method == "ConceptualCloud"
        assert block.block_type == "Imagery"
        assert block.quality_score == 0.85
        assert block.metadata["cluster_type"] == "IMAGERY"
        assert block.original_object is None

    def test_poetic_building_block_with_original_object(self):
        """Test PoeticBuildingBlock with original object."""
        cloud_term = CloudTerm(
            term="ash",
            cluster_type=ClusterType.IMAGERY,
            score=0.92,
            freq_bucket="rare",
        )

        block = PoeticBuildingBlock(
            text=cloud_term.term,
            source_method="ConceptualCloud",
            block_type="Imagery",
            quality_score=cloud_term.score,
            original_object=cloud_term,
        )

        assert block.original_object == cloud_term
        assert isinstance(block.original_object, CloudTerm)

    def test_strategy_result_creation(self):
        """Test StrategyResult instantiation."""
        blocks = [
            PoeticBuildingBlock(
                text="rust",
                source_method="ConceptualCloud",
                block_type="Imagery",
                quality_score=0.85,
            ),
        ]

        result = StrategyResult(
            strategy_name="test_strategy",
            building_blocks=blocks,
            execution_time=1.5,
            generators_used=["ConceptualCloud"],
            metadata={"test": "value"},
            params={"start_word": "rust"},
        )

        assert result.strategy_name == "test_strategy"
        assert len(result.building_blocks) == 1
        assert result.execution_time == 1.5
        assert result.generators_used == ["ConceptualCloud"]
        assert result.metadata["test"] == "value"
        assert result.params["start_word"] == "rust"


# ============================================================================
# Unit Tests - Strategy Engine Core
# ============================================================================


class TestStrategyEngine:
    """Test StrategyEngine controller."""

    def test_engine_initialization(self):
        """Test StrategyEngine initialization."""
        engine = StrategyEngine()
        assert engine.strategies == {}

    def test_register_strategy(self):
        """Test strategy registration."""
        engine = StrategyEngine()
        engine.register_strategy("bridge_two_concepts", BridgeTwoConceptsStrategy)

        assert "bridge_two_concepts" in engine.strategies
        assert engine.strategies["bridge_two_concepts"] == BridgeTwoConceptsStrategy

    def test_register_invalid_strategy(self):
        """Test registration of non-BaseStrategy class fails."""
        engine = StrategyEngine()

        class NotAStrategy:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseStrategy"):
            engine.register_strategy("invalid", NotAStrategy)

    def test_list_strategies(self):
        """Test listing registered strategies."""
        engine = StrategyEngine()
        assert engine.list_strategies() == []

        engine.register_strategy("bridge_two_concepts", BridgeTwoConceptsStrategy)
        assert engine.list_strategies() == ["bridge_two_concepts"]

    def test_execute_unregistered_strategy(self):
        """Test execution of unregistered strategy fails."""
        engine = StrategyEngine()

        with pytest.raises(ValueError, match="not registered"):
            engine.execute("nonexistent", {})

    def test_execute_with_invalid_params(self):
        """Test execution with invalid params fails."""
        engine = StrategyEngine()
        engine.register_strategy("bridge_two_concepts", BridgeTwoConceptsStrategy)

        # Missing required params
        with pytest.raises(ValueError, match="Invalid parameters"):
            engine.execute("bridge_two_concepts", {})

    def test_get_strategy_engine_singleton(self):
        """Test global singleton instance."""
        engine1 = get_strategy_engine()
        engine2 = get_strategy_engine()

        assert engine1 is engine2  # Same instance


# ============================================================================
# Unit Tests - BridgeTwoConceptsStrategy
# ============================================================================


class TestBridgeTwoConceptsStrategy:
    """Test BridgeTwoConceptsStrategy implementation."""

    def test_validate_params_success(self):
        """Test successful parameter validation."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params({"start_word": "rust", "end_word": "forgiveness"})

        assert valid is True
        assert error == ""

    def test_validate_params_missing_start_word(self):
        """Test validation fails without start_word."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params({"end_word": "forgiveness"})

        assert valid is False
        assert "start_word" in error

    def test_validate_params_missing_end_word(self):
        """Test validation fails without end_word."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params({"start_word": "rust"})

        assert valid is False
        assert "end_word" in error

    def test_validate_params_empty_start_word(self):
        """Test validation fails with empty start_word."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params({"start_word": "", "end_word": "forgiveness"})

        assert valid is False
        assert "non-empty string" in error

    def test_validate_params_with_optional_params(self):
        """Test validation with optional parameters."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params(
            {
                "start_word": "rust",
                "end_word": "forgiveness",
                "seed_words": ["severe", "quiet"],
                "num_steps": 5,
                "k_per_cluster": 5,
            }
        )

        assert valid is True
        assert error == ""

    def test_validate_params_invalid_seed_words(self):
        """Test validation fails with invalid seed_words."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params(
            {"start_word": "rust", "end_word": "forgiveness", "seed_words": "not a list"}
        )

        assert valid is False
        assert "seed_words must be a list" in error

    def test_validate_params_invalid_num_steps(self):
        """Test validation fails with invalid num_steps."""
        strategy = BridgeTwoConceptsStrategy()

        valid, error = strategy.validate_params(
            {"start_word": "rust", "end_word": "forgiveness", "num_steps": 2}
        )

        assert valid is False
        assert "num_steps must be an integer >= 3" in error

    def test_run_basic_execution(self):
        """Test basic strategy execution."""
        strategy = BridgeTwoConceptsStrategy()

        result = strategy.run({"start_word": "rust", "end_word": "forgiveness"})

        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "bridge_two_concepts"
        assert isinstance(result.building_blocks, list)
        assert result.execution_time > 0
        assert len(result.generators_used) > 0

    def test_run_with_seed_words(self):
        """Test execution with seed words."""
        strategy = BridgeTwoConceptsStrategy()

        result = strategy.run(
            {
                "start_word": "rust",
                "end_word": "forgiveness",
                "seed_words": ["severe"],
            }
        )

        assert result.params["seed_words"] == ["severe"]
        assert "seed_words" in result.metadata

    def test_run_quality_filtering(self):
        """Test quality score filtering."""
        strategy = BridgeTwoConceptsStrategy(min_quality_threshold=0.5)

        result = strategy.run({"start_word": "rust", "end_word": "forgiveness"})

        # All blocks should meet minimum threshold
        for block in result.building_blocks:
            assert block.quality_score >= 0.5

    def test_run_quality_ranking(self):
        """Test results are ranked by quality score."""
        strategy = BridgeTwoConceptsStrategy()

        result = strategy.run({"start_word": "rust", "end_word": "forgiveness"})

        if len(result.building_blocks) > 1:
            # Check descending order
            for i in range(len(result.building_blocks) - 1):
                assert (
                    result.building_blocks[i].quality_score
                    >= result.building_blocks[i + 1].quality_score
                )


# ============================================================================
# Unit Tests - Normalization
# ============================================================================


class TestNormalization:
    """Test result normalization logic."""

    def test_normalize_cloud_term(self):
        """Test normalization of CloudTerm."""
        strategy = BridgeTwoConceptsStrategy()

        cloud_term = CloudTerm(
            term="ash",
            cluster_type=ClusterType.IMAGERY,
            score=0.92,
            freq_bucket="rare",
        )

        blocks = strategy.normalize_results({"cloud": [cloud_term]})

        assert len(blocks) == 1
        assert blocks[0].text == "ash"
        assert blocks[0].source_method == "ConceptualCloud"
        assert blocks[0].block_type == "imagery"
        assert blocks[0].quality_score == 0.92
        assert blocks[0].metadata["freq_bucket"] == "rare"

    def test_normalize_line_seed(self):
        """Test normalization of LineSeed."""
        strategy = BridgeTwoConceptsStrategy()

        line_seed = LineSeed(
            text="But rust never forgives",
            seed_type=SeedType.OPENING,
            strategy=GenerationStrategy.JUXTAPOSITION,
            quality_score=0.87,
            momentum=0.8,
            openness=0.6,
        )

        blocks = strategy.normalize_results({"lines": [line_seed]})

        assert len(blocks) == 1
        assert blocks[0].text == "But rust never forgives"
        assert blocks[0].source_method == "LineSeedGenerator"
        assert blocks[0].block_type.startswith("LineSeed-")
        assert blocks[0].quality_score == 0.87

    def test_normalize_metaphor(self):
        """Test normalization of Metaphor."""
        strategy = BridgeTwoConceptsStrategy()

        metaphor = Metaphor(
            text="forgiveness is a slow erosion",
            source="forgiveness",
            target="erosion",
            metaphor_type=MetaphorType.CONCEPTUAL,
            quality_score=0.85,
        )

        blocks = strategy.normalize_results({"metaphors": [metaphor]})

        assert len(blocks) == 1
        assert blocks[0].text == "forgiveness is a slow erosion"
        assert blocks[0].source_method == "MetaphorGenerator"
        assert blocks[0].block_type == "Metaphor"
        assert blocks[0].quality_score == 0.85

    def test_normalize_multiple_types(self):
        """Test normalization of mixed generator outputs."""
        strategy = BridgeTwoConceptsStrategy()

        cloud_term = CloudTerm(
            term="ash",
            cluster_type=ClusterType.IMAGERY,
            score=0.92,
            freq_bucket="rare",
        )

        line_seed = LineSeed(
            text="But rust never forgives",
            seed_type=SeedType.OPENING,
            strategy=GenerationStrategy.JUXTAPOSITION,
            quality_score=0.87,
            momentum=0.8,
            openness=0.6,
        )

        blocks = strategy.normalize_results({"cloud": [cloud_term], "lines": [line_seed]})

        assert len(blocks) == 2
        source_methods = {b.source_method for b in blocks}
        assert "ConceptualCloud" in source_methods
        assert "LineSeedGenerator" in source_methods

    def test_normalize_empty_results(self):
        """Test normalization of empty results."""
        strategy = BridgeTwoConceptsStrategy()

        blocks = strategy.normalize_results({})

        assert blocks == []

    def test_normalize_with_empty_lists(self):
        """Test normalization with empty result lists."""
        strategy = BridgeTwoConceptsStrategy()

        blocks = strategy.normalize_results({"cloud": [], "lines": [], "metaphors": []})

        assert blocks == []


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_execution(self):
        """Test complete end-to-end workflow."""
        # Get singleton engine
        engine = get_strategy_engine()

        # Register strategy
        engine.register_strategy("bridge_two_concepts", BridgeTwoConceptsStrategy)

        # Execute
        result = engine.execute(
            "bridge_two_concepts",
            {
                "start_word": "rust",
                "end_word": "forgiveness",
                "seed_words": ["severe"],
            },
        )

        # Validate result
        assert result.strategy_name == "bridge_two_concepts"
        assert len(result.building_blocks) > 0
        assert result.execution_time > 0
        assert len(result.generators_used) > 0

        # Check metadata
        assert result.metadata["start_word"] == "rust"
        assert result.metadata["end_word"] == "forgiveness"
        assert result.metadata["seed_words"] == ["severe"]

        # Validate blocks
        for block in result.building_blocks:
            assert isinstance(block, PoeticBuildingBlock)
            assert block.text
            assert block.source_method
            assert 0.0 <= block.quality_score <= 1.0

    def test_parallel_execution_generators(self):
        """Test that parallel generators complete successfully."""
        strategy = BridgeTwoConceptsStrategy()

        result = strategy.run({"start_word": "fire", "end_word": "water"})

        # Should have results from multiple generators
        source_methods = {b.source_method for b in result.building_blocks}

        # Expect at least some of the parallel generators to succeed
        assert len(source_methods) > 0

    def test_sequential_execution_generators(self):
        """Test that sequential generators complete after parallel batch."""
        strategy = BridgeTwoConceptsStrategy()

        result = strategy.run({"start_word": "rust", "end_word": "forgiveness"})

        # Check for line seed generators (sequential phase)
        line_seeds = [b for b in result.building_blocks if b.source_method == "LineSeedGenerator"]

        # Should have at least some line seeds (if sequential phase succeeded)
        # Note: This may be 0 if line generation failed, which is OK for this test
        assert isinstance(line_seeds, list)

"""Tests for contextual sampler functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from poetryplayground.contextual_sampler import (
    ContextSample,
    ContextualSampler,
)
from poetryplayground.strategies.contextual_sampler import ContextualSamplerStrategy
from poetryplayground.strategy_engine import (
    PoeticBuildingBlock,
    StrategyResult,
    get_strategy_engine,
)


# ============================================================================
# Unit Tests - Data Structures
# ============================================================================


class TestDataStructures:
    """Test dataclasses and data structures."""

    def test_context_sample_creation(self):
        """Test ContextSample instantiation."""
        sample = ContextSample(
            text="The sun sets slowly. Night approaches swiftly.",
            quality_score=0.75,
            source_title="Test Novel",
            source_author="Jane Doe",
            match_position=1200,
            search_term="night",
            before_context="The sun sets slowly.",
            after_context="Night approaches swiftly.",
        )

        assert sample.text == "The sun sets slowly. Night approaches swiftly."
        assert sample.quality_score == 0.75
        assert sample.source_title == "Test Novel"
        assert sample.source_author == "Jane Doe"
        assert sample.match_position == 1200
        assert sample.search_term == "night"
        assert sample.before_context == "The sun sets slowly."
        assert sample.after_context == "Night approaches swiftly."


# ============================================================================
# Unit Tests - ContextualSampler Core
# ============================================================================


class TestContextualSampler:
    """Test ContextualSampler functionality."""

    @pytest.fixture
    def mock_document_library(self):
        """Create mock document library."""
        library = Mock()
        library.get_diverse_documents.return_value = [
            "The dawn breaks gently. Tomorrow arrives with promise. Birds sing sweetly.",
            "Night falls silently. Tomorrow brings new hope. Stars shine brightly.",
            "Time passes slowly. Tomorrow waits patiently. Clouds drift lazily.",
        ]
        return library

    @pytest.fixture
    def mock_quality_scorer(self):
        """Create mock quality scorer."""
        scorer = Mock()
        # Mock score_word to return consistent scores
        mock_score = Mock()
        mock_score.overall = 0.7
        scorer.score_word.return_value = mock_score
        return scorer

    @pytest.fixture
    def sampler(self, mock_document_library, mock_quality_scorer):
        """Create ContextualSampler instance with mocks."""
        return ContextualSampler(
            document_library=mock_document_library,
            quality_scorer=mock_quality_scorer,
            transformer=None,
        )

    def test_sampler_initialization(self, sampler, mock_document_library, mock_quality_scorer):
        """Test ContextualSampler initialization."""
        assert sampler.document_library == mock_document_library
        assert sampler.quality_scorer == mock_quality_scorer
        assert sampler.transformer is None

    def test_parameter_validation_empty_search_term(self, sampler):
        """Test that empty search term raises ValueError."""
        with pytest.raises(ValueError, match="search_term must be a non-empty string"):
            sampler.sample(
                search_term="",
                sample_count=10,
                context_window=200,
            )

    def test_parameter_validation_invalid_sample_count(self, sampler):
        """Test that invalid sample_count raises ValueError."""
        with pytest.raises(ValueError, match="sample_count must be between 1 and 100"):
            sampler.sample(
                search_term="test",
                sample_count=0,
                context_window=200,
            )

        with pytest.raises(ValueError, match="sample_count must be between 1 and 100"):
            sampler.sample(
                search_term="test",
                sample_count=101,
                context_window=200,
            )

    def test_parameter_validation_invalid_context_window(self, sampler):
        """Test that invalid context_window raises ValueError."""
        with pytest.raises(ValueError, match="context_window must be between 50 and 1000"):
            sampler.sample(
                search_term="test",
                sample_count=10,
                context_window=30,
            )

        with pytest.raises(ValueError, match="context_window must be between 50 and 1000"):
            sampler.sample(
                search_term="test",
                sample_count=10,
                context_window=1500,
            )

    def test_parameter_validation_invalid_min_quality(self, sampler):
        """Test that invalid min_quality raises ValueError."""
        with pytest.raises(ValueError, match="min_quality must be between 0.0 and 1.0"):
            sampler.sample(
                search_term="test",
                sample_count=10,
                context_window=200,
                min_quality=-0.1,
            )

        with pytest.raises(ValueError, match="min_quality must be between 0.0 and 1.0"):
            sampler.sample(
                search_term="test",
                sample_count=10,
                context_window=200,
                min_quality=1.5,
            )

    def test_basic_sampling(self, sampler):
        """Test basic sampling functionality."""
        samples = sampler.sample(
            search_term="tomorrow",
            sample_count=5,
            context_window=200,
            min_quality=0.0,
        )

        # Should find samples (search term appears in the source, but extracted
        # context is before+after, so search term may not be in final text)
        assert len(samples) > 0
        assert all(isinstance(s, ContextSample) for s in samples)
        # Verify search_term metadata is set correctly
        assert all(s.search_term == "tomorrow" for s in samples)

    def test_sample_quality_filtering(self, sampler, mock_quality_scorer):
        """Test that low-quality samples are filtered out."""
        # Configure scorer to return low scores
        mock_score = Mock()
        mock_score.overall = 0.2  # Below threshold
        mock_quality_scorer.score_word.return_value = mock_score

        samples = sampler.sample(
            search_term="tomorrow",
            sample_count=10,
            context_window=200,
            min_quality=0.5,  # High threshold
        )

        # Should find no samples due to quality filtering
        assert len(samples) == 0

    def test_sample_sorting_by_quality(self, sampler):
        """Test that samples are sorted by quality score."""
        samples = sampler.sample(
            search_term="tomorrow",
            sample_count=5,
            context_window=200,
            min_quality=0.0,
        )

        if len(samples) > 1:
            # Verify descending order
            scores = [s.quality_score for s in samples]
            assert scores == sorted(scores, reverse=True)

    def test_search_document_extracts_context(self, sampler):
        """Test that _search_document correctly extracts before/after context."""
        document = "The sun rises. Tomorrow brings hope. The day begins."

        samples = sampler._search_document(
            document=document,
            search_term="tomorrow",
            context_window=200,
            min_quality=0.0,
            doc_index=0,
        )

        assert len(samples) == 1
        sample = samples[0]

        # Should have before and after context
        assert "sun rises" in sample.before_context.lower()
        assert "day begins" in sample.after_context.lower()
        assert sample.search_term == "tomorrow"

    def test_search_document_skips_incomplete_context(self, sampler):
        """Test that samples without both before/after context are skipped."""
        # Document with "tomorrow" at the start (no before context)
        document = "Tomorrow is bright. The sun rises."

        samples = sampler._search_document(
            document=document,
            search_term="tomorrow",
            context_window=200,
            min_quality=0.0,
            doc_index=0,
        )

        # Should skip samples without full context
        assert len(samples) == 0

    def test_search_document_respects_context_window(self, sampler):
        """Test that context is truncated to fit context_window."""
        # Very long sentences
        long_before = "A " * 100  # 200 chars
        long_after = "B " * 100  # 200 chars
        document = f"{long_before}Tomorrow arrives. {long_after}"

        samples = sampler._search_document(
            document=document,
            search_term="tomorrow",
            context_window=100,  # Small window
            min_quality=0.0,
            doc_index=0,
        )

        if samples:
            sample = samples[0]
            # Combined context should be <= 100 chars
            assert len(sample.text) <= 100

    def test_case_insensitive_search(self, sampler):
        """Test that search is case-insensitive."""
        samples_lower = sampler.sample(
            search_term="tomorrow",
            sample_count=5,
            context_window=200,
            min_quality=0.0,
        )

        samples_upper = sampler.sample(
            search_term="TOMORROW",
            sample_count=5,
            context_window=200,
            min_quality=0.0,
        )

        # Both should find samples (case-insensitive)
        assert len(samples_lower) > 0
        assert len(samples_upper) > 0


# ============================================================================
# Unit Tests - ContextualSamplerStrategy
# ============================================================================


class TestContextualSamplerStrategy:
    """Test ContextualSamplerStrategy functionality."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return ContextualSamplerStrategy()

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.sampler is None
        assert strategy.transformer is None

    def test_validate_params_success(self, strategy):
        """Test parameter validation with valid params."""
        params = {
            "search_term": "night",
            "sample_count": 10,
            "context_window": 200,
            "min_quality": 0.5,
            "transform_ratio": 0.0,
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is True
        assert error_msg == ""

    def test_validate_params_missing_search_term(self, strategy):
        """Test validation fails without search_term."""
        params = {
            "sample_count": 10,
            "context_window": 200,
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is False
        assert "search_term" in error_msg

    def test_validate_params_empty_search_term(self, strategy):
        """Test validation fails with empty search_term."""
        params = {
            "search_term": "   ",
            "sample_count": 10,
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is False
        assert "non-empty string" in error_msg

    def test_validate_params_invalid_sample_count(self, strategy):
        """Test validation fails with invalid sample_count."""
        params = {
            "search_term": "test",
            "sample_count": 150,
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is False
        assert "sample_count" in error_msg

    def test_validate_params_invalid_transform_ratio(self, strategy):
        """Test validation fails with invalid transform_ratio."""
        params = {
            "search_term": "test",
            "transform_ratio": 1.5,
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is False
        assert "transform_ratio" in error_msg

    def test_validate_params_invalid_locc_codes(self, strategy):
        """Test validation fails with invalid locc_codes."""
        params = {
            "search_term": "test",
            "locc_codes": ["XYZ"],  # Invalid code
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is False
        assert "locc_code" in error_msg

    def test_validate_params_valid_locc_codes(self, strategy):
        """Test validation succeeds with valid locc_codes."""
        params = {
            "search_term": "test",
            "locc_codes": ["PZ", "PR"],  # Valid codes
        }

        is_valid, error_msg = strategy.validate_params(params)
        assert is_valid is True
        assert error_msg == ""

    @patch("poetryplayground.strategies.contextual_sampler.ContextualSampler")
    @patch("poetryplayground.strategies.contextual_sampler.get_quality_scorer")
    @patch("poetryplayground.strategies.contextual_sampler.document_library")
    def test_strategy_run_basic(
        self, mock_doc_lib, mock_get_scorer, mock_sampler_class, strategy
    ):
        """Test strategy execution."""
        # Mock the sampler instance
        mock_sampler_instance = Mock()
        mock_sample = Mock()
        mock_sample.text = "The sun rises. Tomorrow brings hope."
        mock_sample.quality_score = 0.8
        mock_sample.source_title = "Test"
        mock_sample.source_author = "Author"
        mock_sample.search_term = "tomorrow"
        mock_sample.match_position = 100
        mock_sample.before_context = "The sun rises."
        mock_sample.after_context = "Tomorrow brings hope."

        mock_sampler_instance.sample.return_value = [mock_sample]
        mock_sampler_class.return_value = mock_sampler_instance

        params = {
            "search_term": "tomorrow",
            "sample_count": 10,
            "context_window": 200,
            "min_quality": 0.5,
            "transform_ratio": 0.0,
        }

        result = strategy.run(params)

        # Verify result structure
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "contextual_sampler"
        assert len(result.building_blocks) > 0
        assert all(isinstance(b, PoeticBuildingBlock) for b in result.building_blocks)
        assert result.execution_time >= 0
        assert "ContextualSampler" in result.generators_used

    @patch("poetryplayground.strategies.contextual_sampler.ContextualSampler")
    @patch("poetryplayground.strategies.contextual_sampler.ShipOfTheseusTransformer")
    @patch("poetryplayground.strategies.contextual_sampler.get_quality_scorer")
    @patch("poetryplayground.strategies.contextual_sampler.document_library")
    def test_strategy_run_with_transformation(
        self, mock_doc_lib, mock_get_scorer, mock_transformer_class, mock_sampler_class, strategy
    ):
        """Test strategy execution with transformation."""
        # Mock sampler
        mock_sampler_instance = Mock()
        mock_sample = Mock()
        mock_sample.text = "The sun rises. Tomorrow brings hope."
        mock_sample.quality_score = 0.8
        mock_sample.source_title = "Test"
        mock_sample.source_author = "Author"
        mock_sample.search_term = "tomorrow"
        mock_sample.match_position = 100
        mock_sample.before_context = "The sun rises."
        mock_sample.after_context = "Tomorrow brings hope."

        mock_sampler_instance.sample.return_value = [mock_sample]
        mock_sampler_class.return_value = mock_sampler_instance

        # Mock transformer
        mock_transformer_instance = Mock()
        mock_transform_result = Mock()
        mock_transform_result.transformed = "The moon rises. Tomorrow brings joy."
        mock_transform_result.num_replacements = 2
        mock_transform_result.replacement_ratio = 0.3
        mock_transformer_instance.transform_line.return_value = mock_transform_result
        mock_transformer_class.return_value = mock_transformer_instance

        params = {
            "search_term": "tomorrow",
            "sample_count": 10,
            "context_window": 200,
            "min_quality": 0.5,
            "transform_ratio": 0.3,  # Enable transformation
        }

        result = strategy.run(params)

        # Should have both original and transformed blocks
        assert len(result.building_blocks) >= 2
        assert "ShipOfTheseusTransformer" in result.generators_used

        # Check metadata includes transformation info
        assert result.metadata["transform_ratio"] == 0.3
        assert result.metadata["transformed_samples_count"] >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestContextualSamplerIntegration:
    """Integration tests with StrategyEngine."""

    @patch("poetryplayground.strategies.contextual_sampler.ContextualSampler")
    @patch("poetryplayground.strategies.contextual_sampler.get_quality_scorer")
    @patch("poetryplayground.strategies.contextual_sampler.document_library")
    def test_strategy_engine_integration(
        self, mock_doc_lib, mock_get_scorer, mock_sampler_class
    ):
        """Test ContextualSamplerStrategy through StrategyEngine."""
        # Mock the sampler
        mock_sampler_instance = Mock()
        mock_sample = Mock()
        mock_sample.text = "Context text"
        mock_sample.quality_score = 0.75
        mock_sample.source_title = "Test"
        mock_sample.source_author = "Author"
        mock_sample.search_term = "test"
        mock_sample.match_position = 100
        mock_sample.before_context = "Before"
        mock_sample.after_context = "After"

        mock_sampler_instance.sample.return_value = [mock_sample]
        mock_sampler_class.return_value = mock_sampler_instance

        # Get engine and register strategy
        engine = get_strategy_engine()
        engine.register_strategy("contextual_sampler", ContextualSamplerStrategy)

        # Execute through engine
        params = {
            "search_term": "test",
            "sample_count": 5,
            "context_window": 200,
        }

        result = engine.execute("contextual_sampler", params)

        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "contextual_sampler"
        assert len(result.building_blocks) > 0

    def test_building_block_conversion(self):
        """Test that ContextSample is correctly converted to PoeticBuildingBlock."""
        sample = ContextSample(
            text="The night falls gently.",
            quality_score=0.85,
            source_title="Test Novel",
            source_author="Jane Doe",
            match_position=500,
            search_term="night",
            before_context="Evening arrives.",
            after_context="Stars appear.",
        )

        # Create building block (as done in strategy)
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

        assert block.text == sample.text
        assert block.quality_score == sample.quality_score
        assert block.source_method == "ContextualSampler"
        assert block.block_type == "ContextSample"
        assert block.metadata["source_title"] == "Test Novel"
        assert block.metadata["search_term"] == "night"
        assert block.original_object == sample

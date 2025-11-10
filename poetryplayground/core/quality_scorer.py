"""Unified quality scoring system for generated poetry content.

This module provides a centralized framework for evaluating the quality of
words, phrases, and metaphors across all generation modules. It ensures
consistent quality standards and enables fine-grained quality control.

The scoring system considers multiple dimensions:
- Frequency: Balance between too common (boring) and too rare (obscure)
- Novelty: Avoid clichés and overused expressions
- Coherence: Semantic fit within context
- Register: Appropriate formality/tone for poetic use
- Imagery: Concreteness vs abstractness balance

Example:
    >>> scorer = QualityScorer()
    >>> context = GenerationContext(emotional_tone="dark", concreteness_target=0.7)
    >>> score = scorer.score_word("shadow", context)
    >>> print(f"Overall: {score.overall:.2f}, Novelty: {score.novelty:.2f}")
"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set

from wordfreq import word_frequency

from ..logger import logger


class EmotionalTone(Enum):
    """Emotional tone classifications."""

    DARK = "dark"
    LIGHT = "light"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class FormalityLevel(Enum):
    """Formality/register levels."""

    ARCHAIC = "archaic"
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    CASUAL = "casual"


@dataclass
class GenerationContext:
    """Context information for quality scoring.

    Attributes:
        emotional_tone: Desired emotional register
        concreteness_target: Target concreteness (0=abstract, 1=concrete)
        formality_level: Desired formality level
        avoid_cliches: Whether to strictly avoid clichéd expressions
        domain: Optional semantic domain (e.g., "nature", "urban", "emotion")
    """

    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    concreteness_target: float = 0.5  # 0=abstract, 1=concrete
    formality_level: FormalityLevel = FormalityLevel.CONVERSATIONAL
    avoid_cliches: bool = True
    domain: Optional[str] = None


@dataclass
class QualityScore:
    """Comprehensive quality score breakdown.

    All scores are in 0-1 range where 1.0 is highest quality.

    Attributes:
        overall: Weighted combination of all sub-scores
        frequency: Appropriateness of word frequency (0.25-0.75 = ideal)
        novelty: Avoidance of clichés and overused expressions
        coherence: Semantic fit with context
        register: Appropriateness of formality/tone
        imagery: Concreteness score when imagery is desired
        component_scores: Dict of all individual component scores
    """

    overall: float
    frequency: float
    novelty: float
    coherence: float
    register: float
    imagery: float
    component_scores: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable quality score."""
        return (
            f"QualityScore(overall={self.overall:.2f}, "
            f"freq={self.frequency:.2f}, "
            f"novelty={self.novelty:.2f}, "
            f"coherence={self.coherence:.2f})"
        )

    def get_grade(self) -> str:
        """Get letter grade for overall score."""
        if self.overall >= 0.9:
            return "A+"
        elif self.overall >= 0.8:
            return "A"
        elif self.overall >= 0.7:
            return "B"
        elif self.overall >= 0.6:
            return "C"
        elif self.overall >= 0.5:
            return "D"
        else:
            return "F"


class QualityScorer:
    """Unified quality scoring engine for poetry generation.

    This class provides methods to evaluate words, phrases, and metaphors
    using multiple quality dimensions. It loads external data files for
    cliché detection and concreteness ratings.

    Attributes:
        cliche_phrases: Set of known clichéd phrases/patterns
        cliche_words: Set of overused poetry words
        concreteness_cache: Cache of word concreteness ratings
        min_frequency: Minimum acceptable word frequency
        max_frequency: Maximum acceptable word frequency
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize quality scorer.

        Args:
            data_dir: Directory containing quality data files
        """
        if data_dir is None:
            # Default to data/ subdirectory (parent.parent since we're in core/)
            data_dir = Path(__file__).parent.parent / "data"

        self.data_dir = data_dir
        self.cliche_phrases: Set[str] = set()
        self.cliche_words: Set[str] = set()
        self.concreteness_cache: Dict[str, float] = {}

        # Frequency thresholds (Zipf scale, 0-8)
        # Too rare: < 3.0, Too common: > 6.0, Sweet spot: 4.0-5.5
        self.min_frequency = 1e-07  # Very rare
        self.max_frequency = 1e-03  # Very common
        self.ideal_min = 5e-06  # Lower bound of ideal range
        self.ideal_max = 5e-05  # Upper bound of ideal range

        # Load quality data files
        self._load_cliche_database()
        self._load_concreteness_ratings()

        logger.info("QualityScorer initialized")

    def _load_cliche_database(self) -> None:
        """Load cliché database from JSON file."""
        cliche_file = self.data_dir / "poetry_cliches.json"

        if not cliche_file.exists():
            logger.warning(f"Cliché database not found at {cliche_file}")
            # Use minimal built-in set
            self.cliche_phrases = {
                "life is a journey",
                "love is a rose",
                "time is a river",
                "heart of stone",
                "mind like an ocean",
                "death is sleep",
            }
            self.cliche_words = {
                "heart",
                "soul",
                "love",
                "dream",
                "tears",
                "broken",
                "forever",
                "eternity",
                "destiny",
                "fate",
            }
            return

        try:
            with open(cliche_file, encoding="utf-8") as f:
                data = json.load(f)
                self.cliche_phrases = set(data.get("phrases", []))
                self.cliche_words = set(data.get("words", []))
                logger.info(
                    f"Loaded {len(self.cliche_phrases)} cliché phrases, "
                    f"{len(self.cliche_words)} cliché words"
                )
        except Exception as e:
            logger.error(f"Error loading cliché database: {e}")
            self.cliche_phrases = set()
            self.cliche_words = set()

    def _load_concreteness_ratings(self) -> None:
        """Load Brysbaert concreteness ratings."""
        concreteness_file = self.data_dir / "concreteness_ratings.txt"

        if not concreteness_file.exists():
            logger.warning(f"Concreteness ratings not found at {concreteness_file}")
            return

        try:
            with open(concreteness_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        try:
                            score = float(parts[1])
                            # Normalize to 0-1 if needed (Brysbaert is 1-5)
                            if score > 1.0:
                                score = (score - 1.0) / 4.0
                            self.concreteness_cache[word] = score
                        except ValueError:
                            continue

            logger.info(f"Loaded {len(self.concreteness_cache)} concreteness ratings")
        except Exception as e:
            logger.error(f"Error loading concreteness ratings: {e}")

    def score_word(self, word: str, context: Optional[GenerationContext] = None) -> QualityScore:
        """Score a single word's quality.

        Args:
            word: Word to evaluate
            context: Optional context for contextual scoring

        Returns:
            QualityScore with breakdown of all components
        """
        if context is None:
            context = GenerationContext()

        word_lower = word.lower()

        # Calculate component scores
        freq_score = self._score_frequency(word_lower)
        novelty_score = self._score_novelty_word(word_lower)
        register_score = self._score_register(word_lower, context.formality_level)
        imagery_score = self._score_imagery(word_lower, context.concreteness_target)

        # Coherence for single words is based on domain fit (if specified)
        coherence_score = 0.8  # Neutral default for single words

        # Calculate weighted overall score
        # Weights: frequency (0.2), novelty (0.3), register (0.15),
        #          imagery (0.25), coherence (0.1)
        overall = (
            freq_score * 0.2
            + novelty_score * 0.3
            + register_score * 0.15
            + imagery_score * 0.25
            + coherence_score * 0.1
        )

        return QualityScore(
            overall=overall,
            frequency=freq_score,
            novelty=novelty_score,
            coherence=coherence_score,
            register=register_score,
            imagery=imagery_score,
            component_scores={
                "frequency": freq_score,
                "novelty": novelty_score,
                "register": register_score,
                "imagery": imagery_score,
                "coherence": coherence_score,
            },
        )

    def score_phrase(
        self, phrase: str, context: Optional[GenerationContext] = None
    ) -> QualityScore:
        """Score a phrase or multi-word expression.

        Args:
            phrase: Phrase to evaluate
            context: Optional context for contextual scoring

        Returns:
            QualityScore with breakdown of all components
        """
        if context is None:
            context = GenerationContext()

        phrase_lower = phrase.lower().strip()

        # Calculate component scores
        freq_score = self._score_phrase_frequency(phrase_lower)
        novelty_score = self._score_novelty_phrase(phrase_lower)
        coherence_score = 0.7  # Default, can be enhanced with semantic analysis
        register_score = 0.7  # Average of word-level register scores

        # Score individual words for imagery
        words = phrase_lower.split()
        if words:
            word_imagery_scores = [
                self._score_imagery(w, context.concreteness_target) for w in words
            ]
            imagery_score = sum(word_imagery_scores) / len(word_imagery_scores)
        else:
            imagery_score = 0.5

        # Calculate weighted overall score
        overall = (
            freq_score * 0.15
            + novelty_score * 0.35  # Higher weight for phrases (more cliché risk)
            + register_score * 0.1
            + imagery_score * 0.25
            + coherence_score * 0.15
        )

        return QualityScore(
            overall=overall,
            frequency=freq_score,
            novelty=novelty_score,
            coherence=coherence_score,
            register=register_score,
            imagery=imagery_score,
        )

    def _score_frequency(self, word: str) -> float:
        """Score word frequency (too common or too rare = bad).

        Returns score in 0-1 range where 1.0 is ideal frequency.
        """
        freq = word_frequency(word, "en")

        if freq == 0:
            return 0.3  # Unknown word, penalize

        # Ideal range: 5e-06 to 5e-05 (moderately common)
        if self.ideal_min <= freq <= self.ideal_max:
            return 1.0

        # Too rare
        if freq < self.ideal_min:
            # Scale from min_frequency (0.2) to ideal_min (1.0)
            if freq < self.min_frequency:
                return 0.2
            ratio = math.log10(freq / self.min_frequency) / math.log10(
                self.ideal_min / self.min_frequency
            )
            return 0.2 + (ratio * 0.8)

        # Too common
        if freq > self.ideal_max:
            # Scale from ideal_max (1.0) to max_frequency (0.3)
            if freq > self.max_frequency:
                return 0.3
            ratio = math.log10(freq / self.ideal_max) / math.log10(
                self.max_frequency / self.ideal_max
            )
            return 1.0 - (ratio * 0.7)

        return 0.8  # Shouldn't reach here, but reasonable default

    def _score_phrase_frequency(self, phrase: str) -> float:
        """Score phrase frequency based on component words."""
        words = phrase.split()
        if not words:
            return 0.5

        word_scores = [self._score_frequency(w) for w in words]
        return sum(word_scores) / len(word_scores)

    def _score_novelty_word(self, word: str) -> float:
        """Score word novelty (avoidance of clichés).

        Returns 0.0-1.0 where 1.0 = fresh, 0.0 = clichéd.
        """
        if word in self.cliche_words:
            return 0.0  # Completely clichéd

        # Partial penalty for words containing cliché stems
        for cliche in self.cliche_words:
            if cliche in word or word in cliche:
                return 0.4

        return 1.0  # Fresh word

    def _score_novelty_phrase(self, phrase: str) -> float:
        """Score phrase novelty (avoidance of clichéd expressions).

        Returns 0.0-1.0 where 1.0 = fresh, 0.0 = clichéd.
        """
        phrase_lower = phrase.lower().strip()

        # Exact match
        if phrase_lower in self.cliche_phrases:
            return 0.0

        # Check for partial matches
        max_penalty = 0.0
        for cliche in self.cliche_phrases:
            # Check if cliché phrase appears in our phrase
            if cliche in phrase_lower:
                max_penalty = max(max_penalty, 0.8)  # Heavy penalty

            # Check for word overlap (partial cliché)
            cliche_words = set(cliche.split())
            phrase_words = set(phrase_lower.split())
            overlap = len(cliche_words & phrase_words)
            if overlap >= 2:  # At least 2 words match
                overlap_ratio = overlap / len(cliche_words)
                penalty = overlap_ratio * 0.6
                max_penalty = max(max_penalty, penalty)

        return 1.0 - max_penalty

    def _score_register(self, word: str, target_level: FormalityLevel) -> float:
        """Score register/formality appropriateness.

        Currently simplified - can be enhanced with corpus analysis.
        """
        # TODO: Implement detailed register classification
        # For now, return neutral score
        return 0.7

    def _score_imagery(self, word: str, target_concreteness: float) -> float:
        """Score imagery quality based on concreteness.

        Args:
            word: Word to score
            target_concreteness: Target concreteness (0=abstract, 1=concrete)

        Returns:
            Score 0-1 where 1.0 means perfect match to target
        """
        # Get concreteness rating
        concreteness = self.concreteness_cache.get(word.lower(), 0.5)  # Default to neutral

        # Calculate distance from target
        distance = abs(concreteness - target_concreteness)

        # Convert distance to score (closer = better)
        # Maximum distance is 1.0, score should be 0.0
        # Distance 0.0 should give score 1.0
        score = 1.0 - distance

        return max(0.0, min(1.0, score))

    def is_cliche(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is clichéd.

        Args:
            text: Text to check (word or phrase)
            threshold: Novelty threshold below which text is considered cliché

        Returns:
            True if clichéd, False otherwise
        """
        if " " in text.strip():
            # Phrase
            score = self._score_novelty_phrase(text)
        else:
            # Single word
            score = self._score_novelty_word(text)

        return score < threshold

    def get_concreteness(self, word: str) -> float:
        """Get concreteness rating for a word.

        Args:
            word: Word to look up

        Returns:
            Concreteness score 0.0-1.0 (0=abstract, 1=concrete)
        """
        return self.concreteness_cache.get(word.lower(), 0.5)


# Global scorer instance
_scorer_instance = None


def get_quality_scorer() -> QualityScorer:
    """Get global QualityScorer instance (singleton pattern)."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = QualityScorer()
    return _scorer_instance

"""Line and Phrase Seed Generator for poetry ideation.

This module generates evocative incomplete phrases and line beginnings
that serve as creative catalysts rather than finished products.
"""

import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .lexigen import (
    similar_sounding_words,
    similar_meaning_words,
    contextually_linked_words,
    frequently_following_words,
    phonetically_related_words,
    related_rare_words
)
from .utils import filter_word_list, validate_word

from .word_validator import word_validator


class SeedType(Enum):
    """Types of line seeds that can be generated."""
    OPENING = "opening"
    PIVOT = "pivot"
    IMAGE = "image"
    EMOTIONAL = "emotional"
    SONIC = "sonic"
    CLOSING = "closing"
    FRAGMENT = "fragment"


class GenerationStrategy(Enum):
    """Strategies for generating line seeds."""
    JUXTAPOSITION = "juxtaposition"
    SYNESTHESIA = "synesthesia"
    INCOMPLETE_METAPHOR = "incomplete_metaphor"
    RHYTHMIC_BREAK = "rhythmic_break"
    QUESTION_IMPLIED = "question_implied"
    TEMPORAL_SHIFT = "temporal_shift"
    PERSPECTIVE_BLUR = "perspective_blur"


@dataclass
class LineSeed:
    """A line seed with metadata."""
    text: str
    seed_type: SeedType
    strategy: Optional[GenerationStrategy]
    momentum: float  # 0-1, how much forward movement it creates
    openness: float  # 0-1, how many directions it could go
    quality_score: float  # 0-1, overall quality
    notes: Optional[str] = None


class LineSeedGenerator:
    """Generate evocative line beginnings and fragments for poetry."""

    def __init__(self):
        """Initialize the line seed generator."""
        self._init_patterns()
        self._init_connectives()
        self._init_sensory_words()

    def _init_patterns(self):
        """Initialize phrase patterns for different seed types."""
        self.opening_patterns = [
            "The {noun} {verb} {adjective} before...",
            "In {adjective} {noun}, {pronoun} {verb}...",
            "{Temporal} the {noun} {verb}...",
            "Where {noun} meets {noun}, the {noun}...",
            "Through {adjective} {noun}, something {adjective}...",
            "{Pronoun} {verb} the {noun} of {abstract}..."
        ]

        self.fragment_patterns = [
            "...{preposition} {adjective} {noun}...",
            "...the {noun}'s {noun}...",
            "...{verb}ing what we {verb}...",
            "...{adjective}, {comparative} {adjective}...",
            "...through {noun}, {noun}..."
        ]

        self.closing_approaches = [
            "return_to_opening",
            "sudden_concrete",
            "open_question",
            "sensory_snapshot",
            "emotional_truth",
            "paradox"
        ]

    def _init_connectives(self):
        """Initialize connecting words and phrases."""
        self.temporal_markers = [
            "Before", "After", "When", "While", "During",
            "Until", "Since", "As", "Once", "Still"
        ]

        self.spatial_prepositions = [
            "beneath", "between", "through", "across",
            "beyond", "within", "against", "beside"
        ]

        self.abstract_connectors = [
            "almost", "perhaps", "somehow", "barely",
            "nearly", "always", "never", "still"
        ]

    def _init_sensory_words(self):
        """Initialize sensory vocabulary."""
        self.sensory_map = {
            'sight': ['glimmer', 'shadow', 'gleam', 'blur', 'shimmer'],
            'sound': ['whisper', 'echo', 'hum', 'rustle', 'murmur'],
            'touch': ['smooth', 'rough', 'cold', 'warm', 'sharp'],
            'smell': ['smoke', 'rain', 'metal', 'earth', 'salt'],
            'taste': ['bitter', 'sweet', 'acid', 'iron', 'ash']
        }

    def generate_opening_line(self, seed_words: List[str],
                            mood: Optional[str] = None) -> LineSeed:
        """Generate a strong opening line with forward momentum.

        Args:
            seed_words: Words to base generation on
            mood: Optional mood/tone for the line

        Returns:
            LineSeed with opening line
        """
        strategy = random.choice(list(GenerationStrategy))

        if strategy == GenerationStrategy.JUXTAPOSITION:
            line = self._generate_juxtaposition_opening(seed_words)
        elif strategy == GenerationStrategy.TEMPORAL_SHIFT:
            line = self._generate_temporal_opening(seed_words)
        elif strategy == GenerationStrategy.INCOMPLETE_METAPHOR:
            line = self._generate_metaphor_opening(seed_words)
        else:
            line = self._generate_standard_opening(seed_words)

        quality = self._evaluate_quality(line)

        return LineSeed(
            text=line,
            seed_type=SeedType.OPENING,
            strategy=strategy,
            momentum=random.uniform(0.7, 0.95),
            openness=random.uniform(0.6, 0.9),
            quality_score=quality
        )

    def generate_fragment(self, seed_words: List[str],
                         position: str = 'any') -> LineSeed:
        """Generate an evocative incomplete fragment.

        Args:
            seed_words: Words to base generation on
            position: Where in poem ('opening', 'middle', 'closing', 'any')

        Returns:
            LineSeed with fragment
        """
        # Get related words for variety
        expanded_words = []
        for word in seed_words[:2]:  # Limit for performance
            expanded_words.extend(similar_meaning_words(word, sample_size=3))
            expanded_words.extend(phonetically_related_words(word, sample_size=2))

        expanded_words = word_validator.clean_word_list(expanded_words)

        # Combine with seed words, ensuring we have something to work with
        all_words = seed_words + expanded_words
        if not all_words:
            all_words = ['word', 'phrase', 'thought']  # Emergency fallback

        # Build fragment
        pattern = random.choice(self.fragment_patterns)
        fragment = self._fill_pattern_fragment(pattern, all_words)

        quality = self._evaluate_quality(fragment)

        return LineSeed(
            text=fragment,
            seed_type=SeedType.FRAGMENT,
            strategy=None,
            momentum=random.uniform(0.4, 0.7),
            openness=random.uniform(0.7, 1.0),
            quality_score=quality
        )

    def generate_image_seed(self, seed_words: List[str]) -> LineSeed:
        """Generate a vivid but incomplete sensory description.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with image
        """
        # Choose sensory mode
        sense = random.choice(list(self.sensory_map.keys()))
        sensory_words = self.sensory_map[sense]

        # Get contextual words
        context_words = []
        for word in seed_words[:2]:
            context_words.extend(contextually_linked_words(word, sample_size=3))

        context_words = word_validator.clean_word_list(context_words)

        # If no context words found, use seed words as fallback
        if not context_words:
            context_words = seed_words

        # Create image
        templates = [
            f"{random.choice(sensory_words)} {random.choice(context_words)}",
            f"the {random.choice(sensory_words)} of {random.choice(context_words)}",
            f"{random.choice(context_words)} like {random.choice(sensory_words)}",
            f"{random.choice(sensory_words)}, {random.choice(sensory_words)} {random.choice(context_words)}"
        ]

        image = random.choice(templates)
        quality = self._evaluate_quality(image)

        return LineSeed(
            text=image,
            seed_type=SeedType.IMAGE,
            strategy=GenerationStrategy.SYNESTHESIA if random.random() > 0.5 else None,
            momentum=random.uniform(0.3, 0.6),
            openness=random.uniform(0.5, 0.8),
            quality_score=quality
        )

    def generate_pivot_line(self, seed_words: List[str]) -> LineSeed:
        """Generate a line that can change poem direction.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with pivot line
        """
        # Pivot lines often use contrasts or questions
        templates = [
            "But {noun} {verb} {adjective}...",
            "Or was it {noun} that {verb}...",
            "Until the {noun} {verb}...",
            "Then {noun}, then {noun}...",
            "As if {noun} could {verb}..."
        ]

        template = random.choice(templates)
        line = self._fill_pattern(template, seed_words)
        quality = self._evaluate_quality(line)

        return LineSeed(
            text=line,
            seed_type=SeedType.PIVOT,
            strategy=GenerationStrategy.QUESTION_IMPLIED,
            momentum=random.uniform(0.6, 0.9),
            openness=random.uniform(0.8, 1.0),
            quality_score=quality
        )

    def generate_sonic_pattern(self, seed_words: List[str]) -> LineSeed:
        """Generate a rhythmic/sonic template to build from.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with sonic pattern
        """
        # Get phonetically related words for sound consistency
        sound_words = []
        for word in seed_words[:2]:
            sound_words.extend(phonetically_related_words(word, sample_size=4))

        sound_words = word_validator.clean_word_list(sound_words)

        if len(sound_words) >= 3:
            # Create alliteration or assonance
            pattern = f"{sound_words[0]}, {sound_words[1]}, {sound_words[2]}"
        else:
            # Fallback to rhythm pattern
            pattern = f"{seed_words[0]}-{random.choice(['pause', 'beat', 'breath'])}-{seed_words[-1]}"

        quality = self._evaluate_quality(pattern)

        return LineSeed(
            text=pattern,
            seed_type=SeedType.SONIC,
            strategy=GenerationStrategy.RHYTHMIC_BREAK,
            momentum=random.uniform(0.5, 0.8),
            openness=random.uniform(0.4, 0.7),
            quality_score=quality,
            notes="Build on this sound pattern"
        )

    def generate_ending_approach(self, seed_words: List[str],
                                opening_line: Optional[str] = None) -> LineSeed:
        """Suggest approaches for ending the poem.

        Args:
            seed_words: Words to base generation on
            opening_line: Optional opening to echo/transform

        Returns:
            LineSeed with ending approach
        """
        approach = random.choice(self.closing_approaches)

        if approach == "return_to_opening" and opening_line:
            # Transform the opening
            words = opening_line.split()
            if len(words) > 3:
                ending = f"{words[-1]} {words[0]} {random.choice(words[1:-1])}..."
            else:
                ending = f"Again, {opening_line.rstrip('.')}"
        elif approach == "sudden_concrete":
            # Single concrete detail
            concrete_words = [w for w in seed_words if len(w) < 8]
            if concrete_words:
                ending = f"The {random.choice(concrete_words)}."
            else:
                ending = f"Just {seed_words[0]}."
        elif approach == "open_question":
            ending = f"What {random.choice(['remains', 'persists', 'echoes'])} when {seed_words[0]}...?"
        else:
            # Emotional truth or paradox
            ending = f"All {seed_words[0]}, no {seed_words[-1]}"

        quality = self._evaluate_quality(ending)

        return LineSeed(
            text=ending,
            seed_type=SeedType.CLOSING,
            strategy=GenerationStrategy.PERSPECTIVE_BLUR,
            momentum=random.uniform(0.1, 0.3),
            openness=random.uniform(0.2, 0.5),
            quality_score=quality,
            notes=f"Closing approach: {approach}"
        )

    def generate_seed_collection(self, seed_words: List[str],
                                num_seeds: int = 10) -> List[LineSeed]:
        """Generate a collection of varied line seeds.

        Args:
            seed_words: Words to base generation on
            num_seeds: Number of seeds to generate

        Returns:
            List of LineSeeds of various types
        """
        seeds = []

        # Ensure variety by generating different types
        seeds.append(self.generate_opening_line(seed_words))
        seeds.append(self.generate_pivot_line(seed_words))
        seeds.append(self.generate_image_seed(seed_words))
        seeds.append(self.generate_sonic_pattern(seed_words))
        seeds.append(self.generate_ending_approach(seed_words))

        # Fill remaining with fragments and varied types
        remaining = num_seeds - len(seeds)
        for _ in range(remaining):
            seed_type = random.choice([
                self.generate_fragment,
                self.generate_image_seed,
                self.generate_pivot_line
            ])
            seeds.append(seed_type(seed_words))

        # Sort by quality score
        seeds.sort(key=lambda s: s.quality_score, reverse=True)

        return seeds[:num_seeds]

    # Helper methods

    def _generate_juxtaposition_opening(self, seed_words: List[str]) -> str:
        """Generate opening using juxtaposition strategy."""
        if len(seed_words) >= 2:
            # Get contrasting word for first seed word
            related = similar_meaning_words(seed_words[0], sample_size=5)
            if related:
                contrast = related[-1]  # Often least similar is interesting
                return f"Between {seed_words[0]} and {contrast}, the {seed_words[1]}..."
        return f"The {seed_words[0]}'s {random.choice(['shadow', 'echo', 'ghost'])}..."

    def _generate_temporal_opening(self, seed_words: List[str]) -> str:
        """Generate opening using temporal shift."""
        temporal = random.choice(self.temporal_markers)
        verb_options = frequently_following_words(seed_words[0], sample_size=3)
        if verb_options:
            verb = verb_options[0]
        else:
            verb = "was"
        return f"{temporal} {seed_words[0]} {verb}..."

    def _generate_metaphor_opening(self, seed_words: List[str]) -> str:
        """Generate opening with incomplete metaphor."""
        related = contextually_linked_words(seed_words[0], sample_size=3)
        if related:
            return f"The {seed_words[0]} like {related[0]}, but..."
        return f"As if {seed_words[0]} were..."

    def _generate_standard_opening(self, seed_words: List[str]) -> str:
        """Generate standard pattern-based opening."""
        pattern = random.choice(self.opening_patterns)
        return self._fill_pattern(pattern, seed_words)

    def _fill_pattern(self, pattern: str, words: List[str]) -> str:
        """Fill a pattern with appropriate words."""
        filled = pattern

        # Ensure we have words to work with
        if not words:
            words = ['word']  # Fallback word

        # Simple replacement logic (can be enhanced)
        replacements = {
            '{noun}': lambda: random.choice(words) if words else 'word',
            '{verb}': lambda: random.choice(['carries', 'holds', 'breaks', 'turns', 'waits']),
            '{adjective}': lambda: random.choice(['distant', 'quiet', 'sharp', 'soft', 'strange']),
            '{pronoun}': lambda: random.choice(['we', 'they', 'it', 'you']),
            '{Pronoun}': lambda: random.choice(['We', 'They', 'It', 'You']),
            '{Temporal}': lambda: random.choice(self.temporal_markers),
            '{preposition}': lambda: random.choice(self.spatial_prepositions),
            '{abstract}': lambda: random.choice(['memory', 'time', 'silence', 'distance']),
            '{comparative}': lambda: random.choice(['more', 'less', 'almost', 'nearly'])
        }

        for placeholder, func in replacements.items():
            while placeholder in filled:
                filled = filled.replace(placeholder, func(), 1)

        return filled

    def _fill_pattern_fragment(self, pattern: str, words: List[str]) -> str:
        """Fill a fragment pattern with appropriate words."""
        # Similar to _fill_pattern but for fragments
        return self._fill_pattern(pattern, words)

    def _evaluate_quality(self, text: str) -> float:
        """Evaluate the quality of a generated seed.

        Args:
            text: The generated text

        Returns:
            Quality score from 0 to 1
        """
        score = 0.5  # Base score

        # Check for variety in word length
        words = text.split()
        if words:
            lengths = [len(w.strip('.,!?')) for w in words]
            if len(set(lengths)) > len(lengths) * 0.6:
                score += 0.1

        # Check for avoiding clichés
        cliches = ['heart', 'soul', 'love', 'dream', 'tears', 'broken']
        if not any(cliche in text.lower() for cliche in cliches):
            score += 0.2

        # Check for interesting word combinations
        if '...' in text:
            score += 0.1  # Suggests openness

        # Check for concrete imagery
        concrete_indicators = ['the', 'through', 'between', 'of']
        if any(word in text.lower() for word in concrete_indicators):
            score += 0.1

        return min(1.0, score)
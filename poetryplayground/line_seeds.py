"""Line and Phrase Seed Generator for poetry ideation.

This module generates evocative incomplete phrases and line beginnings
that serve as creative catalysts rather than finished products.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .lexigen import (
    contextually_linked_words,
    frequently_following_words,
    phonetically_related_words,
    similar_meaning_words,
)
from .semantic_geodesic import get_semantic_space
from .vocabulary import vocabulary
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

    def __init__(self, use_templates: bool = True):
        """Initialize the line seed generator.

        Args:
            use_templates: If True, use grammatical templates for fragment generation.
                          If False, use legacy pattern-based generation.
        """
        self.use_templates = use_templates
        self.template_generator = None
        self.pos_vocab = None

        # Always initialize pattern-based generation (used as fallback)
        self._init_patterns()
        self._init_connectives()
        self._init_sensory_words()

        # Initialize template-based generation if requested
        if self.use_templates:
            self._init_template_generation()

    def _init_patterns(self):
        """Initialize phrase patterns for different seed types."""
        self.opening_patterns = [
            "The {noun} {verb} {adjective} before...",
            "In {adjective} {noun}, {pronoun} {verb}...",
            "{Temporal} the {noun} {verb}...",
            "Where {noun} meets {noun}, the {noun}...",
            "Through {adjective} {noun}, something {adjective}...",
            "{Pronoun} {verb} the {noun} of {abstract}...",
        ]

        self.fragment_patterns = [
            "...{preposition} {adjective} {noun}...",
            "...the {noun}'s {noun}...",
            "...{verb}ing what we {verb}...",
            "...{adjective}, {comparative} {adjective}...",
            "...through {noun}, {noun}...",
        ]

        self.closing_approaches = [
            "return_to_opening",
            "sudden_concrete",
            "open_question",
            "sensory_snapshot",
            "emotional_truth",
            "paradox",
        ]

    def _init_connectives(self):
        """Initialize connecting words and phrases."""
        self.temporal_markers = [
            "Before",
            "After",
            "When",
            "While",
            "During",
            "Until",
            "Since",
            "As",
            "Once",
            "Still",
        ]

        self.spatial_prepositions = [
            "beneath",
            "between",
            "through",
            "across",
            "beyond",
            "within",
            "against",
            "beside",
        ]

        self.abstract_connectors = [
            "almost",
            "perhaps",
            "somehow",
            "barely",
            "nearly",
            "always",
            "never",
            "still",
        ]

    def _init_sensory_words(self):
        """Initialize sensory vocabulary using shared vocabulary."""
        # Base sensory words, enhanced with atmospheric nouns and evocative verbs
        base_sensory = {
            "sight": ["glimmer", "shadow", "gleam", "blur", "shimmer", "flicker", "dazzle", "fade"],
            "sound": ["whisper", "echo", "hum", "rustle", "murmur", "crackle", "thrum", "drone"],
            "touch": ["smooth", "rough", "cold", "warm", "sharp", "tender", "coarse", "velvet"],
            "smell": ["smoke", "rain", "metal", "earth", "salt", "musk", "pine", "copper"],
            "taste": ["bitter", "sweet", "acid", "iron", "ash", "honey", "brine", "mint"],
        }

        # Enhance with words from shared vocabulary (convert sets to lists for slicing)
        atmospheric_words = list(vocabulary.atmospheric_nouns)[:50]  # Sample from larger set
        evocative_words = list(vocabulary.evocative_verbs)[:50]  # Sample from larger set

        # Add evocative words to appropriate sensory categories based on characteristics
        for word in atmospheric_words + evocative_words:
            if any(sense in word.lower() for sense in ["light", "bright", "dark", "see", "glow"]):
                base_sensory["sight"].append(word)
            elif any(sense in word.lower() for sense in ["sound", "voice", "call", "ring", "sing"]):
                base_sensory["sound"].append(word)
            elif any(sense in word.lower() for sense in ["touch", "feel", "hold", "grasp"]):
                base_sensory["touch"].append(word)
            elif any(sense in word.lower() for sense in ["smell", "scent", "aroma"]):
                base_sensory["smell"].append(word)
            elif any(sense in word.lower() for sense in ["taste", "sweet", "bitter"]):
                base_sensory["taste"].append(word)

        self.sensory_map = base_sensory

    def _init_template_generation(self):
        """Initialize template-based generation components.

        Uses lazy imports to avoid circular dependencies.
        """
        try:
            from .grammatical_templates import TemplateGenerator
            from .logger import logger
            from .pos_vocabulary import POSVocabulary

            logger.info("Initializing template-based line seed generation...")
            self.pos_vocab = POSVocabulary()
            self.template_generator = TemplateGenerator(self.pos_vocab)
            logger.info("Template-based line seed generation ready")
        except Exception as e:
            from .logger import logger

            logger.warning(
                f"Failed to initialize template generation: {e}. "
                "Falling back to pattern-based generation."
            )
            self.use_templates = False
            self.template_generator = None
            self.pos_vocab = None

    def _generate_template_based_fragment(
        self, seed_words: List[str], target_syllables: Optional[int] = None
    ) -> Optional[str]:
        """Generate grammatical fragment using POS templates.

        Args:
            seed_words: Words to guide generation (currently not used in template selection)
            target_syllables: Optional syllable constraint (2-5 for fragments)

        Returns:
            Grammatically coherent fragment like "lonely commuter" or "wind whispers",
            or None if generation fails
        """
        if not self.template_generator:
            return None

        # Choose syllable count if not specified (2-5 syllables for fragments)
        if target_syllables is None:
            target_syllables = random.choice([2, 3, 4, 5])

        # Constrain to reasonable fragment length
        target_syllables = max(2, min(5, target_syllables))

        try:
            # Generate using template system
            line, _template = self.template_generator.generate_line(
                target_syllables, max_attempts=100
            )

            return line if line else None

        except Exception as e:
            from .logger import logger

            logger.debug(f"Template fragment generation failed: {e}")
            return None

    def generate_opening_line(self, seed_words: List[str], mood: Optional[str] = None) -> LineSeed:
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
            quality_score=quality,
        )

    def generate_fragment(self, seed_words: List[str], position: str = "any") -> LineSeed:
        """Generate an evocative incomplete fragment.

        Uses template-based generation if enabled, with fallback to pattern-based.

        Args:
            seed_words: Words to base generation on
            position: Where in poem ('opening', 'middle', 'closing', 'any')

        Returns:
            LineSeed with fragment
        """
        # Try template-based generation first if available
        if self.use_templates and self.template_generator:
            # Try fragments of different lengths (2-5 syllables)
            for _ in range(3):  # Multiple attempts for variety
                template_fragment = self._generate_template_based_fragment(seed_words)
                if template_fragment:
                    quality = self._evaluate_quality(template_fragment)
                    return LineSeed(
                        text=template_fragment,
                        seed_type=SeedType.FRAGMENT,
                        strategy=None,
                        momentum=random.uniform(0.4, 0.7),
                        openness=random.uniform(0.7, 1.0),
                        quality_score=quality,
                    )

        # Fall back to pattern-based generation
        # Get related words for variety - use ALL seed words with larger samples
        expanded_words = []
        for word in seed_words:  # Use all seed words for maximum diversity
            expanded_words.extend(
                similar_meaning_words(word, sample_size=8, min_quality=0.6)
            )
            expanded_words.extend(
                phonetically_related_words(word, sample_size=6, min_quality=0.6)
            )

        expanded_words = word_validator.clean_word_list(expanded_words)

        # Combine with seed words, ensuring we have something to work with
        all_words = seed_words + expanded_words
        if not all_words:
            all_words = ["word", "phrase", "thought"]  # Emergency fallback

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
            quality_score=quality,
        )

    def generate_image_seed(self, seed_words: List[str]) -> LineSeed:
        """Generate a vivid but incomplete sensory description.

        Uses template-based generation if enabled, with fallback to pattern-based.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with image
        """
        # Try template-based generation first if available
        if self.use_templates and self.template_generator:
            # Try generating a 3-4 syllable sensory fragment
            for _ in range(3):  # Multiple attempts for variety
                target_syllables = random.choice([3, 4])
                template_image = self._generate_template_based_fragment(
                    seed_words, target_syllables
                )
                if template_image:
                    quality = self._evaluate_quality(template_image)
                    return LineSeed(
                        text=template_image,
                        seed_type=SeedType.IMAGE,
                        strategy=GenerationStrategy.SYNESTHESIA if random.random() > 0.5 else None,
                        momentum=random.uniform(0.3, 0.6),
                        openness=random.uniform(0.5, 0.8),
                        quality_score=quality,
                    )

        # Fall back to pattern-based generation
        # Choose sensory mode
        sense = random.choice(list(self.sensory_map.keys()))
        sensory_words = self.sensory_map[sense]

        # Get contextual words - use ALL seed words with larger samples
        context_words = []
        for word in seed_words:  # Use all seed words for maximum diversity
            context_words.extend(
                contextually_linked_words(word, sample_size=15, min_quality=0.6)
            )

        context_words = word_validator.clean_word_list(context_words)

        # If no context words found, use seed words as fallback
        if not context_words:
            context_words = seed_words

        # Create image
        templates = [
            f"{random.choice(sensory_words)} {random.choice(context_words)}",
            f"the {random.choice(sensory_words)} of {random.choice(context_words)}",
            f"{random.choice(context_words)} like {random.choice(sensory_words)}",
            f"{random.choice(sensory_words)}, {random.choice(sensory_words)} {random.choice(context_words)}",
        ]

        image = random.choice(templates)
        quality = self._evaluate_quality(image)

        return LineSeed(
            text=image,
            seed_type=SeedType.IMAGE,
            strategy=GenerationStrategy.SYNESTHESIA if random.random() > 0.5 else None,
            momentum=random.uniform(0.3, 0.6),
            openness=random.uniform(0.5, 0.8),
            quality_score=quality,
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
            "As if {noun} could {verb}...",
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
            quality_score=quality,
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
            # Fallback to rhythm pattern - use atmospheric nouns for rhythm words
            rhythm_words = ['pause', 'beat', 'breath', 'silence', 'moment', 'stillness']
            try:
                # Sample from atmospheric nouns related to rhythm/time
                rhythm_word = random.choice(rhythm_words)
            except (AttributeError, IndexError):
                rhythm_word = 'pause'  # Fallback
            pattern = f"{seed_words[0]}-{rhythm_word}-{seed_words[-1]}"

        quality = self._evaluate_quality(pattern)

        return LineSeed(
            text=pattern,
            seed_type=SeedType.SONIC,
            strategy=GenerationStrategy.RHYTHMIC_BREAK,
            momentum=random.uniform(0.5, 0.8),
            openness=random.uniform(0.4, 0.7),
            quality_score=quality,
            notes="Build on this sound pattern",
        )

    def generate_ending_approach(
        self, seed_words: List[str], opening_line: Optional[str] = None
    ) -> LineSeed:
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
            # Use evocative verbs for question endings
            try:
                question_verbs = [v for v in vocabulary.evocative_verbs if len(v) <= 10]
                question_verb = random.choice(question_verbs) if question_verbs else "remains"
            except (AttributeError, IndexError):
                question_verb = random.choice(['remains', 'persists', 'echoes'])
            ending = f"What {question_verb} when {seed_words[0]}...?"
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
            notes=f"Closing approach: {approach}",
        )

    def _is_semantically_similar(
        self, text: str, existing_texts: List[str], threshold: float = 0.90
    ) -> bool:
        """Check if text is semantically similar to any existing texts.

        Args:
            text: New text to check
            existing_texts: List of previously generated texts
            threshold: Similarity threshold (0.90 = very similar, 0.95 = nearly identical)

        Returns:
            True if text is too similar to any existing text
        """
        if not existing_texts:
            return False

        try:
            # Get semantic space (cached singleton)
            semantic_space = get_semantic_space()

            # Get embedding for new text by averaging word vectors
            new_words = text.lower().split()
            new_vectors = []
            for word in new_words:
                if word in semantic_space.space:
                    new_vectors.append(semantic_space.space[word])

            if not new_vectors:
                return False  # No vectors available, assume not similar

            # Average the word vectors for the new text
            import numpy as np
            new_vec = np.mean(new_vectors, axis=0)

            # Check similarity against each existing text
            for existing_text in existing_texts:
                existing_words = existing_text.lower().split()
                existing_vectors = []
                for word in existing_words:
                    if word in semantic_space.space:
                        existing_vectors.append(semantic_space.space[word])

                if not existing_vectors:
                    continue  # Skip if no vectors available

                # Average the word vectors for existing text
                existing_vec = np.mean(existing_vectors, axis=0)

                # Compute cosine similarity
                similarity = np.dot(new_vec, existing_vec) / (
                    np.linalg.norm(new_vec) * np.linalg.norm(existing_vec)
                )

                if similarity >= threshold:
                    return True  # Too similar!

            return False  # Not similar to any existing text

        except Exception:
            # If semantic comparison fails, assume not similar
            return False

    def generate_seed_collection(
        self, seed_words: List[str], num_seeds: int = 10
    ) -> List[LineSeed]:
        """Generate a collection of varied line seeds with deduplication.

        Args:
            seed_words: Words to base generation on
            num_seeds: Number of seeds to generate

        Returns:
            List of LineSeeds of various types (guaranteed unique)
        """
        seeds = []
        seen_texts = set()  # Track exact text
        seen_normalized = set()  # Track normalized text (lowercase, stripped)
        generated_texts = []  # Track all generated texts for semantic comparison

        def add_unique_seed(generator_func, max_attempts=5):
            """Try to generate a unique seed, retry if duplicate or semantically similar."""
            for _attempt in range(max_attempts):
                candidate = generator_func(seed_words)
                text = candidate.text
                normalized = text.lower().strip()

                # Check for exact/normalized duplicates
                if text in seen_texts or normalized in seen_normalized:
                    continue  # Try again

                # Check for semantic similarity (using 0.90 threshold)
                if self._is_semantically_similar(text, generated_texts, threshold=0.90):
                    continue  # Too similar, try again

                # Passed all checks - add the seed!
                seeds.append(candidate)
                seen_texts.add(text)
                seen_normalized.add(normalized)
                generated_texts.append(text)
                return True

            # Failed to generate unique seed after max attempts
            return False

        # Ensure variety by generating different types
        add_unique_seed(self.generate_opening_line)
        add_unique_seed(self.generate_pivot_line)
        add_unique_seed(self.generate_image_seed)
        add_unique_seed(self.generate_sonic_pattern)
        add_unique_seed(self.generate_ending_approach)

        # Fill remaining with fragments and varied types
        remaining = num_seeds - len(seeds)
        for _ in range(remaining):
            seed_type = random.choice(
                [self.generate_fragment, self.generate_image_seed, self.generate_pivot_line]
            )
            add_unique_seed(seed_type)

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
        # Use atmospheric nouns for evocative possessives
        try:
            possessive_nouns = [n for n in vocabulary.atmospheric_nouns if len(n) <= 8]
            possessive_noun = random.choice(possessive_nouns) if possessive_nouns else "shadow"
        except (AttributeError, IndexError):
            possessive_noun = random.choice(['shadow', 'echo', 'ghost'])
        return f"The {seed_words[0]}'s {possessive_noun}..."

    def _generate_temporal_opening(self, seed_words: List[str]) -> str:
        """Generate opening using temporal shift."""
        temporal = random.choice(self.temporal_markers)
        verb_options = frequently_following_words(seed_words[0], sample_size=3)
        verb = verb_options[0] if verb_options else "was"
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
        """Fill a pattern with appropriate words using near-infinite vocabulary pools."""
        filled = pattern

        # Ensure we have words to work with
        if not words:
            words = ["word"]  # Fallback word

        # Dynamic replacement logic using large vocabulary collections (~300-500 words each)
        # Falls back to static lists only if vocabulary access fails

        def get_verb():
            """Get random verb from vocabulary (~298 options)."""
            try:
                return random.choice(list(vocabulary.evocative_verbs))
            except (AttributeError, IndexError):
                # Fallback to static list
                return random.choice([
                    "carries", "holds", "breaks", "turns", "waits", "drifts", "fades",
                    "whispers", "trembles", "shifts", "lingers", "dissolves"
                ])

        def get_adjective():
            """Get random adjective from vocabulary (~hundreds of options)."""
            try:
                # Collect all adjectives from poetic_attributes dictionary
                all_adjectives = []
                for category_attrs in vocabulary.poetic_attributes.values():
                    all_adjectives.extend(category_attrs)
                return random.choice(all_adjectives)
            except (AttributeError, IndexError, ValueError):
                # Fallback to static list
                return random.choice([
                    "distant", "quiet", "sharp", "soft", "strange", "hollow", "bright",
                    "faint", "worn", "ancient", "still", "fleeting", "hidden"
                ])

        def get_abstract():
            """Get random abstract noun from vocabulary (~498 options)."""
            try:
                return random.choice(list(vocabulary.atmospheric_nouns))
            except (AttributeError, IndexError):
                # Fallback to static list
                return random.choice([
                    "memory", "time", "silence", "distance", "shadow", "light", "absence",
                    "presence", "longing", "stillness", "movement", "echo"
                ])

        replacements = {
            "{noun}": lambda: random.choice(words) if words else "word",
            "{verb}": get_verb,
            "{adjective}": get_adjective,
            "{pronoun}": lambda: random.choice([
                "we", "they", "it", "you", "she", "he", "one", "someone",
                "something", "everything", "nothing", "anyone", "anything"
            ]),
            "{Pronoun}": lambda: random.choice([
                "We", "They", "It", "You", "She", "He", "One", "Someone",
                "Something", "Everything", "Nothing", "Anyone", "Anything"
            ]),
            "{Temporal}": lambda: random.choice(self.temporal_markers),
            "{preposition}": lambda: random.choice(self.spatial_prepositions),
            "{abstract}": get_abstract,
            "{comparative}": lambda: random.choice([
                "more", "less", "almost", "nearly", "barely", "scarcely", "hardly",
                "quite", "rather", "somewhat", "partly", "fully", "half", "mostly",
                "entirely", "completely", "utterly", "slightly", "deeply"
            ]),
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
        """Evaluate the quality of a generated seed using comprehensive quality scoring.

        Args:
            text: The generated text

        Returns:
            Quality score from 0 to 1
        """
        from .quality_scorer import get_quality_scorer

        scorer = get_quality_scorer()
        score = 0.0  # Start from zero - must earn points

        # Extract meaningful words (filter out punctuation and common articles)
        words = text.split()
        stop_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "is", "was", "are", "were", "and", "but"}
        meaningful_words = [
            w.strip(".,!?;:").lower() for w in words if w.strip(".,!?;:").lower() not in stop_words
        ]

        if not meaningful_words:
            return 0.3  # Low score for no meaningful content

        # 1. Word Quality: Evaluate individual word quality (40% weight)
        word_qualities = []
        cliche_count = 0
        for word in meaningful_words:
            if word_validator.is_valid_english_word(word, allow_rare=True):
                word_score = scorer.score_word(word)
                word_qualities.append(word_score.overall)
                # Track clichéd words
                if scorer.is_cliche(word, threshold=0.5):
                    cliche_count += 1

        if word_qualities:
            avg_word_quality = sum(word_qualities) / len(word_qualities)
            score += avg_word_quality * 0.4
        else:
            score += 0.2  # Partial credit if words exist but aren't validated

        # 2. Novelty: Penalize clichés heavily (30% weight)
        cliche_ratio = cliche_count / len(meaningful_words) if meaningful_words else 0
        novelty = 1.0 - (cliche_ratio * 0.8)  # Heavy penalty for clichés

        # Also check phrase-level clichés
        if " " in text.strip():
            phrase_score = scorer.score_phrase(text)
            novelty = min(novelty, phrase_score.novelty)

        score += novelty * 0.3

        # 3. Imagery: Prefer concrete, vivid words (20% weight)
        if meaningful_words:
            concreteness_scores = [scorer.get_concreteness(w) for w in meaningful_words]
            avg_concreteness = sum(concreteness_scores) / len(concreteness_scores)
            # Prefer moderately concrete to very concrete (0.6-0.9 ideal for imagery)
            if 0.6 <= avg_concreteness <= 0.9:
                score += 0.2
            elif 0.5 <= avg_concreteness < 0.6:
                score += 0.15
            elif avg_concreteness > 0.9:
                score += 0.15  # Very concrete is good but not ideal
            elif avg_concreteness >= 0.4:
                score += 0.1  # Some concreteness

        # 4. Structural Quality: Variety and openness (10% weight)
        # Check for word length variety
        if len(meaningful_words) > 1:
            lengths = [len(w) for w in meaningful_words]
            variety_ratio = len(set(lengths)) / len(lengths)
            score += variety_ratio * 0.05

        # Bonus for openness markers (..., ?, incomplete structure)
        if "..." in text or text.endswith("..."):
            score += 0.03

        # Bonus for evocative punctuation (comma, dash, question)
        if any(p in text for p in [",", "—", "?", ";"]):
            score += 0.02

        return min(1.0, max(0.0, score))

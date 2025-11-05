"""Grammatical templates for structured poetry generation.

This module provides POS-based templates that define valid grammatical structures
for poetry lines. Templates enforce word order (e.g., ADJ + NOUN) to produce
coherent phrases instead of "syllable soup".

Templates are organized by form (haiku, tanka, senryu) and syllable count,
enabling both grammatical correctness and syllable constraints.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .forms import count_syllables
from .logger import logger
from .pos_vocabulary import POSVocabulary


@dataclass
class GrammaticalTemplate:
    """Represents a grammatical template for a line of poetry.

    A template defines the part-of-speech structure of a line. For example:
    - ['ADJ', 'NOUN'] produces "lonely commuter" (grammatically valid)
    - ['DET', 'NOUN', 'VERB'] produces "the wind blows"
    - ['NOUN', 'PREP', 'DET', 'NOUN'] produces "light in the sky"

    Attributes:
        pattern: List of POS tags defining the grammatical structure
        name: Descriptive name for the template (e.g., "adjective-noun")
        description: Human-readable description of the pattern
        examples: Example lines that match this template
        weight: Sampling weight (higher = more likely to be chosen)
    """

    pattern: List[str]
    name: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self):
        """Validate template after initialization."""
        if not self.pattern:
            raise ValueError("Template pattern cannot be empty")
        if self.weight <= 0:
            raise ValueError("Template weight must be positive")

    def __str__(self) -> str:
        """String representation of template."""
        pattern_str = " + ".join(self.pattern)
        return f"{self.name}: {pattern_str}"


class TemplateLibrary:
    """Library of grammatical templates organized by syllable count.

    Provides curated template sets for different poetry forms, ensuring
    grammatically coherent output while meeting syllable constraints.
    """

    def __init__(self):
        """Initialize template library with default templates."""
        self.templates: Dict[int, List[GrammaticalTemplate]] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default templates for common syllable counts."""
        # 5-syllable templates (haiku line 1 & 3, tanka lines 1 & 3, senryu 1 & 3)
        self.templates[5] = [
            GrammaticalTemplate(
                pattern=["ADJ", "NOUN"],
                name="adjective-noun",
                description="Simple adjective + noun phrase",
                examples=["ancient temple", "autumn twilight", "lonely mountain"],
                weight=2.0,
            ),
            GrammaticalTemplate(
                pattern=["NOUN", "VERB"],
                name="noun-verb",
                description="Subject + action",
                examples=["wind whispers", "rain falling", "clouds gather"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["DET", "ADJ", "NOUN"],
                name="det-adj-noun",
                description="Determiner + adjective + noun",
                examples=["the old bridge", "a quiet stream"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["NOUN", "PREP", "NOUN"],
                name="noun-prep-noun",
                description="Noun phrase with preposition",
                examples=["light through trees", "waves on shore"],
                weight=1.0,
            ),
            GrammaticalTemplate(
                pattern=["ADV", "VERB", "NOUN"],
                name="adv-verb-noun",
                description="Adverb + verb + object",
                examples=["softly calls home", "gently sways grass"],
                weight=1.0,
            ),
            GrammaticalTemplate(
                pattern=["VERB", "DET", "NOUN"],
                name="verb-det-noun",
                description="Action + determined object",
                examples=["beneath the moon", "through the mist"],
                weight=1.0,
            ),
        ]

        # 7-syllable templates (haiku line 2, tanka lines 2 & 4 & 5, senryu line 2)
        self.templates[7] = [
            GrammaticalTemplate(
                pattern=["DET", "ADJ", "NOUN", "VERB"],
                name="det-adj-noun-verb",
                description="Full clause with determiner",
                examples=["the mountain wind descends"],
                weight=2.0,
            ),
            GrammaticalTemplate(
                pattern=["ADJ", "NOUN", "PREP", "DET", "NOUN"],
                name="adj-noun-prep-det-noun",
                description="Noun phrase with prepositional phrase",
                examples=["cherry blossoms on the branch"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["NOUN", "VERB", "ADV"],
                name="noun-verb-adv",
                description="Subject + action + manner",
                examples=["the river flows quietly"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["ADV", "DET", "NOUN", "VERB"],
                name="adv-det-noun-verb",
                description="Adverb + subject + action",
                examples=["softly the rain falls"],
                weight=1.0,
            ),
            GrammaticalTemplate(
                pattern=["VERB", "PREP", "DET", "ADJ", "NOUN"],
                name="verb-prep-det-adj-noun",
                description="Action with detailed prepositional phrase",
                examples=["flowing through the ancient forest"],
                weight=1.0,
            ),
            GrammaticalTemplate(
                pattern=["DET", "NOUN", "PREP", "DET", "NOUN"],
                name="det-noun-prep-det-noun",
                description="Two noun phrases connected",
                examples=["the sound of falling leaves"],
                weight=1.5,
            ),
        ]

        # 3-syllable templates (for shorter phrases)
        self.templates[3] = [
            GrammaticalTemplate(
                pattern=["ADJ", "NOUN"],
                name="adj-noun-short",
                description="Simple adjective + noun",
                examples=["old gate", "spring wind"],
                weight=2.0,
            ),
            GrammaticalTemplate(
                pattern=["NOUN", "VERB"],
                name="noun-verb-short",
                description="Simple subject + verb",
                examples=["bird sings", "leaf falls"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["DET", "NOUN"],
                name="det-noun",
                description="Determiner + noun",
                examples=["the moon", "a crow"],
                weight=1.0,
            ),
        ]

        # 4-syllable templates
        self.templates[4] = [
            GrammaticalTemplate(
                pattern=["ADJ", "ADJ", "NOUN"],
                name="adj-adj-noun",
                description="Two adjectives + noun",
                examples=["cold winter night"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["DET", "NOUN", "VERB"],
                name="det-noun-verb-4",
                description="Determiner + subject + verb",
                examples=["the wind whispers"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["NOUN", "PREP", "NOUN"],
                name="noun-prep-noun-4",
                description="Noun phrase with preposition",
                examples=["snow on mountain"],
                weight=1.0,
            ),
        ]

        # 6-syllable templates
        self.templates[6] = [
            GrammaticalTemplate(
                pattern=["DET", "ADJ", "NOUN", "VERB"],
                name="det-adj-noun-verb-6",
                description="Full clause",
                examples=["the cherry blossom falls"],
                weight=1.5,
            ),
            GrammaticalTemplate(
                pattern=["ADJ", "NOUN", "PREP", "NOUN"],
                name="adj-noun-prep-noun",
                description="Adjective noun phrase with prep",
                examples=["silent temple in the mist"],
                weight=1.0,
            ),
        ]

    def get_templates(self, target_syllables: int) -> List[GrammaticalTemplate]:
        """Get all templates for a specific syllable count.

        Args:
            target_syllables: Target syllable count

        Returns:
            List of templates matching the syllable count (empty if none)
        """
        return self.templates.get(target_syllables, [])

    def add_template(self, target_syllables: int, template: GrammaticalTemplate):
        """Add a custom template to the library.

        Args:
            target_syllables: Syllable count this template is for
            template: The grammatical template to add
        """
        if target_syllables not in self.templates:
            self.templates[target_syllables] = []
        self.templates[target_syllables].append(template)

    def get_all_syllable_counts(self) -> List[int]:
        """Get all syllable counts that have templates.

        Returns:
            Sorted list of syllable counts
        """
        return sorted(self.templates.keys())


class TemplateGenerator:
    """Generates poetry lines using grammatical templates and POS vocabulary.

    Combines template-based generation with syllable-aware vocabulary to produce
    grammatically coherent lines that meet syllable constraints.
    """

    def __init__(
        self, pos_vocab: POSVocabulary, template_library: Optional[TemplateLibrary] = None
    ):
        """Initialize template generator.

        Args:
            pos_vocab: POS-tagged vocabulary for word lookup
            template_library: Template library (uses default if None)
        """
        self.pos_vocab = pos_vocab
        self.library = template_library or TemplateLibrary()

    def generate_line(
        self,
        target_syllables: int,
        max_attempts: int = 500,  # Increased default due to syllable verification
        temperature: float = 1.0,
    ) -> Tuple[Optional[str], Optional[GrammaticalTemplate]]:
        """Generate a line matching syllable constraint using templates.

        Args:
            target_syllables: Target syllable count
            max_attempts: Maximum generation attempts
            temperature: Sampling temperature (higher = more random template choice)

        Returns:
            Tuple of (generated line, template used) or (None, None) if failed
        """
        templates = self.library.get_templates(target_syllables)
        if not templates:
            logger.warning(f"No templates found for {target_syllables} syllables")
            return None, None

        # Try each template multiple times
        for _attempt in range(max_attempts):
            # Select template based on weights and temperature
            template = self._sample_template(templates, temperature)

            # Try to generate line from this template
            line = self._generate_from_template(template, target_syllables)
            if line:
                return line, template

        logger.warning(
            f"Failed to generate line with {target_syllables} syllables after {max_attempts} attempts"
        )
        return None, None

    def _sample_template(
        self, templates: List[GrammaticalTemplate], temperature: float
    ) -> GrammaticalTemplate:
        """Sample a template based on weights and temperature.

        Args:
            templates: List of templates to sample from
            temperature: Sampling temperature (higher = more uniform)

        Returns:
            Selected template
        """
        if temperature <= 0:
            # Deterministic: pick highest weight
            return max(templates, key=lambda t: t.weight)

        # Apply temperature to weights
        weights = [t.weight ** (1.0 / temperature) for t in templates]
        total = sum(weights)
        weights = [w / total for w in weights]

        return random.choices(templates, weights=weights)[0]

    def _generate_from_template(
        self, template: GrammaticalTemplate, target_syllables: int
    ) -> Optional[str]:
        """Generate a line from a specific template.

        Args:
            template: The grammatical template to use
            target_syllables: Target syllable count

        Returns:
            Generated line or None if generation failed
        """
        # Get possible syllable distributions for this template
        syllable_combos = self.pos_vocab.get_syllable_combinations(
            template.pattern,
            target_syllables,
            max_results=200,  # More combinations for better success rate
        )

        if not syllable_combos:
            return None

        # Shuffle for variety
        random.shuffle(syllable_combos)

        # Try each syllable distribution (try all of them if needed)
        for combo in syllable_combos:  # Try all combinations
            # Try multiple word selections for this combo
            for _word_attempt in range(5):  # Try up to 5 different word selections
                words = []
                success = True

                for pos_tag, syllables in zip(template.pattern, combo):
                    # Get words matching POS and syllable count
                    candidates = self.pos_vocab.get_words(pos_tag, syllables)

                    if not candidates:
                        success = False
                        break

                    # Pick a random word
                    word = random.choice(candidates)
                    words.append(word)

                if success:
                    line = " ".join(words)

                    # Verify actual syllable count matches target
                    actual_syllables = sum(count_syllables(word) for word in words)
                    if actual_syllables == target_syllables:
                        return line
                    # Otherwise, try next word selection or combination

        return None

    def generate_lines(
        self,
        syllable_pattern: List[int],
        max_attempts: int = 500,  # Increased default due to syllable verification
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[Optional[GrammaticalTemplate]]]:
        """Generate multiple lines following a syllable pattern.

        Args:
            syllable_pattern: List of syllable counts per line (e.g., [5, 7, 5] for haiku)
            max_attempts: Maximum attempts per line
            temperature: Sampling temperature

        Returns:
            Tuple of (lines list, templates used list)
        """
        lines = []
        templates_used = []

        for target in syllable_pattern:
            line, template = self.generate_line(target, max_attempts, temperature)

            if line is None:
                raise ValueError(
                    f"Failed to generate line with {target} syllables after {max_attempts} attempts"
                )

            lines.append(line)
            templates_used.append(template)

        return lines, templates_used


def create_template_generator(pos_vocab: POSVocabulary) -> TemplateGenerator:
    """Factory function to create a TemplateGenerator.

    Args:
        pos_vocab: POS vocabulary to use

    Returns:
        Configured TemplateGenerator instance
    """
    return TemplateGenerator(pos_vocab)

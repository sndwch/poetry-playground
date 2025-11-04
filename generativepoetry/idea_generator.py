"""
Poetry Idea Generator

Mines classic literature from Project Gutenberg to extract creative seeds for poetry.
Generates configurable numbers of ideas across different categories to overcome
writer's block and spark creative inspiration.
"""

import re
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum

from .decomposer import ParsedText
from .document_library import get_diverse_gutenberg_documents
from .word_validator import WordValidator


class IdeaType(Enum):
    EMOTIONAL_MOMENT = "emotional_moment"
    VIVID_IMAGERY = "vivid_imagery"
    CHARACTER_SITUATION = "character_situation"
    PHILOSOPHICAL_FRAGMENT = "philosophical_fragment"
    SETTING_DESCRIPTION = "setting_description"
    DIALOGUE_SPARK = "dialogue_spark"
    OPENING_LINE = "opening_line"
    SENSORY_DETAIL = "sensory_detail"
    CONFLICT_SCENARIO = "conflict_scenario"
    METAPHYSICAL_CONCEPT = "metaphysical_concept"


@dataclass
class PoetryIdea:
    """A creative seed extracted from classic literature"""
    text: str
    idea_type: IdeaType
    source_preview: str  # Brief preview of source text
    creative_prompt: str  # Suggested way to use this idea
    keywords: List[str] = field(default_factory=list)


@dataclass
class IdeaCollection:
    """Collection of poetry ideas organized by type"""
    emotional_moments: List[PoetryIdea] = field(default_factory=list)
    vivid_imagery: List[PoetryIdea] = field(default_factory=list)
    character_situations: List[PoetryIdea] = field(default_factory=list)
    philosophical_fragments: List[PoetryIdea] = field(default_factory=list)
    setting_descriptions: List[PoetryIdea] = field(default_factory=list)
    dialogue_sparks: List[PoetryIdea] = field(default_factory=list)
    opening_lines: List[PoetryIdea] = field(default_factory=list)
    sensory_details: List[PoetryIdea] = field(default_factory=list)
    conflict_scenarios: List[PoetryIdea] = field(default_factory=list)
    metaphysical_concepts: List[PoetryIdea] = field(default_factory=list)

    def get_ideas_by_type(self, idea_type: IdeaType) -> List[PoetryIdea]:
        """Get all ideas of a specific type"""
        type_map = {
            IdeaType.EMOTIONAL_MOMENT: self.emotional_moments,
            IdeaType.VIVID_IMAGERY: self.vivid_imagery,
            IdeaType.CHARACTER_SITUATION: self.character_situations,
            IdeaType.PHILOSOPHICAL_FRAGMENT: self.philosophical_fragments,
            IdeaType.SETTING_DESCRIPTION: self.setting_descriptions,
            IdeaType.DIALOGUE_SPARK: self.dialogue_sparks,
            IdeaType.OPENING_LINE: self.opening_lines,
            IdeaType.SENSORY_DETAIL: self.sensory_details,
            IdeaType.CONFLICT_SCENARIO: self.conflict_scenarios,
            IdeaType.METAPHYSICAL_CONCEPT: self.metaphysical_concepts,
        }
        return type_map.get(idea_type, [])

    def add_idea(self, idea: PoetryIdea):
        """Add an idea to the appropriate category (with deduplication)"""
        # Check if this text content already exists in any category
        normalized_text = idea.text.strip().lower()

        for existing_type in IdeaType:
            existing_ideas = self.get_ideas_by_type(existing_type)
            for existing_idea in existing_ideas:
                if existing_idea.text.strip().lower() == normalized_text:
                    # Duplicate found - skip this idea
                    return

        # No duplicate found - add the idea
        ideas_list = self.get_ideas_by_type(idea.idea_type)
        ideas_list.append(idea)

    def total_count(self) -> int:
        """Get total number of ideas collected"""
        return sum(len(self.get_ideas_by_type(idea_type)) for idea_type in IdeaType)

    def get_random_mixed_selection(self, count: int) -> List[PoetryIdea]:
        """Get a random mix of ideas across all types"""
        all_ideas = []
        for idea_type in IdeaType:
            all_ideas.extend(self.get_ideas_by_type(idea_type))

        if not all_ideas:
            return []

        return random.sample(all_ideas, min(count, len(all_ideas)))


class PoetryIdeaGenerator:
    """Generates poetry ideas by mining classic literature"""

    def __init__(self):
        self.word_validator = WordValidator()
        self.used_text_signatures = set()

        # Patterns for different types of creative seeds
        self.patterns = {
            IdeaType.EMOTIONAL_MOMENT: [
                r'[A-Z][^.!?]*(?:felt|feeling|emotion|heart|soul|tears|joy|sorrow|fear|love|hate|anger|hope|despair)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:suddenly|then|now|when|as)[^.!?]*(?:realized|understood|knew|felt|remembered)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:eyes|face|voice|hands)[^.!?]*(?:trembled|shook|whispered|cried|smiled)[^.!?]*[.!?]'
            ],
            IdeaType.VIVID_IMAGERY: [
                r'[A-Z][^.!?]*(?:light|shadow|color|golden|silver|crimson|azure|emerald|darkness)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:mountain|river|ocean|forest|field|garden|sky|stars|moon|sun)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:through|across|beneath|above|beyond)[^.!?]*[.!?]'
            ],
            IdeaType.CHARACTER_SITUATION: [
                r'[A-Z][^.!?]*(?:man|woman|child|stranger|figure|person)[^.!?]*(?:stood|sat|walked|ran|appeared)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:he|she)[^.!?]*(?:was|had been|found himself|found herself)[^.!?]*[.!?]'
            ],
            IdeaType.PHILOSOPHICAL_FRAGMENT: [
                r'[A-Z][^.!?]*(?:truth|life|death|time|existence|meaning|purpose|reason|wisdom)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:what is|why do|how can|where does|when will)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:always|never|forever|eternal|infinite|beyond)[^.!?]*[.!?]'
            ],
            IdeaType.SETTING_DESCRIPTION: [
                r'[A-Z][^.!?]*(?:room|house|street|city|village|castle|church|tower)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:morning|evening|night|dawn|dusk|noon|midnight)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:ancient|old|new|forgotten|hidden|distant|remote)[^.!?]*[.!?]'
            ],
            IdeaType.DIALOGUE_SPARK: [
                r'"[A-Z][^"]*(?:what|why|how|where|when|who)[^"]*"',
                r'"[A-Z][^"]*(?:never|always|remember|forget|tell me|listen)[^"]*"',
                r'"[A-Z][^"]{20,100}"'  # Interesting dialogue of medium length
            ],
            IdeaType.OPENING_LINE: [
                r'^[A-Z][^.!?]{30,120}[.!?]',  # First sentences of good length
                r'^\s*[A-Z][^.!?]*(?:once|long ago|in the|there was|it was)[^.!?]*[.!?]'
            ],
            IdeaType.SENSORY_DETAIL: [
                r'[A-Z][^.!?]*(?:smell|scent|taste|sound|touch|feel|hear|see|watch)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:sweet|bitter|sour|rough|smooth|soft|hard|warm|cold|hot)[^.!?]*[.!?]'
            ],
            IdeaType.CONFLICT_SCENARIO: [
                r'[A-Z][^.!?]*(?:against|between|struggle|fight|battle|conflict|oppose)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:but|however|yet|although|despite|nevertheless)[^.!?]*[.!?]'
            ],
            IdeaType.METAPHYSICAL_CONCEPT: [
                r'[A-Z][^.!?]*(?:spirit|essence|being|consciousness|dream|vision|soul)[^.!?]*[.!?]',
                r'[A-Z][^.!?]*(?:beyond|transcend|invisible|ethereal|mystical|divine)[^.!?]*[.!?]'
            ]
        }

        # Creative prompts for each type
        self.creative_prompts = {
            IdeaType.EMOTIONAL_MOMENT: [
                "Write about a moment when you felt this same emotion",
                "Explore what led to this emotional state",
                "Contrast this feeling with its opposite",
                "Describe this emotion through metaphor"
            ],
            IdeaType.VIVID_IMAGERY: [
                "Use this image as the central metaphor for a poem",
                "Describe a memory triggered by this scene",
                "Write about someone experiencing this setting",
                "Contrast this image with something modern"
            ],
            IdeaType.CHARACTER_SITUATION: [
                "Write from this character's perspective",
                "Imagine what happens next in this situation",
                "Explore what brought this character to this moment",
                "Write about watching this scene unfold"
            ],
            IdeaType.PHILOSOPHICAL_FRAGMENT: [
                "Question or challenge this idea",
                "Explore this concept through personal experience",
                "Use this as the theme for a reflective poem",
                "Connect this idea to a specific moment or image"
            ],
            IdeaType.SETTING_DESCRIPTION: [
                "Set a personal memory in this place",
                "Write about someone discovering this location",
                "Describe how this place changes over time",
                "Use this setting to reflect an emotional state"
            ],
            IdeaType.DIALOGUE_SPARK: [
                "Continue this conversation in verse",
                "Write about the silence after these words",
                "Explore who might have said this and why",
                "Use this as the opening or closing of a poem"
            ],
            IdeaType.OPENING_LINE: [
                "Use this as inspiration for your own opening",
                "Write a poem that leads to this realization",
                "Modernize this scenario or setting",
                "Respond to or challenge this statement"
            ],
            IdeaType.SENSORY_DETAIL: [
                "Build a poem around this sensory experience",
                "Connect this sensation to a memory",
                "Use this detail to ground an abstract idea",
                "Explore how this sensation changes over time"
            ],
            IdeaType.CONFLICT_SCENARIO: [
                "Explore both sides of this conflict",
                "Write about the moment before or after this tension",
                "Use this conflict as metaphor for internal struggle",
                "Imagine the resolution or escalation"
            ],
            IdeaType.METAPHYSICAL_CONCEPT: [
                "Ground this abstract idea in concrete imagery",
                "Explore this concept through everyday experiences",
                "Question or reimagine this spiritual idea",
                "Use this as the hidden theme beneath a simple narrative"
            ]
        }

    def generate_ideas(self, num_ideas: int = 20, preferred_types: Optional[List[IdeaType]] = None) -> IdeaCollection:
        """Generate a collection of poetry ideas from classic literature"""
        print(f"Generating {num_ideas} poetry ideas from classic literature...")

        collection = IdeaCollection()
        attempts = 0
        max_attempts = num_ideas * 2  # Allow plenty of attempts

        target_types = preferred_types or list(IdeaType)

        while collection.total_count() < num_ideas and attempts < max_attempts:
            attempts += 1

            try:
                # Get a fresh text
                text = self._get_unique_text()
                if not text:
                    continue

                # Extract ideas from this text
                ideas = self._extract_ideas_from_text(text, target_types)

                # Add the best ideas to our collection
                for idea in ideas:
                    if collection.total_count() >= num_ideas:
                        break
                    collection.add_idea(idea)

                # Brief pause to avoid API rate limits
                time.sleep(0.3)

            except Exception as e:
                print(f"Error processing text: {e}")
                continue

        print(f"Successfully generated {collection.total_count()} ideas from {len(self.used_text_signatures)} different texts")
        return collection

    def _get_unique_text(self) -> Optional[str]:
        """Get a text we haven't used before"""
        max_retries = 5
        for _ in range(max_retries):
            try:
                text = random_gutenberg_document()
                if not text:
                    continue

                # Create signature
                signature = text[:200].replace('\n', ' ').strip()
                if signature in self.used_text_signatures:
                    continue

                self.used_text_signatures.add(signature)
                return text

            except Exception:
                continue

        return None

    def _extract_ideas_from_text(self, text: str, target_types: List[IdeaType]) -> List[PoetryIdea]:
        """Extract creative ideas from a single text"""
        ideas = []
        source_preview = text[:100].replace('\n', ' ').strip()

        # Parse the text
        parsed = ParsedText(text)

        # Look for different types of ideas
        for idea_type in target_types:
            if idea_type not in self.patterns:
                continue

            patterns = self.patterns[idea_type]

            for pattern in patterns:
                # For opening lines, search from the beginning
                if idea_type == IdeaType.OPENING_LINE:
                    search_text = text[:2000]  # First part of text
                else:
                    # For other types, sample from throughout the text
                    search_text = text

                matches = re.findall(pattern, search_text, re.MULTILINE | re.DOTALL)

                for match in matches[:3]:  # Limit per pattern
                    idea_text = match.strip()

                    # Quality filter
                    if not self._is_good_idea(idea_text, idea_type):
                        continue

                    # Create the idea
                    idea = PoetryIdea(
                        text=idea_text,
                        idea_type=idea_type,
                        source_preview=source_preview,
                        creative_prompt=random.choice(self.creative_prompts[idea_type]),
                        keywords=self._extract_keywords(idea_text)
                    )

                    ideas.append(idea)

                    if len(ideas) >= 8:  # Limit per text
                        break

                if len(ideas) >= 8:
                    break

        return ideas

    def _is_good_idea(self, text: str, idea_type: IdeaType) -> bool:
        """Check if extracted text makes a good creative seed"""
        # Basic length check
        if len(text) < 20 or len(text) > 300:
            return False

        # Must contain some alphabetic content
        if not re.search(r'[a-zA-Z]', text):
            return False

        # Avoid overly technical or boring content
        boring_indicators = [
            'copyright', 'transcriber', 'project gutenberg', 'chapter',
            'vol.', 'page', 'footnote', 'table of contents', 'index',
            'isbn', 'published', 'edition', 'printer', 'library'
        ]

        text_lower = text.lower()
        if any(indicator in text_lower for indicator in boring_indicators):
            return False

        # For dialogue, ensure it's not just attribution
        if idea_type == IdeaType.DIALOGUE_SPARK:
            if text.count('"') < 2:  # Should be actual quoted speech
                return False

        # Must have some interesting words
        interesting_word_count = 0
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        for word in words:
            if len(word) > 4 and self.word_validator.is_valid_english_word(word):
                interesting_word_count += 1

        if interesting_word_count < 3:
            return False

        return True

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key words from the idea text"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Filter to interesting words
        keywords = []
        for word in words:
            if (len(word) > 4 and
                word not in {'that', 'with', 'have', 'this', 'will', 'from', 'they', 'been', 'were', 'said'} and
                self.word_validator.is_valid_english_word(word)):
                keywords.append(word)

        # Return unique keywords, limited count
        return list(dict.fromkeys(keywords))[:5]

    def generate_idea_report(self, collection: IdeaCollection) -> str:
        """Generate a formatted report of the idea collection"""
        report = []

        report.append("POETRY IDEA GENERATOR")
        report.append("=" * 60)
        report.append(f"\nGenerated {collection.total_count()} creative seeds from classic literature")

        # Show ideas by category
        for idea_type in IdeaType:
            ideas = collection.get_ideas_by_type(idea_type)
            if not ideas:
                continue

            # Format the category name nicely
            category_name = idea_type.value.replace('_', ' ').title()

            report.append(f"\n{category_name.upper()}:")
            report.append("-" * 40)

            for i, idea in enumerate(ideas[:3], 1):  # Show first 3 of each type
                report.append(f"{i}. \"{idea.text}\"")
                report.append(f"   Prompt: {idea.creative_prompt}")
                if idea.keywords:
                    report.append(f"   Keywords: {', '.join(idea.keywords[:3])}")
                report.append("")

            if len(ideas) > 3:
                report.append(f"   (Plus {len(ideas) - 3} more {category_name.lower()} ideas)")
                report.append("")

        report.append("CREATIVE SUGGESTIONS:")
        report.append("-" * 30)
        report.append("• Pick an idea that resonates and write for 10 minutes")
        report.append("• Combine two different types of ideas into one poem")
        report.append("• Use an idea as a starting point, then let it evolve")
        report.append("• Modernize or personalize a classic scenario")

        return "\n".join(report)
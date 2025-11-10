"""
Poetry Idea Generator

Mines classic literature from Project Gutenberg to extract creative seeds for poetry.
Generates configurable numbers of ideas across different categories to overcome
writer's block and spark creative inspiration.
"""

import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .config import DocumentConfig
from .core.document_library import get_diverse_gutenberg_documents
from .core.word_validator import WordValidator
from .decomposer import ParsedText


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
    quality_score: float = 0.5  # Overall quality score (0-1)


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

    def get_random_mixed_selection(
        self, count: int, prefer_quality: bool = True
    ) -> List[PoetryIdea]:
        """Get a random mix of ideas across all types.

        Args:
            count: Number of ideas to return
            prefer_quality: If True, weight selection toward higher quality ideas

        Returns:
            List of PoetryIdeas
        """
        all_ideas = []
        for idea_type in IdeaType:
            all_ideas.extend(self.get_ideas_by_type(idea_type))

        if not all_ideas:
            return []

        if prefer_quality:
            # Sort by quality and take from top, but with some randomness
            all_ideas.sort(key=lambda x: x.quality_score, reverse=True)
            # Take from top 60% to maintain variety while favoring quality
            selection_pool_size = max(count, int(len(all_ideas) * 0.6))
            selection_pool = all_ideas[:selection_pool_size]
            return random.sample(selection_pool, min(count, len(selection_pool)))
        else:
            return random.sample(all_ideas, min(count, len(all_ideas)))


class PoetryIdeaGenerator:
    """Generates poetry ideas by mining classic literature"""

    def __init__(self):
        self.word_validator = WordValidator()

        # Patterns for different types of creative seeds
        self.patterns = {
            IdeaType.EMOTIONAL_MOMENT: [
                r"[A-Z][^.!?]*(?:felt|feeling|emotion|heart|soul|tears|joy|sorrow|fear|love|hate|anger|hope|despair)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:suddenly|then|now|when|as)[^.!?]*(?:realized|understood|knew|felt|remembered)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:eyes|face|voice|hands)[^.!?]*(?:trembled|shook|whispered|cried|smiled)[^.!?]*[.!?]",
            ],
            IdeaType.VIVID_IMAGERY: [
                r"[A-Z][^.!?]*(?:light|shadow|color|golden|silver|crimson|azure|emerald|darkness)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:mountain|river|ocean|forest|field|garden|sky|stars|moon|sun)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:through|across|beneath|above|beyond)[^.!?]*[.!?]",
            ],
            IdeaType.CHARACTER_SITUATION: [
                r"[A-Z][^.!?]*(?:man|woman|child|stranger|figure|person)[^.!?]*(?:stood|sat|walked|ran|appeared)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:he|she)[^.!?]*(?:was|had been|found himself|found herself)[^.!?]*[.!?]",
            ],
            IdeaType.PHILOSOPHICAL_FRAGMENT: [
                r"[A-Z][^.!?]*(?:truth|life|death|time|existence|meaning|purpose|reason|wisdom)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:what is|why do|how can|where does|when will)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:always|never|forever|eternal|infinite|beyond)[^.!?]*[.!?]",
            ],
            IdeaType.SETTING_DESCRIPTION: [
                r"[A-Z][^.!?]*(?:room|house|street|city|village|castle|church|tower)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:morning|evening|night|dawn|dusk|noon|midnight)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:ancient|old|new|forgotten|hidden|distant|remote)[^.!?]*[.!?]",
            ],
            IdeaType.DIALOGUE_SPARK: [
                r'"[A-Z][^"]*(?:what|why|how|where|when|who)[^"]*"',
                r'"[A-Z][^"]*(?:never|always|remember|forget|tell me|listen)[^"]*"',
                r'"[A-Z][^"]{20,100}"',  # Interesting dialogue of medium length
            ],
            IdeaType.OPENING_LINE: [
                r"^[A-Z][^.!?]{30,120}[.!?]",  # First sentences of good length
                r"^\s*[A-Z][^.!?]*(?:once|long ago|in the|there was|it was)[^.!?]*[.!?]",
            ],
            IdeaType.SENSORY_DETAIL: [
                r"[A-Z][^.!?]*(?:smell|scent|taste|sound|touch|feel|hear|see|watch)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:sweet|bitter|sour|rough|smooth|soft|hard|warm|cold|hot)[^.!?]*[.!?]",
            ],
            IdeaType.CONFLICT_SCENARIO: [
                r"[A-Z][^.!?]*(?:against|between|struggle|fight|battle|conflict|oppose)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:but|however|yet|although|despite|nevertheless)[^.!?]*[.!?]",
            ],
            IdeaType.METAPHYSICAL_CONCEPT: [
                r"[A-Z][^.!?]*(?:spirit|essence|being|consciousness|dream|vision|soul)[^.!?]*[.!?]",
                r"[A-Z][^.!?]*(?:beyond|transcend|invisible|ethereal|mystical|divine)[^.!?]*[.!?]",
            ],
        }

        # Creative prompts for each type
        self.creative_prompts = {
            IdeaType.EMOTIONAL_MOMENT: [
                "Write about a moment when you felt this same emotion",
                "Explore what led to this emotional state",
                "Contrast this feeling with its opposite",
                "Describe this emotion through metaphor",
            ],
            IdeaType.VIVID_IMAGERY: [
                "Use this image as the central metaphor for a poem",
                "Describe a memory triggered by this scene",
                "Write about someone experiencing this setting",
                "Contrast this image with something modern",
            ],
            IdeaType.CHARACTER_SITUATION: [
                "Write from this character's perspective",
                "Imagine what happens next in this situation",
                "Explore what brought this character to this moment",
                "Write about watching this scene unfold",
            ],
            IdeaType.PHILOSOPHICAL_FRAGMENT: [
                "Question or challenge this idea",
                "Explore this concept through personal experience",
                "Use this as the theme for a reflective poem",
                "Connect this idea to a specific moment or image",
            ],
            IdeaType.SETTING_DESCRIPTION: [
                "Set a personal memory in this place",
                "Write about someone discovering this location",
                "Describe how this place changes over time",
                "Use this setting to reflect an emotional state",
            ],
            IdeaType.DIALOGUE_SPARK: [
                "Continue this conversation in verse",
                "Write about the silence after these words",
                "Explore who might have said this and why",
                "Use this as the opening or closing of a poem",
            ],
            IdeaType.OPENING_LINE: [
                "Use this as inspiration for your own opening",
                "Write a poem that leads to this realization",
                "Modernize this scenario or setting",
                "Respond to or challenge this statement",
            ],
            IdeaType.SENSORY_DETAIL: [
                "Build a poem around this sensory experience",
                "Connect this sensation to a memory",
                "Use this detail to ground an abstract idea",
                "Explore how this sensation changes over time",
            ],
            IdeaType.CONFLICT_SCENARIO: [
                "Explore both sides of this conflict",
                "Write about the moment before or after this tension",
                "Use this conflict as metaphor for internal struggle",
                "Imagine the resolution or escalation",
            ],
            IdeaType.METAPHYSICAL_CONCEPT: [
                "Ground this abstract idea in concrete imagery",
                "Explore this concept through everyday experiences",
                "Question or reimagine this spiritual idea",
                "Use this as the hidden theme beneath a simple narrative",
            ],
        }

    def generate_ideas(
        self, num_ideas: int = 20, preferred_types: Optional[List[IdeaType]] = None
    ) -> IdeaCollection:
        """Generate a collection of poetry ideas from classic literature using adaptive scaling"""
        print(f"Generating {num_ideas} poetry ideas from classic literature...")

        collection = IdeaCollection()
        target_types = preferred_types or list(IdeaType)

        # Start with initial batch of documents
        initial_batch_size = max(3, num_ideas // 10)  # Start with reasonable batch
        print(f"📚 Retrieving {initial_batch_size} diverse documents for idea extraction...")

        documents = get_diverse_gutenberg_documents(
            count=initial_batch_size, min_length=DocumentConfig.MIN_LENGTH_IDEAS
        )
        if not documents:
            print("❌ Failed to retrieve documents for idea extraction")
            return collection

        print(f"✓ Successfully retrieved {len(documents)} diverse documents")

        # Process initial documents
        documents_processed = 0
        for doc_index, text in enumerate(documents, 1):
            print(f"  🔍 Extracting ideas from document {doc_index}/{len(documents)}...")
            documents_processed += 1

            try:
                ideas = self._extract_ideas_from_text(text, target_types)
                for idea in ideas:
                    if collection.total_count() >= num_ideas:
                        break
                    collection.add_idea(idea)

                print(
                    f"    ✓ Found {len(ideas)} potential ideas, collection now has {collection.total_count()}"
                )

                if collection.total_count() >= num_ideas:
                    break

            except Exception as e:
                print(f"    ⚠ Error processing document {doc_index}: {e}")
                continue

        # Apply adaptive scaling: get more documents if yield is low
        min_target = num_ideas
        while collection.total_count() < min_target:
            remaining_needed = min_target - collection.total_count()
            additional_batch = min(
                DocumentConfig.MAX_ADAPTIVE_BATCH,
                max(DocumentConfig.MIN_ADAPTIVE_BATCH, remaining_needed // 5),
            )

            print(
                f"  📚 Found {collection.total_count()} ideas, need {remaining_needed} more. Retrieving {additional_batch} additional documents..."
            )

            additional_docs = get_diverse_gutenberg_documents(
                count=additional_batch, min_length=DocumentConfig.MIN_LENGTH_IDEAS
            )

            if not additional_docs:
                print("  ⚠ Could not retrieve additional documents")
                break

            for text in additional_docs:
                documents_processed += 1
                print(f"  🔍 Extracting ideas from additional document {documents_processed}...")

                try:
                    ideas = self._extract_ideas_from_text(text, target_types)
                    for idea in ideas:
                        if collection.total_count() >= min_target:
                            break
                        collection.add_idea(idea)

                    print(
                        f"    ✓ Found {len(ideas)} additional ideas, collection now has {collection.total_count()}"
                    )

                    if collection.total_count() >= min_target:
                        break

                except Exception as e:
                    print(f"    ⚠ Error processing additional document {documents_processed}: {e}")
                    continue

        print(
            f"🎉 Successfully generated {collection.total_count()} ideas from {documents_processed} diverse texts!"
        )
        return collection

    def _extract_ideas_from_text(self, text: str, target_types: List[IdeaType]) -> List[PoetryIdea]:
        """Extract creative ideas from a single text"""
        ideas = []
        source_preview = text[:100].replace("\n", " ").strip()

        # Parse the text
        ParsedText(text)

        # Look for different types of ideas
        for idea_type in target_types:
            if idea_type not in self.patterns:
                continue

            patterns = self.patterns[idea_type]

            for pattern in patterns:
                # For opening lines, search from the beginning; otherwise use full text
                search_text = text[:2000] if idea_type == IdeaType.OPENING_LINE else text

                matches = re.findall(pattern, search_text, re.MULTILINE | re.DOTALL)

                for match in matches[:3]:  # Limit per pattern
                    idea_text = match.strip()

                    # Quality filter
                    if not self._is_good_idea(idea_text, idea_type):
                        continue

                    # Calculate quality score for this idea
                    quality = self._evaluate_idea_quality(idea_text, idea_type)

                    # Create the idea with quality score
                    idea = PoetryIdea(
                        text=idea_text,
                        idea_type=idea_type,
                        source_preview=source_preview,
                        creative_prompt=random.choice(self.creative_prompts[idea_type]),
                        keywords=self._extract_keywords(idea_text),
                        quality_score=quality,
                    )

                    ideas.append(idea)

                    if len(ideas) >= 8:  # Limit per text
                        break

                if len(ideas) >= 8:
                    break

        return ideas

    def _is_good_idea(self, text: str, idea_type: IdeaType) -> bool:
        """Check if extracted text makes a good creative seed using comprehensive quality scoring.

        Args:
            text: The extracted text to evaluate
            idea_type: The type of idea being evaluated

        Returns:
            True if the idea meets quality thresholds
        """
        # Basic length check
        if len(text) < 20 or len(text) > 300:
            return False

        # Must contain some alphabetic content
        if not re.search(r"[a-zA-Z]", text):
            return False

        # Avoid overly technical or boring content
        boring_indicators = [
            "copyright",
            "transcriber",
            "project gutenberg",
            "chapter",
            "vol.",
            "page",
            "footnote",
            "table of contents",
            "index",
            "isbn",
            "published",
            "edition",
            "printer",
            "library",
        ]

        text_lower = text.lower()
        if any(indicator in text_lower for indicator in boring_indicators):
            return False

        # For dialogue, ensure it's not just attribution
        if idea_type == IdeaType.DIALOGUE_SPARK and text.count('"') < 2:
            return False  # Should be actual quoted speech

        # Apply comprehensive quality filtering
        quality_score = self._evaluate_idea_quality(text, idea_type)

        # Quality threshold varies by type
        # Imagery and sensory details need higher quality (more vivid)
        if idea_type in [IdeaType.VIVID_IMAGERY, IdeaType.SENSORY_DETAIL]:
            return quality_score >= 0.6
        # Philosophical and metaphysical can be more abstract
        elif idea_type in [IdeaType.PHILOSOPHICAL_FRAGMENT, IdeaType.METAPHYSICAL_CONCEPT]:
            return quality_score >= 0.5
        # Everything else uses moderate threshold
        else:
            return quality_score >= 0.55

    def _evaluate_idea_quality(self, text: str, idea_type: IdeaType) -> float:
        """Evaluate the quality of a poetry idea using comprehensive quality scoring.

        Args:
            text: The idea text to evaluate
            idea_type: The type of idea

        Returns:
            Quality score from 0 to 1
        """
        from .quality_scorer import get_quality_scorer

        scorer = get_quality_scorer()
        score = 0.0

        # Extract meaningful words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        stop_words = {
            "a",
            "an",
            "the",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "is",
            "was",
            "are",
            "were",
            "and",
            "but",
            "or",
            "with",
        }
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]

        if not meaningful_words:
            return 0.3

        # 1. Word Quality (35% weight)
        word_qualities = []
        cliche_count = 0
        for word in meaningful_words[:15]:  # Limit for performance
            if self.word_validator.is_valid_english_word(word, allow_rare=True):
                word_score = scorer.score_word(word)
                word_qualities.append(word_score.overall)
                if scorer.is_cliche(word, threshold=0.5):
                    cliche_count += 1

        if word_qualities:
            avg_word_quality = sum(word_qualities) / len(word_qualities)
            score += avg_word_quality * 0.35

        # 2. Novelty: Penalize clichés (25% weight)
        cliche_ratio = cliche_count / len(meaningful_words) if meaningful_words else 0
        novelty = 1.0 - (cliche_ratio * 0.7)  # Moderate penalty

        # Check phrase-level clichés
        phrase_score = scorer.score_phrase(text[:150])  # Limit length for performance
        novelty = min(novelty, phrase_score.novelty)

        score += novelty * 0.25

        # 3. Type-specific quality (40% weight)
        # Different idea types have different quality criteria
        type_score = 0.5  # Base

        if idea_type in [IdeaType.VIVID_IMAGERY, IdeaType.SENSORY_DETAIL]:
            # Prefer concrete, vivid imagery
            concreteness_scores = [scorer.get_concreteness(w) for w in meaningful_words[:10]]
            if concreteness_scores:
                avg_concrete = sum(concreteness_scores) / len(concreteness_scores)
                # Ideal: 0.7-0.95 (very concrete)
                if 0.7 <= avg_concrete <= 0.95:
                    type_score = 1.0
                elif 0.6 <= avg_concrete < 0.7:
                    type_score = 0.8
                elif avg_concrete > 0.5:
                    type_score = 0.6

        elif idea_type in [IdeaType.PHILOSOPHICAL_FRAGMENT, IdeaType.METAPHYSICAL_CONCEPT]:
            # Can be more abstract, but should have some concrete grounding
            concreteness_scores = [scorer.get_concreteness(w) for w in meaningful_words[:10]]
            if concreteness_scores:
                avg_concrete = sum(concreteness_scores) / len(concreteness_scores)
                # Ideal: 0.3-0.6 (moderately abstract with some concrete)
                if 0.3 <= avg_concrete <= 0.6:
                    type_score = 1.0
                elif avg_concrete < 0.3:
                    type_score = 0.7  # Too abstract
                else:
                    type_score = 0.8  # Concrete is okay

        elif idea_type == IdeaType.OPENING_LINE:
            # Should have good momentum and not be clichéd
            if novelty > 0.7:
                type_score = 0.9
            elif novelty > 0.5:
                type_score = 0.7
            else:
                type_score = 0.5

        elif idea_type == IdeaType.EMOTIONAL_MOMENT:
            # Balance of concrete and abstract, avoid cliché emotions
            concreteness_scores = [scorer.get_concreteness(w) for w in meaningful_words[:10]]
            if concreteness_scores:
                avg_concrete = sum(concreteness_scores) / len(concreteness_scores)
                # Ideal: 0.5-0.8 (balanced)
                if 0.5 <= avg_concrete <= 0.8 and novelty > 0.6:
                    type_score = 1.0
                elif novelty > 0.5:
                    type_score = 0.7
                else:
                    type_score = 0.5

        else:
            # Default: balanced scoring
            type_score = 0.7

        score += type_score * 0.4

        return min(1.0, max(0.0, score))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key words from the idea text"""
        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        # Filter to interesting words
        keywords = []
        for word in words:
            if (
                len(word) > 4
                and word
                not in {
                    "that",
                    "with",
                    "have",
                    "this",
                    "will",
                    "from",
                    "they",
                    "been",
                    "were",
                    "said",
                }
                and self.word_validator.is_valid_english_word(word)
            ):
                keywords.append(word)

        # Return unique keywords, limited count
        return list(dict.fromkeys(keywords))[:5]

    def generate_idea_report(self, collection: IdeaCollection) -> str:
        """Generate a formatted report of the idea collection"""
        report = []

        report.append("POETRY IDEA GENERATOR")
        report.append("=" * 60)
        report.append(
            f"\nGenerated {collection.total_count()} creative seeds from classic literature"
        )

        # Show ideas by category
        for idea_type in IdeaType:
            ideas = collection.get_ideas_by_type(idea_type)
            if not ideas:
                continue

            # Format the category name nicely
            category_name = idea_type.value.replace("_", " ").title()

            report.append(f"\n{category_name.upper()}:")
            report.append("-" * 40)

            for i, idea in enumerate(ideas[:3], 1):  # Show first 3 of each type
                report.append(f'{i}. "{idea.text}"')
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

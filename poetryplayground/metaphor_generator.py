"""Dynamic Metaphor Generator using Project Gutenberg texts and cross-domain connections."""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .config import DocumentConfig, QualityConfig
from .decomposer import ParsedText
from .document_library import get_diverse_gutenberg_documents
from .grammatical_templates import MetaphorPatterns
from .lexigen import (
    contextually_linked_words,
    frequently_following_words,
    related_rare_words,
    similar_meaning_words,
)
from .vocabulary import vocabulary
from .word_validator import word_validator

logger = logging.getLogger(__name__)


class MetaphorType(Enum):
    """Types of metaphors that can be generated."""

    SIMILE = "simile"  # X is like Y
    DIRECT = "direct"  # X is Y
    IMPLIED = "implied"  # Using verbs associated with Y for X
    POSSESSIVE = "possessive"  # Y's X or X of Y
    APPOSITIVE = "appositive"  # X, that Y
    COMPOUND = "compound"  # X-Y
    EXTENDED = "extended"  # Multi-line development
    CONCEPTUAL = "conceptual"  # Abstract mapping
    SYNESTHETIC = "synesthetic"  # Cross-sensory


@dataclass
class Metaphor:
    """A generated metaphor with metadata."""

    text: str
    source: str  # The tenor (what's being described)
    target: str  # The vehicle (what it's compared to)
    metaphor_type: MetaphorType
    quality_score: float
    grounds: List[str] = None  # Connecting attributes
    source_text: Optional[str] = None  # If derived from Gutenberg


class MetaphorGenerator:
    """Generate fresh metaphors using Gutenberg texts and domain crossing."""

    def __init__(self):
        """Initialize the metaphor generator."""
        self._init_domains()
        self._init_patterns()
        self._init_verb_associations()
        self._metaphor_cache = []
        self._gutenberg_patterns = []

    def _init_domains(self):
        """Initialize semantic domain categories from centralized vocabulary."""
        self.domains = vocabulary.concept_domains

    def _init_patterns(self):
        """Initialize metaphor pattern templates from grammatical_templates."""
        metaphor_patterns = MetaphorPatterns()

        # Import patterns from centralized template library
        self.simile_patterns = metaphor_patterns.simile_patterns
        self.direct_patterns = metaphor_patterns.direct_patterns
        self.possessive_patterns = metaphor_patterns.possessive_patterns
        self.appositive_patterns = metaphor_patterns.appositive_patterns
        self.compound_patterns = metaphor_patterns.compound_patterns
        self.conceptual_patterns = metaphor_patterns.conceptual_patterns

    def _init_verb_associations(self):
        """Initialize verb associations using shared vocabulary."""
        self.verb_associations = vocabulary.domain_verb_associations

    def extract_metaphor_patterns(self, num_texts: int = 3, verbose: bool = True) -> List[str]:
        """Extract metaphorical patterns from multiple Gutenberg texts.

        Args:
            num_texts: Number of different texts to sample from (default: 3)
            verbose: Whether to print progress messages (default: True for CLI, False for TUI)

        Returns:
            List of extracted metaphor patterns from diverse sources
        """
        all_metaphors = []

        # Get diverse documents using centralized system
        if verbose:
            print(f"📚 Retrieving {num_texts} diverse documents for metaphor extraction...")
        documents = get_diverse_gutenberg_documents(
            count=num_texts, min_length=DocumentConfig.MIN_LENGTH_METAPHORS
        )

        if not documents:
            if verbose:
                print("❌ Failed to retrieve documents for metaphor extraction")
            return all_metaphors

        if verbose:
            print(f"✓ Successfully retrieved {len(documents)} diverse documents")

        # Process each document using helper method
        for doc_index, text in enumerate(documents, 1):
            found_metaphors = self._extract_metaphors_from_text(
                text, doc_index, len(documents), verbose=verbose
            )
            if found_metaphors:
                all_metaphors.extend(found_metaphors[: QualityConfig.MAX_METAPHORS_PER_TEXT])

        # Apply adaptive scaling: get more documents if yield is low
        min_target = max(15, num_texts * 5)  # Higher expectations for metaphors
        documents_processed = len(documents)

        while len(all_metaphors) < min_target:
            remaining_needed = min_target - len(all_metaphors)
            additional_batch = min(
                DocumentConfig.MAX_ADAPTIVE_BATCH,
                max(DocumentConfig.MIN_ADAPTIVE_BATCH, remaining_needed // 3),
            )

            if verbose:
                print(
                    f"  📚 Found {len(all_metaphors)} metaphors, need {remaining_needed} more. Retrieving {additional_batch} additional documents..."
                )

            additional_docs = get_diverse_gutenberg_documents(
                count=additional_batch, min_length=DocumentConfig.MIN_LENGTH_METAPHORS
            )

            if not additional_docs:
                if verbose:
                    print("  ⚠ Could not retrieve additional documents")
                break

            for text in additional_docs:
                documents_processed += 1
                found_metaphors = self._extract_metaphors_from_text(
                    text, documents_processed, None, is_additional=True, verbose=verbose
                )
                if found_metaphors:
                    all_metaphors.extend(found_metaphors[: QualityConfig.MAX_METAPHORS_PER_TEXT])

                if len(all_metaphors) >= min_target:
                    break

        # Remove duplicates, apply quality filtering, and store for later use
        unique_metaphors = []
        seen_pairs = set()
        quality_filtered = 0

        for metaphor in all_metaphors:
            source, target = metaphor[0], metaphor[1]
            pair = (source.lower(), target.lower())

            if pair not in seen_pairs and len(source) > 2 and len(target) > 2:
                # Apply quality filter - only keep high-quality metaphors
                quality = self._score_metaphor(source, target, [])
                if quality >= 0.5:  # Quality threshold for final selection
                    seen_pairs.add(pair)
                    unique_metaphors.append(metaphor)
                else:
                    quality_filtered += 1

        if quality_filtered > 0 and verbose:
            print(f"  🎯 Filtered out {quality_filtered} low-quality metaphors")

        if verbose:
            print(
                f"🎉 Extracted {len(unique_metaphors)} unique metaphor patterns from {documents_processed} diverse texts!"
            )

        self._gutenberg_patterns.extend(unique_metaphors)
        return unique_metaphors

    def _get_text_signature(self, text: str) -> str:
        """Create a signature for a text to identify unique documents"""
        # Use first 200 characters as signature (after cleaning)
        clean_text = re.sub(r"\s+", " ", text[:500]).strip()
        return clean_text[:200]

    def _is_valid_metaphor_pair(self, source: str, target: str, check_quality: bool = True) -> bool:
        """Check if a source-target pair makes a valid metaphor.

        Args:
            source: The tenor (what's being described)
            target: The vehicle (what it's compared to)
            check_quality: Whether to apply quality filtering (default: True)

        Returns:
            True if pair is valid and high-quality enough
        """
        # Filter out common non-metaphorical phrases
        invalid_pairs = {
            ("it", "that"),
            ("this", "that"),
            ("he", "she"),
            ("one", "another"),
            ("some", "other"),
            ("man", "woman"),
        }

        if (source, target) in invalid_pairs:
            return False

        # Both should be meaningful words
        if len(source) < 3 or len(target) < 3:
            return False

        # Use word validator for basic checks
        if not (
            word_validator.is_valid_english_word(source)
            and word_validator.is_valid_english_word(target)
        ):
            return False

        # Apply quality filtering if requested
        if check_quality:
            from .quality_scorer import get_quality_scorer

            scorer = get_quality_scorer()

            # Reject if the phrase itself is a cliché
            metaphor_phrase = f"{source} is {target}"
            if scorer.is_cliche(metaphor_phrase, threshold=0.6):
                return False

            # Reject if both words are individually clichéd
            if scorer.is_cliche(source, threshold=0.5) and scorer.is_cliche(target, threshold=0.5):
                return False

            # Require minimum word quality
            source_quality = scorer.score_word(source).overall
            target_quality = scorer.score_word(target).overall
            avg_quality = (source_quality + target_quality) / 2

            if avg_quality < 0.4:  # Quality threshold
                return False

        return True

    def _extract_metaphors_from_text(
        self,
        text: str,
        doc_index: int,
        total_docs: Optional[int] = None,
        is_additional: bool = False,
        verbose: bool = True,
    ) -> List[Tuple[str, str, str]]:
        """Extract metaphors from a single text using defined patterns."""
        patterns = [
            r"(\w+)\s+(?:is|was|are|were)\s+like\s+(?:a\s+|an\s+|the\s+)?(\w+)",
            r"(\w+)\s+as\s+(?:a\s+|an\s+|the\s+)?(\w+)",
            r"(\w+),\s+(?:a\s+|an\s+|that\s+|this\s+)(\w+)",
            r"the\s+(\w+)\s+of\s+(\w+)",
            r"(\w+)\s+(?:resembles|mirrors|echoes)\s+(?:a\s+|an\s+|the\s+)?(\w+)",
        ]

        try:
            if verbose:
                if is_additional:
                    print(f"  🔍 Extracting metaphors from additional document {doc_index}...")
                else:
                    print(f"  🔍 Extracting metaphors from document {doc_index}/{total_docs}...")

            parsed = ParsedText(text)
            found_metaphors = []

            # Sample sentences from throughout the text, not just the beginning
            sentences_to_check = (
                parsed.sentences[:50]
                + parsed.sentences[len(parsed.sentences) // 2 : len(parsed.sentences) // 2 + 50]
            )

            for sentence in sentences_to_check:
                for pattern in patterns:
                    matches = re.findall(pattern, sentence.lower())
                    for match in matches:
                        if len(match) == 2:
                            source, target = match
                            if self._is_valid_metaphor_pair(source, target):
                                found_metaphors.append((source, target, sentence))

            # Sort by quality using comprehensive quality scoring
            if found_metaphors:
                # Score each metaphor and sort by quality
                scored_metaphors = []
                for source, target, sentence in found_metaphors:
                    # Calculate quality score for this metaphor
                    quality = self._score_metaphor(source, target, [])

                    # Also consider sentence clarity (shorter is often clearer)
                    clarity_bonus = max(0, 1.0 - (len(sentence) / 200))  # Bonus for concise context

                    # Combined score: 80% metaphor quality, 20% clarity
                    combined_score = (quality * 0.8) + (clarity_bonus * 0.2)

                    scored_metaphors.append((source, target, sentence, combined_score))

                # Sort by combined score (descending)
                scored_metaphors.sort(key=lambda x: x[3], reverse=True)

                # Convert back to original format (drop the score)
                found_metaphors = [(s, t, sent) for s, t, sent, _ in scored_metaphors]

                if verbose:
                    if is_additional:
                        print(
                            f"    ✓ Found {len(found_metaphors)} additional metaphor patterns (quality-sorted)"
                        )
                    else:
                        print(
                            f"    ✓ Found {len(found_metaphors)} metaphor patterns (quality-sorted)"
                        )
            else:
                if verbose:
                    print("    ✓ Found 0 metaphor patterns")

            return found_metaphors

        except Exception as e:
            if verbose:
                print(f"    ⚠ Error processing document {doc_index}: {e}")
            return []

    def generate_metaphor_batch(self, source_words: List[str], count: int = 10) -> List[Metaphor]:
        """Generate a batch of metaphors from source words.

        Args:
            source_words: Words to create metaphors for
            count: Number of metaphors to generate

        Returns:
            List of Metaphor objects sorted by quality
        """
        metaphors = []

        for source in source_words:
            # Get various target domains
            targets = self._find_target_domains(source)

            for target in targets[:5]:  # Limit targets per source
                # Generate different types
                metaphors.extend(
                    [
                        self._generate_simile(source, target),
                        self._generate_direct_metaphor(source, target),
                        self._generate_implied_metaphor(source, target),
                        self._generate_possessive_metaphor(source, target),
                        self._generate_compound_metaphor(source, target),
                        self._generate_conceptual_metaphor(source, target),
                    ]
                )

        # Remove None values
        metaphors = [m for m in metaphors if m is not None]

        # Sort by quality
        metaphors.sort(key=lambda m: m.quality_score, reverse=True)

        return metaphors[:count]

    def _find_target_domains(self, source: str) -> List[str]:
        """Find suitable target domains for a source word."""
        targets = []

        # Method 1: Use semantic opposites or contrasts
        for _domain, words in self.domains.items():
            if source not in words:
                targets.extend(random.sample(words, min(2, len(words))))

        # Method 2: Use contextually linked words
        linked = contextually_linked_words(source, sample_size=5)
        if linked:
            targets.extend(linked)

        # Method 3: Use rare related words for unexpected connections
        rare = related_rare_words(source, sample_size=3)
        if rare:
            targets.extend(rare)

        # Method 4: Random domain sampling for surprise
        random_domain = random.choice(list(self.domains.keys()))
        targets.extend(random.sample(self.domains[random_domain], 2))

        # Clean and validate
        targets = word_validator.clean_word_list(targets)

        return list(set(targets))  # Remove duplicates

    def _generate_simile(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate a simile metaphor."""
        pattern = random.choice(self.simile_patterns)

        # Get connecting attributes
        grounds = self._find_connecting_attributes(source, target)

        # Build the simile
        if "{adjective}" in pattern:
            adjective = grounds[0] if grounds else "mysterious"
            text = pattern.format(source=source, target=target, adjective=adjective)
        else:
            text = pattern.format(source=source, target=target)

        # Add elaboration if we have grounds
        if grounds:
            elaboration = random.choice(
                [
                    f", both {grounds[0]}",
                    f" in its {grounds[0]}ness",
                    f": {grounds[0]}, {grounds[1] if len(grounds) > 1 else 'endless'}",
                ]
            )
            text += elaboration

        quality = self._score_metaphor(source, target, grounds)

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.SIMILE,
            quality_score=quality,
            grounds=grounds,
        )

    def _generate_direct_metaphor(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate a direct metaphor."""
        pattern = random.choice(self.direct_patterns)
        text = pattern.format(source=source, target=target)

        grounds = self._find_connecting_attributes(source, target)
        quality = self._score_metaphor(source, target, grounds)

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.DIRECT,
            quality_score=quality,
            grounds=grounds,
        )

    def _generate_implied_metaphor(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate an implied metaphor using verbs."""
        # Find verbs associated with target
        target_verbs = []

        # Check our verb associations
        for key, verbs in self.verb_associations.items():
            if key in target.lower():
                target_verbs.extend(verbs)
                break

        # If no specific verbs, get from frequently following words
        if not target_verbs:
            following = frequently_following_words(target, sample_size=10)
            # Filter for likely verbs (simple heuristic)
            target_verbs = [w for w in following if w.endswith(("s", "ed", "ing"))]

        if not target_verbs:
            target_verbs = ["becomes", "transforms", "emerges"]

        verb = random.choice(target_verbs)
        text = f"{source} {verb}"

        # Add object if appropriate
        if random.random() > 0.5:
            abstract_object = random.choice(vocabulary.abstract_entities)
            text += f" {abstract_object}"

        grounds = self._find_connecting_attributes(source, target)
        quality = self._score_metaphor(source, target, grounds) * 0.9  # Slightly lower for implied

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.IMPLIED,
            quality_score=quality,
            grounds=grounds,
        )

    def _generate_possessive_metaphor(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate a possessive metaphor."""
        pattern = random.choice(self.possessive_patterns)
        text = pattern.format(source=source, target=target)

        grounds = self._find_connecting_attributes(source, target)
        quality = self._score_metaphor(source, target, grounds)

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.POSSESSIVE,
            quality_score=quality,
            grounds=grounds,
        )

    def _generate_compound_metaphor(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate a compound metaphor (hyphenated or fused terms)."""
        pattern = random.choice(self.compound_patterns)
        text = pattern.format(source=source, target=target)

        grounds = self._find_connecting_attributes(source, target)
        # Compound metaphors are terse and striking - slight quality bonus
        quality = self._score_metaphor(source, target, grounds) * 1.05

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.COMPOUND,
            quality_score=min(1.0, quality),
            grounds=grounds,
        )

    def _generate_conceptual_metaphor(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate a conceptual metaphor (abstract domain mapping)."""
        pattern = random.choice(self.conceptual_patterns)
        text = pattern.format(source=source, target=target)

        grounds = self._find_connecting_attributes(source, target)
        # Conceptual metaphors are intellectually rich
        quality = self._score_metaphor(source, target, grounds) * 1.1

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.CONCEPTUAL,
            quality_score=min(1.0, quality),
            grounds=grounds,
        )

    def generate_extended_metaphor(self, source: str, target: str) -> Metaphor:
        """Generate an extended, multi-line metaphor."""
        lines = []

        # Opening comparison
        lines.append(f"{source.capitalize()} is {target}.")

        # Develop with attributes
        grounds = self._find_connecting_attributes(source, target)

        # Add elaborations
        if grounds:
            lines.append(
                f"Both {grounds[0]}, both {grounds[1] if len(grounds) > 1 else 'endless'}."
            )

        # Add action
        target_verbs = self.verb_associations.get(target.lower(), ["moves", "shifts", "changes"])
        if target_verbs:
            verb = random.choice(target_verbs)
            lines.append(f"It {verb} through {random.choice(vocabulary.abstract_entities[:8])}.")

        # Closing transformation
        transformation = random.choice(
            [
                f"Until {source} and {target} are one.",
                f"Where {target} ends, {source} begins.",
                f"In this light, all {source} becomes {target}.",
            ]
        )
        lines.append(transformation)

        text = "\n".join(lines)
        quality = self._score_metaphor(source, target, grounds) * 1.2  # Bonus for extended

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.EXTENDED,
            quality_score=min(1.0, quality),
            grounds=grounds,
        )

    def mine_gutenberg_for_domain(self, domain: str, sample_size: int = 5) -> List[Tuple[str, str]]:
        """Mine Gutenberg texts for metaphors in a specific domain.

        Args:
            domain: The domain to search for
            sample_size: Number of texts to sample

        Returns:
            List of (metaphor_text, source_text) tuples
        """
        found_metaphors = []

        for _ in range(sample_size):
            patterns = self.extract_metaphor_patterns()
            for source, target, sentence in patterns:
                if domain.lower() in source.lower() or domain.lower() in target.lower():
                    found_metaphors.append((f"{source} like {target}", sentence))

        return found_metaphors

    def _find_connecting_attributes(self, source: str, target: str) -> List[str]:
        """Find attributes that connect source and target.

        Enhanced to use word embeddings for semantic similarity when available.
        """
        from .quality_scorer import get_quality_scorer

        attributes = []
        scorer = get_quality_scorer()

        # Try multiple approaches to find connections

        # 1. Get descriptive words for both (expanded search)
        try:
            source_context = contextually_linked_words(source, sample_size=15, datamuse_api_max=25)
            target_context = contextually_linked_words(target, sample_size=15, datamuse_api_max=25)

            # Find overlaps
            if source_context and target_context:
                overlap = set(source_context) & set(target_context)
                # Filter by quality - only keep good connecting words
                quality_attributes = []
                for attr in overlap:
                    if not scorer.is_cliche(attr):  # Avoid clichéd connections
                        quality_score = scorer.score_word(attr).overall
                        if quality_score > 0.5:  # Quality threshold
                            quality_attributes.append((attr, quality_score))

                # Sort by quality
                quality_attributes.sort(key=lambda x: x[1], reverse=True)
                attributes.extend([attr for attr, _ in quality_attributes[:5]])
        except Exception:
            pass

        # 2. If no overlap, try similar meaning words
        if not attributes:
            try:
                source_similar = similar_meaning_words(source, sample_size=10)
                target_similar = similar_meaning_words(target, sample_size=10)

                # Check if any similar words overlap
                if source_similar and target_similar:
                    overlap = set(source_similar) & set(target_similar)
                    # Filter by quality
                    quality_overlap = [w for w in overlap if not scorer.is_cliche(w)]
                    attributes.extend(quality_overlap)
            except Exception:
                pass

        # 3. Try semantic bridge words using concreteness preference
        if len(attributes) < 2:
            try:
                # Generate candidate attributes from context
                candidates = []
                for word in [source, target]:
                    try:
                        context = contextually_linked_words(word, sample_size=10)
                        candidates.extend(context)
                    except Exception:
                        pass

                # Score candidates by how well they bridge the gap
                if candidates:
                    bridge_candidates = []
                    for candidate in set(candidates):
                        if not scorer.is_cliche(candidate):
                            # Prefer concrete connecting words for vivid imagery
                            concreteness = scorer.get_concreteness(candidate)
                            quality = scorer.score_word(candidate).overall
                            # Combined score: prefer concrete + high quality
                            bridge_score = (concreteness * 0.6) + (quality * 0.4)
                            bridge_candidates.append((candidate, bridge_score))

                    # Take top candidates
                    bridge_candidates.sort(key=lambda x: x[1], reverse=True)
                    attributes.extend([c for c, _ in bridge_candidates[:3]])
            except Exception:
                pass

        # 4. Generate thematic attributes based on word characteristics
        if not attributes:
            attributes = self._generate_thematic_attributes(source, target)

        # 5. Last resort: use shared vocabulary for diverse poetic attributes
        # But filter for quality
        if not attributes:
            candidates = vocabulary.get_random_attributes(count=10)
            quality_candidates = [c for c in candidates if not scorer.is_cliche(c)]
            attributes = quality_candidates[:3] if quality_candidates else candidates[:3]

        return attributes[:3]  # Limit to 3 best

    def _generate_thematic_attributes(self, source: str, target: str) -> List[str]:
        """Generate attributes based on the thematic nature of the words."""
        attributes = []

        # Time-related words
        time_words = {"morning", "evening", "night", "dawn", "dusk", "noon", "midnight", "season"}
        emotion_words = {"disappointment", "joy", "sorrow", "anger", "fear", "love", "hope"}
        nature_words = {"ocean", "mountain", "forest", "river", "sky", "earth", "storm", "calm"}

        # Generate contextual attributes
        all_words = {source.lower(), target.lower()}

        if any(word in time_words for word in all_words):
            attributes.extend(vocabulary.get_thematic_words("temporal", count=2))
        elif any(word in emotion_words for word in all_words):
            attributes.extend(vocabulary.get_thematic_words("emotional", count=2))
        elif any(word in nature_words for word in all_words):
            attributes.extend(vocabulary.get_thematic_words("natural", count=2))
        else:
            # Use shared vocabulary for general attributes
            attributes.extend(vocabulary.get_random_attributes(count=2))

        return attributes

    def _score_metaphor(self, source: str, target: str, grounds: List[str]) -> float:
        """Score a metaphor for quality using comprehensive quality system."""
        from .quality_scorer import get_quality_scorer

        scorer = get_quality_scorer()
        score = 0.5  # Base score

        # 1. Novelty: Check against comprehensive cliché database
        metaphor_phrase = f"{source} is {target}"
        if scorer.is_cliche(metaphor_phrase, threshold=0.6):
            score -= 0.3  # Heavy penalty for clichéd metaphors
        else:
            score += 0.2  # Bonus for fresh pairing

        # Also check individual words for overuse
        if scorer.is_cliche(source) or scorer.is_cliche(target):
            score -= 0.1  # Moderate penalty for clichéd words

        # 2. Coherence: Bonus for having connecting grounds
        if grounds:
            # More grounds = stronger connection
            score += 0.1 * min(len(grounds), 3)

            # Quality of grounds matters
            ground_qualities = [scorer.score_word(g).overall for g in grounds[:3]]
            if ground_qualities:
                avg_ground_quality = sum(ground_qualities) / len(ground_qualities)
                score += 0.1 * avg_ground_quality

        # 3. Concreteness balance: Prefer concrete target for vivid imagery
        target_concreteness = scorer.get_concreteness(target)
        source_concreteness = scorer.get_concreteness(source)

        # Ideal: abstract source, concrete target (e.g., "love is a rose")
        if source_concreteness < 0.6 and target_concreteness > 0.7:
            score += 0.15  # Bonus for good imagery
        # Penalize abstract-to-abstract (weak imagery)
        elif source_concreteness < 0.5 and target_concreteness < 0.5:
            score -= 0.1
        # Also good: concrete-to-concrete (clear comparison)
        elif source_concreteness > 0.7 and target_concreteness > 0.7:
            score += 0.05

        # 4. Word quality: High-quality words make better metaphors
        source_quality = scorer.score_word(source).overall
        target_quality = scorer.score_word(target).overall
        avg_word_quality = (source_quality + target_quality) / 2
        score += 0.1 * avg_word_quality

        # 5. Semantic distance: Reward unexpected but coherent connections
        # (Keep existing logic but adjust weight)
        try:
            source_meaning = similar_meaning_words(source, sample_size=5)
            if source_meaning and target not in source_meaning:
                score += 0.1  # Bonus for non-obvious connection
        except Exception:
            pass

        return max(0.0, min(1.0, score))

    def generate_synesthetic_metaphor(self, source: str) -> Optional[Metaphor]:
        """Generate a cross-sensory (synesthetic) metaphor."""
        # Pick two different senses
        senses = random.sample(list(vocabulary.enhanced_sensory_domains.keys()), 2)
        sense1_word = vocabulary.get_sensory_words(senses[0], count=1)[0]
        sense2_word = vocabulary.get_sensory_words(senses[1], count=1)[0]

        text = f"{source}, {sense1_word} as {sense2_word}"

        return Metaphor(
            text=text,
            source=source,
            target=sense2_word,
            metaphor_type=MetaphorType.SYNESTHETIC,
            quality_score=0.7,
            grounds=[senses[0], senses[1]],
        )

"""Dynamic Metaphor Generator using Project Gutenberg texts and cross-domain connections."""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import spacy

from .config import DocumentConfig, QualityConfig
from .core.document_library import get_diverse_gutenberg_documents
from .core.lexigen import (
    contextually_linked_words,
    frequently_following_words,
    related_rare_words,
    similar_meaning_words,
)
from .core.quality_scorer import get_quality_scorer
from .core.vocabulary import vocabulary
from .core.word_validator import word_validator
from .decomposer import ParsedText
from .grammatical_templates import MetaphorPatterns
from .setup_models import lazy_ensure_spacy_model

logger = logging.getLogger(__name__)


# Lazy-load spaCy model for POS tagging
def _get_spacy_nlp():
    """Get spaCy NLP model for POS tagging, downloading if needed."""
    lazy_ensure_spacy_model("en_core_web_sm", "English language model (small)")
    # Keep POS tagger, disable unnecessary components for performance
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return nlp


# Initialize on first use
_spacy_nlp = None


def get_spacy_nlp():
    """Get cached spaCy NLP model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = _get_spacy_nlp()
    return _spacy_nlp


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

    def _get_concreteness(self, word: str) -> float:
        """Get concreteness score for a word (0=abstract, 1=concrete).

        Args:
            word: Word to check

        Returns:
            Concreteness score from 0 (abstract) to 1 (concrete), defaults to 0.5 if unknown
        """
        scorer = get_quality_scorer()
        return scorer.concreteness_cache.get(word.lower(), 0.5)

    def _is_abstract_noun(self, word: str, threshold: float = 0.5) -> bool:
        """Check if a word is an abstract noun.

        Args:
            word: Word to check
            threshold: Concreteness threshold below which word is considered abstract

        Returns:
            True if word is likely an abstract noun
        """
        return self._get_concreteness(word) < threshold

    def _is_concrete_noun(self, word: str, threshold: float = 0.8) -> bool:
        """Check if a word is a concrete noun.

        Args:
            word: Word to check
            threshold: Concreteness threshold above which word is considered concrete

        Returns:
            True if word is likely a concrete noun
        """
        return self._get_concreteness(word) > threshold

    def _is_poetic_possessive(self, source: str, target: str) -> bool:
        """Check if a possessive construction (X of Y) is poetic.

        Poetic possessives mix different types of imagery AND avoid literal relationships.

        Poetic examples:
        - "silence of stone" ✓ (mix + not semantically related)
        - "shadow of desire" ✓ (mix + not literal)

        Non-poetic examples:
        - "door of house" ✗ (both concrete + too similar)
        - "strength of muscles" ✗ (literal/semantic relationship)
        - "truth of reason" ✗ (both abstract)

        Args:
            source: The first noun
            target: The second noun

        Returns:
            True if this is likely a poetic possessive
        """
        # Get concreteness scores
        source_score = self._get_concreteness(source)
        target_score = self._get_concreteness(target)

        # 1. REJECT if both are high-confidence CONCRETE
        # Simple rule: concrete + concrete = literal, not poetic
        if source_score > 0.6 and target_score > 0.6:
            return False  # e.g., "trunk of tree", "door of house" - LITERAL

        # 2. REJECT if both are high-confidence ABSTRACT
        # Simple rule: abstract + abstract = vague, not poetic
        if source_score < 0.4 and target_score < 0.4:
            return False  # e.g., "truth of reason" - VAGUE

        # 3. REJECT if either word is in the "dead zone" (0.4-0.6)
        # Words in this range are neither clearly concrete nor abstract
        # This includes unknown words (default 0.5) which lack clear imagery
        if 0.4 <= source_score <= 0.6 or 0.4 <= target_score <= 0.6:
            return False  # e.g., "chapter of nine", "unknownword123 of stone" - NEUTRAL

        # 4. REJECT if they are TOO LITERAL (semantically similar)
        try:
            similar_words = similar_meaning_words(source, sample_size=10)
            if target in similar_words:
                return False  # e.g., "strength of muscles" - LITERAL
        except Exception:
            pass  # Continue if API fails

        # 5. Otherwise, it passes all checks - likely poetic
        # Good metaphors mix concrete + abstract or are semantically distant
        return True

    def _is_poetic_simile(self, source: str, target: str, sentence: str) -> bool:
        """Check if a simile construction (X as Y) is poetic.

        Poetic similes typically have:
        - Adjective/noun as noun (e.g., "swift as wind", "cold as ice")

        Non-poetic similes:
        - Verb patterns (e.g., "regarded as different")
        - Numeric patterns (e.g., "one as ten")

        Args:
            source: The first word
            target: The second word
            sentence: Full sentence for context

        Returns:
            True if this is likely a poetic simile
        """
        # 1. REJECT if they are TOO LITERAL (semantically similar)
        try:
            similar_words = similar_meaning_words(source, sample_size=10)
            if target in similar_words:
                return False  # e.g., "strong as strength" - LITERAL
        except Exception:
            pass  # Continue if API fails

        # 2. Use spaCy to check POS tags
        try:
            nlp = get_spacy_nlp()
            doc = nlp(sentence.lower())

            # Find the tokens
            source_token = None
            target_token = None

            for token in doc:
                if token.text == source.lower() and source_token is None:
                    source_token = token
                if token.text == target.lower():
                    target_token = token

            if not source_token or not target_token:
                return False  # Fail-closed: if we can't analyze it, reject it

            # Check POS tags - we want adjective/noun patterns
            # Reject if source is a verb or number
            if source_token.pos_ in ["VERB", "NUM", "AUX"]:
                return False

            # Reject if target is a verb or number
            if target_token.pos_ in ["VERB", "NUM", "AUX", "ADJ"]:
                return False

            # Accept if source is ADJ/NOUN and target is NOUN, otherwise reject
            return source_token.pos_ in ["ADJ", "NOUN"] and target_token.pos_ == "NOUN"

        except Exception:
            # If spaCy fails, fail-closed and reject
            return False

    def extract_metaphor_patterns(
        self, num_texts: int = 3, verbose: bool = True, target_count: int = 15
    ) -> List[Metaphor]:
        """Extract metaphorical patterns from multiple Gutenberg texts.

        Args:
            num_texts: Number of different texts to sample from (default: 3)
            verbose: Whether to print progress messages (default: True for CLI, False for TUI)
            target_count: Target number of metaphors to extract (default: 15)

        Returns:
            List of Metaphor objects with quality scores, types, and context
        """
        all_metaphors = []

        # Get diverse documents using centralized system
        # Filter by literary LoCC codes: PZ (Fiction), PR (English Lit), PS (American Lit)
        if verbose:
            print(
                f"📚 Retrieving {num_texts} diverse literary documents for metaphor extraction..."
            )
        documents = get_diverse_gutenberg_documents(
            count=num_texts,
            min_length=DocumentConfig.MIN_LENGTH_METAPHORS,
            locc_codes=["PZ", "PR", "PS"],  # Fiction and literature only
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
        min_target = target_count  # Use user's requested count
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
                count=additional_batch,
                min_length=DocumentConfig.MIN_LENGTH_METAPHORS,
                locc_codes=["PZ", "PR", "PS"],  # Fiction and literature only
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

        # Remove duplicates based on source-target pairs
        unique_metaphors = []
        seen_pairs = set()
        quality_filtered = 0

        for metaphor in all_metaphors:
            pair = (metaphor.source.lower(), metaphor.target.lower())

            if pair not in seen_pairs and len(metaphor.source) > 2 and len(metaphor.target) > 2:
                # Quality score already calculated in _extract_metaphors_from_text
                # Apply threshold of 0.5 for final selection
                if metaphor.quality_score >= 0.5:
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
            # Common literal possessive constructions
            ("university", "paris"),
            ("university", "oxford"),
            ("university", "cambridge"),
            ("institute", "france"),
            ("member", "parliament"),
            ("officer", "legion"),
            ("professor", "law"),
            ("end", "november"),
            ("end", "december"),
            ("end", "january"),
            ("end", "year"),
            ("end", "month"),
            ("door", "house"),
            ("roof", "house"),
            ("floor", "house"),
            ("wall", "house"),
            ("sir", "word"),
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
            from .core.quality_scorer import get_quality_scorer

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
    ) -> List[Metaphor]:
        """Extract metaphors from a single text using POS-first pattern matching.

        This uses spaCy to POS-tag sentences first, then looks for specific
        grammatical patterns that indicate metaphorical language.

        Returns:
            List of Metaphor objects with quality scores and type classification
        """
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
                # HYBRID APPROACH: Use regex to find CANDIDATES, then apply SMART FILTERS

                # Pattern 1: Possessives (X of Y)
                possessive_pattern = r"\b(\w+)\s+of\s+(\w+)\b"
                for match in re.finditer(possessive_pattern, sentence.lower()):
                    source, target = match.groups()

                    # Filter 0: Skip function words (articles, pronouns, etc.)
                    function_words = {
                        "the",
                        "a",
                        "an",
                        "this",
                        "that",
                        "these",
                        "those",
                        "my",
                        "your",
                        "his",
                        "her",
                        "its",
                        "our",
                        "their",
                    }
                    if source in function_words or target in function_words:
                        continue

                    # Filter 1: NER - Skip proper nouns
                    if word_validator.is_proper_noun(source) or word_validator.is_proper_noun(
                        target
                    ):
                        continue

                    # Filter 2: Concreteness - Require poetic abstract+concrete mix
                    if not self._is_poetic_possessive(source, target):
                        continue

                    # Filter 3: Final validation
                    if self._is_valid_metaphor_pair(source, target):
                        found_metaphors.append((source, target, sentence, MetaphorType.POSSESSIVE))

                # Pattern 2: Similes (X as Y)
                simile_pattern = r"\b(\w+)\s+as\s+(\w+)\b"
                for match in re.finditer(simile_pattern, sentence.lower()):
                    source, target = match.groups()

                    # Filter 1: NER - Skip proper nouns
                    if word_validator.is_proper_noun(source) or word_validator.is_proper_noun(
                        target
                    ):
                        continue

                    # Filter 2: POS - Check if it's a poetic simile (ADJ/NOUN as NOUN, not VERB/NUM)
                    if not self._is_poetic_simile(source, target, sentence):
                        continue

                    # Filter 3: Final validation
                    if self._is_valid_metaphor_pair(source, target):
                        found_metaphors.append((source, target, sentence, MetaphorType.SIMILE))

            # Sort by quality using comprehensive quality scoring
            if found_metaphors:
                # Score each metaphor and create Metaphor objects
                metaphor_objects = []
                for source, target, sentence, metaphor_type in found_metaphors:
                    # Calculate quality score for this metaphor
                    quality = self._score_metaphor(source, target, [])

                    # Also consider sentence clarity (shorter is often clearer)
                    clarity_bonus = max(0, 1.0 - (len(sentence) / 200))  # Bonus for concise context

                    # Combined score: 80% metaphor quality, 20% clarity
                    combined_score = (quality * 0.8) + (clarity_bonus * 0.2)

                    # Create Metaphor object with all metadata
                    metaphor_obj = Metaphor(
                        text=f"{source} → {target}",
                        source=source,
                        target=target,
                        metaphor_type=metaphor_type,
                        quality_score=combined_score,
                        grounds=[],  # Could be enhanced later
                        source_text=sentence,  # Store full context sentence
                    )
                    metaphor_objects.append(metaphor_obj)

                # Sort by quality score (descending)
                metaphor_objects.sort(key=lambda m: m.quality_score, reverse=True)

                if verbose:
                    if is_additional:
                        print(
                            f"    ✓ Found {len(metaphor_objects)} additional metaphor patterns (quality-sorted)"
                        )
                    else:
                        print(
                            f"    ✓ Found {len(metaphor_objects)} metaphor patterns (quality-sorted)"
                        )

                return metaphor_objects
            else:
                if verbose:
                    print("    ✓ Found 0 metaphor patterns")
                return []

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

    def generate_metaphor_from_pair(self, source: str, target: str) -> Optional[Metaphor]:
        """Generate a high-quality metaphor from a specific source-target pair.

        This is a targeted metaphor generation method designed for use by the
        GeneratePoemScaffold strategy. It bypasses Gutenberg text extraction
        and directly creates a metaphor using the generator's quality-aware
        internal logic.

        The method reuses the full "smart" pipeline:
        - Finds connecting attributes via _find_connecting_attributes()
        - Scores using _score_metaphor() with comprehensive quality checks
        - Validates against NER, POS, concreteness, and semantic distance
        - Returns None if the metaphor doesn't meet quality thresholds

        Args:
            source: The tenor (what's being described), e.g., "rust"
            target: The vehicle (what it's compared to), e.g., "memory"

        Returns:
            A high-quality Metaphor object, or None if no suitable metaphor can be formed

        Example:
            >>> gen = MetaphorGenerator()
            >>> metaphor = gen.generate_metaphor_from_pair("rust", "memory")
            >>> print(metaphor.text)
            "rust as a quiet memory"
            >>> print(metaphor.quality_score)
            0.88
        """
        # Validate inputs - both source and target must be valid words
        if not word_validator.is_valid_english_word(
            source
        ) or not word_validator.is_valid_english_word(target):
            logger.debug(f"Invalid source or target word: {source}, {target}")
            return None

        # Find connecting attributes using the smart semantic search
        grounds = self._find_connecting_attributes(source, target)

        # If no grounds found, the metaphor is too weak to be useful
        if not grounds:
            logger.debug(f"No connecting grounds found for {source} -> {target}")
            return None

        # Score the metaphor using the comprehensive quality system
        quality = self._score_metaphor(source, target, grounds)

        # Quality threshold: Only return high-quality metaphors (> 0.5)
        # This ensures the scaffold provides genuinely useful creative material
        if quality < 0.5:
            logger.debug(f"Metaphor quality too low ({quality:.2f}): {source} -> {target}")
            return None

        # Choose an appropriate metaphor pattern based on word characteristics
        # Prefer possessive for concrete-abstract pairs, direct for others
        target_concreteness = self._get_concreteness(target)
        source_concreteness = self._get_concreteness(source)

        if target_concreteness > 0.7 and source_concreteness < 0.5:
            # Abstract -> Concrete: Use possessive pattern ("X of Y")
            pattern = random.choice(self.possessive_patterns)
            metaphor_type = MetaphorType.POSSESSIVE
        elif abs(target_concreteness - source_concreteness) < 0.3:
            # Similar concreteness: Use direct pattern ("X is Y")
            pattern = random.choice(self.direct_patterns)
            metaphor_type = MetaphorType.DIRECT
        else:
            # Mixed: Use simile pattern ("X like Y")
            pattern = random.choice(self.simile_patterns)
            metaphor_type = MetaphorType.SIMILE

        # Build the metaphor text
        text = pattern.format(source=source, target=target)

        # Create and return the Metaphor object
        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=metaphor_type,
            quality_score=min(1.0, quality),
            grounds=grounds,
            source_text=None,  # Not derived from Gutenberg
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
        from .core.quality_scorer import get_quality_scorer

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

        # Use vocabulary.concept_domains instead of hardcoded sets
        all_words = {source.lower(), target.lower()}

        # Check if words are in time domain
        time_words = vocabulary.concept_domains.get("time", set())
        if any(word in time_words for word in all_words):
            attributes.extend(vocabulary.get_thematic_words("temporal", count=2))
        # Check if words are in nature domain
        elif any(word in vocabulary.concept_domains.get("nature", set()) for word in all_words):
            attributes.extend(vocabulary.get_thematic_words("natural", count=2))
        # Check if words have emotional connotations
        elif any(
            any(word in keywords for keywords in vocabulary.emotional_keywords.values())
            for word in all_words
        ):
            attributes.extend(vocabulary.get_thematic_words("emotional", count=2))
        else:
            # Use shared vocabulary for general attributes
            attributes.extend(vocabulary.get_random_attributes(count=2))

        return attributes

    def _score_metaphor(self, source: str, target: str, grounds: List[str]) -> float:
        """Score a metaphor for quality using comprehensive quality system."""
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

    def generate_metaphor_with_constraints(
        self,
        source_words: Optional[List[str]] = None,
        semantic_domains: Optional[List[str]] = None,
        metaphor_types: Optional[List[str]] = None,
        min_quality_score: float = 0.6,
        count: int = 10,
    ) -> List[Metaphor]:
        """Generate metaphors with template-based constraints.

        This method exposes public API for constrained metaphor generation,
        enabling template-aware poem generation.

        Args:
            source_words: Optional list of source words to use. If None, samples from semantic_domains
            semantic_domains: Optional list of semantic domains to constrain vocabulary
                            (e.g., ["nature", "emotion"]). If None, uses all domains.
            metaphor_types: Optional list of metaphor types to generate
                          (e.g., ["simile", "direct", "possessive"]). If None, generates all types.
            min_quality_score: Minimum quality threshold (0.0-1.0)
            count: Number of metaphors to return

        Returns:
            List of Metaphor objects matching constraints, sorted by quality score

        Example:
            >>> gen = MetaphorGenerator()
            >>> metaphors = gen.generate_metaphor_with_constraints(
            ...     semantic_domains=["nature", "emotion"],
            ...     metaphor_types=["simile", "direct"],
            ...     min_quality_score=0.7,
            ...     count=5
            ... )
        """
        # 1. Determine source words
        if source_words is None:
            source_words = []
            # Sample from semantic domains if provided
            if semantic_domains:
                for domain in semantic_domains:
                    if domain in self.domains:
                        source_words.extend(
                            random.sample(self.domains[domain], min(3, len(self.domains[domain])))
                        )
            else:
                # Sample from all domains
                for domain_words in self.domains.values():
                    source_words.extend(random.sample(domain_words, min(2, len(domain_words))))

            # Clean and limit
            source_words = word_validator.clean_word_list(source_words)
            source_words = source_words[:10]  # Limit to prevent explosion

        # 2. Determine metaphor types to generate
        type_generators = {
            "simile": self._generate_simile,
            "direct": self._generate_direct_metaphor,
            "implied": self._generate_implied_metaphor,
            "possessive": self._generate_possessive_metaphor,
            "compound": self._generate_compound_metaphor,
            "conceptual": self._generate_conceptual_metaphor,
        }

        if metaphor_types:
            # Filter to requested types
            active_generators = {k: v for k, v in type_generators.items() if k in metaphor_types}
        else:
            # Use all types
            active_generators = type_generators

        # 3. Generate metaphors
        metaphors = []

        for source in source_words:
            # Find target words
            if semantic_domains:
                # Constrain targets to semantic domains
                targets = []
                for domain in semantic_domains:
                    if domain in self.domains:
                        # Sample from this domain, excluding source
                        domain_words = [w for w in self.domains[domain] if w != source]
                        if domain_words:
                            targets.extend(random.sample(domain_words, min(2, len(domain_words))))
            else:
                # Use existing method (cross-domain)
                targets = self._find_target_domains(source)

            # Generate using active generators
            for target in targets[:5]:  # Limit targets per source
                for gen_func in active_generators.values():
                    metaphor = gen_func(source, target)
                    if metaphor and metaphor.quality_score >= min_quality_score:
                        metaphors.append(metaphor)

        # 4. Sort by quality and return top results
        metaphors.sort(key=lambda m: m.quality_score, reverse=True)
        return metaphors[:count]

    def generate_metaphor_from_template(
        self,
        line_template: "LineTemplate",  # type: ignore  # noqa: F821  # Deprecated method
        source_words: Optional[List[str]] = None,
        count: int = 5,
    ) -> List[Metaphor]:
        """Generate metaphors matching a LineTemplate's constraints.

        This is a convenience method that extracts constraints from a LineTemplate
        and calls generate_metaphor_with_constraints().

        Args:
            line_template: LineTemplate with semantic_domain, metaphor_type, min_quality_score
            source_words: Optional list of source words. If None, samples from template's domain
            count: Number of metaphors to generate

        Returns:
            List of Metaphor objects matching template constraints

        Example:
            >>> from poetryplayground.poem_template import LineTemplate, LineType
            >>> template = LineTemplate(
            ...     syllable_count=5,
            ...     pos_pattern=["DET", "NOUN", "VERB"],
            ...     line_type=LineType.IMAGE,
            ...     metaphor_type="simile",
            ...     semantic_domain="nature",
            ...     min_quality_score=0.7
            ... )
            >>> gen = MetaphorGenerator()
            >>> metaphors = gen.generate_metaphor_from_template(template)
        """
        # Extract constraints from template
        semantic_domains = (
            [line_template.semantic_domain] if line_template.semantic_domain else None
        )
        metaphor_types = [line_template.metaphor_type] if line_template.metaphor_type else None
        min_quality_score = line_template.min_quality_score

        return self.generate_metaphor_with_constraints(
            source_words=source_words,
            semantic_domains=semantic_domains,
            metaphor_types=metaphor_types,
            min_quality_score=min_quality_score,
            count=count,
        )

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

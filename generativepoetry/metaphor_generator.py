"""Dynamic Metaphor Generator using Project Gutenberg texts and cross-domain connections."""

import random
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

from .lexigen import (
    similar_meaning_words,
    contextually_linked_words,
    frequently_following_words,
    related_rare_words
)
from .decomposer import (
    get_gutenberg_document,
    random_gutenberg_document,
    ParsedText
)
from .utils import filter_word_list

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
        """Initialize semantic domain categories."""
        self.domains = {
            'nature': ['ocean', 'forest', 'storm', 'garden', 'river', 'mountain',
                      'desert', 'rain', 'snow', 'tree', 'flower', 'sky'],
            'architecture': ['cathedral', 'bridge', 'tower', 'ruins', 'door',
                           'window', 'wall', 'foundation', 'arch', 'castle'],
            'time': ['clock', 'season', 'dawn', 'century', 'moment', 'hour',
                    'twilight', 'midnight', 'autumn', 'eternity'],
            'body': ['heart', 'bones', 'blood', 'breath', 'skin', 'eyes',
                    'hands', 'pulse', 'nerves', 'spine'],
            'cosmos': ['stars', 'void', 'orbit', 'constellation', 'moon',
                      'comet', 'galaxy', 'nebula', 'eclipse', 'gravity'],
            'technology': ['engine', 'wire', 'signal', 'machine', 'circuit',
                          'network', 'code', 'algorithm', 'data', 'static'],
            'textiles': ['thread', 'weave', 'fray', 'pattern', 'fabric',
                        'silk', 'tapestry', 'knot', 'stitch', 'loom'],
            'music': ['symphony', 'discord', 'rhythm', 'silence', 'melody',
                     'harmony', 'crescendo', 'note', 'chord', 'echo'],
            'light': ['shadow', 'glow', 'gleam', 'flicker', 'radiance',
                     'twilight', 'beam', 'reflection', 'prism', 'darkness'],
            'container': ['vessel', 'bowl', 'cage', 'box', 'envelope',
                         'bottle', 'frame', 'shell', 'cocoon', 'womb']
        }

    def _init_patterns(self):
        """Initialize metaphor pattern templates."""
        self.simile_patterns = [
            "{source} is like {target}",
            "{source} like {target}",
            "{source}, like {target},",
            "as {adjective} as {target}",
            "{source} resembles {target}"
        ]

        self.direct_patterns = [
            "{source} is {target}",
            "{source}: {target}",
            "{source}, a {target}",
            "the {target} of {source}",
            "{source} becomes {target}"
        ]

        self.possessive_patterns = [
            "{target}'s {source}",
            "the {source} of {target}",
            "{target}-{source}",
            "{source}, {target}'s child"
        ]

        self.appositive_patterns = [
            "{source}, that {target}",
            "{source}, the {target}",
            "{source} — {target} —",
            "{source} (a {target})"
        ]

    def _init_verb_associations(self):
        """Initialize verb associations for implied metaphors."""
        self.verb_associations = {
            'ocean': ['drowns', 'floods', 'ebbs', 'flows', 'crashes', 'swells'],
            'fire': ['burns', 'consumes', 'flickers', 'ignites', 'smolders'],
            'bird': ['soars', 'nests', 'migrates', 'sings', 'preens'],
            'machine': ['grinds', 'processes', 'outputs', 'breaks down', 'calibrates'],
            'plant': ['grows', 'blooms', 'withers', 'roots', 'branches'],
            'fabric': ['unravels', 'frays', 'weaves', 'tears', 'mends'],
            'light': ['illuminates', 'blinds', 'reveals', 'pierces', 'fades'],
            'music': ['harmonizes', 'crescendos', 'resonates', 'echoes', 'syncopates']
        }

    def extract_metaphor_patterns(self, text_id: Optional[int] = None) -> List[str]:
        """Extract metaphorical patterns from Gutenberg texts.

        Args:
            text_id: Optional Gutenberg text ID. If None, uses random text.

        Returns:
            List of extracted metaphor patterns
        """
        try:
            if text_id:
                text = get_gutenberg_document(text_id)
            else:
                text = random_gutenberg_document()

            if not text:
                return []

            # Parse text
            parsed = ParsedText(text)

            # Patterns to find metaphors
            patterns = [
                r'(\w+)\s+(?:is|was|are|were)\s+like\s+(?:a\s+|an\s+|the\s+)?(\w+)',
                r'(\w+)\s+as\s+(?:a\s+|an\s+|the\s+)?(\w+)',
                r'(\w+),\s+(?:a\s+|an\s+|that\s+|this\s+)(\w+)',
                r'the\s+(\w+)\s+of\s+(\w+)',
            ]

            found_metaphors = []
            for sentence in parsed.sentences[:100]:  # Limit to first 100 sentences
                for pattern in patterns:
                    matches = re.findall(pattern, sentence.lower())
                    for match in matches:
                        if len(match) == 2:
                            source, target = match
                            if self._is_valid_metaphor_pair(source, target):
                                found_metaphors.append((source, target, sentence))

            # Store for later use
            self._gutenberg_patterns.extend(found_metaphors)
            return found_metaphors

        except Exception as e:
            logger.debug(f"Error extracting from Gutenberg: {e}")
            return []

    def _is_valid_metaphor_pair(self, source: str, target: str) -> bool:
        """Check if a source-target pair makes a valid metaphor."""
        # Filter out common non-metaphorical phrases
        invalid_pairs = {
            ('it', 'that'), ('this', 'that'), ('he', 'she'),
            ('one', 'another'), ('some', 'other'), ('man', 'woman')
        }

        if (source, target) in invalid_pairs:
            return False

        # Both should be meaningful words
        if len(source) < 3 or len(target) < 3:
            return False

        # Use word validator
        return (word_validator.is_valid_english_word(source) and
               word_validator.is_valid_english_word(target))

        return True

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
                metaphors.extend([
                    self._generate_simile(source, target),
                    self._generate_direct_metaphor(source, target),
                    self._generate_implied_metaphor(source, target),
                    self._generate_possessive_metaphor(source, target)
                ])

        # Remove None values
        metaphors = [m for m in metaphors if m is not None]

        # Sort by quality
        metaphors.sort(key=lambda m: m.quality_score, reverse=True)

        return metaphors[:count]

    def _find_target_domains(self, source: str) -> List[str]:
        """Find suitable target domains for a source word."""
        targets = []

        # Method 1: Use semantic opposites or contrasts
        for domain, words in self.domains.items():
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
            elaboration = random.choice([
                f", both {grounds[0]}",
                f" in its {grounds[0]}ness",
                f": {grounds[0]}, {grounds[1] if len(grounds) > 1 else 'endless'}"
            ])
            text += elaboration

        quality = self._score_metaphor(source, target, grounds)

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.SIMILE,
            quality_score=quality,
            grounds=grounds
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
            grounds=grounds
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
            target_verbs = [w for w in following if w.endswith(('s', 'ed', 'ing'))]

        if not target_verbs:
            target_verbs = ['becomes', 'transforms', 'emerges']

        verb = random.choice(target_verbs)
        text = f"{source} {verb}"

        # Add object if appropriate
        if random.random() > 0.5:
            objects = ['silence', 'memory', 'time', 'space', 'meaning']
            text += f" {random.choice(objects)}"

        grounds = self._find_connecting_attributes(source, target)
        quality = self._score_metaphor(source, target, grounds) * 0.9  # Slightly lower for implied

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.IMPLIED,
            quality_score=quality,
            grounds=grounds
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
            grounds=grounds
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
            lines.append(f"Both {grounds[0]}, both {grounds[1] if len(grounds) > 1 else 'endless'}.")

        # Add action
        target_verbs = self.verb_associations.get(target.lower(), ['moves', 'shifts', 'changes'])
        if target_verbs:
            verb = random.choice(target_verbs)
            lines.append(f"It {verb} through {random.choice(['time', 'space', 'memory', 'silence'])}.")

        # Closing transformation
        transformation = random.choice([
            f"Until {source} and {target} are one.",
            f"Where {target} ends, {source} begins.",
            f"In this light, all {source} becomes {target}."
        ])
        lines.append(transformation)

        text = "\n".join(lines)
        quality = self._score_metaphor(source, target, grounds) * 1.2  # Bonus for extended

        return Metaphor(
            text=text,
            source=source,
            target=target,
            metaphor_type=MetaphorType.EXTENDED,
            quality_score=min(1.0, quality),
            grounds=grounds
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
        """Find attributes that connect source and target."""
        attributes = []

        # Get descriptive words for both
        source_context = contextually_linked_words(source, sample_size=10)
        target_context = contextually_linked_words(target, sample_size=10)

        # Find overlaps
        if source_context and target_context:
            overlap = set(source_context) & set(target_context)
            attributes.extend(list(overlap))

        # Add some poetic attributes if none found
        if not attributes:
            poetic_attributes = [
                'endless', 'silent', 'forgotten', 'luminous', 'fractured',
                'eternal', 'fleeting', 'hidden', 'transparent', 'infinite'
            ]
            attributes = random.sample(poetic_attributes, 2)

        return attributes[:3]  # Limit to 3

    def _score_metaphor(self, source: str, target: str, grounds: List[str]) -> float:
        """Score a metaphor for quality."""
        score = 0.5  # Base score

        # Novelty: Penalize very common pairings
        common_pairs = {
            ('life', 'journey'), ('love', 'fire'), ('time', 'river'),
            ('mind', 'ocean'), ('death', 'sleep'), ('heart', 'stone')
        }
        if (source, target) not in common_pairs:
            score += 0.2

        # Coherence: Bonus for having connecting grounds
        if grounds:
            score += 0.1 * min(len(grounds), 3)

        # Vividness: Bonus for concrete imagery
        concrete_words = set()
        for domain_words in self.domains.values():
            concrete_words.update(domain_words)

        if target in concrete_words:
            score += 0.1

        # Semantic distance: Reward unexpected connections
        source_meaning = similar_meaning_words(source, sample_size=5)
        if source_meaning and target not in source_meaning:
            score += 0.1

        return min(1.0, score)

    def generate_synesthetic_metaphor(self, source: str) -> Optional[Metaphor]:
        """Generate a cross-sensory (synesthetic) metaphor."""
        sensory_domains = {
            'sight': ['color', 'bright', 'dark', 'shimmer', 'blur'],
            'sound': ['loud', 'quiet', 'melody', 'echo', 'whisper'],
            'touch': ['rough', 'smooth', 'cold', 'warm', 'sharp'],
            'taste': ['sweet', 'bitter', 'sour', 'salt', 'umami'],
            'smell': ['fragrant', 'pungent', 'fresh', 'musty', 'acrid']
        }

        # Pick two different senses
        senses = random.sample(list(sensory_domains.keys()), 2)
        sense1_word = random.choice(sensory_domains[senses[0]])
        sense2_word = random.choice(sensory_domains[senses[1]])

        text = f"{source}, {sense1_word} as {sense2_word}"

        return Metaphor(
            text=text,
            source=source,
            target=sense2_word,
            metaphor_type=MetaphorType.SYNESTHETIC,
            quality_score=0.7,
            grounds=[senses[0], senses[1]]
        )
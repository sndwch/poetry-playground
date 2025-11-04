"""
Resonant Fragment Miner

Discovers evocative sentence fragments from classic literature that could
serve as seeds for compression poetry. Mines multiple sentence patterns
for fragments with poetic weight and causal resonance.

Example fragments: "The door slammed", "When silence fell", "Every shadow whispered"
"""

import re
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from .word_validator import WordValidator
from .decomposer import random_gutenberg_document
from .vocabulary import vocabulary


@dataclass
class ResonantFragment:
    """A discovered fragment with poetic potential"""
    text: str                    # The fragment text
    pattern_type: str           # Which pattern matched (causality, temporal, etc.)
    word_count: int             # Length in words
    source_preview: str         # Brief context from source
    emotional_tone: str         # Detected emotional quality
    poetic_score: float         # How evocative it seems (0-1)


@dataclass
class FragmentCollection:
    """Collection of fragments organized by pattern type"""
    causality: List[ResonantFragment] = field(default_factory=list)
    temporal: List[ResonantFragment] = field(default_factory=list)
    universal: List[ResonantFragment] = field(default_factory=list)
    singular: List[ResonantFragment] = field(default_factory=list)
    modal: List[ResonantFragment] = field(default_factory=list)

    def total_count(self) -> int:
        return len(self.causality) + len(self.temporal) + len(self.universal) + len(self.singular) + len(self.modal)

    def get_all_fragments(self) -> List[ResonantFragment]:
        return self.causality + self.temporal + self.universal + self.singular + self.modal


class ResonantFragmentMiner:
    """Mine evocative fragments from classic literature"""

    def __init__(self):
        self.word_validator = WordValidator()

        # Fragment patterns to search for (more flexible)
        self.patterns = {
            'causality': [
                r'The [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',           # "The door slammed."
                r'The [a-zA-Z]+ had [a-zA-Z]+[^.!?]{0,50}[.!?]',       # "The light had faded."
                r'[A-Z][a-zA-Z]+ [a-zA-Z]+ed[^.!?]{0,40}[.!?]',        # "Something crashed."
            ],
            'temporal': [
                r'When [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',          # "When silence fell."
                r'As [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',            # "As darkness crept."
                r'Until [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',         # "Until morning broke."
                r'Before [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',        # "Before dawn arrived."
                r'After [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',         # "After thunder roared."
            ],
            'universal': [
                r'Every [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',         # "Every shadow whispered."
                r'All [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',           # "All hope vanished."
                r'No [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',            # "No sound escaped."
                r'Each [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',          # "Each breath echoed."
            ],
            'singular': [
                r'One [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',           # "One candle burned."
                r'Only [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]',          # "Only silence remained."
                r'Something [a-zA-Z]+[^.!?]{0,50}[.!?]',                # "Something stirred."
                r'Nothing [a-zA-Z]+[^.!?]{0,50}[.!?]',                  # "Nothing moved."
            ],
            'modal': [
                r'[A-Z][a-zA-Z]+ would [a-zA-Z]+[^.!?]{0,50}[.!?]',    # "Memory would return."
                r'[A-Z][a-zA-Z]+ could [a-zA-Z]+[^.!?]{0,50}[.!?]',    # "Fire could consume."
                r'[A-Z][a-zA-Z]+ might [a-zA-Z]+[^.!?]{0,50}[.!?]',    # "Wind might whisper."
            ]
        }

        # Use shared vocabulary for emotional tone classification

    def mine_fragments(self, target_count: int = 50, num_texts: int = 5) -> FragmentCollection:
        """Mine resonant fragments from classic literature"""
        print(f"🔍 Mining {target_count} resonant fragments from {num_texts} classic texts...")

        collection = FragmentCollection()
        attempts = 0
        max_attempts = num_texts * 2

        while collection.total_count() < target_count and attempts < max_attempts:
            attempts += 1

            try:
                # Get a random Gutenberg text
                text = random_gutenberg_document()
                if not text or len(text) < 1000:
                    continue

                print(f"  Scanning text {attempts}...")

                # Mine fragments from this text
                fragments = self._extract_fragments_from_text(text)

                # Add the best fragments to our collection
                for fragment in fragments:
                    if collection.total_count() >= target_count:
                        break
                    self._add_fragment_to_collection(fragment, collection)

                # Brief pause
                time.sleep(0.5)

            except Exception as e:
                print(f"    Warning: Error processing text {attempts}: {e}")
                continue

        print(f"Successfully mined {collection.total_count()} fragments!")
        return collection

    def _extract_fragments_from_text(self, text: str) -> List[ResonantFragment]:
        """Extract resonant fragments from a single text"""
        fragments = []
        source_preview = text[:100].replace('\n', ' ').strip()

        # Clean the text - remove excessive whitespace and weird characters
        cleaned_text = re.sub(r'\s+', ' ', text)
        cleaned_text = re.sub(r'[^\w\s.!?,:;\'"-]', ' ', cleaned_text)

        # Search for each pattern type
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text, re.MULTILINE)

                for match in matches[:10]:  # Limit per pattern to avoid spam
                    fragment_text = match.strip()

                    # Quality filter
                    if not self._is_good_fragment(fragment_text):
                        continue

                    # Create fragment
                    fragment = ResonantFragment(
                        text=fragment_text,
                        pattern_type=pattern_type,
                        word_count=len(fragment_text.split()),
                        source_preview=source_preview,
                        emotional_tone=self._detect_emotional_tone(fragment_text),
                        poetic_score=self._calculate_poetic_score(fragment_text)
                    )

                    fragments.append(fragment)

        return fragments

    def _is_good_fragment(self, text: str) -> bool:
        """Check if fragment is worth keeping"""
        if not text or len(text.strip()) < 5:
            return False

        words = text.split()

        # Check length constraints (more permissive)
        if len(words) < 2 or len(words) > 18:
            return False

        # Must end with punctuation
        if not text.strip().endswith(('.', '!', '?')):
            return False

        # Avoid obvious non-literary content
        if any(char in text for char in ['(', ')', '[', ']']) or any(word.isdigit() for word in words):
            return False

        # Use shared vocabulary for evocative word detection
        has_interesting = any(vocabulary.is_evocative_word(word) for word in words)

        # Accept if contains interesting words OR if it just seems like good English prose
        seems_literary = len([w for w in words if len(w) > 3]) >= len(words) * 0.4

        return has_interesting or seems_literary

    def _detect_emotional_tone(self, text: str) -> str:
        """Detect the emotional tone of a fragment"""
        return vocabulary.get_emotional_tone(text)

    def _calculate_poetic_score(self, text: str) -> float:
        """Calculate how poetic/evocative a fragment seems"""
        score = 0.5  # Base score
        words = text.split()

        # Bonus for any evocative words (using shared vocabulary)
        evocative_count = sum(1 for word in words if vocabulary.is_evocative_word(word))
        score += min(0.3, evocative_count * 0.1)  # Up to 0.3 bonus

        # Bonus for emotional tone (not neutral)
        tone = vocabulary.get_emotional_tone(text)
        if tone != 'neutral':
            score += 0.2

        # Bonus for shorter fragments (more compressed)
        if len(words) <= 6:
            score += 0.1

        return min(1.0, score)

    def _add_fragment_to_collection(self, fragment: ResonantFragment, collection: FragmentCollection):
        """Add fragment to the appropriate category in collection"""
        if fragment.pattern_type == 'causality':
            collection.causality.append(fragment)
        elif fragment.pattern_type == 'temporal':
            collection.temporal.append(fragment)
        elif fragment.pattern_type == 'universal':
            collection.universal.append(fragment)
        elif fragment.pattern_type == 'singular':
            collection.singular.append(fragment)
        elif fragment.pattern_type == 'modal':
            collection.modal.append(fragment)

    def format_fragment_collection(self, collection: FragmentCollection) -> str:
        """Format the fragment collection for display"""
        if collection.total_count() == 0:
            return "No fragments found."

        report = []
        report.append("🔍 RESONANT FRAGMENT COLLECTION")
        report.append("=" * 60)
        report.append(f"Total fragments discovered: {collection.total_count()}")
        report.append("")

        # Display each category
        categories = [
            ('CAUSALITY', collection.causality, "Direct cause-and-effect patterns"),
            ('TEMPORAL', collection.temporal, "Time-based transitions and flow"),
            ('UNIVERSAL', collection.universal, "Sweeping statements and absolutes"),
            ('SINGULAR', collection.singular, "Focus on individual elements"),
            ('MODAL', collection.modal, "Conditional and possibility expressions")
        ]

        for category_name, fragments, description in categories:
            if not fragments:
                continue

            report.append(f"{category_name} FRAGMENTS:")
            report.append("-" * 40)
            report.append(f"{description}")
            report.append("")

            # Sort by poetic score (best first)
            sorted_fragments = sorted(fragments, key=lambda f: f.poetic_score, reverse=True)

            for i, fragment in enumerate(sorted_fragments[:12], 1):  # Show top 12 per category
                tone_emoji = {
                    'dark': '🌑', 'light': '☀️', 'dynamic': '⚡',
                    'quiet': '🌙', 'mysterious': '🔮', 'neutral': '📝'
                }.get(fragment.emotional_tone, '📝')

                report.append(f"{i:2d}. \"{fragment.text}\"")
                report.append(f"     {tone_emoji} {fragment.emotional_tone} | "
                            f"⭐ {fragment.poetic_score:.2f} | "
                            f"📏 {fragment.word_count} words")
                report.append("")

        report.append("CREATIVE SUGGESTIONS:")
        report.append("-" * 30)
        report.append("• Use fragments as opening lines for new poems")
        report.append("• Combine fragments from different categories")
        report.append("• Build on the emotional tone of high-scoring fragments")
        report.append("• Adapt the sentence structure for your own ideas")

        return "\n".join(report)


# Alias for backward compatibility with CLI
CausalPoetryGenerator = ResonantFragmentMiner
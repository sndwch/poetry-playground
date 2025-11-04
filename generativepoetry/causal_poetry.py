"""
Resonant Fragment Miner

Discovers evocative sentence fragments from classic literature that could
serve as seeds for compression poetry. Mines multiple sentence patterns
for fragments with poetic weight and causal resonance.

Example fragments: "The door slammed", "When silence fell", "Every shadow whispered"
"""

import re
from dataclasses import dataclass, field
from typing import List

from .config import DocumentConfig, QualityConfig
from .document_library import get_diverse_gutenberg_documents
from .vocabulary import vocabulary
from .word_validator import WordValidator


@dataclass
class ResonantFragment:
    """A discovered fragment with poetic potential"""

    text: str  # The fragment text
    pattern_type: str  # Which pattern matched (causality, temporal, etc.)
    word_count: int  # Length in words
    source_preview: str  # Brief context from source
    emotional_tone: str  # Detected emotional quality
    poetic_score: float  # How evocative it seems (0-1)


@dataclass
class FragmentCollection:
    """Collection of fragments organized by pattern type"""

    causality: List[ResonantFragment] = field(default_factory=list)
    temporal: List[ResonantFragment] = field(default_factory=list)
    universal: List[ResonantFragment] = field(default_factory=list)
    singular: List[ResonantFragment] = field(default_factory=list)
    modal: List[ResonantFragment] = field(default_factory=list)

    def total_count(self) -> int:
        return (
            len(self.causality)
            + len(self.temporal)
            + len(self.universal)
            + len(self.singular)
            + len(self.modal)
        )

    def get_all_fragments(self) -> List[ResonantFragment]:
        return self.causality + self.temporal + self.universal + self.singular + self.modal


class ResonantFragmentMiner:
    """Mine evocative fragments from classic literature"""

    def __init__(self):
        self.word_validator = WordValidator()

        # Fragment patterns - more flexible to capture diverse literary language
        self.patterns = {
            "causality": [
                r"The [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "The door slammed."
                r"The [a-zA-Z]+ had [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "The light had faded."
                r"[A-Z][a-zA-Z]+ [a-zA-Z]+ed[^.!?]{0,40}[.!?]",  # "Something crashed."
                r"[A-Z][a-zA-Z]+ [a-zA-Z]+ [a-zA-Z]+[^.!?]{0,40}[.!?]",  # "Fire burns everything."
                r"It [a-zA-Z]+[^.!?]{0,40}[.!?]",  # "It shattered."
                r"She [a-zA-Z]+[^.!?]{0,40}[.!?]",  # "She whispered."
                r"He [a-zA-Z]+[^.!?]{0,40}[.!?]",  # "He vanished."
            ],
            "temporal": [
                r"When [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "When silence fell."
                r"As [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "As darkness crept."
                r"Until [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Until morning broke."
                r"Before [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Before dawn arrived."
                r"After [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "After thunder roared."
                r"Then [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Then everything changed."
                r"Now [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Now darkness falls."
                r"Soon [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Soon hope returned."
            ],
            "universal": [
                r"Every [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Every shadow whispered."
                r"All [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "All hope vanished."
                r"No [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "No sound escaped."
                r"Each [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Each breath echoed."
                r"Any [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Any movement ceased."
                r"None [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "None dared speak."
            ],
            "singular": [
                r"One [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "One candle burned."
                r"Only [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Only silence remained."
                r"Something [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Something stirred."
                r"Nothing [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Nothing moved."
                r"Someone [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Someone whispered."
                r"Anyone [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Anyone could see."
            ],
            "modal": [
                r"[A-Z][a-zA-Z]+ would [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Memory would return."
                r"[A-Z][a-zA-Z]+ could [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Fire could consume."
                r"[A-Z][a-zA-Z]+ might [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Wind might whisper."
                r"[A-Z][a-zA-Z]+ should [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Love should endure."
                r"[A-Z][a-zA-Z]+ may [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Hope may return."
                r"[A-Z][a-zA-Z]+ must [a-zA-Z]+[^.!?]{0,50}[.!?]",  # "Truth must prevail."
            ],
        }

        # Use shared vocabulary for emotional tone classification

    def mine_fragments(self, target_count: int = 50, num_texts: int = 5) -> FragmentCollection:
        """Mine resonant fragments using adaptive text retrieval to maintain quality"""
        print(f"🔍 Mining {target_count} resonant fragments with adaptive text retrieval...")

        collection = FragmentCollection()
        seen_fragments = set()  # Deduplication tracking
        initial_texts = num_texts

        # Start with initial batch
        print(f"  📚 Retrieving {initial_texts} diverse documents...")
        documents = get_diverse_gutenberg_documents(
            count=initial_texts, min_length=DocumentConfig.MIN_LENGTH_FRAGMENTS
        )

        if not documents:
            print("❌ Failed to retrieve any documents. Check internet connection.")
            return collection

        print(f"  ✓ Successfully retrieved {len(documents)} diverse documents")

        # Mine fragments from initial documents
        documents_processed = 0
        for text in documents:
            documents_processed += 1
            print(
                f"  📖 Mining from document {documents_processed}... (found {collection.total_count()}/{target_count})"
            )

            fragments_added = self._mine_fragments_from_single_text(
                text, collection, seen_fragments, target_count
            )
            print(f"    ✓ Added {fragments_added} high-quality fragments")

            # Check if we've reached our target
            if collection.total_count() >= target_count:
                break

        # Adaptive retrieval: keep getting more texts until quota is met
        adaptive_batch_size = 3  # Start with 3 additional docs

        while collection.total_count() < target_count:
            needed = target_count - collection.total_count()

            # Scale batch size based on how many we still need
            if needed > 50:
                adaptive_batch_size = 5  # Larger batches for big gaps
            elif needed > 20:
                adaptive_batch_size = 3  # Medium batches
            else:
                adaptive_batch_size = 2  # Smaller batches when close

            print(
                f"  📚 Need {needed} more fragments, retrieving {adaptive_batch_size} additional documents..."
            )

            additional_docs = get_diverse_gutenberg_documents(
                count=adaptive_batch_size, min_length=DocumentConfig.MIN_LENGTH_FRAGMENTS
            )

            if not additional_docs:
                print(
                    "  ⚠ Could not retrieve additional documents - network issue or exhausted sources"
                )
                break

            for text in additional_docs:
                documents_processed += 1
                print(
                    f"  📖 Mining from document {documents_processed}... (found {collection.total_count()}/{target_count})"
                )

                fragments_added = self._mine_fragments_from_single_text(
                    text, collection, seen_fragments, target_count
                )
                print(f"    ✓ Added {fragments_added} high-quality fragments")

                if collection.total_count() >= target_count:
                    break

        print(
            f"🎉 Successfully mined {collection.total_count()} unique high-quality fragments from {documents_processed} diverse texts!"
        )
        return collection

    def _mine_fragments_from_single_text(
        self, text: str, collection: FragmentCollection, seen_fragments: set, target_count: int
    ) -> int:
        """Mine fragments from a single text and add to collection"""
        added_count = 0

        try:
            # Mine fragments from this text
            fragments = self._extract_fragments_from_text(text)

            # Sort by quality and take best ones from this text
            fragments.sort(key=lambda f: f.poetic_score, reverse=True)

            for fragment in fragments:
                if collection.total_count() >= target_count:
                    break

                # Check for duplicates (normalize text)
                normalized_text = fragment.text.lower().strip()
                if normalized_text in seen_fragments:
                    continue

                # Additional quality check
                if self._is_high_quality_fragment(fragment):
                    seen_fragments.add(normalized_text)
                    self._add_fragment_to_collection(fragment, collection)
                    added_count += 1

        except Exception as e:
            print(f"    ⚠ Warning: Error processing text: {e}")

        return added_count

    def _extract_fragments_from_text(self, text: str) -> List[ResonantFragment]:
        """Extract resonant fragments from a single text"""
        fragments = []

        # Sample from different sections of the text for variety
        text_len = len(text)
        if text_len > 10000:
            # Take samples from beginning, middle, and end
            sections = [
                text[: text_len // 3],
                text[text_len // 3 : 2 * text_len // 3],
                text[2 * text_len // 3 :],
            ]
            sample_text = " ".join(sections)
        else:
            sample_text = text

        source_preview = sample_text[:100].replace("\n", " ").strip()

        # Clean the text - remove excessive whitespace and weird characters
        cleaned_text = re.sub(r"\s+", " ", sample_text)
        cleaned_text = re.sub(r'[^\w\s.!?,:;\'"-]', " ", cleaned_text)

        # Search for each pattern type
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_text, re.MULTILINE)

                for match in matches[:25]:  # Increased limit for better yield
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
                        poetic_score=self._calculate_poetic_score(fragment_text),
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
        if not text.strip().endswith((".", "!", "?")):
            return False

        # Avoid obvious non-literary content
        if any(char in text for char in ["(", ")", "[", "]"]) or any(
            word.isdigit() for word in words
        ):
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
        if tone != "neutral":
            score += 0.2

        # Bonus for shorter fragments (more compressed)
        if len(words) <= 6:
            score += 0.1

        return min(1.0, score)

    def _is_high_quality_fragment(self, fragment: ResonantFragment) -> bool:
        """Additional quality check for fragments with higher standards"""
        text = fragment.text
        words = text.split()

        # Minimum quality score threshold - maintain high standards
        if fragment.poetic_score < QualityConfig.FRAGMENT_QUALITY_THRESHOLD:
            return False

        # Reject overly generic fragments (relaxed)
        generic_patterns = [
            r"^(What|How|Where|When|Why)\s+(could|would|should|might)",  # "What could it mean?"
            r"^\w+\s+(said|asked|replied|answered)\s*(to|that)",  # Dialogue tags with continuation
            r"^(I|We|You)\s+(was|were|am|is|are)\s+\w+\.$",  # Simple self-referential statements
            r"^\w+\s+\w+\.$",  # Only two words total
        ]

        for pattern in generic_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False

        # Reject fragments with too many proper names (characters)
        proper_nouns = [
            word
            for word in words
            if word[0].isupper() and word.lower() not in ["the", "a", "an", "i"]
        ]
        if len(proper_nouns) > 2:  # Too character-specific
            return False

        # Prefer fragments with interesting imagery or action
        has_imagery = any(vocabulary.is_evocative_word(word.lower()) for word in words)
        has_action = any(word.endswith(("ed", "ing", "s")) for word in words if len(word) > 3)

        return has_imagery or has_action

    def _add_fragment_to_collection(
        self, fragment: ResonantFragment, collection: FragmentCollection
    ):
        """Add fragment to the appropriate category in collection"""
        if fragment.pattern_type == "causality":
            collection.causality.append(fragment)
        elif fragment.pattern_type == "temporal":
            collection.temporal.append(fragment)
        elif fragment.pattern_type == "universal":
            collection.universal.append(fragment)
        elif fragment.pattern_type == "singular":
            collection.singular.append(fragment)
        elif fragment.pattern_type == "modal":
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
            ("CAUSALITY", collection.causality, "Direct cause-and-effect patterns"),
            ("TEMPORAL", collection.temporal, "Time-based transitions and flow"),
            ("UNIVERSAL", collection.universal, "Sweeping statements and absolutes"),
            ("SINGULAR", collection.singular, "Focus on individual elements"),
            ("MODAL", collection.modal, "Conditional and possibility expressions"),
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

            # Show all fragments if total count > 50, otherwise limit to top 12 per category
            display_limit = (
                len(sorted_fragments)
                if collection.total_count() > 50
                else min(12, len(sorted_fragments))
            )
            for i, fragment in enumerate(sorted_fragments[:display_limit], 1):
                tone_emoji = {
                    "dark": "🌑",
                    "light": "☀️",
                    "dynamic": "⚡",
                    "quiet": "🌙",
                    "mysterious": "🔮",
                    "neutral": "📝",
                }.get(fragment.emotional_tone, "📝")

                report.append(f'{i:2d}. "{fragment.text}"')
                report.append(
                    f"     {tone_emoji} {fragment.emotional_tone} | "
                    f"⭐ {fragment.poetic_score:.2f} | "
                    f"📏 {fragment.word_count} words"
                )
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

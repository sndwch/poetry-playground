"""Generate personalized line seeds that match an author's stylistic fingerprint.

This module extends the standard LineSeedGenerator to create line seeds that not only
have high quality scores but also match the stylistic patterns found in a personal
corpus. The personalization works by:

1. **Hybrid Vocabulary Pool**: Blends words from the author's fingerprint with generic
   vocabulary, controlled by a strictness parameter (0.0 = fully generic, 1.0 = only
   fingerprint words).

2. **Style Matching**: Evaluates candidates across 4 dimensions:
   - Line length patterns (typical syllable counts)
   - POS sequence patterns (syntactic structures)
   - Concreteness ratio (abstract vs concrete language)
   - Phonetic patterns (alliteration, rhyme frequency)

3. **Dual Scoring**: Final ranking combines universal quality (70%) with style fit (30%),
   ensuring seeds are both craft-worthy and personally authentic.

Example usage:
    >>> from poetryplayground.personal_corpus_analyzer import PersonalCorpusAnalyzer
    >>> analyzer = PersonalCorpusAnalyzer()
    >>> fingerprint = analyzer.analyze_directory("~/my_poems")
    >>>
    >>> generator = PersonalizedLineSeedGenerator(fingerprint, strictness=0.7)
    >>> seeds = generator.generate_personalized_collection(count=20)
    >>>
    >>> for seed in seeds:
    ...     print(f"{seed.text} (quality={seed.quality_score:.2f}, style={seed.style_fit_score:.2f})")
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import spacy

from poetryplayground.corpus_analyzer import StyleFingerprint
from poetryplayground.lexicon import get_lexicon_data
from poetryplayground.line_seeds import (
    GenerationStrategy,
    LineSeed,
    LineSeedGenerator,
    SeedType,
)
from poetryplayground.quality_scorer import get_quality_scorer
from poetryplayground.semantic_geodesic import SemanticSpace


class PersonalizedLineSeedGenerator:
    """Generate line seeds that match a personal stylistic fingerprint.

    This generator creates line seeds using the same strategies as LineSeedGenerator,
    but biases word selection toward the author's vocabulary and evaluates candidates
    against style patterns extracted from their corpus.

    Attributes:
        fingerprint: The StyleFingerprint containing style patterns
        strictness: 0.0-1.0, how much to weight fingerprint vocabulary vs generic
        base_generator: Standard LineSeedGenerator for core functionality
        quality_scorer: Universal quality scorer instance
        semantic_space: Semantic space for word relationships
        hybrid_vocab: Combined vocabulary pool (fingerprint + generic)
    """

    def __init__(
        self,
        fingerprint: StyleFingerprint,
        strictness: float = 0.7,
        semantic_space: Optional[SemanticSpace] = None,
    ):
        """Initialize the personalized line seed generator.

        Args:
            fingerprint: StyleFingerprint from analyzing personal corpus
            strictness: 0.0-1.0, vocabulary bias (0=generic, 1=only fingerprint)
            semantic_space: Optional SemanticSpace instance (creates if None)

        Raises:
            ValueError: If strictness not in [0.0, 1.0]
        """
        if not 0.0 <= strictness <= 1.0:
            raise ValueError(f"strictness must be in [0.0, 1.0], got {strictness}")

        self.fingerprint = fingerprint
        self.strictness = strictness
        self.base_generator = LineSeedGenerator()
        self.quality_scorer = get_quality_scorer()
        self.semantic_space = semantic_space or SemanticSpace()

        # Load spaCy for POS tagging
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            # Fallback: use semantic_space.nlp if available
            self.nlp = getattr(semantic_space, "nlp", None)

        # Build fingerprint vocabulary mapping (word -> frequency)
        self.fingerprint_vocab = self._build_fingerprint_vocab()

        # Build hybrid vocabulary pool
        self.hybrid_vocab = self._build_hybrid_vocabulary_pool()

        # Build POS-to-words cache for efficient style-aware generation
        self.pos_word_cache = self._build_pos_word_cache()

    def generate_personalized_collection(
        self,
        count: int = 20,
        min_quality: float = 0.5,
        min_style_fit: float = 0.4,
    ) -> List[LineSeed]:
        """Generate a collection of personalized line seeds.

        Creates line seeds using PERSONALIZED vocabulary (not generic base generator),
        then scores each by both quality and style fit. Final ranking: 0.7 * quality + 0.3 * style_fit.

        Args:
            count: Number of seeds to generate
            min_quality: Minimum quality score threshold (0.0-1.0)
            min_style_fit: Minimum style fit threshold (0.0-1.0)

        Returns:
            List of LineSeed objects with style_fit_score and style_components populated,
            sorted by combined score (descending)
        """
        # Select seed words from fingerprint vocabulary (top frequent words)
        seed_words = list(self.fingerprint_vocab.keys())[:50]  # Top 50 most frequent

        # Generate larger set of candidates (3x desired count) using PERSONALIZED methods
        candidate_count = count * 3
        candidates = []
        seen_texts = set()  # Deduplication
        last_generators = []  # Track last N generator types for variety

        # Generator types with weights (reduce sonic_pattern frequency)
        generator_types = [
            (self.generate_fragment, 3),  # Weight 3
            (self.generate_image_seed, 3),  # Weight 3
            (self.generate_opening_line, 2),  # Weight 2
            (self.generate_pivot_line, 2),  # Weight 2
            (self.generate_sonic_pattern, 1),  # Weight 1 (reduce repetition)
            (self.generate_ending_approach, 2),  # Weight 2
        ]

        # Generate using overridden personalized methods (NOT base_generator!)
        attempts = 0
        max_attempts = candidate_count * 2  # Allow retries for deduplication
        while len(candidates) < candidate_count and attempts < max_attempts:
            attempts += 1

            # Weighted random choice
            generators, weights = zip(*generator_types)
            seed_type = random.choices(generators, weights=weights, k=1)[0]

            # Avoid using same generator 3 times in a row
            if len(last_generators) >= 3 and all(g == seed_type for g in last_generators[-3:]):
                continue

            try:
                candidate = seed_type(seed_words)

                # Deduplication: skip if we've seen this text
                if candidate.text.lower().strip() in seen_texts:
                    continue

                candidates.append(candidate)
                seen_texts.add(candidate.text.lower().strip())
                last_generators.append(seed_type)

                # Keep last_generators list bounded
                if len(last_generators) > 5:
                    last_generators.pop(0)

            except Exception:
                # Skip if generation fails
                continue

        # Score each candidate by quality and style fit
        scored_seeds = []
        for seed in candidates:
            # Calculate style fit
            style_fit, style_components = self._calculate_style_fit_score(seed.text)

            # Check thresholds
            if seed.quality_score < min_quality or style_fit < min_style_fit:
                continue

            # Calculate combined score: 70% quality + 30% style fit
            combined_score = 0.7 * seed.quality_score + 0.3 * style_fit

            # Update seed with style information
            seed.style_fit_score = style_fit
            seed.style_components = style_components

            scored_seeds.append((combined_score, seed))

        # Sort by combined score (descending)
        scored_seeds.sort(key=lambda x: x[0], reverse=True)

        # Return top N seeds
        return [seed for _, seed in scored_seeds[:count]]

    # -------------------------------------------------------------------------
    # Phase 2: Hybrid Vocabulary Pool & Biased Selection
    # -------------------------------------------------------------------------

    def _build_fingerprint_vocab(self) -> Dict[str, int]:
        """Build vocabulary mapping from fingerprint data.

        Returns:
            Dictionary mapping word -> frequency count in personal corpus
        """
        vocab = {}

        # Add most common words
        for word, count in self.fingerprint.vocabulary.most_common_words:
            vocab[word] = count

        # Add content words (may overlap, keep higher count)
        for word, count in self.fingerprint.vocabulary.most_common_content_words:
            if word not in vocab or count > vocab[word]:
                vocab[word] = count

        return vocab

    def _build_hybrid_vocabulary_pool(self) -> Set[str]:
        """Build vocabulary pool blending fingerprint and generic words.

        Creates a weighted combination:
        - strictness% from fingerprint vocabulary (most_common_words + content_words)
        - (1-strictness)% from semantic_space generic vocabulary

        Returns:
            Set of words to use for generation
        """
        target_size = 10000  # Reasonable vocabulary size

        # Sort fingerprint words by frequency
        fingerprint_words = sorted(
            self.fingerprint_vocab.keys(), key=lambda w: self.fingerprint_vocab[w], reverse=True
        )

        # Get generic words from semantic space
        generic_words = self.semantic_space.index_to_word

        # Calculate how many from each source
        n_fingerprint = int(target_size * self.strictness)

        # Combine (fingerprint words get priority)
        vocab = set()

        # Add fingerprint words (up to n_fingerprint)
        vocab.update(fingerprint_words[:n_fingerprint])

        # Add generic words that aren't already in vocab
        for word in generic_words:
            if word not in vocab:
                vocab.add(word)
                if len(vocab) >= target_size:
                    break

        return vocab

    def _build_pos_word_cache(self) -> Dict[str, List[str]]:
        """Build cache mapping POS tags to words from hybrid vocabulary.

        Uses lexicon's POS cache for efficiency instead of running spaCy on every word.

        Returns:
            Dictionary mapping POS tag (NOUN, VERB, ADJ, etc.) to list of words
        """
        lexicon = get_lexicon_data()
        pos_cache = {}

        # Build reverse mapping: POS -> [words]
        for word in self.hybrid_vocab:
            pos_tag = lexicon.pos_cache.get(word)
            if pos_tag:
                if pos_tag not in pos_cache:
                    pos_cache[pos_tag] = []
                pos_cache[pos_tag].append(word)

        return pos_cache

    def _select_biased_word(
        self,
        candidates: List[str],
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Select word from candidates, biased toward fingerprint vocabulary.

        Selection strategy:
        1. Filter to words in hybrid_vocab
        2. Score by: quality + style_bonus (if in fingerprint)
        3. Return highest scoring candidate

        Args:
            candidates: List of candidate words
            context: Optional context for semantic relevance

        Returns:
            Selected word, or None if no valid candidates
        """
        if not candidates:
            return None

        # Filter to words in hybrid vocabulary
        valid_candidates = [w for w in candidates if w.lower() in self.hybrid_vocab]

        if not valid_candidates:
            return None

        # Score each candidate
        scored = []
        for word in valid_candidates:
            # Get quality score
            quality = self.quality_scorer.score_word(word)

            # Add bonus if word is in personal fingerprint
            style_bonus = 0.0
            if word in self.fingerprint_vocab:
                # Higher frequency in personal corpus = higher bonus
                freq = self.fingerprint_vocab[word]
                # Normalize bonus to 0.0-0.3 range
                # Typical word counts are 1-100, so scale accordingly
                style_bonus = min(0.3, freq / 100.0)

            # Combined score: quality + style bonus
            total_score = quality.overall + style_bonus

            scored.append((word, total_score))

        # Return highest scoring candidate
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    # -------------------------------------------------------------------------
    # Personalized Word Selection Helpers (constrained to hybrid_vocab)
    # -------------------------------------------------------------------------

    def _get_personalized_similar_words(
        self, word: str, sample_size: int = 5, min_quality: float = 0.5
    ) -> List[str]:
        """Get similar meaning words constrained to hybrid vocabulary.

        Args:
            word: Source word
            sample_size: Number of words to return
            min_quality: Minimum quality threshold

        Returns:
            List of similar words from hybrid_vocab
        """
        from poetryplayground.lexigen import similar_meaning_words

        # Get generic similar words
        candidates = similar_meaning_words(
            word, sample_size=sample_size * 3, min_quality=min_quality
        )

        # Filter to hybrid vocab
        filtered = [w for w in candidates if w.lower() in self.hybrid_vocab]

        # Use biased selection to prefer fingerprint words
        result = []
        for _ in range(min(sample_size, len(filtered))):
            selected = self._select_biased_word(filtered)
            if selected and selected not in result:
                result.append(selected)
                filtered.remove(selected)

        return result

    def _get_personalized_phonetic_words(
        self, word: str, sample_size: int = 5, min_quality: float = 0.5
    ) -> List[str]:
        """Get phonetically related words constrained to hybrid vocabulary.

        Args:
            word: Source word
            sample_size: Number of words to return
            min_quality: Minimum quality threshold

        Returns:
            List of phonetically related words from hybrid_vocab
        """
        from poetryplayground.lexigen import phonetically_related_words

        # Get generic phonetic words
        candidates = phonetically_related_words(
            word, sample_size=sample_size * 3, min_quality=min_quality
        )

        # Filter to hybrid vocab
        filtered = [w for w in candidates if w.lower() in self.hybrid_vocab]

        # Use biased selection
        result = []
        for _ in range(min(sample_size, len(filtered))):
            selected = self._select_biased_word(filtered)
            if selected and selected not in result:
                result.append(selected)
                filtered.remove(selected)

        return result

    def _get_personalized_contextual_words(
        self, word: str, sample_size: int = 10, min_quality: float = 0.5
    ) -> List[str]:
        """Get contextually linked words constrained to hybrid vocabulary.

        Args:
            word: Source word
            sample_size: Number of words to return
            min_quality: Minimum quality threshold

        Returns:
            List of contextual words from hybrid_vocab
        """
        from poetryplayground.lexigen import contextually_linked_words

        # Get generic contextual words
        candidates = contextually_linked_words(
            word, sample_size=sample_size * 3, min_quality=min_quality
        )

        # Filter to hybrid vocab
        filtered = [w for w in candidates if w.lower() in self.hybrid_vocab]

        # Use biased selection
        result = []
        for _ in range(min(sample_size, len(filtered))):
            selected = self._select_biased_word(filtered)
            if selected and selected not in result:
                result.append(selected)
                filtered.remove(selected)

        return result

    def _get_personalized_words_by_pos(self, pos_tag: str, sample_size: int = 20) -> List[str]:
        """Get words of specific POS tag from hybrid vocabulary.

        Uses the pre-built POS cache for efficiency.

        Args:
            pos_tag: POS tag (NOUN, VERB, ADJ, etc.)
            sample_size: Number of words to return

        Returns:
            List of words with given POS tag from hybrid_vocab
        """
        # Use cached POS mapping - much faster than runtime spaCy processing
        candidates = self.pos_word_cache.get(pos_tag, [])

        if not candidates:
            return []

        # Return sample (with shuffling for variety)
        if len(candidates) <= sample_size:
            return candidates[:]

        # Shuffle and return sample
        shuffled = candidates[:]
        random.shuffle(shuffled)
        return shuffled[:sample_size]

    # -------------------------------------------------------------------------
    # Personalized Generation Methods (Override base generator)
    # -------------------------------------------------------------------------

    def generate_fragment(self, seed_words: List[str], position: str = "any") -> LineSeed:
        """Generate fragment using semantic clusters and fingerprint POS patterns.

        Builds from semantically related words instead of random POS matches.

        Args:
            seed_words: Words to base generation on
            position: Where in poem (unused currently)

        Returns:
            LineSeed with semantically coherent, style-matched fragment
        """
        # Use fingerprint's actual POS patterns (syntactic structure)
        if not self.fingerprint.vocabulary.dominant_pos_3grams:
            # Fallback if no POS data
            return self.base_generator.generate_fragment(seed_words, position)

        # Pick a random dominant POS pattern from fingerprint
        pos_pattern, _ = random.choice(self.fingerprint.vocabulary.dominant_pos_3grams)

        # Pick a base word from fingerprint to build semantic cluster around
        if not self.fingerprint_vocab:
            return self.base_generator.generate_fragment(seed_words, position)

        base_word = random.choice(list(self.fingerprint_vocab.keys()))

        # Get semantically related words (this creates coherence!)
        related_words = self._get_personalized_similar_words(base_word, sample_size=15)
        related_words.append(base_word)  # Include base word itself

        if not related_words:
            return self.base_generator.generate_fragment(seed_words, position)

        # Build fragment using ONLY semantically related words that match POS pattern
        lexicon = get_lexicon_data()
        fragment_words = []

        for pos_tag in pos_pattern:
            # Find a related word that matches the required POS tag
            selected_word = None
            for word in related_words:
                if lexicon.pos_cache.get(word) == pos_tag:
                    selected_word = word
                    related_words.remove(word)  # Don't reuse same word
                    break

            if selected_word:
                fragment_words.append(selected_word)

        if not fragment_words:
            # Fallback if generation failed
            return self.base_generator.generate_fragment(seed_words, position)

        fragment = " ".join(fragment_words)
        quality = self.base_generator._evaluate_quality(fragment)

        return LineSeed(
            text=fragment,
            seed_type=SeedType.FRAGMENT,
            strategy=None,
            momentum=random.uniform(0.4, 0.7),
            openness=random.uniform(0.7, 1.0),
            quality_score=quality,
        )

    def generate_image_seed(self, seed_words: List[str]) -> LineSeed:
        """Generate image using semantic clusters and fingerprint patterns.

        Builds from semantically related concrete words for coherent imagery.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with semantically coherent, concrete image
        """
        # Pick a concrete base word from fingerprint to build semantic cluster around
        concrete_base_words = [
            w for w in self.fingerprint_vocab if self.quality_scorer.get_concreteness(w) > 0.6
        ]

        if not concrete_base_words:
            concrete_base_words = list(self.fingerprint_vocab.keys())[:20]

        if not concrete_base_words:
            return self.base_generator.generate_image_seed(seed_words)

        base_word = random.choice(concrete_base_words)

        # Get semantically related concrete words
        related_words = self._get_personalized_contextual_words(base_word, sample_size=15)
        related_words.append(base_word)

        # Filter to concrete words only
        related_words = [w for w in related_words if self.quality_scorer.get_concreteness(w) > 0.5]

        if not related_words:
            return self.base_generator.generate_image_seed(seed_words)

        # Use fingerprint POS pattern if available
        if self.fingerprint.vocabulary.dominant_pos_3grams:
            image_patterns = [
                p
                for p, _ in self.fingerprint.vocabulary.dominant_pos_3grams
                if "NOUN" in p or "ADJ" in p
            ]
            pos_pattern = random.choice(image_patterns) if image_patterns else ("DET", "NOUN")
        else:
            pos_pattern = ("DET", "ADJ", "NOUN")

        # Build image using ONLY semantically related concrete words
        lexicon = get_lexicon_data()
        image_words = []

        for pos_tag in pos_pattern:
            # Find a related word that matches the required POS tag
            selected_word = None
            for word in related_words:
                if lexicon.pos_cache.get(word) == pos_tag:
                    selected_word = word
                    related_words.remove(word)  # Don't reuse
                    break

            if selected_word:
                image_words.append(selected_word)

        if not image_words:
            return self.base_generator.generate_image_seed(seed_words)

        image = " ".join(image_words)
        quality = self.base_generator._evaluate_quality(image)

        return LineSeed(
            text=image,
            seed_type=SeedType.IMAGE,
            strategy=GenerationStrategy.SYNESTHESIA if random.random() > 0.5 else None,
            momentum=random.uniform(0.3, 0.6),
            openness=random.uniform(0.5, 0.8),
            quality_score=quality,
        )

    def generate_opening_line(self, seed_words: List[str], mood: Optional[str] = None) -> LineSeed:
        """Generate opening targeting fingerprint syllable counts and POS patterns.

        Args:
            seed_words: Words to base generation on
            mood: Optional mood (unused currently)

        Returns:
            LineSeed with style-matched opening
        """
        strategy = random.choice(list(GenerationStrategy))

        # Sample target syllable count from fingerprint distribution
        line_length_dist = self.fingerprint.metrics.line_length_distribution
        if line_length_dist:
            # Weighted random choice from distribution
            syllable_counts = list(line_length_dist.keys())
            weights = list(line_length_dist.values())
            target_syllables = random.choices(syllable_counts, weights=weights, k=1)[0]
        else:
            target_syllables = 7  # Default

        # Choose POS pattern (prefer 4-grams for longer lines, 3-grams for shorter)
        patterns_available = []
        if self.fingerprint.vocabulary.dominant_pos_4grams and target_syllables >= 8:
            patterns_available = self.fingerprint.vocabulary.dominant_pos_4grams
        if self.fingerprint.vocabulary.dominant_pos_3grams:
            patterns_available = self.fingerprint.vocabulary.dominant_pos_3grams

        if not patterns_available:
            return self.base_generator.generate_opening_line(seed_words, mood)

        # Try to build a line that approximately matches target syllables
        lexicon = get_lexicon_data()
        for _ in range(5):  # Try up to 5 times
            pos_pattern, _ = random.choice(patterns_available)

            # Build line slot-by-slot
            line_words = []
            total_syllables = 0

            for pos_tag in pos_pattern:
                candidates = self._get_personalized_words_by_pos(pos_tag, sample_size=50)
                if not candidates:
                    continue

                selected_word = self._select_biased_word(candidates)
                if not selected_word:
                    continue

                # Count syllables
                word_syllables = lexicon.syllable_cache.get(
                    selected_word, len(selected_word) // 3 + 1
                )

                # Stop if we'd overshoot the target significantly
                if total_syllables + word_syllables > target_syllables + 2:
                    break

                line_words.append(selected_word)
                total_syllables += word_syllables

            # Check if we got close enough to target
            if line_words and abs(total_syllables - target_syllables) <= 3:
                line = " ".join(line_words)
                quality = self.base_generator._evaluate_quality(line)

                return LineSeed(
                    text=line,
                    seed_type=SeedType.OPENING,
                    strategy=strategy,
                    momentum=random.uniform(0.7, 0.95),
                    openness=random.uniform(0.6, 0.9),
                    quality_score=quality,
                )

        # Fallback if we couldn't match target
        return self.base_generator.generate_opening_line(seed_words, mood)

    def generate_pivot_line(self, seed_words: List[str]) -> LineSeed:
        """Generate pivot using semantic clusters and fingerprint patterns.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with semantically coherent pivot
        """
        # Choose a pivot word
        pivot_words = ["But", "Or", "Until", "When", "If", "Though"]
        pivot_word = random.choice(pivot_words)

        # Pick a base word and get semantic cluster
        if not self.fingerprint_vocab:
            fingerprint_words = ["threshold", "silence"]
        else:
            fingerprint_words = list(self.fingerprint_vocab.keys())[:20]

        base_word = random.choice(fingerprint_words)
        related_words = self._get_personalized_similar_words(base_word, sample_size=10)
        related_words.append(base_word)

        if not related_words:
            line = f"{pivot_word} {base_word}"
        elif self.fingerprint.vocabulary.dominant_pos_3grams:
            # Build pivot using semantic cluster
            pos_pattern, _ = random.choice(self.fingerprint.vocabulary.dominant_pos_3grams)
            lexicon = get_lexicon_data()
            line_words = [pivot_word]

            for pos_tag in pos_pattern:
                selected_word = None
                for word in related_words:
                    if lexicon.pos_cache.get(word) == pos_tag:
                        selected_word = word
                        related_words.remove(word)
                        break

                if selected_word:
                    line_words.append(selected_word)

            line = " ".join(line_words) if len(line_words) > 1 else f"{pivot_word} {base_word}"
        else:
            line = f"{pivot_word} {random.choice(related_words)}"

        quality = self.base_generator._evaluate_quality(line)

        return LineSeed(
            text=line,
            seed_type=SeedType.PIVOT,
            strategy=GenerationStrategy.QUESTION_IMPLIED,
            momentum=random.uniform(0.6, 0.9),
            openness=random.uniform(0.8, 1.0),
            quality_score=quality,
        )

    def generate_sonic_pattern(self, seed_words: List[str]) -> LineSeed:
        """Generate sonic pattern using personalized vocabulary.

        Args:
            seed_words: Words to base generation on

        Returns:
            LineSeed with personalized sonic pattern
        """
        # Get phonetically related words from personalized vocab
        sound_words = []
        for word in seed_words[:2]:
            sound_words.extend(self._get_personalized_phonetic_words(word, sample_size=3))

        if len(sound_words) >= 3:
            pattern = f"{sound_words[0]}, {sound_words[1]}, {sound_words[2]}"
        elif sound_words:
            pattern = f"{sound_words[0]}-{seed_words[0]}"
        else:
            pattern = f"{seed_words[0]}-{seed_words[-1]}"

        quality = self.base_generator._evaluate_quality(pattern)

        return LineSeed(
            text=pattern,
            seed_type=SeedType.SONIC,
            strategy=GenerationStrategy.RHYTHMIC_BREAK,
            momentum=random.uniform(0.5, 0.8),
            openness=random.uniform(0.4, 0.7),
            quality_score=quality,
        )

    def generate_ending_approach(
        self, seed_words: List[str], opening_line: Optional[str] = None
    ) -> LineSeed:
        """Generate ending using fingerprint vocabulary (prefer concrete nouns).

        Args:
            seed_words: Words to base generation on
            opening_line: Optional opening to reference

        Returns:
            LineSeed with style-matched ending (short, concrete)
        """
        # Get concrete nouns from fingerprint for endings
        concrete_nouns = [
            w for w in self.fingerprint_vocab if self.quality_scorer.get_concreteness(w) > 0.6
        ]

        if not concrete_nouns:
            concrete_nouns = list(self.fingerprint_vocab.keys())[:10]

        # Ending patterns (short, direct)
        patterns = [
            lambda: f"The {random.choice(concrete_nouns)}.",
            lambda: f"Just {random.choice(concrete_nouns)}.",
            lambda: f"All {random.choice(concrete_nouns)}, no {random.choice(concrete_nouns)}",
        ]

        ending = random.choice(patterns)()
        quality = self.base_generator._evaluate_quality(ending)

        return LineSeed(
            text=ending,
            seed_type=SeedType.CLOSING,
            strategy=GenerationStrategy.PERSPECTIVE_BLUR,
            momentum=random.uniform(0.1, 0.3),
            openness=random.uniform(0.2, 0.5),
            quality_score=quality,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Style Matching Components (4 dimensions, 25% each)
    # -------------------------------------------------------------------------

    def _calculate_line_length_fit(self, text: str) -> float:
        """Calculate how well text matches fingerprint line length patterns.

        Compares syllable count against fingerprint.line_metrics distribution:
        - Perfect match to peak: 1.0
        - Within 1 syllable: 0.7-0.9
        - Within 2 syllables: 0.5-0.7
        - Further: decays exponentially

        Args:
            text: The line text to evaluate

        Returns:
            Score 0.0-1.0 (1.0 = perfect fit to typical line length)
        """
        lexicon = get_lexicon_data()

        # Count syllables in text
        words = text.lower().split()
        total_syllables = 0
        for word in words:
            syllable_count = lexicon.syllable_cache.get(word, len(word) // 3 + 1)
            total_syllables += syllable_count

        # Get line length distribution from fingerprint
        line_length_dist = self.fingerprint.metrics.line_length_distribution

        if not line_length_dist:
            return 0.5  # No data, assume neutral

        # Find most common line length (peak)
        peak_length = max(line_length_dist, key=line_length_dist.get)

        # Calculate distance from peak
        distance = abs(total_syllables - peak_length)

        # Score based on distance
        distance_scores = {
            0: 1.0,
            1: 0.8,
            2: 0.6,
            3: 0.4,
            4: 0.2,
        }
        return distance_scores.get(distance, 0.1)

    def _calculate_pos_pattern_fit(self, text: str) -> float:
        """Calculate how well text matches fingerprint POS patterns.

        Uses Jaccard similarity between candidate line's POS 3-grams and
        fingerprint's dominant POS 3-grams:
        - Extract all 3-grams from candidate line
        - Compare with top 15 dominant 3-grams from fingerprint
        - Score = intersection / union (Jaccard similarity)

        Args:
            text: The line text to evaluate

        Returns:
            Score 0.0-1.0 (1.0 = perfect match to dominant patterns)
        """
        if not self.nlp:
            return 0.5  # Can't analyze without NLP model

        # Get POS tags
        doc = self.nlp(text)
        pos_tags = [token.pos_ for token in doc]

        if len(pos_tags) < 3:
            return 0.5

        # Extract 3-grams from candidate line
        candidate_3grams = set()
        for i in range(len(pos_tags) - 2):
            trigram = (pos_tags[i], pos_tags[i + 1], pos_tags[i + 2])
            candidate_3grams.add(trigram)

        # Get fingerprint dominant 3-grams
        dominant_3grams = self.fingerprint.vocabulary.dominant_pos_3grams

        if not dominant_3grams:
            return 0.5  # No data

        # Convert to set of tuples (ignore counts)
        fingerprint_3grams = {ngram for ngram, _ in dominant_3grams}

        # Calculate Jaccard similarity: intersection / union
        if not candidate_3grams or not fingerprint_3grams:
            return 0.5

        intersection = len(candidate_3grams & fingerprint_3grams)
        union = len(candidate_3grams | fingerprint_3grams)

        return intersection / union if union > 0 else 0.5

    def _calculate_concreteness_fit(self, text: str) -> float:
        """Calculate how well text matches fingerprint concreteness.

        Compares candidate line's average concreteness against fingerprint's
        average concreteness:
        - Perfect match: 1.0
        - Within 0.1 deviation: 0.8+
        - Within 0.2 deviation: 0.6+
        - Further: decays

        Args:
            text: The line text to evaluate

        Returns:
            Score 0.0-1.0 (1.0 = matches author's concreteness level)
        """
        # Get fingerprint target concreteness
        target_concreteness = self.fingerprint.themes.avg_concreteness

        if target_concreteness == 0.0:
            return 0.5  # No data

        # Calculate concreteness for this text
        words = text.lower().split()
        concreteness_scores = []

        for word in words:
            if word in self.quality_scorer.concreteness_cache:
                concreteness_scores.append(self.quality_scorer.concreteness_cache[word])

        if not concreteness_scores:
            return 0.5  # No data for this text

        candidate_concreteness = sum(concreteness_scores) / len(concreteness_scores)

        # Calculate absolute deviation from target
        deviation = abs(candidate_concreteness - target_concreteness)

        # Score based on deviation (closer = better)
        if deviation < 0.05:
            return 1.0
        elif deviation < 0.1:
            return 0.9
        elif deviation < 0.15:
            return 0.7
        elif deviation < 0.2:
            return 0.6
        elif deviation < 0.3:
            return 0.4
        else:
            return 0.2

    def _calculate_phonetic_fit(self, text: str) -> float:
        """Calculate how well text matches fingerprint phonetic patterns.

        Evaluates:
        - Alliteration: consecutive words starting with same sound
        - Internal rhyme: words sharing similar endings

        Note: Since fingerprint doesn't currently track phonetic patterns,
        this returns a simple heuristic score based on presence of
        sound patterns (higher = more alliteration/rhyme).

        Args:
            text: The line text to evaluate

        Returns:
            Score 0.0-1.0 (0.5 = baseline, higher = more phonetic patterns)
        """
        words = text.lower().split()

        if len(words) < 2:
            return 0.5

        # Count alliteration (consecutive words starting with same letter)
        alliteration_count = 0
        for i in range(len(words) - 1):
            if words[i] and words[i + 1] and words[i][0] == words[i + 1][0]:
                alliteration_count += 1

        # Count potential rhymes (words ending with same 2+ characters)
        rhyme_count = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if len(words[i]) >= 3 and len(words[j]) >= 3 and words[i][-2:] == words[j][-2:]:
                    rhyme_count += 1

        # Normalize: baseline is 0.5, increase for patterns
        phonetic_density = (alliteration_count + rhyme_count) / len(words)

        # Scale to 0.3-0.7 range (0.5 baseline)
        return 0.5 + (phonetic_density * 0.2)

    def _calculate_style_fit_score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Calculate overall style fit by combining 4 components.

        Final score: average of 4 components (equal weight, 25% each):
        - line_length_fit
        - pos_pattern_fit
        - concreteness_fit
        - phonetic_fit

        Args:
            text: The line text to evaluate

        Returns:
            Tuple of (overall_score, component_breakdown_dict)
        """
        components = {
            "line_length": self._calculate_line_length_fit(text),
            "pos_pattern": self._calculate_pos_pattern_fit(text),
            "concreteness": self._calculate_concreteness_fit(text),
            "phonetic": self._calculate_phonetic_fit(text),
        }

        # Average all components (equal weight, 25% each)
        overall = sum(components.values()) / len(components)

        return overall, components

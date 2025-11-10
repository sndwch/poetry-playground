"""Template extraction from existing poems.

This module analyzes existing poems and extracts structural templates
that can be used to generate similar poems.

Example:
    >>> from poetryplayground.template_extractor import TemplateExtractor
    >>> extractor = TemplateExtractor()
    >>> poem = '''
    ... The old pond—
    ... A frog jumps in,
    ... Water's sound!
    ... '''
    >>> template = extractor.extract_template(poem, title="Basho Haiku")
    >>> print(template.syllable_pattern)
    [4, 5, 3]
"""

from collections import Counter
from typing import List, Optional

import spacy

from poetryplayground.core.quality_scorer import (
    EmotionalTone,
    FormalityLevel,
    get_quality_scorer,
)
from poetryplayground.core.vocabulary import vocabulary
from poetryplayground.forms import count_syllables
from poetryplayground.logger import logger
from poetryplayground.poem_template import (
    LineTemplate,
    LineType,
    PoemTemplate,
)
from poetryplayground.setup_models import lazy_ensure_spacy_model


class TemplateExtractor:
    """Extract structural templates from existing poems.

    Analyzes poems to extract:
    - Line count and syllable patterns
    - POS (part-of-speech) patterns
    - Semantic domains
    - Metaphor types
    - Emotional tone and formality
    - Line types (opening, pivot, closing, etc.)

    Attributes:
        nlp: spaCy language model for linguistic analysis
        quality_scorer: Quality scorer for concreteness and tone analysis
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the template extractor.

        Args:
            spacy_model: spaCy model to use for analysis (default: en_core_web_sm)
        """
        # Load spaCy model
        lazy_ensure_spacy_model(spacy_model, "English language model")
        self.nlp = spacy.load(spacy_model)

        # Get quality scorer
        self.quality_scorer = get_quality_scorer()

        logger.info(f"TemplateExtractor initialized with {spacy_model}")

    def extract_template(
        self,
        poem_text: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        source: Optional[str] = None,
    ) -> PoemTemplate:
        """Extract a template from a poem.

        Args:
            poem_text: The poem text to analyze
            title: Optional title for the template
            author: Optional author name
            source: Optional source description

        Returns:
            PoemTemplate extracted from the poem

        Example:
            >>> extractor = TemplateExtractor()
            >>> poem = "The old pond\\nA frog jumps in\\nWater's sound"
            >>> template = extractor.extract_template(poem, title="Haiku")
        """
        # Clean and split into lines
        lines = self._clean_and_split_lines(poem_text)

        if not lines:
            raise ValueError("Poem text contains no valid lines")

        # Analyze each line
        line_templates = []
        syllable_pattern = []
        all_words = []

        for i, line in enumerate(lines):
            line_template = self._analyze_line(line, i, len(lines))
            line_templates.append(line_template)
            syllable_pattern.append(line_template.syllable_count)
            all_words.extend(line.split())

        # Extract semantic domains
        semantic_domains = self._extract_semantic_domains(all_words)

        # Detect metaphor types
        metaphor_types = self._detect_metaphor_types(lines)

        # Analyze emotional tone
        emotional_tone = self._analyze_emotional_tone(all_words)

        # Analyze formality level
        formality_level = self._analyze_formality(all_words, lines)

        # Calculate overall concreteness
        concreteness_ratio = self._calculate_concreteness_ratio(all_words)

        # Determine minimum quality score from words
        min_quality_score = self._estimate_quality_threshold(all_words)

        # Create template
        template = PoemTemplate(
            title=title or "Extracted Template",
            source=source or "extracted",
            author=author or "unknown",
            lines=len(lines),
            line_templates=line_templates,
            syllable_pattern=syllable_pattern,
            semantic_domains=semantic_domains,
            metaphor_types=metaphor_types,
            emotional_tone=emotional_tone,
            formality_level=formality_level,
            concreteness_ratio=concreteness_ratio,
            min_quality_score=min_quality_score,
            notes=f"Extracted from {len(lines)}-line poem",
        )

        logger.info(f"Extracted template '{template.title}' with {len(lines)} lines")
        return template

    def _clean_and_split_lines(self, text: str) -> List[str]:
        """Clean and split poem text into lines.

        Args:
            text: Raw poem text

        Returns:
            List of cleaned, non-empty lines
        """
        # Split by newlines
        lines = text.strip().split("\n")

        # Clean each line (strip whitespace, remove empty lines)
        cleaned = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(line)

        return cleaned

    def _analyze_line(self, line: str, line_index: int, total_lines: int) -> LineTemplate:
        """Analyze a single line and create a LineTemplate.

        Args:
            line: The line text
            line_index: Zero-based index of this line
            total_lines: Total number of lines in poem

        Returns:
            LineTemplate for this line
        """
        # Count syllables
        syllable_count = count_syllables(line)

        # Get POS pattern using spaCy
        doc = self.nlp(line)
        pos_pattern = [token.pos_ for token in doc if not token.is_punct]

        # Determine line type based on position
        line_type = self._classify_line_type(line, line_index, total_lines)

        # Detect metaphor in this line
        metaphor_type = self._detect_line_metaphor_type(line)

        # Extract semantic domain for this line
        words = [token.text.lower() for token in doc if token.is_alpha]
        semantic_domain = self._get_primary_semantic_domain(words)

        # Calculate concreteness for this line
        concreteness_target = self._calculate_line_concreteness(words)

        # Estimate quality threshold for this line
        min_quality_score = self._estimate_line_quality(words)

        return LineTemplate(
            syllable_count=syllable_count,
            pos_pattern=pos_pattern,
            line_type=line_type,
            metaphor_type=metaphor_type,
            semantic_domain=semantic_domain,
            concreteness_target=concreteness_target,
            min_quality_score=min_quality_score,
        )

    def _classify_line_type(self, line: str, line_index: int, total_lines: int) -> LineType:
        """Classify the functional type of a line.

        Args:
            line: The line text
            line_index: Zero-based index of this line
            total_lines: Total number of lines

        Returns:
            LineType classification
        """
        # First line is typically opening
        if line_index == 0:
            return LineType.OPENING

        # Last line is typically closing
        if line_index == total_lines - 1:
            return LineType.CLOSING

        # Middle line in 3-line poem is often a pivot
        if total_lines == 3 and line_index == 1:
            return LineType.PIVOT

        # For longer poems, check for transition markers
        if any(marker in line.lower() for marker in ["but", "yet", "however", "though", "still"]):
            return LineType.TRANSITION

        # Check for emotional markers
        emotional_words = ["heart", "love", "fear", "joy", "pain", "sorrow", "hope", "dream"]
        if any(word in line.lower() for word in emotional_words):
            return LineType.EMOTIONAL

        # Check for sound patterns (alliteration, repetition)
        words = line.lower().split()
        if len(words) >= 2:
            first_letters = [w[0] for w in words if w]
            if len(set(first_letters)) < len(first_letters) * 0.6:  # High repetition
                return LineType.SONIC

        # Default to image
        return LineType.IMAGE

    def _detect_line_metaphor_type(self, line: str) -> Optional[str]:
        """Detect metaphor type in a line.

        Args:
            line: The line text

        Returns:
            Metaphor type string, or None if no clear metaphor detected
        """
        line_lower = line.lower()

        # Simile patterns
        if " like " in line_lower or " as " in line_lower:
            return "simile"

        # Direct metaphor patterns ("X is Y")
        if " is " in line_lower or " are " in line_lower:
            return "direct"

        # Possessive patterns ("X of Y", "Y's X")
        if " of " in line_lower or "'s " in line_lower:
            return "possessive"

        # Appositive patterns (comma-separated)
        if "," in line and not line.endswith(","):
            return "appositive"

        # Compound patterns (hyphenated words)
        if "-" in line_lower:
            return "compound"

        return None

    def _extract_semantic_domains(self, words: List[str]) -> List[str]:
        """Extract semantic domains from words.

        Args:
            words: List of words from the poem

        Returns:
            List of semantic domain names
        """
        domain_counts = Counter()

        # Clean words
        words_clean = [w.lower() for w in words if w.isalpha()]

        # Check each word against vocabulary domains
        for domain, domain_words in vocabulary.concept_domains.items():
            domain_words_set = {w.lower() for w in domain_words}
            matches = sum(1 for word in words_clean if word in domain_words_set)
            if matches > 0:
                domain_counts[domain] = matches

        # Return top domains (at least 1 match)
        if not domain_counts:
            return ["general"]

        # Return domains with most matches (top 3)
        top_domains = [domain for domain, _ in domain_counts.most_common(3)]
        return top_domains

    def _get_primary_semantic_domain(self, words: List[str]) -> Optional[str]:
        """Get primary semantic domain for a line.

        Args:
            words: List of words from the line

        Returns:
            Primary domain name, or None
        """
        domains = self._extract_semantic_domains(words)
        return domains[0] if domains else None

    def _detect_metaphor_types(self, lines: List[str]) -> List[str]:
        """Detect metaphor types across all lines.

        Args:
            lines: List of poem lines

        Returns:
            List of unique metaphor types found
        """
        metaphor_types = set()

        for line in lines:
            metaphor_type = self._detect_line_metaphor_type(line)
            if metaphor_type:
                metaphor_types.add(metaphor_type)

        return sorted(metaphor_types)

    def _analyze_emotional_tone(self, words: List[str]) -> EmotionalTone:
        """Analyze emotional tone of the poem.

        Args:
            words: List of words from the poem

        Returns:
            EmotionalTone classification
        """
        # Define word sets for different tones
        dark_words = {
            "death",
            "dark",
            "night",
            "shadow",
            "fear",
            "pain",
            "sorrow",
            "grief",
            "black",
            "void",
            "empty",
            "cold",
            "alone",
            "lost",
        }

        light_words = {
            "light",
            "bright",
            "joy",
            "hope",
            "love",
            "sun",
            "dawn",
            "smile",
            "laugh",
            "warm",
            "happy",
            "peace",
            "dream",
            "bloom",
        }

        # Count matches
        words_clean = {w.lower() for w in words if w.isalpha()}
        dark_count = len(words_clean & dark_words)
        light_count = len(words_clean & light_words)

        # Classify
        if dark_count > light_count * 1.5:
            return EmotionalTone.DARK
        elif light_count > dark_count * 1.5:
            return EmotionalTone.LIGHT
        elif dark_count > 0 and light_count > 0:
            return EmotionalTone.MIXED
        else:
            return EmotionalTone.NEUTRAL

    def _analyze_formality(self, words: List[str], lines: List[str]) -> FormalityLevel:
        """Analyze formality level of the poem.

        Args:
            words: List of words from the poem
            lines: List of lines for structural analysis

        Returns:
            FormalityLevel classification
        """
        # Check for archaic words
        archaic_words = {
            "thee",
            "thou",
            "thy",
            "thine",
            "ye",
            "hath",
            "doth",
            "art",
            "ere",
            "midst",
            "nigh",
            "'tis",
            "whence",
        }

        # Check for casual markers
        casual_markers = {
            "yeah",
            "gonna",
            "wanna",
            "gotta",
            "kinda",
            "sorta",
            "ok",
            "okay",
            "hey",
            "yep",
            "nope",
        }

        words_clean = {w.lower() for w in words}

        # Check for archaic
        if words_clean & archaic_words:
            return FormalityLevel.ARCHAIC

        # Check for casual
        if words_clean & casual_markers:
            return FormalityLevel.CASUAL

        # Check average word length (longer = more formal)
        avg_word_length = sum(len(w) for w in words if w.isalpha()) / max(len(words), 1)

        if avg_word_length > 6:
            return FormalityLevel.FORMAL
        else:
            return FormalityLevel.CONVERSATIONAL

    def _calculate_concreteness_ratio(self, words: List[str]) -> float:
        """Calculate overall concreteness ratio.

        Args:
            words: List of words from the poem

        Returns:
            Concreteness ratio (0=abstract, 1=concrete)
        """
        words_clean = [w.lower() for w in words if w.isalpha() and len(w) > 2]

        if not words_clean:
            return 0.5

        concreteness_scores = [self.quality_scorer.get_concreteness(word) for word in words_clean]

        return sum(concreteness_scores) / len(concreteness_scores)

    def _calculate_line_concreteness(self, words: List[str]) -> float:
        """Calculate concreteness for a single line.

        Args:
            words: List of words from the line

        Returns:
            Concreteness ratio (0=abstract, 1=concrete)
        """
        return self._calculate_concreteness_ratio(words)

    def _estimate_quality_threshold(self, words: List[str]) -> float:
        """Estimate minimum quality score threshold from words.

        Args:
            words: List of words from the poem

        Returns:
            Estimated quality threshold (0.0-1.0)
        """
        from poetryplayground.core.quality_scorer import GenerationContext

        words_clean = [w.lower() for w in words if w.isalpha() and len(w) > 2]

        if not words_clean:
            return 0.6

        # Score each word
        context = GenerationContext()
        scores = []

        for word in words_clean[:20]:  # Limit to first 20 words for performance
            try:
                quality = self.quality_scorer.score_word(word, context)
                scores.append(quality.overall)
            except Exception:
                continue

        if not scores:
            return 0.6

        # Use 25th percentile as threshold (lower bound of quality)
        sorted_scores = sorted(scores)
        threshold_index = max(0, len(sorted_scores) // 4)
        return sorted_scores[threshold_index]

    def _estimate_line_quality(self, words: List[str]) -> float:
        """Estimate quality threshold for a single line.

        Args:
            words: List of words from the line

        Returns:
            Estimated quality threshold (0.0-1.0)
        """
        return self._estimate_quality_threshold(words)

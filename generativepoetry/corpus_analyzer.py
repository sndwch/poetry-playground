"""
Personal Corpus Analyzer for Generative Poetry

Analyzes a personal collection of poetry to identify patterns, style fingerprints,
and vocabulary characteristics. Provides insights for maintaining authentic voice
while exploring creative expansions.
"""

import os
import re
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from pathlib import Path

import nltk
import spacy
from wordfreq import word_frequency
from datamuse import datamuse

from .word_validator import WordValidator
from .lexigen import similar_meaning_words, contextually_linked_words, similar_sounding_words


@dataclass
class PoetryMetrics:
    """Container for various poetry analysis metrics"""
    total_poems: int = 0
    total_lines: int = 0
    total_words: int = 0
    avg_lines_per_poem: float = 0.0
    avg_words_per_line: float = 0.0
    avg_syllables_per_line: float = 0.0
    vocabulary_size: int = 0
    vocabulary_richness: float = 0.0  # Type-token ratio

    # Line structure patterns
    line_length_distribution: Dict[int, int] = field(default_factory=dict)
    stanza_patterns: List[int] = field(default_factory=list)

    # Stylistic elements
    punctuation_frequency: Dict[str, int] = field(default_factory=dict)
    capitalization_patterns: Dict[str, int] = field(default_factory=dict)
    enjambment_frequency: float = 0.0


@dataclass
class VocabularyProfile:
    """Detailed vocabulary analysis"""
    most_common_words: List[Tuple[str, int]] = field(default_factory=list)
    most_common_content_words: List[Tuple[str, int]] = field(default_factory=list)
    signature_words: List[Tuple[str, float]] = field(default_factory=list)  # High frequency relative to general usage
    rare_words: List[Tuple[str, int]] = field(default_factory=list)

    # Part-of-speech preferences
    pos_distribution: Dict[str, int] = field(default_factory=dict)
    preferred_adjectives: List[Tuple[str, int]] = field(default_factory=list)
    preferred_verbs: List[Tuple[str, int]] = field(default_factory=list)
    preferred_nouns: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class ThematicProfile:
    """Thematic and semantic analysis"""
    semantic_clusters: List[Tuple[str, List[str]]] = field(default_factory=list)
    metaphor_patterns: List[str] = field(default_factory=list)
    emotional_register: Dict[str, float] = field(default_factory=dict)
    concrete_vs_abstract: Dict[str, int] = field(default_factory=dict)


@dataclass
class VocabularyExpansion:
    """Suggested vocabulary expansions using Datamuse API"""
    original_word: str
    similar_meaning: List[str] = field(default_factory=list)
    contextual_links: List[str] = field(default_factory=list)
    sound_echoes: List[str] = field(default_factory=list)

    def all_alternatives(self) -> List[str]:
        """Get all alternative words"""
        return self.similar_meaning + self.contextual_links + self.sound_echoes


@dataclass
class InspiredStanza:
    """A generated stanza inspired by corpus patterns"""
    original_fragments: List[str]
    expanded_stanza: str
    substitution_map: Dict[str, str]  # original -> new word
    inspiration_type: str  # 'semantic', 'contextual', 'sonic'


@dataclass
class StyleFingerprint:
    """Complete style analysis of personal corpus"""
    metrics: PoetryMetrics = field(default_factory=PoetryMetrics)
    vocabulary: VocabularyProfile = field(default_factory=VocabularyProfile)
    themes: ThematicProfile = field(default_factory=ThematicProfile)

    # Compositional patterns
    opening_patterns: List[str] = field(default_factory=list)
    closing_patterns: List[str] = field(default_factory=list)
    transitional_phrases: List[str] = field(default_factory=list)

    # Inspiration expansions
    vocabulary_expansions: List[VocabularyExpansion] = field(default_factory=list)
    inspired_stanzas: List[InspiredStanza] = field(default_factory=list)


class PersonalCorpusAnalyzer:
    """Analyzes personal poetry collections to create detailed style profiles"""

    def __init__(self):
        """Initialize the analyzer with NLP tools"""
        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        except OSError:
            print("Warning: spaCy model not found. Some analysis features will be limited.")
            self.nlp = None

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')

        self.word_validator = WordValidator()
        self.datamuse_api = datamuse.Datamuse()

        # Extended stop words including function words and common contractions
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            'not', 'no', 'yes', 'so', 'too', 'very', 'just', 'only', 'even', 'also', 'still', 'more', 'most',
            'some', 'any', 'all', 'every', 'each', 'few', 'many', 'much', 'several', 'both', 'either', 'neither',
            'if', 'then', 'else', 'than', 'as', 'like', 'since', 'until', 'while', 'before', 'after', 'during',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'once', 'here', 'there', 'everywhere',
            'am', 'get', 'got', 'say', 'said', 'go', 'went', 'come', 'came', 'take', 'took', 'make', 'made',
            'done', 'doing', 'put', 'know', 'knew', 'think', 'thought', 'see', 'saw', 'look', 'looked',
            'way', 'one', 'two', 'three', 'first', 'last', 'next', 'back', 'away', 'around', 'about',
            'into', 'from', 'through', 'across', 'between', 'among', 'within', 'without', 'above', 'below'
        ])

    def analyze_directory(self, directory_path: str) -> StyleFingerprint:
        """Analyze all poetry files in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        poems = []
        for file_path in directory.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Skip empty files
                        poems.append({
                            'title': file_path.stem,
                            'content': content,
                            'path': str(file_path)
                        })
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

        if not poems:
            raise ValueError("No readable poetry files found in directory")

        return self.analyze_poems(poems)

    def analyze_poems(self, poems: List[Dict[str, str]]) -> StyleFingerprint:
        """Analyze a collection of poems"""
        fingerprint = StyleFingerprint()

        # Prepare text data
        all_text = []
        all_lines = []
        poem_structures = []

        for poem in poems:
            content = self._clean_poem_text(poem['content'])
            # Expand contractions for better analysis
            content = self._expand_contractions(content)
            lines = [line.strip() for line in content.split('\n') if line.strip()]

            all_text.append(content)
            all_lines.extend(lines)
            poem_structures.append(len(lines))

        # Calculate basic metrics
        fingerprint.metrics = self._calculate_metrics(poem_structures, all_lines)

        # Analyze vocabulary
        fingerprint.vocabulary = self._analyze_vocabulary(all_text, all_lines)

        # Analyze themes and patterns
        fingerprint.themes = self._analyze_themes(all_text, all_lines)

        # Identify compositional patterns
        self._analyze_compositional_patterns(all_lines, fingerprint)

        # Generate vocabulary expansions and inspired stanzas
        self._generate_vocabulary_expansions(fingerprint)
        self._generate_inspired_stanzas(all_lines, fingerprint)

        return fingerprint

    def _clean_poem_text(self, text: str) -> str:
        """Clean poem text while preserving formatting"""
        # Remove markdown formatting but preserve line breaks
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
        text = re.sub(r'\\$', '', text, flags=re.MULTILINE)     # Remove line continuation

        # Normalize whitespace while preserving intentional spacing
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions to improve analysis accuracy"""
        # First normalize smart quotes to regular apostrophes
        text = text.replace("'", "'").replace("'", "'").replace(chr(8217), "'").replace(chr(8216), "'")

        contractions = {
            "didn't": "did not",
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "couldn't": "could not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "doesn't": "does not",
            "i'm": "i am",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "where's": "where is",
            "when's": "when is",
            "how's": "how is",
            "who's": "who is",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "we'll": "we will",
            "they'll": "they will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "i'd": "i would",
            "you'd": "you would",
            "we'd": "we would",
            "they'd": "they would",
            "he'd": "he would",
            "she'd": "she would"
        }

        # Create pattern that matches contractions (case insensitive)
        contraction_pattern = r'\b(' + '|'.join(re.escape(key) for key in contractions.keys()) + r')\b'

        def replace_contraction(match):
            contraction = match.group(1).lower()
            expansion = contractions.get(contraction, contraction)
            # Preserve original capitalization
            if match.group(1)[0].isupper():
                expansion = expansion.capitalize()
            return expansion

        return re.sub(contraction_pattern, replace_contraction, text, flags=re.IGNORECASE)

    def _is_valid_word_for_analysis(self, word: str) -> bool:
        """Check if a word should be included in analysis"""
        # Skip very short words
        if len(word) < 2:
            return False

        # Skip words that are likely contractions fragments or typos
        invalid_fragments = {
            'didn', 'doesn', 'wouldn', 'couldn', 'shouldn', 'hasn', 'haven',
            'hadn', 'isn', 'aren', 'wasn', 'weren', 've', 'll', 're', 'd', 't', 's'
        }

        if word in invalid_fragments:
            return False

        # Skip words that are mostly punctuation or numbers
        if not word.isalpha():
            return False

        return True

    def _is_poetry_appropriate(self, word: str) -> bool:
        """Filter out words that might not be appropriate for poetry inspiration"""
        word_lower = word.lower()

        # Filter out crude slang and inappropriate terms
        inappropriate_words = {
            'marijuana', 'cannabis', 'pot', 'weed', 'dope', 'drug', 'drugs',
            'shit', 'crap', 'damn', 'hell', 'bitch', 'piss', 'fuck',
            'sex', 'sexual', 'porn', 'naked', 'rape',
            'kill', 'murder', 'death', 'dead', 'corpse', 'suicide',
            'money', 'cash', 'dollar', 'profit', 'business', 'corporate',
            'internet', 'website', 'computer', 'software', 'app',
            'covid', 'pandemic', 'virus', 'disease', 'cancer'
        }

        if word_lower in inappropriate_words:
            return False

        # Filter out words that are too technical or modern
        if len(word) > 12:  # Very long words are often technical
            return False

        # Filter out words with numbers or special characters
        if not word.isalpha():
            return False

        # Prefer words that are reasonably common but not too mundane
        freq = word_frequency(word_lower, 'en')
        if freq < 1e-7 or freq > 1e-3:  # Too rare or too common
            return False

        return True

    def _calculate_metrics(self, poem_structures: List[int], all_lines: List[str]) -> PoetryMetrics:
        """Calculate basic structural metrics"""
        metrics = PoetryMetrics()

        metrics.total_poems = len(poem_structures)
        metrics.total_lines = len(all_lines)
        metrics.avg_lines_per_poem = statistics.mean(poem_structures) if poem_structures else 0

        # Word and syllable analysis
        total_words = 0
        total_syllables = 0
        line_lengths = []

        for line in all_lines:
            words = line.split()
            word_count = len(words)
            total_words += word_count
            line_lengths.append(word_count)

            # Estimate syllables (simple heuristic)
            syllables = sum(self._count_syllables(word) for word in words)
            total_syllables += syllables

        metrics.total_words = total_words
        metrics.avg_words_per_line = total_words / len(all_lines) if all_lines else 0
        metrics.avg_syllables_per_line = total_syllables / len(all_lines) if all_lines else 0

        # Line length distribution
        length_counter = Counter(line_lengths)
        metrics.line_length_distribution = dict(length_counter)

        # Stanza patterns (simplified)
        metrics.stanza_patterns = poem_structures

        # Vocabulary richness (type-token ratio)
        all_words = ' '.join(all_lines).lower().split()
        unique_words = set(all_words)
        metrics.vocabulary_size = len(unique_words)
        metrics.vocabulary_richness = len(unique_words) / len(all_words) if all_words else 0

        # Punctuation and capitalization patterns
        all_text = ' '.join(all_lines)
        punctuation_chars = '.,!?;:-()[]"\'…'
        for char in punctuation_chars:
            metrics.punctuation_frequency[char] = all_text.count(char)

        # Capitalization patterns
        caps_words = [word for line in all_lines for word in line.split() if word and word[0].isupper()]
        metrics.capitalization_patterns['initial_caps'] = len(caps_words)
        metrics.capitalization_patterns['all_caps'] = len([w for w in caps_words if w.isupper()])

        return metrics

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting heuristic"""
        word = word.lower().strip('.,!?;:"\'')
        if not word:
            return 0

        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False

        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1

        return max(1, syllables)

    def _analyze_vocabulary(self, all_text: List[str], all_lines: List[str]) -> VocabularyProfile:
        """Analyze vocabulary patterns and preferences"""
        profile = VocabularyProfile()

        # Get all words
        all_words = []
        content_words = []

        for text in all_text:
            # Use more careful word extraction
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            # Filter out single characters and fragments
            valid_words = [w for w in words if len(w) >= 2 and self._is_valid_word_for_analysis(w)]
            all_words.extend(valid_words)

            # Filter content words (non-stop words)
            content_words.extend([w for w in valid_words if w not in self.stop_words and len(w) > 2])

        # Most common words
        word_counter = Counter(all_words)
        profile.most_common_words = word_counter.most_common(50)

        content_counter = Counter(content_words)
        profile.most_common_content_words = content_counter.most_common(30)

        # Signature words (high frequency relative to general usage)
        signature_words = []
        for word, count in content_counter.most_common(100):
            if count >= 2:  # Appears at least twice
                general_freq = word_frequency(word, 'en')
                if general_freq > 0:
                    corpus_freq = count / len(content_words)
                    signature_score = corpus_freq / general_freq
                    if signature_score > 10:  # Much more frequent than general usage
                        signature_words.append((word, signature_score))

        profile.signature_words = sorted(signature_words, key=lambda x: x[1], reverse=True)[:20]

        # Rare words (low general frequency but used in corpus)
        rare_words = []
        for word, count in content_counter.items():
            general_freq = word_frequency(word, 'en')
            if 0 < general_freq < 1e-6 and count >= 2:  # Rare but used multiple times
                rare_words.append((word, count))

        profile.rare_words = sorted(rare_words, key=lambda x: x[1], reverse=True)[:20]

        # Part-of-speech analysis (if spaCy available)
        if self.nlp:
            pos_counter = Counter()
            adjectives = Counter()
            verbs = Counter()
            nouns = Counter()

            for text in all_text:
                doc = self.nlp(text)
                for token in doc:
                    if token.is_alpha and not token.is_stop:
                        pos_counter[token.pos_] += 1

                        if token.pos_ == 'ADJ':
                            adjectives[token.lemma_.lower()] += 1
                        elif token.pos_ == 'VERB':
                            verbs[token.lemma_.lower()] += 1
                        elif token.pos_ in ['NOUN', 'PROPN']:
                            nouns[token.lemma_.lower()] += 1

            profile.pos_distribution = dict(pos_counter)
            profile.preferred_adjectives = adjectives.most_common(20)
            profile.preferred_verbs = verbs.most_common(20)
            profile.preferred_nouns = nouns.most_common(20)

        return profile

    def _analyze_themes(self, all_text: List[str], all_lines: List[str]) -> ThematicProfile:
        """Analyze thematic patterns and semantic clusters"""
        profile = ThematicProfile()

        # Simple semantic clustering based on word co-occurrence
        if self.nlp:
            # Collect content words by poem
            poem_words = []
            for text in all_text:
                doc = self.nlp(text)
                words = [token.lemma_.lower() for token in doc
                        if token.is_alpha and not token.is_stop and len(token.text) > 2]
                poem_words.append(words)

            # Find words that frequently appear together
            co_occurrence = defaultdict(Counter)
            for words in poem_words:
                for i, word1 in enumerate(words):
                    for word2 in words[i+1:i+6]:  # Look at nearby words
                        if word1 != word2:
                            co_occurrence[word1][word2] += 1
                            co_occurrence[word2][word1] += 1

            # Create semantic clusters
            clusters = []
            processed = set()

            for word, related in co_occurrence.items():
                if word not in processed and len(related) >= 3:
                    cluster_words = [word]
                    for related_word, count in related.most_common(5):
                        if count >= 2:
                            cluster_words.append(related_word)

                    if len(cluster_words) >= 3:
                        clusters.append((word, cluster_words[1:]))
                        processed.update(cluster_words)

            profile.semantic_clusters = clusters[:10]

        # Look for metaphor patterns (simple heuristics)
        metaphor_indicators = ['like', 'as', 'is', 'was', 'becomes', 'turns into']
        metaphor_patterns = []

        for line in all_lines:
            line_lower = line.lower()
            for indicator in metaphor_indicators:
                if indicator in line_lower:
                    # Extract potential metaphor
                    pattern = re.search(fr'(.{{0,20}}){re.escape(indicator)}(.{{0,30}})', line_lower)
                    if pattern:
                        metaphor_patterns.append(pattern.group(0).strip())

        profile.metaphor_patterns = list(set(metaphor_patterns))[:15]

        # Emotional register analysis (basic sentiment indicators)
        emotional_words = {
            'joy': ['joy', 'happy', 'bright', 'light', 'laugh', 'dance', 'sing', 'hope', 'love'],
            'melancholy': ['dark', 'shadow', 'empty', 'hollow', 'fade', 'lost', 'quiet', 'still'],
            'intensity': ['fire', 'burn', 'rage', 'storm', 'wild', 'fierce', 'sharp', 'cut'],
            'contemplation': ['think', 'wonder', 'perhaps', 'maybe', 'consider', 'ponder']
        }

        emotion_scores = {emotion: 0 for emotion in emotional_words}
        total_words = sum(len(text.split()) for text in all_text)

        for emotion, words in emotional_words.items():
            count = sum(text.lower().count(word) for text in all_text for word in words)
            emotion_scores[emotion] = count / total_words if total_words > 0 else 0

        profile.emotional_register = emotion_scores

        # Concrete vs abstract analysis
        concrete_indicators = ['see', 'hear', 'touch', 'smell', 'taste', 'hand', 'eye', 'skin']
        abstract_indicators = ['think', 'feel', 'believe', 'hope', 'dream', 'memory', 'soul']

        concrete_count = sum(text.lower().count(word) for text in all_text for word in concrete_indicators)
        abstract_count = sum(text.lower().count(word) for text in all_text for word in abstract_indicators)

        profile.concrete_vs_abstract = {
            'concrete': concrete_count,
            'abstract': abstract_count,
            'ratio': concrete_count / max(abstract_count, 1)
        }

        return profile

    def _analyze_compositional_patterns(self, all_lines: List[str], fingerprint: StyleFingerprint):
        """Analyze how poems are structured and composed"""
        if not all_lines:
            return

        # Opening patterns (first lines or first few words)
        opening_patterns = []
        closing_patterns = []

        # This is simplified - in a real implementation, you'd group by poems
        # For now, we'll look at patterns in line beginnings and endings

        line_starts = Counter()
        line_ends = Counter()

        for line in all_lines:
            words = line.split()
            if words:
                # First 1-3 words
                if len(words) >= 1:
                    line_starts[words[0].lower()] += 1
                if len(words) >= 2:
                    line_starts[f"{words[0]} {words[1]}".lower()] += 1

                # Last 1-2 words
                line_ends[words[-1].lower().rstrip('.,!?;:"')] += 1
                if len(words) >= 2:
                    last_two = f"{words[-2]} {words[-1]}".lower().rstrip('.,!?;:"')
                    line_ends[last_two] += 1

        # Keep patterns that appear multiple times
        fingerprint.opening_patterns = [pattern for pattern, count in line_starts.most_common(20) if count >= 2]
        fingerprint.closing_patterns = [pattern for pattern, count in line_ends.most_common(15) if count >= 2]

        # Transitional phrases
        transitions = ['but', 'and', 'yet', 'still', 'then', 'now', 'here', 'there', 'when', 'where']
        transition_phrases = []

        for line in all_lines:
            line_lower = line.lower()
            for trans in transitions:
                if line_lower.startswith(trans + ' '):
                    phrase = line_lower[:30]  # First 30 chars
                    transition_phrases.append(phrase)

        fingerprint.transitional_phrases = list(set(transition_phrases))[:10]

    def generate_style_report(self, fingerprint: StyleFingerprint) -> str:
        """Generate a human-readable style analysis report"""
        report = []

        report.append("PERSONAL POETRY STYLE ANALYSIS")
        report.append("=" * 50)

        # Basic metrics
        m = fingerprint.metrics
        report.append(f"\nCORPUS OVERVIEW:")
        report.append(f"  Total poems: {m.total_poems}")
        report.append(f"  Total lines: {m.total_lines}")
        report.append(f"  Total words: {m.total_words}")
        report.append(f"  Average lines per poem: {m.avg_lines_per_poem:.1f}")
        report.append(f"  Average words per line: {m.avg_words_per_line:.1f}")
        report.append(f"  Vocabulary size: {m.vocabulary_size}")
        report.append(f"  Vocabulary richness: {m.vocabulary_richness:.3f}")

        # Vocabulary insights
        v = fingerprint.vocabulary
        report.append(f"\nVOCABULARY PROFILE:")

        if v.signature_words:
            report.append(f"  Signature words (distinctively yours):")
            for word, score in v.signature_words[:10]:
                report.append(f"    • {word} (signature strength: {score:.1f}x)")

        if v.most_common_content_words:
            report.append(f"  Most frequent content words:")
            for word, count in v.most_common_content_words[:10]:
                report.append(f"    • {word} ({count} times)")

        if v.preferred_adjectives:
            report.append(f"  Preferred adjectives:")
            adj_list = [f"{word}({count})" for word, count in v.preferred_adjectives[:8]]
            report.append(f"    {', '.join(adj_list)}")

        # Thematic analysis
        t = fingerprint.themes
        report.append(f"\nTHEMATIC PATTERNS:")

        if t.semantic_clusters:
            report.append(f"  Word clusters (related concepts):")
            for theme, words in t.semantic_clusters[:5]:
                report.append(f"    • {theme}: {', '.join(words[:5])}")

        if t.emotional_register:
            report.append(f"  Emotional register:")
            for emotion, score in t.emotional_register.items():
                if score > 0:
                    report.append(f"    • {emotion}: {score:.4f}")

        concrete_ratio = t.concrete_vs_abstract.get('ratio', 0)
        if concrete_ratio > 0:
            tendency = "concrete" if concrete_ratio > 1 else "abstract"
            report.append(f"  Imagery tendency: {tendency} ({concrete_ratio:.2f}:1 ratio)")

        # Compositional patterns
        report.append(f"\nCOMPOSITIONAL STYLE:")

        if fingerprint.opening_patterns:
            report.append(f"  Common opening patterns:")
            for pattern in fingerprint.opening_patterns[:8]:
                report.append(f"    • '{pattern}'")

        if fingerprint.transitional_phrases:
            report.append(f"  Transitional phrases:")
            for phrase in fingerprint.transitional_phrases[:6]:
                report.append(f"    • '{phrase}'")

        # Structural patterns
        if m.line_length_distribution:
            most_common_length = max(m.line_length_distribution.items(), key=lambda x: x[1])
            report.append(f"  Preferred line length: {most_common_length[0]} words")

        avg_poem_length = m.avg_lines_per_poem
        if avg_poem_length > 0:
            if avg_poem_length < 10:
                length_style = "brief, concentrated"
            elif avg_poem_length < 25:
                length_style = "moderate length"
            else:
                length_style = "extended, expansive"
            report.append(f"  Poem length tendency: {length_style} ({avg_poem_length:.1f} lines avg)")

        return "\n".join(report)

    def suggest_expansions(self, fingerprint: StyleFingerprint) -> List[str]:
        """Suggest creative expansions while maintaining authentic voice"""
        suggestions = []

        # Based on vocabulary analysis
        if fingerprint.vocabulary.signature_words:
            top_sig_words = [word for word, _ in fingerprint.vocabulary.signature_words[:5]]
            suggestions.append(f"Explore variations of your signature words: {', '.join(top_sig_words)}")

        # Based on thematic clusters
        if fingerprint.themes.semantic_clusters:
            unused_combinations = []
            for theme, words in fingerprint.themes.semantic_clusters[:3]:
                unused_combinations.append(f"{theme} + {words[0]}")
            suggestions.append(f"Try unexpected combinations: {', '.join(unused_combinations)}")

        # Based on emotional register
        emotions = fingerprint.themes.emotional_register
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            opposite_emotions = {
                'joy': 'melancholy',
                'melancholy': 'joy',
                'intensity': 'contemplation',
                'contemplation': 'intensity'
            }
            opposite = opposite_emotions.get(dominant_emotion[0])
            if opposite:
                suggestions.append(f"Consider exploring {opposite} to balance your {dominant_emotion[0]} tendency")

        # Based on structure
        avg_length = fingerprint.metrics.avg_lines_per_poem
        if avg_length < 15:
            suggestions.append("Try writing a longer piece to explore extended development of ideas")
        elif avg_length > 30:
            suggestions.append("Experiment with condensed, haiku-like brevity")

        # Based on concrete/abstract ratio
        ratio = fingerprint.themes.concrete_vs_abstract.get('ratio', 1)
        if ratio > 2:
            suggestions.append("Balance concrete imagery with more abstract concepts")
        elif ratio < 0.5:
            suggestions.append("Ground abstract ideas with more sensory, concrete details")

        return suggestions

    def _generate_vocabulary_expansions(self, fingerprint: StyleFingerprint):
        """Generate vocabulary expansions using Datamuse API for signature words"""
        print("Generating vocabulary expansions...")

        # Take top signature words for expansion
        top_words = [word for word, _ in fingerprint.vocabulary.signature_words[:12]]

        for word in top_words:
            try:
                expansion = VocabularyExpansion(original_word=word)

                # Get similar meaning words
                similar = similar_meaning_words(word, sample_size=6, datamuse_api_max=20)
                expansion.similar_meaning = [w for w in similar if w != word and self._is_poetry_appropriate(w)][:4]

                # Get contextually linked words
                contextual = contextually_linked_words(word, sample_size=6, datamuse_api_max=20)
                expansion.contextual_links = [w for w in contextual if w != word and self._is_poetry_appropriate(w)][:4]

                # Get sound echoes (similar sounding)
                sonic = similar_sounding_words(word, sample_size=5, datamuse_api_max=15)
                expansion.sound_echoes = [w for w in sonic if w != word and self._is_poetry_appropriate(w)][:3]

                # Only add if we found some alternatives
                if expansion.all_alternatives():
                    fingerprint.vocabulary_expansions.append(expansion)

            except Exception as e:
                print(f"Could not expand '{word}': {e}")
                continue

    def _generate_inspired_stanzas(self, all_lines: List[str], fingerprint: StyleFingerprint):
        """Generate sample stanzas using vocabulary expansions"""
        print("Generating inspired stanzas...")

        if not fingerprint.vocabulary_expansions:
            return

        # Find interesting source lines with signature words
        source_lines = []
        signature_words = set(word for word, _ in fingerprint.vocabulary.signature_words[:15])

        for line in all_lines:
            line_words = set(re.findall(r'\b[a-zA-Z]+\b', line.lower()))
            if signature_words & line_words and len(line.split()) >= 4:
                source_lines.append(line)

        # Generate different types of inspired stanzas
        for inspiration_type in ['semantic', 'contextual', 'sonic']:
            try:
                stanza = self._create_inspired_stanza(source_lines, fingerprint, inspiration_type)
                if stanza:
                    fingerprint.inspired_stanzas.append(stanza)
            except Exception as e:
                print(f"Could not generate {inspiration_type} stanza: {e}")

    def _create_inspired_stanza(self, source_lines: List[str], fingerprint: StyleFingerprint,
                               inspiration_type: str) -> Optional[InspiredStanza]:
        """Create a single inspired stanza using vocabulary substitutions"""
        import random

        if len(source_lines) < 3:
            return None

        # Pick 3-4 random source lines for the stanza
        selected_lines = random.sample(source_lines, min(4, len(source_lines)))
        original_fragments = selected_lines.copy()

        # Build substitution map based on inspiration type
        substitution_map = {}

        for expansion in fingerprint.vocabulary_expansions[:8]:  # Use top expansions
            original_word = expansion.original_word

            # Choose replacement based on inspiration type
            if inspiration_type == 'semantic' and expansion.similar_meaning:
                replacement = random.choice(expansion.similar_meaning)
            elif inspiration_type == 'contextual' and expansion.contextual_links:
                replacement = random.choice(expansion.contextual_links)
            elif inspiration_type == 'sonic' and expansion.sound_echoes:
                replacement = random.choice(expansion.sound_echoes)
            else:
                continue

            substitution_map[original_word] = replacement

        if not substitution_map:
            return None

        # Apply substitutions to create new stanza
        new_lines = []
        for line in selected_lines:
            new_line = line
            for original, replacement in substitution_map.items():
                # Case-sensitive replacement
                new_line = re.sub(r'\b' + re.escape(original) + r'\b', replacement, new_line, flags=re.IGNORECASE)
                # Handle capitalization
                new_line = re.sub(r'\b' + re.escape(original.capitalize()) + r'\b', replacement.capitalize(), new_line)

            new_lines.append(new_line)

        expanded_stanza = '\n'.join(new_lines)

        return InspiredStanza(
            original_fragments=original_fragments,
            expanded_stanza=expanded_stanza,
            substitution_map=substitution_map,
            inspiration_type=inspiration_type
        )

    def generate_inspiration_report(self, fingerprint: StyleFingerprint) -> str:
        """Generate a report focused on creative inspiration and vocabulary expansion"""
        report = []

        report.append("CREATIVE INSPIRATION GENERATOR")
        report.append("=" * 60)

        # Vocabulary expansions
        if fingerprint.vocabulary_expansions:
            report.append(f"\nVOCABULARY EXPANSIONS:")
            report.append("Explore these alternatives to your signature words:")

            for expansion in fingerprint.vocabulary_expansions[:8]:
                report.append(f"\n• '{expansion.original_word}' →")

                if expansion.similar_meaning:
                    report.append(f"    Similar meaning: {', '.join(expansion.similar_meaning)}")

                if expansion.contextual_links:
                    report.append(f"    Contextually linked: {', '.join(expansion.contextual_links)}")

                if expansion.sound_echoes:
                    report.append(f"    Sound echoes: {', '.join(expansion.sound_echoes)}")

        # Inspired stanzas
        if fingerprint.inspired_stanzas:
            report.append(f"\n\nINSPIRED STANZA VARIATIONS:")
            report.append("Sample stanzas reimagined with expanded vocabulary:")

            for i, stanza in enumerate(fingerprint.inspired_stanzas, 1):
                report.append(f"\n{i}. {stanza.inspiration_type.title()} Variation:")
                report.append("-" * 40)
                report.append(stanza.expanded_stanza)

                if stanza.substitution_map:
                    subs = [f"{k}→{v}" for k, v in list(stanza.substitution_map.items())[:3]]
                    report.append(f"    Substitutions: {', '.join(subs)}")

        # Creative prompts
        report.append(f"\n\nCREATIVE PROMPTS:")
        report.append("-" * 30)

        if fingerprint.vocabulary.signature_words:
            top_words = [word for word, _ in fingerprint.vocabulary.signature_words[:5]]
            report.append(f"• Write a poem replacing '{top_words[0]}' with its alternatives")
            report.append(f"• Combine '{top_words[1]}' with contextually distant words")
            report.append(f"• Echo the sound of '{top_words[2]}' throughout a piece")

        if fingerprint.themes.semantic_clusters:
            clusters = fingerprint.themes.semantic_clusters[:2]
            report.append(f"• Cross-pollinate word groups: {clusters[0][0]} + {clusters[1][0] if len(clusters) > 1 else 'new territory'}")

        return "\n".join(report)
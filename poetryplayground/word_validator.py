"""Enhanced word validation to ensure quality output."""

import nltk
from nltk.corpus import brown, words
from typing import Dict, Optional
from wordfreq import word_frequency

# Optional imports for advanced features
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from .logger import logger

# Try to ensure NLTK data is available
try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words", quiet=True)

try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


class WordValidator:
    """Validate words to ensure they are real English words."""

    def __init__(self):
        """Initialize word validator with dictionaries and NLP tools."""
        # Load English word lists
        try:
            self._nltk_words = {word.lower() for word in words.words()}
        except Exception:
            self._nltk_words = set()

        try:
            # Get common words from Brown corpus
            self._brown_words = {
                word.lower() for word in brown.words() if word.isalpha() and len(word) > 2
            }
        except Exception:
            self._brown_words = set()

        # Initialize spaCy for NER (if available)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("WordValidator: spaCy NER enabled")
            except Exception as e:
                logger.warning(f"WordValidator: Could not load spaCy model: {e}")

        # Initialize VADER sentiment analyzer (if available)
        self.sentiment_analyzer = None
        if VADER_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("WordValidator: VADER sentiment analysis enabled")
            except Exception as e:
                logger.warning(f"WordValidator: Could not initialize VADER: {e}")

        # Common proper nouns and names to exclude
        self._proper_nouns = {
            "allen",
            "connor",
            "connors",
            "fischer",
            "fisher",
            "fraser",
            "frasier",
            "bamburgh",
            "billingham",
            "bellingham",
            "dublin",
            "hamburg",
            "darcy",
            "fenner",
            "feiner",
            "fancher",
            "fleischer",
            "fluker",
            "chevalier",
            "marcell",
            "eakin",
            "cole",
            "ayer",
            "albion",
            "cambs",
            "gms",
            "fuehrer",
            "futur",
            "knut",
            "nair",
            "amma",
            "mani",
            "nere",
            "narc",
            "nare",
            "lublin",
            "vale",
            "fletcher",
            "griff",
            "tennent",
            "tenant",
            "tope",
            "swope",
            "fuser",
        }

        # Non-English or problematic words
        self._excluded_words = {
            "knut",
            "fuehrer",
            "futur",
            "witte",
            "wayde",
            "thay",
            "wid",
            "geet",
            "girt",
            "dern",
            "jut",
            "comers",
            "conners",
            "spew",
            "spat",
            "shutt",
            "kut",
            "nir",
            "gms",
            "bratt",
            "brat",
            "doesnt",
            "dont",
            "cant",
            "wont",
            "shouldnt",
            "couldnt",  # Missing apostrophes
            "tope",
            "hap",
            "amma",
            "nare",
            "narc",
        }

        # Minimum word frequency for common words (more restrictive)
        self.min_frequency = 5e-06  # Much higher than current 4e-08

    def is_valid_english_word(self, word: str, allow_rare: bool = False) -> bool:
        """
        Check if a word is a valid English word.

        Args:
            word: Word to validate
            allow_rare: Whether to allow rare words

        Returns:
            True if word is valid English word
        """
        word_lower = word.lower().strip()

        # Basic checks
        if len(word_lower) < 2 or len(word_lower) > 15:
            return False

        # Check for invalid characters
        if not word_lower.isalpha():
            return False

        # Exclude proper nouns and known bad words
        if word_lower in self._proper_nouns or word_lower in self._excluded_words:
            return False

        # Check against NLTK words dictionary
        if (
            self._nltk_words
            and word_lower not in self._nltk_words
            and self._brown_words
            and word_lower not in self._brown_words
            and "'" not in word_lower
        ):
            return False

        # Check word frequency
        freq = word_frequency(word_lower, "en")
        if freq < self.min_frequency and not allow_rare:
            return False

        # Additional check: must start with a common letter pattern
        return word_lower[:2] not in ["xz", "qx", "qz", "zx", "vx"]

    def is_proper_noun(self, word: str, use_nlp: bool = True) -> bool:
        """Check if word is likely a proper noun.

        Args:
            word: Word to check
            use_nlp: Whether to use spaCy NER (if available)

        Returns:
            True if word appears to be a proper noun
        """
        # Check if in our known proper nouns list (fallback)
        if word.lower() in self._proper_nouns:
            return True

        # Use spaCy NER if available and requested
        if use_nlp and self.nlp:
            try:
                doc = self.nlp(word)
                if len(doc) > 0:
                    # Check if it's tagged as a named entity
                    # Common entity types: PERSON, GPE (Geo-Political Entity),
                    # ORG, LOC, FAC, NORP, EVENT
                    ent_type = doc[0].ent_type_
                    if ent_type in ['PERSON', 'GPE', 'ORG', 'LOC', 'FAC', 'NORP', 'EVENT']:
                        return True

                    # Also check POS tag
                    if doc[0].pos_ == 'PROPN':  # Proper noun
                        return True
            except Exception as e:
                logger.debug(f"NER check failed for '{word}': {e}")

        # Fallback: check if it's capitalized and not in dictionary
        return bool(word[0].isupper() and word.lower() not in self._nltk_words)

    def clean_word_list(self, words: list, allow_rare: bool = False, exclude_words: list = None) -> list:
        """
        Clean a list of words, removing invalid ones.

        Args:
            words: List of words to clean
            allow_rare: Whether to allow rare words
            exclude_words: Optional list of words to exclude

        Returns:
            List of valid English words
        """
        if exclude_words is None:
            exclude_words = []

        # Normalize exclude list to lowercase
        exclude_set = {w.lower() for w in exclude_words if isinstance(w, str)}

        cleaned = []
        for word in words:
            if isinstance(word, str):
                word_lower = word.lower()
                if word_lower not in exclude_set and self.is_valid_english_word(word, allow_rare):
                    cleaned.append(word_lower)
        return cleaned

    def get_word_quality_score(self, word: str) -> float:
        """
        Get a quality score for a word (0-1).
        Higher scores mean better/more common words.
        """
        if not self.is_valid_english_word(word, allow_rare=True):
            return 0.0

        # Base score on frequency
        freq = word_frequency(word.lower(), "en")

        # Convert frequency to 0-1 scale (log scale)
        # Most common words have frequency ~1e-3, very rare ~1e-8
        if freq > 0:
            import math

            # Map log frequency from [-8, -3] to [0, 1]
            log_freq = math.log10(freq)
            score = (log_freq + 8) / 5  # Normalize to 0-1
            score = max(0, min(1, score))  # Clamp to 0-1
        else:
            score = 0.0

        # Penalty for very short or very long words
        length = len(word)
        if length < 3:
            score *= 0.7
        elif length > 12:
            score *= 0.8

        # Bonus for words in Brown corpus (common usage)
        if self._brown_words and word.lower() in self._brown_words:
            score = min(1.0, score * 1.2)

        return score

    def get_sentiment(self, word: str) -> Dict[str, float]:
        """Get sentiment scores for a word.

        Args:
            word: Word to analyze

        Returns:
            Dictionary with sentiment scores:
            - pos: positive score (0-1)
            - neg: negative score (0-1)
            - neu: neutral score (0-1)
            - compound: compound score (-1 to 1)
        """
        if not self.sentiment_analyzer:
            # Return neutral if VADER not available
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

        try:
            scores = self.sentiment_analyzer.polarity_scores(word)
            return scores
        except Exception as e:
            logger.debug(f"Sentiment analysis failed for '{word}': {e}")
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

    def has_positive_sentiment(self, word: str, threshold: float = 0.1) -> bool:
        """Check if word has positive sentiment.

        Args:
            word: Word to check
            threshold: Minimum compound score to consider positive

        Returns:
            True if word is positively valenced
        """
        sentiment = self.get_sentiment(word)
        return sentiment["compound"] > threshold

    def has_negative_sentiment(self, word: str, threshold: float = -0.1) -> bool:
        """Check if word has negative sentiment.

        Args:
            word: Word to check
            threshold: Maximum compound score to consider negative

        Returns:
            True if word is negatively valenced
        """
        sentiment = self.get_sentiment(word)
        return sentiment["compound"] < threshold

    def filter_by_sentiment(self, words: list, emotional_tone: str) -> list:
        """Filter words by emotional tone.

        Args:
            words: List of words to filter
            emotional_tone: Desired tone ("positive", "negative", "neutral", "any")

        Returns:
            Filtered list of words matching the tone
        """
        if emotional_tone == "any":
            return words

        filtered = []
        for word in words:
            sentiment = self.get_sentiment(word)
            compound = sentiment["compound"]

            if emotional_tone == "positive" and compound > 0.1:
                filtered.append(word)
            elif emotional_tone == "negative" and compound < -0.1:
                filtered.append(word)
            elif emotional_tone == "neutral" and -0.1 <= compound <= 0.1:
                filtered.append(word)

        return filtered


# Global validator instance
word_validator = WordValidator()

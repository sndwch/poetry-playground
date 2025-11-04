"""Enhanced word validation to ensure quality output."""

import nltk
from nltk.corpus import brown, words
from wordfreq import word_frequency

# Try to ensure NLTK data is available
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class WordValidator:
    """Validate words to ensure they are real English words."""

    def __init__(self):
        """Initialize word validator with dictionaries."""
        # Load English word lists
        try:
            self._nltk_words = {word.lower() for word in words.words()}
        except Exception:
            self._nltk_words = set()

        try:
            # Get common words from Brown corpus
            self._brown_words = {word.lower() for word in brown.words()
                                   if word.isalpha() and len(word) > 2}
        except Exception:
            self._brown_words = set()

        # Common proper nouns and names to exclude
        self._proper_nouns = {
            'allen', 'connor', 'connors', 'fischer', 'fisher', 'fraser', 'frasier',
            'bamburgh', 'billingham', 'bellingham', 'dublin', 'hamburg',
            'darcy', 'fenner', 'feiner', 'fancher', 'fleischer', 'fluker',
            'chevalier', 'marcell', 'eakin', 'cole', 'ayer',
            'albion', 'cambs', 'gms', 'fuehrer', 'futur', 'knut', 'nair',
            'amma', 'mani', 'nere', 'narc', 'nare', 'lublin', 'vale',
            'fletcher', 'griff', 'tennent', 'tenant', 'tope', 'swope',
            'fuser'
        }

        # Non-English or problematic words
        self._excluded_words = {
            'knut', 'fuehrer', 'futur', 'witte', 'wayde', 'thay', 'wid',
            'geet', 'girt', 'dern', 'jut', 'comers', 'conners', 'spew',
            'spat', 'shutt', 'kut', 'nir', 'gms', 'bratt', 'brat',
            'doesnt', 'dont', 'cant', 'wont', 'shouldnt', 'couldnt',  # Missing apostrophes
            'tope', 'hap', 'amma', 'nare', 'narc'
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
        if self._nltk_words and word_lower not in self._nltk_words:
            # Also check if it's in Brown corpus (common words)
            if self._brown_words and word_lower not in self._brown_words:
                # Allow contractions with apostrophes
                if "'" not in word_lower:
                    return False

        # Check word frequency
        freq = word_frequency(word_lower, 'en')
        if freq < self.min_frequency and not allow_rare:
            return False

        # Additional check: must start with a common letter pattern
        return word_lower[:2] not in ['xz', 'qx', 'qz', 'zx', 'vx']

    def is_proper_noun(self, word: str) -> bool:
        """Check if word is likely a proper noun."""
        # Check if in our known proper nouns list
        if word.lower() in self._proper_nouns:
            return True

        # Check if it's capitalized and not in dictionary
        return bool(word[0].isupper() and word.lower() not in self._nltk_words)

    def clean_word_list(self, words: list, allow_rare: bool = False) -> list:
        """
        Clean a list of words, removing invalid ones.

        Args:
            words: List of words to clean
            allow_rare: Whether to allow rare words

        Returns:
            List of valid English words
        """
        cleaned = []
        for word in words:
            if isinstance(word, str) and self.is_valid_english_word(word, allow_rare):
                cleaned.append(word.lower())
        return cleaned

    def get_word_quality_score(self, word: str) -> float:
        """
        Get a quality score for a word (0-1).
        Higher scores mean better/more common words.
        """
        if not self.is_valid_english_word(word, allow_rare=True):
            return 0.0

        # Base score on frequency
        freq = word_frequency(word.lower(), 'en')

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


# Global validator instance
word_validator = WordValidator()

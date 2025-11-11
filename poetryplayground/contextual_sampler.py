"""Contextual Sampler: Extract context-aware text samples from Project Gutenberg.

This module provides tools for finding specific words or phrases in classic literature
and extracting the surrounding context (sentences before and after) as creative prompts.

Example:
    >>> from poetryplayground.contextual_sampler import ContextualSampler
    >>> from poetryplayground.core.document_library import document_library
    >>> from poetryplayground.core.quality_scorer import get_quality_scorer
    >>>
    >>> sampler = ContextualSampler(document_library, get_quality_scorer())
    >>> samples = sampler.sample(
    ...     search_term="tomorrow",
    ...     sample_count=10,
    ...     context_window=200,
    ...     min_quality=0.5
    ... )
    >>> for sample in samples:
    ...     print(f"{sample.quality_score:.2f}: {sample.text}")
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import nltk

from poetryplayground.logger import logger

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

module_logger = logging.getLogger(__name__)


@dataclass
class ContextSample:
    """A text sample with context around a search term.

    Attributes:
        text: The extracted context text (before + after sentences)
        quality_score: Quality score from 0.0 to 1.0
        source_title: Title of the source document
        source_author: Author of the source document
        match_position: Character position of the match in the original document
        search_term: The search term that was found
        before_context: The sentence before the match
        after_context: The sentence after the match
    """

    text: str
    quality_score: float
    source_title: str
    source_author: str
    match_position: int
    search_term: str
    before_context: str
    after_context: str


class ContextualSampler:
    """Extract context-aware samples from Project Gutenberg literature.

    This class searches for specific words or phrases in classic texts and extracts
    the surrounding context (sentences before and after) as creative prompts.
    """

    def __init__(self, document_library, quality_scorer, transformer=None):
        """Initialize the contextual sampler.

        Args:
            document_library: DocumentLibrary instance for accessing texts
            quality_scorer: QualityScorer instance for scoring samples
            transformer: Optional ShipOfTheseusTransformer for transformations
        """
        self.document_library = document_library
        self.quality_scorer = quality_scorer
        self.transformer = transformer

    def sample(
        self,
        search_term: str,
        sample_count: int,
        context_window: int,
        min_quality: float = 0.5,
        locc_codes: Optional[List[str]] = None,
    ) -> List[ContextSample]:
        """Extract contextual samples from literature.

        Args:
            search_term: Word or phrase to search for
            sample_count: Number of samples to retrieve
            context_window: Maximum characters of context around the search term
            min_quality: Minimum quality score (0.0-1.0)
            locc_codes: Optional list of Library of Congress Classification codes
                       (e.g., ["PZ", "PR", "PS"] for fiction and literature)

        Returns:
            List of ContextSample objects, sorted by quality score (descending)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not search_term or not search_term.strip():
            raise ValueError("search_term must be a non-empty string")
        if sample_count < 1 or sample_count > 100:
            raise ValueError("sample_count must be between 1 and 100")
        if context_window < 50 or context_window > 1000:
            raise ValueError("context_window must be between 50 and 1000")
        if min_quality < 0.0 or min_quality > 1.0:
            raise ValueError("min_quality must be between 0.0 and 1.0")

        search_term = search_term.strip()
        logger.info(
            f"ContextualSampler: Searching for '{search_term}' "
            f"(target: {sample_count} samples, min_quality: {min_quality:.2f})"
        )

        # Calculate how many documents we need to fetch
        # Fetch more than needed to account for filtering
        fetch_count = min(sample_count * 2, 20)

        # Get diverse documents from the corpus
        try:
            documents = self.document_library.get_diverse_documents(
                count=fetch_count, min_length=5000, locc_codes=locc_codes
            )
            logger.info(f"Retrieved {len(documents)} documents from corpus")
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

        # Search documents in parallel
        all_samples = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit search tasks for each document (enumerate for tracking)
            futures = {
                executor.submit(
                    self._search_document,
                    doc,
                    search_term,
                    context_window,
                    min_quality,
                    doc_index,
                ): doc_index
                for doc_index, doc in enumerate(documents)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                doc_index = futures[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                    logger.debug(f"Found {len(samples)} samples in document {doc_index + 1}")
                except Exception as e:
                    logger.warning(f"Error searching document {doc_index + 1}: {e}")
                    continue

        # Sort by quality score (descending)
        all_samples.sort(key=lambda s: s.quality_score, reverse=True)

        # Return top N samples
        result = all_samples[:sample_count]
        logger.info(
            f"ContextualSampler: Found {len(result)} samples (from {len(all_samples)} total)"
        )

        return result

    def _search_document(
        self,
        document: str,
        search_term: str,
        context_window: int,
        min_quality: float,
        doc_index: int = 0,
    ) -> List[ContextSample]:
        """Search a single document for the search term and extract contexts.

        Args:
            document: Document text (string)
            search_term: Term to search for
            context_window: Maximum characters of context
            min_quality: Minimum quality threshold
            doc_index: Document index for identification (optional)

        Returns:
            List of ContextSample objects from this document
        """
        samples = []

        # Document is just a string (text), not a dict
        text = document
        # Use placeholder metadata since DocumentLibrary doesn't provide it
        title = f"Project Gutenberg Document #{doc_index + 1}"
        author = "Unknown Author"

        if not text:
            return samples

        # Split into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Failed to tokenize document {title}: {e}")
            return samples

        # Search for the term
        search_pattern = re.compile(re.escape(search_term), re.IGNORECASE)

        for i, sentence in enumerate(sentences):
            # Check if search term appears in this sentence
            match = search_pattern.search(sentence)
            if not match:
                continue

            # Extract context (sentence before and after)
            before_context = ""
            after_context = ""

            if i > 0:
                before_context = sentences[i - 1]
            if i < len(sentences) - 1:
                after_context = sentences[i + 1]

            # Skip if we don't have both before and after context
            if not before_context or not after_context:
                continue

            # Combine context (limit to context_window)
            combined_text = f"{before_context} {after_context}"
            if len(combined_text) > context_window:
                # Truncate to fit within window
                half_window = context_window // 2
                before_context = before_context[-half_window:]
                after_context = after_context[:half_window]
                combined_text = f"{before_context} {after_context}"

            # Score the sample
            quality_score = self._score_sample(combined_text, search_term)

            # Filter by minimum quality
            if quality_score < min_quality:
                continue

            # Calculate match position in original text
            match_position = text.find(sentence)

            # Create sample
            sample = ContextSample(
                text=combined_text.strip(),
                quality_score=quality_score,
                source_title=title,
                source_author=author,
                match_position=match_position,
                search_term=search_term,
                before_context=before_context.strip(),
                after_context=after_context.strip(),
            )

            samples.append(sample)

        return samples

    def _score_sample(self, context_text: str, search_term: str) -> float:
        """Calculate quality score for a context sample.

        Args:
            context_text: The combined context text
            search_term: The search term

        Returns:
            Quality score from 0.0 to 1.0
        """
        # Extract words from context
        words = re.findall(r"\b\w+\b", context_text.lower())

        if not words:
            return 0.0

        # Score each word and average
        scores = []
        for word in words:
            try:
                word_score = self.quality_scorer.score_word(word)
                scores.append(word_score.overall)
            except Exception:
                # Skip words that can't be scored
                continue

        if not scores:
            return 0.5  # Default for unknown words

        # Calculate average quality
        avg_quality = sum(scores) / len(scores)

        # Bonus for literary/poetic words (high quality score)
        literary_bonus = 0.0
        if avg_quality > 0.7:
            literary_bonus = 0.1

        # Penalty for very common/boring words (low quality score)
        boring_penalty = 0.0
        if avg_quality < 0.3:
            boring_penalty = 0.1

        final_score = avg_quality + literary_bonus - boring_penalty

        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, final_score))

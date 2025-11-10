"""
Centralized Document Library for Project Gutenberg Text Management

Provides robust, diverse document retrieval with caching, metadata filtering,
and anti-repetition tracking to solve the "same document" problem across modules.
"""

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from gutenbergpy.textget import get_text_by_id, strip_headers

from .config import DocumentConfig, PerformanceConfig
from .quality_scorer import get_quality_scorer

# Configure logging to be less verbose
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Only show warnings and errors


@dataclass
class DocumentInfo:
    """Information about a retrieved document"""

    id: int
    title: str
    author: str
    length: int
    genre: str
    language: str
    quality_score: float


class DocumentLibrary:
    """Centralized document retrieval and management system"""

    def __init__(self):
        self.good_document_ids: Set[int] = set()
        self.bad_document_ids: Set[int] = set()
        self.recently_used: List[int] = []
        self.document_cache: Dict[int, str] = {}
        self.document_metadata: Dict[int, DocumentInfo] = {}

        # Curated list of known good document ranges
        self.quality_ranges = [
            (1, 1000),  # Early classics
            (2000, 3000),  # Literature
            (5000, 6000),  # Fiction
            (10000, 15000),  # More literature
            (20000, 25000),  # Poetry and drama
            (30000, 35000),  # Modern classics
            (40000, 45000),  # 20th century
            (50000, 55000),  # Contemporary
        ]

        self.max_recent = PerformanceConfig.MAX_RECENT_TRACKING
        self.max_cache = PerformanceConfig.MAX_DOCUMENT_CACHE

    def get_diverse_documents(
        self, count: int = 5, min_length: int = DocumentConfig.MIN_LENGTH_LIBRARY_DEFAULT
    ) -> List[str]:
        """Get multiple diverse documents, ensuring variety"""
        documents = []
        attempts = 0
        max_attempts = count * 10  # Allow for failures

        while len(documents) < count and attempts < max_attempts:
            attempts += 1

            doc = self.get_single_document(
                min_length=min_length, avoid_recent=True, force_different=True
            )
            if doc and len(doc) >= min_length:
                documents.append(doc)

            # Brief pause to avoid overwhelming the API
            time.sleep(PerformanceConfig.API_DELAY_SECONDS)
        return documents

    def get_single_document(
        self,
        min_length: int = DocumentConfig.MIN_LENGTH_GENERAL,
        avoid_recent: bool = True,
        force_different: bool = False,
    ) -> Optional[str]:
        """Get a single document with anti-repetition measures"""

        for _ in range(PerformanceConfig.MAX_PROCESSING_ATTEMPTS):
            try:
                # Choose from quality ranges for better content
                range_start, range_end = random.choice(self.quality_ranges)
                document_id = random.randint(range_start, range_end)

                # Skip if recently used and we want diversity
                if avoid_recent and document_id in self.recently_used:
                    continue

                # Skip known bad documents
                if document_id in self.bad_document_ids:
                    continue

                # Check cache first
                if document_id in self.document_cache:
                    text = self.document_cache[document_id]
                    if len(text) >= min_length:
                        self._track_usage(document_id)
                        return text

                # Fetch new document
                raw_text = get_text_by_id(document_id)
                if not raw_text:
                    self.bad_document_ids.add(document_id)
                    continue

                # Process the text
                text = strip_headers(raw_text)
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="ignore")

                # Clean and validate
                text = self._clean_text(text)

                if len(text) < min_length:
                    self.bad_document_ids.add(document_id)
                    continue

                # Quality scoring (threshold: 0.5)
                quality_score = self._score_text_quality(text)
                if quality_score < 0.5:
                    self.bad_document_ids.add(document_id)
                    continue

                # Success! Cache and track with quality score
                self.good_document_ids.add(document_id)
                self._cache_document(document_id, text, quality_score)
                self._track_usage(document_id)

                return text

            except Exception:
                if "document_id" in locals():
                    self.bad_document_ids.add(document_id)
                continue
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r"\n\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove control characters but keep basic punctuation
        text = re.sub(r'[^\w\s.!?,:;\'"-]', " ", text)

        return text.strip()

    def _extract_metadata(self, text: str, document_id: int) -> tuple[str, str]:
        """Extract title and author from Project Gutenberg text.

        Returns:
            Tuple of (title, author)
        """
        lines = text.split('\n')[:50]  # Check first 50 lines

        title = "Unknown"
        author = "Unknown"

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for title patterns
            if 'Title:' in line:
                title = line.split('Title:', 1)[1].strip()
            elif not title or title == "Unknown":
                # Sometimes title is just on the first non-empty line
                if line and len(line) > 3 and not any(x in line.lower() for x in ['project gutenberg', 'ebook', 'copyright']):
                    if i < 5:  # Only consider first 5 lines for implicit title
                        title = line

            # Look for author patterns
            if 'Author:' in line:
                author = line.split('Author:', 1)[1].strip()
            elif 'by ' in line.lower() and len(line) < 100:
                parts = line.lower().split('by ')
                if len(parts) > 1:
                    potential_author = parts[1].strip()
                    if len(potential_author) > 2 and len(potential_author) < 50:
                        author = potential_author.title()

        # Clean up extracted values
        title = title[:200] if title else f"Document {document_id}"
        author = author[:100] if author else "Unknown"

        return title, author

    def _score_text_quality(self, text: str) -> float:
        """Score text quality using comprehensive metrics (0-1).

        Evaluates:
        - Length adequacy
        - Paragraph structure
        - Word count
        - Non-numeric content
        - Sentence structure quality
        - Vocabulary quality (via QualityScorer)

        Returns:
            Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        scores = []

        # 1. Length score (0-1)
        length_score = min(len(text) / 5000, 1.0)  # Ideal: 5000+ chars
        scores.append(length_score)

        # 2. Paragraph structure score (0-1)
        paragraph_count = len(re.findall(r"\n\n", text))
        paragraph_score = min(paragraph_count / 10, 1.0)  # Ideal: 10+ paragraphs
        scores.append(paragraph_score)

        # 3. Word count score (0-1)
        words = text.split()
        word_count_score = min(len(words) / 2000, 1.0)  # Ideal: 2000+ words
        scores.append(word_count_score)

        # 4. Non-numeric content score (0-1)
        if len(words) >= 100:
            numeric_ratio = len([w for w in words[:100] if any(c.isdigit() for c in w)]) / 100
            non_numeric_score = max(1.0 - (numeric_ratio * 2), 0.0)  # Penalize heavy numeric content
        else:
            non_numeric_score = 0.5
        scores.append(non_numeric_score)

        # 5. Sentence structure score (0-1)
        sentences = re.split(r"[.!?]+", text[:2000])
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            # Ideal range: 10-30 words per sentence
            if 10 <= avg_sentence_length <= 30:
                sentence_score = 1.0
            elif 5 <= avg_sentence_length < 10:
                sentence_score = 0.7
            elif 30 < avg_sentence_length <= 40:
                sentence_score = 0.7
            else:
                sentence_score = 0.3
        else:
            sentence_score = 0.0
        scores.append(sentence_score)

        # 6. Vocabulary quality score (0-1)
        # Sample words from the document to evaluate vocabulary quality
        scorer = get_quality_scorer()
        sample_words = [w.lower() for w in words[:200] if w.isalpha() and len(w) > 3]
        if sample_words:
            vocab_scores = [scorer.score_word(w).overall for w in sample_words[:50]]
            vocab_quality_score = sum(vocab_scores) / len(vocab_scores) if vocab_scores else 0.5
        else:
            vocab_quality_score = 0.5
        scores.append(vocab_quality_score)

        # Return weighted average
        # Give more weight to vocabulary quality and structure
        weights = [0.1, 0.1, 0.1, 0.15, 0.2, 0.35]  # Vocab quality gets 35%
        weighted_score = sum(s * w for s, w in zip(scores, weights))

        return weighted_score

    def _cache_document(self, document_id: int, text: str, quality_score: float):
        """Cache document with metadata and size management.

        Args:
            document_id: Gutenberg document ID
            text: Document text content
            quality_score: Computed quality score (0-1)
        """
        if len(self.document_cache) >= self.max_cache:
            # Remove oldest cached document
            oldest_id = min(self.document_cache.keys())
            del self.document_cache[oldest_id]
            if oldest_id in self.document_metadata:
                del self.document_metadata[oldest_id]

        self.document_cache[document_id] = text

        # Extract and store metadata
        title, author = self._extract_metadata(text, document_id)
        self.document_metadata[document_id] = DocumentInfo(
            id=document_id,
            title=title,
            author=author,
            length=len(text),
            genre="unknown",  # Genre detection could be added later
            language="en",    # Assume English for now
            quality_score=quality_score
        )

    def _track_usage(self, document_id: int):
        """Track recently used documents for diversity"""
        if document_id in self.recently_used:
            self.recently_used.remove(document_id)

        self.recently_used.append(document_id)

        # Keep only recent history
        if len(self.recently_used) > self.max_recent:
            self.recently_used.pop(0)

    def get_document_metadata(self, document_id: int) -> Optional[DocumentInfo]:
        """Get metadata for a specific document.

        Args:
            document_id: The Gutenberg document ID

        Returns:
            DocumentInfo if available, None otherwise
        """
        return self.document_metadata.get(document_id)

    def get_documents_by_quality(self, min_quality: float = 0.0) -> List[DocumentInfo]:
        """Get all cached documents sorted by quality score (highest first).

        Args:
            min_quality: Minimum quality threshold (default 0.0)

        Returns:
            List of DocumentInfo sorted by quality score descending
        """
        filtered = [
            info for info in self.document_metadata.values()
            if info.quality_score >= min_quality
        ]
        return sorted(filtered, key=lambda x: x.quality_score, reverse=True)

    def get_stats(self) -> dict:
        """Get statistics about the document library"""
        quality_scores = [info.quality_score for info in self.document_metadata.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        return {
            "good_documents": len(self.good_document_ids),
            "bad_documents": len(self.bad_document_ids),
            "cached_documents": len(self.document_cache),
            "recently_used": len(self.recently_used),
            "documents_with_metadata": len(self.document_metadata),
            "average_quality_score": round(avg_quality, 3),
            "cache_size_mb": sum(len(text) for text in self.document_cache.values()) / 1024 / 1024,
        }

    def clear_cache(self):
        """Clear all caches (for testing or memory management)"""
        self.document_cache.clear()
        self.recently_used.clear()
        pass  # Cache cleared silently


# Global instance for easy access across modules
document_library = DocumentLibrary()


# Backwards-compatible functions for existing code
def random_gutenberg_document(language_filter="en") -> str:
    """Backwards-compatible function using the new document library"""
    return document_library.get_single_document(min_length=DocumentConfig.MIN_LENGTH_GENERAL) or ""


def get_diverse_gutenberg_documents(
    count: int = 5, min_length: int = DocumentConfig.MIN_LENGTH_LIBRARY_DEFAULT
) -> List[str]:
    """Get multiple diverse documents - use this instead of calling random_gutenberg_document in loops"""
    return document_library.get_diverse_documents(count=count, min_length=min_length)


def get_gutenberg_document_stats() -> dict:
    """Get document library statistics"""
    return document_library.get_stats()

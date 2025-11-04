"""
Centralized Document Library for Project Gutenberg Text Management

Provides robust, diverse document retrieval with caching, metadata filtering,
and anti-repetition tracking to solve the "same document" problem across modules.
"""

import random
import re
import time
from typing import List, Set, Optional, Dict, Tuple
from dataclasses import dataclass
from gutenbergpy.textget import get_text_by_id, strip_headers
import logging

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
            (1, 1000),      # Early classics
            (2000, 3000),   # Literature
            (5000, 6000),   # Fiction
            (10000, 15000), # More literature
            (20000, 25000), # Poetry and drama
            (30000, 35000), # Modern classics
            (40000, 45000), # 20th century
            (50000, 55000)  # Contemporary
        ]

        self.max_recent = 50  # Track last 50 used documents
        self.max_cache = 20   # Cache up to 20 documents

    def get_diverse_documents(self, count: int = 5, min_length: int = 5000) -> List[str]:
        """Get multiple diverse documents, ensuring variety"""
        documents = []
        attempts = 0
        max_attempts = count * 10  # Allow for failures

        while len(documents) < count and attempts < max_attempts:
            attempts += 1

            doc = self.get_single_document(min_length=min_length,
                                         avoid_recent=True,
                                         force_different=True)
            if doc and len(doc) >= min_length:
                documents.append(doc)

            # Brief pause to avoid overwhelming the API
            time.sleep(0.2)
        return documents

    def get_single_document(self, min_length: int = 1000,
                          avoid_recent: bool = True,
                          force_different: bool = False) -> Optional[str]:
        """Get a single document with anti-repetition measures"""

        for attempt in range(20):  # More attempts than original
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
                    text = text.decode('utf-8', errors='ignore')

                # Clean and validate
                text = self._clean_text(text)

                if len(text) < min_length:
                    self.bad_document_ids.add(document_id)
                    continue

                # Quality check
                if not self._is_quality_text(text):
                    self.bad_document_ids.add(document_id)
                    continue

                # Success! Cache and track
                self.good_document_ids.add(document_id)
                self._cache_document(document_id, text)
                self._track_usage(document_id)

                return text

            except Exception as e:
                if 'document_id' in locals():
                    self.bad_document_ids.add(document_id)
                continue
        return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove control characters but keep basic punctuation
        text = re.sub(r'[^\w\s.!?,:;\'"-]', ' ', text)

        return text.strip()

    def _is_quality_text(self, text: str) -> bool:
        """Check if text meets quality standards"""
        # Must have reasonable length
        if len(text) < 1000:
            return False

        # Must have paragraphs (not just lists or tables)
        paragraph_count = len(re.findall(r'\n\n', text))
        if paragraph_count < 5:
            return False

        # Must have reasonable word distribution
        words = text.split()
        if len(words) < 500:
            return False

        # Must not be mostly technical/numeric content
        numeric_ratio = len([w for w in words[:100] if any(c.isdigit() for c in w)]) / 100
        if numeric_ratio > 0.3:
            return False

        # Must have reasonable sentence structure
        sentences = re.split(r'[.!?]+', text[:2000])
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length < 5 or avg_sentence_length > 50:
            return False

        return True

    def _cache_document(self, document_id: int, text: str):
        """Cache document with size management"""
        if len(self.document_cache) >= self.max_cache:
            # Remove oldest cached document
            oldest_id = min(self.document_cache.keys())
            del self.document_cache[oldest_id]

        self.document_cache[document_id] = text

    def _track_usage(self, document_id: int):
        """Track recently used documents for diversity"""
        if document_id in self.recently_used:
            self.recently_used.remove(document_id)

        self.recently_used.append(document_id)

        # Keep only recent history
        if len(self.recently_used) > self.max_recent:
            self.recently_used.pop(0)

    def get_stats(self) -> dict:
        """Get statistics about the document library"""
        return {
            'good_documents': len(self.good_document_ids),
            'bad_documents': len(self.bad_document_ids),
            'cached_documents': len(self.document_cache),
            'recently_used': len(self.recently_used),
            'cache_size_mb': sum(len(text) for text in self.document_cache.values()) / 1024 / 1024
        }

    def clear_cache(self):
        """Clear all caches (for testing or memory management)"""
        self.document_cache.clear()
        self.recently_used.clear()
        pass  # Cache cleared silently


# Global instance for easy access across modules
document_library = DocumentLibrary()


# Backwards-compatible functions for existing code
def random_gutenberg_document(language_filter='en') -> str:
    """Backwards-compatible function using the new document library"""
    return document_library.get_single_document(min_length=1000) or ""


def get_diverse_gutenberg_documents(count: int = 5, min_length: int = 5000) -> List[str]:
    """Get multiple diverse documents - use this instead of calling random_gutenberg_document in loops"""
    return document_library.get_diverse_documents(count=count, min_length=min_length)


def get_gutenberg_document_stats() -> dict:
    """Get document library statistics"""
    return document_library.get_stats()
"""Tests for document_library.py quality scoring and metadata extraction (Phase 7)."""

import pytest

from poetryplayground.core.document_library import DocumentInfo, DocumentLibrary


class TestDocumentQualityScoring:
    """Test document quality scoring functionality."""

    @pytest.fixture
    def library(self):
        """Create a fresh DocumentLibrary instance for each test."""
        return DocumentLibrary()

    def test_score_text_quality_returns_float(self, library):
        """Test that _score_text_quality returns a float score."""
        sample_text = """
        This is a sample document with multiple paragraphs.
        It contains reasonable content for testing purposes.

        This is the second paragraph with more content.
        The text should have reasonable length and structure.

        Third paragraph continues with literary content.
        We need enough text to pass quality checks.

        Fourth paragraph adds more depth and variety.
        Literary works often have diverse vocabulary.

        Fifth paragraph completes the minimum requirements.
        Quality assessment considers multiple factors.

        Sixth paragraph ensures adequate length.
        The document should score reasonably well.
        """

        score = library._score_text_quality(sample_text)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_text_quality_penalizes_short_text(self, library):
        """Test that short texts receive lower scores."""
        short_text = "Too short"
        long_text = " ".join(["quality content"] * 500)

        short_score = library._score_text_quality(short_text)
        long_score = library._score_text_quality(long_text)

        assert short_score < long_score

    def test_score_text_quality_penalizes_numeric_content(self, library):
        """Test that heavily numeric content receives lower scores."""
        numeric_text = " ".join([f"number{i}" for i in range(100)])
        literary_text = """
        The moonlight danced across the quiet waters, casting silver shadows
        upon the ancient stones. In the distance, a nightingale sang its
        melancholy song, echoing through the forgotten gardens where roses
        once bloomed in wild abandon.
        """ * 20

        numeric_score = library._score_text_quality(numeric_text)
        literary_score = library._score_text_quality(literary_text)

        # Literary text should score higher
        assert literary_score > numeric_score

    def test_score_text_quality_evaluates_sentence_structure(self, library):
        """Test that sentence structure affects quality score."""
        good_structure = """
        The ancient library stood silent in the moonlight. Its weathered
        stones told stories of centuries past. Within its walls, countless
        volumes waited patiently for curious readers.

        Each book contained worlds of imagination and knowledge. Some were
        bound in leather, others in cloth. All shared the timeless quality
        of written wisdom.
        """ * 10

        poor_structure = "a b c d e. " * 100  # Very short sentences

        good_score = library._score_text_quality(good_structure)
        poor_score = library._score_text_quality(poor_structure)

        # Good structure should score higher
        assert good_score > poor_score

    def test_score_text_quality_uses_vocabulary_scoring(self, library):
        """Test that vocabulary quality affects the score."""
        # Text with richer vocabulary
        rich_vocab = """
        The penumbra of twilight descended upon the vestige of the ancient
        citadel. Ephemeral shadows danced across weathered ramparts while
        the sibilant wind whispered through desolate corridors.

        In the gloaming, a solitary raven perched upon the crumbling
        parapet, its obsidian plumage gleaming faintly. The bird's
        melancholy cry echoed through the abandoned halls.
        """ * 10

        # Text with common vocabulary
        common_vocab = """
        The dark time came to the old building. Small shadows moved on the
        walls while the wind made sounds in the empty rooms.

        In the evening, a black bird sat on the broken wall. The bird made
        a sad sound in the empty halls.
        """ * 10

        rich_score = library._score_text_quality(rich_vocab)
        common_score = library._score_text_quality(common_vocab)

        # Rich vocabulary should score higher
        assert rich_score > common_score


class TestMetadataExtraction:
    """Test metadata extraction from Project Gutenberg texts."""

    @pytest.fixture
    def library(self):
        """Create a fresh DocumentLibrary instance for each test."""
        return DocumentLibrary()

    def test_extract_metadata_finds_title(self, library):
        """Test that title extraction works."""
        sample_text = """
        Title: Pride and Prejudice

        Author: Jane Austen

        It is a truth universally acknowledged...
        """

        title, author = library._extract_metadata(sample_text, document_id=1)

        assert isinstance(title, str)
        assert isinstance(author, str)
        assert "Pride and Prejudice" in title

    def test_extract_metadata_finds_author(self, library):
        """Test that author extraction works."""
        sample_text = """
        Title: Moby Dick

        Author: Herman Melville

        Call me Ishmael...
        """

        _title, author = library._extract_metadata(sample_text, document_id=1)

        assert "Herman Melville" in author or "melville" in author.lower()

    def test_extract_metadata_handles_missing_fields(self, library):
        """Test that extraction handles missing metadata gracefully."""
        sample_text = "Just some text without metadata headers."

        title, author = library._extract_metadata(sample_text, document_id=42)

        # Should return defaults without crashing
        assert isinstance(title, str)
        assert isinstance(author, str)
        assert len(title) > 0
        assert len(author) > 0

    def test_extract_metadata_handles_by_pattern(self, library):
        """Test extraction of 'by Author' pattern."""
        sample_text = """
        MOBY DICK
        by Herman Melville

        Call me Ishmael...
        """

        title, author = library._extract_metadata(sample_text, document_id=1)

        assert isinstance(title, str)
        assert isinstance(author, str)
        # Author should be extracted from "by" pattern
        assert len(author) > 3


class TestDocumentCachingWithMetadata:
    """Test document caching with quality scores and metadata."""

    @pytest.fixture
    def library(self):
        """Create a fresh DocumentLibrary instance for each test."""
        return DocumentLibrary()

    def test_cache_document_stores_metadata(self, library):
        """Test that caching a document also stores its metadata."""
        sample_text = """
        Title: Test Document

        Author: Test Author

        This is a test document with reasonable content for quality scoring.
        It has multiple paragraphs and sufficient length.

        Second paragraph adds more content.
        Quality should be adequate for testing.
        """ * 20

        quality_score = library._score_text_quality(sample_text)
        library._cache_document(1, sample_text, quality_score)

        # Check that metadata was stored
        metadata = library.get_document_metadata(1)

        assert metadata is not None
        assert isinstance(metadata, DocumentInfo)
        assert metadata.id == 1
        assert metadata.quality_score == quality_score
        assert isinstance(metadata.title, str)
        assert isinstance(metadata.author, str)

    def test_cache_document_stores_quality_score(self, library):
        """Test that quality score is correctly stored in metadata."""
        sample_text = "Test content " * 500
        quality_score = 0.75

        library._cache_document(1, sample_text, quality_score)
        metadata = library.get_document_metadata(1)

        assert metadata is not None
        assert metadata.quality_score == quality_score

    def test_get_document_metadata_returns_none_for_missing(self, library):
        """Test that getting metadata for non-existent document returns None."""
        metadata = library.get_document_metadata(99999)

        assert metadata is None


class TestQualityAwareDocumentRetrieval:
    """Test quality-based document retrieval methods."""

    @pytest.fixture
    def library_with_docs(self):
        """Create a library with some cached documents."""
        library = DocumentLibrary()

        # Cache some documents with different quality scores
        for i, score in enumerate([0.9, 0.6, 0.8, 0.4, 0.7], start=1):
            text = f"Document {i} content " * 500
            library._cache_document(i, text, score)

        return library

    def test_get_documents_by_quality_returns_sorted_list(self, library_with_docs):
        """Test that documents are returned sorted by quality."""
        docs = library_with_docs.get_documents_by_quality()

        assert isinstance(docs, list)
        assert len(docs) == 5

        # Should be sorted by quality descending
        for i in range(len(docs) - 1):
            assert docs[i].quality_score >= docs[i + 1].quality_score

    def test_get_documents_by_quality_filters_by_threshold(self, library_with_docs):
        """Test that quality threshold filters correctly."""
        # Get only high-quality documents (>= 0.7)
        high_quality_docs = library_with_docs.get_documents_by_quality(min_quality=0.7)

        assert all(doc.quality_score >= 0.7 for doc in high_quality_docs)
        assert len(high_quality_docs) == 3  # Documents with scores 0.9, 0.8, 0.7

    def test_get_documents_by_quality_empty_for_high_threshold(self, library_with_docs):
        """Test that very high threshold returns empty list."""
        perfect_docs = library_with_docs.get_documents_by_quality(min_quality=1.0)

        assert isinstance(perfect_docs, list)
        assert len(perfect_docs) == 0  # No documents with perfect score


class TestDocumentLibraryStats:
    """Test enhanced statistics with quality information."""

    @pytest.fixture
    def library_with_docs(self):
        """Create a library with some cached documents."""
        library = DocumentLibrary()

        for i, score in enumerate([0.8, 0.6, 0.7, 0.9], start=1):
            text = f"Document {i} " * 500
            library._cache_document(i, text, score)
            library.good_document_ids.add(i)

        return library

    def test_get_stats_includes_quality_metrics(self, library_with_docs):
        """Test that stats include quality information."""
        stats = library_with_docs.get_stats()

        assert "average_quality_score" in stats
        assert "documents_with_metadata" in stats

        # Check average quality is calculated correctly
        # (0.8 + 0.6 + 0.7 + 0.9) / 4 = 0.75
        assert stats["average_quality_score"] == pytest.approx(0.75, abs=0.01)
        assert stats["documents_with_metadata"] == 4

    def test_get_stats_handles_empty_library(self):
        """Test that stats work with empty library."""
        library = DocumentLibrary()
        stats = library.get_stats()

        assert stats["average_quality_score"] == 0.0
        assert stats["documents_with_metadata"] == 0


class TestDocumentInfoDataclass:
    """Test DocumentInfo dataclass structure."""

    def test_document_info_creation(self):
        """Test creating DocumentInfo instance."""
        doc_info = DocumentInfo(
            id=1,
            title="Test Title",
            author="Test Author",
            length=1000,
            genre="fiction",
            language="en",
            quality_score=0.85
        )

        assert doc_info.id == 1
        assert doc_info.title == "Test Title"
        assert doc_info.author == "Test Author"
        assert doc_info.length == 1000
        assert doc_info.genre == "fiction"
        assert doc_info.language == "en"
        assert doc_info.quality_score == 0.85

    def test_document_info_has_all_fields(self):
        """Test that DocumentInfo has expected fields."""
        doc_info = DocumentInfo(
            id=1,
            title="Title",
            author="Author",
            length=100,
            genre="poetry",
            language="en",
            quality_score=0.5
        )

        # All required fields should exist
        assert hasattr(doc_info, "id")
        assert hasattr(doc_info, "title")
        assert hasattr(doc_info, "author")
        assert hasattr(doc_info, "length")
        assert hasattr(doc_info, "genre")
        assert hasattr(doc_info, "language")
        assert hasattr(doc_info, "quality_score")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

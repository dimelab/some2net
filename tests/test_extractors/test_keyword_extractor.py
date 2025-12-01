"""
Tests for KeywordExtractor using RAKE (Rapid Automatic Keyword Extraction).
"""

import pytest
from src.core.extractors import KeywordExtractor


class TestKeywordExtractorBasic:
    """Test basic functionality of KeywordExtractor."""

    def test_initialization_default(self):
        """Test extractor initialization with defaults."""
        extractor = KeywordExtractor()

        assert extractor.get_extractor_type() == 'keyword'
        assert extractor.min_keywords == 5
        assert extractor.max_keywords == 20
        assert extractor.language == 'english'
        assert extractor.max_phrase_length == 3
        assert extractor.min_phrase_length == 1

    def test_initialization_custom(self):
        """Test extractor initialization with custom parameters."""
        extractor = KeywordExtractor(
            min_keywords=3,
            max_keywords=10,
            language='danish',
            max_phrase_length=4,
            min_phrase_length=2
        )

        assert extractor.min_keywords == 3
        assert extractor.max_keywords == 10
        assert extractor.language == 'danish'
        assert extractor.max_phrase_length == 4
        assert extractor.min_phrase_length == 2

    def test_collect_single_text(self):
        """Test collecting a single text for an author."""
        extractor = KeywordExtractor()

        extractor.collect_text("@user1", "machine learning and data science")

        assert extractor.get_author_count() == 1
        assert extractor.get_text_count("@user1") == 1

    def test_collect_multiple_texts(self):
        """Test collecting multiple texts for an author."""
        extractor = KeywordExtractor()

        texts = [
            "machine learning algorithms",
            "deep learning neural networks",
            "artificial intelligence systems"
        ]
        extractor.collect_texts("@user1", texts)

        assert extractor.get_author_count() == 1
        assert extractor.get_text_count("@user1") == 3

    def test_collect_multiple_authors(self):
        """Test collecting texts for multiple authors."""
        extractor = KeywordExtractor()

        extractor.collect_texts("@user1", ["text1", "text2"])
        extractor.collect_texts("@user2", ["text3", "text4", "text5"])

        assert extractor.get_author_count() == 2
        assert extractor.get_text_count("@user1") == 2
        assert extractor.get_text_count("@user2") == 3
        assert extractor.get_text_count() == 5


class TestKeywordExtraction:
    """Test keyword extraction functionality."""

    def test_extract_per_author_basic(self):
        """Test extracting keywords for a single author."""
        extractor = KeywordExtractor(min_keywords=3, max_keywords=5)

        texts = [
            "machine learning and artificial intelligence",
            "deep learning neural networks for computer vision",
            "machine learning algorithms for data analysis"
        ]
        extractor.collect_texts("@user1", texts)

        keywords = extractor.extract_per_author("@user1")

        assert isinstance(keywords, list)
        assert len(keywords) >= 3
        assert len(keywords) <= 5

        # Check keyword structure
        for kw in keywords:
            assert 'text' in kw
            assert 'type' in kw
            assert 'score' in kw
            assert kw['type'] == 'KEYWORD'
            assert 0 <= kw['score'] <= 1

    def test_extract_per_author_empty(self):
        """Test extracting keywords for author with no texts."""
        extractor = KeywordExtractor()

        keywords = extractor.extract_per_author("@nonexistent")

        assert keywords == []

    def test_extract_all_authors(self):
        """Test extracting keywords for all authors."""
        extractor = KeywordExtractor(min_keywords=2, max_keywords=5)

        extractor.collect_texts("@user1", [
            "machine learning data science",
            "artificial intelligence neural networks"
        ])
        extractor.collect_texts("@user2", [
            "climate change environmental protection",
            "renewable energy sustainable development"
        ])

        results = extractor.extract_all_authors(show_progress=False)

        assert len(results) == 2
        assert '@user1' in results
        assert '@user2' in results

        # User1 should have ML/AI related keywords
        user1_keywords = [kw['text'] for kw in results['@user1']]
        assert len(user1_keywords) >= 2

        # User2 should have climate/energy related keywords
        user2_keywords = [kw['text'] for kw in results['@user2']]
        assert len(user2_keywords) >= 2

    def test_keyword_relevance(self):
        """Test that extracted keywords are relevant to content."""
        extractor = KeywordExtractor(min_keywords=3, max_keywords=8)

        # Text about programming
        extractor.collect_texts("@programmer", [
            "Python programming language for web development",
            "JavaScript framework for frontend applications",
            "Python libraries for data science and machine learning",
            "Programming best practices and code quality"
        ])

        keywords = extractor.extract_per_author("@programmer")
        keyword_texts = [kw['text'].lower() for kw in keywords]

        # Should extract programming-related terms
        # At least one of these should be in top keywords
        relevant_terms = ['python', 'programming', 'javascript', 'data', 'machine', 'learning']
        has_relevant = any(
            any(term in keyword for term in relevant_terms)
            for keyword in keyword_texts
        )
        assert has_relevant

    def test_phrase_extraction(self):
        """Test that multi-word phrases are extracted."""
        extractor = KeywordExtractor(
            min_keywords=3,
            max_keywords=10,
            max_phrase_length=3  # Allow up to 3-word phrases
        )

        extractor.collect_texts("@user1", [
            "machine learning is amazing",
            "deep learning neural networks",
            "machine learning algorithms"
        ])

        keywords = extractor.extract_per_author("@user1")
        keyword_texts = [kw['text'] for kw in keywords]

        # RAKE should extract multi-word phrases
        phrases = [kw for kw in keyword_texts if ' ' in kw]
        assert len(phrases) > 0

        # Should extract relevant phrases
        assert any('machine learning' in kw or 'deep learning' in kw for kw in keyword_texts)


class TestKeywordExtractorEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_from_text_not_supported(self):
        """Test that extract_from_text raises NotImplementedError."""
        extractor = KeywordExtractor()

        with pytest.raises(NotImplementedError):
            extractor.extract_from_text("some text")

    def test_extract_batch_not_supported(self):
        """Test that extract_batch raises NotImplementedError."""
        extractor = KeywordExtractor()

        with pytest.raises(NotImplementedError):
            extractor.extract_batch(["text1", "text2"])

    def test_collect_empty_author(self):
        """Test collecting texts with empty author name."""
        extractor = KeywordExtractor()

        extractor.collect_text("", "some text")

        assert extractor.get_author_count() == 0

    def test_collect_empty_texts(self):
        """Test collecting empty texts."""
        extractor = KeywordExtractor()

        extractor.collect_texts("@user1", ["", "  ", None])

        # Should filter out empty texts
        assert extractor.get_text_count("@user1") == 0

    def test_extract_from_short_text(self):
        """Test extracting keywords from very short text."""
        extractor = KeywordExtractor(min_keywords=5, max_keywords=10)

        extractor.collect_texts("@user1", ["hello world"])

        keywords = extractor.extract_per_author("@user1")

        # Should return fewer keywords than min_keywords for short text
        # But should not error
        assert isinstance(keywords, list)

    def test_clear_collected_texts(self):
        """Test clearing collected texts."""
        extractor = KeywordExtractor()

        extractor.collect_texts("@user1", ["text1", "text2"])
        extractor.collect_texts("@user2", ["text3"])

        assert extractor.get_author_count() == 2

        extractor.clear()

        assert extractor.get_author_count() == 0
        assert extractor.get_text_count() == 0


class TestKeywordExtractorConfig:
    """Test configuration and metadata methods."""

    def test_get_config(self):
        """Test get_config returns correct configuration."""
        extractor = KeywordExtractor(
            min_keywords=3,
            max_keywords=15,
            language='danish',
            max_phrase_length=4
        )

        config = extractor.get_config()

        assert config['type'] == 'keyword'
        assert config['algorithm'] == 'RAKE'
        assert config['min_keywords'] == 3
        assert config['max_keywords'] == 15
        assert config['language'] == 'danish'
        assert config['max_phrase_length'] == 4

    def test_get_extractor_type(self):
        """Test get_extractor_type returns 'keyword'."""
        extractor = KeywordExtractor()

        assert extractor.get_extractor_type() == 'keyword'


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

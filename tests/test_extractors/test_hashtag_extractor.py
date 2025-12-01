"""
Unit tests for HashtagExtractor.
"""

import pytest
from src.core.extractors import HashtagExtractor


class TestHashtagExtractor:
    """Test suite for HashtagExtractor."""

    def test_single_hashtag(self):
        """Test extraction of a single hashtag."""
        extractor = HashtagExtractor()
        text = "I love #python programming!"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '#python'
        assert result[0]['type'] == 'HASHTAG'
        assert result[0]['score'] == 1.0

    def test_multiple_hashtags(self):
        """Test extraction of multiple hashtags."""
        extractor = HashtagExtractor()
        text = "Learning #python and #machinelearning is fun! #AI"
        result = extractor.extract_from_text(text)

        assert len(result) == 3
        hashtags = [r['text'] for r in result]
        assert '#python' in hashtags
        assert '#machinelearning' in hashtags
        assert '#ai' in hashtags

    def test_case_normalization(self):
        """Test case normalization."""
        extractor = HashtagExtractor(normalize_case=True)
        text = "#Python #PYTHON #python"
        result = extractor.extract_from_text(text)

        # Should deduplicate to single lowercase version
        assert len(result) == 1
        assert result[0]['text'] == '#python'

    def test_case_preservation(self):
        """Test case preservation when normalization is off."""
        extractor = HashtagExtractor(normalize_case=False)
        text = "#Python #PYTHON #python"
        result = extractor.extract_from_text(text)

        # Should keep all three
        assert len(result) == 3
        hashtags = [r['text'] for r in result]
        assert '#Python' in hashtags
        assert '#PYTHON' in hashtags
        assert '#python' in hashtags

    def test_unicode_hashtags(self):
        """Test extraction of Unicode hashtags."""
        extractor = HashtagExtractor()
        text = "Testing #日本語 and #français hashtags"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        hashtags = [r['text'] for r in result]
        assert '#日本語' in hashtags
        assert '#français' in hashtags

    def test_empty_text(self):
        """Test handling of empty text."""
        extractor = HashtagExtractor()
        result = extractor.extract_from_text("")
        assert result == []

    def test_none_text(self):
        """Test handling of None text."""
        extractor = HashtagExtractor()
        result = extractor.extract_from_text(None)
        assert result == []

    def test_no_hashtags(self):
        """Test text with no hashtags."""
        extractor = HashtagExtractor()
        text = "This text has no hashtags at all."
        result = extractor.extract_from_text(text)
        assert result == []

    def test_duplicate_hashtags_in_same_text(self):
        """Test that duplicate hashtags in same text are deduplicated."""
        extractor = HashtagExtractor()
        text = "I love #python and #python is great! #python rocks!"
        result = extractor.extract_from_text(text)

        # Should only return one instance
        assert len(result) == 1
        assert result[0]['text'] == '#python'

    def test_hashtag_with_numbers(self):
        """Test hashtags containing numbers."""
        extractor = HashtagExtractor()
        text = "Looking forward to #COVID19 ending and #2023goals"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        hashtags = [r['text'] for r in result]
        assert '#covid19' in hashtags
        assert '#2023goals' in hashtags

    def test_batch_extraction(self):
        """Test batch extraction."""
        extractor = HashtagExtractor()
        texts = [
            "I love #python",
            "Learning #machinelearning",
            "No hashtags here"
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[0]) == 1
        assert results[0][0]['text'] == '#python'
        assert len(results[1]) == 1
        assert results[1][0]['text'] == '#machinelearning'
        assert len(results[2]) == 0

    def test_get_extractor_type(self):
        """Test extractor type identifier."""
        extractor = HashtagExtractor()
        assert extractor.get_extractor_type() == 'hashtag'

    def test_get_config(self):
        """Test configuration retrieval."""
        extractor = HashtagExtractor(normalize_case=False)
        config = extractor.get_config()

        assert 'normalize_case' in config
        assert config['normalize_case'] is False

    def test_hashtag_at_start(self):
        """Test hashtag at the start of text."""
        extractor = HashtagExtractor()
        text = "#python is great"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '#python'

    def test_hashtag_at_end(self):
        """Test hashtag at the end of text."""
        extractor = HashtagExtractor()
        text = "I love programming with #python"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '#python'

    def test_hashtag_with_underscores(self):
        """Test hashtags with underscores."""
        extractor = HashtagExtractor()
        text = "Check out #machine_learning and #data_science"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        hashtags = [r['text'] for r in result]
        assert '#machine_learning' in hashtags
        assert '#data_science' in hashtags

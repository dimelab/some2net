"""
Unit tests for ExactMatchExtractor.
"""

import pytest
from src.core.extractors import ExactMatchExtractor


class TestExactMatchExtractor:
    """Test suite for ExactMatchExtractor."""

    def test_simple_text(self):
        """Test extraction of simple text."""
        extractor = ExactMatchExtractor()
        text = "Hello world"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'Hello world'
        assert result[0]['type'] == 'EXACT'
        assert result[0]['score'] == 1.0

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        extractor = ExactMatchExtractor(strip_whitespace=True)
        text = "  Hello world  "
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'Hello world'

    def test_keep_whitespace(self):
        """Test keeping whitespace when requested."""
        extractor = ExactMatchExtractor(strip_whitespace=False)
        text = "  Hello world  "
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '  Hello world  '

    def test_empty_text(self):
        """Test handling of empty text."""
        extractor = ExactMatchExtractor()
        result = extractor.extract_from_text("")
        assert result == []

    def test_none_text(self):
        """Test handling of None text."""
        extractor = ExactMatchExtractor()
        result = extractor.extract_from_text(None)
        assert result == []

    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        extractor = ExactMatchExtractor(strip_whitespace=True)
        result = extractor.extract_from_text("   ")
        assert result == []

    def test_multiline_text(self):
        """Test handling of multiline text."""
        extractor = ExactMatchExtractor()
        text = "Line 1\nLine 2\nLine 3"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == text

    def test_special_characters(self):
        """Test handling of special characters."""
        extractor = ExactMatchExtractor()
        text = "Hello! @user #hashtag https://example.com"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == text

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        extractor = ExactMatchExtractor()
        text = "こんにちは世界 Bonjour le monde"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == text

    def test_numeric_text(self):
        """Test handling of numeric text."""
        extractor = ExactMatchExtractor()
        text = "12345"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '12345'

    def test_very_long_text(self):
        """Test handling of very long text."""
        extractor = ExactMatchExtractor()
        text = "A" * 1000
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == text

    def test_batch_extraction(self):
        """Test batch extraction."""
        extractor = ExactMatchExtractor()
        texts = [
            "Text 1",
            "Text 2",
            "Text 3"
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[0]) == 1
        assert results[0][0]['text'] == 'Text 1'
        assert len(results[1]) == 1
        assert results[1][0]['text'] == 'Text 2'
        assert len(results[2]) == 1
        assert results[2][0]['text'] == 'Text 3'

    def test_batch_with_empty_texts(self):
        """Test batch extraction with some empty texts."""
        extractor = ExactMatchExtractor()
        texts = [
            "Text 1",
            "",
            "Text 3"
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[0]) == 1
        assert len(results[1]) == 0  # Empty text
        assert len(results[2]) == 1

    def test_get_extractor_type(self):
        """Test extractor type identifier."""
        extractor = ExactMatchExtractor()
        assert extractor.get_extractor_type() == 'exact'

    def test_get_config(self):
        """Test configuration retrieval."""
        extractor = ExactMatchExtractor(strip_whitespace=False)
        config = extractor.get_config()

        assert 'strip_whitespace' in config
        assert config['strip_whitespace'] is False

    def test_categorical_data(self):
        """Test use case with categorical data."""
        extractor = ExactMatchExtractor()
        categories = ["positive", "negative", "neutral"]
        results = extractor.extract_batch(categories)

        assert len(results) == 3
        assert results[0][0]['text'] == 'positive'
        assert results[1][0]['text'] == 'negative'
        assert results[2][0]['text'] == 'neutral'

    def test_sentiment_labels(self):
        """Test use case with sentiment labels."""
        extractor = ExactMatchExtractor()
        text = "Positive"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'Positive'
        assert result[0]['type'] == 'EXACT'

    def test_newline_handling(self):
        """Test that newlines are preserved."""
        extractor = ExactMatchExtractor(strip_whitespace=False)
        text = "\nHello\n"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == '\nHello\n'

    def test_newline_stripping(self):
        """Test that leading/trailing newlines are stripped when requested."""
        extractor = ExactMatchExtractor(strip_whitespace=True)
        text = "\nHello\n"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'Hello'

    def test_tabs_and_spaces(self):
        """Test handling of tabs and spaces."""
        extractor = ExactMatchExtractor(strip_whitespace=True)
        text = "\t  Hello  \t"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'Hello'

    def test_internal_whitespace_preserved(self):
        """Test that internal whitespace is always preserved."""
        extractor = ExactMatchExtractor(strip_whitespace=True)
        text = "  Hello   world  "
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        # Internal whitespace should be preserved, only leading/trailing stripped
        assert result[0]['text'] == 'Hello   world'

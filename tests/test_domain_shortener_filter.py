"""
Test that domain extractor filters out URL shorteners.
"""

import pytest
from src.core.extractors import DomainExtractor


class TestDomainShortenerFilter:
    """Test URL shortener filtering."""

    def test_filters_tco_links(self):
        """Test that t.co links are filtered out."""
        extractor = DomainExtractor(filter_shorteners=True)

        text = "Check this out https://t.co/abc123def"
        results = extractor.extract_from_text(text)

        # Should return empty - t.co is filtered
        assert len(results) == 0

    def test_filters_bitly_links(self):
        """Test that bit.ly links are filtered out."""
        extractor = DomainExtractor(filter_shorteners=True)

        text = "Click here https://bit.ly/xyz123"
        results = extractor.extract_from_text(text)

        assert len(results) == 0

    def test_filters_multiple_shorteners(self):
        """Test that multiple shorteners are filtered."""
        extractor = DomainExtractor(filter_shorteners=True)

        text = "Links: https://t.co/abc https://goo.gl/xyz https://ow.ly/def"
        results = extractor.extract_from_text(text)

        assert len(results) == 0

    def test_keeps_real_domains(self):
        """Test that real domains are kept."""
        extractor = DomainExtractor(filter_shorteners=True)

        text = "Visit https://www.dr.dk/news/article"
        results = extractor.extract_from_text(text)

        assert len(results) == 1
        assert results[0]['text'] == 'dr.dk'  # www stripped by default

    def test_mixed_shortened_and_real(self):
        """Test that only real domains are kept when mixed."""
        extractor = DomainExtractor(filter_shorteners=True)

        text = "Shortened https://t.co/abc and real https://www.nytimes.com/article"
        results = extractor.extract_from_text(text)

        # Should only have nytimes.com
        assert len(results) == 1
        assert results[0]['text'] == 'nytimes.com'

    def test_filter_can_be_disabled(self):
        """Test that filtering can be disabled."""
        extractor = DomainExtractor(filter_shorteners=False)

        text = "Link https://t.co/abc123"
        results = extractor.extract_from_text(text)

        # Should include t.co when filtering is disabled
        assert len(results) == 1
        assert results[0]['text'] == 't.co'

    def test_all_common_shorteners_blocked(self):
        """Test that all common shorteners are blocked."""
        extractor = DomainExtractor(filter_shorteners=True)

        shorteners = [
            'https://t.co/abc',
            'https://bit.ly/xyz',
            'https://tinyurl.com/abc',
            'https://goo.gl/xyz',
            'https://ow.ly/abc',
            'https://is.gd/xyz',
            'https://youtu.be/abc',
            'https://lnkd.in/xyz'
        ]

        for url in shorteners:
            results = extractor.extract_from_text(url)
            assert len(results) == 0, f"Failed to filter {url}"

    def test_config_shows_filter_setting(self):
        """Test that config shows filter setting."""
        extractor = DomainExtractor(filter_shorteners=True)

        config = extractor.get_config()

        assert 'filter_shorteners' in config
        assert config['filter_shorteners'] is True
        assert 'num_shorteners_blocked' in config
        assert config['num_shorteners_blocked'] > 20  # Should have 25+


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

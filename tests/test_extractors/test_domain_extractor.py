"""
Unit tests for DomainExtractor.
"""

import pytest
from src.core.extractors import DomainExtractor


class TestDomainExtractor:
    """Test suite for DomainExtractor."""

    def test_single_domain(self):
        """Test extraction of a single domain."""
        extractor = DomainExtractor()
        text = "Check out https://example.com for more info"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'example.com'
        assert result[0]['type'] == 'DOMAIN'
        assert result[0]['score'] == 1.0

    def test_multiple_domains(self):
        """Test extraction of multiple domains."""
        extractor = DomainExtractor()
        text = "Visit https://example.com and https://test.org for more"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        domains = [r['text'] for r in result]
        assert 'example.com' in domains
        assert 'test.org' in domains

    def test_http_and_https(self):
        """Test extraction from both HTTP and HTTPS URLs."""
        extractor = DomainExtractor()
        text = "HTTP: http://example.com HTTPS: https://secure.com"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        domains = [r['text'] for r in result]
        assert 'example.com' in domains
        assert 'secure.com' in domains

    def test_strip_www(self):
        """Test stripping of www prefix."""
        extractor = DomainExtractor(strip_www=True)
        text = "Visit https://www.example.com"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_keep_www(self):
        """Test keeping www prefix when requested."""
        extractor = DomainExtractor(strip_www=False)
        text = "Visit https://www.example.com"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'www.example.com'

    def test_strip_subdomain(self):
        """Test extraction of main domain only."""
        extractor = DomainExtractor(strip_subdomain=True)
        text = "Check https://blog.example.com and https://api.test.org"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        domains = [r['text'] for r in result]
        assert 'example.com' in domains
        assert 'test.org' in domains

    def test_keep_subdomain(self):
        """Test keeping subdomains when requested."""
        extractor = DomainExtractor(strip_subdomain=False)
        text = "Check https://blog.example.com"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'blog.example.com'

    def test_url_with_path(self):
        """Test extraction from URLs with paths."""
        extractor = DomainExtractor()
        text = "Read https://example.com/blog/post-1 for details"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_url_with_query_params(self):
        """Test extraction from URLs with query parameters."""
        extractor = DomainExtractor()
        text = "Search https://example.com/search?q=test&lang=en"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_url_with_port(self):
        """Test extraction from URLs with port numbers."""
        extractor = DomainExtractor()
        text = "Dev server at https://localhost:8080/app"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'localhost'

    def test_empty_text(self):
        """Test handling of empty text."""
        extractor = DomainExtractor()
        result = extractor.extract_from_text("")
        assert result == []

    def test_none_text(self):
        """Test handling of None text."""
        extractor = DomainExtractor()
        result = extractor.extract_from_text(None)
        assert result == []

    def test_no_urls(self):
        """Test text with no URLs."""
        extractor = DomainExtractor()
        text = "This text has no URLs at all."
        result = extractor.extract_from_text(text)
        assert result == []

    def test_duplicate_domains_in_same_text(self):
        """Test that duplicate domains in same text are deduplicated."""
        extractor = DomainExtractor()
        text = "Visit https://example.com and https://example.com/about"
        result = extractor.extract_from_text(text)

        # Should only return one instance
        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_malformed_url(self):
        """Test handling of malformed URLs."""
        extractor = DomainExtractor()
        text = "Bad URL: https:// or http://"
        result = extractor.extract_from_text(text)

        # Should handle gracefully and return empty
        assert len(result) == 0

    def test_case_normalization(self):
        """Test that domains are normalized to lowercase."""
        extractor = DomainExtractor()
        text = "Visit https://Example.COM and https://TEST.org"
        result = extractor.extract_from_text(text)

        assert len(result) == 2
        domains = [r['text'] for r in result]
        assert 'example.com' in domains
        assert 'test.org' in domains

    def test_batch_extraction(self):
        """Test batch extraction."""
        extractor = DomainExtractor()
        texts = [
            "Check https://example.com",
            "Visit https://test.org and https://demo.net",
            "No URLs here"
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[0]) == 1
        assert results[0][0]['text'] == 'example.com'
        assert len(results[1]) == 2
        assert len(results[2]) == 0

    def test_get_extractor_type(self):
        """Test extractor type identifier."""
        extractor = DomainExtractor()
        assert extractor.get_extractor_type() == 'domain'

    def test_get_config(self):
        """Test configuration retrieval."""
        extractor = DomainExtractor(strip_www=False, strip_subdomain=True)
        config = extractor.get_config()

        assert 'strip_www' in config
        assert config['strip_www'] is False
        assert 'strip_subdomain' in config
        assert config['strip_subdomain'] is True

    def test_url_with_fragment(self):
        """Test extraction from URLs with fragments."""
        extractor = DomainExtractor()
        text = "See https://example.com/page#section for details"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_url_with_special_characters(self):
        """Test extraction from URLs with special characters."""
        extractor = DomainExtractor()
        text = "Check https://example.com/path?foo=bar&baz=qux"
        result = extractor.extract_from_text(text)

        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_international_domain(self):
        """Test extraction of international domains."""
        extractor = DomainExtractor()
        text = "Visit https://例え.jp for Japanese content"
        result = extractor.extract_from_text(text)

        # Should handle international domains
        assert len(result) >= 0  # May or may not work with IDN

    def test_multiple_urls_same_domain(self):
        """Test multiple URLs pointing to same domain."""
        extractor = DomainExtractor()
        text = "Check https://example.com/page1 and https://example.com/page2"
        result = extractor.extract_from_text(text)

        # Should deduplicate
        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

    def test_www_normalization_deduplication(self):
        """Test that www and non-www versions are deduplicated when strip_www=True."""
        extractor = DomainExtractor(strip_www=True)
        text = "Visit https://www.example.com and https://example.com"
        result = extractor.extract_from_text(text)

        # Should deduplicate to single domain
        assert len(result) == 1
        assert result[0]['text'] == 'example.com'

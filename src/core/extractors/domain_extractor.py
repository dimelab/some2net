"""
Domain extractor for social media text.

Extracts domains from URLs in text and creates edges from authors to domains.
"""

import re
from typing import List, Dict
from urllib.parse import urlparse
from .base_extractor import BaseExtractor


class DomainExtractor(BaseExtractor):
    """Extract domains from URLs in text."""

    def __init__(self, strip_www: bool = True, strip_subdomain: bool = False):
        """
        Initialize domain extractor.

        Args:
            strip_www: If True, removes 'www.' prefix from domains
            strip_subdomain: If True, extracts only the main domain (e.g., example.com from sub.example.com)
        """
        self.strip_www = strip_www
        self.strip_subdomain = strip_subdomain

    def _extract_main_domain(self, domain: str) -> str:
        """
        Extract the main domain from a full domain.

        Args:
            domain: Full domain (e.g., 'sub.example.com')

        Returns:
            str: Main domain (e.g., 'example.com')
        """
        # Simple heuristic: take last two parts
        # This won't work perfectly for all TLDs (e.g., .co.uk) but is good enough
        parts = domain.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return domain

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Extract domains from URLs in a single text.

        Args:
            text: Input text to extract domains from
            **kwargs: Additional parameters (unused)

        Returns:
            List[Dict]: List of extracted domains with metadata
        """
        if not text:
            return []

        # Pattern for URLs (http/https)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)

        results = []
        seen_domains = set()  # Track unique domains in this text

        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc

                if not domain:
                    continue

                # Remove port number if present
                if ':' in domain:
                    domain = domain.split(':')[0]

                # Strip www. if requested
                if self.strip_www and domain.lower().startswith('www.'):
                    domain = domain[4:]

                # Extract main domain if requested
                if self.strip_subdomain:
                    domain = self._extract_main_domain(domain)

                # Normalize to lowercase
                domain = domain.lower()

                # Avoid duplicates within the same text
                if domain and domain not in seen_domains:
                    results.append({
                        'text': domain,
                        'type': 'DOMAIN',
                        'score': 1.0
                    })
                    seen_domains.add(domain)

            except Exception:
                # Skip malformed URLs
                continue

        return results

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Extract domains from a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Processing batch size (unused for this simple extractor)
            show_progress: Show progress bar (unused for this simple extractor)
            **kwargs: Additional parameters

        Returns:
            List[List[Dict]]: List of extraction results, one per input text
        """
        return [self.extract_from_text(text, **kwargs) for text in texts]

    def get_extractor_type(self) -> str:
        """
        Return extractor type identifier.

        Returns:
            str: 'domain'
        """
        return 'domain'

    def get_config(self) -> Dict:
        """
        Return extractor configuration.

        Returns:
            Dict: Configuration parameters
        """
        return {
            'strip_www': self.strip_www,
            'strip_subdomain': self.strip_subdomain
        }

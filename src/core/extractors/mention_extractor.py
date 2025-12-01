"""
Mention extractor for social media text.

Extracts user mentions (e.g., @username) from text and creates
edges from authors to mentioned users.
"""

import re
from typing import List, Dict
from .base_extractor import BaseExtractor


class MentionExtractor(BaseExtractor):
    """Extract user mentions from text."""

    def __init__(self, normalize_case: bool = True, exclude_emails: bool = True):
        """
        Initialize mention extractor.

        Args:
            normalize_case: If True, converts mentions to lowercase
            exclude_emails: If True, filters out email addresses
        """
        self.normalize_case = normalize_case
        self.exclude_emails = exclude_emails

    def _is_likely_email(self, text: str) -> bool:
        """
        Check if a string is likely an email address.

        Args:
            text: Text to check

        Returns:
            bool: True if likely an email
        """
        # Simple heuristic: if there's a dot after the @, likely an email
        if '.' in text:
            # Check if there's a dot in the domain part
            parts = text.split('@')
            if len(parts) == 2 and '.' in parts[1]:
                return True
        return False

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Extract mentions from a single text.

        Args:
            text: Input text to extract mentions from
            **kwargs: Additional parameters (unused)

        Returns:
            List[Dict]: List of extracted mentions with metadata
        """
        if not text:
            return []

        # Pattern: @ followed by word characters (Unicode-aware)
        # Matches @username but not email@domain.com if exclude_emails is True
        pattern = r'@(\w+)'
        mentions = re.findall(pattern, text, re.UNICODE)

        results = []
        seen = set()  # Track unique mentions in this text

        for mention in mentions:
            # Check if this looks like an email address (before processing)
            if self.exclude_emails:
                # Look for the mention in the original text with context
                # to check for email patterns
                context_pattern = rf'@{re.escape(mention)}[^\s]*'
                matches = re.findall(context_pattern, text, re.UNICODE)
                if matches and any(self._is_likely_email(m) for m in matches):
                    continue

            # Remove @ symbol and normalize
            # This allows matching between authors and mentioned users
            if self.normalize_case:
                mention_text = mention.lower()
            else:
                mention_text = mention

            # Avoid duplicates within the same text
            if mention_text not in seen:
                results.append({
                    'text': mention_text,
                    'type': 'MENTION',
                    'score': 1.0
                })
                seen.add(mention_text)

        return results

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Extract mentions from a batch of texts.

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
            str: 'mention'
        """
        return 'mention'

    def get_config(self) -> Dict:
        """
        Return extractor configuration.

        Returns:
            Dict: Configuration parameters
        """
        return {
            'normalize_case': self.normalize_case,
            'exclude_emails': self.exclude_emails
        }

"""
Hashtag extractor for social media text.

Extracts hashtags (e.g., #python, #machinelearning) from text and creates
edges from authors to hashtags.
"""

import re
from typing import List, Dict
from .base_extractor import BaseExtractor


class HashtagExtractor(BaseExtractor):
    """Extract hashtags from text."""

    def __init__(self, normalize_case: bool = True):
        """
        Initialize hashtag extractor.

        Args:
            normalize_case: If True, converts hashtags to lowercase
        """
        self.normalize_case = normalize_case

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Extract hashtags from a single text.

        Args:
            text: Input text to extract hashtags from
            **kwargs: Additional parameters (unused)

        Returns:
            List[Dict]: List of extracted hashtags with metadata
        """
        if not text:
            return []

        # Pattern: # followed by word characters (Unicode-aware)
        # Supports international characters in hashtags
        pattern = r'#(\w+)'
        hashtags = re.findall(pattern, text, re.UNICODE)

        results = []
        seen = set()  # Track unique hashtags in this text

        for tag in hashtags:
            if self.normalize_case:
                tag_text = f"#{tag.lower()}"
            else:
                tag_text = f"#{tag}"

            # Avoid duplicates within the same text
            if tag_text not in seen:
                results.append({
                    'text': tag_text,
                    'type': 'HASHTAG',
                    'score': 1.0
                })
                seen.add(tag_text)

        return results

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Extract hashtags from a batch of texts.

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
            str: 'hashtag'
        """
        return 'hashtag'

    def get_config(self) -> Dict:
        """
        Return extractor configuration.

        Returns:
            Dict: Configuration parameters
        """
        return {
            'normalize_case': self.normalize_case
        }

"""
Exact match extractor for social media text.

Returns the exact text value as-is without any extraction.
Useful for pre-classified data or categorical values.
"""

from typing import List, Dict
from .base_extractor import BaseExtractor


class ExactMatchExtractor(BaseExtractor):
    """Return exact text value without extraction."""

    def __init__(self, strip_whitespace: bool = True):
        """
        Initialize exact match extractor.

        Args:
            strip_whitespace: If True, strips leading/trailing whitespace
        """
        self.strip_whitespace = strip_whitespace

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Return the exact text as a single extracted item.

        Args:
            text: Input text to use as-is
            **kwargs: Additional parameters (unused)

        Returns:
            List[Dict]: List with single item containing the exact text
        """
        if not text:
            return []

        if self.strip_whitespace:
            processed_text = text.strip()
        else:
            processed_text = text

        # Return empty if only whitespace
        if not processed_text:
            return []

        return [{
            'text': processed_text,
            'type': 'EXACT',
            'score': 1.0
        }]

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Extract exact matches from a batch of texts.

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
            str: 'exact'
        """
        return 'exact'

    def get_config(self) -> Dict:
        """
        Return extractor configuration.

        Returns:
            Dict: Configuration parameters
        """
        return {
            'strip_whitespace': self.strip_whitespace
        }

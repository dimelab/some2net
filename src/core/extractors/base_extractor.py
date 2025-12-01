"""
Base extractor abstract class for all extraction methods.

This module defines the interface that all extractors must implement,
ensuring consistent behavior across different extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseExtractor(ABC):
    """Abstract base class for all extraction methods."""

    @abstractmethod
    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Extract items from a single text.

        Args:
            text: Input text to extract from
            **kwargs: Extractor-specific parameters

        Returns:
            List[Dict]: Each dict contains:
                - 'text': extracted item text (str)
                - 'type': item type/category (str)
                - 'score': confidence/relevance score (float, 0-1)

        Example:
            [
                {'text': '#python', 'type': 'HASHTAG', 'score': 1.0},
                {'text': '#datascience', 'type': 'HASHTAG', 'score': 1.0}
            ]
        """
        pass

    @abstractmethod
    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Extract items from a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Processing batch size
            show_progress: Show progress bar
            **kwargs: Extractor-specific parameters

        Returns:
            List[List[Dict]]: List of extraction results, one per input text
        """
        pass

    @abstractmethod
    def get_extractor_type(self) -> str:
        """
        Return extractor type identifier.

        Returns:
            str: One of: 'ner', 'hashtag', 'mention', 'domain', 'keyword', 'exact'
        """
        pass

    def get_config(self) -> Dict:
        """
        Return extractor configuration (optional, for serialization).

        Returns:
            Dict: Configuration parameters
        """
        return {}

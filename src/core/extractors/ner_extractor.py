"""
NER Extractor Module

Wraps the existing NEREngine to conform to BaseExtractor interface.
"""

from typing import List, Dict, Optional
import logging

from .base_extractor import BaseExtractor
from ..ner_engine import NEREngine

logger = logging.getLogger(__name__)


class NERExtractor(BaseExtractor):
    """
    Named Entity Recognition extractor.

    Wraps the existing NEREngine to provide a consistent interface
    with other extractors.

    Example:
        >>> extractor = NERExtractor(
        ...     model_name="Davlan/xlm-roberta-base-ner-hrl",
        ...     confidence_threshold=0.85
        ... )
        >>> entities = extractor.extract_from_text("Apple Inc. is in California.")
        >>> print(entities)
        [
            {'text': 'Apple Inc.', 'type': 'ORG', 'score': 0.95},
            {'text': 'California', 'type': 'LOC', 'score': 0.92}
        ]
    """

    def __init__(
        self,
        model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        cache_dir: str = "./cache/ner_results",
        enable_cache: bool = True
    ):
        """
        Initialize NER extractor.

        Args:
            model_name: HuggingFace model name for NER
            device: Device to use ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence score for entities
            cache_dir: Directory for caching NER results
            enable_cache: Enable/disable result caching
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache

        # Initialize NER engine
        self.ner_engine = NEREngine(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
            cache_dir=cache_dir,
            enable_cache=enable_cache
        )

        logger.info(f"NERExtractor initialized with model: {model_name}")

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Extract named entities from a single text.

        Args:
            text: Input text to extract entities from
            **kwargs: Additional parameters (ignored)

        Returns:
            List of entity dictionaries with 'text', 'type', and 'score'

        Example:
            >>> entities = extractor.extract_from_text("John lives in Paris")
            >>> print(entities)
            [
                {'text': 'John', 'type': 'PER', 'score': 0.95},
                {'text': 'Paris', 'type': 'LOC', 'score': 0.92}
            ]
        """
        if not text or not text.strip():
            return []

        # Use NER engine to extract entities
        entities = self.ner_engine.extract_entities(text)

        # Entities from NER engine already have the correct format:
        # {'text': str, 'type': str, 'score': float}
        return entities

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Extract entities from a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Processing batch size
            show_progress: Show progress bar
            **kwargs: Additional parameters
                - detect_languages: Whether to detect languages (default: True)

        Returns:
            List of entity lists, one per input text

        Example:
            >>> texts = ["John lives in Paris", "Microsoft is in Seattle"]
            >>> results = extractor.extract_batch(texts)
            >>> print(len(results))
            2
        """
        if not texts:
            return []

        # Get optional parameters
        detect_languages = kwargs.get('detect_languages', True)

        # Use NER engine's batch processing
        entities_batch, languages = self.ner_engine.extract_entities_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            detect_languages=detect_languages
        )

        # NER engine returns entities in the correct format
        return entities_batch

    def get_extractor_type(self) -> str:
        """
        Return extractor type identifier.

        Returns:
            'ner'
        """
        return 'ner'

    def get_config(self) -> Dict:
        """
        Return extractor configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            'type': 'ner',
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'enable_cache': self.enable_cache
        }

    def clear_cache(self):
        """Clear the NER results cache."""
        if self.ner_engine and hasattr(self.ner_engine, 'clear_cache'):
            self.ner_engine.clear_cache()
            logger.info("NER cache cleared")

    def get_cache_info(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache information
        """
        if self.ner_engine and hasattr(self.ner_engine, 'get_cache_info'):
            return self.ner_engine.get_cache_info()
        return {}


# Example usage
if __name__ == "__main__":
    # Create NER extractor
    extractor = NERExtractor(
        model_name="Davlan/xlm-roberta-base-ner-hrl",
        confidence_threshold=0.85
    )

    # Test single extraction
    text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
    entities = extractor.extract_from_text(text)

    print("Single text extraction:")
    print(f"Text: {text}")
    print(f"Entities found: {len(entities)}")
    for entity in entities:
        print(f"  {entity['text']} ({entity['type']}): {entity['score']:.2f}")

    # Test batch extraction
    texts = [
        "Microsoft was founded by Bill Gates in Seattle.",
        "Paris is the capital of France.",
        "Google announced a new AI model."
    ]

    print("\n\nBatch extraction:")
    results = extractor.extract_batch(texts, show_progress=False)

    for i, (text, entities) in enumerate(zip(texts, results)):
        print(f"\nText {i+1}: {text}")
        print(f"Entities: {len(entities)}")
        for entity in entities:
            print(f"  {entity['text']} ({entity['type']}): {entity['score']:.2f}")

    # Print configuration
    print("\n\nExtractor configuration:")
    config = extractor.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

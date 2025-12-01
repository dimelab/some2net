"""
Keyword Extractor Module

Extracts keywords using RAKE (Rapid Automatic Keyword Extraction).
Requires two-pass processing: first collect texts per author, then extract keywords.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from rake_nltk import Rake
    # Download required NLTK data if not available
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK stopwords downloaded")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        print("✅ NLTK punkt downloaded")
except ImportError:
    Rake = None

from .base_extractor import BaseExtractor


class KeywordExtractor(BaseExtractor):
    """
    Extract keywords using RAKE (Rapid Automatic Keyword Extraction).

    RAKE is an unsupervised, domain-independent keyword extraction algorithm
    that identifies key phrases based on word co-occurrence and frequency.

    This extractor requires a two-pass approach:
    1. First pass: Collect all texts for each author
    2. Second pass: Extract keywords for each author

    Example:
        >>> extractor = KeywordExtractor(min_keywords=5, max_keywords=20)
        >>>
        >>> # First pass: collect texts
        >>> extractor.collect_texts("@user1", ["text1", "text2", "text3"])
        >>> extractor.collect_texts("@user2", ["text4", "text5"])
        >>>
        >>> # Second pass: extract keywords
        >>> keywords = extractor.extract_all_authors()
        >>> print(keywords["@user1"])  # List of keyword dicts
    """

    def __init__(
        self,
        min_keywords: int = 5,
        max_keywords: int = 20,
        language: str = 'english',
        max_phrase_length: int = 3,
        min_phrase_length: int = 1,
        ranking_metric: str = 'degree_to_frequency_ratio'
    ):
        """
        Initialize keyword extractor using RAKE.

        Args:
            min_keywords: Minimum number of keywords to extract per author
            max_keywords: Maximum number of keywords to extract per author
            language: Language for stopwords ('english', 'danish', etc.)
            max_phrase_length: Maximum number of words in a keyword phrase
            min_phrase_length: Minimum number of words in a keyword phrase
            ranking_metric: Metric for ranking keywords
                - 'degree_to_frequency_ratio' (default, best for most cases)
                - 'word_degree' (favors longer phrases)
                - 'word_frequency' (favors frequent words)
        """
        if Rake is None:
            raise ImportError(
                "rake-nltk is required for keyword extraction. "
                "Install it with: pip install rake-nltk"
            )

        self.min_keywords = min_keywords
        self.max_keywords = max_keywords
        self.language = language
        self.max_phrase_length = max_phrase_length
        self.min_phrase_length = min_phrase_length
        self.ranking_metric = ranking_metric

        # Storage for author texts (first pass)
        self.author_texts = defaultdict(list)

        logger.info(f"KeywordExtractor initialized: {min_keywords}-{max_keywords} keywords per author using RAKE")

    def _create_rake_instance(self) -> Rake:
        """Create a new RAKE instance with configured parameters."""
        # In newer versions of rake-nltk, ranking_metric can be passed as int:
        # 0 = DEGREE_TO_FREQUENCY_RATIO (default)
        # 1 = WORD_DEGREE
        # 2 = WORD_FREQUENCY

        # Try to use class constants if they exist, otherwise use integer values
        try:
            metric_map = {
                'degree_to_frequency_ratio': Rake.DEGREE_TO_FREQUENCY_RATIO,
                'word_degree': Rake.WORD_DEGREE,
                'word_frequency': Rake.WORD_FREQUENCY
            }
            metric = metric_map.get(
                self.ranking_metric,
                Rake.DEGREE_TO_FREQUENCY_RATIO
            )
        except AttributeError:
            # Fallback to integer values for older/newer versions
            metric_map = {
                'degree_to_frequency_ratio': 0,
                'word_degree': 1,
                'word_frequency': 2
            }
            metric = metric_map.get(self.ranking_metric, 0)

        return Rake(
            language=self.language,
            max_length=self.max_phrase_length,
            min_length=self.min_phrase_length,
            ranking_metric=metric
        )

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        """
        Not supported for keyword extraction.
        Use collect_texts() and extract_per_author() instead.

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            Empty list (not supported)

        Raises:
            NotImplementedError: This method is not supported for keyword extraction
        """
        raise NotImplementedError(
            "Keyword extraction requires collecting all texts per author. "
            "Use collect_texts() and extract_per_author() instead."
        )

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        **kwargs
    ) -> List[List[Dict]]:
        """
        Not supported for keyword extraction.
        Use collect_texts() and extract_all_authors() instead.

        Args:
            texts: List of input texts
            batch_size: Batch size (ignored)
            show_progress: Show progress (ignored)
            **kwargs: Additional parameters

        Returns:
            Empty list (not supported)

        Raises:
            NotImplementedError: This method is not supported for keyword extraction
        """
        raise NotImplementedError(
            "Keyword extraction requires collecting all texts per author. "
            "Use collect_texts() and extract_all_authors() instead."
        )

    def collect_texts(self, author: str, texts: List[str]):
        """
        Collect texts for an author (first pass).

        Args:
            author: Author identifier
            texts: List of texts to add for this author
        """
        if not author or not author.strip():
            logger.warning("Empty author name, skipping text collection")
            return

        author = str(author).strip()

        # Filter out empty or whitespace-only texts
        valid_texts = [t for t in texts if t and str(t).strip()]

        if valid_texts:
            self.author_texts[author].extend(valid_texts)
            logger.debug(f"Collected {len(valid_texts)} texts for {author} (total: {len(self.author_texts[author])})")

    def collect_text(self, author: str, text: str):
        """
        Collect a single text for an author.

        Args:
            author: Author identifier
            text: Single text to add
        """
        self.collect_texts(author, [text])

    def extract_per_author(
        self,
        author: str,
        use_collected: bool = True,
        texts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Extract keywords for a specific author using RAKE.

        Args:
            author: Author identifier
            use_collected: Whether to use previously collected texts (default: True)
            texts: Optional list of texts to use instead of collected texts

        Returns:
            List of keyword dictionaries with 'text', 'type', and 'score'
        """
        # Determine which texts to use
        if texts is not None:
            author_texts = texts
        elif use_collected:
            author_texts = self.author_texts.get(author, [])
        else:
            logger.warning(f"No texts provided for author {author}")
            return []

        if not author_texts:
            logger.warning(f"No texts found for author {author}")
            return []

        # Filter empty texts
        author_texts = [str(t).strip() for t in author_texts if t and str(t).strip()]

        if not author_texts:
            logger.warning(f"No valid texts for author {author} after filtering")
            return []

        # Combine all texts for this author
        combined_text = ' '.join(author_texts)

        try:
            # Create RAKE instance
            rake = self._create_rake_instance()

            # Extract keywords
            rake.extract_keywords_from_text(combined_text)

            # Get ranked phrases with scores
            ranked_phrases = rake.get_ranked_phrases_with_scores()

            if not ranked_phrases:
                logger.warning(f"No keywords extracted for {author}")
                return []

            # Sort by score (descending)
            ranked_phrases.sort(key=lambda x: x[0], reverse=True)

            # Determine how many keywords to extract
            # At least min_keywords, at most max_keywords
            num_keywords = max(
                self.min_keywords,
                min(len(ranked_phrases), self.max_keywords)
            )

            # Normalize scores to 0-1 range
            max_score = ranked_phrases[0][0] if ranked_phrases else 1.0
            min_score = ranked_phrases[-1][0] if ranked_phrases else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0

            # Return top keywords
            results = []
            for score, phrase in ranked_phrases[:num_keywords]:
                # Normalize score to 0-1 range
                normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5

                results.append({
                    'text': phrase,
                    'type': 'KEYWORD',
                    'score': float(normalized_score)
                })

            logger.debug(f"Extracted {len(results)} keywords for {author}")
            return results

        except Exception as e:
            logger.error(f"Error extracting keywords for {author}: {e}")
            return []

    def extract_all_authors(self, show_progress: bool = True) -> Dict[str, List[Dict]]:
        """
        Extract keywords for all collected authors.

        Args:
            show_progress: Whether to show a progress bar

        Returns:
            Dictionary mapping author to list of keyword dictionaries
        """
        results = {}

        authors = list(self.author_texts.keys())

        if show_progress:
            authors = tqdm(authors, desc="Extracting keywords")

        for author in authors:
            results[author] = self.extract_per_author(author, use_collected=True)

        total_keywords = sum(len(keywords) for keywords in results.values())
        logger.info(f"Extracted keywords for {len(results)} authors (total: {total_keywords} keywords)")

        return results

    def get_extractor_type(self) -> str:
        """
        Return extractor type identifier.

        Returns:
            'keyword'
        """
        return 'keyword'

    def get_config(self) -> Dict:
        """
        Return extractor configuration.

        Returns:
            Dictionary with configuration parameters
        """
        return {
            'type': 'keyword',
            'algorithm': 'RAKE',
            'min_keywords': self.min_keywords,
            'max_keywords': self.max_keywords,
            'language': self.language,
            'max_phrase_length': self.max_phrase_length,
            'min_phrase_length': self.min_phrase_length,
            'ranking_metric': self.ranking_metric
        }

    def get_author_count(self) -> int:
        """
        Get the number of authors with collected texts.

        Returns:
            Number of authors
        """
        return len(self.author_texts)

    def get_text_count(self, author: Optional[str] = None) -> int:
        """
        Get the number of collected texts.

        Args:
            author: Optional author to get count for (if None, returns total)

        Returns:
            Number of texts
        """
        if author is not None:
            return len(self.author_texts.get(author, []))
        else:
            return sum(len(texts) for texts in self.author_texts.values())

    def clear(self):
        """Clear all collected texts."""
        self.author_texts.clear()
        logger.info("Cleared all collected texts")

    def reset(self):
        """Alias for clear()."""
        self.clear()


# Example usage
if __name__ == "__main__":
    # Create extractor
    extractor = KeywordExtractor(
        min_keywords=5,
        max_keywords=10,
        language='english'
    )

    # Simulate collecting texts from posts
    posts = [
        {"author": "@user1", "text": "Machine learning and artificial intelligence are transforming data science"},
        {"author": "@user1", "text": "Deep learning models using neural networks for computer vision"},
        {"author": "@user1", "text": "Python programming for machine learning and data analysis"},
        {"author": "@user2", "text": "Climate change is affecting global weather patterns"},
        {"author": "@user2", "text": "Renewable energy and sustainable development are crucial"},
        {"author": "@user2", "text": "Environmental protection and conservation efforts worldwide"}
    ]

    # First pass: collect texts
    for post in posts:
        extractor.collect_text(post["author"], post["text"])

    print(f"Collected texts for {extractor.get_author_count()} authors")
    print(f"Total texts: {extractor.get_text_count()}")

    # Second pass: extract keywords
    keywords = extractor.extract_all_authors(show_progress=False)

    # Display results
    for author, author_keywords in keywords.items():
        print(f"\n{author}:")
        for kw in author_keywords:
            print(f"  {kw['text']}: {kw['score']:.3f}")

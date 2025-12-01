"""
Keyword Extractor Module

Extracts keywords using TF-IDF on unigrams and bigrams.
Requires two-pass processing: first collect texts per author, then extract keywords.
"""

from typing import List, Dict, Optional
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from tqdm import tqdm

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class KeywordExtractor(BaseExtractor):
    """
    Extract keywords using TF-IDF on unigrams and bigrams.

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
        stop_words: str = 'english',
        ngram_range: tuple = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95
    ):
        """
        Initialize keyword extractor.

        Args:
            min_keywords: Minimum number of keywords to extract per author
            max_keywords: Maximum number of keywords to extract per author
            stop_words: Stop words language ('english', 'none', etc.)
            ngram_range: Range of n-grams (default: (1,2) for unigrams and bigrams)
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms (as fraction)
        """
        self.min_keywords = min_keywords
        self.max_keywords = max_keywords
        self.stop_words = stop_words if stop_words != 'none' else None
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        # Storage for author texts (first pass)
        self.author_texts = defaultdict(list)

        logger.info(f"KeywordExtractor initialized: {min_keywords}-{max_keywords} keywords per author")

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
        Extract keywords for a specific author.

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
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_keywords * 2,  # Extract more, then filter
                stop_words=self.stop_words,
                min_df=self.min_df,
                lowercase=True,
                max_df=self.max_df
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Sort by score
            keyword_scores = sorted(
                zip(feature_names, scores),
                key=lambda x: x[1],
                reverse=True
            )

            # Determine how many keywords to extract
            # At least min_keywords, at most max_keywords
            num_keywords = max(
                self.min_keywords,
                min(len(keyword_scores), self.max_keywords)
            )

            # Return top keywords
            results = [
                {
                    'text': keyword,
                    'type': 'KEYWORD',
                    'score': float(score)
                }
                for keyword, score in keyword_scores[:num_keywords]
                if score > 0  # Only include keywords with non-zero scores
            ]

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
            'min_keywords': self.min_keywords,
            'max_keywords': self.max_keywords,
            'stop_words': self.stop_words,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df
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
        stop_words='english'
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

"""
Keyword Extractor Module

Extracts keywords using RAKE (Rapid Automatic Keyword Extraction).
Requires two-pass processing: first collect texts per author, then extract keywords.
"""

from typing import List, Dict, Optional
from collections import defaultdict, Counter
import logging
from tqdm import tqdm
import math
import re

logger = logging.getLogger(__name__)

try:
    from rake_nltk import Rake
    # Download required NLTK data if not available
    import nltk

    # List of required NLTK resources for RAKE
    required_resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]

    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK {resource_name}...")
            try:
                nltk.download(resource_name, quiet=True)
                print(f"✅ NLTK {resource_name} downloaded")
            except Exception as e:
                print(f"⚠️ Warning: Could not download {resource_name}: {e}")
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
        ranking_metric: str = 'degree_to_frequency_ratio',
        min_char_length: int = 3,
        use_tfidf: bool = True,
        filter_common_words: bool = True
    ):
        """
        Initialize keyword extractor using RAKE with optional TF-IDF weighting.

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
            min_char_length: Minimum character length for keywords (default: 3)
            use_tfidf: Whether to use TF-IDF weighting to boost distinctive keywords (default: True)
            filter_common_words: Whether to filter very common single-word keywords (default: True)
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
        self.min_char_length = min_char_length
        self.use_tfidf = use_tfidf
        self.filter_common_words = filter_common_words

        # Storage for author texts (first pass)
        self.author_texts = defaultdict(list)

        # Extended stopwords for common Danish and English words
        # These are filtered in addition to NLTK's stopwords
        self.additional_stopwords = {
            'danish': [
                # Common words often missed by NLTK
                'så', 'kan', 'ja', 'tak', 'enig', 'hvorfor', 'lige', 'bare', 'godt', 'rigtig',
                'ved', 'har', 'var', 'vil', 'blevet', 'været', 'måske', 'altså', 'okay', 'ok',
                'dog', 'endnu', 'flere', 'fx', 'helt', 'hmm', 'hej', 'hey', 'jo', 'mener',
                'netop', 'også', 'rent', 'selvfølgelig', 'synes', 'tror', 'ud', 'åh',
                # Very short common words
                'af', 'at', 'de', 'den', 'der', 'det', 'du', 'en', 'er', 'et', 'for',
                'han', 'hun', 'i', 'jeg', 'med', 'men', 'og', 'på', 'til', 'vi',
                # Numbers and short words
                'nu', 'ny', 'om', 'op', 'os', 'se', 'si', 'to', 'tre', 'fire', 'fem'
            ],
            'english': [
                'yes', 'no', 'ok', 'okay', 'yeah', 'yep', 'nope', 'thanks', 'please',
                'just', 'like', 'really', 'very', 'much', 'well', 'actually', 'basically',
                'literally', 'totally', 'definitely', 'probably', 'maybe', 'perhaps',
                'hmm', 'umm', 'uh', 'ah', 'oh', 'hey', 'hi', 'hello'
            ]
        }

        logger.info(f"KeywordExtractor initialized: {min_keywords}-{max_keywords} keywords per author using RAKE + {'TF-IDF' if use_tfidf else 'RAKE only'}")

    def _get_stopwords(self) -> set:
        """
        Get comprehensive stopwords list for the language.

        Returns:
            Set of stopwords combining NLTK and custom stopwords
        """
        stopwords_set = set()

        # Try to get NLTK stopwords
        try:
            from nltk.corpus import stopwords
            nltk_stopwords = stopwords.words(self.language)
            stopwords_set.update(nltk_stopwords)
            logger.debug(f"Loaded {len(nltk_stopwords)} NLTK stopwords for {self.language}")
        except Exception as e:
            logger.debug(f"Could not load NLTK stopwords: {e}")

        # Add our custom stopwords
        custom_stopwords = self.additional_stopwords.get(self.language, [])
        stopwords_set.update(custom_stopwords)

        logger.debug(f"Total stopwords for {self.language}: {len(stopwords_set)}")
        return stopwords_set

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

        # Get comprehensive stopwords list
        stopwords_list = list(self._get_stopwords())

        return Rake(
            language=self.language,
            max_length=self.max_phrase_length,
            min_length=self.min_phrase_length,
            ranking_metric=metric,
            stopwords=stopwords_list  # Pass custom stopwords to RAKE
        )

    def _filter_keyword(self, keyword: str) -> bool:
        """
        Filter out unwanted keywords including URLs, mentions, hashtags, etc.

        Args:
            keyword: Keyword to check

        Returns:
            True if keyword should be kept, False if it should be filtered out
        """
        keyword_lower = keyword.lower().strip()

        # Filter by character length
        if len(keyword_lower) < self.min_char_length:
            return False

        # Filter URLs (http://, https://, www., domain.com patterns)
        url_pattern = r'(https?://|www\.|[a-z0-9-]+\.(com|org|net|edu|gov|io|co|dk|de|uk|fr|es|it|nl|se|no))'
        if re.search(url_pattern, keyword_lower):
            return False

        # Filter social media artifacts
        # Hashtags
        if keyword_lower.startswith('#'):
            return False
        # Mentions
        if keyword_lower.startswith('@'):
            return False
        # Email addresses
        if '@' in keyword_lower and '.' in keyword_lower:
            return False

        # Filter keywords that are mostly numbers or special characters
        # Allow some special chars for languages like Danish (å, æ, ø)
        alphanumeric_ratio = sum(c.isalnum() for c in keyword_lower) / len(keyword_lower) if keyword_lower else 0
        if alphanumeric_ratio < 0.7:  # At least 70% should be letters/numbers
            return False

        # Filter additional common words if enabled
        if self.filter_common_words:
            # Check against additional stopwords for this language
            lang_stopwords = self.additional_stopwords.get(self.language, [])
            if keyword_lower in lang_stopwords:
                return False

            # Filter very short single words (likely common words)
            words = keyword_lower.split()
            if len(words) == 1 and len(keyword_lower) <= 2:
                return False

        return True

    def _calculate_tfidf_scores(self, all_keywords: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate TF-IDF scores for keywords across all authors.

        Args:
            all_keywords: Dict mapping author to list of keywords

        Returns:
            Dict mapping keyword to TF-IDF score
        """
        # Count document frequency (how many authors use each keyword)
        df = Counter()
        all_docs = []

        for author, keywords in all_keywords.items():
            unique_keywords = set(k.lower() for k in keywords)
            all_docs.append(unique_keywords)
            for keyword in unique_keywords:
                df[keyword] += 1

        num_docs = len(all_docs)

        # Calculate IDF scores
        idf_scores = {}
        for keyword, doc_freq in df.items():
            # IDF = log(N / df) where N is total documents, df is document frequency
            idf_scores[keyword] = math.log(num_docs / doc_freq) if doc_freq > 0 else 0

        return idf_scores

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

            # Filter out unwanted keywords
            filtered_phrases = [
                (score, phrase) for score, phrase in ranked_phrases
                if self._filter_keyword(phrase)
            ]

            if not filtered_phrases:
                logger.warning(f"No keywords remaining after filtering for {author}")
                return []

            # Sort by score (descending)
            filtered_phrases.sort(key=lambda x: x[0], reverse=True)

            # Determine how many keywords to extract
            # At least min_keywords (if available), at most max_keywords
            num_keywords = min(
                max(self.min_keywords, len(filtered_phrases)),
                self.max_keywords
            )

            # Normalize scores to 0-1 range
            max_score = filtered_phrases[0][0] if filtered_phrases else 1.0
            min_score = filtered_phrases[-1][0] if filtered_phrases else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0

            # Return top keywords (without TF-IDF at this stage)
            results = []
            for score, phrase in filtered_phrases[:num_keywords]:
                # Normalize score to 0-1 range
                normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5

                results.append({
                    'text': phrase,
                    'type': 'KEYWORD',
                    'score': float(normalized_score),
                    'raw_score': float(score)  # Keep raw score for TF-IDF weighting
                })

            logger.debug(f"Extracted {len(results)} keywords for {author}")
            return results

        except Exception as e:
            logger.error(f"Error extracting keywords for {author}: {e}")
            return []

    def extract_all_authors(self, show_progress: bool = True) -> Dict[str, List[Dict]]:
        """
        Extract keywords for all collected authors with optional TF-IDF weighting.

        Args:
            show_progress: Whether to show a progress bar

        Returns:
            Dictionary mapping author to list of keyword dictionaries
        """
        results = {}

        authors = list(self.author_texts.keys())

        if show_progress:
            authors = tqdm(authors, desc="Extracting keywords")

        # First pass: Extract keywords for each author
        for author in authors:
            results[author] = self.extract_per_author(author, use_collected=True)

        # Second pass: Apply TF-IDF weighting if enabled
        if self.use_tfidf and len(results) > 1:
            # Collect all keywords for TF-IDF calculation
            all_keywords = {
                author: [kw['text'] for kw in keywords]
                for author, keywords in results.items()
            }

            # Calculate IDF scores
            idf_scores = self._calculate_tfidf_scores(all_keywords)

            # Re-rank keywords using TF-IDF
            for author, keywords in results.items():
                if not keywords:
                    continue

                # Calculate TF-IDF scores
                for kw in keywords:
                    keyword_lower = kw['text'].lower()
                    idf = idf_scores.get(keyword_lower, 0)
                    # TF = normalized RAKE score, IDF from calculation
                    tfidf = kw['score'] * (1 + idf)  # Add 1 to avoid zero scores
                    kw['tfidf_score'] = float(tfidf)

                # Re-sort by TF-IDF score
                keywords.sort(key=lambda x: x.get('tfidf_score', x['score']), reverse=True)

                # Update final scores to TF-IDF scores
                for kw in keywords:
                    kw['score'] = kw.get('tfidf_score', kw['score'])
                    # Clean up temporary scores
                    kw.pop('tfidf_score', None)
                    kw.pop('raw_score', None)

                results[author] = keywords

        total_keywords = sum(len(keywords) for keywords in results.values())
        logger.info(f"Extracted keywords for {len(results)} authors (total: {total_keywords} keywords, TF-IDF: {self.use_tfidf})")

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
            'algorithm': 'RAKE + TF-IDF' if self.use_tfidf else 'RAKE',
            'min_keywords': self.min_keywords,
            'max_keywords': self.max_keywords,
            'language': self.language,
            'max_phrase_length': self.max_phrase_length,
            'min_phrase_length': self.min_phrase_length,
            'min_char_length': self.min_char_length,
            'ranking_metric': self.ranking_metric,
            'use_tfidf': self.use_tfidf,
            'filter_common_words': self.filter_common_words
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

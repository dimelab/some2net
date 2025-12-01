"""
Extractors module for different extraction methods.

This module provides a unified interface for various extraction strategies
including hashtags, mentions, domains, and exact matches.
"""

from .base_extractor import BaseExtractor
from .hashtag_extractor import HashtagExtractor
from .mention_extractor import MentionExtractor
from .domain_extractor import DomainExtractor
from .exact_match_extractor import ExactMatchExtractor
from .keyword_extractor import KeywordExtractor
from .ner_extractor import NERExtractor

__all__ = [
    'BaseExtractor',
    'HashtagExtractor',
    'MentionExtractor',
    'DomainExtractor',
    'ExactMatchExtractor',
    'KeywordExtractor',
    'NERExtractor',
]

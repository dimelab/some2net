# Implementation Plan: Multi-Method Extraction & Metadata Features

**Date**: 2025-12-01
**Author**: Claude Code
**Project**: some2net - Social Network Analytics Library

---

## Table of Contents

1. [Overview](#overview)
2. [Current Architecture](#current-architecture)
3. [Proposed Changes](#proposed-changes)
4. [New Features](#new-features)
5. [Architecture Design](#architecture-design)
6. [Implementation Details](#implementation-details)
7. [File Structure](#file-structure)
8. [Implementation Phases](#implementation-phases)
9. [Testing Strategy](#testing-strategy)
10. [Considerations & Edge Cases](#considerations--edge-cases)

---

## Overview

### Goals

Add support for multiple extraction methods beyond NER, allowing users to build networks based on:
- **Hashtag extraction**: Links authors with hashtags they use
- **Mention extraction**: Links authors with users they mention (@username)
- **Domain extraction**: Links authors with domains from URLs they share
- **Keyword extraction**: Links authors with TF-IDF weighted keywords (5-20 per author)
- **Exact match**: Links authors with exact text values (no extraction)

Additionally, add the ability to attach metadata from CSV/NDJSON columns to both nodes and edges.

### Success Criteria

- [ ] User can select extraction method in UI
- [ ] All extraction methods follow a consistent interface
- [ ] Metadata columns can be selected and attached to nodes/edges
- [ ] Existing NER functionality remains unchanged
- [ ] All extractors have comprehensive tests
- [ ] Documentation updated with examples

---

## Current Architecture

### Data Flow

```
Input File (CSV/NDJSON)
    ↓
DataLoader (chunked reading)
    ↓
NEREngine (entity extraction)
    ↓
EntityResolver (deduplication)
    ↓
EntityLinker (optional Wikipedia linking)
    ↓
NetworkBuilder (graph construction)
    ↓
Export (GEXF, GraphML, JSON, etc.)
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `DataLoader` | Load CSV/NDJSON files | `src/core/data_loader.py` |
| `NEREngine` | Named entity extraction | `src/core/ner_engine.py` |
| `EntityResolver` | Entity deduplication | `src/core/entity_resolver.py` |
| `EntityLinker` | Wikipedia/Wikidata linking | `src/core/entity_linker.py` |
| `NetworkBuilder` | Graph construction | `src/core/network_builder.py` |
| `SocialNetworkPipeline` | Orchestration | `src/core/pipeline.py` |
| Streamlit App | User interface | `src/cli/app.py` |

### Current Limitations

1. **Single extraction method**: Only NER is supported
2. **No metadata support**: Cannot attach additional column data to nodes/edges
3. **Tightly coupled**: NER logic is hardcoded in pipeline

---

## Proposed Changes

### 1. Extraction Method Abstraction

Create a **base extractor class** that all extraction methods implement. This allows the pipeline to work with any extractor uniformly.

**Benefits**:
- Consistent interface across all extraction methods
- Easy to add new extraction methods in the future
- Existing NER can be wrapped without modification
- Testable in isolation

### 2. Metadata Column Support

Extend `NetworkBuilder` and `Pipeline` to accept metadata from input columns and attach it to nodes and edges.

**Benefits**:
- Rich network analysis with contextual data
- Support for temporal analysis (timestamps)
- User identifiers, post IDs, sentiment scores, etc.

---

## New Features

### Feature 1: Hashtag Extraction

**Description**: Extract hashtags (e.g., `#python`, `#machinelearning`) from text and create edges from authors to hashtags.

**Example Network**:
```
@alice → #python (weight: 5)
@alice → #datascience (weight: 3)
@bob → #python (weight: 2)
```

**Configuration Options**:
- Case normalization (keep `#Python` vs normalize to `#python`)
- Minimum frequency filter

### Feature 2: Mention Extraction

**Description**: Extract user mentions (e.g., `@username`) from text and create edges from authors to mentioned users.

**Example Network**:
```
@alice → @bob (weight: 3)
@alice → @charlie (weight: 1)
@bob → @alice (weight: 2)
```

**Configuration Options**:
- Platform-specific patterns (Twitter, TikTok, etc.)
- Self-mention handling

### Feature 3: Domain Extraction

**Description**: Extract domains from URLs in text and create edges from authors to domains.

**Example Network**:
```
@alice → nytimes.com (weight: 5)
@alice → bbc.com (weight: 2)
@bob → nytimes.com (weight: 3)
```

**Configuration Options**:
- Subdomain handling (keep `www.` vs strip)
- URL shortener expansion (future)

### Feature 4: Keyword Extraction (TF-IDF)

**Description**: Extract 5-20 keywords per author using TF-IDF on unigrams and bigrams from all their posts.

**Example Network**:
```
@alice → "machine learning" (weight: 0.85)
@alice → "neural networks" (weight: 0.72)
@alice → "python" (weight: 0.65)
```

**Configuration Options**:
- Min/max keywords per author (5-20 default)
- Stop words language
- N-gram range (1-2 default)

**Special Considerations**:
- Requires all texts per author before extraction
- Two-pass processing needed
- Higher memory usage than other methods

### Feature 5: Exact Match

**Description**: Use the raw text value as-is without any extraction. Creates edge from author to the exact text content.

**Example Network**:
```
@alice → "I love programming!" (weight: 1)
@alice → "Python is great" (weight: 1)
```

**Use Cases**:
- Sentiment categories (if pre-classified)
- Topic labels
- Any categorical data

---

## Architecture Design

### Extractor Base Class

**Location**: `src/core/extractors/base_extractor.py`

```python
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
```

### Extractor Implementations

#### 1. NER Extractor (Wrapper)

**Location**: `src/core/extractors/ner_extractor.py`

Wraps the existing `NEREngine` to conform to `BaseExtractor` interface.

#### 2. Hashtag Extractor

**Location**: `src/core/extractors/hashtag_extractor.py`

```python
import re
from typing import List, Dict
from .base_extractor import BaseExtractor

class HashtagExtractor(BaseExtractor):
    """Extract hashtags from text."""

    def __init__(self, normalize_case: bool = True):
        self.normalize_case = normalize_case

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        if not text:
            return []

        # Pattern: # followed by word characters (Unicode-aware)
        pattern = r'#(\w+)'
        hashtags = re.findall(pattern, text, re.UNICODE)

        results = []
        for tag in hashtags:
            if self.normalize_case:
                tag_text = f"#{tag.lower()}"
            else:
                tag_text = f"#{tag}"

            results.append({
                'text': tag_text,
                'type': 'HASHTAG',
                'score': 1.0
            })

        return results

    def extract_batch(self, texts: List[str], **kwargs) -> List[List[Dict]]:
        return [self.extract_from_text(text) for text in texts]

    def get_extractor_type(self) -> str:
        return 'hashtag'
```

#### 3. Mention Extractor

**Location**: `src/core/extractors/mention_extractor.py`

Similar to hashtag extractor, but matches `@username` pattern.

#### 4. Domain Extractor

**Location**: `src/core/extractors/domain_extractor.py`

```python
import re
from typing import List, Dict
from urllib.parse import urlparse
from .base_extractor import BaseExtractor

class DomainExtractor(BaseExtractor):
    """Extract domains from URLs in text."""

    def __init__(self, strip_www: bool = True):
        self.strip_www = strip_www

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        if not text:
            return []

        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)

        results = []
        seen_domains = set()

        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc

                if not domain:
                    continue

                if self.strip_www and domain.startswith('www.'):
                    domain = domain[4:]

                if domain not in seen_domains:
                    results.append({
                        'text': domain,
                        'type': 'DOMAIN',
                        'score': 1.0
                    })
                    seen_domains.add(domain)
            except:
                continue

        return results

    def extract_batch(self, texts: List[str], **kwargs) -> List[List[Dict]]:
        return [self.extract_from_text(text) for text in texts]

    def get_extractor_type(self) -> str:
        return 'domain'
```

#### 5. Keyword Extractor (TF-IDF)

**Location**: `src/core/extractors/keyword_extractor.py`

```python
from typing import List, Dict
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from .base_extractor import BaseExtractor

class KeywordExtractor(BaseExtractor):
    """Extract keywords using TF-IDF on unigrams and bigrams."""

    def __init__(
        self,
        min_keywords: int = 5,
        max_keywords: int = 20,
        stop_words: str = 'english',
        ngram_range: tuple = (1, 2)
    ):
        self.min_keywords = min_keywords
        self.max_keywords = max_keywords
        self.stop_words = stop_words if stop_words != 'none' else None
        self.ngram_range = ngram_range
        self.author_texts = defaultdict(list)

    def collect_texts(self, author: str, texts: List[str]):
        """Collect texts for an author (first pass)."""
        self.author_texts[author].extend(texts)

    def extract_per_author(self, author: str) -> List[Dict]:
        """Extract keywords for a specific author."""
        texts = self.author_texts.get(author, [])

        if not texts:
            return []

        # Filter empty texts
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return []

        combined_text = ' '.join(texts)

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_keywords * 2,
                stop_words=self.stop_words,
                min_df=1,
                lowercase=True,
                max_df=0.95
            )

            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            keyword_scores = sorted(
                zip(feature_names, scores),
                key=lambda x: x[1],
                reverse=True
            )

            num_keywords = max(
                self.min_keywords,
                min(len(keyword_scores), self.max_keywords)
            )

            return [
                {
                    'text': keyword,
                    'type': 'KEYWORD',
                    'score': float(score)
                }
                for keyword, score in keyword_scores[:num_keywords]
            ]
        except Exception as e:
            return []

    def extract_all_authors(self) -> Dict[str, List[Dict]]:
        """Extract keywords for all collected authors."""
        return {author: self.extract_per_author(author)
                for author in self.author_texts.keys()}

    def get_extractor_type(self) -> str:
        return 'keyword'
```

#### 6. Exact Match Extractor

**Location**: `src/core/extractors/exact_match_extractor.py`

```python
from typing import List, Dict
from .base_extractor import BaseExtractor

class ExactMatchExtractor(BaseExtractor):
    """Return exact text value without extraction."""

    def extract_from_text(self, text: str, **kwargs) -> List[Dict]:
        if not text or not text.strip():
            return []

        return [{
            'text': text.strip(),
            'type': 'EXACT',
            'score': 1.0
        }]

    def extract_batch(self, texts: List[str], **kwargs) -> List[List[Dict]]:
        return [self.extract_from_text(text) for text in texts]

    def get_extractor_type(self) -> str:
        return 'exact'
```

---

## Implementation Details

### Pipeline Modifications

**Location**: `src/core/pipeline.py`

**Key Changes**:

1. Add `extraction_method` and `extractor_config` parameters to `__init__()`
2. Create `_create_extractor()` factory method
3. Add `node_metadata_columns` and `edge_metadata_columns` to `process_file()`
4. Handle keyword extraction specially (two-pass)
5. Extract and pass metadata to `NetworkBuilder`

**Example**:

```python
class SocialNetworkPipeline:
    def __init__(
        self,
        extraction_method: str = "ner",
        extractor_config: Optional[Dict] = None,
        # ... existing parameters
    ):
        self.extraction_method = extraction_method
        self.extractor = self._create_extractor(extraction_method, extractor_config)
        # ... rest of init

    def _create_extractor(self, method: str, config: Optional[Dict]) -> BaseExtractor:
        """Factory method to create appropriate extractor."""
        from .extractors import (
            NERExtractor, HashtagExtractor, MentionExtractor,
            DomainExtractor, KeywordExtractor, ExactMatchExtractor
        )

        config = config or {}

        if method == "ner":
            return NERExtractor(**config)
        elif method == "hashtag":
            return HashtagExtractor(**config)
        elif method == "mention":
            return MentionExtractor(**config)
        elif method == "domain":
            return DomainExtractor(**config)
        elif method == "keyword":
            return KeywordExtractor(**config)
        elif method == "exact":
            return ExactMatchExtractor(**config)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

    def process_file(
        self,
        # ... existing parameters
        node_metadata_columns: Optional[List[str]] = None,
        edge_metadata_columns: Optional[List[str]] = None
    ) -> Tuple[nx.DiGraph, Dict]:
        """Process file with optional metadata."""

        # For keyword extraction, use two-pass approach
        if self.extraction_method == "keyword":
            return self._process_file_keyword(...)
        else:
            return self._process_file_standard(...)
```

### NetworkBuilder Modifications

**Location**: `src/core/network_builder.py`

**Key Changes**:

```python
def add_post(
    self,
    author: str,
    entities: List[Dict],
    post_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    node_metadata: Optional[Dict] = None,  # NEW
    edge_metadata: Optional[Dict] = None   # NEW
):
    """Add post to network with optional metadata."""

    # Add author node with metadata
    if not self.graph.has_node(author):
        node_attrs = {
            'node_type': 'author',
            'label': author,
            'mention_count': 0,
            'post_count': 0
        }
        if node_metadata:
            node_attrs.update(node_metadata)
        self.graph.add_node(author, **node_attrs)

    # Process entities with edge metadata
    for entity in entities:
        self._add_entity_edge(
            author, entity['text'], entity['type'], entity['score'],
            post_id, timestamp, edge_metadata
        )
```

### Streamlit UI Updates

**Location**: `src/cli/app.py`

**Major Additions**:

1. **Extraction method selector**:

```python
extraction_method = st.selectbox(
    "Choose Extraction Method",
    ["NER (Named Entities)", "Hashtags", "Mentions (@username)",
     "URL Domains", "Keywords (TF-IDF)", "Exact Match"]
)
```

2. **Method-specific configuration panels**
3. **Metadata column selectors**:

```python
node_metadata_cols = st.multiselect(
    "Node Metadata Columns",
    available_columns,
    help="Attach these columns as attributes to nodes"
)

edge_metadata_cols = st.multiselect(
    "Edge Metadata Columns",
    available_columns,
    help="Attach these columns as attributes to edges"
)
```

---

## File Structure

```
some2net/
├── src/
│   ├── core/
│   │   ├── extractors/                    # NEW
│   │   │   ├── __init__.py
│   │   │   ├── base_extractor.py
│   │   │   ├── ner_extractor.py
│   │   │   ├── hashtag_extractor.py
│   │   │   ├── mention_extractor.py
│   │   │   ├── domain_extractor.py
│   │   │   ├── keyword_extractor.py
│   │   │   └── exact_match_extractor.py
│   │   ├── pipeline.py                   # MODIFIED
│   │   ├── network_builder.py            # MODIFIED
│   │   └── ... (other modules)
│   ├── cli/
│   │   └── app.py                        # MODIFIED
│   └── utils/
│       └── ...
├── tests/
│   ├── test_extractors/                  # NEW
│   │   ├── test_hashtag_extractor.py
│   │   ├── test_mention_extractor.py
│   │   ├── test_domain_extractor.py
│   │   ├── test_keyword_extractor.py
│   │   └── test_exact_match_extractor.py
│   └── test_metadata_integration.py      # NEW
├── examples/
│   ├── example_hashtag_network.py        # NEW
│   └── example_keyword_network.py        # NEW
├── IMPLEMENTATION_PLAN.md                # THIS FILE
└── README.md                             # UPDATE
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Create extractor abstraction and simple extractors

**Tasks**:
1. Create `src/core/extractors/` directory
2. Implement `base_extractor.py`
3. Implement simple extractors: hashtag, mention, domain, exact
4. Write unit tests for each
5. Create `__init__.py` for easy imports

**Deliverable**: Working simple extractors with tests

---

### Phase 2: Advanced Extraction + Metadata (Week 2)

**Goal**: Add keyword extraction and metadata support

**Tasks**:
1. Implement `keyword_extractor.py` with TF-IDF
2. Wrap existing NER as `ner_extractor.py`
3. Modify `NetworkBuilder` to accept metadata
4. Write integration tests for metadata
5. Update exporters to preserve metadata

**Deliverable**: Keyword extraction + metadata support

---

### Phase 3: Pipeline Integration (Week 3)

**Goal**: Integrate extractors into pipeline

**Tasks**:
1. Modify `SocialNetworkPipeline.__init__()`
2. Implement `_create_extractor()` factory
3. Update `process_file()` with metadata parameters
4. Implement two-pass processing for keywords
5. Write end-to-end integration tests

**Deliverable**: Fully functional pipeline

---

### Phase 4: UI & Documentation (Week 4)

**Goal**: Update UI and documentation

**Tasks**:
1. Add extraction method selector to UI
2. Add method-specific configuration panels
3. Add metadata column selection UI
4. Create example scripts
5. Update documentation

**Deliverable**: Complete UI and docs

---

## Testing Strategy

### Unit Tests

**For each extractor**:
- Extract single item
- Extract multiple items
- Handle Unicode
- Case normalization
- Empty text handling
- No matches in text

### Integration Tests

**Pipeline with different methods**:
- Process file with each extraction method
- Metadata flow through pipeline
- Export with all formats
- Large dataset performance

### End-to-End Tests

- Full workflow with each method
- Full workflow with metadata
- Export verification

---

## Considerations & Edge Cases

### 1. Keyword Extraction
- **Challenge**: Requires all texts per author
- **Solution**: Two-pass processing, first collect then extract
- **Edge case**: Author with single short post may not reach min_keywords

### 2. Metadata Handling
- **Challenge**: Different data types, missing values, name conflicts
- **Solution**: Convert to strings, use None for missing, prefix with `meta_`
- **Edge case**: Column doesn't exist in some chunks → skip gracefully

### 3. Hashtag & Mention
- **Challenge**: Unicode support, platform-specific patterns
- **Solution**: Unicode-aware regex, configurable patterns
- **Edge case**: Email addresses contain @ → validate and filter

### 4. Domain Extraction
- **Challenge**: URL shorteners, malformed URLs, non-HTTP
- **Solution**: Robust parsing with exception handling
- **Edge case**: URL with port → remove port number

### 5. Memory Management
- **Challenge**: Keyword extraction stores all texts
- **Solution**: Stream texts, clear after processing
- **Monitoring**: Log memory usage, warn on large datasets

---

## Success Metrics

### Development
- [ ] All extractors implemented
- [ ] Test coverage >80%
- [ ] Documentation complete
- [ ] UI updated

### Performance
- [ ] Simple extractors: <1s for 10K posts
- [ ] Keyword extraction: <60s for 10K posts
- [ ] Metadata overhead: <10%

### User Experience
- [ ] Clear method selection UI
- [ ] Helpful tooltips
- [ ] Informative errors
- [ ] Export preserves metadata

---

## Conclusion

This plan provides a comprehensive roadmap for adding multi-method extraction and metadata support while maintaining backward compatibility and code quality. The modular design allows for easy addition of new extraction methods in the future.

**Next Steps**: Begin Phase 1 implementation with base extractor and simple extractors.

# Quick Start Code Examples

This document provides ready-to-use code snippets to jumpstart your implementation.

## 1. Basic setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="social-network-analytics",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Social media network analytics with multilingual NER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/social-network-analytics",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Sociology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sna-web=cli.app:main",
            "sna-cli=cli.cli:main",
        ],
    },
)
```

## 2. Data Loader (src/core/data_loader.py)

```python
"""Data loading utilities for CSV and NDJSON files."""
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional, Tuple
import chardet


class DataLoader:
    """Load and validate social media data files."""
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize DataLoader.
        
        Args:
            chunk_size: Number of rows to read per chunk
        """
        self.chunk_size = chunk_size
    
    def detect_encoding(self, filepath: str, sample_size: int = 10000) -> str:
        """
        Detect file encoding.
        
        Args:
            filepath: Path to file
            sample_size: Bytes to sample for detection
            
        Returns:
            Detected encoding (e.g., 'utf-8', 'latin-1')
        """
        with open(filepath, 'rb') as f:
            raw_data = f.read(sample_size)
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def load_csv(
        self, 
        filepath: str,
        author_col: str,
        text_col: str,
        encoding: Optional[str] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Load CSV file in chunks.
        
        Args:
            filepath: Path to CSV file
            author_col: Name of author column
            text_col: Name of text column
            encoding: File encoding (auto-detect if None)
            
        Yields:
            DataFrame chunks with validated columns
        """
        if encoding is None:
            encoding = self.detect_encoding(filepath)
        
        try:
            # Try reading with detected encoding
            reader = pd.read_csv(
                filepath,
                chunksize=self.chunk_size,
                encoding=encoding,
                on_bad_lines='skip'  # Skip malformed lines
            )
        except UnicodeDecodeError:
            # Fallback to latin-1 which never fails
            reader = pd.read_csv(
                filepath,
                chunksize=self.chunk_size,
                encoding='latin-1',
                on_bad_lines='skip'
            )
        
        for chunk in reader:
            # Strip whitespace from column names
            chunk.columns = chunk.columns.str.strip()
            
            # Validate required columns
            self._validate_columns(chunk, author_col, text_col)
            
            # Clean the data
            chunk = self._clean_chunk(chunk, author_col, text_col)
            
            yield chunk
    
    def load_ndjson(
        self,
        filepath: str,
        author_col: str,
        text_col: str,
        encoding: Optional[str] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Load NDJSON file in chunks.
        
        Args:
            filepath: Path to NDJSON file
            author_col: Name of author column
            text_col: Name of text column
            encoding: File encoding (auto-detect if None)
            
        Yields:
            DataFrame chunks with validated columns
        """
        if encoding is None:
            encoding = self.detect_encoding(filepath)
        
        reader = pd.read_json(
            filepath,
            lines=True,
            chunksize=self.chunk_size,
            encoding=encoding
        )
        
        for chunk in reader:
            # Strip whitespace from column names
            chunk.columns = chunk.columns.str.strip()
            
            # Validate required columns
            self._validate_columns(chunk, author_col, text_col)
            
            # Clean the data
            chunk = self._clean_chunk(chunk, author_col, text_col)
            
            yield chunk
    
    def _validate_columns(self, df: pd.DataFrame, author_col: str, text_col: str) -> None:
        """
        Validate that required columns exist.
        
        Args:
            df: DataFrame to validate
            author_col: Name of author column
            text_col: Name of text column
            
        Raises:
            ValueError: If required columns are missing
        """
        missing = []
        if author_col not in df.columns:
            missing.append(author_col)
        if text_col not in df.columns:
            missing.append(text_col)
        
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
    
    def _clean_chunk(self, df: pd.DataFrame, author_col: str, text_col: str) -> pd.DataFrame:
        """
        Clean chunk data.
        
        Args:
            df: DataFrame chunk
            author_col: Name of author column
            text_col: Name of text column
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing author or text
        df = df.dropna(subset=[author_col, text_col])
        
        # Convert to string and strip whitespace
        df[author_col] = df[author_col].astype(str).str.strip()
        df[text_col] = df[text_col].astype(str).str.strip()
        
        # Remove empty strings
        df = df[df[author_col] != '']
        df = df[df[text_col] != '']
        
        # Add post ID if not present
        if 'post_id' not in df.columns:
            df['post_id'] = range(len(df))
        
        return df


# Example usage
if __name__ == "__main__":
    loader = DataLoader(chunk_size=1000)
    
    # Load CSV
    for chunk in loader.load_csv("data.csv", "author", "text"):
        print(f"Loaded {len(chunk)} rows")
        print(chunk.head())
        break
```

## 3. NER Engine (src/core/ner_engine.py)

```python
"""Named Entity Recognition engine using Hugging Face transformers."""
from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm


class NEREngine:
    """Extract named entities from text using transformer models."""
    
    def __init__(
        self,
        model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85
    ):
        """
        Initialize NER engine.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence score for entities
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = -1
            else:
                self.device = 0
        else:
            self.device = -1
        
        # Load model
        print(f"Loading NER model: {model_name}")
        print(f"Device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        self.nlp = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",  # Merge sub-word tokens
            device=self.device
        )
        
        print("Model loaded successfully!")
    
    def extract_entities_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[Dict]]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            show_progress: Show progress bar
            
        Returns:
            List of entity lists, one per input text
        """
        all_results = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting entities")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            try:
                # Run NER on batch
                batch_results = self.nlp(batch)
                
                # Filter by confidence and clean results
                cleaned_results = [
                    self._clean_entities(result)
                    for result in batch_results
                ]
                
                all_results.extend(cleaned_results)
                
                # Clear GPU cache
                if self.device >= 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Add empty results for failed batch
                all_results.extend([[] for _ in batch])
        
        return all_results
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from single text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries
        """
        try:
            results = self.nlp(text)
            return self._clean_entities(results)
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Clean and filter entity results.
        
        Args:
            entities: Raw entity results from pipeline
            
        Returns:
            Cleaned entity list
        """
        cleaned = []
        
        for entity in entities:
            # Filter by confidence
            if entity['score'] < self.confidence_threshold:
                continue
            
            # Standardize entity type labels
            entity_type = entity['entity_group']
            
            # Map to standard labels
            if entity_type in ['PER', 'PERSON']:
                entity_type = 'PER'
            elif entity_type in ['LOC', 'LOCATION']:
                entity_type = 'LOC'
            elif entity_type in ['ORG', 'ORGANIZATION']:
                entity_type = 'ORG'
            else:
                # Skip MISC and other types
                continue
            
            cleaned.append({
                'text': entity['word'].strip(),
                'type': entity_type,
                'score': entity['score'],
                'start': entity['start'],
                'end': entity['end']
            })
        
        return cleaned


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = NEREngine()
    
    # Single text
    text = "John Smith works at Microsoft in Copenhagen."
    entities = engine.extract_entities(text)
    print("Entities:", entities)
    
    # Batch processing
    texts = [
        "Angela Merkel visited Paris last week.",
        "Apple Inc. announced a new product in California.",
        "The president will meet with officials in Washington."
    ]
    results = engine.extract_entities_batch(texts)
    for text, entities in zip(texts, results):
        print(f"\nText: {text}")
        print(f"Entities: {entities}")
```

## 4. Entity Resolver (src/core/entity_resolver.py)

```python
"""Entity resolution and deduplication."""
from typing import Dict, Set, Optional
import re
from Levenshtein import ratio


class EntityResolver:
    """Resolve and deduplicate named entities."""
    
    def __init__(self, fuzzy_threshold: float = 0.9):
        """
        Initialize resolver.
        
        Args:
            fuzzy_threshold: Minimum similarity for fuzzy matching (0-1)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.entity_map: Dict[str, str] = {}  # normalized -> canonical
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for matching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation at start/end
        text = text.strip('.,!?;:')
        
        return text
    
    def get_canonical_form(self, entity_text: str) -> str:
        """
        Get canonical form of entity.
        
        Args:
            entity_text: Entity text to resolve
            
        Returns:
            Canonical form (or original if new)
        """
        normalized = self.normalize_text(entity_text)
        
        # Check for exact match
        if normalized in self.entity_map:
            return self.entity_map[normalized]
        
        # Check for fuzzy match
        for known_norm, canonical in self.entity_map.items():
            similarity = ratio(normalized, known_norm)
            if similarity >= self.fuzzy_threshold:
                # Map this variant to same canonical form
                self.entity_map[normalized] = canonical
                return canonical
        
        # New entity - use original text as canonical
        self.entity_map[normalized] = entity_text
        return entity_text
    
    def is_author_mention(self, author_name: str, entity_text: str) -> bool:
        """
        Check if entity matches an author name.
        
        Args:
            author_name: Author name/handle
            entity_text: Extracted entity text
            
        Returns:
            True if entity likely refers to author
        """
        # Normalize both
        norm_author = self.normalize_text(author_name)
        norm_entity = self.normalize_text(entity_text)
        
        # Remove @ symbol from handles
        norm_author = norm_author.lstrip('@')
        norm_entity = norm_entity.lstrip('@')
        
        # Check if one contains the other
        if norm_author in norm_entity or norm_entity in norm_author:
            return True
        
        # Check fuzzy similarity
        if ratio(norm_author, norm_entity) >= self.fuzzy_threshold:
            return True
        
        # Check if entity is substring of author (e.g., "Smith" in "John Smith")
        author_parts = norm_author.split()
        entity_parts = norm_entity.split()
        
        # If entity is single word and matches any author name part
        if len(entity_parts) == 1 and entity_parts[0] in author_parts:
            return True
        
        return False
    
    def reset(self):
        """Clear all cached entity mappings."""
        self.entity_map.clear()


# Example usage
if __name__ == "__main__":
    resolver = EntityResolver()
    
    # Test normalization
    print(resolver.get_canonical_form("John Smith"))
    print(resolver.get_canonical_form("john smith"))  # Should return "John Smith"
    print(resolver.get_canonical_form("John  Smith"))  # Should return "John Smith"
    
    # Test author matching
    print(resolver.is_author_mention("@johndoe", "John Doe"))  # True
    print(resolver.is_author_mention("Jane Smith", "Smith"))  # True
    print(resolver.is_author_mention("Alice", "Bob"))  # False
```

## 5. Network Builder (src/core/network_builder.py)

```python
"""Network construction from entity mentions."""
import networkx as nx
from typing import List, Dict, Optional, Set
from collections import defaultdict
from .entity_resolver import EntityResolver


class NetworkBuilder:
    """Build social network from author-entity mentions."""
    
    def __init__(self, use_fuzzy_matching: bool = True):
        """
        Initialize network builder.
        
        Args:
            use_fuzzy_matching: Enable fuzzy entity matching
        """
        self.graph = nx.DiGraph()
        self.resolver = EntityResolver() if use_fuzzy_matching else None
        self.author_set: Set[str] = set()
        
        # Track statistics
        self.posts_processed = 0
        self.entities_extracted = 0
    
    def add_author(self, author_name: str) -> None:
        """
        Add author node to graph.
        
        Args:
            author_name: Author identifier
        """
        if author_name not in self.graph:
            self.graph.add_node(
                author_name,
                node_type='author',
                label=author_name,
                post_count=0,
                mention_count=0
            )
            self.author_set.add(author_name)
    
    def add_post(
        self,
        author: str,
        entities: List[Dict],
        post_id: Optional[str] = None
    ) -> None:
        """
        Add post information to network.
        
        Args:
            author: Post author
            entities: List of extracted entities
            post_id: Optional post identifier
        """
        # Add author node
        self.add_author(author)
        
        # Increment post count
        self.graph.nodes[author]['post_count'] += 1
        self.posts_processed += 1
        
        # Process each entity
        for entity in entities:
            entity_text = entity['text']
            entity_type = entity['type']
            
            # Resolve entity to canonical form
            if self.resolver:
                canonical_entity = self.resolver.get_canonical_form(entity_text)
            else:
                canonical_entity = entity_text
            
            # Check if entity is another author
            is_author_mention = False
            if self.resolver:
                for known_author in self.author_set:
                    if self.resolver.is_author_mention(known_author, entity_text):
                        canonical_entity = known_author
                        is_author_mention = True
                        break
            
            # Determine node type
            if is_author_mention:
                node_type = 'author'
                edge_entity_type = 'AUTHOR'
            else:
                node_type = entity_type.lower()  # 'per', 'loc', 'org'
                edge_entity_type = entity_type
            
            # Add entity node if not exists
            if canonical_entity not in self.graph:
                self.graph.add_node(
                    canonical_entity,
                    node_type=node_type,
                    label=canonical_entity,
                    mention_count=0
                )
            
            # Increment mention count
            self.graph.nodes[canonical_entity]['mention_count'] += 1
            self.entities_extracted += 1
            
            # Add or update edge
            if self.graph.has_edge(author, canonical_entity):
                # Increment weight
                self.graph[author][canonical_entity]['weight'] += 1
                
                # Add post ID to list
                if post_id:
                    self.graph[author][canonical_entity]['source_posts'].append(post_id)
            else:
                # Create new edge
                self.graph.add_edge(
                    author,
                    canonical_entity,
                    weight=1,
                    entity_type=edge_entity_type,
                    source_posts=[post_id] if post_id else []
                )
    
    def get_statistics(self) -> Dict:
        """
        Calculate network statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'posts_processed': self.posts_processed,
            'entities_extracted': self.entities_extracted,
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'authors': sum(1 for _, data in self.graph.nodes(data=True) 
                          if data['node_type'] == 'author'),
            'persons': sum(1 for _, data in self.graph.nodes(data=True) 
                          if data['node_type'] == 'per'),
            'locations': sum(1 for _, data in self.graph.nodes(data=True) 
                            if data['node_type'] == 'loc'),
            'organizations': sum(1 for _, data in self.graph.nodes(data=True) 
                                if data['node_type'] == 'org'),
        }
        
        # Calculate density
        if stats['total_nodes'] > 1:
            stats['density'] = nx.density(self.graph)
        else:
            stats['density'] = 0.0
        
        # Get top mentioned entities
        mentions = [(node, data['mention_count']) 
                   for node, data in self.graph.nodes(data=True)
                   if data['node_type'] != 'author']
        mentions.sort(key=lambda x: x[1], reverse=True)
        stats['top_mentioned'] = mentions[:10]
        
        return stats
    
    def get_graph(self) -> nx.DiGraph:
        """
        Get the constructed network.
        
        Returns:
            NetworkX DiGraph
        """
        return self.graph


# Example usage
if __name__ == "__main__":
    builder = NetworkBuilder()
    
    # Simulate adding posts
    builder.add_post(
        author="@user1",
        entities=[
            {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
            {'text': 'Microsoft', 'type': 'ORG', 'score': 0.90}
        ],
        post_id="post_001"
    )
    
    builder.add_post(
        author="@user2",
        entities=[
            {'text': 'John Smith', 'type': 'PER', 'score': 0.92},
            {'text': 'New York', 'type': 'LOC', 'score': 0.88}
        ],
        post_id="post_002"
    )
    
    # Get statistics
    stats = builder.get_statistics()
    print("Network Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get graph
    G = builder.get_graph()
    print(f"\nNodes: {list(G.nodes(data=True))[:3]}")
    print(f"Edges: {list(G.edges(data=True))[:3]}")
```

## 6. Basic Streamlit App (src/cli/app.py)

```python
"""Streamlit web interface for social network analytics."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.ner_engine import NEREngine
from core.network_builder import NetworkBuilder
from utils.exporters import export_graphml, export_statistics


def main():
    st.set_page_config(
        page_title="Social Network Analytics",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 Social Network Analytics")
    st.markdown("Extract social networks from social media posts using NER")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    model_name = st.sidebar.selectbox(
        "NER Model",
        [
            "Davlan/xlm-roberta-base-ner-hrl",
            "Babelscape/wikineural-multilingual-ner"
        ]
    )
    
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.85,
        step=0.05
    )
    
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=8,
        max_value=128,
        value=32,
        step=8
    )
    
    # File upload
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or NDJSON file",
        type=['csv', 'ndjson']
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Detect file type
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'ndjson'
        
        # Preview data
        st.subheader("Data Preview")
        if file_type == 'csv':
            preview_df = pd.read_csv(temp_path, nrows=5)
        else:
            preview_df = pd.read_json(temp_path, lines=True, nrows=5)
        
        st.dataframe(preview_df)
        
        # Column selection
        st.header("2. Select Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            author_col = st.selectbox(
                "Author Column",
                options=preview_df.columns
            )
        
        with col2:
            text_col = st.selectbox(
                "Text Column",
                options=preview_df.columns
            )
        
        # Entity type selection
        st.header("3. Entity Types")
        st.write("All entity types (PER, LOC, ORG) will be extracted")
        
        # Process button
        st.header("4. Process Data")
        if st.button("🚀 Start Processing", type="primary"):
            process_data(
                temp_path,
                file_type,
                author_col,
                text_col,
                model_name,
                confidence,
                batch_size
            )


def process_data(
    filepath: str,
    file_type: str,
    author_col: str,
    text_col: str,
    model_name: str,
    confidence: float,
    batch_size: int
):
    """Process uploaded data and build network."""
    
    try:
        # Initialize components
        with st.spinner("Loading NER model..."):
            engine = NEREngine(
                model_name=model_name,
                confidence_threshold=confidence
            )
        
        st.success("Model loaded!")
        
        loader = DataLoader(chunk_size=10000)
        builder = NetworkBuilder()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load and process data
        total_posts = 0
        processed_posts = 0
        
        # Get total row count for progress
        if file_type == 'csv':
            total_posts = sum(1 for _ in open(filepath)) - 1  # Subtract header
            chunks = loader.load_csv(filepath, author_col, text_col)
        else:
            total_posts = sum(1 for _ in open(filepath))
            chunks = loader.load_ndjson(filepath, author_col, text_col)
        
        # Process chunks
        for chunk in chunks:
            # Extract texts
            texts = chunk[text_col].tolist()
            authors = chunk[author_col].tolist()
            post_ids = chunk.get('post_id', range(len(chunk))).tolist()
            
            # Run NER
            entities_batch = engine.extract_entities_batch(
                texts,
                batch_size=batch_size,
                show_progress=False
            )
            
            # Build network
            for author, entities, post_id in zip(authors, entities_batch, post_ids):
                builder.add_post(author, entities, str(post_id))
            
            # Update progress
            processed_posts += len(chunk)
            progress = min(processed_posts / total_posts, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processed {processed_posts:,} / {total_posts:,} posts")
        
        st.success(f"✅ Processing complete! Processed {processed_posts:,} posts")
        
        # Display statistics
        st.header("5. Results")
        stats = builder.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", f"{stats['total_nodes']:,}")
        with col2:
            st.metric("Total Edges", f"{stats['total_edges']:,}")
        with col3:
            st.metric("Authors", f"{stats['authors']:,}")
        with col4:
            st.metric("Entities", f"{stats['entities_extracted']:,}")
        
        # Top mentioned entities
        st.subheader("Top Mentioned Entities")
        if stats['top_mentioned']:
            top_df = pd.DataFrame(
                stats['top_mentioned'],
                columns=['Entity', 'Mentions']
            )
            st.dataframe(top_df, use_container_width=True)
        
        # Export options
        st.header("6. Download Results")
        
        graph = builder.get_graph()
        output_dir = Path("/tmp/sna_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export GraphML
        graphml_path = output_dir / "network.graphml"
        export_graphml(graph, str(graphml_path))
        
        with open(graphml_path, 'rb') as f:
            st.download_button(
                label="📥 Download GraphML (for Gephi)",
                data=f,
                file_name="network.graphml",
                mime="application/xml"
            )
        
        # Export statistics
        stats_path = output_dir / "statistics.json"
        export_statistics(stats, str(stats_path))
        
        with open(stats_path, 'rb') as f:
            st.download_button(
                label="📊 Download Statistics (JSON)",
                data=f,
                file_name="statistics.json",
                mime="application/json"
            )
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
```

## 7. requirements.txt

```
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
pandas>=2.0.0
networkx>=3.0
numpy>=1.24.0

# Web interface
streamlit>=1.25.0

# NLP utilities
langdetect>=1.0.9
python-Levenshtein>=0.21.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
chardet>=5.0.0

# Development (optional)
pytest>=7.0
pytest-cov>=4.0
black>=23.0
flake8>=6.0
```

## 8. README.md Template

```markdown
# Social Network Analytics

Extract social networks from social media posts using multilingual Named Entity Recognition (NER).

## Features

- 🌍 Multilingual NER (Danish, English, and 8 other languages)
- 📊 Network construction (author → entity mentions)
- 🚀 GPU-accelerated processing
- 📁 Support for CSV and NDJSON files
- 📈 Multiple export formats (GraphML, GEXF, JSON)
- 🖥️ Easy-to-use web interface

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/social-network-analytics.git
cd social-network-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Quick Start

### Web Interface

```bash
sna-web
```

Then open your browser to `http://localhost:8501`

### Command Line

```bash
sna-cli data.csv --author-col "author" --text-col "text" --output-dir ./output
```

### Python API

```python
from social_network_analytics import process_social_media_data

graph, stats = process_social_media_data(
    filepath="data.csv",
    author_column="author",
    text_column="text",
    output_dir="./output"
)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB RAM minimum
- 10GB disk space for models

## License

MIT License - see LICENSE file for details
```

---

These code examples provide a solid foundation to start building your social network analytics library. Each module is self-contained and includes example usage. Start by implementing these core components, then add the additional features as needed.

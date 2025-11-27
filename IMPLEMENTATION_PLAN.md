# Social Network Analytics - Implementation Plan

## Overview
This document provides a step-by-step implementation plan for building the social media network analytics library as specified in `.clinerules`.

## Timeline Estimate
- **Phase 1** (Core): 2-3 weeks
- **Phase 2** (UI): 1 week  
- **Phase 3** (Polish): 1 week
- **Total**: 4-5 weeks for full prototype

---

## Phase 1: Core Functionality (Weeks 1-3)

### Step 1.1: Project Setup (Day 1)
**Objective**: Initialize project structure and dependencies

**Tasks**:
1. Create directory structure as specified in `.clinerules`
2. Initialize git repository
3. Create `setup.py` with package metadata
4. Create `requirements.txt` with core dependencies
5. Create basic `README.md`
6. Set up virtual environment

**Code Template - setup.py**:
```python
from setuptools import setup, find_packages

setup(
    name='social-network-analytics',
    version='0.1.0',
    description='Social media network analytics with NER',
    author='Your Name',
    python_requires='>=3.9',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'pandas>=2.0.0',
        'networkx>=3.0',
        'streamlit>=1.25.0',
        'langdetect>=1.0.9',
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'python-levenshtein>=0.21.0',
    ],
    entry_points={
        'console_scripts': [
            'sna-web=cli.app:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
```

**Deliverable**: Working project skeleton with installable package

---

### Step 1.2: Data Loader Module (Days 2-3)
**Objective**: Implement robust file loading for CSV and NDJSON

**File**: `src/core/data_loader.py`

**Key Functions**:
1. `load_csv(filepath, chunksize=10000)` - Iterator for CSV files
2. `load_ndjson(filepath, chunksize=10000)` - Iterator for NDJSON
3. `detect_encoding(filepath)` - Auto-detect file encoding
4. `validate_columns(df, author_col, text_col)` - Verify required columns

**Implementation Notes**:
- Use `pandas.read_csv()` with `chunksize` for memory efficiency
- For NDJSON, use `pd.read_json(lines=True, chunksize=...)`
- Handle encoding: UTF-8, Latin-1, CP1252
- Strip whitespace from column names
- Handle missing values (drop or fill with empty string)

**Test Cases**:
- Small CSV (100 rows)
- Large CSV (100k+ rows)
- NDJSON with various encodings
- Files with missing values
- Malformed files (should raise clear errors)

**Deliverable**: Tested data loading module that handles both formats efficiently

---

### Step 1.3: NER Engine Module (Days 4-7)
**Objective**: Implement NER extraction using Hugging Face transformers

**File**: `src/core/ner_engine.py`

**Key Classes**:
```python
class NEREngine:
    def __init__(self, model_name, device='cuda', confidence_threshold=0.85):
        """Initialize NER model"""
        
    def extract_entities(self, texts, batch_size=32):
        """
        Extract entities from list of texts
        Returns: List of dicts with entities per text
        """
        
    def _aggregate_entities(self, ner_results):
        """Aggregate sub-word tokens into full entities"""
```

**Recommended Model**: `Davlan/xlm-roberta-base-ner-hrl`
- Supports: English, German, Dutch, Spanish, Italian, French, Polish, Portuguese, Danish, Norwegian
- Entity types: PER, LOC, ORG, MISC
- Good balance of speed and accuracy

**Alternative**: `Babelscape/wikineural-multilingual-ner`
- More languages but slightly larger

**Implementation Steps**:
1. Load model and tokenizer from Hugging Face
2. Implement batching logic for GPU efficiency
3. Handle tokenization and entity aggregation
4. Filter entities by confidence threshold
5. Post-process entities (normalize whitespace, remove punctuation)
6. Cache model locally in `models/` directory

**Key Considerations**:
- Use `pipeline('ner', model=..., aggregation_strategy='simple')`
- Aggregation strategy 'simple' merges sub-tokens (e.g., "##ensen" → "Jensen")
- Clear GPU cache between batches for memory management
- Add progress bar with `tqdm`

**Test Cases**:
- English text with known entities
- Danish text with known entities
- Mixed language text
- Very long posts (>512 tokens - should truncate)
- Empty/null text (should return empty list)

**Deliverable**: Working NER engine that extracts PER/LOC/ORG from multilingual text

---

### Step 1.4: Entity Resolution Module (Days 8-9)
**Objective**: Deduplicate entities and match author names

**File**: `src/core/entity_resolver.py`

**Key Functions**:
```python
class EntityResolver:
    def __init__(self, fuzzy_threshold=0.9):
        """Initialize resolver with fuzzy matching threshold"""
        
    def normalize_entity(self, entity_text):
        """Normalize entity for matching (lowercase, whitespace)"""
        
    def deduplicate_entities(self, entity_list):
        """
        Remove duplicate entities
        Returns: Dict mapping normalized -> canonical form
        """
        
    def match_author_entities(self, author_name, entity_text):
        """Check if entity matches author (with fuzzy matching)"""
```

**Implementation**:
1. Normalize entities: lowercase, strip, collapse whitespace
2. Use dict to map normalized → original (first occurrence)
3. Optional: Levenshtein distance for fuzzy matching
4. For author matching: check if author name substring in entity or vice versa

**Test Cases**:
- Exact duplicates: "John Smith" vs "john smith"
- Whitespace variants: "New York" vs "New  York"
- Fuzzy matches: "Copenhagen" vs "Copenhagn" (typo)
- Author-entity matching: author="@johndoe", entity="John Doe"

**Deliverable**: Entity deduplication and author-matching system

---

### Step 1.5: Network Builder Module (Days 10-12)
**Objective**: Construct directed network from NER results

**File**: `src/core/network_builder.py`

**Key Class**:
```python
class NetworkBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_resolver = EntityResolver()
        
    def add_post(self, author, entities, post_id=None, timestamp=None):
        """Add post data to network"""
        
    def finalize_network(self):
        """Compute statistics and prepare for export"""
        
    def get_statistics(self):
        """Return network statistics dict"""
```

**Network Structure**:
- Nodes: authors + entities (PER/LOC/ORG)
- Node attributes: 
  - `node_type`: 'author', 'person', 'location', 'organization'
  - `label`: display name
  - `mention_count`: number of times mentioned
- Edges: directed from author to entity
- Edge attributes:
  - `weight`: number of mentions
  - `entity_type`: PER/LOC/ORG/AUTHOR
  - `source_posts`: list of post IDs

**Implementation Steps**:
1. For each post, extract author and entities
2. Add author node if not exists
3. For each entity:
   - Normalize and deduplicate
   - Add entity node if not exists
   - Add/update edge from author to entity (increment weight)
4. Special case: if entity matches another author, create author-author edge
5. Store metadata in node/edge attributes

**Test Cases**:
- Single author, multiple entities
- Multiple authors mentioning same entity
- Author mentioning another author
- Edge weight accumulation

**Deliverable**: Complete network construction pipeline

---

### Step 1.6: Export Module (Days 13-14)
**Objective**: Export network in multiple formats

**File**: `src/utils/exporters.py`

**Key Functions**:
```python
def export_graphml(graph, filepath):
    """Export to GraphML (Gephi-compatible)"""
    
def export_gexf(graph, filepath):
    """Export to GEXF format"""
    
def export_json(graph, filepath):
    """Export to node-link JSON (D3.js compatible)"""
    
def export_edgelist(graph, filepath):
    """Export to CSV edge list"""
    
def export_statistics(stats, filepath):
    """Export statistics to JSON"""
```

**Implementation**:
- Use `networkx.write_graphml()` for GraphML
- Use `networkx.write_gexf()` for GEXF
- Use `networkx.node_link_data()` for JSON
- Custom CSV writer for edge list
- Include all node/edge attributes

**Test Cases**:
- Export and re-import in Gephi
- Verify all attributes are preserved
- Test with large networks (10k+ nodes)

**Deliverable**: Multi-format export functionality

---

### Step 1.7: Integration & Pipeline (Day 15)
**Objective**: Connect all modules into end-to-end pipeline

**File**: `src/core/pipeline.py`

**Key Function**:
```python
def process_social_media_data(
    filepath,
    author_column,
    text_column,
    file_format='csv',
    output_dir='./output',
    model_name='Davlan/xlm-roberta-base-ner-hrl',
    batch_size=32,
    confidence_threshold=0.85,
    progress_callback=None
):
    """
    End-to-end processing pipeline
    
    Returns:
        graph: NetworkX graph object
        statistics: Dict of network statistics
    """
```

**Pipeline Steps**:
1. Load data in chunks
2. For each chunk:
   - Extract text and author
   - Run NER on batch of texts
   - Add results to network builder
   - Update progress
3. Finalize network
4. Compute statistics
5. Export to all formats
6. Return graph and stats

**Deliverable**: Working end-to-end pipeline that can be called programmatically

---

## Phase 2: User Interface (Week 4)

### Step 2.1: Streamlit Web Interface (Days 16-19)
**Objective**: Create minimal web UI for non-technical users

**File**: `src/cli/app.py`

**Page Structure**:
1. **Header**: Title, description
2. **Configuration Section**:
   - File uploader (CSV/NDJSON)
   - Author column selector (dropdown)
   - Text column selector (dropdown)
   - Entity type checkboxes (PER, LOC, ORG)
   - Advanced settings (expandable):
     - Confidence threshold slider
     - Batch size input
     - Model selection dropdown
3. **Processing Section**:
   - "Process Data" button
   - Progress bar
   - Status messages
4. **Results Section**:
   - Network statistics (cards/metrics)
   - Download buttons (GraphML, GEXF, JSON, CSV)
   - Basic network visualization (optional, using networkx + matplotlib)

**Key Streamlit Components**:
```python
import streamlit as st
import pandas as pd

st.title("Social Network Analytics")
st.write("Extract social networks from social media posts")

uploaded_file = st.file_uploader("Upload CSV or NDJSON", type=['csv', 'ndjson'])

if uploaded_file:
    # Preview data
    df_preview = pd.read_csv(uploaded_file, nrows=10)
    st.write("Data Preview:", df_preview)
    
    # Column selectors
    author_col = st.selectbox("Select author column", df_preview.columns)
    text_col = st.selectbox("Select text column", df_preview.columns)
    
    # Process button
    if st.button("Process Data"):
        with st.spinner("Processing..."):
            # Run pipeline
            progress_bar = st.progress(0)
            # ... processing logic ...
            st.success("Processing complete!")
```

**Implementation Notes**:
- Use `st.session_state` to cache model loading
- Display processing time
- Show sample of extracted entities
- Provide download buttons using `st.download_button()`

**Deliverable**: Functional web interface accessible via browser

---

### Step 2.2: Command-Line Interface (Days 19-20)
**Objective**: Create CLI for power users

**File**: `src/cli/cli.py`

**Implementation**:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Social Network Analytics CLI'
    )
    parser.add_argument('input_file', help='Path to CSV or NDJSON file')
    parser.add_argument('--author-col', required=True, help='Author column name')
    parser.add_argument('--text-col', required=True, help='Text column name')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--format', choices=['csv', 'ndjson'], default='csv')
    parser.add_argument('--model', default='Davlan/xlm-roberta-base-ner-hrl')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--confidence', type=float, default=0.85)
    
    args = parser.parse_args()
    
    # Run pipeline
    process_social_media_data(
        filepath=args.input_file,
        author_column=args.author_col,
        text_column=args.text_col,
        # ... other args
    )
    
if __name__ == '__main__':
    main()
```

**Deliverable**: CLI tool for batch processing

---

## Phase 3: Polish & Testing (Week 5)

### Step 3.1: Error Handling (Days 21-22)
**Tasks**:
1. Add try-except blocks throughout
2. Create custom exception classes
3. Log errors to file
4. Display user-friendly error messages
5. Create error report export

**Key Areas**:
- File reading errors (encoding, format)
- Column not found errors
- GPU/CUDA errors (fallback to CPU)
- Model loading errors
- Network export errors

---

### Step 3.2: Testing (Days 22-24)
**Tasks**:
1. Write unit tests for each module
2. Create integration tests
3. Generate test data (Danish + English)
4. Test edge cases
5. Performance testing with large files

**Test Structure**:
```
tests/
├── test_data_loader.py
├── test_ner_engine.py
├── test_entity_resolver.py
├── test_network_builder.py
├── test_exporters.py
├── test_pipeline.py
└── fixtures/
    ├── sample_small.csv
    ├── sample_danish.csv
    └── sample_multilingual.ndjson
```

---

### Step 3.3: Documentation (Day 25)
**Tasks**:
1. Complete README.md with:
   - Installation guide
   - Quick start
   - Usage examples
   - Configuration guide
2. Add docstrings to all functions
3. Create example Jupyter notebook
4. Add inline comments
5. Create CHANGELOG.md

---

## Development Best Practices

### Code Quality
- Use type hints throughout
- Follow PEP 8 style guide
- Use Black for code formatting
- Use pylint for linting
- Aim for 80%+ test coverage

### Git Workflow
- Use feature branches
- Meaningful commit messages
- Tag releases (v0.1.0, etc.)

### Performance Optimization
- Profile code with `cProfile`
- Monitor GPU memory usage
- Use batch processing for all operations
- Cache model and intermediate results

### Debugging
- Add verbose logging mode
- Use logging module (not print statements)
- Include timing information
- Create debug mode for step-by-step execution

---

## Deployment Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Example data included
- [ ] Requirements.txt finalized
- [ ] Setup.py tested on fresh environment
- [ ] README instructions verified

### Release
- [ ] Tag release in git
- [ ] Create GitHub release
- [ ] Upload to PyPI (optional)
- [ ] Create demo video
- [ ] Deploy demo instance (optional)

---

## Troubleshooting Guide

### Common Issues

**Issue**: Model download fails
- **Solution**: Pre-download model, provide local path option

**Issue**: GPU out of memory
- **Solution**: Reduce batch size, add automatic fallback

**Issue**: Encoding errors in CSV
- **Solution**: Try multiple encodings, use chardet library

**Issue**: Network too large to visualize
- **Solution**: Add filtering options, export subgraphs

**Issue**: NER missing obvious entities
- **Solution**: Lower confidence threshold, try different model

---

## Performance Benchmarks

**Target Performance** (on NVIDIA RTX 3080):
- 10,000 posts: < 5 minutes
- 100,000 posts: < 30 minutes
- 1,000,000 posts: < 5 hours

**Optimization Strategies**:
1. Increase batch size (32 → 64)
2. Use mixed precision (FP16)
3. Parallelize file reading and NER processing
4. Cache entity normalization results

---

## Future Roadmap (Post-Prototype)

### Version 0.2.0
- Database backend (SQLite/PostgreSQL)
- Advanced entity resolution (ML-based coreference)
- Temporal network analysis
- Interactive visualization with Plotly

### Version 0.3.0
- REST API
- Docker containerization
- Multi-user support
- Real-time processing

### Version 1.0.0
- Production-ready
- Advanced analytics (centrality, community detection)
- Sentiment analysis integration
- Platform-specific data collectors

---

## Resources

### Documentation Links
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Recommended Models
1. **Davlan/xlm-roberta-base-ner-hrl** (recommended)
   - 10 high-resource languages including Danish
   - F1: ~90% on English, ~85% on Danish
   
2. **Babelscape/wikineural-multilingual-ner**
   - 9 languages, Wikipedia-trained
   - Good for rare entities

3. **xlm-roberta-large-finetuned-conll03-english**
   - English-focused but works multilingually
   - Higher accuracy but slower

### Testing Data Sources
- [Danish Twitter Corpus](https://github.com/certainlyio/nordic_bert)
- [Multilingual Named Entity Resources](https://github.com/dice-group/FOX)

---

## Contact & Support

For questions or issues:
1. Check documentation
2. Review example notebooks
3. Search existing GitHub issues
4. Create new issue with:
   - Python version
   - GPU info
   - Error message
   - Sample data (if possible)

---

**End of Implementation Plan**

This plan provides a structured approach to building the social network analytics library. Adjust timelines based on your expertise and available time. Focus on getting Phase 1 working robustly before moving to UI development.

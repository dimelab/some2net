# Social Network Analytics (SNA)

A Python library for constructing social networks from social media posts using Named Entity Recognition (NER). Extract mentions of persons, locations, and organizations from multilingual text and generate network graphs for analysis in Gephi or other network analysis tools.

## Features

- 🌍 **Multilingual NER** - Supports 10+ languages including Danish and English
- 📊 **Network Construction** - Automatic graph building from entity mentions
- 🚀 **GPU Acceleration** - Fast processing with CUDA support
- 💾 **Smart Caching** - Disk-based caching for faster reprocessing
- 📈 **Interactive Visualization** - Force Atlas 2 layout with Plotly
- 📁 **Multiple Export Formats** - GEXF (primary), GraphML, JSON, CSV
- 🎯 **Batch Processing** - Memory-efficient handling of large datasets
- 🌐 **Web Interface** - Easy-to-use Streamlit UI
- 📝 **Language Detection** - Automatic per-post language identification

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 6GB+ GPU VRAM (for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/social-network-analytics.git
cd social-network-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test import
python -c "from core.ner_engine import NEREngine; print('Success!')"
```

## Quick Start

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run src/cli/app.py
```

Then:
1. Upload your CSV or NDJSON file
2. Select author and text columns
3. Choose entity types (PER, LOC, ORG)
4. Click "Process Data"
5. Download network in GEXF format

### Python API

```python
from core.data_loader import DataLoader
from core.ner_engine import NEREngine
from core.entity_resolver import EntityResolver
from core.network_builder import NetworkBuilder
from utils.exporters import export_gexf

# Initialize components
loader = DataLoader()
engine = NEREngine(enable_cache=True)
resolver = EntityResolver()
builder = NetworkBuilder()

# Load data
data_chunks = loader.load_csv(
    "data.csv",
    author_column="author",
    text_column="text",
    chunksize=10000
)

# Process each chunk
for chunk in data_chunks:
    # Extract entities
    entities_batch, languages = engine.extract_entities_batch(
        chunk['text'].tolist(),
        detect_languages=True
    )

    # Build network
    for idx, row in chunk.iterrows():
        builder.add_post(
            author=row['author'],
            entities=entities_batch[idx],
            post_id=row.get('post_id'),
        )

# Export network
graph = builder.finalize_network()
export_gexf(graph, "output/network.gexf")

# Get statistics
stats = builder.get_statistics()
print(f"Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
```

## Input Data Format

### CSV Format

```csv
author,text,post_id,timestamp
@user1,"John Smith announced Microsoft's new office in Copenhagen",1,2024-01-01
@user2,"Angela Merkel visited Paris to meet Emmanuel Macron",2,2024-01-02
```

Required columns:
- **Author column**: User identifier or author name
- **Text column**: Post content for NER extraction

Optional columns:
- **post_id**: Unique identifier for posts
- **timestamp**: For temporal analysis

### NDJSON Format

```json
{"author": "@user1", "text": "John Smith works at Microsoft", "post_id": "1"}
{"author": "@user2", "text": "Meeting in Copenhagen tomorrow", "post_id": "2"}
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "Davlan/xlm-roberta-base-ner-hrl"
  confidence_threshold: 0.85

processing:
  batch_size: 32
  chunk_size: 10000

cache:
  enabled: true
  max_age_days: 30
```

## Output Formats

### GEXF (Primary Format)
Gephi's native format, recommended for network analysis:
```python
from utils.exporters import export_gexf
export_gexf(graph, "network.gexf")
```

### GraphML
NetworkX standard format:
```python
from utils.exporters import export_graphml
export_graphml(graph, "network.graphml")
```

### JSON (D3.js compatible)
```python
from utils.exporters import export_json
export_json(graph, "network.json")
```

### CSV Edge List
```python
from utils.exporters import export_edgelist
export_edgelist(graph, "network.csv")
```

## Network Structure

### Node Types
- **Author nodes**: Post authors (blue)
- **Person nodes**: Extracted PER entities (orange)
- **Location nodes**: Extracted LOC entities (green)
- **Organization nodes**: Extracted ORG entities (red)

### Edge Types
- **Author → Entity**: Directed edges when author mentions entity
- **Author → Author**: When one author mentions another's name
- **Edge weights**: Number of mentions (frequency)

### Node Attributes
- `node_type`: 'author', 'person', 'location', 'organization'
- `label`: Display name
- `mention_count`: Total times mentioned

### Edge Attributes
- `weight`: Mention frequency
- `entity_type`: PER/LOC/ORG/AUTHOR
- `source_posts`: List of post IDs

## Performance

### Benchmarks (NVIDIA RTX 3080)
- 10,000 posts: ~2-5 minutes
- 100,000 posts: ~20-30 minutes
- 1,000,000 posts: ~3-5 hours

### Memory Usage
- Model loading: ~1GB GPU memory
- Processing batch (32): ~2GB GPU memory
- Network (100k nodes): ~500MB RAM

### Optimization Tips
1. Increase `batch_size` if you have more GPU memory
2. Enable `use_fp16` for faster inference
3. Use caching for repeated processing
4. Reduce `chunk_size` if running out of RAM

## Troubleshooting

### CUDA Out of Memory
```yaml
# In config.yaml, reduce batch size
processing:
  batch_size: 16  # or 8
```

### Model Download Issues
```python
# Manually download model
from transformers import AutoModelForTokenClassification, AutoTokenizer
model_name = "Davlan/xlm-roberta-base-ner-hrl"
model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir="./models")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
```

### Encoding Errors
The data loader automatically tries multiple encodings (UTF-8, Latin-1, CP1252). For persistent issues:
```python
loader = DataLoader()
df = loader.load_csv("file.csv", encoding='iso-8859-1')
```

## Error Handling

The library includes comprehensive error handling with user-friendly messages:

### Custom Exceptions

```python
from src.core.exceptions import (
    FileNotFoundError,
    ColumnNotFoundError,
    NERProcessingError,
    format_error_for_user
)

try:
    # Your code here
    process_data("data.csv", "author", "text")

except FileNotFoundError as e:
    print(format_error_for_user(e, include_details=True))
    # Output: ❌ Error: File not found: data.csv
    #            Please check the file path and ensure the file exists.

except ColumnNotFoundError as e:
    print(format_error_for_user(e, include_details=True))
    # Output: ❌ Error: Column 'author' not found in data
    #            Available columns: 'user', 'text', 'timestamp'
```

### Error Tracking

Track and export errors during processing:

```python
from src.utils.logger import ErrorTracker

tracker = ErrorTracker()

for post in posts:
    try:
        process_post(post)
    except Exception as e:
        tracker.add_error(e, context="post processing", post_id=post.id)
        continue

# Export error report
if tracker.has_errors():
    tracker.export_to_json("./output/errors.json")
    tracker.export_to_text("./output/errors.txt")
    print(f"Completed with {len(tracker)} errors")
```

### Logging

Centralized logging with file and console output:

```python
from src.utils.logger import setup_logger

logger = setup_logger("my_app", level="INFO", log_dir="./logs")

logger.info("Processing started")
logger.warning("Low confidence in entity detection")
logger.error("Failed to process post #12345")
```

See [ERROR_HANDLING_GUIDE.md](ERROR_HANDLING_GUIDE.md) for complete documentation.

## Testing

The library includes 400+ tests with comprehensive coverage:

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit -v              # Unit tests only
pytest -m integration -v       # Integration tests only
pytest -m "not slow" -v        # Skip slow tests
pytest -m edge_case -v         # Edge case tests
pytest -m performance -v       # Performance benchmarks

# With coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing tests/
```

### Test Categories

- **Unit Tests**: Individual module testing (250+ tests)
- **Integration Tests**: End-to-end pipeline testing (35+ tests)
- **Edge Case Tests**: Boundary conditions and unusual inputs (48 tests)
- **Error Handling Tests**: Exception and error tracking (42 tests)
- **Performance Tests**: Benchmarks and scalability (16 tests)

### Performance Benchmarks

- ✅ 1,000 posts: < 5 seconds
- ✅ 10,000 posts: < 10 seconds
- ✅ Throughput: > 100 posts/second
- ✅ Linear scaling verified

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed testing documentation.

## Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run full test suite
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test files
pytest tests/test_error_handling.py -v
pytest tests/test_edge_cases.py -v
pytest tests/test_performance.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
pylint src/

# Type checking
mypy src/
```

## Project Structure

```
social-network-analytics/
├── src/
│   ├── core/                    # Core processing modules
│   │   ├── data_loader.py       # CSV/NDJSON loading
│   │   ├── ner_engine.py        # Named entity recognition
│   │   ├── entity_resolver.py   # Entity deduplication
│   │   ├── network_builder.py   # Graph construction
│   │   ├── pipeline.py          # End-to-end pipeline
│   │   └── exceptions.py        # Custom exceptions
│   ├── utils/                   # Utility modules
│   │   ├── exporters.py         # Network export
│   │   ├── visualizer.py        # Visualization
│   │   └── logger.py            # Logging & error tracking
│   └── cli/                     # User interfaces
│       ├── app.py               # Streamlit web UI
│       └── cli.py               # Command-line tool
├── tests/                       # Test suite (400+ tests)
│   ├── conftest.py              # Shared fixtures
│   ├── test_error_handling.py   # Error handling tests
│   ├── test_edge_cases.py       # Edge case tests
│   ├── test_performance.py      # Performance benchmarks
│   └── test_*.py                # Module-specific tests
├── examples/                    # Example data and scripts
│   ├── sample_data.csv          # Example CSV
│   ├── test_*.py                # Example usage
│   └── usage_example.ipynb      # Jupyter notebook
├── models/                      # Model cache (auto-created)
├── cache/                       # NER results cache
├── logs/                        # Application logs
├── output/                      # Export output directory
├── config.yaml                  # Configuration file
├── pytest.ini                   # Test configuration
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── ERROR_HANDLING_GUIDE.md      # Error handling docs
├── TESTING_GUIDE.md             # Testing documentation
├── CHANGELOG.md                 # Version history
└── setup.py                     # Package setup
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{social_network_analytics,
  author = {Your Name},
  title = {Social Network Analytics: NER-based Social Media Network Construction},
  year = {2024},
  url = {https://github.com/yourusername/social-network-analytics}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for NER models
- [NetworkX](https://networkx.org/) for network analysis
- [Streamlit](https://streamlit.io/) for web interface
- NER model: [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl)

## Support

- Documentation: See `/docs` folder
- Issues: [GitHub Issues](https://github.com/yourusername/social-network-analytics/issues)
- Email: your.email@example.com

## Roadmap

- [ ] Database backend for persistent storage
- [ ] Advanced entity resolution with coreference
- [ ] Temporal network analysis
- [ ] Sentiment analysis integration
- [ ] REST API
- [ ] Docker containerization
- [ ] Community detection algorithms
- [ ] Real-time processing support

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See CONTRIBUTING.md for detailed guidelines.

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

## Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
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
│   ├── core/              # Core processing modules
│   │   ├── data_loader.py
│   │   ├── ner_engine.py
│   │   ├── entity_resolver.py
│   │   └── network_builder.py
│   ├── utils/             # Utility modules
│   │   ├── exporters.py
│   │   └── visualizer.py
│   └── cli/               # User interfaces
│       └── app.py
├── tests/                 # Test suite
├── examples/              # Example data and scripts
├── models/                # Model cache
├── cache/                 # NER results cache
├── logs/                  # Application logs
├── config.yaml            # Configuration
└── requirements.txt       # Dependencies
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

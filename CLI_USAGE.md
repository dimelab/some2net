# CLI Usage Guide

## Social Network Analytics - Command-Line Interface

The `sna-cli` tool provides a powerful command-line interface for batch processing social media data and extracting social networks using Named Entity Recognition.

## Installation

```bash
# Install the package
pip install -e .

# Or install from requirements
pip install -r requirements.txt

# The CLI will be available as 'sna-cli'
sna-cli --help
```

## Basic Usage

### Minimal Command

```bash
sna-cli input.csv --author username --text tweet_text
```

This will:
- Process `input.csv`
- Extract entities from `tweet_text` column
- Create network with authors from `username` column
- Save results to `./output/` directory

### Common Usage

```bash
sna-cli tweets.csv \
  --author username \
  --text text \
  --output ./results \
  --model Davlan/xlm-roberta-base-ner-hrl \
  --confidence 0.85
```

## Command-Line Arguments

### Required Arguments

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `input_file` | - | Path to CSV or NDJSON file | `tweets.csv` |
| `--author` | `-a` | Author/username column name | `--author username` |
| `--text` | `-t` | Text/content column name | `--text tweet_text` |

### Output Options

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--output` | `-o` | `./output` | Output directory path |
| `--format` | `-f` | `auto` | File format: `csv`, `ndjson`, `auto` |
| `--export-formats` | - | `all` | Export formats: `gexf`, `graphml`, `json`, `edgelist`, `statistics`, `all` |

### Model Options

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | `Davlan/xlm-roberta-base-ner-hrl` | HuggingFace NER model |
| `--confidence` | `-c` | `0.85` | Minimum confidence threshold (0.0-1.0) |

### Processing Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` `-b` | `32` | Batch size for NER (8-128) |
| `--chunk-size` | `10000` | Rows per chunk (memory management) |

### Feature Toggles

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-cache` | Enabled | Disable NER result caching |
| `--no-entity-resolver` | Enabled | Disable entity deduplication |
| `--no-author-edges` | Enabled | Disable author-to-author edges |
| `--no-language-detection` | Enabled | Disable language detection |

### Display Options

| Argument | Short | Description |
|----------|-------|-------------|
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress all output except errors |
| `--progress` | - | Show progress bar (default: enabled) |
| `--no-progress` | - | Hide progress bar |
| `--version` | - | Show version and exit |

## Examples

### Example 1: Basic Processing

Process a CSV file with default settings:

```bash
sna-cli tweets.csv --author user --text content
```

**Output**:
```
╔══════════════════════════════════════════════════════════════╗
║          Social Network Analytics - CLI Tool                 ║
║  Extract social networks from social media data using NER    ║
╚══════════════════════════════════════════════════════════════╝

📋 Configuration:
  Input file:           tweets.csv
  Author column:        user
  Text column:          content
  Output directory:     ./output
  ...

🚀 Starting processing...

======================================================================
✅ Processing Complete!
======================================================================

📊 Results:
  Posts processed:      10,000
  Entities extracted:   15,234
  Network nodes:        1,523
  Network edges:        3,456
  ...
```

### Example 2: NDJSON with Custom Output

Process NDJSON data with custom output directory:

```bash
sna-cli data.ndjson \
  --author username \
  --text tweet_text \
  --output ./results/run1 \
  --format ndjson
```

### Example 3: High Confidence, Small Batches

Process with higher confidence threshold and smaller batches:

```bash
sna-cli tweets.csv \
  --author user \
  --text text \
  --confidence 0.95 \
  --batch-size 16 \
  --chunk-size 5000
```

### Example 4: Disable Features

Disable caching and author edges:

```bash
sna-cli tweets.csv \
  --author user \
  --text text \
  --no-cache \
  --no-author-edges
```

### Example 5: Verbose Output

Enable detailed logging:

```bash
sna-cli tweets.csv \
  --author user \
  --text text \
  --verbose
```

**Output includes detailed progress**:
```
  [1,000 posts | 125.3 posts/sec] Processed chunk 1
  [2,000 posts | 130.1 posts/sec] Processed chunk 2
  [3,000 posts | 128.7 posts/sec] Processed chunk 3
  ...
```

### Example 6: Export Specific Formats

Export only GEXF and JSON formats:

```bash
sna-cli tweets.csv \
  --author user \
  --text text \
  --export-formats gexf json
```

### Example 7: Quiet Mode

Suppress all output (useful for scripts):

```bash
sna-cli tweets.csv \
  --author user \
  --text text \
  --quiet

# Check exit code
echo $?  # 0 = success, non-zero = error
```

### Example 8: Alternative Model

Use a different NER model:

```bash
sna-cli tweets.csv \
  --author user \
  --text text \
  --model Babelscape/wikineural-multilingual-ner
```

### Example 9: Large File Processing

Process a very large file with optimized settings:

```bash
sna-cli large_dataset.csv \
  --author username \
  --text content \
  --chunk-size 50000 \
  --batch-size 64 \
  --output ./large_results
```

### Example 10: Complete Configuration

Full configuration example:

```bash
sna-cli social_media_data.csv \
  --author user_handle \
  --text post_content \
  --output ./final_results \
  --model Davlan/xlm-roberta-base-ner-hrl \
  --confidence 0.90 \
  --batch-size 32 \
  --chunk-size 10000 \
  --export-formats gexf graphml json statistics \
  --verbose
```

## Output Files

The CLI creates the following files in the output directory:

### Network Files

| File | Format | Description | Primary Use |
|------|--------|-------------|-------------|
| `network.gexf` | GEXF | **Primary format** | Gephi visualization |
| `network.graphml` | GraphML | Alternative format | yEd, Cytoscape |
| `network.json` | JSON | Node-link format | D3.js, web visualizations |
| `network_edgelist.csv` | CSV | Edge list | Excel, R, Python analysis |

### Statistics File

| File | Format | Description |
|------|--------|-------------|
| `network_statistics.json` | JSON | Complete network statistics and metadata |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error (file not found, processing error, etc.) |
| `130` | Interrupted by user (Ctrl+C) |

## Error Handling

### File Not Found

```bash
$ sna-cli missing.csv --author user --text text
❌ Error: Input file not found: missing.csv
```

### Invalid Confidence

```bash
$ sna-cli data.csv --author user --text text --confidence 1.5
❌ Error: Confidence threshold must be between 0.0 and 1.0
```

### Missing Column

```bash
$ sna-cli data.csv --author wrong_column --text text
❌ Error: Column 'wrong_column' not found in input file
```

### Interrupted Processing

```bash
$ sna-cli large.csv --author user --text text
^C
⚠️  Processing interrupted by user
```

## Performance Tips

### For Large Files (>100MB)

```bash
sna-cli large.csv \
  --author user \
  --text text \
  --chunk-size 50000 \
  --batch-size 64 \
  --no-progress  # Faster without progress display
```

### For GPU Acceleration

```bash
# Ensure CUDA is available
# Use larger batch sizes
sna-cli data.csv \
  --author user \
  --text text \
  --batch-size 128
```

### For Memory-Constrained Systems

```bash
sna-cli data.csv \
  --author user \
  --text text \
  --chunk-size 5000 \
  --batch-size 16
```

## Integration with Other Tools

### Bash Script

```bash
#!/bin/bash

# Process multiple files
for file in data/*.csv; do
  echo "Processing $file..."
  sna-cli "$file" \
    --author username \
    --text text \
    --output "./results/$(basename $file .csv)" \
    --quiet

  if [ $? -eq 0 ]; then
    echo "✓ $file processed successfully"
  else
    echo "✗ $file failed"
  fi
done
```

### Python Script

```python
import subprocess
import sys

# Run CLI from Python
result = subprocess.run([
    'sna-cli',
    'data.csv',
    '--author', 'user',
    '--text', 'text',
    '--output', './results',
    '--quiet'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Success!")
else:
    print(f"Error: {result.stderr}")
    sys.exit(1)
```

### Make Target

```makefile
# Makefile
.PHONY: process
process:
	sna-cli data/tweets.csv \
		--author username \
		--text text \
		--output results/run_$(shell date +%Y%m%d) \
		--verbose

.PHONY: clean
clean:
	rm -rf results/*
```

## Comparison with Web Interface

| Feature | CLI | Web Interface |
|---------|-----|---------------|
| Batch processing | ✅ Excellent | ❌ Not designed for it |
| Automation | ✅ Easy to script | ❌ Manual only |
| Configuration | ✅ Full control | ✅ Interactive |
| Visualization | ❌ Export only | ✅ Interactive preview |
| Progress tracking | ✅ Text-based | ✅ Graphical |
| Large files | ✅ Optimized | ⚠️ May struggle |
| Remote usage | ✅ SSH-friendly | ⚠️ Needs port forwarding |

## Troubleshooting

### Model Download Issues

**Problem**: First run downloads ~1GB model
**Solution**: Pre-download with Python:

```python
from transformers import pipeline
ner = pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl")
```

### Memory Errors

**Problem**: `Out of memory` error
**Solution**: Reduce chunk and batch sizes:

```bash
sna-cli data.csv --author user --text text \
  --chunk-size 1000 --batch-size 8
```

### GPU Not Detected

**Problem**: Using CPU despite having GPU
**Solution**: Check CUDA installation and PyTorch build

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Advanced Usage

### Custom Model Fine-tuned for Your Domain

```bash
sna-cli domain_data.csv \
  --author user \
  --text text \
  --model /path/to/your/finetuned/model
```

### Pipeline Multiple Datasets

```bash
# Process and combine results
for dataset in tweets posts comments; do
  sna-cli "${dataset}.csv" \
    --author user \
    --text content \
    --output "./results/${dataset}"
done

# Manually combine GEXF files in Gephi or networkx
```

## Getting Help

```bash
# Show all options
sna-cli --help

# Show version
sna-cli --version

# Verbose error messages
sna-cli data.csv --author user --text text --verbose
```

## See Also

- [Main README](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- Web interface: `sna-web` or `streamlit run src/cli/app.py`

# Technical Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                      │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Web UI                │  Command-Line Interface       │
│  - File upload                   │  - Batch processing           │
│  - Column selection              │  - Automation support         │
│  - Progress tracking             │  - Script integration         │
│  - Results download              │                               │
└────────────────┬────────────────┴───────────────┬───────────────┘
                 │                                 │
                 └────────────┬────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────────┐
│                      ORCHESTRATION LAYER                           │
├───────────────────────────────────────────────────────────────────┤
│                     pipeline.py                                    │
│  - Coordinate data flow                                           │
│  - Manage processing batches                                      │
│  - Handle errors and retries                                      │
│  - Progress reporting                                             │
└────────────────┬──────────────────────────────┬──────────────────┘
                 │                               │
┌────────────────▼───────────┐     ┌───────────▼──────────────────┐
│    DATA INGESTION LAYER     │     │    ANALYTICS LAYER           │
├────────────────────────────┤     ├──────────────────────────────┤
│  data_loader.py            │     │  ner_engine.py               │
│  - CSV reader (chunked)    │     │  - Model loading             │
│  - NDJSON reader           │     │  - Batch processing          │
│  - Encoding detection      │     │  - Entity extraction         │
│  - Column validation       │     │  - Token aggregation         │
│  - Memory management       │     └───────────┬──────────────────┘
└────────────────┬───────────┘                 │
                 │                              │
                 │     ┌────────────────────────▼──────────────────┐
                 │     │    ENTITY PROCESSING LAYER                │
                 │     ├───────────────────────────────────────────┤
                 │     │  entity_resolver.py                       │
                 │     │  - Entity normalization                   │
                 │     │  - Deduplication                          │
                 │     │  - Author-entity matching                 │
                 │     │  - Fuzzy matching (optional)              │
                 │     └───────────┬───────────────────────────────┘
                 │                 │
                 └─────────────────┼──────────────────────────┐
                                   │                          │
┌──────────────────────────────────▼──────────────────────────▼─────┐
│                    NETWORK CONSTRUCTION LAYER                      │
├───────────────────────────────────────────────────────────────────┤
│  network_builder.py                                               │
│  - Node creation (authors, entities)                              │
│  - Edge creation (mentions)                                       │
│  - Weight accumulation                                            │
│  - Metadata storage                                               │
│  - Statistics computation                                         │
└────────────────┬──────────────────────────────────────────────────┘
                 │
┌────────────────▼──────────────────────────────────────────────────┐
│                    EXPORT LAYER                                    │
├───────────────────────────────────────────────────────────────────┤
│  exporters.py                                                     │
│  - GraphML export (Gephi)                                         │
│  - GEXF export                                                    │
│  - JSON export (D3.js)                                            │
│  - CSV edge list                                                  │
│  - Statistics JSON                                                │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────┐
│ Input Files │
│ .csv/.ndjson│
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Data Loader        │
│  - Read in chunks   │
│  - Validate columns │
│  - Handle encoding  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐         ┌──────────────────┐
│  Text Batches       │────────>│  NER Engine      │
│  [text1, text2,...] │         │  - Tokenization  │
└─────────────────────┘         │  - GPU inference │
                                │  - Entity extract│
                                └────────┬─────────┘
                                         │
                                         ▼
                                ┌─────────────────────┐
                                │ Entity Results      │
                                │ [{text, label,      │
                                │   score, span},...] │
                                └────────┬────────────┘
                                         │
                                         ▼
                                ┌─────────────────────┐
                                │ Entity Resolver     │
                                │ - Normalize         │
                                │ - Deduplicate       │
                                │ - Match authors     │
                                └────────┬────────────┘
                                         │
       ┌─────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Network Builder     │
│ For each post:      │
│   - Add author node │
│   - Add entity nodes│
│   - Create edges    │
│   - Update weights  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ NetworkX Graph      │
│ - Nodes with attrs  │
│ - Edges with attrs  │
│ - Metadata          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Export to formats   │
│ - GraphML           │
│ - GEXF              │
│ - JSON              │
│ - CSV               │
└─────────────────────┘
```

## Component Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│                    External Dependencies                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  PyTorch   │  │Transformers│  │ Pandas  │  │ NetworkX │  │
│  │  (CUDA)    │  │(HuggingFace)│ │         │  │          │  │
│  └────────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                               │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Streamlit  │  │  Tqdm    │  │Langdetect│  │Levenshtein│ │
│  │            │  │(Progress)│  │          │  │ (Fuzzy)  │  │
│  └────────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Application Modules                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Core:                        Utils:                          │
│  ├── data_loader.py          ├── preprocessing.py            │
│  ├── ner_engine.py           ├── validators.py               │
│  ├── entity_resolver.py      └── exporters.py                │
│  ├── network_builder.py                                      │
│  └── pipeline.py             CLI:                            │
│                              ├── app.py (Streamlit)          │
│  Models:                     └── cli.py (argparse)           │
│  └── model_manager.py                                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Processing Pipeline Flowchart

```
START
  │
  ▼
[Load Configuration]
  │
  ├─ Model name
  ├─ Batch size
  ├─ Confidence threshold
  └─ Output directory
  │
  ▼
[Initialize NER Engine]
  │
  ├─ Download model (if needed)
  ├─ Load to GPU
  └─ Warm up model
  │
  ▼
[Open Input File]
  │
  ├─ Detect format (CSV/NDJSON)
  ├─ Detect encoding
  └─ Create chunked reader
  │
  ▼
[Initialize Network Builder]
  │
  └─ Create empty graph
  │
  ▼
┌─────────────────────────┐
│ FOR EACH CHUNK          │
├─────────────────────────┤
│  │                      │
│  ▼                      │
│ [Extract Text & Author] │
│  │                      │
│  ├─ Validate columns    │
│  ├─ Handle missing data │
│  └─ Clean text          │
│  │                      │
│  ▼                      │
│ [Batch NER Processing]  │
│  │                      │
│  ├─ Tokenize texts      │
│  ├─ GPU inference       │
│  ├─ Aggregate entities  │
│  └─ Filter by threshold │
│  │                      │
│  ▼                      │
│ [Resolve Entities]      │
│  │                      │
│  ├─ Normalize names     │
│  ├─ Deduplicate         │
│  └─ Match authors       │
│  │                      │
│  ▼                      │
│ [Update Network]        │
│  │                      │
│  ├─ Add/update nodes    │
│  ├─ Add/update edges    │
│  └─ Increment weights   │
│  │                      │
│  ▼                      │
│ [Update Progress]       │
│  │                      │
│  └─ Report % complete   │
│                         │
└────────┬────────────────┘
         │
         ▼
[Finalize Network]
  │
  ├─ Calculate statistics
  ├─ Compute centrality
  └─ Identify components
  │
  ▼
[Export Results]
  │
  ├─ Save GraphML
  ├─ Save GEXF
  ├─ Save JSON
  ├─ Save CSV
  └─ Save statistics
  │
  ▼
[Display Results]
  │
  ├─ Show statistics
  ├─ Show top entities
  └─ Provide download links
  │
  ▼
END
```

## Network Structure

```
Network: G = (V, E)

Vertices (V):
  V = V_authors ∪ V_entities
  
  V_authors: Set of post authors
    Attributes:
      - node_type: 'author'
      - label: author name/handle
      - post_count: number of posts
      - mention_count: times mentioned by others
      
  V_entities: Set of extracted entities
    V_entities = V_persons ∪ V_locations ∪ V_organizations
    
    Attributes:
      - node_type: 'person'|'location'|'organization'
      - label: entity text
      - mention_count: number of mentions
      - first_seen: timestamp of first mention

Edges (E):
  E: Set of directed edges (author → entity)
  
  Attributes:
    - weight: number of times author mentioned entity
    - entity_type: 'PER'|'LOC'|'ORG'|'AUTHOR'
    - source_posts: list of post IDs
    - first_mention: timestamp of first mention
    - last_mention: timestamp of last mention

Special Cases:
  - Author-to-Author edges: when author X mentions author Y's name
  - Self-loops: not created (author cannot mention themselves)
  - Multiple mentions: accumulate weight, don't create duplicate edges
```

## Memory Management Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Optimization                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Input Stage:                                            │
│     - Read files in chunks (10k rows default)               │
│     - Don't load entire file into memory                    │
│     - Process and discard chunks after use                  │
│                                                              │
│  2. NER Stage:                                              │
│     - Process texts in batches (32 default)                 │
│     - Clear GPU cache after each batch:                     │
│       torch.cuda.empty_cache()                              │
│     - Use FP16 for inference (halves memory)                │
│                                                              │
│  3. Network Building:                                       │
│     - NetworkX keeps full graph in memory                   │
│     - For very large networks (>1M nodes):                  │
│       * Consider streaming to disk                          │
│       * Use igraph instead of NetworkX                      │
│                                                              │
│  4. Export Stage:                                           │
│     - Stream write to files (don't buffer)                  │
│     - Use generators for edge iteration                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Estimated Memory Usage:
  - Model: ~1GB GPU memory
  - Batch processing: ~2GB GPU memory
  - Network (100k nodes): ~500MB RAM
  - Network (1M nodes): ~5GB RAM
  - Peak usage: ~8GB RAM + 3GB GPU for 1M nodes
```

## Error Handling Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Error Handling Strategy                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Input Validation Errors (UserError):                    │
│     - File not found                                        │
│     - Invalid format                                        │
│     - Missing columns                                       │
│     → Action: Display clear message, don't proceed          │
│                                                              │
│  2. Processing Errors (RecoverableError):                   │
│     - Single post parsing error                             │
│     - Encoding issue in one row                             │
│     - NER failure on one text                               │
│     → Action: Log, skip item, continue processing           │
│                                                              │
│  3. System Errors (CriticalError):                          │
│     - GPU out of memory                                     │
│     - Model loading failure                                 │
│     - Disk full during export                               │
│     → Action: Log, attempt recovery, fail gracefully        │
│                                                              │
│  4. Network Errors (NetworkError):                          │
│     - Model download timeout                                │
│     - HuggingFace API unreachable                           │
│     → Action: Retry with backoff, use cached model          │
│                                                              │
│  Logging Structure:                                         │
│    logs/                                                    │
│    ├── app.log          (all logs)                         │
│    ├── errors.log       (errors only)                      │
│    └── processing.log   (processing details)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Configuration Management

```
Configuration Hierarchy (priority order):

1. Command-line arguments (highest priority)
   --model, --batch-size, --confidence, etc.

2. Environment variables
   SNA_MODEL_NAME, SNA_BATCH_SIZE, etc.

3. Config file (config.yaml)
   model:
     name: "Davlan/xlm-roberta-base-ner-hrl"
     confidence_threshold: 0.85
   processing:
     batch_size: 32
     chunk_size: 10000
   output:
     formats: ['graphml', 'gexf', 'json']

4. Default values (lowest priority)
   Hard-coded in application

Example config.yaml:
─────────────────────
model:
  name: "Davlan/xlm-roberta-base-ner-hrl"
  cache_dir: "./models"
  device: "cuda"  # or "cpu"
  confidence_threshold: 0.85

processing:
  batch_size: 32
  chunk_size: 10000
  max_sequence_length: 512
  use_fp16: true

entity_resolution:
  fuzzy_matching: true
  fuzzy_threshold: 0.9
  normalize_case: true

output:
  directory: "./output"
  formats:
    - graphml
    - gexf
    - json
  include_statistics: true
  
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "logs/app.log"
```

## Performance Benchmarks

```
Expected Performance (NVIDIA RTX 3080, 10GB VRAM):

Posts/second by text length:
┌──────────────────┬──────────┬──────────┬──────────┐
│ Text Length      │ Batch=16 │ Batch=32 │ Batch=64 │
├──────────────────┼──────────┼──────────┼──────────┤
│ Short (<50 tok)  │  ~400/s  │  ~600/s  │  ~800/s  │
│ Medium (50-200)  │  ~200/s  │  ~300/s  │  ~400/s  │
│ Long (200-512)   │  ~100/s  │  ~150/s  │  ~200/s  │
└──────────────────┴──────────┴──────────┴──────────┘

Total Processing Time:
┌────────────┬────────────┬────────────┬────────────┐
│ Total Posts│ Avg Length │ Time       │ Throughput │
├────────────┼────────────┼────────────┼────────────┤
│ 10,000     │ Medium     │ ~1 min     │ ~167/s     │
│ 100,000    │ Medium     │ ~8 min     │ ~208/s     │
│ 1,000,000  │ Medium     │ ~2 hours   │ ~139/s     │
└────────────┴────────────┴────────────┴────────────┘

Note: Times include file I/O, NER processing, and network construction.
Network export adds ~5-10% overhead.

CPU-only Performance:
Expect 5-10x slower than GPU (10k posts: ~5-10 minutes)
```

## Security Considerations

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Guidelines                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Input Validation:                                       │
│     - Limit file size (default: 500MB)                      │
│     - Validate file extensions                              │
│     - Sanitize file paths (prevent directory traversal)     │
│     - Check for malicious content in uploads                │
│                                                              │
│  2. Data Privacy:                                           │
│     - Don't log post content                                │
│     - Clear session data after processing                   │
│     - No persistent storage of user data                    │
│     - Warn users about data sensitivity                     │
│                                                              │
│  3. Model Security:                                         │
│     - Verify model checksums after download                 │
│     - Use official HuggingFace Hub only                     │
│     - Don't execute arbitrary model code                    │
│                                                              │
│  4. Network Security:                                       │
│     - Run on localhost only (development)                   │
│     - Use HTTPS in production                               │
│     - Implement rate limiting                               │
│     - Add CORS headers appropriately                        │
│                                                              │
│  5. Dependency Security:                                    │
│     - Pin dependency versions                               │
│     - Regularly update dependencies                         │
│     - Scan for vulnerabilities (pip-audit)                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Scaling Considerations

```
Current Architecture (Prototype):
  - Single machine
  - In-memory processing
  - Session-based
  - No persistence

For Production Scale:

1. Horizontal Scaling:
   ┌─────────────┐
   │ Load Balancer│
   └──────┬──────┘
          │
    ┌─────┴─────┬─────────┐
    │           │         │
   ▼           ▼         ▼
  [Worker 1] [Worker 2] [Worker 3]
     │           │         │
     └───────────┴─────────┘
              │
              ▼
        [Database]
        [Redis Cache]

2. Data Pipeline:
   Input → Queue (RabbitMQ) → Workers → Storage

3. Distributed Processing:
   - Apache Spark for large-scale NER
   - Dask for distributed dataframes
   - Ray for parallel processing

4. Network Storage:
   - Graph database (Neo4j, ArangoDB)
   - Time-series DB for temporal analysis

5. Monitoring:
   - Prometheus + Grafana
   - Error tracking (Sentry)
   - Performance monitoring (APM)
```

---

This architecture is designed for:
- **Modularity**: Each component is independent
- **Testability**: Clear interfaces for unit testing
- **Scalability**: Can be extended to distributed systems
- **Maintainability**: Clean separation of concerns
- **Performance**: Optimized for GPU acceleration

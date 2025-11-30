# Entity Linking Phase 3: Advanced Features

## Overview

Phase 3 introduces advanced features for entity linking, focusing on improved disambiguation, custom knowledge bases, and entity relationship extraction. These features enable more accurate entity resolution and richer network analysis.

**Status**: ✅ Complete
**Date**: 2025-11-30

## What's New in Phase 3

### 🎯 1. Advanced Disambiguation

Context-aware entity disambiguation using sentence embeddings to improve linking accuracy for ambiguous entities.

**Key Features**:
- Semantic similarity between context and candidates
- Entity co-occurrence patterns for disambiguation
- Re-ranking of candidates based on context

**Use Case**: Disambiguating "Paris" (France vs Texas) based on surrounding context.

### 📚 2. Custom Knowledge Base

Support for domain-specific entity databases to handle organization-specific entities and aliases.

**Key Features**:
- JSON-based custom entity definitions
- Alias support for entity variants
- Type-aware entity lookup
- Priority lookup (custom KB checked before Wikipedia)

**Use Case**: Organization-specific entity names, internal project codes, or domain terminology.

### 🔗 3. Entity Relationship Extraction

Automatic extraction of relationships between co-mentioned entities.

**Key Features**:
- Co-mention relationship detection
- Confidence scoring for relationships
- Context preservation
- Network export capabilities

**Use Case**: Building entity-entity networks for relationship analysis.

### 📊 4. Co-occurrence Network Analysis

Track and analyze entity co-occurrence patterns over time.

**Key Features**:
- Automatic co-occurrence tracking
- Network graph generation
- Threshold-based filtering
- Export for visualization

**Use Case**: Discovering which entities frequently appear together.

---

## Installation & Setup

### Dependencies

Phase 3 adds one new dependency:

```bash
pip install sentence-transformers>=2.2.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### Model Downloads

The first time you use advanced disambiguation, it will download the multilingual sentence transformer model (~500MB):

```python
# Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# Size: ~500MB
# Languages: 50+ languages
```

---

## Feature 1: Advanced Disambiguation

### Basic Usage

```python
from src.core.entity_linker import EntityLinker

# Initialize with advanced disambiguation
linker = EntityLinker(
    enable_advanced_disambiguation=True,
    enable_cache=True
)

# Link entity with context
result = linker.link_entity(
    entity_text="Paris",
    entity_type="LOC",
    language="en",
    context="I visited Paris, the capital of France, last summer."
)

print(f"Linked to: {result['canonical_name']}")
print(f"Disambiguation method: {result['disambiguation_method']}")
# Output: "advanced" (used context-aware disambiguation)
```

### How It Works

1. **Baseline Model**: mGENRE generates top-k candidates
2. **Context Embedding**: Encodes the context sentence
3. **Candidate Scoring**: Computes semantic similarity
4. **Re-ranking**: Adjusts confidence scores
5. **Co-occurrence Boost**: Adds bonus for known entity pairs

```python
# Example: Disambiguating "Paris"
candidates = [
    "Paris, Texas" (confidence: 0.6),
    "Paris, France" (confidence: 0.5)
]

context = "The capital of France is beautiful"

# After advanced disambiguation:
# Paris, France gets boosted due to semantic similarity
# Final selection: "Paris, France" (adjusted confidence: 0.85)
```

### Configuration

```python
linker = EntityLinker(
    enable_advanced_disambiguation=True,
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # Alternative models:
    # "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    # "sentence-transformers/distiluse-base-multilingual-cased-v2"
)
```

### Performance Impact

| Mode | Speed | Accuracy | Use When |
|------|-------|----------|----------|
| Baseline | 100% | Good | Simple entities, clear context |
| Advanced | 70% | Better | Ambiguous entities, rich context |

**Recommendation**: Use advanced disambiguation only when context is available and entities are ambiguous.

---

## Feature 2: Custom Knowledge Base

### Creating a Knowledge Base

Create a JSON file with your custom entities:

```json
{
  "Meta": {
    "canonical_name": "Meta Platforms",
    "wikidata_id": "Q380",
    "wikipedia_url": "https://en.wikipedia.org/wiki/Meta_Platforms",
    "type": "ORG",
    "aliases": ["Facebook", "Meta Inc", "Meta Platforms Inc"],
    "language_variants": {
      "en": "Meta Platforms",
      "da": "Meta Platforms"
    }
  },
  "Project Phoenix": {
    "canonical_name": "Internal Project Phoenix",
    "wikidata_id": "CUSTOM_001",
    "type": "ORG",
    "aliases": ["Phoenix", "Proj Phoenix", "PHX"]
  }
}
```

**Format Specification**:
- **canonical_name** (required): Official entity name
- **wikidata_id** (optional): Wikidata QID or custom ID
- **wikipedia_url** (optional): Wikipedia link
- **type** (recommended): PER/LOC/ORG for type checking
- **aliases** (optional): Alternative names/spellings
- **language_variants** (optional): Translations

### Using Custom Knowledge Base

```python
from src.core.entity_linker import EntityLinker

# Initialize with custom KB
linker = EntityLinker(
    custom_kb_path="path/to/custom_kb.json",
    enable_cache=True
)

# Entities in KB are matched first
result = linker.link_entity("Facebook", "ORG", "en")

print(result['canonical_name'])  # "Meta Platforms"
print(result['wikidata_id'])     # "Q380"
print(result['disambiguation_method'])  # "custom_kb"
```

### Alias Lookup

Aliases are automatically indexed:

```python
# All of these resolve to "Meta Platforms"
linker.link_entity("Meta", "ORG", "en")
linker.link_entity("Facebook", "ORG", "en")
linker.link_entity("Meta Inc", "ORG", "en")
```

### Type-Aware Matching

```python
# Type must match (if specified in KB)
result = linker.link_entity("Meta", "ORG", "en")  # ✅ Matches
result = linker.link_entity("Meta", "PER", "en")  # ❌ No match (wrong type)
```

### Example: Domain-Specific Entities

See `examples/custom_knowledge_base_example.json` for a complete example.

### Use Cases

1. **Internal Organization Names**: Company-specific entity mappings
2. **Project Codes**: Map internal codes to full names
3. **Abbreviations**: Handle domain-specific abbreviations
4. **Corrections**: Override incorrect Wikipedia matches
5. **Private Entities**: Entities not in Wikipedia

---

## Feature 3: Entity Relationship Extraction

### Basic Usage

```python
from src.core.entity_linker import EntityLinker

linker = EntityLinker()

# Process entities from text
entities = [
    {
        'text': 'Paris',
        'is_linked': True,
        'wikidata_id': 'Q90',
        'canonical_name': 'Paris',
        'linking_confidence': 0.95
    },
    {
        'text': 'France',
        'is_linked': True,
        'wikidata_id': 'Q142',
        'canonical_name': 'France',
        'linking_confidence': 0.93
    }
]

text = "Paris is the capital of France"

# Extract relationships
relationships = linker.extract_entity_relationships(
    entities,
    text,
    relationship_types=['co-mention']
)

for rel in relationships:
    print(f"{rel['source_name']} <-> {rel['target_name']}")
    print(f"  Type: {rel['relationship_type']}")
    print(f"  Confidence: {rel['confidence']}")
```

**Output**:
```
Paris <-> France
  Type: co-mention
  Confidence: 0.93
```

### Relationship Format

```python
{
    'source_entity': 'Q90',           # Wikidata ID
    'source_name': 'Paris',           # Canonical name
    'target_entity': 'Q142',          # Wikidata ID
    'target_name': 'France',          # Canonical name
    'relationship_type': 'co-mention',
    'confidence': 0.93,               # Min of both confidences
    'context': 'Paris is the capital...'  # First 200 chars
}
```

### Relationship Types

**Phase 3** supports:
- `co-mention`: Entities mentioned together

**Future phases** could add:
- `located-in`: Geographical relationships
- `works-for`: Employment relationships
- `part-of`: Organizational hierarchy
- Custom relationship types

### Use Cases

1. **Entity Network Analysis**: Build networks from co-mentions
2. **Relationship Discovery**: Find entity associations
3. **Context Analysis**: Understand entity relationships in text
4. **Knowledge Graph Construction**: Build domain knowledge graphs

---

## Feature 4: Co-occurrence Network

### Automatic Tracking

Co-occurrence is tracked automatically during linking:

```python
linker = EntityLinker()

# As you link entities with co_entities parameter
result = linker.link_entity(
    entity_text="Paris",
    entity_type="LOC",
    language="en",
    co_entities=["France", "Europe"]  # Other entities in context
)

# Co-occurrence is tracked in linker.entity_cooccurrence
```

### Retrieving the Network

```python
# Get co-occurrence network
network = linker.get_entity_network(min_cooccurrence=2)

for entity_id, connections in network.items():
    print(f"\n{entity_id}:")
    for connected_entity, count in connections:
        print(f"  → {connected_entity}: {count} co-occurrences")
```

**Output**:
```
Q90:
  → Q142: 15 co-occurrences
  → Q46: 8 co-occurrences
```

### Exporting Network Data

```python
# Save to JSON for analysis/visualization
linker.save_cooccurrence_data("cooccurrence_network.json")
```

**Output Format** (`cooccurrence_network.json`):
```json
{
  "Q90": {
    "Q142": 15,
    "Q46": 8
  },
  "Q142": {
    "Q90": 15,
    "Q183": 12
  }
}
```

### Visualization

Use the exported data with network visualization tools:

```python
import networkx as nx
import json

# Load co-occurrence data
with open("cooccurrence_network.json") as f:
    data = json.load(f)

# Create network graph
G = nx.Graph()
for entity, connections in data.items():
    for connected, weight in connections.items():
        G.add_edge(entity, connected, weight=weight)

# Export to Gephi format
nx.write_gexf(G, "entity_network.gexf")
```

### Use Cases

1. **Topic Detection**: Identify entity clusters
2. **Community Finding**: Discover entity communities
3. **Temporal Analysis**: Track relationship changes over time
4. **Influence Analysis**: Find central entities

---

## Complete Example

Here's a comprehensive example using all Phase 3 features:

```python
from src.core.entity_linker import EntityLinker

# Initialize with all Phase 3 features
linker = EntityLinker(
    enable_advanced_disambiguation=True,
    custom_kb_path="custom_entities.json",
    enable_cache=True
)

# Sample text
text = "Meta CEO Mark Zuckerberg announced new features in Copenhagen."

# Extracted entities (from NER)
entities = [
    {'text': 'Meta', 'type': 'ORG', 'language': 'en'},
    {'text': 'Mark Zuckerberg', 'type': 'PER', 'language': 'en'},
    {'text': 'Copenhagen', 'type': 'LOC', 'language': 'en'}
]

# Link entities with advanced features
enhanced_entities = []
entity_names = [e['text'] for e in entities]

for entity in entities:
    result = linker.link_entity(
        entity_text=entity['text'],
        entity_type=entity['type'],
        language=entity['language'],
        context=text,  # For advanced disambiguation
        co_entities=entity_names  # For co-occurrence tracking
    )

    if result:
        entity.update(result)
        entity['is_linked'] = True
        enhanced_entities.append(entity)

# Extract relationships
relationships = linker.extract_entity_relationships(
    enhanced_entities,
    text
)

# Results
print("=== Linked Entities ===")
for ent in enhanced_entities:
    print(f"{ent['text']} → {ent['canonical_name']}")
    print(f"  Method: {ent['disambiguation_method']}")
    print(f"  Wikidata: {ent['wikidata_id']}")

print("\n=== Relationships ===")
for rel in relationships:
    print(f"{rel['source_name']} <-> {rel['target_name']}")

# Export co-occurrence network
linker.save_cooccurrence_data("cooccurrence.json")
network = linker.get_entity_network(min_cooccurrence=1)
print(f"\n=== Network: {len(network)} entities with connections ===")
```

---

## Testing

### Running Tests

```bash
# Run all entity linker tests (including Phase 3)
pytest tests/test_entity_linker.py -v

# Run only Phase 3 tests
pytest tests/test_entity_linker.py::TestPhase3CustomKnowledgeBase -v
pytest tests/test_entity_linker.py::TestPhase3AdvancedDisambiguation -v
pytest tests/test_entity_linker.py::TestPhase3EntityRelationships -v
pytest tests/test_entity_linker.py::TestPhase3CooccurrenceNetwork -v
```

### Test Coverage

Phase 3 tests cover:
- ✅ Custom KB loading and alias resolution
- ✅ Advanced disambiguation with context
- ✅ Entity relationship extraction
- ✅ Co-occurrence network generation
- ✅ Edge cases and error handling

---

## Performance Considerations

### Advanced Disambiguation

**Memory Impact**:
- Embedding model: ~500MB RAM
- Minimal additional memory per entity

**Speed Impact**:
- Adds ~30-50ms per entity (with context)
- Negligible if context is empty or disabled

**Recommendation**:
```python
# Use when needed
if entity_is_ambiguous and context_available:
    enable_advanced_disambiguation = True
else:
    enable_advanced_disambiguation = False  # Faster
```

### Custom Knowledge Base

**Memory**: Negligible (typically <1MB for thousands of entities)
**Speed**: Very fast (O(1) hash lookup)
**Recommendation**: Always use if you have domain-specific entities

### Co-occurrence Tracking

**Memory**: O(n²) worst case, but sparse in practice
**Speed**: Negligible overhead during linking
**Recommendation**: Enable by default, export periodically

---

## Best Practices

### 1. Custom Knowledge Base

```python
# DO: Keep KB focused on your domain
kb = {
    "important_entity_1": {...},
    "important_entity_2": {...}
}

# DON'T: Duplicate Wikipedia entities
kb = {
    "Paris": {...}  # Already in Wikipedia
}
```

### 2. Advanced Disambiguation

```python
# DO: Provide good context
result = linker.link_entity(
    "Paris",
    context="I visited Paris, the capital of France"  # Full sentence
)

# DON'T: Use without context
result = linker.link_entity(
    "Paris",
    context=""  # No benefit from advanced mode
)
```

### 3. Relationship Extraction

```python
# DO: Use with linked entities only
relationships = linker.extract_entity_relationships(
    [e for e in entities if e.get('is_linked')],
    text
)

# DON'T: Include unlinked entities
relationships = linker.extract_entity_relationships(
    all_entities,  # May include unlinked
    text
)
```

### 4. Co-occurrence Networks

```python
# DO: Export periodically
if entity_count % 1000 == 0:
    linker.save_cooccurrence_data(f"cooccur_{entity_count}.json")

# DO: Use appropriate thresholds
network = linker.get_entity_network(min_cooccurrence=3)  # Filter noise
```

---

## Troubleshooting

### Advanced Disambiguation Issues

**Problem**: "Failed to load embedding model"

**Solutions**:
```bash
# Install sentence-transformers
pip install sentence-transformers>=2.2.0

# Or disable advanced disambiguation
linker = EntityLinker(enable_advanced_disambiguation=False)
```

**Problem**: "CUDA out of memory" with advanced mode

**Solutions**:
```python
# Use CPU for embedder
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or reduce batch processing
```

### Custom KB Issues

**Problem**: Entities not matching

**Solutions**:
- Check JSON syntax is valid
- Ensure entity names are normalized (lowercased automatically)
- Verify aliases are in the aliases list
- Check entity type matches (if specified)

**Problem**: "Cannot load custom KB file"

```python
# Verify file exists and is valid JSON
import json
with open("custom_kb.json") as f:
    data = json.load(f)  # Should not raise error
```

---

## Limitations

### Current Limitations

1. **Advanced Disambiguation**:
   - Only uses sentence-level context (not document-level)
   - Limited to semantic similarity (no entity descriptions)
   - Requires good context sentences

2. **Custom Knowledge Base**:
   - JSON format only (no database backend)
   - All entities loaded into memory
   - No hierarchical organization

3. **Relationship Extraction**:
   - Only co-mention relationships
   - No relationship direction (undirected)
   - No relationship classification beyond type

4. **Co-occurrence Networks**:
   - In-memory storage (not persistent by default)
   - No temporal tracking (when entities co-occurred)
   - No context aggregation

### Future Enhancements

**Phase 4 (Planned)**:
- Document-level context for disambiguation
- Entity description retrieval from Wikidata
- Typed relationship extraction (e.g., "works-for", "located-in")
- Database backend for custom KB
- Temporal co-occurrence analysis
- Relationship classification using NLP

---

## Migration from Phase 2

Phase 3 is fully backward compatible. No changes needed to existing code.

### Optional Upgrades

```python
# Phase 2 code (still works)
linker = EntityLinker()
result = linker.link_entity("Paris", "LOC", "en")

# Phase 3 enhancements (optional)
linker = EntityLinker(
    enable_advanced_disambiguation=True,  # NEW
    custom_kb_path="custom.json"          # NEW
)
result = linker.link_entity(
    "Paris", "LOC", "en",
    context="I visited Paris, France",    # NEW
    co_entities=["France", "Europe"]      # NEW
)

# Use new features
relationships = linker.extract_entity_relationships(entities, text)  # NEW
network = linker.get_entity_network()                                 # NEW
```

---

## API Reference

### EntityLinker (Phase 3 Extensions)

#### New Initialization Parameters

```python
EntityLinker(
    # ... Phase 1 & 2 parameters ...
    enable_advanced_disambiguation: bool = False,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    custom_kb_path: Optional[str] = None
)
```

#### New/Updated Methods

**`link_entity()`** - Now supports:
```python
link_entity(
    entity_text: str,
    entity_type: str = "ENTITY",
    language: str = "en",
    context: Optional[str] = None,
    co_entities: Optional[List[str]] = None  # NEW
) -> Optional[Dict]
```

**`extract_entity_relationships()`** - NEW
```python
extract_entity_relationships(
    entities: List[Dict],
    text: str,
    relationship_types: Optional[List[str]] = None
) -> List[Dict]
```

**`get_entity_network()`** - NEW
```python
get_entity_network(
    min_cooccurrence: int = 2
) -> Dict[str, List[Tuple[str, int]]]
```

**`save_cooccurrence_data()`** - NEW
```python
save_cooccurrence_data(output_path: str) -> None
```

---

## Changelog

### Phase 3.0 (2025-11-30)

**Added**:
- ✅ Advanced disambiguation using sentence embeddings
- ✅ Custom knowledge base support with JSON
- ✅ Entity relationship extraction
- ✅ Co-occurrence network tracking and export
- ✅ Co-entity parameter for context-aware linking
- ✅ Comprehensive tests for all Phase 3 features
- ✅ Example custom knowledge base

**Updated**:
- `requirements.txt`: Added sentence-transformers
- `link_entity()`: New co_entities parameter
- Entity linking results: Added disambiguation_method field

**Documentation**:
- ✅ ENTITY_LINKING_PHASE3.md (this document)
- ✅ Updated ENTITY_LINKING.md with Phase 3 status

---

## References

### Papers

- **mGENRE**: De Cao et al. (2021) - "Multilingual Autoregressive Entity Linking"
- **Sentence-BERT**: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

### Resources

- [mGENRE Model](https://huggingface.co/facebook/mgenre-wiki)
- [Sentence Transformers](https://www.sbert.net/)
- [Wikidata API](https://www.wikidata.org/w/api.php)

---

**Version**: Phase 3.0
**Status**: ✅ Production Ready
**Last Updated**: 2025-11-30

**Phases**:
- Phase 1: Standalone entity linking ✅
- Phase 2: Pipeline integration with Wikidata ✅
- Phase 3: Advanced features ✅
- Phase 4: Future enhancements (planned)

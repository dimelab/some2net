"""
Example script demonstrating NER Engine functionality.

This script shows how to:
1. Initialize the NER engine
2. Extract entities from single text
3. Process batches of text
4. Use caching for efficiency
5. Detect languages
6. Integrate with DataLoader

Run after installing dependencies:
    pip install -r requirements.txt
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.ner_engine import NEREngine
from src.core.data_loader import DataLoader


def example_basic_extraction():
    """Example 1: Basic entity extraction from single text."""
    print("=" * 70)
    print("Example 1: Basic Entity Extraction")
    print("=" * 70)

    # Initialize NER engine
    # Note: First run will download model (~1GB), may take a few minutes
    engine = NEREngine(
        enable_cache=True,
        confidence_threshold=0.85
    )

    # Extract entities from single text
    text = "John Smith announced that Microsoft will open a new office in Copenhagen."

    print(f"\nText: {text}\n")

    entities = engine.extract_entities(text)

    print(f"Found {len(entities)} entities:\n")
    for entity in entities:
        print(f"  • {entity['text']:20s} ({entity['type']}) - confidence: {entity['score']:.3f}")

    print()


def example_batch_processing():
    """Example 2: Batch processing multiple texts."""
    print("=" * 70)
    print("Example 2: Batch Processing")
    print("=" * 70)

    engine = NEREngine(enable_cache=True)

    texts = [
        "Barack Obama met with Angela Merkel in Berlin yesterday.",
        "Apple Inc. and Microsoft are competing in the AI market.",
        "The conference will be held in Paris next month.",
        "Tesla opened a new factory in Austin, Texas.",
        "Elon Musk announced the new product at SpaceX headquarters."
    ]

    print(f"\nProcessing {len(texts)} texts in batch...\n")

    # Process all texts in one batch
    results, languages = engine.extract_entities_batch(
        texts,
        batch_size=5,
        show_progress=True,
        detect_languages=True
    )

    # Display results
    for i, (text, entities, lang) in enumerate(zip(texts, results, languages), 1):
        print(f"\n{i}. Text: {text}")
        print(f"   Language: {lang}")
        print(f"   Entities ({len(entities)}):")
        for entity in entities:
            print(f"     • {entity['text']} ({entity['type']})")

    print()


def example_language_detection():
    """Example 3: Language detection."""
    print("=" * 70)
    print("Example 3: Language Detection")
    print("=" * 70)

    engine = NEREngine(enable_cache=True)

    multilingual_texts = [
        "This is an English sentence about London.",
        "Dette er en dansk sætning om København.",
        "Esta es una oración española sobre Madrid.",
        "Das ist ein deutscher Satz über Berlin.",
        "C'est une phrase française sur Paris."
    ]

    print("\nDetecting languages:\n")

    for text in multilingual_texts:
        lang = engine.detect_language(text)
        print(f"  {lang:5s} | {text}")

    print()


def example_caching():
    """Example 4: Caching demonstration."""
    print("=" * 70)
    print("Example 4: Caching for Efficiency")
    print("=" * 70)

    engine = NEREngine(enable_cache=True)

    text = "Microsoft CEO Satya Nadella visited the London office."

    print(f"\nText: {text}\n")

    # First extraction (cache miss)
    print("First extraction (cache miss):")
    result1 = engine.extract_entities(text)
    print(f"  Found {len(result1)} entities")

    # Check cache
    stats = engine.get_cache_stats()
    print(f"  Cache: {stats['size']} entries, {stats['size_bytes']:,} bytes")

    # Second extraction (cache hit)
    print("\nSecond extraction (cache hit):")
    result2 = engine.extract_entities(text)
    print(f"  Found {len(result2)} entities (same as before)")

    # Results are identical
    assert result1 == result2, "Cache should return identical results"

    print("\n  ✓ Cache hit - much faster!")
    print()


def example_with_dataloader():
    """Example 5: Integration with DataLoader."""
    print("=" * 70)
    print("Example 5: Integration with DataLoader")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\nError: {example_file} not found")
        print("Please ensure sample_data.csv exists in examples/ directory")
        return

    # Initialize components
    loader = DataLoader()
    engine = NEREngine(enable_cache=True)

    print(f"\nLoading: {example_file.name}\n")

    # Track results
    total_posts = 0
    total_entities = 0
    entity_types = {'PER': 0, 'LOC': 0, 'ORG': 0}

    # Process in chunks
    for chunk in loader.load_csv(
        example_file,
        author_column='author',
        text_column='text',
        chunksize=5
    ):
        texts = chunk['text'].tolist()
        total_posts += len(texts)

        # Extract entities from chunk
        entities_batch, languages = engine.extract_entities_batch(
            texts,
            show_progress=False,
            detect_languages=True
        )

        # Aggregate results
        for entities in entities_batch:
            total_entities += len(entities)
            for entity in entities:
                entity_types[entity['type']] += 1

    # Display summary
    print(f"Summary:")
    print(f"  Posts processed: {total_posts}")
    print(f"  Total entities:  {total_entities}")
    print(f"  By type:")
    print(f"    Persons (PER):        {entity_types['PER']}")
    print(f"    Locations (LOC):      {entity_types['LOC']}")
    print(f"    Organizations (ORG):  {entity_types['ORG']}")

    # Cache statistics
    stats = engine.get_cache_stats()
    print(f"\n  Cache: {stats['size']} entries, {stats['size_bytes']:,} bytes")
    print()


def example_entity_types():
    """Example 6: Different entity types."""
    print("=" * 70)
    print("Example 6: Entity Types")
    print("=" * 70)

    engine = NEREngine(enable_cache=True, confidence_threshold=0.80)

    # Examples highlighting different entity types
    examples = [
        ("Persons", "Barack Obama and Angela Merkel met with Joe Biden."),
        ("Locations", "I visited Paris, London, Copenhagen, and Berlin."),
        ("Organizations", "Microsoft, Google, Apple, and Amazon are tech giants."),
        ("Mixed", "Elon Musk founded SpaceX in California and Tesla in Austin.")
    ]

    for category, text in examples:
        print(f"\n{category}:")
        print(f"  Text: {text}")

        entities = engine.extract_entities(text)

        # Group by type
        by_type = {'PER': [], 'LOC': [], 'ORG': []}
        for entity in entities:
            by_type[entity['type']].append(entity['text'])

        print(f"  Persons:       {', '.join(by_type['PER']) or 'None'}")
        print(f"  Locations:     {', '.join(by_type['LOC']) or 'None'}")
        print(f"  Organizations: {', '.join(by_type['ORG']) or 'None'}")

    print()


def example_confidence_threshold():
    """Example 7: Effect of confidence threshold."""
    print("=" * 70)
    print("Example 7: Confidence Threshold")
    print("=" * 70)

    text = "John Smith works at Microsoft in Seattle."

    print(f"\nText: {text}\n")

    thresholds = [0.50, 0.75, 0.85, 0.95]

    for threshold in thresholds:
        engine = NEREngine(
            enable_cache=False,  # Don't cache for fair comparison
            confidence_threshold=threshold
        )

        entities = engine.extract_entities(text)

        print(f"Threshold {threshold:.2f}: {len(entities)} entities found")
        for entity in entities:
            print(f"  • {entity['text']:15s} ({entity['type']}) - {entity['score']:.3f}")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NER Engine Examples")
    print("=" * 70 + "\n")

    print("NOTE: First run will download the NER model (~1GB).")
    print("This may take a few minutes depending on your connection.\n")

    try:
        # Run examples
        example_basic_extraction()
        example_batch_processing()
        example_language_detection()
        example_caching()
        example_entity_types()
        example_confidence_threshold()
        example_with_dataloader()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

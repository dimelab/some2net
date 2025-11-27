"""
Example script demonstrating Entity Resolution functionality.

This script shows how to:
1. Use simple normalized matching
2. Deduplicate entities across posts
3. Match entities to authors
4. Integrate with NER engine
5. Track entity statistics

Run after installing dependencies:
    pip install -r requirements.txt
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.entity_resolver import EntityResolver
from src.core.data_loader import DataLoader
from src.core.ner_engine import NEREngine


def example_basic_normalization():
    """Example 1: Basic text normalization."""
    print("=" * 70)
    print("Example 1: Basic Text Normalization")
    print("=" * 70)

    resolver = EntityResolver()

    texts = [
        "John Smith",
        "john smith",
        "JOHN SMITH",
        "John  Smith",  # Extra space
        "  John Smith  ",  # Leading/trailing space
        "'John Smith'",  # Quotes
        "John Smith.",  # Period
    ]

    print("\nNormalizing various text forms:\n")
    for text in texts:
        normalized = resolver.normalize_text(text)
        print(f"  '{text:20s}' → '{normalized}'")

    print("\n✓ All normalize to: 'john smith'")
    print()


def example_canonical_forms():
    """Example 2: Canonical form resolution."""
    print("=" * 70)
    print("Example 2: Canonical Form Resolution")
    print("=" * 70)

    resolver = EntityResolver()

    print("\nSimulating entities extracted from different posts:\n")

    # Simulate posts mentioning the same entity in different forms
    posts = [
        ("Post 1", "John Smith"),
        ("Post 2", "john smith"),
        ("Post 3", "JOHN SMITH"),
        ("Post 4", "John  Smith"),
    ]

    canonicals = []
    for post_name, entity_text in posts:
        canonical = resolver.get_canonical_form(entity_text)
        canonicals.append(canonical)
        print(f"  {post_name}: '{entity_text:15s}' → Canonical: '{canonical}'")

    print(f"\n✓ All mapped to same canonical form: '{canonicals[0]}'")
    print(f"✓ First occurrence's capitalization preserved")
    print()


def example_multiple_entities():
    """Example 3: Multiple different entities."""
    print("=" * 70)
    print("Example 3: Multiple Different Entities")
    print("=" * 70)

    resolver = EntityResolver()

    entities = [
        "John Smith",
        "jane doe",
        "JOHN SMITH",  # Duplicate of first
        "Microsoft Corporation",
        "Jane Doe",  # Duplicate of second
        "microsoft corporation",  # Duplicate of fourth
        "Copenhagen",
        "copenhagen",  # Duplicate of seventh
    ]

    print("\nResolving entities:\n")

    canonical_map = {}
    for entity in entities:
        canonical = resolver.get_canonical_form(entity)

        if canonical not in canonical_map:
            canonical_map[canonical] = []
        canonical_map[canonical].append(entity)

    for canonical, variations in canonical_map.items():
        print(f"  Canonical: '{canonical}'")
        print(f"    Variations: {variations}")
        print()

    stats = resolver.get_statistics()
    print(f"✓ Total unique entities: {stats['unique_entities']}")
    print()


def example_author_matching():
    """Example 4: Author-entity matching."""
    print("=" * 70)
    print("Example 4: Author-Entity Matching")
    print("=" * 70)

    resolver = EntityResolver()

    test_cases = [
        ("@johnsmith", "John Smith", "Should match (name in handle)"),
        ("@jane_doe", "Jane Doe", "Should match (name in handle)"),
        ("@alice", "Alice", "Should match (exact)"),
        ("@bob_the_builder", "Bob", "Should match (partial)"),
        ("John Smith", "Smith", "Should match (last name)"),
        ("@johndoe", "Jane Smith", "Should NOT match"),
        ("@alice", "Bob", "Should NOT match"),
    ]

    print("\nTesting author-entity matching:\n")

    for author, entity, description in test_cases:
        is_match = resolver.is_author_mention(author, entity)
        symbol = "✓" if is_match else "✗"
        print(f"  {symbol} Author: '{author:20s}' Entity: '{entity:15s}'")
        print(f"     {description}")
        print()


def example_with_ner_engine():
    """Example 5: Integration with NER Engine."""
    print("=" * 70)
    print("Example 5: Integration with NER Engine")
    print("=" * 70)

    print("\nNOTE: This will download NER model on first run (~1GB)\n")

    resolver = EntityResolver()

    try:
        engine = NEREngine(enable_cache=True)

        texts = [
            "John Smith announced that Microsoft will open in Copenhagen.",
            "Microsoft CEO visited Copenhagen last week.",
            "John Smith met with officials in copenhagen.",
        ]

        print("Extracting and resolving entities:\n")

        all_canonicals = []

        for i, text in enumerate(texts, 1):
            print(f"Post {i}: {text}")

            # Extract entities
            entities = engine.extract_entities(text)

            # Resolve entities
            print(f"  Extracted entities:")
            for entity in entities:
                canonical = resolver.get_canonical_form(entity['text'])
                all_canonicals.append(canonical)
                print(f"    '{entity['text']:15s}' ({entity['type']}) → '{canonical}'")
            print()

        # Count entity mentions
        from collections import Counter
        mention_counts = Counter(all_canonicals)

        print("Entity mention counts:")
        for entity, count in mention_counts.most_common():
            print(f"  '{entity}': {count} mentions")

        print()

    except Exception as e:
        print(f"⚠️  Could not load NER engine: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")
        print()


def example_complete_pipeline():
    """Example 6: Complete pipeline with DataLoader."""
    print("=" * 70)
    print("Example 6: Complete Pipeline (CSV → NER → Resolution)")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        print("   Please ensure sample_data.csv exists in examples/ directory")
        return

    try:
        # Initialize components
        loader = DataLoader()
        engine = NEREngine(enable_cache=True)
        resolver = EntityResolver()

        print(f"\nProcessing: {example_file.name}\n")

        # Track entities
        entity_mentions = {}
        author_self_mentions = []

        # Process data
        for chunk in loader.load_csv(
            example_file,
            author_column='author',
            text_column='text',
            chunksize=5
        ):
            authors = chunk['author'].tolist()
            texts = chunk['text'].tolist()

            # Extract entities
            entities_batch, _ = engine.extract_entities_batch(
                texts,
                show_progress=False
            )

            # Resolve and track entities
            for author, entities in zip(authors, entities_batch):
                for entity in entities:
                    # Resolve to canonical form
                    canonical = resolver.get_canonical_form(entity['text'])

                    # Track mentions
                    key = (canonical, entity['type'])
                    if key not in entity_mentions:
                        entity_mentions[key] = 0
                    entity_mentions[key] += 1

                    # Check if author mentions themselves
                    if resolver.is_author_mention(author, entity['text']):
                        author_self_mentions.append({
                            'author': author,
                            'entity': canonical,
                            'type': entity['type']
                        })

        # Display results
        print("Top 10 most mentioned entities:")
        sorted_entities = sorted(
            entity_mentions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for (entity, etype), count in sorted_entities[:10]:
            print(f"  {count:2d}× '{entity}' ({etype})")

        print(f"\nAuthor self-mentions detected: {len(author_self_mentions)}")
        if author_self_mentions:
            print("Examples:")
            for mention in author_self_mentions[:3]:
                print(f"  {mention['author']} → {mention['entity']} ({mention['type']})")

        # Statistics
        stats = resolver.get_statistics()
        print(f"\nResolution statistics:")
        print(f"  Unique entities after resolution: {stats['unique_entities']}")
        print()

    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("   Make sure NER model is installed and dependencies are available")
        print()


def example_entity_deduplication():
    """Example 7: Entity deduplication in action."""
    print("=" * 70)
    print("Example 7: Entity Deduplication")
    print("=" * 70)

    resolver = EntityResolver()

    # Simulate extracting entities from multiple posts
    print("\nSimulating entity extraction from 5 posts:\n")

    posts_entities = [
        ["John Smith", "Microsoft", "Copenhagen"],
        ["john smith", "MICROSOFT", "copenhagen"],  # Same entities, different case
        ["Jane Doe", "Google", "Paris"],
        ["JOHN SMITH", "Microsoft Corporation", "Copenhagen"],  # Some same, some different
        ["Jane Doe", "google", "paris"],  # Same as post 3
    ]

    all_entities_before = []
    all_entities_after = []

    for i, entities in enumerate(posts_entities, 1):
        print(f"Post {i}: {entities}")
        all_entities_before.extend(entities)

        resolved = [resolver.get_canonical_form(e) for e in entities]
        all_entities_after.extend(resolved)

        print(f"  Resolved: {resolved}")
        print()

    # Count unique entities
    unique_before = len(set(all_entities_before))
    unique_after = len(set(all_entities_after))

    print(f"Summary:")
    print(f"  Total mentions: {len(all_entities_before)}")
    print(f"  Unique before resolution: {unique_before}")
    print(f"  Unique after resolution: {unique_after}")
    print(f"  Duplicates eliminated: {unique_before - unique_after}")
    print()


def example_statistics():
    """Example 8: Using statistics."""
    print("=" * 70)
    print("Example 8: Entity Resolution Statistics")
    print("=" * 70)

    resolver = EntityResolver()

    # Add various entities
    entities = [
        "John Smith", "jane doe", "Microsoft", "Copenhagen",
        "JOHN SMITH", "Jane Doe", "microsoft", "copenhagen",
        "Apple Inc.", "Paris", "Google", "Berlin"
    ]

    print("\nAdding entities:\n")
    for entity in entities:
        canonical = resolver.get_canonical_form(entity)
        print(f"  '{entity:20s}' → '{canonical}'")

    print("\nStatistics:")
    stats = resolver.get_statistics()
    print(f"  Unique entities: {stats['unique_entities']}")
    print(f"  Normalized forms (sample): {stats['normalized_forms'][:5]}")

    print("\nResetting resolver...")
    resolver.reset()

    stats_after = resolver.get_statistics()
    print(f"  Unique entities after reset: {stats_after['unique_entities']}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Entity Resolver Examples")
    print("=" * 70 + "\n")

    try:
        # Run examples
        example_basic_normalization()
        example_canonical_forms()
        example_multiple_entities()
        example_author_matching()
        example_entity_deduplication()
        example_statistics()
        example_with_ner_engine()
        example_complete_pipeline()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

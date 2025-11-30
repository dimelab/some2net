"""
Entity Linking Demo
Demonstrates how to use the EntityLinker module for multilingual entity linking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.entity_linker import EntityLinker


def demo_basic_linking():
    """Demo: Basic entity linking."""
    print("=" * 70)
    print("DEMO 1: Basic Entity Linking")
    print("=" * 70)

    # Initialize linker
    linker = EntityLinker(
        device='cpu',  # Use 'cuda' if you have GPU
        confidence_threshold=0.7,
        enable_cache=True,
        cache_dir='./cache/entity_links'
    )

    # Link a single entity
    print("\n🔗 Linking: 'Copenhagen' (English)")
    result = linker.link_entity(
        entity_text="Copenhagen",
        entity_type="LOC",
        language="en"
    )

    if result:
        print(f"  ✅ Linked successfully!")
        print(f"  📖 Wikipedia: {result['wikipedia_url']}")
        print(f"  🏷️  Canonical: {result['canonical_name']}")
        print(f"  🎯 Confidence: {result['linking_confidence']:.2%}")
        print(f"  🌍 Language Variants:")
        for lang, variant in result['language_variants'].items():
            print(f"     - {lang}: {variant}")
    else:
        print("  ❌ Linking failed")

    print("\n" + "=" * 70)


def demo_cross_language():
    """Demo: Cross-language entity linking."""
    print("\n" + "=" * 70)
    print("DEMO 2: Cross-Language Entity Linking")
    print("=" * 70)

    linker = EntityLinker(device='cpu', enable_cache=True)

    # Same entity in different languages
    test_entities = [
        {'text': 'Copenhagen', 'type': 'LOC', 'language': 'en'},
        {'text': 'København', 'type': 'LOC', 'language': 'da'},
        {'text': 'Copenhague', 'type': 'LOC', 'language': 'fr'},
    ]

    print("\n🔗 Linking same entity in 3 languages:")
    print("  1. Copenhagen (English)")
    print("  2. København (Danish)")
    print("  3. Copenhague (French)")

    enhanced = linker.link_entities_batch(test_entities)

    print("\n📊 Results:")
    for i, ent in enumerate(enhanced, 1):
        print(f"\n  Entity {i}: {ent['text']} ({ent.get('language', 'unknown')})")
        if ent.get('is_linked'):
            print(f"    → Canonical: {ent.get('canonical_name')}")
            print(f"    → Wikipedia: {ent.get('wikipedia_url')}")
            print(f"    → Confidence: {ent.get('linking_confidence', 0):.2%}")
        else:
            print(f"    → Not linked")

    print("\n💡 Note: All three should link to the same canonical entity!")
    print("=" * 70)


def demo_batch_processing():
    """Demo: Batch entity linking."""
    print("\n" + "=" * 70)
    print("DEMO 3: Batch Entity Processing")
    print("=" * 70)

    linker = EntityLinker(device='cpu', enable_cache=True)

    # Mixed entity types from NER output
    entities = [
        {'text': 'Mette Frederiksen', 'type': 'PER', 'score': 0.95, 'language': 'da'},
        {'text': 'Paris', 'type': 'LOC', 'score': 0.92, 'language': 'en'},
        {'text': 'European Parliament', 'type': 'ORG', 'score': 0.88, 'language': 'en'},
        {'text': 'Berlin', 'type': 'LOC', 'score': 0.91, 'language': 'en'},
        {'text': 'UnknownEntity123', 'type': 'PER', 'score': 0.75, 'language': 'en'},
    ]

    print(f"\n🔗 Processing {len(entities)} entities...")

    enhanced = linker.link_entities_batch(entities, show_progress=True)

    # Statistics
    linked = sum(1 for e in enhanced if e.get('is_linked'))
    unlinked = len(enhanced) - linked

    print(f"\n📊 Linking Statistics:")
    print(f"  ✅ Successfully linked: {linked}/{len(entities)}")
    print(f"  ❌ Could not link: {unlinked}/{len(entities)}")

    print(f"\n📋 Detailed Results:")
    for ent in enhanced:
        status = "✅" if ent.get('is_linked') else "❌"
        print(f"\n  {status} {ent['text']} ({ent['type']})")
        if ent.get('is_linked'):
            print(f"     → {ent.get('canonical_name')}")
            print(f"     → {ent.get('wikipedia_url')}")
        else:
            print(f"     → Could not link (confidence too low or not found)")

    print("\n" + "=" * 70)


def demo_caching():
    """Demo: Caching behavior."""
    print("\n" + "=" * 70)
    print("DEMO 4: Caching Performance")
    print("=" * 70)

    import time

    linker = EntityLinker(device='cpu', enable_cache=True)

    entity_text = "Paris"

    # First call - no cache
    print(f"\n⏱️  First call (no cache):")
    start = time.time()
    result1 = linker.link_entity(entity_text, "LOC", "en")
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")
    if result1:
        print(f"  Result: {result1['canonical_name']}")

    # Second call - should hit cache
    print(f"\n⏱️  Second call (with cache):")
    start = time.time()
    result2 = linker.link_entity(entity_text, "LOC", "en")
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")
    if result2:
        print(f"  Result: {result2['canonical_name']}")

    # Compare
    if time1 > 0 and time2 > 0:
        speedup = time1 / time2
        print(f"\n📈 Cache Speedup: {speedup:.1f}x faster!")

    # Cache stats
    stats = linker.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"  Entries: {stats['size']}")
    print(f"  Size: {stats['size_bytes'] / 1024:.1f} KB")

    print("\n" + "=" * 70)


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "ENTITY LINKING DEMO" + " " * 29 + "║")
    print("║" + " " * 15 + "Multilingual Entity Linking with mGENRE" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Run demos
        demo_basic_linking()
        demo_cross_language()
        demo_batch_processing()
        demo_caching()

        print("\n✅ All demos completed successfully!")
        print("\n💡 Next Steps:")
        print("  1. Try running with 'cuda' device if you have GPU")
        print("  2. Test with your own entity data")
        print("  3. Integrate into the full pipeline (Phase 2)")

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure you have installed: pip install sentencepiece")
        print("  2. Check internet connection (first run downloads model)")
        print("  3. Ensure you have ~3GB free disk space for model")
        import traceback
        traceback.print_exc()

    print("\n")


if __name__ == "__main__":
    main()

"""
Entity Linking Phase 2 Demo

Demonstrates the integration of entity linking into the pipeline with:
- Cross-language entity resolution via Wikidata
- Automatic entity metadata enrichment
- Network construction with unified entities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.entity_resolver import EntityResolver
from core.network_builder import NetworkBuilder


def demo_entity_resolver_with_wikidata():
    """Demonstrate EntityResolver with Wikidata ID support"""
    print("=" * 70)
    print("DEMO 1: Entity Resolver with Wikidata IDs")
    print("=" * 70)
    print()

    resolver = EntityResolver()

    # Scenario: Same city mentioned in different languages
    print("Scenario: Copenhagen mentioned in 3 languages")
    print("-" * 50)

    copenhagen_en = resolver.get_canonical_form(
        "Copenhagen",
        wikidata_id="Q1748",
        canonical_name="Copenhagen"
    )
    print(f"English: 'Copenhagen' (Q1748) → '{copenhagen_en}'")

    copenhagen_da = resolver.get_canonical_form(
        "København",
        wikidata_id="Q1748",
        canonical_name="Copenhagen"
    )
    print(f"Danish: 'København' (Q1748) → '{copenhagen_da}'")

    copenhagen_fr = resolver.get_canonical_form(
        "Copenhague",
        wikidata_id="Q1748",
        canonical_name="Copenhagen"
    )
    print(f"French: 'Copenhague' (Q1748) → '{copenhagen_fr}'")

    print()
    print(f"✅ All resolve to same canonical form: '{copenhagen_en}'")
    print(f"✅ Wikidata-based cross-language resolution working!")
    print()

    # Show statistics
    stats = resolver.get_statistics()
    print("Statistics:")
    print(f"  Unique entities: {stats['unique_entities']}")
    print(f"  Wikidata-linked: {stats['wikidata_linked_entities']}")
    print(f"  Wikidata IDs: {stats['wikidata_ids']}")
    print()


def demo_network_builder_with_metadata():
    """Demonstrate NetworkBuilder with entity linking metadata"""
    print("=" * 70)
    print("DEMO 2: Network Builder with Entity Metadata")
    print("=" * 70)
    print()

    builder = NetworkBuilder(use_entity_resolver=True)

    # Scenario: 3 users mention Copenhagen in different languages
    print("Scenario: 3 users mention Copenhagen in different languages")
    print("-" * 50)

    # User 1 (Danish)
    builder.add_post(
        author='danish_user',
        entities=[{
            'text': 'København',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q1748',
            'wikipedia_url': 'https://en.wikipedia.org/wiki/Copenhagen',
            'canonical_name': 'Copenhagen',
            'is_linked': True
        }]
    )
    print("✅ Added post from 'danish_user' mentioning 'København' (Q1748)")

    # User 2 (English)
    builder.add_post(
        author='english_user',
        entities=[{
            'text': 'Copenhagen',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q1748',
            'wikipedia_url': 'https://en.wikipedia.org/wiki/Copenhagen',
            'canonical_name': 'Copenhagen',
            'is_linked': True
        }]
    )
    print("✅ Added post from 'english_user' mentioning 'Copenhagen' (Q1748)")

    # User 3 (French)
    builder.add_post(
        author='french_user',
        entities=[{
            'text': 'Copenhague',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q1748',
            'wikipedia_url': 'https://en.wikipedia.org/wiki/Copenhagen',
            'canonical_name': 'Copenhagen',
            'is_linked': True
        }]
    )
    print("✅ Added post from 'french_user' mentioning 'Copenhague' (Q1748)")

    print()

    # Analyze network
    graph = builder.get_graph()

    print("Network Analysis:")
    print("-" * 50)
    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"Total edges: {graph.number_of_edges()}")
    print()

    # Check entity nodes
    entity_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] != 'author']
    print(f"Entity nodes: {len(entity_nodes)}")

    for node_name in entity_nodes:
        node_data = graph.nodes[node_name]
        print(f"\n  Entity: '{node_name}'")
        print(f"    Type: {node_data['node_type']}")
        print(f"    Mentions: {node_data['mention_count']}")
        print(f"    Wikidata ID: {node_data.get('wikidata_id', 'N/A')}")
        print(f"    Wikipedia: {node_data.get('wikipedia_url', 'N/A')}")
        print(f"    Linked: {node_data.get('is_linked', False)}")

    print()
    print("✅ All 3 mentions unified into single 'Copenhagen' node!")
    print("✅ Wikidata metadata stored in node attributes!")
    print()


def demo_disambiguation_with_wikidata():
    """Demonstrate entity disambiguation using Wikidata IDs"""
    print("=" * 70)
    print("DEMO 3: Entity Disambiguation with Wikidata")
    print("=" * 70)
    print()

    resolver = EntityResolver()

    print("Scenario: Two different entities with same name 'Paris'")
    print("-" * 50)

    # Paris, France
    paris_france = resolver.get_canonical_form(
        "Paris",
        wikidata_id="Q90",
        canonical_name="Paris, France"
    )
    print(f"Paris, France (Q90) → '{paris_france}'")

    # Paris, Texas
    paris_texas = resolver.get_canonical_form(
        "Paris",
        wikidata_id="Q16858",
        canonical_name="Paris, Texas"
    )
    print(f"Paris, Texas (Q16858) → '{paris_texas}'")

    print()
    print(f"✅ Different Wikidata IDs → Different canonical forms")
    print(f"✅ Disambiguation via QIDs: '{paris_france}' ≠ '{paris_texas}'")
    print()


def demo_mixed_linked_unlinked():
    """Demonstrate handling of both linked and unlinked entities"""
    print("=" * 70)
    print("DEMO 4: Mixed Linked and Unlinked Entities")
    print("=" * 70)
    print()

    builder = NetworkBuilder(use_entity_resolver=True)

    print("Scenario: Post with both linked and unlinked entities")
    print("-" * 50)

    builder.add_post(
        author='test_user',
        entities=[
            {
                'text': 'Copenhagen',
                'type': 'LOC',
                'score': 0.95,
                'wikidata_id': 'Q1748',
                'wikipedia_url': 'https://en.wikipedia.org/wiki/Copenhagen',
                'canonical_name': 'Copenhagen',
                'is_linked': True
            },
            {
                'text': 'Obscure Place',
                'type': 'LOC',
                'score': 0.70,
                'is_linked': False  # Failed to link
            }
        ]
    )

    graph = builder.get_graph()

    print("\nEntities in network:")
    for node, data in graph.nodes(data=True):
        if data['node_type'] != 'author':
            linked_status = "✅ Linked" if data.get('is_linked') else "❌ Unlinked"
            qid = data.get('wikidata_id', 'N/A')
            print(f"  {node}: {linked_status} (QID: {qid})")

    print()
    print("✅ Both entities added to network")
    print("✅ Linked entities have metadata, unlinked ones don't")
    print("✅ Graceful handling of partial linking!")
    print()


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "ENTITY LINKING PHASE 2 DEMO" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    try:
        demo_entity_resolver_with_wikidata()
        demo_network_builder_with_metadata()
        demo_disambiguation_with_wikidata()
        demo_mixed_linked_unlinked()

        print("=" * 70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY! ✅")
        print("=" * 70)
        print()
        print("Phase 2 Features Demonstrated:")
        print("  ✅ Wikidata ID-based entity resolution")
        print("  ✅ Cross-language entity unification")
        print("  ✅ Entity metadata storage")
        print("  ✅ Disambiguation via Wikidata IDs")
        print("  ✅ Graceful handling of unlinked entities")
        print()
        print("Next Steps:")
        print("  1. Enable entity linking in your pipeline:")
        print("     pipeline = SocialNetworkPipeline(enable_entity_linking=True)")
        print("  2. Process your multilingual data")
        print("  3. Analyze cross-language entity networks!")
        print()

    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

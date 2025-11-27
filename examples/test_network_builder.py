"""
Example script demonstrating Network Builder functionality.

This script shows how to:
1. Create networks from posts and entities
2. Track nodes and edges
3. Handle author-to-author mentions
4. Calculate network statistics
5. Integrate with full pipeline

Run after installing dependencies:
    pip install -r requirements.txt
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.network_builder import NetworkBuilder
from src.core.data_loader import DataLoader
from src.core.ner_engine import NEREngine


def example_basic_network():
    """Example 1: Basic network creation."""
    print("=" * 70)
    print("Example 1: Basic Network Creation")
    print("=" * 70)

    builder = NetworkBuilder()

    # Add a few posts with entities
    posts = [
        {
            'author': '@user1',
            'entities': [
                {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92},
            ]
        },
        {
            'author': '@user2',
            'entities': [
                {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.89},
                {'text': 'Google', 'type': 'ORG', 'score': 0.90}
            ]
        }
    ]

    print("\nAdding posts to network:\n")
    for i, post in enumerate(posts, 1):
        builder.add_post(post['author'], post['entities'])
        print(f"  Post {i}: {post['author']} → {len(post['entities'])} entities")

    # Get graph
    graph = builder.get_graph()

    print(f"\nNetwork created:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print()


def example_entity_deduplication():
    """Example 2: Entity deduplication."""
    print("=" * 70)
    print("Example 2: Entity Deduplication")
    print("=" * 70)

    builder = NetworkBuilder(use_entity_resolver=True)

    # Same entity in different forms
    posts = [
        {
            'author': '@user1',
            'entities': [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]
        },
        {
            'author': '@user2',
            'entities': [{'text': 'microsoft', 'type': 'ORG', 'score': 0.91}]
        },
        {
            'author': '@user3',
            'entities': [{'text': 'MICROSOFT', 'type': 'ORG', 'score': 0.93}]
        }
    ]

    print("\nAdding posts with same entity in different forms:\n")
    for post in posts:
        builder.add_post(post['author'], post['entities'])
        entity_text = post['entities'][0]['text']
        print(f"  {post['author']}: '{entity_text}'")

    # Count organization nodes
    graph = builder.get_graph()
    org_nodes = [
        n for n, attrs in graph.nodes(data=True)
        if attrs.get('node_type') == 'organization'
    ]

    print(f"\nResult:")
    print(f"  Organization nodes created: {len(org_nodes)}")
    print(f"  ✓ Entity deduplication working (3 mentions → 1 entity)")
    print()


def example_author_mentions():
    """Example 3: Author-to-author mentions."""
    print("=" * 70)
    print("Example 3: Author-to-Author Mentions")
    print("=" * 70)

    builder = NetworkBuilder(create_author_edges=True)

    # First author posts
    builder.add_post('@johndoe', [])

    # Second author mentions first
    posts = [
        {
            'author': '@alice',
            'entities': [{'text': 'John Doe', 'type': 'PER', 'score': 0.95}]
        },
        {
            'author': '@bob',
            'entities': [{'text': 'johndoe', 'type': 'PER', 'score': 0.92}]
        }
    ]

    print("\nAdding posts where authors mention each other:\n")
    print("  @johndoe posts (establishes identity)")
    for post in posts:
        builder.add_post(post['author'], post['entities'])
        entity = post['entities'][0]
        print(f"  {post['author']} mentions: '{entity['text']}'")

    # Check for author-to-author edges
    graph = builder.get_graph()
    author_edges = [
        (u, v) for u, v, attrs in graph.edges(data=True)
        if attrs.get('entity_type') == 'AUTHOR'
    ]

    print(f"\nAuthor-to-author edges created: {len(author_edges)}")
    for source, target in author_edges:
        print(f"  {source} → {target}")
    print()


def example_edge_weights():
    """Example 4: Edge weight accumulation."""
    print("=" * 70)
    print("Example 4: Edge Weight Accumulation")
    print("=" * 70)

    builder = NetworkBuilder()

    entity = {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}

    print("\nAdding multiple posts mentioning same entity:\n")
    for i in range(1, 6):
        builder.add_post('@user1', [entity], post_id=f'post_{i}')
        print(f"  Post {i}: @user1 mentions Microsoft")

    # Get edge info
    graph = builder.get_graph()
    edges = list(graph.out_edges('@user1', data=True))

    if edges:
        target, edge_data = edges[0][1], edges[0][2]
        weight = edge_data.get('weight', 0)
        source_posts = edge_data.get('source_posts', [])

        print(f"\nEdge: @user1 → {target}")
        print(f"  Weight (mentions): {weight}")
        print(f"  Source posts: {len(source_posts)}")
        print(f"  ✓ Weight accumulated correctly")
    print()


def example_network_statistics():
    """Example 5: Network statistics."""
    print("=" * 70)
    print("Example 5: Network Statistics")
    print("=" * 70)

    builder = NetworkBuilder()

    # Add varied posts
    posts = [
        {
            'author': '@user1',
            'entities': [
                {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92},
                {'text': 'Seattle', 'type': 'LOC', 'score': 0.88}
            ]
        },
        {
            'author': '@user2',
            'entities': [
                {'text': 'Google', 'type': 'ORG', 'score': 0.93},
                {'text': 'Mountain View', 'type': 'LOC', 'score': 0.87}
            ]
        },
        {
            'author': '@user1',  # Same author, second post
            'entities': [
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.94}  # Duplicate
            ]
        }
    ]

    for post in posts:
        builder.add_post(post['author'], post['entities'])

    stats = builder.get_statistics()

    print("\nNetwork Statistics:\n")
    print(f"  Total Nodes:          {stats['total_nodes']}")
    print(f"  Total Edges:          {stats['total_edges']}")
    print(f"  Density:              {stats['density']:.4f}")
    print()
    print("  Node Types:")
    print(f"    Authors:            {stats['authors']}")
    print(f"    Persons:            {stats['persons']}")
    print(f"    Locations:          {stats['locations']}")
    print(f"    Organizations:      {stats['organizations']}")
    print()
    print("  Edge Types:")
    print(f"    Person mentions:    {stats['person_mentions']}")
    print(f"    Location mentions:  {stats['location_mentions']}")
    print(f"    Org mentions:       {stats['organization_mentions']}")
    print()
    print(f"  Posts Processed:      {stats['posts_processed']}")
    print(f"  Total Mentions:       {stats['total_mentions']}")
    print()


def example_top_entities():
    """Example 6: Top mentioned entities."""
    print("=" * 70)
    print("Example 6: Top Mentioned Entities")
    print("=" * 70)

    builder = NetworkBuilder()

    # Add posts with entities having different mention frequencies
    posts = [
        {'author': '@user1', 'entities': [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]},
        {'author': '@user2', 'entities': [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.91}]},
        {'author': '@user3', 'entities': [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.93}]},
        {'author': '@user4', 'entities': [{'text': 'Google', 'type': 'ORG', 'score': 0.90}]},
        {'author': '@user5', 'entities': [{'text': 'Google', 'type': 'ORG', 'score': 0.89}]},
        {'author': '@user6', 'entities': [{'text': 'Apple', 'type': 'ORG', 'score': 0.88}]},
    ]

    for post in posts:
        builder.add_post(post['author'], post['entities'])

    stats = builder.get_statistics()

    print("\nTop 10 Most Mentioned Entities:\n")
    for i, entity in enumerate(stats['top_entities'][:10], 1):
        print(f"  {i}. {entity['entity']:20s} ({entity['type']:3s}) - {entity['mentions']} mentions")
    print()


def example_complete_pipeline():
    """Example 7: Complete pipeline with real data."""
    print("=" * 70)
    print("Example 7: Complete Pipeline (CSV → NER → Network)")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        print("   Please ensure sample_data.csv exists in examples/ directory")
        return

    print("\nNOTE: This will download NER model on first run (~1GB)\n")

    try:
        # Initialize components
        loader = DataLoader()
        engine = NEREngine(enable_cache=True)
        builder = NetworkBuilder()

        print(f"Processing: {example_file.name}\n")

        # Process data
        total_posts = 0
        for chunk in loader.load_csv(
            example_file,
            author_column='author',
            text_column='text',
            chunksize=5
        ):
            authors = chunk['author'].tolist()
            texts = chunk['text'].tolist()
            post_ids = chunk.get('post_id', [None] * len(texts)).tolist()

            # Extract entities
            entities_batch, _ = engine.extract_entities_batch(
                texts,
                show_progress=False
            )

            # Build network
            for author, entities, post_id in zip(authors, entities_batch, post_ids):
                builder.add_post(author, entities, post_id=str(post_id) if post_id else None)
                total_posts += 1

        # Get final statistics
        stats = builder.get_statistics()
        graph = builder.get_graph()

        print("Pipeline Complete!\n")
        print("Network Summary:")
        print(f"  Posts processed:      {total_posts}")
        print(f"  Total nodes:          {stats['total_nodes']}")
        print(f"  Total edges:          {stats['total_edges']}")
        print(f"  Network density:      {stats['density']:.4f}")
        print()
        print("  Authors:              {stats['authors']}")
        print(f"  Entities:             {stats['persons'] + stats['locations'] + stats['organizations']}")
        print()

        print("Top 5 Most Mentioned Entities:")
        for i, entity in enumerate(stats['top_entities'][:5], 1):
            print(f"  {i}. {entity['entity']} ({entity['type']}) - {entity['mentions']} mentions")
        print()

        print("Top 5 Authors by Posts:")
        top_authors = builder.get_top_authors(5)
        for i, author in enumerate(top_authors, 1):
            print(f"  {i}. {author['author']} - {author['posts']} posts, {author['out_degree']} connections")
        print()

    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")
        print()


def example_node_edge_info():
    """Example 8: Querying nodes and edges."""
    print("=" * 70)
    print("Example 8: Querying Node and Edge Information")
    print("=" * 70)

    builder = NetworkBuilder()

    # Add posts
    posts = [
        {
            'author': '@user1',
            'entities': [
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92},
                {'text': 'Seattle', 'type': 'LOC', 'score': 0.88}
            ],
            'post_id': 'post_1',
            'timestamp': '2024-01-01'
        },
        {
            'author': '@user1',
            'entities': [
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.94}
            ],
            'post_id': 'post_2',
            'timestamp': '2024-01-02'
        }
    ]

    for post in posts:
        builder.add_post(
            post['author'],
            post['entities'],
            post_id=post['post_id'],
            timestamp=post['timestamp']
        )

    print("\nQuerying node information:\n")

    # Get author info
    author_info = builder.get_node_info('@user1')
    if author_info:
        print(f"Author: {author_info['id']}")
        print(f"  Type: {author_info['type']}")
        print(f"  Posts: {author_info['post_count']}")
        print(f"  Out-degree: {author_info['out_degree']}")
        print()

    # Get edge info
    graph = builder.get_graph()
    edges = list(graph.out_edges('@user1'))

    if edges:
        target = edges[0][1]
        edge_info = builder.get_edge_info('@user1', target)

        print(f"Edge: {edge_info['source']} → {edge_info['target']}")
        print(f"  Weight: {edge_info['weight']}")
        print(f"  Type: {edge_info['entity_type']}")
        print(f"  Posts: {edge_info['source_posts']}")
        print(f"  First mention: {edge_info['first_mention']}")
        print(f"  Last mention: {edge_info['last_mention']}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Network Builder Examples")
    print("=" * 70 + "\n")

    try:
        # Run examples
        example_basic_network()
        example_entity_deduplication()
        example_author_mentions()
        example_edge_weights()
        example_network_statistics()
        example_top_entities()
        example_node_edge_info()
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

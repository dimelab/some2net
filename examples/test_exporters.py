"""
Example script demonstrating network export functionality.

This script shows how to:
1. Export networks in different formats (GEXF, GraphML, JSON, CSV)
2. Export statistics
3. Use export_all_formats for convenience
4. Integrate with NetworkBuilder output
5. Load exported files in other tools

Run after installing dependencies:
    pip install -r requirements.txt
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.network_builder import NetworkBuilder
from src.utils.exporters import (
    export_gexf,
    export_graphml,
    export_json,
    export_edgelist,
    export_adjacency_matrix,
    export_statistics,
    export_all_formats
)
import networkx as nx


def example_create_sample_network():
    """Create a sample network for demonstration."""
    print("=" * 70)
    print("Creating Sample Network")
    print("=" * 70)

    builder = NetworkBuilder()

    # Add posts
    posts = [
        {
            'author': '@user1',
            'entities': [
                {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92},
                {'text': 'Seattle', 'type': 'LOC', 'score': 0.88}
            ],
            'post_id': 'post_1',
            'timestamp': '2024-01-01'
        },
        {
            'author': '@user2',
            'entities': [
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.93},
                {'text': 'Google', 'type': 'ORG', 'score': 0.91}
            ],
            'post_id': 'post_2',
            'timestamp': '2024-01-02'
        },
        {
            'author': '@user1',
            'entities': [
                {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.89},
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.94}
            ],
            'post_id': 'post_3',
            'timestamp': '2024-01-03'
        },
        {
            'author': '@user3',
            'entities': [
                {'text': 'John Smith', 'type': 'PER', 'score': 0.96},
                {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.90}
            ],
            'post_id': 'post_4',
            'timestamp': '2024-01-04'
        }
    ]

    print(f"\nAdding {len(posts)} posts to network...\n")
    for post in posts:
        builder.add_post(
            author=post['author'],
            entities=post['entities'],
            post_id=post['post_id'],
            timestamp=post['timestamp']
        )

    graph = builder.get_graph()
    stats = builder.get_statistics()

    print(f"Network created:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Authors: {stats['authors']}")
    print(f"  Entities: {stats['persons'] + stats['locations'] + stats['organizations']}")
    print()

    return graph, stats


def example_export_gexf(graph, stats):
    """Example 1: Export to GEXF format (primary format for Gephi)."""
    print("=" * 70)
    print("Example 1: Export to GEXF Format (Primary)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    filepath = str(output_dir / "network.gexf")

    print(f"\nExporting network to GEXF format...")
    print(f"  Output: {filepath}")

    export_gexf(graph, filepath)

    # Verify by loading
    loaded = nx.read_gexf(filepath)
    print(f"\n✓ GEXF export successful")
    print(f"  Nodes in file: {len(loaded.nodes)}")
    print(f"  Edges in file: {len(loaded.edges)}")
    print(f"\nUsage: Open {filepath} in Gephi for visualization")
    print()


def example_export_graphml(graph, stats):
    """Example 2: Export to GraphML format."""
    print("=" * 70)
    print("Example 2: Export to GraphML Format")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    filepath = str(output_dir / "network.graphml")

    print(f"\nExporting network to GraphML format...")
    print(f"  Output: {filepath}")

    export_graphml(graph, filepath)

    # Verify by loading
    loaded = nx.read_graphml(filepath)
    print(f"\n✓ GraphML export successful")
    print(f"  Nodes in file: {len(loaded.nodes)}")
    print(f"  Edges in file: {len(loaded.edges)}")
    print(f"\nUsage: Open {filepath} in yEd, Cytoscape, or other tools")
    print()


def example_export_json(graph, stats):
    """Example 3: Export to JSON format (D3.js compatible)."""
    print("=" * 70)
    print("Example 3: Export to JSON Format (D3.js)")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    filepath = str(output_dir / "network.json")

    print(f"\nExporting network to JSON format...")
    print(f"  Output: {filepath}")

    export_json(graph, filepath)

    # Show JSON structure
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n✓ JSON export successful")
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Links: {len(data['links'])}")
    print(f"\nJSON Structure:")
    print(f"  - nodes: [{len(data['nodes'])} items]")
    print(f"  - links: [{len(data['links'])} items]")
    print(f"\nSample node:")
    if data['nodes']:
        import pprint
        pprint.pprint(data['nodes'][0], indent=4)
    print(f"\nUsage: Load in D3.js, Vis.js, or web visualizations")
    print()


def example_export_edgelist(graph, stats):
    """Example 4: Export to CSV edge list."""
    print("=" * 70)
    print("Example 4: Export to CSV Edge List")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    filepath = str(output_dir / "network_edgelist.csv")

    print(f"\nExporting network to CSV edge list...")
    print(f"  Output: {filepath}")

    export_edgelist(graph, filepath)

    # Show first few rows
    import csv
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    print(f"\n✓ Edge list export successful")
    print(f"  Total edges: {len(rows) - 1}")
    print(f"\nFirst 3 rows:")
    for row in rows[:4]:
        print(f"  {', '.join(row)}")
    print(f"\nUsage: Import in Excel, R, Python pandas, etc.")
    print()


def example_export_adjacency_matrix(graph, stats):
    """Example 5: Export adjacency matrix."""
    print("=" * 70)
    print("Example 5: Export Adjacency Matrix")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    filepath = str(output_dir / "network_adjacency.csv")

    print(f"\nExporting adjacency matrix...")
    print(f"  Output: {filepath}")

    export_adjacency_matrix(graph, filepath)

    print(f"\n✓ Adjacency matrix export successful")
    print(f"  Dimensions: {len(graph.nodes)} × {len(graph.nodes)}")
    print(f"\nUsage: Matrix analysis, igraph, NetworkX, etc.")
    print()


def example_export_statistics(graph, stats):
    """Example 6: Export network statistics."""
    print("=" * 70)
    print("Example 6: Export Network Statistics")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    filepath = str(output_dir / "network_statistics.json")

    print(f"\nExporting network statistics...")
    print(f"  Output: {filepath}")

    export_statistics(stats, filepath)

    # Show statistics
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        loaded_stats = json.load(f)

    print(f"\n✓ Statistics export successful")
    print(f"\nKey Statistics:")
    print(f"  Total nodes:          {loaded_stats['total_nodes']}")
    print(f"  Total edges:          {loaded_stats['total_edges']}")
    print(f"  Network density:      {loaded_stats['density']:.4f}")
    print(f"  Authors:              {loaded_stats['authors']}")
    print(f"  Persons:              {loaded_stats['persons']}")
    print(f"  Locations:            {loaded_stats['locations']}")
    print(f"  Organizations:        {loaded_stats['organizations']}")
    print(f"\nTop 3 Entities:")
    for i, entity in enumerate(loaded_stats['top_entities'][:3], 1):
        print(f"  {i}. {entity['entity']} ({entity['type']}) - {entity['mentions']} mentions")
    print()


def example_export_all_formats(graph, stats):
    """Example 7: Export all formats at once."""
    print("=" * 70)
    print("Example 7: Export All Formats at Once")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output" / "all_formats"

    print(f"\nExporting network in all formats...")
    print(f"  Output directory: {output_dir}")
    print()

    files = export_all_formats(graph, stats, str(output_dir), "complete_network")

    print(f"\n✓ Exported {len(files)} formats:")
    for format_name, filepath in files.items():
        file_path = Path(filepath)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        print(f"  - {format_name:15s}: {file_path.name:40s} ({file_size:,} bytes)")
    print()


def example_complete_pipeline():
    """Example 8: Complete pipeline from CSV to exports."""
    print("=" * 70)
    print("Example 8: Complete Pipeline (CSV → Network → Export)")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        print("   Skipping complete pipeline example")
        return

    print("\nNOTE: This will download NER model on first run (~1GB)\n")

    try:
        from src.core.data_loader import DataLoader
        from src.core.ner_engine import NEREngine

        # Initialize
        print("Initializing components...")
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

        # Get results
        graph = builder.get_graph()
        stats = builder.get_statistics()

        print("Pipeline Complete!\n")
        print("Network Summary:")
        print(f"  Posts processed:      {total_posts}")
        print(f"  Total nodes:          {stats['total_nodes']}")
        print(f"  Total edges:          {stats['total_edges']}")
        print()

        # Export all formats
        output_dir = Path(__file__).parent / "output" / "pipeline_output"
        files = export_all_formats(graph, stats, str(output_dir), "pipeline_network")

        print(f"Exported {len(files)} formats to: {output_dir}")
        print()

    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")
        print()


def example_load_exported_network():
    """Example 9: Load and analyze exported network."""
    print("=" * 70)
    print("Example 9: Load Exported Network Files")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    gexf_file = output_dir / "network.gexf"
    json_file = output_dir / "network.json"
    stats_file = output_dir / "network_statistics.json"

    if not gexf_file.exists():
        print("\n⚠️  No exported files found. Run other examples first.")
        return

    # Load GEXF
    print("\nLoading GEXF file...")
    G = nx.read_gexf(str(gexf_file))
    print(f"  Loaded graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Analyze
    print("\nNetwork Analysis:")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / len(G.nodes):.2f}")
    print(f"  Number of weakly connected components: {nx.number_weakly_connected_components(G)}")

    # Find most connected node
    degrees = dict(G.degree())
    most_connected = max(degrees.items(), key=lambda x: x[1])
    print(f"  Most connected node: {most_connected[0]} (degree: {most_connected[1]})")

    # Load statistics
    if stats_file.exists():
        import json
        print(f"\nLoading statistics from JSON...")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"  Posts processed: {stats['posts_processed']}")
        print(f"  Network density: {stats['density']:.4f}")

    print()


def main():
    """Run all export examples."""
    print("\n" + "=" * 70)
    print("Network Export Examples")
    print("=" * 70 + "\n")

    try:
        # Create sample network
        graph, stats = example_create_sample_network()

        # Individual export examples
        example_export_gexf(graph, stats)
        example_export_graphml(graph, stats)
        example_export_json(graph, stats)
        example_export_edgelist(graph, stats)
        example_export_adjacency_matrix(graph, stats)
        example_export_statistics(graph, stats)

        # Batch export
        example_export_all_formats(graph, stats)

        # Load and analyze
        example_load_exported_network()

        # Complete pipeline (optional - requires NER model)
        # example_complete_pipeline()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print(f"\nExported files location: {Path(__file__).parent / 'output'}")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

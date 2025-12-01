"""
Example: Creating a Hashtag Network

This example demonstrates how to create a network based on hashtag extraction
from social media posts. The network shows which authors use which hashtags,
with edge weights representing frequency of use.
"""

from src.core.pipeline import SocialNetworkPipeline
import tempfile
import csv
from pathlib import Path


def create_sample_data():
    """Create sample social media data with hashtags."""
    temp_dir = tempfile.mkdtemp()
    filepath = Path(temp_dir) / "social_media_hashtags.csv"

    data = [
        ['author', 'text', 'timestamp', 'sentiment'],
        ['@alice', 'I love #python and #datascience! #machinelearning is amazing', '2024-01-01', 'positive'],
        ['@bob', 'Working on #python projects today #coding #webdev', '2024-01-02', 'neutral'],
        ['@charlie', '#javascript #webdev #frontend development is fun', '2024-01-03', 'positive'],
        ['@alice', 'Another day of #datascience and #python coding', '2024-01-04', 'positive'],
        ['@dave', '#machinelearning and #AI are transforming everything', '2024-01-05', 'neutral'],
        ['@bob', '#python #django #webdev stack is powerful', '2024-01-06', 'positive'],
        ['@alice', 'Deep dive into #machinelearning algorithms #datascience', '2024-01-07', 'neutral'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


def main():
    """Run hashtag network extraction example."""

    print("=" * 70)
    print("HASHTAG NETWORK EXTRACTION EXAMPLE")
    print("=" * 70)

    # Step 1: Create sample data
    print("\n1. Creating sample data...")
    filepath = create_sample_data()
    print(f"   Created: {filepath}")

    # Step 2: Initialize pipeline with hashtag extraction
    print("\n2. Initializing pipeline with hashtag extractor...")
    pipeline = SocialNetworkPipeline(
        extraction_method="hashtag",
        extractor_config={
            'normalize_case': True  # Normalize #Python, #PYTHON to #python
        }
    )
    print("   Pipeline initialized!")

    # Step 3: Process file with metadata
    print("\n3. Processing social media data...")
    graph, stats = pipeline.process_file(
        filepath,
        author_column='author',
        text_column='text',
        node_metadata_columns=['sentiment'],  # Attach sentiment to author nodes
        edge_metadata_columns=['timestamp'],  # Attach timestamp to edges
        show_progress=False
    )
    print("   Processing complete!")

    # Step 4: Display results
    print("\n" + "=" * 70)
    print("NETWORK STATISTICS")
    print("=" * 70)
    print(f"\nTotal Nodes: {stats['total_nodes']}")
    print(f"  - Authors: {stats['authors']}")
    print(f"  - Hashtags: {stats['total_nodes'] - stats['authors']}")
    print(f"\nTotal Edges: {stats['total_edges']}")
    print(f"Network Density: {stats['density']:.4f}")
    print(f"Average Degree: {stats['average_degree']:.2f}")

    print(f"\nPosts Processed: {stats['processing_metadata']['total_posts']}")
    print(f"Hashtags Extracted: {stats['processing_metadata']['entities_extracted']}")

    # Step 5: Show top hashtags
    print("\n" + "=" * 70)
    print("TOP HASHTAGS BY MENTIONS")
    print("=" * 70)

    # Count hashtag mentions
    hashtag_counts = {}
    for node in graph.nodes():
        if node.startswith('#'):
            hashtag_counts[node] = graph.nodes[node].get('mention_count', 0)

    # Sort and display
    sorted_hashtags = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (hashtag, count) in enumerate(sorted_hashtags[:10], 1):
        print(f"{i:2d}. {hashtag:20s} - {count} mentions")

    # Step 6: Show author-hashtag relationships
    print("\n" + "=" * 70)
    print("AUTHOR-HASHTAG RELATIONSHIPS")
    print("=" * 70)

    for author in ['@alice', '@bob', '@charlie', '@dave']:
        if author in graph:
            hashtags = [target for _, target in graph.out_edges(author)]
            if hashtags:
                print(f"\n{author}:")
                for hashtag in hashtags:
                    weight = graph.edges[author, hashtag].get('weight', 1)
                    print(f"  - {hashtag} (used {weight} times)")

    # Step 7: Export network
    print("\n" + "=" * 70)
    print("EXPORTING NETWORK")
    print("=" * 70)

    output_dir = "./output/hashtag_network"
    files = pipeline.export_network(
        output_dir=output_dir,
        base_name="hashtag_network",
        formats=['gexf', 'graphml', 'json', 'statistics']
    )

    print("\nExported files:")
    for fmt, path in files.items():
        print(f"  {fmt:15s}: {path}")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Open the GEXF file in Gephi for visualization")
    print("  2. Use the GraphML file in other network analysis tools")
    print("  3. Use the JSON file for web-based visualizations (D3.js)")
    print("  4. Review the statistics JSON for detailed metrics")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

"""
Example: Network with Rich Metadata

This example demonstrates how to attach metadata from CSV columns to nodes
and edges in the network. This is useful for enriching networks with contextual
information like timestamps, sentiment scores, engagement metrics, etc.
"""

from src.core.pipeline import SocialNetworkPipeline
import tempfile
import csv
from pathlib import Path


def create_sample_data():
    """Create sample social media data with rich metadata."""
    temp_dir = tempfile.mkdtemp()
    filepath = Path(temp_dir) / "social_media_metadata.csv"

    data = [
        ['post_id', 'author', 'text', 'timestamp', 'likes', 'retweets', 'sentiment', 'location', 'verified'],
        ['1', '@alice', 'Loving #python today!', '2024-01-01 10:30', '150', '45', 'positive', 'New York', 'True'],
        ['2', '@bob', 'Check out #machinelearning!', '2024-01-01 14:20', '89', '23', 'neutral', 'San Francisco', 'False'],
        ['3', '@alice', '#datascience is amazing', '2024-01-02 09:15', '203', '67', 'positive', 'New York', 'True'],
        ['4', '@charlie', 'Working with #python', '2024-01-02 16:45', '45', '12', 'neutral', 'London', 'False'],
        ['5', '@alice', 'New #machinelearning project!', '2024-01-03 11:00', '312', '98', 'positive', 'New York', 'True'],
        ['6', '@dave', 'Learning #datascience basics', '2024-01-03 13:30', '67', '19', 'neutral', 'Boston', 'False'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


def main():
    """Run metadata-enriched network extraction example."""

    print("=" * 70)
    print("METADATA-ENRICHED NETWORK EXAMPLE")
    print("=" * 70)

    # Step 1: Create sample data
    print("\n1. Creating sample data with rich metadata...")
    filepath = create_sample_data()
    print(f"   Created: {filepath}")

    # Step 2: Initialize pipeline
    print("\n2. Initializing pipeline with hashtag extractor...")
    pipeline = SocialNetworkPipeline(
        extraction_method="hashtag",
        extractor_config={'normalize_case': True}
    )
    print("   Pipeline initialized!")

    # Step 3: Process file WITH metadata
    print("\n3. Processing data with metadata attachment...")
    print("   Node metadata: location, verified status")
    print("   Edge metadata: timestamp, likes, retweets, sentiment")

    graph, stats = pipeline.process_file(
        filepath,
        author_column='author',
        text_column='text',
        # Metadata to attach to NODES (authors)
        node_metadata_columns=['location', 'verified'],
        # Metadata to attach to EDGES (author -> hashtag relationships)
        edge_metadata_columns=['timestamp', 'likes', 'retweets', 'sentiment'],
        show_progress=False
    )
    print("   Processing complete!")

    # Step 4: Display basic statistics
    print("\n" + "=" * 70)
    print("NETWORK STATISTICS")
    print("=" * 70)
    print(f"\nTotal Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print(f"Posts Processed: {stats['processing_metadata']['total_posts']}")

    # Step 5: Show author metadata
    print("\n" + "=" * 70)
    print("AUTHOR METADATA")
    print("=" * 70)

    for author in ['@alice', '@bob', '@charlie', '@dave']:
        if author in graph:
            node_data = graph.nodes[author]
            location = node_data.get('location', 'Unknown')
            verified = node_data.get('verified', 'Unknown')
            post_count = node_data.get('post_count', 0)

            print(f"\n{author}:")
            print(f"  Location: {location}")
            print(f"  Verified: {verified}")
            print(f"  Posts: {post_count}")

    # Step 6: Show edge metadata
    print("\n" + "=" * 70)
    print("EDGE METADATA (Author-Hashtag Relationships)")
    print("=" * 70)

    print("\nExample edges with metadata:")
    edge_count = 0
    for source, target, data in graph.edges(data=True):
        if edge_count >= 5:  # Show first 5 edges
            break

        timestamp = data.get('timestamp', 'N/A')
        likes = data.get('likes', 'N/A')
        retweets = data.get('retweets', 'N/A')
        sentiment = data.get('sentiment', 'N/A')
        weight = data.get('weight', 1)

        print(f"\n{source} → {target}")
        print(f"  Weight: {weight} (times mentioned)")
        print(f"  Timestamp: {timestamp}")
        print(f"  Likes: {likes}")
        print(f"  Retweets: {retweets}")
        print(f"  Sentiment: {sentiment}")

        edge_count += 1

    # Step 7: Analyze metadata patterns
    print("\n" + "=" * 70)
    print("METADATA ANALYSIS")
    print("=" * 70)

    # Count by location
    locations = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'author':
            loc = data.get('location', 'Unknown')
            locations[loc] = locations.get(loc, 0) + 1

    print("\nAuthors by location:")
    for loc, count in sorted(locations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {loc:20s}: {count} authors")

    # Count verified vs unverified
    verified_count = sum(1 for _, data in graph.nodes(data=True)
                        if data.get('node_type') == 'author' and data.get('verified') == 'True')
    unverified_count = sum(1 for _, data in graph.nodes(data=True)
                          if data.get('node_type') == 'author' and data.get('verified') != 'True')

    print(f"\nVerification status:")
    print(f"  Verified: {verified_count}")
    print(f"  Unverified: {unverified_count}")

    # Step 8: Export network (metadata is preserved!)
    print("\n" + "=" * 70)
    print("EXPORTING NETWORK")
    print("=" * 70)

    output_dir = "./output/metadata_network"
    files = pipeline.export_network(
        output_dir=output_dir,
        base_name="metadata_network",
        formats=['gexf', 'graphml', 'json', 'statistics']
    )

    print("\nExported files (with metadata preserved):")
    for fmt, path in files.items():
        print(f"  {fmt:15s}: {path}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
Metadata enrichment enables:

1. TEMPORAL ANALYSIS
   - Track when specific hashtags were used
   - Analyze trends over time
   - Find peak activity periods

2. ENGAGEMENT ANALYSIS
   - Identify high-engagement hashtags
   - Correlate sentiment with engagement
   - Find influential posts

3. GEOGRAPHIC ANALYSIS
   - Map hashtag usage by location
   - Identify regional trends
   - Analyze geographic spread

4. INFLUENCE ANALYSIS
   - Weight verified accounts differently
   - Track verified vs. unverified patterns
   - Identify thought leaders

All this metadata is preserved in exported formats and can be used
for filtering, coloring, and sizing nodes/edges in visualization tools
like Gephi!
    """)

    print("=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

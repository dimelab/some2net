"""
Example: Creating a Keyword Network using RAKE

This example demonstrates how to create a network based on keyword extraction
using RAKE (Rapid Automatic Keyword Extraction) from social media posts.
The network shows which authors are associated with which keywords,
with edge weights representing keyword relevance scores.

Note: This method uses two-pass processing - first collecting all texts per author,
then extracting keywords for each author's combined content.
"""

from src.core.pipeline import SocialNetworkPipeline
import tempfile
import csv
from pathlib import Path


def create_sample_data():
    """Create sample social media data with varied content."""
    temp_dir = tempfile.mkdtemp()
    filepath = Path(temp_dir) / "social_media_keywords.csv"

    data = [
        ['author', 'text', 'timestamp', 'platform'],
        ['@alice', 'Machine learning algorithms are transforming how we analyze data and make predictions', '2024-01-01', 'twitter'],
        ['@alice', 'Deep learning neural networks are particularly good at image recognition tasks', '2024-01-02', 'twitter'],
        ['@alice', 'The future of artificial intelligence depends on better training data and computing power', '2024-01-03', 'twitter'],
        ['@bob', 'Web development has evolved significantly with modern JavaScript frameworks like React and Vue', '2024-01-04', 'mastodon'],
        ['@bob', 'Building responsive websites requires understanding CSS grid and flexbox layouts', '2024-01-05', 'mastodon'],
        ['@bob', 'Frontend development tools make it easier to create interactive user interfaces', '2024-01-06', 'mastodon'],
        ['@charlie', 'Climate change is the most pressing environmental challenge of our generation', '2024-01-07', 'twitter'],
        ['@charlie', 'Renewable energy sources like solar and wind power are becoming more cost effective', '2024-01-08', 'twitter'],
        ['@charlie', 'Sustainable development requires balancing economic growth with environmental protection', '2024-01-09', 'twitter'],
        ['@dave', 'Quantum computing could revolutionize cryptography and drug discovery in the coming decades', '2024-01-10', 'mastodon'],
        ['@dave', 'Understanding quantum mechanics is essential for developing new computing paradigms', '2024-01-11', 'mastodon'],
        ['@alice', 'Natural language processing enables computers to understand and generate human text', '2024-01-12', 'twitter'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


def main():
    """Run keyword network extraction example."""

    print("=" * 70)
    print("KEYWORD NETWORK EXTRACTION EXAMPLE")
    print("=" * 70)

    # Step 1: Create sample data
    print("\n1. Creating sample data...")
    filepath = create_sample_data()
    print(f"   Created: {filepath}")

    # Step 2: Initialize pipeline with keyword extraction
    print("\n2. Initializing pipeline with keyword extractor...")
    print("\n   Available methods:")
    print("   - 'rake': RAKE algorithm (extracts multi-word phrases)")
    print("   - 'tfidf': Standard TF-IDF (extracts single words)")

    # You can choose between 'rake' and 'tfidf' methods
    method = 'rake'  # Change to 'tfidf' for TF-IDF method

    print(f"\n   Using method: {method}")

    if method == 'rake':
        extractor_config = {
            'method': 'rake',
            'min_keywords': 3,          # Extract at least 3 keywords per author
            'max_keywords': 10,         # Extract at most 10 keywords per author
            'language': 'english',      # Use English stopwords
            'max_phrase_length': 3,     # Up to 3-word phrases (RAKE only)
            'min_phrase_length': 1,     # Single words allowed (RAKE only)
            'use_tfidf': True,          # Apply TF-IDF weighting to RAKE scores
        }
    else:  # tfidf
        extractor_config = {
            'method': 'tfidf',
            'min_keywords': 3,          # Extract at least 3 keywords per author
            'max_keywords': 10,         # Extract at most 10 keywords per author
            'language': 'english',      # Use English stopwords
        }

    pipeline = SocialNetworkPipeline(
        extraction_method="keyword",
        extractor_config=extractor_config
    )
    print("   Pipeline initialized!")

    # Step 3: Process file with metadata
    print("\n3. Processing social media data (two-pass for keywords)...")
    print("   This may take longer as it collects all texts per author first...")
    graph, stats = pipeline.process_file(
        filepath,
        author_column='author',
        text_column='text',
        node_metadata_columns=['platform'],   # Attach platform to author nodes
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
    print(f"  - Keywords: {stats['total_nodes'] - stats['authors']}")
    print(f"\nTotal Edges: {stats['total_edges']}")
    print(f"Network Density: {stats['density']:.4f}")
    print(f"Average Degree: {stats['average_degree']:.2f}")

    print(f"\nPosts Processed: {stats['processing_metadata']['total_posts']}")
    print(f"Keywords Extracted: {stats['processing_metadata']['entities_extracted']}")

    # Step 5: Show top keywords
    print("\n" + "=" * 70)
    print("TOP KEYWORDS BY RELEVANCE")
    print("=" * 70)

    # Collect all keywords with their total weights
    keyword_weights = {}
    for node in graph.nodes():
        # Keywords are nodes that are not authors (don't start with @)
        if not node.startswith('@'):
            # Sum up weights from all edges pointing to this keyword
            total_weight = sum(
                graph.edges[source, node].get('weight', 0)
                for source in graph.predecessors(node)
            )
            keyword_weights[node] = total_weight

    # Sort and display
    sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
    for i, (keyword, weight) in enumerate(sorted_keywords[:15], 1):
        print(f"{i:2d}. {keyword:30s} - Total weight: {weight:.3f}")

    # Step 6: Show author-keyword relationships
    print("\n" + "=" * 70)
    print("AUTHOR-KEYWORD PROFILES")
    print("=" * 70)

    for author in ['@alice', '@bob', '@charlie', '@dave']:
        if author in graph:
            print(f"\n{author}:")

            # Get keywords for this author with their scores
            keywords_with_scores = [
                (target, graph.edges[author, target].get('score', 0))
                for _, target in graph.out_edges(author)
            ]

            # Sort by score (relevance)
            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)

            # Display top keywords
            for keyword, score in keywords_with_scores[:5]:
                print(f"  - {keyword:30s} (relevance: {score:.3f})")

    # Step 7: Find authors with similar keyword profiles
    print("\n" + "=" * 70)
    print("SHARED INTERESTS (Common Keywords)")
    print("=" * 70)

    authors = [n for n in graph.nodes() if n.startswith('@')]

    for i, author1 in enumerate(authors):
        keywords1 = set(target for _, target in graph.out_edges(author1))

        for author2 in authors[i+1:]:
            keywords2 = set(target for _, target in graph.out_edges(author2))

            # Find common keywords
            common = keywords1 & keywords2

            if common:
                print(f"\n{author1} ↔ {author2}:")
                for keyword in sorted(common):
                    score1 = graph.edges[author1, keyword].get('score', 0)
                    score2 = graph.edges[author2, keyword].get('score', 0)
                    avg_score = (score1 + score2) / 2
                    print(f"  - {keyword:30s} (avg relevance: {avg_score:.3f})")

    # Step 8: Export network
    print("\n" + "=" * 70)
    print("EXPORTING NETWORK")
    print("=" * 70)

    output_dir = "./output/keyword_network"
    files = pipeline.export_network(
        output_dir=output_dir,
        base_name="keyword_network",
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
    print("  2. Use node colors to represent different author communities")
    print("  3. Filter keywords by relevance score (edge weight)")
    print("  4. Find authors with similar interests based on shared keywords")
    print("  5. Analyze how keyword usage changes over time (using timestamps)")
    print("\nVisualization tips:")
    print("  - Authors are typically central nodes with many keyword connections")
    print("  - Keyword nodes with many incoming edges are popular topics")
    print("  - Edge weights represent keyword relevance (higher = more important)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

"""
Example script demonstrating the complete pipeline functionality.

This script shows how to:
1. Use the simple convenience function
2. Use the full SocialNetworkPipeline class
3. Process CSV and NDJSON files
4. Handle progress callbacks
5. Export to multiple formats
6. Access and analyze results

Run after installing dependencies:
    pip install -r requirements.txt
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import SocialNetworkPipeline, process_social_media_data


def example_simple_processing():
    """Example 1: Simple one-line processing."""
    print("=" * 70)
    print("Example 1: Simple One-Line Processing")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        print("   Skipping this example")
        return

    print("\nNOTE: This will download NER model on first run (~1GB)\n")

    output_dir = str(Path(__file__).parent / "output" / "simple_pipeline")

    print(f"Processing: {example_file.name}")
    print(f"Output directory: {output_dir}\n")

    try:
        # One-line processing
        graph, stats, files = process_social_media_data(
            str(example_file),
            author_column='author',
            text_column='text',
            output_dir=output_dir
        )

        print("\n✓ Pipeline Complete!\n")
        print("Network Statistics:")
        print(f"  Total nodes:          {stats['total_nodes']}")
        print(f"  Total edges:          {stats['total_edges']}")
        print(f"  Authors:              {stats['authors']}")
        print(f"  Persons:              {stats['persons']}")
        print(f"  Locations:            {stats['locations']}")
        print(f"  Organizations:        {stats['organizations']}")
        print(f"  Network density:      {stats['density']:.4f}")
        print()

        print("Processing Summary:")
        metadata = stats['processing_metadata']
        print(f"  Posts processed:      {metadata['total_posts']}")
        print(f"  Chunks processed:     {metadata['total_chunks']}")
        print(f"  Entities extracted:   {metadata['entities_extracted']}")
        print(f"  Errors:               {len(metadata['errors'])}")
        print()

        print("Exported Files:")
        for fmt, filepath in files.items():
            file_size = Path(filepath).stat().st_size
            print(f"  {fmt:15s}: {Path(filepath).name:40s} ({file_size:,} bytes)")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")


def example_advanced_pipeline():
    """Example 2: Advanced pipeline usage with full control."""
    print("=" * 70)
    print("Example 2: Advanced Pipeline with Full Control")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        return

    print("\nUsing SocialNetworkPipeline class for fine-grained control\n")

    try:
        # Create pipeline with custom settings
        pipeline = SocialNetworkPipeline(
            model_name="Davlan/xlm-roberta-base-ner-hrl",
            confidence_threshold=0.90,  # Higher threshold
            enable_cache=True,
            use_entity_resolver=True,
            create_author_edges=True
        )

        print("Pipeline Configuration:")
        print(f"  Confidence threshold: 0.90")
        print(f"  Entity resolution:    Enabled")
        print(f"  Author edges:         Enabled")
        print(f"  NER caching:          Enabled")
        print()

        # Process file
        print("Processing file...\n")
        graph, stats = pipeline.process_file(
            str(example_file),
            author_column='author',
            text_column='text',
            chunksize=5,
            batch_size=16,
            show_progress=True
        )

        print("\n✓ Processing complete!\n")

        # Analyze results
        print("Network Analysis:")
        print(f"  Total nodes:          {len(graph.nodes)}")
        print(f"  Total edges:          {len(graph.edges)}")

        # Get author information
        authors = [
            (node, attrs)
            for node, attrs in graph.nodes(data=True)
            if attrs.get('node_type') == 'author'
        ]

        print(f"\nAuthors ({len(authors)}):")
        for author, attrs in sorted(authors, key=lambda x: x[1]['post_count'], reverse=True)[:5]:
            print(f"  {author:15s}: {attrs['post_count']} posts, {graph.out_degree(author)} connections")

        # Get top entities
        top_entities = stats['top_entities'][:5]
        print(f"\nTop Entities:")
        for i, entity in enumerate(top_entities, 1):
            print(f"  {i}. {entity['entity']:20s} ({entity['type']:15s}) - {entity['mentions']} mentions")

        # Export to specific formats
        output_dir = str(Path(__file__).parent / "output" / "advanced_pipeline")
        print(f"\nExporting to: {output_dir}")

        files = pipeline.export_network(
            output_dir,
            base_name="advanced_network",
            formats=['gexf', 'json', 'statistics']
        )

        print(f"✓ Exported {len(files)} formats")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_with_progress_callback():
    """Example 3: Pipeline with progress tracking."""
    print("=" * 70)
    print("Example 3: Pipeline with Progress Tracking")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        return

    print("\nDemonstrating progress callback functionality\n")

    try:
        pipeline = SocialNetworkPipeline()

        # Define progress callback
        def progress_callback(current, total, status):
            print(f"  Progress: {current} posts processed | {status}")

        # Process with callback
        graph, stats = pipeline.process_file(
            str(example_file),
            author_column='author',
            text_column='text',
            chunksize=3,
            show_progress=False,
            progress_callback=progress_callback
        )

        print(f"\n✓ Processed {stats['processing_metadata']['total_posts']} posts")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_batch_processing():
    """Example 4: Process multiple files."""
    print("=" * 70)
    print("Example 4: Batch Processing Multiple Files")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        return

    print("\nProcessing multiple files with the same pipeline\n")

    try:
        pipeline = SocialNetworkPipeline(enable_cache=True)

        files_to_process = [
            (example_file, 'author', 'text', 'Dataset 1')
        ]

        # Check if Danish sample exists
        danish_file = Path(__file__).parent / 'sample_danish.csv'
        if danish_file.exists():
            files_to_process.append(
                (danish_file, 'author', 'text', 'Dataset 2 (Danish)')
            )

        for i, (filepath, author_col, text_col, name) in enumerate(files_to_process, 1):
            print(f"Processing {name}:")
            print(f"  File: {Path(filepath).name}")

            # Reset for each file
            if i > 1:
                pipeline.reset()

            graph, stats = pipeline.process_file(
                str(filepath),
                author_column=author_col,
                text_column=text_col,
                show_progress=False
            )

            print(f"  ✓ Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
            print()

    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_ndjson_processing():
    """Example 5: Process NDJSON format."""
    print("=" * 70)
    print("Example 5: Processing NDJSON Format")
    print("=" * 70)

    # Create a sample NDJSON file
    import json
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    ndjson_file = output_dir / "sample.ndjson"

    sample_data = [
        {'post_id': '1', 'username': '@alice', 'tweet': 'Meeting with John Smith at Microsoft.'},
        {'post_id': '2', 'username': '@bob', 'tweet': 'Visiting Copenhagen next week.'},
        {'post_id': '3', 'username': '@charlie', 'tweet': 'Google and Apple are innovating.'}
    ]

    print(f"\nCreating sample NDJSON file: {ndjson_file.name}")

    with open(ndjson_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Created with {len(sample_data)} posts\n")

    try:
        print("Processing NDJSON file...\n")

        graph, stats, files = process_social_media_data(
            str(ndjson_file),
            author_column='username',
            text_column='tweet',
            file_format='ndjson',
            output_dir=str(output_dir / "ndjson_output")
        )

        print("✓ NDJSON processing complete!\n")
        print(f"Network: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        print(f"Exported to: {len(files)} formats")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_custom_export():
    """Example 6: Custom export configuration."""
    print("=" * 70)
    print("Example 6: Custom Export Configuration")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        return

    print("\nExporting only specific formats\n")

    try:
        # Process with specific export formats
        output_dir = str(Path(__file__).parent / "output" / "custom_export")

        graph, stats, files = process_social_media_data(
            str(example_file),
            author_column='author',
            text_column='text',
            output_dir=output_dir,
            export_formats=['gexf', 'json']  # Only GEXF and JSON
        )

        print("✓ Processing complete!\n")
        print("Exported formats:")
        for fmt, filepath in files.items():
            print(f"  ✓ {fmt}: {Path(filepath).name}")

        print(f"\nNote: Only exported {len(files)} formats (as requested)")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_error_handling():
    """Example 7: Error handling and recovery."""
    print("=" * 70)
    print("Example 7: Error Handling and Recovery")
    print("=" * 70)

    print("\nDemonstrating graceful error handling\n")

    try:
        pipeline = SocialNetworkPipeline()

        # Try to process non-existent file
        print("1. Attempting to process non-existent file:")
        try:
            pipeline.process_file(
                "nonexistent.csv",
                author_column='author',
                text_column='text'
            )
        except FileNotFoundError as e:
            print(f"   ✓ Caught error: {type(e).__name__}")

        # Try to export before processing
        print("\n2. Attempting to export before processing:")
        try:
            pipeline.export_network("./output")
        except RuntimeError as e:
            print(f"   ✓ Caught error: {type(e).__name__}")

        print("\n✓ Error handling working correctly")
        print()

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def example_analyze_results():
    """Example 8: Analyze pipeline results."""
    print("=" * 70)
    print("Example 8: Analyzing Pipeline Results")
    print("=" * 70)

    example_file = Path(__file__).parent / 'sample_data.csv'

    if not example_file.exists():
        print(f"\n⚠️  {example_file} not found")
        return

    print("\nDetailed analysis of pipeline results\n")

    try:
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            str(example_file),
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Network-level analysis
        print("Network Metrics:")
        print(f"  Density:              {stats['density']:.4f}")
        print(f"  Average degree:       {(stats['avg_in_degree'] + stats['avg_out_degree']) / 2:.2f}")
        print(f"  Connected components: {stats['connected_components']}")
        print(f"  Largest component:    {stats['largest_component_size']} nodes")

        # Node type distribution
        print("\nNode Type Distribution:")
        print(f"  Authors:              {stats['authors']:3d} ({stats['authors']/stats['total_nodes']*100:.1f}%)")
        print(f"  Persons:              {stats['persons']:3d} ({stats['persons']/stats['total_nodes']*100:.1f}%)")
        print(f"  Locations:            {stats['locations']:3d} ({stats['locations']/stats['total_nodes']*100:.1f}%)")
        print(f"  Organizations:        {stats['organizations']:3d} ({stats['organizations']/stats['total_nodes']*100:.1f}%)")

        # Edge type distribution
        print("\nEdge Type Distribution:")
        total_edges = stats['total_edges']
        if total_edges > 0:
            print(f"  Person mentions:      {stats['person_mentions']:3d} ({stats['person_mentions']/total_edges*100:.1f}%)")
            print(f"  Location mentions:    {stats['location_mentions']:3d} ({stats['location_mentions']/total_edges*100:.1f}%)")
            print(f"  Org mentions:         {stats['organization_mentions']:3d} ({stats['organization_mentions']/total_edges*100:.1f}%)")
            print(f"  Author mentions:      {stats['author_mentions']:3d} ({stats['author_mentions']/total_edges*100:.1f}%)")

        # Processing efficiency
        print("\nProcessing Efficiency:")
        metadata = stats['processing_metadata']
        print(f"  Posts processed:      {metadata['total_posts']}")
        print(f"  Entities extracted:   {metadata['entities_extracted']}")
        print(f"  Avg entities/post:    {metadata['entities_extracted']/metadata['total_posts']:.2f}")
        print(f"  Chunks processed:     {metadata['total_chunks']}")

        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Run all pipeline examples."""
    print("\n" + "=" * 70)
    print("Social Network Pipeline Examples")
    print("=" * 70 + "\n")

    try:
        # Run examples
        example_simple_processing()
        example_advanced_pipeline()
        example_with_progress_callback()
        example_batch_processing()
        example_ndjson_processing()
        example_custom_export()
        example_error_handling()
        example_analyze_results()

        print("=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print(f"\nOutput files location: {Path(__file__).parent / 'output'}")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

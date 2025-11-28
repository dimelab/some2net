"""
Command-Line Interface for Social Network Analytics

A powerful CLI tool for batch processing social media data and
constructing networks from named entities.

Usage:
    sna-cli input.csv --author username --text tweet_text
    sna-cli data.ndjson --author user --text content --output ./results
    sna-cli input.csv -a author -t text --model Davlan/xlm-roberta-base-ner-hrl
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import time

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import process_social_media_data


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""

    parser = argparse.ArgumentParser(
        prog='sna-cli',
        description='Social Network Analytics - Extract networks from social media data',
        epilog='Example: sna-cli tweets.csv --author username --text text --output ./results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV or NDJSON file'
    )

    parser.add_argument(
        '--author', '-a',
        type=str,
        required=True,
        dest='author_column',
        help='Name of the author/username column'
    )

    parser.add_argument(
        '--text', '-t',
        type=str,
        required=True,
        dest='text_column',
        help='Name of the text/content column'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        dest='output_dir',
        help='Output directory for results (default: ./output)'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['csv', 'ndjson', 'auto'],
        default='auto',
        dest='file_format',
        help='Input file format (default: auto-detect from extension)'
    )

    # NER model options
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Davlan/xlm-roberta-base-ner-hrl',
        dest='model_name',
        help='HuggingFace NER model name (default: Davlan/xlm-roberta-base-ner-hrl)'
    )

    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.85,
        dest='confidence_threshold',
        help='Minimum confidence threshold for entities (default: 0.85)'
    )

    # Processing options
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        dest='batch_size',
        help='Batch size for NER processing (default: 32)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10000,
        dest='chunksize',
        help='Number of rows to process per chunk (default: 10000)'
    )

    # Feature toggles
    parser.add_argument(
        '--no-cache',
        action='store_false',
        dest='enable_cache',
        help='Disable NER result caching'
    )

    parser.add_argument(
        '--no-entity-resolver',
        action='store_false',
        dest='use_entity_resolver',
        help='Disable entity deduplication'
    )

    parser.add_argument(
        '--no-author-edges',
        action='store_false',
        dest='create_author_edges',
        help='Disable author-to-author edge creation'
    )

    parser.add_argument(
        '--no-language-detection',
        action='store_false',
        dest='detect_languages',
        help='Disable language detection'
    )

    # Export options
    parser.add_argument(
        '--export-formats',
        type=str,
        nargs='+',
        choices=['gexf', 'graphml', 'json', 'edgelist', 'statistics', 'all'],
        default=['all'],
        help='Export formats (default: all)'
    )

    # Display options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    parser.add_argument(
        '--progress',
        action='store_true',
        default=True,
        help='Show progress bar (default: enabled)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_false',
        dest='progress',
        help='Hide progress bar'
    )

    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    return parser


def validate_args(args) -> bool:
    """Validate command-line arguments."""

    # Check input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input_file}", file=sys.stderr)
        return False

    # Check confidence threshold range
    if not 0.0 <= args.confidence_threshold <= 1.0:
        print(f"❌ Error: Confidence threshold must be between 0.0 and 1.0", file=sys.stderr)
        return False

    # Check batch size
    if args.batch_size < 1:
        print(f"❌ Error: Batch size must be positive", file=sys.stderr)
        return False

    # Check chunk size
    if args.chunksize < 1:
        print(f"❌ Error: Chunk size must be positive", file=sys.stderr)
        return False

    return True


def print_banner():
    """Print CLI banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          Social Network Analytics - CLI Tool                 ║
║  Extract social networks from social media data using NER    ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config(args):
    """Print configuration summary."""
    print("📋 Configuration:")
    print(f"  Input file:           {args.input_file}")
    print(f"  Author column:        {args.author_column}")
    print(f"  Text column:          {args.text_column}")
    print(f"  Output directory:     {args.output_dir}")
    print(f"  File format:          {args.file_format}")
    print()
    print("🤖 Model Settings:")
    print(f"  Model:                {args.model_name}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Chunk size:           {args.chunksize}")
    print()
    print("⚙️  Features:")
    print(f"  NER caching:          {'✓' if args.enable_cache else '✗'}")
    print(f"  Entity deduplication: {'✓' if args.use_entity_resolver else '✗'}")
    print(f"  Author edges:         {'✓' if args.create_author_edges else '✗'}")
    print(f"  Language detection:   {'✓' if args.detect_languages else '✗'}")
    print()


def create_progress_callback(verbose: bool = False):
    """Create progress callback for pipeline."""

    start_time = time.time()
    last_update = [0]  # Use list to modify in closure

    def progress_callback(current: int, total: int, status: str):
        if verbose:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            print(f"  [{current:,} posts | {rate:.1f} posts/sec] {status}")
        else:
            # Simple progress dots
            if current - last_update[0] >= 1000:
                print(".", end="", flush=True)
                last_update[0] = current

    return progress_callback


def main():
    """Main CLI entry point."""

    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    # Quiet mode
    if args.quiet:
        sys.stdout = open('/dev/null', 'w')

    # Print banner
    if not args.quiet:
        print_banner()

    # Validate arguments
    if not validate_args(args):
        sys.exit(1)

    # Print configuration
    if not args.quiet:
        print_config(args)

    # Determine file format
    file_format = args.file_format
    if file_format == 'auto':
        file_format = None  # Let pipeline auto-detect

    # Prepare export formats
    export_formats = None if 'all' in args.export_formats else args.export_formats

    try:
        # Start processing
        print("🚀 Starting processing...")
        print()

        start_time = time.time()

        # Create progress callback
        progress_cb = create_progress_callback(verbose=args.verbose) if args.progress else None

        # Process data using pipeline
        graph, stats, files = process_social_media_data(
            filepath=args.input_file,
            author_column=args.author_column,
            text_column=args.text_column,
            output_dir=args.output_dir,
            file_format=file_format,
            model_name=args.model_name,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            chunksize=args.chunksize,
            export_formats=export_formats,
            progress_callback=progress_cb
        )

        elapsed_time = time.time() - start_time

        # Print completion
        if not args.quiet:
            print()
            print()
            print("=" * 70)
            print("✅ Processing Complete!")
            print("=" * 70)
            print()

            # Statistics
            metadata = stats['processing_metadata']
            print("📊 Results:")
            print(f"  Posts processed:      {metadata['total_posts']:,}")
            print(f"  Entities extracted:   {metadata['entities_extracted']:,}")
            print(f"  Network nodes:        {stats['total_nodes']:,}")
            print(f"  Network edges:        {stats['total_edges']:,}")
            print(f"  Network density:      {stats['density']:.4f}")
            print()

            # Breakdown
            print("📈 Entity Breakdown:")
            print(f"  Authors:              {stats['authors']:,}")
            print(f"  Persons:              {stats['persons']:,}")
            print(f"  Locations:            {stats['locations']:,}")
            print(f"  Organizations:        {stats['organizations']:,}")
            print()

            # Performance
            print("⚡ Performance:")
            print(f"  Time elapsed:         {elapsed_time:.1f}s")
            print(f"  Processing speed:     {metadata['total_posts'] / elapsed_time:.1f} posts/second")
            print(f"  Chunks processed:     {metadata['total_chunks']}")
            print()

            # Errors (if any)
            if metadata['errors']:
                print(f"⚠️  Warnings: {len(metadata['errors'])} errors encountered")
                if args.verbose:
                    print("  First 5 errors:")
                    for i, error in enumerate(metadata['errors'][:5], 1):
                        print(f"    {i}. {error}")
                print()

            # Exported files
            print(f"📁 Exported {len(files)} files to: {args.output_dir}")
            for format_name, filepath in sorted(files.items()):
                file_size = Path(filepath).stat().st_size
                size_mb = file_size / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_size / 1024:.1f} KB"
                print(f"  ✓ {format_name:12s}: {Path(filepath).name:30s} ({size_str})")
            print()

            # Top entities
            if stats.get('top_entities'):
                print("🏆 Top 10 Mentioned Entities:")
                for i, entity in enumerate(stats['top_entities'][:10], 1):
                    type_emoji = {
                        'person': '👤',
                        'location': '📍',
                        'organization': '🏢'
                    }.get(entity['type'], '🏷️')
                    print(f"  {i:2d}. {type_emoji} {entity['entity']:30s} - {entity['mentions']:,} mentions")
                print()

            print("=" * 70)
            print("✨ Analysis complete! Open the GEXF file in Gephi for visualization.")
            print("=" * 70)

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}", file=sys.stderr)

        if args.verbose:
            import traceback
            print("\nDetailed error:", file=sys.stderr)
            traceback.print_exc()

        sys.exit(1)


if __name__ == '__main__':
    main()

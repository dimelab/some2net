"""
Pipeline Module

End-to-end pipeline integrating all components:
- DataLoader: Load CSV/NDJSON files
- NEREngine: Extract named entities
- EntityResolver: Deduplicate entities
- NetworkBuilder: Construct network graph
- Exporters: Export to multiple formats

Provides a simple interface for processing social media data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import networkx as nx

from .data_loader import DataLoader
from .ner_engine import NEREngine
from .network_builder import NetworkBuilder

try:
    from ..utils.exporters import export_all_formats
except ImportError:
    # Fallback for when running as script
    from utils.exporters import export_all_formats

logger = logging.getLogger(__name__)


class SocialNetworkPipeline:
    """
    End-to-end pipeline for social network construction from social media data.

    Pipeline Steps:
    1. Load data from file (CSV/NDJSON)
    2. Extract named entities using NER
    3. Build network graph with entity resolution
    4. Calculate network statistics
    5. Export to multiple formats
    """

    def __init__(
        self,
        model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        enable_cache: bool = True,
        use_entity_resolver: bool = True,
        create_author_edges: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            model_name: HuggingFace model for NER
            device: Device for NER ('cuda', 'cpu', or None for auto-detect)
            confidence_threshold: Minimum confidence for entity extraction
            enable_cache: Whether to cache NER results
            use_entity_resolver: Whether to deduplicate entities
            create_author_edges: Whether to create author-to-author edges
        """
        logger.info("Initializing Social Network Pipeline")

        # Initialize components
        self.data_loader = DataLoader()
        self.ner_engine = NEREngine(
            model_name=model_name,
            device=device,
            confidence_threshold=confidence_threshold,
            enable_cache=enable_cache
        )
        self.network_builder = NetworkBuilder(
            use_entity_resolver=use_entity_resolver,
            create_author_edges=create_author_edges
        )

        # Pipeline state
        self.graph: Optional[nx.DiGraph] = None
        self.statistics: Optional[Dict] = None
        self.processing_metadata: Dict = {
            'total_posts': 0,
            'total_chunks': 0,
            'entities_extracted': 0,
            'errors': []
        }

        logger.info("Pipeline initialized successfully")

    def process_file(
        self,
        filepath: str,
        author_column: str,
        text_column: str,
        file_format: Optional[str] = None,
        chunksize: int = 10000,
        batch_size: int = 32,
        detect_languages: bool = True,
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Process a social media data file end-to-end.

        Args:
            filepath: Path to input file (CSV or NDJSON)
            author_column: Name of author column
            text_column: Name of text column
            file_format: File format ('csv' or 'ndjson', auto-detected if None)
            chunksize: Number of rows to process per chunk
            batch_size: Batch size for NER processing
            detect_languages: Whether to detect languages
            show_progress: Whether to show progress bars
            progress_callback: Optional callback(current, total, status_msg)

        Returns:
            Tuple of (graph, statistics)
        """
        logger.info(f"Starting pipeline processing: {filepath}")

        # Reset state
        self._reset_state()

        # Validate file
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format if not provided
        if file_format is None:
            file_format = 'ndjson' if file_path.suffix.lower() in ['.ndjson', '.jsonl'] else 'csv'

        logger.info(f"Processing as {file_format.upper()} with chunks of {chunksize}")

        # Step 1: Load data in chunks
        try:
            if file_format.lower() == 'csv':
                chunks = self.data_loader.load_csv(
                    filepath,
                    author_column=author_column,
                    text_column=text_column,
                    chunksize=chunksize
                )
            elif file_format.lower() in ['ndjson', 'jsonl']:
                chunks = self.data_loader.load_ndjson(
                    filepath,
                    author_column=author_column,
                    text_column=text_column,
                    chunksize=chunksize
                )
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            self.processing_metadata['errors'].append(f"File loading error: {e}")
            raise

        # Step 2-4: Process chunks
        chunk_num = 0
        for chunk in chunks:
            chunk_num += 1
            self.processing_metadata['total_chunks'] = chunk_num

            try:
                self._process_chunk(
                    chunk,
                    author_column,
                    text_column,
                    batch_size,
                    detect_languages,
                    show_progress,
                    chunk_num
                )

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(
                        self.processing_metadata['total_posts'],
                        -1,  # Total unknown for streaming
                        f"Processed chunk {chunk_num}"
                    )

            except Exception as e:
                error_msg = f"Error processing chunk {chunk_num}: {e}"
                logger.error(error_msg)
                self.processing_metadata['errors'].append(error_msg)
                # Continue processing other chunks
                continue

        # Step 5: Finalize network
        logger.info("Finalizing network and calculating statistics")
        self.graph = self.network_builder.get_graph()
        self.statistics = self.network_builder.get_statistics()

        # Add processing metadata to statistics
        self.statistics['processing_metadata'] = self.processing_metadata

        logger.info(f"Pipeline complete: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        return self.graph, self.statistics

    def _process_chunk(
        self,
        chunk,
        author_column: str,
        text_column: str,
        batch_size: int,
        detect_languages: bool,
        show_progress: bool,
        chunk_num: int
    ):
        """Process a single chunk of data."""

        # Extract authors and texts
        authors = chunk[author_column].tolist()
        texts = chunk[text_column].tolist()

        # Get post IDs if available
        post_ids = chunk.get('post_id', chunk.get('id', [None] * len(texts))).tolist()

        # Get timestamps if available
        timestamps = chunk.get('timestamp', chunk.get('created_at', [None] * len(texts))).tolist()

        logger.debug(f"Chunk {chunk_num}: Processing {len(texts)} posts")

        # Step 2: Extract entities using NER
        try:
            entities_batch, languages = self.ner_engine.extract_entities_batch(
                texts,
                batch_size=batch_size,
                show_progress=show_progress and chunk_num == 1,  # Only show progress for first chunk
                detect_languages=detect_languages
            )

            self.processing_metadata['entities_extracted'] += sum(len(ent) for ent in entities_batch)

        except Exception as e:
            error_msg = f"NER extraction error in chunk {chunk_num}: {e}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            raise

        # Step 3: Build network
        for i, (author, entities) in enumerate(zip(authors, entities_batch)):
            try:
                post_id = post_ids[i] if i < len(post_ids) else None
                timestamp = timestamps[i] if i < len(timestamps) else None

                self.network_builder.add_post(
                    author=author,
                    entities=entities,
                    post_id=str(post_id) if post_id is not None else None,
                    timestamp=str(timestamp) if timestamp is not None else None
                )

                self.processing_metadata['total_posts'] += 1

            except Exception as e:
                error_msg = f"Error adding post to network: {e}"
                logger.warning(error_msg)
                self.processing_metadata['errors'].append(error_msg)
                # Continue with other posts
                continue

    def export_network(
        self,
        output_dir: str,
        base_name: str = "network",
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export the network to various formats.

        Args:
            output_dir: Output directory path
            base_name: Base name for output files
            formats: List of formats to export (default: all)
                     Options: 'gexf', 'graphml', 'json', 'edgelist', 'statistics'

        Returns:
            Dictionary mapping format to filepath
        """
        if self.graph is None or self.statistics is None:
            raise RuntimeError("No network to export. Run process_file() first.")

        logger.info(f"Exporting network to {output_dir}")

        # Export all formats or specified formats
        if formats is None:
            # Export all
            files = export_all_formats(
                self.graph,
                self.statistics,
                output_dir,
                base_name
            )
        else:
            # Export specific formats
            try:
                from ..utils.exporters import (
                    export_gexf, export_graphml, export_json,
                    export_edgelist, export_statistics
                )
            except ImportError:
                from utils.exporters import (
                    export_gexf, export_graphml, export_json,
                    export_edgelist, export_statistics
                )

            files = {}
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            format_map = {
                'gexf': (export_gexf, f"{base_name}.gexf"),
                'graphml': (export_graphml, f"{base_name}.graphml"),
                'json': (export_json, f"{base_name}.json"),
                'edgelist': (export_edgelist, f"{base_name}_edgelist.csv"),
                'statistics': (export_statistics, f"{base_name}_statistics.json")
            }

            for fmt in formats:
                if fmt in format_map:
                    export_func, filename = format_map[fmt]
                    filepath = str(output_path / filename)

                    if fmt == 'statistics':
                        export_func(self.statistics, filepath)
                    else:
                        export_func(self.graph, filepath)

                    files[fmt] = filepath

        logger.info(f"Exported {len(files)} formats")
        return files

    def get_graph(self) -> Optional[nx.DiGraph]:
        """Get the constructed network graph."""
        return self.graph

    def get_statistics(self) -> Optional[Dict]:
        """Get network statistics."""
        return self.statistics

    def get_processing_metadata(self) -> Dict:
        """Get processing metadata (posts processed, errors, etc.)."""
        return self.processing_metadata

    def reset(self):
        """Reset the pipeline state."""
        logger.info("Resetting pipeline state")
        self._reset_state()
        self.network_builder.reset()
        self.ner_engine.clear_cache()

    def _reset_state(self):
        """Reset internal state."""
        self.graph = None
        self.statistics = None
        self.processing_metadata = {
            'total_posts': 0,
            'total_chunks': 0,
            'entities_extracted': 0,
            'errors': []
        }


# Convenience function for simple use cases
def process_social_media_data(
    filepath: str,
    author_column: str,
    text_column: str,
    output_dir: str = "./output",
    file_format: Optional[str] = None,
    model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
    batch_size: int = 32,
    confidence_threshold: float = 0.85,
    chunksize: int = 10000,
    export_formats: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[nx.DiGraph, Dict, Dict[str, str]]:
    """
    Process social media data file end-to-end with one function call.

    Args:
        filepath: Path to input file (CSV or NDJSON)
        author_column: Name of author column
        text_column: Name of text column
        output_dir: Directory for output files
        file_format: File format ('csv' or 'ndjson', auto-detected if None)
        model_name: HuggingFace model for NER
        batch_size: Batch size for NER processing
        confidence_threshold: Minimum confidence for entities
        chunksize: Number of rows per chunk
        export_formats: List of export formats (default: all)
        progress_callback: Optional callback(current, total, status_msg)

    Returns:
        Tuple of (graph, statistics, exported_files)

    Example:
        >>> graph, stats, files = process_social_media_data(
        ...     'data.csv',
        ...     author_column='username',
        ...     text_column='tweet_text',
        ...     output_dir='./output'
        ... )
        >>> print(f"Created network with {len(graph.nodes)} nodes")
    """
    # Create pipeline
    pipeline = SocialNetworkPipeline(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        enable_cache=True,
        use_entity_resolver=True,
        create_author_edges=True
    )

    # Process file
    graph, statistics = pipeline.process_file(
        filepath=filepath,
        author_column=author_column,
        text_column=text_column,
        file_format=file_format,
        chunksize=chunksize,
        batch_size=batch_size,
        detect_languages=True,
        show_progress=True,
        progress_callback=progress_callback
    )

    # Export network
    exported_files = pipeline.export_network(
        output_dir=output_dir,
        base_name="network",
        formats=export_formats
    )

    return graph, statistics, exported_files


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python pipeline.py <file> <author_column> <text_column> [output_dir]")
        print("\nExample:")
        print("  python pipeline.py data.csv username text ./output")
        sys.exit(1)

    filepath = sys.argv[1]
    author_col = sys.argv[2]
    text_col = sys.argv[3]
    output = sys.argv[4] if len(sys.argv) > 4 else "./output"

    print(f"\nProcessing: {filepath}")
    print(f"Author column: {author_col}")
    print(f"Text column: {text_col}")
    print(f"Output directory: {output}\n")

    # Process
    graph, stats, files = process_social_media_data(
        filepath,
        author_col,
        text_col,
        output_dir=output
    )

    # Print results
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Authors: {stats['authors']}")
    print(f"  Entities: {stats['persons'] + stats['locations'] + stats['organizations']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"\nPosts Processed: {stats['processing_metadata']['total_posts']}")
    print(f"Entities Extracted: {stats['processing_metadata']['entities_extracted']}")

    print(f"\nExported Files:")
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")

    print("\n" + "=" * 70)

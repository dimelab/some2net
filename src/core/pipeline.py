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
from .entity_linker import EntityLinker

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
        extraction_method: str = "ner",
        extractor_config: Optional[Dict] = None,
        model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
        device: Optional[str] = None,
        confidence_threshold: float = 0.85,
        enable_cache: bool = True,
        use_entity_resolver: bool = True,
        create_author_edges: bool = True,
        enable_entity_linking: bool = False,
        entity_linking_config: Optional[Dict] = None
    ):
        """
        Initialize the pipeline.

        Args:
            extraction_method: Method for extraction ('ner', 'hashtag', 'mention', 'domain', 'keyword', 'exact')
            extractor_config: Configuration dict for the chosen extractor
            model_name: HuggingFace model for NER (used if extraction_method='ner')
            device: Device for NER ('cuda', 'cpu', or None for auto-detect)
            confidence_threshold: Minimum confidence for entity extraction (NER only)
            enable_cache: Whether to cache NER results (NER only)
            use_entity_resolver: Whether to deduplicate entities
            create_author_edges: Whether to create author-to-author edges
            enable_entity_linking: Whether to enable entity linking (Phase 2, NER only)
            entity_linking_config: Configuration for entity linker
                {
                    'confidence_threshold': 0.7,
                    'enable_cache': True,
                    'top_k': 5
                }
        """
        logger.info("Initializing Social Network Pipeline")

        # Store extraction method
        self.extraction_method = extraction_method
        self.extractor_config = extractor_config or {}

        # Initialize components
        self.data_loader = DataLoader()

        # Create the appropriate extractor
        self.extractor = self._create_extractor(
            extraction_method,
            extractor_config,
            model_name,
            device,
            confidence_threshold,
            enable_cache
        )

        # Keep NER engine for backward compatibility (will be removed eventually)
        if extraction_method == "ner":
            # NERExtractor wraps NEREngine, so we need to access the wrapped engine
            self.ner_engine = self.extractor.ner_engine
        else:
            self.ner_engine = None

        self.network_builder = NetworkBuilder(
            use_entity_resolver=use_entity_resolver,
            create_author_edges=create_author_edges
        )

        # Initialize entity linker if enabled (only for NER)
        self.enable_entity_linking = enable_entity_linking and extraction_method == "ner"
        self.entity_linker: Optional[EntityLinker] = None

        if self.enable_entity_linking:
            logger.info("Entity linking enabled - initializing EntityLinker")
            # Default config
            default_config = {
                'confidence_threshold': 0.7,
                'enable_cache': True,
                'device': device,
                'top_k': 5
            }
            # Merge with user config
            if entity_linking_config:
                default_config.update(entity_linking_config)

            self.entity_linker = EntityLinker(**default_config)
            logger.info("EntityLinker initialized successfully")

        # Pipeline state
        self.graph: Optional[nx.DiGraph] = None
        self.statistics: Optional[Dict] = None
        self.processing_metadata: Dict = {
            'total_posts': 0,
            'total_chunks': 0,
            'entities_extracted': 0,
            'entities_linked': 0,
            'errors': []
        }

        logger.info(f"Pipeline initialized successfully with extraction method: {extraction_method}")

    def _create_extractor(
        self,
        method: str,
        config: Optional[Dict],
        model_name: str,
        device: Optional[str],
        confidence_threshold: float,
        enable_cache: bool
    ):
        """
        Factory method to create appropriate extractor.

        Args:
            method: Extraction method name
            config: Extractor-specific configuration
            model_name: Model name for NER
            device: Device for NER
            confidence_threshold: Confidence threshold for NER
            enable_cache: Enable cache for NER

        Returns:
            BaseExtractor instance
        """
        from .extractors import (
            NERExtractor, HashtagExtractor, MentionExtractor,
            DomainExtractor, KeywordExtractor, ExactMatchExtractor
        )

        config = config or {}

        if method == "ner":
            # For NER, use the legacy NEREngine wrapped in NERExtractor
            # Pass model_name, device, etc. through config
            ner_config = {
                'model_name': model_name,
                'device': device,
                'confidence_threshold': confidence_threshold,
                'enable_cache': enable_cache
            }
            ner_config.update(config)
            return NERExtractor(**ner_config)

        elif method == "hashtag":
            return HashtagExtractor(**config)

        elif method == "mention":
            return MentionExtractor(**config)

        elif method == "domain":
            return DomainExtractor(**config)

        elif method == "keyword":
            return KeywordExtractor(**config)

        elif method == "exact":
            return ExactMatchExtractor(**config)

        else:
            raise ValueError(
                f"Unknown extraction method: {method}. "
                f"Available methods: 'ner', 'hashtag', 'mention', 'domain', 'keyword', 'exact'"
            )

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
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        node_metadata_columns: Optional[List[str]] = None,
        edge_metadata_columns: Optional[List[str]] = None
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Process a social media data file end-to-end.

        Args:
            filepath: Path to input file (CSV or NDJSON)
            author_column: Name of author column
            text_column: Name of text column
            file_format: File format ('csv' or 'ndjson', auto-detected if None)
            chunksize: Number of rows to process per chunk
            batch_size: Batch size for extraction processing
            detect_languages: Whether to detect languages (NER only)
            show_progress: Whether to show progress bars
            progress_callback: Optional callback(current, total, status_msg)
            node_metadata_columns: Optional list of column names to attach as node metadata
            edge_metadata_columns: Optional list of column names to attach as edge metadata

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

        # Route to appropriate processing method based on extraction type
        if self.extraction_method == "keyword":
            return self._process_file_keyword(
                filepath=filepath,
                author_column=author_column,
                text_column=text_column,
                file_format=file_format,
                chunksize=chunksize,
                batch_size=batch_size,
                show_progress=show_progress,
                progress_callback=progress_callback,
                node_metadata_columns=node_metadata_columns,
                edge_metadata_columns=edge_metadata_columns
            )
        else:
            return self._process_file_standard(
                filepath=filepath,
                author_column=author_column,
                text_column=text_column,
                file_format=file_format,
                chunksize=chunksize,
                batch_size=batch_size,
                detect_languages=detect_languages,
                show_progress=show_progress,
                progress_callback=progress_callback,
                node_metadata_columns=node_metadata_columns,
                edge_metadata_columns=edge_metadata_columns
            )

    def _process_file_standard(
        self,
        filepath: str,
        author_column: str,
        text_column: str,
        file_format: str,
        chunksize: int,
        batch_size: int,
        detect_languages: bool,
        show_progress: bool,
        progress_callback: Optional[Callable[[int, int, str], None]],
        node_metadata_columns: Optional[List[str]],
        edge_metadata_columns: Optional[List[str]]
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Process file using standard extraction methods (all except keyword).

        This is the standard single-pass processing approach where we extract
        entities from text as we stream through the file.
        """
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
                    chunk_num,
                    node_metadata_columns,
                    edge_metadata_columns
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

    def _process_file_keyword(
        self,
        filepath: str,
        author_column: str,
        text_column: str,
        file_format: str,
        chunksize: int,
        batch_size: int,
        show_progress: bool,
        progress_callback: Optional[Callable[[int, int, str], None]],
        node_metadata_columns: Optional[List[str]],
        edge_metadata_columns: Optional[List[str]]
    ) -> Tuple[nx.DiGraph, Dict]:
        """
        Process file using keyword extraction (two-pass approach).

        Keyword extraction requires collecting all texts per author before
        extracting keywords, so we need a two-pass approach:
        1. First pass: Collect all texts per author
        2. Extract keywords per author
        3. Build network from author-keyword pairs
        """
        from .extractors import KeywordExtractor

        logger.info("Using two-pass processing for keyword extraction")

        # Ensure extractor is KeywordExtractor
        if not isinstance(self.extractor, KeywordExtractor):
            raise RuntimeError("Extractor must be KeywordExtractor for keyword processing")

        # Step 1: Load data in chunks and collect texts per author
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

        # First pass: Collect texts per author
        logger.info("First pass: Collecting texts per author")
        chunk_num = 0
        author_metadata_map = {}  # Store metadata for each author

        for chunk in chunks:
            chunk_num += 1
            self.processing_metadata['total_chunks'] = chunk_num

            try:
                # Extract authors and texts
                author_data = chunk[author_column]
                text_data = chunk[text_column]

                # Convert to list if it's a pandas Series
                if hasattr(author_data, 'tolist'):
                    authors = author_data.tolist()
                else:
                    authors = list(author_data) if not isinstance(author_data, list) else author_data

                if hasattr(text_data, 'tolist'):
                    texts = text_data.tolist()
                else:
                    texts = list(text_data) if not isinstance(text_data, list) else text_data

                # Collect texts for each author
                for author, text in zip(authors, texts):
                    author = str(author).strip()
                    if author and text:
                        self.extractor.collect_texts(author, [str(text)])
                        self.processing_metadata['total_posts'] += 1

                        # Collect node metadata for this author (if not already collected)
                        if node_metadata_columns and author not in author_metadata_map:
                            metadata = {}
                            for col in node_metadata_columns:
                                if col in chunk.columns:
                                    # Get first value for this author
                                    idx = authors.index(author) if author in authors else None
                                    if idx is not None and idx < len(chunk):
                                        val = chunk[col].iloc[idx] if hasattr(chunk[col], 'iloc') else chunk[col][idx]
                                        metadata[col] = val
                            author_metadata_map[author] = metadata

                if progress_callback:
                    progress_callback(
                        self.processing_metadata['total_posts'],
                        -1,
                        f"Collecting texts - chunk {chunk_num}"
                    )

            except Exception as e:
                error_msg = f"Error collecting texts in chunk {chunk_num}: {e}"
                logger.error(error_msg)
                self.processing_metadata['errors'].append(error_msg)
                continue

        # Second pass: Extract keywords for all authors
        logger.info("Second pass: Extracting keywords per author")
        try:
            author_keywords = self.extractor.extract_all_authors()
            self.processing_metadata['entities_extracted'] = sum(len(kws) for kws in author_keywords.values())

            # Build network from author-keyword pairs
            for author, keywords in author_keywords.items():
                # Get metadata for this author
                node_metadata = author_metadata_map.get(author)

                # Add to network (edge_metadata not applicable for keyword extraction)
                self.network_builder.add_post(
                    author=author,
                    entities=keywords,
                    node_metadata=node_metadata,
                    edge_metadata=None
                )

        except Exception as e:
            error_msg = f"Error extracting keywords: {e}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            raise

        # Step 3: Finalize network
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
        chunk_num: int,
        node_metadata_columns: Optional[List[str]] = None,
        edge_metadata_columns: Optional[List[str]] = None
    ):
        """Process a single chunk of data."""

        # Debug: Log chunk info
        logger.info(f"Chunk {chunk_num}: {len(chunk)} rows, {len(chunk.columns)} columns")
        logger.info(f"Looking for author column: '{author_column}'")
        logger.info(f"Looking for text column: '{text_column}'")
        logger.info(f"Available columns: {list(chunk.columns)[:10]}...")  # First 10 columns

        # Check if chunk is empty
        if len(chunk) == 0:
            logger.warning(f"Chunk {chunk_num} is empty, skipping")
            return

        # Check if columns exist
        if author_column not in chunk.columns:
            raise KeyError(f"Author column '{author_column}' not found in data. Available columns: {list(chunk.columns)[:20]}")
        if text_column not in chunk.columns:
            raise KeyError(f"Text column '{text_column}' not found in data. Available columns: {list(chunk.columns)[:20]}")

        # Extract authors and texts - handle both Series and list types
        author_data = chunk[author_column]
        text_data = chunk[text_column]

        # Convert to list if it's a pandas Series
        if hasattr(author_data, 'tolist'):
            authors = author_data.tolist()
        else:
            authors = list(author_data) if not isinstance(author_data, list) else author_data

        if hasattr(text_data, 'tolist'):
            texts = text_data.tolist()
        else:
            texts = list(text_data) if not isinstance(text_data, list) else text_data

        logger.info(f"Chunk {chunk_num}: Extracted {len(authors)} authors, {len(texts)} texts")
        logger.info(f"Chunk {chunk_num}: Author data type: {type(author_data)}, Text data type: {type(text_data)}")

        # Get post IDs if available
        post_id_data = chunk.get('post_id', chunk.get('id', [None] * len(texts)))
        if hasattr(post_id_data, 'tolist'):
            post_ids = post_id_data.tolist()
        else:
            post_ids = list(post_id_data) if not isinstance(post_id_data, list) else post_id_data

        # Get timestamps if available
        timestamp_data = chunk.get('timestamp', chunk.get('created_at', [None] * len(texts)))
        if hasattr(timestamp_data, 'tolist'):
            timestamps = timestamp_data.tolist()
        else:
            timestamps = list(timestamp_data) if not isinstance(timestamp_data, list) else timestamp_data

        logger.debug(f"Chunk {chunk_num}: Processing {len(texts)} posts")

        # Step 2: Extract entities using the configured extractor
        try:
            # Use extractor interface - check if it's NER for special handling
            if self.extraction_method == "ner":
                # NER has special language detection support
                entities_batch, languages = self.ner_engine.extract_entities_batch(
                    texts,
                    batch_size=batch_size,
                    show_progress=show_progress and chunk_num == 1,  # Only show progress for first chunk
                    detect_languages=detect_languages
                )
            else:
                # Use generic extractor interface
                entities_batch = self.extractor.extract_batch(
                    texts,
                    batch_size=batch_size,
                    show_progress=show_progress and chunk_num == 1
                )
                languages = [None] * len(texts)  # No language detection for non-NER

            self.processing_metadata['entities_extracted'] += sum(len(ent) for ent in entities_batch)

        except Exception as e:
            error_msg = f"Extraction error in chunk {chunk_num}: {e}"
            logger.error(error_msg)
            self.processing_metadata['errors'].append(error_msg)
            raise

        # Step 2.5: Link entities to Wikipedia/Wikidata if enabled (NER only)
        if self.enable_entity_linking and self.entity_linker:
            try:
                logger.debug(f"Chunk {chunk_num}: Linking entities to Wikipedia/Wikidata")

                # Flatten entities for batch processing
                all_entities = []
                entity_indices = []  # Track which post each entity belongs to

                for post_idx, entities in enumerate(entities_batch):
                    for entity in entities:
                        # Add language info if available
                        if detect_languages and post_idx < len(languages):
                            entity['language'] = languages[post_idx]
                        all_entities.append(entity)
                        entity_indices.append(post_idx)

                # Link entities in batch
                if all_entities:
                    linked_entities = self.entity_linker.link_entities_batch(
                        all_entities,
                        batch_size=batch_size,
                        show_progress=False  # Don't show linking progress for now
                    )

                    # Count successfully linked entities
                    linked_count = sum(1 for e in linked_entities if e.get('is_linked', False))
                    self.processing_metadata['entities_linked'] += linked_count

                    # Reconstruct entities_batch with linked information
                    linked_entities_batch = [[] for _ in range(len(entities_batch))]
                    for entity, post_idx in zip(linked_entities, entity_indices):
                        linked_entities_batch[post_idx].append(entity)

                    entities_batch = linked_entities_batch

                    logger.debug(f"Chunk {chunk_num}: Linked {linked_count}/{len(all_entities)} entities")

            except Exception as e:
                error_msg = f"Entity linking error in chunk {chunk_num}: {e}"
                logger.warning(error_msg)
                self.processing_metadata['errors'].append(error_msg)
                # Continue with unlinked entities

        # Step 3: Extract metadata if specified
        node_metadata_list = []
        edge_metadata_list = []

        for i in range(len(authors)):
            # Extract node metadata for this post
            node_metadata = {}
            if node_metadata_columns:
                for col in node_metadata_columns:
                    if col in chunk.columns:
                        if hasattr(chunk[col], 'iloc'):
                            val = chunk[col].iloc[i]
                        else:
                            val = chunk[col][i] if i < len(chunk[col]) else None
                        node_metadata[col] = val

            # Extract edge metadata for this post
            edge_metadata = {}
            if edge_metadata_columns:
                for col in edge_metadata_columns:
                    if col in chunk.columns:
                        if hasattr(chunk[col], 'iloc'):
                            val = chunk[col].iloc[i]
                        else:
                            val = chunk[col][i] if i < len(chunk[col]) else None
                        edge_metadata[col] = val

            node_metadata_list.append(node_metadata if node_metadata else None)
            edge_metadata_list.append(edge_metadata if edge_metadata else None)

        # Step 4: Build network
        for i, (author, entities) in enumerate(zip(authors, entities_batch)):
            try:
                post_id = post_ids[i] if i < len(post_ids) else None
                timestamp = timestamps[i] if i < len(timestamps) else None
                node_metadata = node_metadata_list[i] if i < len(node_metadata_list) else None
                edge_metadata = edge_metadata_list[i] if i < len(edge_metadata_list) else None

                self.network_builder.add_post(
                    author=author,
                    entities=entities,
                    post_id=str(post_id) if post_id is not None else None,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    node_metadata=node_metadata,
                    edge_metadata=edge_metadata
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
        if self.ner_engine:
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

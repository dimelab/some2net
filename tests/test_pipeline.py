"""
Unit tests for the pipeline module.

Tests the end-to-end integration of all components:
- DataLoader
- NEREngine
- EntityResolver
- NetworkBuilder
- Exporters
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import csv
import json
import networkx as nx

from src.core.pipeline import SocialNetworkPipeline, process_social_media_data


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    filepath = Path(temp_dir) / "test_data.csv"

    data = [
        ['post_id', 'author', 'text', 'timestamp'],
        ['1', '@user1', 'I met John Smith at Microsoft yesterday.', '2024-01-01'],
        ['2', '@user2', 'Copenhagen is a beautiful city.', '2024-01-02'],
        ['3', '@user1', 'Microsoft and Google are competitors.', '2024-01-03'],
        ['4', '@user3', 'John Smith works at Microsoft.', '2024-01-04'],
        ['5', '@user2', 'I visited Copenhagen and Paris.', '2024-01-05']
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


@pytest.fixture
def sample_ndjson_file(temp_dir):
    """Create a sample NDJSON file for testing."""
    filepath = Path(temp_dir) / "test_data.ndjson"

    data = [
        {'post_id': '1', 'author': '@user1', 'text': 'I met John Smith at Microsoft.'},
        {'post_id': '2', 'author': '@user2', 'text': 'Copenhagen is beautiful.'},
        {'post_id': '3', 'author': '@user1', 'text': 'Microsoft and Google compete.'}
    ]

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    return str(filepath)


class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_pipeline_init_default(self):
        """Test pipeline initialization with defaults."""
        pipeline = SocialNetworkPipeline()

        assert pipeline.data_loader is not None
        assert pipeline.ner_engine is not None
        assert pipeline.network_builder is not None
        assert pipeline.graph is None
        assert pipeline.statistics is None

    def test_pipeline_init_custom_model(self):
        """Test pipeline initialization with custom model."""
        pipeline = SocialNetworkPipeline(
            model_name="Davlan/xlm-roberta-base-ner-hrl",
            confidence_threshold=0.9
        )

        assert pipeline.ner_engine is not None

    def test_pipeline_init_entity_resolver_disabled(self):
        """Test pipeline with entity resolver disabled."""
        pipeline = SocialNetworkPipeline(use_entity_resolver=False)

        assert pipeline.network_builder.entity_resolver is None

    def test_pipeline_init_author_edges_disabled(self):
        """Test pipeline with author edges disabled."""
        pipeline = SocialNetworkPipeline(create_author_edges=False)

        assert pipeline.network_builder.create_author_edges is False


class TestPipelineProcessing:
    """Test pipeline processing functionality."""

    def test_process_csv_file_basic(self, sample_csv_file):
        """Test processing a basic CSV file."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Verify graph created
        assert graph is not None
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes) > 0
        assert len(graph.edges) >= 0

        # Verify statistics
        assert stats is not None
        assert 'total_nodes' in stats
        assert 'total_edges' in stats
        assert 'processing_metadata' in stats

    def test_process_ndjson_file(self, sample_ndjson_file):
        """Test processing an NDJSON file."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_ndjson_file,
            author_column='author',
            text_column='text',
            file_format='ndjson',
            show_progress=False
        )

        assert graph is not None
        assert len(graph.nodes) > 0

    def test_process_file_auto_detect_format(self, sample_csv_file):
        """Test auto-detection of file format."""
        pipeline = SocialNetworkPipeline()

        # Don't specify file_format
        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        assert graph is not None

    def test_process_file_with_chunking(self, sample_csv_file):
        """Test processing with small chunk size."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            chunksize=2,  # Small chunks
            show_progress=False
        )

        assert graph is not None
        assert stats['processing_metadata']['total_chunks'] >= 2

    def test_process_file_with_progress_callback(self, sample_csv_file):
        """Test processing with progress callback."""
        pipeline = SocialNetworkPipeline()

        progress_calls = []

        def callback(current, total, status):
            progress_calls.append((current, total, status))

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False,
            progress_callback=callback
        )

        # Callback should have been called
        assert len(progress_calls) > 0


class TestPipelineStatistics:
    """Test pipeline statistics generation."""

    def test_statistics_structure(self, sample_csv_file):
        """Test that statistics have expected structure."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check required fields
        required_fields = [
            'total_nodes', 'total_edges', 'density',
            'authors', 'persons', 'locations', 'organizations',
            'posts_processed', 'processing_metadata'
        ]

        for field in required_fields:
            assert field in stats

    def test_processing_metadata(self, sample_csv_file):
        """Test processing metadata is collected."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        metadata = stats['processing_metadata']
        assert 'total_posts' in metadata
        assert 'total_chunks' in metadata
        assert 'entities_extracted' in metadata
        assert 'errors' in metadata

        # Should have processed some posts
        assert metadata['total_posts'] > 0


class TestPipelineExport:
    """Test pipeline export functionality."""

    def test_export_network_all_formats(self, sample_csv_file, temp_dir):
        """Test exporting network in all formats."""
        pipeline = SocialNetworkPipeline()

        # Process file
        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Export
        output_dir = str(Path(temp_dir) / "output")
        files = pipeline.export_network(output_dir, base_name="test_network")

        # Check files created
        assert len(files) > 0
        assert 'gexf' in files
        assert 'graphml' in files
        assert 'json' in files

        # Verify files exist
        for filepath in files.values():
            assert Path(filepath).exists()

    def test_export_network_specific_formats(self, sample_csv_file, temp_dir):
        """Test exporting specific formats only."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        output_dir = str(Path(temp_dir) / "output")
        files = pipeline.export_network(
            output_dir,
            base_name="test",
            formats=['gexf', 'json']
        )

        # Only requested formats
        assert len(files) == 2
        assert 'gexf' in files
        assert 'json' in files
        assert 'graphml' not in files

    def test_export_before_processing_raises_error(self, temp_dir):
        """Test that exporting before processing raises error."""
        pipeline = SocialNetworkPipeline()

        with pytest.raises(RuntimeError, match="No network to export"):
            pipeline.export_network(temp_dir)


class TestPipelineGetters:
    """Test pipeline getter methods."""

    def test_get_graph(self, sample_csv_file):
        """Test get_graph method."""
        pipeline = SocialNetworkPipeline()

        # Before processing
        assert pipeline.get_graph() is None

        # After processing
        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        retrieved_graph = pipeline.get_graph()
        assert retrieved_graph is not None
        assert retrieved_graph is graph

    def test_get_statistics(self, sample_csv_file):
        """Test get_statistics method."""
        pipeline = SocialNetworkPipeline()

        # Before processing
        assert pipeline.get_statistics() is None

        # After processing
        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        retrieved_stats = pipeline.get_statistics()
        assert retrieved_stats is not None
        assert retrieved_stats is stats

    def test_get_processing_metadata(self, sample_csv_file):
        """Test get_processing_metadata method."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        metadata = pipeline.get_processing_metadata()
        assert 'total_posts' in metadata
        assert metadata['total_posts'] > 0


class TestPipelineReset:
    """Test pipeline reset functionality."""

    def test_reset_clears_state(self, sample_csv_file):
        """Test that reset clears pipeline state."""
        pipeline = SocialNetworkPipeline()

        # Process file
        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        assert pipeline.graph is not None
        assert pipeline.statistics is not None

        # Reset
        pipeline.reset()

        # State should be cleared
        assert pipeline.graph is None
        assert pipeline.statistics is None
        assert pipeline.processing_metadata['total_posts'] == 0


class TestConvenienceFunction:
    """Test the process_social_media_data convenience function."""

    def test_process_function_basic(self, sample_csv_file, temp_dir):
        """Test basic usage of convenience function."""
        output_dir = str(Path(temp_dir) / "output")

        graph, stats, files = process_social_media_data(
            sample_csv_file,
            author_column='author',
            text_column='text',
            output_dir=output_dir
        )

        # Check outputs
        assert graph is not None
        assert stats is not None
        assert len(files) > 0

        # Verify files exported
        for filepath in files.values():
            assert Path(filepath).exists()

    def test_process_function_custom_params(self, sample_csv_file, temp_dir):
        """Test convenience function with custom parameters."""
        output_dir = str(Path(temp_dir) / "output")

        graph, stats, files = process_social_media_data(
            sample_csv_file,
            author_column='author',
            text_column='text',
            output_dir=output_dir,
            batch_size=16,
            chunksize=2,
            confidence_threshold=0.9,
            export_formats=['gexf', 'json']
        )

        assert graph is not None
        assert len(files) == 2
        assert 'gexf' in files
        assert 'json' in files


class TestErrorHandling:
    """Test pipeline error handling."""

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        pipeline = SocialNetworkPipeline()

        with pytest.raises(FileNotFoundError):
            pipeline.process_file(
                "nonexistent.csv",
                author_column='author',
                text_column='text',
                show_progress=False
            )

    def test_invalid_file_format(self, sample_csv_file):
        """Test handling of invalid file format."""
        pipeline = SocialNetworkPipeline()

        with pytest.raises(ValueError, match="Unsupported file format"):
            pipeline.process_file(
                sample_csv_file,
                author_column='author',
                text_column='text',
                file_format='xml',  # Unsupported
                show_progress=False
            )

    def test_missing_column(self, sample_csv_file):
        """Test handling of missing column."""
        pipeline = SocialNetworkPipeline()

        with pytest.raises(Exception):  # KeyError or similar
            pipeline.process_file(
                sample_csv_file,
                author_column='nonexistent_column',
                text_column='text',
                show_progress=False
            )


class TestEndToEndIntegration:
    """Test complete end-to-end pipeline integration."""

    def test_full_pipeline_csv_to_export(self, sample_csv_file, temp_dir):
        """Test complete pipeline: CSV → NER → Network → Export."""
        output_dir = str(Path(temp_dir) / "output")

        # Process
        graph, stats, files = process_social_media_data(
            sample_csv_file,
            author_column='author',
            text_column='text',
            output_dir=output_dir
        )

        # Verify network
        assert len(graph.nodes) > 0
        assert len(graph.edges) >= 0

        # Verify authors present
        authors = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'author']
        assert len(authors) > 0

        # Verify statistics
        assert stats['total_nodes'] == len(graph.nodes)
        assert stats['total_edges'] == len(graph.edges)
        assert stats['posts_processed'] > 0

        # Verify exports
        assert 'gexf' in files
        gexf_path = Path(files['gexf'])
        assert gexf_path.exists()

        # Load and verify GEXF
        loaded_graph = nx.read_gexf(str(gexf_path))
        assert len(loaded_graph.nodes) == len(graph.nodes)

    def test_pipeline_preserves_attributes(self, sample_csv_file, temp_dir):
        """Test that pipeline preserves node and edge attributes."""
        pipeline = SocialNetworkPipeline()

        graph, stats = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check author nodes have required attributes
        for node, attrs in graph.nodes(data=True):
            if attrs.get('node_type') == 'author':
                assert 'label' in attrs
                assert 'post_count' in attrs
                assert 'mention_count' in attrs

        # Check edges have required attributes
        for u, v, attrs in graph.edges(data=True):
            assert 'weight' in attrs
            assert attrs['weight'] >= 1

    def test_multiple_file_processing(self, sample_csv_file, temp_dir):
        """Test processing multiple files sequentially."""
        pipeline = SocialNetworkPipeline()

        # Process first file
        graph1, stats1 = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        nodes1 = len(graph1.nodes)

        # Reset and process again
        pipeline.reset()

        graph2, stats2 = pipeline.process_file(
            sample_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Should have same number of nodes
        assert len(graph2.nodes) == nodes1

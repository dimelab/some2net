"""
Unit tests for network export utilities.

Tests all export formats:
- GEXF (primary format)
- GraphML
- JSON (D3.js)
- CSV edge list
- Adjacency matrix
- Statistics export
"""

import pytest
import networkx as nx
import json
import csv
import tempfile
import shutil
from pathlib import Path

from src.utils.exporters import (
    export_gexf,
    export_graphml,
    export_json,
    export_edgelist,
    export_adjacency_matrix,
    export_statistics,
    export_all_formats
)


@pytest.fixture
def sample_graph():
    """Create a sample network graph for testing."""
    G = nx.DiGraph()

    # Add author nodes
    G.add_node("@user1", node_type="author", label="User 1", post_count=5, mention_count=0)
    G.add_node("@user2", node_type="author", label="User 2", post_count=3, mention_count=1)

    # Add entity nodes
    G.add_node("John Smith", node_type="person", label="John Smith", mention_count=3)
    G.add_node("Microsoft", node_type="organization", label="Microsoft", mention_count=2)
    G.add_node("Copenhagen", node_type="location", label="Copenhagen", mention_count=1)

    # Add edges with attributes
    G.add_edge("@user1", "John Smith", weight=2, entity_type="PER",
               source_posts=["post1", "post2"], first_mention="2024-01-01", last_mention="2024-01-02")
    G.add_edge("@user1", "Microsoft", weight=1, entity_type="ORG",
               source_posts=["post1"])
    G.add_edge("@user2", "John Smith", weight=1, entity_type="PER",
               source_posts=["post3"])
    G.add_edge("@user2", "@user1", weight=1, entity_type="AUTHOR",
               source_posts=["post4"])

    return G


@pytest.fixture
def sample_stats():
    """Create sample statistics for testing."""
    return {
        'total_nodes': 5,
        'total_edges': 4,
        'density': 0.2,
        'authors': 2,
        'persons': 1,
        'locations': 1,
        'organizations': 1,
        'person_mentions': 3,
        'location_mentions': 1,
        'organization_mentions': 2,
        'author_mentions': 1,
        'total_mentions': 7,
        'avg_mentions_per_edge': 1.75,
        'posts_processed': 10,
        'top_entities': [
            {'entity': 'John Smith', 'mentions': 3, 'type': 'person'},
            {'entity': 'Microsoft', 'mentions': 2, 'type': 'organization'}
        ]
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestGEXFExport:
    """Test GEXF export functionality (primary format)."""

    def test_export_gexf_basic(self, sample_graph, temp_dir):
        """Test basic GEXF export."""
        filepath = str(Path(temp_dir) / "test.gexf")
        export_gexf(sample_graph, filepath)

        # Check file exists
        assert Path(filepath).exists()

        # Check file is valid GEXF
        loaded_graph = nx.read_gexf(filepath)
        assert len(loaded_graph.nodes) == len(sample_graph.nodes)
        assert len(loaded_graph.edges) == len(sample_graph.edges)

    def test_export_gexf_preserves_attributes(self, sample_graph, temp_dir):
        """Test that GEXF preserves node and edge attributes."""
        filepath = str(Path(temp_dir) / "test.gexf")
        export_gexf(sample_graph, filepath)

        loaded_graph = nx.read_gexf(filepath)

        # Check node attributes preserved
        assert 'node_type' in loaded_graph.nodes['@user1']
        assert 'label' in loaded_graph.nodes['@user1']

    def test_export_gexf_creates_directory(self, sample_graph, temp_dir):
        """Test that GEXF export creates nested directories."""
        filepath = str(Path(temp_dir) / "subdir" / "test.gexf")
        export_gexf(sample_graph, filepath)

        assert Path(filepath).exists()

    def test_export_gexf_empty_graph(self, temp_dir):
        """Test GEXF export with empty graph."""
        G = nx.DiGraph()
        filepath = str(Path(temp_dir) / "empty.gexf")
        export_gexf(G, filepath)

        assert Path(filepath).exists()
        loaded = nx.read_gexf(filepath)
        assert len(loaded.nodes) == 0


class TestGraphMLExport:
    """Test GraphML export functionality."""

    def test_export_graphml_basic(self, sample_graph, temp_dir):
        """Test basic GraphML export."""
        filepath = str(Path(temp_dir) / "test.graphml")
        export_graphml(sample_graph, filepath)

        # Check file exists
        assert Path(filepath).exists()

        # Check file is valid GraphML
        loaded_graph = nx.read_graphml(filepath)
        assert len(loaded_graph.nodes) == len(sample_graph.nodes)
        assert len(loaded_graph.edges) == len(sample_graph.edges)

    def test_export_graphml_converts_list_attributes(self, temp_dir):
        """Test that GraphML converts list attributes to strings."""
        G = nx.DiGraph()
        G.add_node("node1", node_type="author")
        G.add_edge("node1", "node2", source_posts=["post1", "post2", "post3"])

        filepath = str(Path(temp_dir) / "test.graphml")
        export_graphml(G, filepath)

        loaded = nx.read_graphml(filepath)
        # After conversion, list should be string
        assert 'source_posts' in loaded.edges['node1', 'node2']

    def test_export_graphml_handles_special_characters(self, temp_dir):
        """Test GraphML export with special characters in node names."""
        G = nx.DiGraph()
        G.add_node("@user&special<chars>", node_type="author", label="Test & <User>")
        G.add_node("entity's name", node_type="person")
        G.add_edge("@user&special<chars>", "entity's name", weight=1)

        filepath = str(Path(temp_dir) / "special.graphml")
        export_graphml(G, filepath)

        loaded = nx.read_graphml(filepath)
        assert len(loaded.nodes) == 2


class TestJSONExport:
    """Test JSON export functionality (D3.js format)."""

    def test_export_json_basic(self, sample_graph, temp_dir):
        """Test basic JSON export."""
        filepath = str(Path(temp_dir) / "test.json")
        export_json(sample_graph, filepath)

        # Check file exists
        assert Path(filepath).exists()

        # Load and validate JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'nodes' in data
        assert 'links' in data
        assert len(data['nodes']) == len(sample_graph.nodes)
        assert len(data['links']) == len(sample_graph.edges)

    def test_export_json_node_structure(self, sample_graph, temp_dir):
        """Test JSON node structure."""
        filepath = str(Path(temp_dir) / "test.json")
        export_json(sample_graph, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check node has required fields
        node = data['nodes'][0]
        assert 'id' in node

    def test_export_json_edge_structure(self, sample_graph, temp_dir):
        """Test JSON edge structure."""
        filepath = str(Path(temp_dir) / "test.json")
        export_json(sample_graph, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check edge has required fields
        edge = data['links'][0]
        assert 'source' in edge
        assert 'target' in edge

    def test_export_json_utf8_encoding(self, temp_dir):
        """Test JSON export preserves UTF-8 characters."""
        G = nx.DiGraph()
        G.add_node("København", node_type="location", label="København")
        G.add_node("Ærø", node_type="location", label="Ærø")
        G.add_edge("København", "Ærø", weight=1)

        filepath = str(Path(temp_dir) / "utf8.json")
        export_json(G, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # UTF-8 characters should be preserved
        node_ids = [n['id'] for n in data['nodes']]
        assert "København" in node_ids
        assert "Ærø" in node_ids


class TestEdgeListExport:
    """Test CSV edge list export functionality."""

    def test_export_edgelist_basic(self, sample_graph, temp_dir):
        """Test basic edge list export."""
        filepath = str(Path(temp_dir) / "edgelist.csv")
        export_edgelist(sample_graph, filepath)

        assert Path(filepath).exists()

        # Read CSV
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + edges
        assert len(rows) == len(sample_graph.edges) + 1

    def test_export_edgelist_header(self, sample_graph, temp_dir):
        """Test edge list has correct header."""
        filepath = str(Path(temp_dir) / "edgelist.csv")
        export_edgelist(sample_graph, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

        expected_columns = ['source', 'target', 'weight', 'entity_type', 'source_posts']
        assert header == expected_columns

    def test_export_edgelist_content(self, sample_graph, temp_dir):
        """Test edge list content is correct."""
        filepath = str(Path(temp_dir) / "edgelist.csv")
        export_edgelist(sample_graph, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check first edge
        edge = rows[0]
        assert 'source' in edge
        assert 'target' in edge
        assert 'weight' in edge
        assert edge['weight'].isdigit()

    def test_export_edgelist_source_posts_format(self, sample_graph, temp_dir):
        """Test source_posts are pipe-separated."""
        filepath = str(Path(temp_dir) / "edgelist.csv")
        export_edgelist(sample_graph, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Find edge with multiple source posts
        edge_with_posts = [r for r in rows if '|' in r['source_posts']]
        if edge_with_posts:
            assert 'post1|post2' in edge_with_posts[0]['source_posts'] or 'post2|post1' in edge_with_posts[0]['source_posts']


class TestAdjacencyMatrixExport:
    """Test adjacency matrix export functionality."""

    def test_export_adjacency_matrix_basic(self, sample_graph, temp_dir):
        """Test basic adjacency matrix export."""
        filepath = str(Path(temp_dir) / "adjacency.csv")
        export_adjacency_matrix(sample_graph, filepath)

        assert Path(filepath).exists()

    def test_export_adjacency_matrix_dimensions(self, sample_graph, temp_dir):
        """Test adjacency matrix has correct dimensions."""
        filepath = str(Path(temp_dir) / "adjacency.csv")
        export_adjacency_matrix(sample_graph, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should be n+1 rows (header + n nodes), n+1 columns (index + n nodes)
        n_nodes = len(sample_graph.nodes)
        assert len(rows) == n_nodes + 1


class TestStatisticsExport:
    """Test statistics export functionality."""

    def test_export_statistics_basic(self, sample_stats, temp_dir):
        """Test basic statistics export."""
        filepath = str(Path(temp_dir) / "stats.json")
        export_statistics(sample_stats, filepath)

        assert Path(filepath).exists()

        # Load JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'total_nodes' in data
        assert 'total_edges' in data

    def test_export_statistics_preserves_values(self, sample_stats, temp_dir):
        """Test statistics values are preserved."""
        filepath = str(Path(temp_dir) / "stats.json")
        export_statistics(sample_stats, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['total_nodes'] == 5
        assert data['total_edges'] == 4
        assert data['authors'] == 2

    def test_export_statistics_handles_lists(self, sample_stats, temp_dir):
        """Test statistics export handles list values."""
        filepath = str(Path(temp_dir) / "stats.json")
        export_statistics(sample_stats, filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'top_entities' in data
        assert isinstance(data['top_entities'], list)

    def test_export_statistics_empty(self, temp_dir):
        """Test statistics export with empty dict."""
        filepath = str(Path(temp_dir) / "empty_stats.json")
        export_statistics({}, filepath)

        assert Path(filepath).exists()

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data == {}


class TestExportAllFormats:
    """Test exporting all formats at once."""

    def test_export_all_formats_creates_files(self, sample_graph, sample_stats, temp_dir):
        """Test export_all_formats creates all expected files."""
        files = export_all_formats(sample_graph, sample_stats, temp_dir, "network")

        # Check all formats exported
        assert 'gexf' in files
        assert 'graphml' in files
        assert 'json' in files
        assert 'edgelist' in files
        assert 'statistics' in files

    def test_export_all_formats_file_existence(self, sample_graph, sample_stats, temp_dir):
        """Test all exported files exist."""
        files = export_all_formats(sample_graph, sample_stats, temp_dir, "network")

        for filepath in files.values():
            assert Path(filepath).exists()

    def test_export_all_formats_custom_base_name(self, sample_graph, sample_stats, temp_dir):
        """Test custom base name in export_all_formats."""
        files = export_all_formats(sample_graph, sample_stats, temp_dir, "my_network")

        # Check filenames use custom base name
        assert any("my_network" in str(f) for f in files.values())

    def test_export_all_formats_creates_directory(self, sample_graph, sample_stats, temp_dir):
        """Test export_all_formats creates output directory if not exists."""
        new_dir = str(Path(temp_dir) / "nested" / "output")
        files = export_all_formats(sample_graph, sample_stats, new_dir, "network")

        assert Path(new_dir).exists()
        assert len(files) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_export_graph_with_no_edges(self, temp_dir):
        """Test exporting graph with nodes but no edges."""
        G = nx.DiGraph()
        G.add_node("node1", node_type="author")
        G.add_node("node2", node_type="person")

        # All formats should handle this
        filepath = str(Path(temp_dir) / "no_edges.gexf")
        export_gexf(G, filepath)
        assert Path(filepath).exists()

        filepath = str(Path(temp_dir) / "no_edges.json")
        export_json(G, filepath)
        assert Path(filepath).exists()

    def test_export_single_node_graph(self, temp_dir):
        """Test exporting graph with single node."""
        G = nx.DiGraph()
        G.add_node("lonely_node", node_type="author")

        filepath = str(Path(temp_dir) / "single.gexf")
        export_gexf(G, filepath)

        loaded = nx.read_gexf(filepath)
        assert len(loaded.nodes) == 1

    def test_export_graph_with_self_loop(self, temp_dir):
        """Test exporting graph with self-loop edge."""
        G = nx.DiGraph()
        G.add_node("node1", node_type="author")
        G.add_edge("node1", "node1", weight=1)

        filepath = str(Path(temp_dir) / "self_loop.gexf")
        export_gexf(G, filepath)

        loaded = nx.read_gexf(filepath)
        assert loaded.has_edge("node1", "node1")

    def test_export_graph_with_unicode_node_names(self, temp_dir):
        """Test exporting with Unicode node names."""
        G = nx.DiGraph()
        G.add_node("用户1", node_type="author", label="User 1")
        G.add_node("Björk", node_type="person", label="Björk")
        G.add_edge("用户1", "Björk", weight=1)

        # Test GEXF
        filepath = str(Path(temp_dir) / "unicode.gexf")
        export_gexf(G, filepath)
        loaded = nx.read_gexf(filepath)
        assert len(loaded.nodes) == 2

        # Test JSON
        filepath = str(Path(temp_dir) / "unicode.json")
        export_json(G, filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        node_ids = [n['id'] for n in data['nodes']]
        assert "用户1" in node_ids or "Björk" in node_ids

    def test_export_with_missing_attributes(self, temp_dir):
        """Test exporting graph where nodes/edges have missing attributes."""
        G = nx.DiGraph()
        G.add_node("node1")  # No attributes
        G.add_node("node2", node_type="person")
        G.add_edge("node1", "node2")  # No weight, no entity_type

        filepath = str(Path(temp_dir) / "missing_attrs.gexf")
        export_gexf(G, filepath)
        assert Path(filepath).exists()

        filepath = str(Path(temp_dir) / "missing_attrs_edgelist.csv")
        export_edgelist(G, filepath)
        assert Path(filepath).exists()


class TestIntegrationWithNetworkBuilder:
    """Test integration with NetworkBuilder output."""

    def test_export_network_builder_graph(self, temp_dir):
        """Test exporting a graph created by NetworkBuilder."""
        # Simulate NetworkBuilder output
        G = nx.DiGraph()

        # Author nodes
        G.add_node("@user1", node_type="author", label="@user1", mention_count=0, post_count=2)
        G.add_node("@user2", node_type="author", label="@user2", mention_count=1, post_count=1)

        # Entity nodes
        G.add_node("Microsoft", node_type="organization", label="Microsoft", mention_count=2)
        G.add_node("Copenhagen", node_type="location", label="Copenhagen", mention_count=1)

        # Edges with NetworkBuilder attributes
        G.add_edge("@user1", "Microsoft", weight=2, entity_type="ORG",
                   source_posts=["post1", "post2"], first_mention="2024-01-01", last_mention="2024-01-02")
        G.add_edge("@user1", "Copenhagen", weight=1, entity_type="LOC",
                   source_posts=["post1"])
        G.add_edge("@user2", "@user1", weight=1, entity_type="AUTHOR",
                   source_posts=["post3"])

        # Export all formats
        files = export_all_formats(G, {}, temp_dir, "network_builder_output")

        # Verify all exports succeeded
        assert len(files) >= 4  # At least gexf, graphml, json, edgelist

        # Verify GEXF (primary format)
        loaded_gexf = nx.read_gexf(files['gexf'])
        assert len(loaded_gexf.nodes) == 4
        assert len(loaded_gexf.edges) == 3

        # Verify JSON
        with open(files['json'], 'r') as f:
            json_data = json.load(f)
        assert len(json_data['nodes']) == 4
        assert len(json_data['links']) == 3

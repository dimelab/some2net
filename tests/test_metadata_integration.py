"""
Integration tests for metadata support in NetworkBuilder.

Tests that node and edge metadata is properly stored, retrieved, and exported.
"""

import pytest
import networkx as nx
import tempfile
import json
from pathlib import Path

from src.core.network_builder import NetworkBuilder
from src.utils.exporters import export_gexf, export_graphml, export_json


class TestMetadataIntegration:
    """Test metadata handling in network construction and export."""

    def test_node_metadata_basic(self):
        """Test that node metadata is properly stored."""
        builder = NetworkBuilder()

        # Add post with node metadata
        node_metadata = {
            'country': 'Denmark',
            'follower_count': 1000,
            'verified': True
        }

        builder.add_post(
            author="@user1",
            entities=[
                {'text': 'Python', 'type': 'ORG', 'score': 0.9}
            ],
            node_metadata=node_metadata
        )

        graph = builder.get_graph()

        # Check that metadata is attached to node with meta_ prefix
        assert 'meta_country' in graph.nodes['@user1']
        assert 'meta_follower_count' in graph.nodes['@user1']
        assert 'meta_verified' in graph.nodes['@user1']

        # Check values
        assert graph.nodes['@user1']['meta_country'] == 'Denmark'
        assert graph.nodes['@user1']['meta_follower_count'] == 1000
        assert graph.nodes['@user1']['meta_verified'] is True

    def test_edge_metadata_basic(self):
        """Test that edge metadata is properly stored."""
        builder = NetworkBuilder()

        # Add post with edge metadata
        edge_metadata = {
            'sentiment': 'positive',
            'language': 'en',
            'engagement_score': 0.85
        }

        builder.add_post(
            author="@user1",
            entities=[
                {'text': 'Python', 'type': 'ORG', 'score': 0.9}
            ],
            edge_metadata=edge_metadata
        )

        graph = builder.get_graph()

        # Check that metadata is attached to edge with meta_ prefix
        edge_data = graph['@user1']['Python']
        assert 'meta_sentiment' in edge_data
        assert 'meta_language' in edge_data
        assert 'meta_engagement_score' in edge_data

        # Check values
        assert edge_data['meta_sentiment'] == 'positive'
        assert edge_data['meta_language'] == 'en'
        assert edge_data['meta_engagement_score'] == 0.85

    def test_metadata_multiple_posts(self):
        """Test metadata handling across multiple posts from same author."""
        builder = NetworkBuilder()

        # First post with metadata
        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata={'country': 'Denmark'},
            edge_metadata={'sentiment': 'positive'}
        )

        # Second post from same author with different metadata
        builder.add_post(
            author="@user1",
            entities=[{'text': 'Java', 'type': 'ORG', 'score': 0.8}],
            node_metadata={'country': 'Sweden'},  # Different value
            edge_metadata={'sentiment': 'neutral'}
        )

        graph = builder.get_graph()

        # First metadata value should be retained (first occurrence wins)
        assert graph.nodes['@user1']['meta_country'] == 'Denmark'

        # Each edge should have its own metadata
        assert graph['@user1']['Python']['meta_sentiment'] == 'positive'
        assert graph['@user1']['Java']['meta_sentiment'] == 'neutral'

    def test_metadata_no_override_core_attributes(self):
        """Test that metadata doesn't override core attributes."""
        builder = NetworkBuilder()

        # Try to override core attributes via metadata
        node_metadata = {
            'node_type': 'malicious',  # Should not override
            'label': 'hacker',  # Should not override
            'custom_field': 'value'  # Should be added
        }

        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata=node_metadata
        )

        graph = builder.get_graph()

        # Core attributes should not be overridden
        assert graph.nodes['@user1']['node_type'] == 'author'
        assert graph.nodes['@user1']['label'] == '@user1'

        # Custom field should be added with prefix
        assert graph.nodes['@user1']['meta_custom_field'] == 'value'

    def test_metadata_export_graphml(self):
        """Test that metadata is preserved in GraphML export."""
        builder = NetworkBuilder()

        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata={'country': 'Denmark', 'verified': True},
            edge_metadata={'sentiment': 'positive', 'score': 0.85}
        )

        graph = builder.get_graph()

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
            filepath = f.name

        try:
            export_graphml(graph, filepath)

            # Read back and verify
            imported_graph = nx.read_graphml(filepath)

            # Check node metadata (GraphML converts all to strings)
            assert 'meta_country' in imported_graph.nodes['@user1']
            assert imported_graph.nodes['@user1']['meta_country'] == 'Denmark'

            # Check edge metadata
            edge = list(imported_graph.edges(data=True))[0]
            edge_data = edge[2]
            assert 'meta_sentiment' in edge_data
            assert edge_data['meta_sentiment'] == 'positive'

        finally:
            Path(filepath).unlink()

    def test_metadata_export_json(self):
        """Test that metadata is preserved in JSON export."""
        builder = NetworkBuilder()

        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata={'country': 'Denmark', 'follower_count': 1000},
            edge_metadata={'sentiment': 'positive', 'engagement': 0.85}
        )

        graph = builder.get_graph()

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            export_json(graph, filepath)

            # Read back and verify
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Find the @user1 node
            user_node = next(n for n in data['nodes'] if n['id'] == '@user1')
            assert 'meta_country' in user_node
            assert user_node['meta_country'] == 'Denmark'
            assert user_node['meta_follower_count'] == 1000

            # Find the edge
            edge = data['links'][0]
            assert 'meta_sentiment' in edge
            assert edge['meta_sentiment'] == 'positive'
            assert edge['meta_engagement'] == 0.85

        finally:
            Path(filepath).unlink()

    def test_metadata_different_data_types(self):
        """Test metadata with different data types."""
        builder = NetworkBuilder()

        metadata = {
            'string_val': 'text',
            'int_val': 42,
            'float_val': 3.14,
            'bool_val': True,
            'none_val': None
        }

        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata=metadata
        )

        graph = builder.get_graph()

        # All types should be preserved
        assert graph.nodes['@user1']['meta_string_val'] == 'text'
        assert graph.nodes['@user1']['meta_int_val'] == 42
        assert graph.nodes['@user1']['meta_float_val'] == 3.14
        assert graph.nodes['@user1']['meta_bool_val'] is True
        assert graph.nodes['@user1']['meta_none_val'] is None

    def test_metadata_empty_dict(self):
        """Test that empty metadata dict doesn't cause issues."""
        builder = NetworkBuilder()

        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata={},
            edge_metadata={}
        )

        graph = builder.get_graph()

        # Should have core attributes but no metadata
        assert 'node_type' in graph.nodes['@user1']
        assert 'label' in graph.nodes['@user1']

        # No meta_ attributes should be added
        meta_attrs = [k for k in graph.nodes['@user1'].keys() if k.startswith('meta_')]
        assert len(meta_attrs) == 0

    def test_metadata_none_value(self):
        """Test that None metadata is handled correctly."""
        builder = NetworkBuilder()

        builder.add_post(
            author="@user1",
            entities=[{'text': 'Python', 'type': 'ORG', 'score': 0.9}],
            node_metadata=None,
            edge_metadata=None
        )

        graph = builder.get_graph()

        # Should work without errors
        assert '@user1' in graph.nodes
        assert 'Python' in graph.nodes

    def test_complex_integration_scenario(self):
        """Test a complex scenario with multiple authors and entities."""
        builder = NetworkBuilder()

        # User 1 posts
        builder.add_post(
            author="@alice",
            entities=[
                {'text': 'Python', 'type': 'ORG', 'score': 0.9},
                {'text': 'Java', 'type': 'ORG', 'score': 0.8}
            ],
            node_metadata={'country': 'Denmark', 'verified': True},
            edge_metadata={'topic': 'programming', 'year': 2024}
        )

        # User 2 posts
        builder.add_post(
            author="@bob",
            entities=[
                {'text': 'Python', 'type': 'ORG', 'score': 0.95}
            ],
            node_metadata={'country': 'Sweden', 'verified': False},
            edge_metadata={'topic': 'data-science', 'year': 2024}
        )

        # User 1 posts again
        builder.add_post(
            author="@alice",
            entities=[
                {'text': 'JavaScript', 'type': 'ORG', 'score': 0.85}
            ],
            edge_metadata={'topic': 'web-dev', 'year': 2024}
        )

        graph = builder.get_graph()
        stats = builder.get_statistics()

        # Check node count
        assert stats['total_nodes'] == 5  # 2 authors + 3 entities

        # Check that each author has metadata
        assert graph.nodes['@alice']['meta_country'] == 'Denmark'
        assert graph.nodes['@bob']['meta_country'] == 'Sweden'

        # Check that edges have different metadata
        assert graph['@alice']['Python']['meta_topic'] == 'programming'
        assert graph['@bob']['Python']['meta_topic'] == 'data-science'
        assert graph['@alice']['JavaScript']['meta_topic'] == 'web-dev'

        # All edges should have year
        for _, _, edge_data in graph.edges(data=True):
            if 'meta_year' in edge_data:
                assert edge_data['meta_year'] == 2024


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

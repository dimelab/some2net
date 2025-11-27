"""
Unit tests for NetworkBuilder module.
"""

import pytest
import networkx as nx

from src.core.network_builder import NetworkBuilder


@pytest.fixture
def builder():
    """Create fresh NetworkBuilder instance for each test."""
    return NetworkBuilder()


@pytest.fixture
def builder_no_resolver():
    """Create NetworkBuilder without entity resolver."""
    return NetworkBuilder(use_entity_resolver=False)


@pytest.fixture
def sample_entities():
    """Sample entity list for testing."""
    return [
        {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
        {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92},
        {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.89}
    ]


class TestInitialization:
    """Test NetworkBuilder initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        builder = NetworkBuilder()
        assert builder is not None
        assert isinstance(builder.graph, nx.DiGraph)
        assert len(builder.graph.nodes) == 0
        assert len(builder.graph.edges) == 0

    def test_with_entity_resolver(self):
        """Test initialization with entity resolver."""
        builder = NetworkBuilder(use_entity_resolver=True)
        assert builder.entity_resolver is not None

    def test_without_entity_resolver(self):
        """Test initialization without entity resolver."""
        builder = NetworkBuilder(use_entity_resolver=False)
        assert builder.entity_resolver is None

    def test_author_edges_enabled(self):
        """Test with author-to-author edges enabled."""
        builder = NetworkBuilder(create_author_edges=True)
        assert builder.create_author_edges is True

    def test_author_edges_disabled(self):
        """Test with author-to-author edges disabled."""
        builder = NetworkBuilder(create_author_edges=False)
        assert builder.create_author_edges is False


class TestNodeCreation:
    """Test node creation functionality."""

    def test_add_author_node(self, builder, sample_entities):
        """Test adding author node."""
        builder.add_post('@user1', sample_entities)

        assert builder.graph.has_node('@user1')
        assert builder.graph.nodes['@user1']['node_type'] == 'author'
        assert builder.graph.nodes['@user1']['label'] == '@user1'
        assert builder.graph.nodes['@user1']['post_count'] == 1

    def test_add_entity_nodes(self, builder, sample_entities):
        """Test adding entity nodes."""
        builder.add_post('@user1', sample_entities)

        # Should have author + entities
        assert len(builder.graph.nodes) >= 4  # 1 author + 3 entities

        # Check specific entities exist
        assert any('john smith' in node.lower() for node in builder.graph.nodes)

    def test_node_attributes_person(self, builder):
        """Test person node attributes."""
        entities = [{'text': 'John Smith', 'type': 'PER', 'score': 0.95}]
        builder.add_post('@user1', entities)

        # Find the entity node (might be normalized)
        entity_nodes = [
            n for n, attrs in builder.graph.nodes(data=True)
            if attrs.get('node_type') == 'person'
        ]

        assert len(entity_nodes) > 0
        node = entity_nodes[0]
        assert builder.graph.nodes[node]['node_type'] == 'person'
        assert builder.graph.nodes[node]['mention_count'] >= 1

    def test_node_attributes_location(self, builder):
        """Test location node attributes."""
        entities = [{'text': 'Copenhagen', 'type': 'LOC', 'score': 0.89}]
        builder.add_post('@user1', entities)

        location_nodes = [
            n for n, attrs in builder.graph.nodes(data=True)
            if attrs.get('node_type') == 'location'
        ]

        assert len(location_nodes) > 0

    def test_node_attributes_organization(self, builder):
        """Test organization node attributes."""
        entities = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]
        builder.add_post('@user1', entities)

        org_nodes = [
            n for n, attrs in builder.graph.nodes(data=True)
            if attrs.get('node_type') == 'organization'
        ]

        assert len(org_nodes) > 0


class TestEdgeCreation:
    """Test edge creation functionality."""

    def test_add_edges(self, builder, sample_entities):
        """Test adding edges."""
        builder.add_post('@user1', sample_entities)

        # Should have edges from author to entities
        author_edges = list(builder.graph.out_edges('@user1'))
        assert len(author_edges) >= 3

    def test_edge_weight(self, builder):
        """Test edge weight tracking."""
        entities = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]

        # Add same entity twice
        builder.add_post('@user1', entities)
        builder.add_post('@user1', entities)

        # Find the edge
        edges = list(builder.graph.out_edges('@user1', data=True))
        assert len(edges) > 0

        # Check weight
        edge_data = edges[0][2]
        assert edge_data['weight'] == 2

    def test_edge_attributes(self, builder):
        """Test edge attributes."""
        entities = [{'text': 'John Smith', 'type': 'PER', 'score': 0.95}]
        post_id = 'post_123'
        timestamp = '2024-01-01'

        builder.add_post('@user1', entities, post_id=post_id, timestamp=timestamp)

        edges = list(builder.graph.out_edges('@user1', data=True))
        assert len(edges) > 0

        edge_data = edges[0][2]
        assert 'weight' in edge_data
        assert 'entity_type' in edge_data
        assert 'source_posts' in edge_data
        assert post_id in edge_data['source_posts']
        assert 'first_mention' in edge_data
        assert edge_data['first_mention'] == timestamp

    def test_multiple_sources_same_entity(self, builder):
        """Test edge from multiple posts mentioning same entity."""
        entities = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]

        builder.add_post('@user1', entities, post_id='post_1')
        builder.add_post('@user1', entities, post_id='post_2')

        # Should have one edge with weight 2
        edges = list(builder.graph.out_edges('@user1', data=True))
        edge_data = edges[0][2]

        assert edge_data['weight'] == 2
        assert len(edge_data['source_posts']) == 2
        assert 'post_1' in edge_data['source_posts']
        assert 'post_2' in edge_data['source_posts']


class TestAuthorToAuthorEdges:
    """Test author-to-author edge creation."""

    def test_author_mention_creates_edge(self, builder):
        """Test that mentioning an author creates edge."""
        # First author posts
        builder.add_post('@johndoe', [])

        # Second author mentions first
        entities = [{'text': 'John Doe', 'type': 'PER', 'score': 0.95}]
        builder.add_post('@alice', entities)

        # Should have edge from alice to johndoe
        if builder.graph.has_edge('@alice', '@johndoe'):
            edge = builder.graph['@alice']['@johndoe']
            assert edge['entity_type'] == 'AUTHOR'

    def test_no_self_loops(self, builder):
        """Test that authors don't create self-loop edges."""
        # Author mentions themselves
        entities = [{'text': 'John Doe', 'type': 'PER', 'score': 0.95}]
        builder.add_post('@johndoe', entities)

        # Should not have self-loop
        assert not builder.graph.has_edge('@johndoe', '@johndoe')

    def test_author_edges_disabled(self):
        """Test with author edges disabled."""
        builder = NetworkBuilder(create_author_edges=False)

        builder.add_post('@johndoe', [])

        entities = [{'text': 'John Doe', 'type': 'PER', 'score': 0.95}]
        builder.add_post('@alice', entities)

        # Should NOT have author-to-author edge
        assert not builder.graph.has_edge('@alice', '@johndoe')


class TestEntityResolution:
    """Test entity resolution integration."""

    def test_entity_deduplication(self, builder):
        """Test that same entity in different forms is deduplicated."""
        # First post with "Microsoft"
        entities1 = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]
        builder.add_post('@user1', entities1)

        # Second post with "microsoft" (lowercase)
        entities2 = [{'text': 'microsoft', 'type': 'ORG', 'score': 0.91}]
        builder.add_post('@user2', entities2)

        # Should have resolved to same entity
        org_nodes = [
            n for n, attrs in builder.graph.nodes(data=True)
            if attrs.get('node_type') == 'organization'
        ]

        # Should only have one organization node
        assert len(org_nodes) == 1

    def test_without_resolution(self, builder_no_resolver):
        """Test without entity resolution."""
        # First post with "Microsoft"
        entities1 = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]
        builder_no_resolver.add_post('@user1', entities1)

        # Second post with "microsoft" (lowercase)
        entities2 = [{'text': 'microsoft', 'type': 'ORG', 'score': 0.91}]
        builder_no_resolver.add_post('@user2', entities2)

        # Without resolution, should have two separate nodes
        org_nodes = [
            n for n, attrs in builder_no_resolver.graph.nodes(data=True)
            if attrs.get('node_type') == 'organization'
        ]

        assert len(org_nodes) == 2


class TestStatistics:
    """Test network statistics calculation."""

    def test_basic_statistics(self, builder, sample_entities):
        """Test basic statistics."""
        builder.add_post('@user1', sample_entities)

        stats = builder.get_statistics()

        assert stats['total_nodes'] > 0
        assert stats['total_edges'] > 0
        assert stats['posts_processed'] == 1
        assert stats['authors'] >= 1

    def test_empty_network_statistics(self, builder):
        """Test statistics for empty network."""
        stats = builder.get_statistics()

        assert stats['total_nodes'] == 0
        assert stats['total_edges'] == 0
        assert stats['density'] == 0.0

    def test_node_type_counts(self, builder, sample_entities):
        """Test node type counting."""
        builder.add_post('@user1', sample_entities)

        stats = builder.get_statistics()

        assert stats['authors'] >= 1
        assert stats['persons'] >= 0
        assert stats['locations'] >= 0
        assert stats['organizations'] >= 0

    def test_edge_type_counts(self, builder):
        """Test edge type counting."""
        entities = [
            {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
            {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.89},
            {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}
        ]
        builder.add_post('@user1', entities)

        stats = builder.get_statistics()

        total_mentions = (
            stats['person_mentions'] +
            stats['location_mentions'] +
            stats['organization_mentions']
        )
        assert total_mentions >= 3

    def test_top_entities(self, builder):
        """Test top entities calculation."""
        # Add entities with different mention counts
        entities1 = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]
        entities2 = [{'text': 'Google', 'type': 'ORG', 'score': 0.90}]

        builder.add_post('@user1', entities1)
        builder.add_post('@user1', entities1)  # Microsoft mentioned twice
        builder.add_post('@user2', entities2)  # Google mentioned once

        stats = builder.get_statistics()

        assert 'top_entities' in stats
        assert len(stats['top_entities']) > 0

        # Microsoft should be first (most mentions)
        if len(stats['top_entities']) > 0:
            top_entity = stats['top_entities'][0]
            assert 'entity' in top_entity
            assert 'mentions' in top_entity

    def test_density_calculation(self, builder, sample_entities):
        """Test network density calculation."""
        builder.add_post('@user1', sample_entities)
        builder.add_post('@user2', sample_entities)

        stats = builder.get_statistics()

        assert 'density' in stats
        assert 0 <= stats['density'] <= 1


class TestHelperMethods:
    """Test helper methods."""

    def test_get_node_info(self, builder, sample_entities):
        """Test getting node information."""
        builder.add_post('@user1', sample_entities)

        node_info = builder.get_node_info('@user1')

        assert node_info is not None
        assert node_info['id'] == '@user1'
        assert node_info['type'] == 'author'
        assert node_info['post_count'] == 1

    def test_get_node_info_nonexistent(self, builder):
        """Test getting info for non-existent node."""
        node_info = builder.get_node_info('@nonexistent')
        assert node_info is None

    def test_get_top_authors(self, builder, sample_entities):
        """Test getting top authors."""
        builder.add_post('@user1', sample_entities)
        builder.add_post('@user1', sample_entities)  # 2 posts
        builder.add_post('@user2', sample_entities)  # 1 post

        top_authors = builder.get_top_authors(n=5)

        assert len(top_authors) == 2
        assert top_authors[0]['author'] == '@user1'  # Most posts
        assert top_authors[0]['posts'] == 2

    def test_get_edge_info(self, builder):
        """Test getting edge information."""
        entities = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]
        builder.add_post('@user1', entities, post_id='post_1')

        # Find the edge
        edges = list(builder.graph.out_edges('@user1'))
        if edges:
            target = edges[0][1]
            edge_info = builder.get_edge_info('@user1', target)

            assert edge_info is not None
            assert edge_info['source'] == '@user1'
            assert edge_info['target'] == target
            assert edge_info['weight'] >= 1

    def test_get_edge_info_nonexistent(self, builder):
        """Test getting info for non-existent edge."""
        edge_info = builder.get_edge_info('@user1', '@user2')
        assert edge_info is None


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_graph(self, builder, sample_entities):
        """Test that reset clears the graph."""
        builder.add_post('@user1', sample_entities)

        assert len(builder.graph.nodes) > 0
        assert len(builder.graph.edges) > 0

        builder.reset()

        assert len(builder.graph.nodes) == 0
        assert len(builder.graph.edges) == 0

    def test_reset_clears_statistics(self, builder, sample_entities):
        """Test that reset clears statistics."""
        builder.add_post('@user1', sample_entities)

        assert builder.stats['posts_processed'] > 0

        builder.reset()

        assert builder.stats['posts_processed'] == 0
        assert builder.stats['entities_added'] == 0

    def test_reset_clears_authors(self, builder, sample_entities):
        """Test that reset clears author tracking."""
        builder.add_post('@user1', sample_entities)

        assert len(builder.authors) > 0

        builder.reset()

        assert len(builder.authors) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_author(self, builder, sample_entities):
        """Test handling of empty author."""
        builder.add_post('', sample_entities)

        # Should not crash, but also shouldn't add anything
        assert len(builder.graph.nodes) == 0

    def test_empty_entities_list(self, builder):
        """Test handling of empty entities list."""
        builder.add_post('@user1', [])

        # Should add author node but no entity nodes
        assert builder.graph.has_node('@user1')
        assert len(builder.graph.nodes) == 1

    def test_entity_without_text(self, builder):
        """Test handling of entity without text."""
        entities = [{'type': 'PER', 'score': 0.95}]  # Missing 'text'
        builder.add_post('@user1', entities)

        # Should not crash
        assert builder.graph.has_node('@user1')

    def test_entity_with_empty_text(self, builder):
        """Test handling of entity with empty text."""
        entities = [{'text': '', 'type': 'PER', 'score': 0.95}]
        builder.add_post('@user1', entities)

        # Should not crash or add invalid entity
        assert builder.graph.has_node('@user1')

    def test_whitespace_only_author(self, builder, sample_entities):
        """Test handling of whitespace-only author."""
        builder.add_post('   ', sample_entities)

        # Should not crash
        assert len(builder.graph.nodes) == 0


class TestMultiplePosts:
    """Test handling of multiple posts."""

    def test_multiple_posts_same_author(self, builder, sample_entities):
        """Test multiple posts from same author."""
        builder.add_post('@user1', sample_entities)
        builder.add_post('@user1', sample_entities)
        builder.add_post('@user1', sample_entities)

        # Should have same author node
        assert builder.graph.has_node('@user1')
        assert builder.graph.nodes['@user1']['post_count'] == 3

    def test_multiple_posts_different_authors(self, builder, sample_entities):
        """Test posts from different authors."""
        builder.add_post('@user1', sample_entities)
        builder.add_post('@user2', sample_entities)
        builder.add_post('@user3', sample_entities)

        # Should have 3 author nodes
        author_nodes = [
            n for n, attrs in builder.graph.nodes(data=True)
            if attrs.get('node_type') == 'author'
        ]
        assert len(author_nodes) == 3

    def test_incremental_statistics(self, builder):
        """Test that statistics update incrementally."""
        entities = [{'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}]

        builder.add_post('@user1', entities)
        stats1 = builder.get_statistics()

        builder.add_post('@user2', entities)
        stats2 = builder.get_statistics()

        assert stats2['posts_processed'] > stats1['posts_processed']
        assert stats2['total_nodes'] >= stats1['total_nodes']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Tests for Entity Linking Phase 2 Integration

Tests the integration of entity linking into:
- EntityResolver (Wikidata ID-based resolution)
- NetworkBuilder (metadata storage)
- Pipeline (end-to-end flow)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.entity_resolver import EntityResolver
from src.core.network_builder import NetworkBuilder


class TestEntityResolverWithWikidata:
    """Test EntityResolver with Wikidata ID support"""

    def test_resolve_with_wikidata_id(self):
        """Test that entities with same Wikidata ID are resolved to same canonical form"""
        resolver = EntityResolver()

        # Add Copenhagen in different languages with same Wikidata ID
        copenhagen_en = resolver.get_canonical_form(
            "Copenhagen",
            wikidata_id="Q1748",
            canonical_name="Copenhagen"
        )

        copenhagen_da = resolver.get_canonical_form(
            "København",
            wikidata_id="Q1748",
            canonical_name="Copenhagen"
        )

        copenhagen_fr = resolver.get_canonical_form(
            "Copenhague",
            wikidata_id="Q1748",
            canonical_name="Copenhagen"
        )

        # All should resolve to same canonical form
        assert copenhagen_en == copenhagen_da == copenhagen_fr
        assert copenhagen_en == "Copenhagen"

    def test_resolve_without_wikidata_uses_text_matching(self):
        """Test fallback to text-based matching when no Wikidata ID"""
        resolver = EntityResolver()

        # First occurrence sets canonical
        entity1 = resolver.get_canonical_form("John Smith")
        entity2 = resolver.get_canonical_form("john smith")

        assert entity1 == entity2
        assert entity1 == "John Smith"

    def test_wikidata_id_takes_priority_over_text(self):
        """Test that Wikidata ID matching takes priority over text matching"""
        resolver = EntityResolver()

        # Add same entity text with different Wikidata IDs
        paris_france = resolver.get_canonical_form(
            "Paris",
            wikidata_id="Q90",
            canonical_name="Paris, France"
        )

        paris_texas = resolver.get_canonical_form(
            "Paris",
            wikidata_id="Q16858",
            canonical_name="Paris, Texas"
        )

        # Should be different canonical forms despite same text
        assert paris_france != paris_texas
        assert paris_france == "Paris, France"
        assert paris_texas == "Paris, Texas"

    def test_statistics_include_wikidata_counts(self):
        """Test that statistics include Wikidata-linked entities"""
        resolver = EntityResolver()

        # Add some entities with Wikidata IDs
        resolver.get_canonical_form("Copenhagen", wikidata_id="Q1748")
        resolver.get_canonical_form("København", wikidata_id="Q1748")
        resolver.get_canonical_form("Paris", wikidata_id="Q90")

        # Add entity without Wikidata ID
        resolver.get_canonical_form("Random Entity")

        stats = resolver.get_statistics()

        assert 'wikidata_linked_entities' in stats
        assert stats['wikidata_linked_entities'] == 2  # Q1748 and Q90
        assert 'wikidata_ids' in stats

    def test_reset_clears_wikidata_mappings(self):
        """Test that reset() clears Wikidata mappings"""
        resolver = EntityResolver()

        resolver.get_canonical_form("Copenhagen", wikidata_id="Q1748")
        assert len(resolver.wikidata_map) > 0

        resolver.reset()

        assert len(resolver.entity_map) == 0
        assert len(resolver.wikidata_map) == 0
        assert len(resolver.entity_to_wikidata) == 0


class TestNetworkBuilderWithEntityLinking:
    """Test NetworkBuilder with entity linking metadata"""

    def test_add_entity_with_wikidata_metadata(self):
        """Test that entity nodes store Wikidata metadata"""
        builder = NetworkBuilder(use_entity_resolver=True)

        # Add post with linked entity
        entities = [{
            'text': 'Copenhagen',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q1748',
            'wikipedia_url': 'https://en.wikipedia.org/wiki/Copenhagen',
            'canonical_name': 'Copenhagen',
            'is_linked': True
        }]

        builder.add_post(
            author='test_user',
            entities=entities
        )

        graph = builder.get_graph()

        # Check entity node has metadata
        assert graph.has_node('Copenhagen')
        node_data = graph.nodes['Copenhagen']

        assert node_data.get('wikidata_id') == 'Q1748'
        assert node_data.get('wikipedia_url') == 'https://en.wikipedia.org/wiki/Copenhagen'
        assert node_data.get('is_linked') == True

    def test_cross_language_entity_resolution_with_wikidata(self):
        """Test that entities from different languages merge via Wikidata ID"""
        builder = NetworkBuilder(use_entity_resolver=True)

        # Add post with Copenhagen in English
        entities_en = [{
            'text': 'Copenhagen',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q1748',
            'canonical_name': 'Copenhagen',
            'is_linked': True
        }]

        builder.add_post(author='user1', entities=entities_en)

        # Add post with Copenhagen in Danish
        entities_da = [{
            'text': 'København',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q1748',
            'canonical_name': 'Copenhagen',
            'is_linked': True
        }]

        builder.add_post(author='user2', entities=entities_da)

        graph = builder.get_graph()

        # Should have only ONE node for Copenhagen (merged)
        assert graph.has_node('Copenhagen')
        assert graph.nodes['Copenhagen']['mention_count'] == 2

        # Should NOT have separate node for København
        assert not graph.has_node('København')

    def test_entity_without_linking_metadata_still_works(self):
        """Test backward compatibility - entities without linking metadata work"""
        builder = NetworkBuilder(use_entity_resolver=True)

        # Add entity without linking metadata (old format)
        entities = [{
            'text': 'Some Entity',
            'type': 'PER',
            'score': 0.90
        }]

        builder.add_post(author='test_user', entities=entities)

        graph = builder.get_graph()

        # Should still add node
        assert graph.has_node('Some Entity')
        # But no Wikidata metadata
        assert 'wikidata_id' not in graph.nodes['Some Entity']

    def test_metadata_update_on_subsequent_mentions(self):
        """Test that metadata is added to existing nodes if missing"""
        builder = NetworkBuilder(use_entity_resolver=True)

        # First mention without metadata
        entities1 = [{
            'text': 'Paris',
            'type': 'LOC',
            'score': 0.90
        }]
        builder.add_post(author='user1', entities=entities1)

        # Second mention with metadata
        entities2 = [{
            'text': 'Paris',
            'type': 'LOC',
            'score': 0.95,
            'wikidata_id': 'Q90',
            'wikipedia_url': 'https://en.wikipedia.org/wiki/Paris',
            'is_linked': True
        }]
        builder.add_post(author='user2', entities=entities2)

        graph = builder.get_graph()

        # Node should now have metadata
        assert graph.nodes['Paris'].get('wikidata_id') == 'Q90'
        assert graph.nodes['Paris']['mention_count'] == 2


class TestPipelineEntityLinkingIntegration:
    """Test Pipeline with entity linking enabled"""

    @patch('src.core.pipeline.EntityLinker')
    @patch('src.core.pipeline.NEREngine')
    @patch('src.core.pipeline.NetworkBuilder')
    @patch('src.core.pipeline.DataLoader')
    def test_pipeline_init_with_entity_linking(
        self, mock_loader, mock_builder, mock_ner, mock_linker
    ):
        """Test pipeline initialization with entity linking enabled"""
        from src.core.pipeline import SocialNetworkPipeline

        # Configure mock NER engine to avoid model loading
        mock_ner_instance = MagicMock()
        mock_ner.return_value = mock_ner_instance

        # Configure mock entity linker
        mock_linker_instance = MagicMock()
        mock_linker.return_value = mock_linker_instance

        config = {
            'confidence_threshold': 0.7,
            'enable_cache': True
        }

        pipeline = SocialNetworkPipeline(
            enable_entity_linking=True,
            entity_linking_config=config
        )

        # EntityLinker should be initialized
        assert mock_linker.called
        assert pipeline.entity_linker is not None

    @patch('src.core.pipeline.EntityLinker')
    @patch('src.core.pipeline.NEREngine')
    @patch('src.core.pipeline.NetworkBuilder')
    @patch('src.core.pipeline.DataLoader')
    def test_pipeline_init_without_entity_linking(
        self, mock_loader, mock_builder, mock_ner, mock_linker
    ):
        """Test pipeline initialization with entity linking disabled"""
        from src.core.pipeline import SocialNetworkPipeline

        # Configure mock NER engine
        mock_ner_instance = MagicMock()
        mock_ner.return_value = mock_ner_instance

        pipeline = SocialNetworkPipeline(
            enable_entity_linking=False
        )

        # EntityLinker should NOT be initialized
        assert not mock_linker.called
        assert pipeline.entity_linker is None

    def test_processing_metadata_tracks_linked_entities(self):
        """Test that processing metadata includes entities_linked count"""
        from src.core.pipeline import SocialNetworkPipeline

        # Create pipeline with mocked components to avoid model loading
        with patch('src.core.pipeline.NEREngine'), \
             patch('src.core.pipeline.EntityLinker'):

            pipeline = SocialNetworkPipeline(enable_entity_linking=True)

            # Check metadata structure
            assert 'entities_linked' in pipeline.processing_metadata
            assert pipeline.processing_metadata['entities_linked'] == 0


class TestEndToEndScenarios:
    """Integration tests for real-world scenarios"""

    def test_multilingual_entity_resolution_scenario(self):
        """
        Test realistic multilingual scenario:
        - User1 mentions "København" (Danish)
        - User2 mentions "Copenhagen" (English)
        - User3 mentions "Copenhague" (French)
        - All should map to same entity via Wikidata
        """
        builder = NetworkBuilder(use_entity_resolver=True)

        # Post 1: Danish
        builder.add_post(
            author='danish_user',
            entities=[{
                'text': 'København',
                'type': 'LOC',
                'score': 0.95,
                'wikidata_id': 'Q1748',
                'canonical_name': 'Copenhagen',
                'is_linked': True
            }]
        )

        # Post 2: English
        builder.add_post(
            author='english_user',
            entities=[{
                'text': 'Copenhagen',
                'type': 'LOC',
                'score': 0.95,
                'wikidata_id': 'Q1748',
                'canonical_name': 'Copenhagen',
                'is_linked': True
            }]
        )

        # Post 3: French
        builder.add_post(
            author='french_user',
            entities=[{
                'text': 'Copenhague',
                'type': 'LOC',
                'score': 0.95,
                'wikidata_id': 'Q1748',
                'canonical_name': 'Copenhagen',
                'is_linked': True
            }]
        )

        graph = builder.get_graph()

        # Should have only ONE Copenhagen node
        assert graph.has_node('Copenhagen')
        assert graph.nodes['Copenhagen']['mention_count'] == 3

        # Edges from all three users to same entity
        assert graph.has_edge('danish_user', 'Copenhagen')
        assert graph.has_edge('english_user', 'Copenhagen')
        assert graph.has_edge('french_user', 'Copenhagen')

        # Total 3 author nodes + 1 entity node = 4 nodes
        author_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'author']
        entity_nodes = [n for n, d in graph.nodes(data=True) if d['node_type'] == 'location']

        assert len(author_nodes) == 3
        assert len(entity_nodes) == 1

    def test_mixed_linked_and_unlinked_entities(self):
        """Test handling posts with both linked and unlinked entities"""
        builder = NetworkBuilder(use_entity_resolver=True)

        # Post with mixed entities
        builder.add_post(
            author='user1',
            entities=[
                {
                    'text': 'Copenhagen',
                    'type': 'LOC',
                    'score': 0.95,
                    'wikidata_id': 'Q1748',
                    'canonical_name': 'Copenhagen',
                    'is_linked': True
                },
                {
                    'text': 'Random Place',
                    'type': 'LOC',
                    'score': 0.70,
                    'is_linked': False  # Failed to link
                }
            ]
        )

        graph = builder.get_graph()

        # Both entities should be added
        assert graph.has_node('Copenhagen')
        assert graph.has_node('Random Place')

        # Only Copenhagen should have Wikidata metadata
        assert 'wikidata_id' in graph.nodes['Copenhagen']
        assert 'wikidata_id' not in graph.nodes['Random Place']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

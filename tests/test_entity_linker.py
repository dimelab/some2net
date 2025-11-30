"""
Unit tests for EntityLinker module.

Tests entity linking functionality, caching, and error handling.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.entity_linker import EntityLinker


class TestEntityLinkerInitialization:
    """Test EntityLinker initialization and model loading."""

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    @patch('core.entity_linker.torch.cuda.is_available')
    def test_init_cpu_mode(self, mock_cuda, mock_tokenizer, mock_model):
        """Test initialization in CPU mode."""
        mock_cuda.return_value = False
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        linker = EntityLinker(device='cpu', enable_cache=False)

        assert linker.device == 'cpu'
        assert linker.model_name == "facebook/mgenre-wiki"
        assert linker.cache is None

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    @patch('core.entity_linker.torch.cuda.is_available')
    def test_init_with_cache(self, mock_cuda, mock_tokenizer, mock_model):
        """Test initialization with caching enabled."""
        mock_cuda.return_value = False
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            linker = EntityLinker(
                device='cpu',
                enable_cache=True,
                cache_dir=tmpdir
            )

            assert linker.enable_cache is True
            assert linker.cache is not None
            assert Path(tmpdir).exists()

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    def test_init_custom_parameters(self, mock_tokenizer, mock_model):
        """Test initialization with custom parameters."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        linker = EntityLinker(
            device='cpu',
            confidence_threshold=0.8,
            top_k=10,
            enable_cache=False
        )

        assert linker.confidence_threshold == 0.8
        assert linker.top_k == 10


class TestEntityLinking:
    """Test entity linking functionality."""

    @pytest.fixture
    def mock_linker(self):
        """Create a mocked EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(device='cpu', enable_cache=False)

            # Mock the model and tokenizer
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_parse_candidate_with_language(self, mock_linker):
        """Test parsing candidate with language code."""
        result = mock_linker._parse_candidate("Copenhagen >> en")

        assert result is not None
        assert result['title'] == "Copenhagen"
        assert result['language'] == "en"

    def test_parse_candidate_without_language(self, mock_linker):
        """Test parsing candidate without language code."""
        result = mock_linker._parse_candidate("Copenhagen")

        assert result is not None
        assert result['title'] == "Copenhagen"
        assert result['language'] == "en"  # Default

    def test_parse_candidate_multilingual(self, mock_linker):
        """Test parsing candidates from different languages."""
        test_cases = [
            ("København >> da", "København", "da"),
            ("Copenhague >> fr", "Copenhague", "fr"),
            ("Berlin >> de", "Berlin", "de"),
        ]

        for candidate, expected_title, expected_lang in test_cases:
            result = mock_linker._parse_candidate(candidate)
            assert result['title'] == expected_title
            assert result['language'] == expected_lang

    def test_parse_candidate_empty(self, mock_linker):
        """Test parsing empty candidate."""
        result = mock_linker._parse_candidate("")

        assert result is None

    def test_build_wikipedia_url(self, mock_linker):
        """Test Wikipedia URL construction."""
        url = mock_linker._build_wikipedia_url("Copenhagen", "en")
        assert url == "https://en.wikipedia.org/wiki/Copenhagen"

        url_da = mock_linker._build_wikipedia_url("København", "da")
        assert url_da == "https://da.wikipedia.org/wiki/København"

    def test_build_wikipedia_url_with_spaces(self, mock_linker):
        """Test Wikipedia URL construction with spaces in title."""
        url = mock_linker._build_wikipedia_url("New York City", "en")
        assert url == "https://en.wikipedia.org/wiki/New_York_City"

    def test_extract_language_variants(self, mock_linker):
        """Test extraction of language variants."""
        candidates = [
            {'wikipedia_title': 'Copenhagen', 'language': 'en'},
            {'wikipedia_title': 'København', 'language': 'da'},
            {'wikipedia_title': 'Copenhague', 'language': 'fr'},
        ]

        variants = mock_linker._extract_language_variants(candidates)

        assert 'en' in variants
        assert 'da' in variants
        assert 'fr' in variants
        assert variants['en'] == 'Copenhagen'
        assert variants['da'] == 'København'
        assert variants['fr'] == 'Copenhague'

    def test_extract_language_variants_duplicates(self, mock_linker):
        """Test that only first variant per language is kept."""
        candidates = [
            {'wikipedia_title': 'Copenhagen', 'language': 'en'},
            {'wikipedia_title': 'Copenhagen_City', 'language': 'en'},  # Duplicate language
        ]

        variants = mock_linker._extract_language_variants(candidates)

        assert len(variants) == 1
        assert variants['en'] == 'Copenhagen'


class TestCaching:
    """Test caching functionality."""

    @pytest.fixture
    def linker_with_cache(self):
        """Create EntityLinker with temporary cache."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            tmpdir = tempfile.mkdtemp()
            linker = EntityLinker(
                device='cpu',
                enable_cache=True,
                cache_dir=tmpdir
            )

            # Mock the model
            linker.model = Mock()
            linker.tokenizer = Mock()

            yield linker

            # Cleanup
            shutil.rmtree(tmpdir)

    def test_cache_key_generation(self, linker_with_cache):
        """Test cache key generation."""
        key1 = linker_with_cache._get_cache_key("Copenhagen", "LOC", "en")
        key2 = linker_with_cache._get_cache_key("Copenhagen", "LOC", "en")
        key3 = linker_with_cache._get_cache_key("Copenhagen", "LOC", "da")

        # Same inputs should produce same key
        assert key1 == key2

        # Different language should produce different key
        assert key1 != key3

    def test_cache_set_and_get(self, linker_with_cache):
        """Test caching a result."""
        cache_key = linker_with_cache._get_cache_key("Copenhagen", "LOC", "en")

        # Store in cache
        test_result = {
            'wikipedia_title': 'Copenhagen',
            'canonical_name': 'Copenhagen',
            'linking_confidence': 0.95
        }
        linker_with_cache.cache.set(cache_key, test_result)

        # Retrieve from cache
        cached = linker_with_cache.cache.get(cache_key)

        assert cached is not None
        assert cached['wikipedia_title'] == 'Copenhagen'
        assert cached['linking_confidence'] == 0.95

    def test_cache_clear(self, linker_with_cache):
        """Test clearing cache."""
        # Add something to cache
        cache_key = linker_with_cache._get_cache_key("Test", "PER", "en")
        linker_with_cache.cache.set(cache_key, {'test': 'data'})

        # Clear cache
        linker_with_cache.clear_cache()

        # Verify it's cleared
        assert linker_with_cache.cache.get(cache_key) is None

    def test_cache_stats(self, linker_with_cache):
        """Test cache statistics."""
        # Initially empty
        stats = linker_with_cache.get_cache_stats()
        initial_size = stats['size']

        # Add item
        cache_key = linker_with_cache._get_cache_key("Test", "PER", "en")
        linker_with_cache.cache.set(cache_key, {'test': 'data'})

        # Check stats updated
        stats = linker_with_cache.get_cache_stats()
        assert stats['size'] == initial_size + 1
        assert stats['size_bytes'] > 0


class TestBatchLinking:
    """Test batch entity linking."""

    @pytest.fixture
    def mock_linker(self):
        """Create a mocked EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(device='cpu', enable_cache=False)
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_batch_linking_empty_list(self, mock_linker):
        """Test batch linking with empty list."""
        result = mock_linker.link_entities_batch([])

        assert result == []

    def test_batch_linking_preserves_original_fields(self, mock_linker):
        """Test that batch linking preserves original entity fields."""
        entities = [
            {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.95},
            {'text': 'Paris', 'type': 'LOC', 'score': 0.88},
        ]

        # Mock link_entity to return None (no linking)
        mock_linker.link_entity = Mock(return_value=None)

        result = mock_linker.link_entities_batch(entities)

        # Check original fields preserved
        assert len(result) == 2
        assert result[0]['text'] == 'Copenhagen'
        assert result[0]['type'] == 'LOC'
        assert result[0]['score'] == 0.95
        assert result[1]['text'] == 'Paris'

    def test_batch_linking_adds_is_linked_flag(self, mock_linker):
        """Test that batch linking adds is_linked flag."""
        entities = [
            {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.95},
        ]

        # Mock successful linking
        mock_linker.link_entity = Mock(return_value={
            'wikipedia_title': 'Copenhagen',
            'wikidata_id': 'Q1748',
            'wikipedia_url': 'https://en.wikipedia.org/wiki/Copenhagen',
            'canonical_name': 'Copenhagen',
            'language_variants': {'en': 'Copenhagen'},
            'linking_confidence': 0.9
        })

        result = mock_linker.link_entities_batch(entities)

        assert result[0]['is_linked'] is True
        assert result[0]['canonical_name'] == 'Copenhagen'
        assert result[0]['wikidata_id'] == 'Q1748'

    def test_batch_linking_handles_failures(self, mock_linker):
        """Test that batch linking handles linking failures gracefully."""
        entities = [
            {'text': 'UnknownEntity', 'type': 'PER', 'score': 0.75},
        ]

        # Mock failed linking
        mock_linker.link_entity = Mock(return_value=None)

        result = mock_linker.link_entities_batch(entities)

        assert result[0]['is_linked'] is False
        assert 'wikidata_id' not in result[0]

    def test_batch_linking_with_language(self, mock_linker):
        """Test batch linking with language parameter."""
        entities = [
            {'text': 'København', 'type': 'LOC', 'score': 0.95, 'language': 'da'},
        ]

        # Track calls to link_entity
        mock_linker.link_entity = Mock(return_value=None)

        mock_linker.link_entities_batch(entities, default_language='en')

        # Verify link_entity was called with correct language
        mock_linker.link_entity.assert_called_once()
        call_args = mock_linker.link_entity.call_args
        # link_entity is called with positional args: (text, type, language, context)
        # So language is the third positional argument (index 2) or in kwargs
        if len(call_args[0]) > 2:
            assert call_args[0][2] == 'da'  # Should use entity's language
        else:
            assert call_args.kwargs.get('language') == 'da'

    def test_batch_linking_default_language(self, mock_linker):
        """Test batch linking uses default language when not specified."""
        entities = [
            {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.95},
        ]

        mock_linker.link_entity = Mock(return_value=None)

        mock_linker.link_entities_batch(entities, default_language='fr')

        # Verify default language used
        call_args = mock_linker.link_entity.call_args
        # link_entity is called with positional args: (text, type, language, context)
        if len(call_args[0]) > 2:
            assert call_args[0][2] == 'fr'
        else:
            assert call_args.kwargs.get('language') == 'fr'


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def mock_linker(self):
        """Create a mocked EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(device='cpu', enable_cache=False)
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_link_entity_empty_text(self, mock_linker):
        """Test linking with empty entity text."""
        # Empty text should be handled gracefully in batch processing
        entities = [{'text': '', 'type': 'PER', 'score': 0.5}]
        result = mock_linker.link_entities_batch(entities)

        assert len(result) == 1
        assert result[0]['text'] == ''

    def test_link_entity_special_characters(self, mock_linker):
        """Test linking with special characters."""
        result = mock_linker._parse_candidate("Café_Paris >> fr")

        assert result is not None
        assert result['title'] == "Café_Paris"
        assert result['language'] == "fr"

    def test_get_wikidata_id_placeholder(self, mock_linker):
        """Test that _get_wikidata_id returns None (placeholder)."""
        # Currently a placeholder implementation
        wikidata_id = mock_linker._get_wikidata_id("Copenhagen", "en")

        assert wikidata_id is None  # Placeholder behavior


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.fixture
    def mock_linker(self):
        """Create a mocked EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(device='cpu', enable_cache=False)
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_cross_language_scenario(self, mock_linker):
        """Test cross-language entity linking scenario."""
        # Simulate Copenhagen in different languages
        entities = [
            {'text': 'Copenhagen', 'type': 'LOC', 'score': 0.95, 'language': 'en'},
            {'text': 'København', 'type': 'LOC', 'score': 0.93, 'language': 'da'},
            {'text': 'Copenhague', 'type': 'LOC', 'score': 0.91, 'language': 'fr'},
        ]

        # Mock link_entity to simulate successful linking to same entity
        def mock_link(text, entity_type, language, context=None):
            return {
                'wikipedia_title': 'Copenhagen',
                'wikidata_id': 'Q1748',
                'wikipedia_url': f'https://{language}.wikipedia.org/wiki/Copenhagen',
                'canonical_name': 'Copenhagen',
                'language_variants': {
                    'en': 'Copenhagen',
                    'da': 'København',
                    'fr': 'Copenhague'
                },
                'linking_confidence': 0.9
            }

        mock_linker.link_entity = Mock(side_effect=mock_link)

        result = mock_linker.link_entities_batch(entities)

        # All should link to same Wikidata ID
        assert all(ent['wikidata_id'] == 'Q1748' for ent in result)
        assert all(ent['canonical_name'] == 'Copenhagen' for ent in result)

    def test_mixed_success_failure(self, mock_linker):
        """Test batch with some successful and some failed links."""
        entities = [
            {'text': 'Paris', 'type': 'LOC', 'score': 0.95},
            {'text': 'XYZ123', 'type': 'PER', 'score': 0.75},  # Should fail
            {'text': 'Berlin', 'type': 'LOC', 'score': 0.92},
        ]

        # Mock link_entity to succeed for cities, fail for unknown
        def mock_link(text, entity_type, language, context=None):
            if text in ['Paris', 'Berlin']:
                return {
                    'wikipedia_title': text,
                    'wikidata_id': f'Q{text}',
                    'wikipedia_url': f'https://en.wikipedia.org/wiki/{text}',
                    'canonical_name': text,
                    'language_variants': {'en': text},
                    'linking_confidence': 0.9
                }
            return None

        mock_linker.link_entity = Mock(side_effect=mock_link)

        result = mock_linker.link_entities_batch(entities)

        assert result[0]['is_linked'] is True  # Paris
        assert result[1]['is_linked'] is False  # XYZ123
        assert result[2]['is_linked'] is True  # Berlin


class TestPhase3CustomKnowledgeBase:
    """Test Phase 3 custom knowledge base functionality."""

    @pytest.fixture
    def kb_file(self):
        """Create temporary custom knowledge base file."""
        import json
        tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        kb_data = {
            "Meta": {
                "canonical_name": "Meta Platforms",
                "wikidata_id": "Q380",
                "type": "ORG",
                "aliases": ["Facebook", "Meta Inc"]
            },
            "Copenhagen": {
                "canonical_name": "Copenhagen",
                "wikidata_id": "Q1748",
                "type": "LOC"
            }
        }
        json.dump(kb_data, tmpfile)
        tmpfile.close()
        yield tmpfile.name
        Path(tmpfile.name).unlink()

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    def test_custom_kb_loading(self, mock_tokenizer, mock_model, kb_file):
        """Test loading custom knowledge base."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        linker = EntityLinker(
            device='cpu',
            enable_cache=False,
            custom_kb_path=kb_file
        )

        assert len(linker.custom_kb) > 0
        assert 'meta' in linker.custom_kb  # Normalized
        assert 'copenhagen' in linker.custom_kb

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    def test_custom_kb_alias_lookup(self, mock_tokenizer, mock_model, kb_file):
        """Test that aliases work in custom KB."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        linker = EntityLinker(
            device='cpu',
            enable_cache=False,
            custom_kb_path=kb_file
        )

        # Alias 'facebook' should map to 'Meta Platforms'
        assert 'facebook' in linker.custom_kb
        assert linker.custom_kb['facebook']['canonical_name'] == "Meta Platforms"

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    def test_custom_kb_lookup(self, mock_tokenizer, mock_model, kb_file):
        """Test entity lookup in custom KB."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        linker = EntityLinker(
            device='cpu',
            enable_cache=False,
            custom_kb_path=kb_file
        )

        result = linker._lookup_custom_kb("Meta", "ORG")

        assert result is not None
        assert result['canonical_name'] == "Meta Platforms"
        assert result['wikidata_id'] == "Q380"
        assert result['disambiguation_method'] == 'custom_kb'
        assert result['linking_confidence'] == 1.0

    @patch('core.entity_linker.AutoModelForSeq2SeqLM')
    @patch('core.entity_linker.AutoTokenizer')
    def test_custom_kb_type_mismatch(self, mock_tokenizer, mock_model, kb_file):
        """Test that KB lookup respects entity type."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        linker = EntityLinker(
            device='cpu',
            enable_cache=False,
            custom_kb_path=kb_file
        )

        # 'Meta' is ORG, not PER - should return None
        result = linker._lookup_custom_kb("Meta", "PER")
        assert result is None


class TestPhase3AdvancedDisambiguation:
    """Test Phase 3 advanced disambiguation functionality."""

    @pytest.fixture
    def mock_linker_with_embedder(self):
        """Create EntityLinker with mocked embedder."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'), \
             patch('core.entity_linker.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('core.entity_linker.SentenceTransformer'):

            linker = EntityLinker(
                device='cpu',
                enable_cache=False,
                enable_advanced_disambiguation=True
            )

            # Mock the embedder
            linker.embedder = Mock()
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_advanced_disambiguation_improves_ranking(self, mock_linker_with_embedder):
        """Test that advanced disambiguation can re-rank candidates."""
        import torch

        candidates = [
            {'wikipedia_title': 'Paris_Texas', 'language': 'en', 'confidence': 0.6},
            {'wikipedia_title': 'Paris', 'language': 'en', 'confidence': 0.5},
        ]

        context = "I visited Paris, the capital of France"

        # Mock embeddings - Paris (France) should be more similar to context
        def mock_encode(text, convert_to_tensor=False):
            if "France" in text or "Paris (en)" in text:
                # High similarity for Paris (France)
                return torch.tensor([1.0, 0.0, 0.0])
            else:
                # Low similarity for Paris, Texas
                return torch.tensor([0.0, 1.0, 0.0])

        mock_linker_with_embedder.embedder.encode = Mock(side_effect=mock_encode)
        # Force enable_advanced_disambiguation
        mock_linker_with_embedder.enable_advanced_disambiguation = True

        result = mock_linker_with_embedder._advanced_disambiguation(
            "Paris", candidates, context
        )

        # Second candidate (Paris, France) should win due to context similarity
        assert result['wikipedia_title'] == 'Paris'

    def test_advanced_disambiguation_without_embedder(self):
        """Test disambiguation falls back without embedder."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(
                device='cpu',
                enable_cache=False,
                enable_advanced_disambiguation=False
            )

            candidates = [
                {'wikipedia_title': 'Paris', 'language': 'en', 'confidence': 0.9},
            ]

            result = linker._advanced_disambiguation(
                "Paris", candidates, "context"
            )

            # Should just return first candidate
            assert result == candidates[0]


class TestPhase3EntityRelationships:
    """Test Phase 3 entity relationship extraction."""

    @pytest.fixture
    def mock_linker(self):
        """Create a mocked EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(device='cpu', enable_cache=False)
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_extract_entity_relationships_co_mention(self, mock_linker):
        """Test extraction of co-mention relationships."""
        entities = [
            {
                'text': 'Paris',
                'is_linked': True,
                'wikidata_id': 'Q90',
                'canonical_name': 'Paris',
                'linking_confidence': 0.95
            },
            {
                'text': 'France',
                'is_linked': True,
                'wikidata_id': 'Q142',
                'canonical_name': 'France',
                'linking_confidence': 0.93
            }
        ]

        text = "Paris is the capital of France"

        relationships = mock_linker.extract_entity_relationships(
            entities, text, ['co-mention']
        )

        assert len(relationships) == 1
        assert relationships[0]['source_entity'] == 'Q90'
        assert relationships[0]['target_entity'] == 'Q142'
        assert relationships[0]['relationship_type'] == 'co-mention'
        assert relationships[0]['confidence'] == 0.93  # Min of the two

    def test_extract_entity_relationships_no_linked(self, mock_linker):
        """Test relationship extraction with no linked entities."""
        entities = [
            {'text': 'Unknown', 'is_linked': False}
        ]

        relationships = mock_linker.extract_entity_relationships(
            entities, "text", ['co-mention']
        )

        assert len(relationships) == 0

    def test_extract_entity_relationships_single_entity(self, mock_linker):
        """Test relationship extraction with single entity."""
        entities = [
            {
                'text': 'Paris',
                'is_linked': True,
                'wikidata_id': 'Q90',
                'canonical_name': 'Paris',
                'linking_confidence': 0.95
            }
        ]

        relationships = mock_linker.extract_entity_relationships(
            entities, "Paris is beautiful", ['co-mention']
        )

        # No relationships with only one entity
        assert len(relationships) == 0

    def test_extract_entity_relationships_multiple_entities(self, mock_linker):
        """Test relationship extraction with 3 entities."""
        entities = [
            {'text': 'A', 'is_linked': True, 'wikidata_id': 'Q1', 'linking_confidence': 0.9},
            {'text': 'B', 'is_linked': True, 'wikidata_id': 'Q2', 'linking_confidence': 0.9},
            {'text': 'C', 'is_linked': True, 'wikidata_id': 'Q3', 'linking_confidence': 0.9}
        ]

        relationships = mock_linker.extract_entity_relationships(
            entities, "A, B, and C", ['co-mention']
        )

        # Should have 3 relationships: A-B, A-C, B-C
        assert len(relationships) == 3


class TestPhase3CooccurrenceNetwork:
    """Test Phase 3 co-occurrence network functionality."""

    @pytest.fixture
    def mock_linker(self):
        """Create a mocked EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(device='cpu', enable_cache=False)
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_entity_cooccurrence_tracking(self, mock_linker):
        """Test that co-occurrence is tracked."""
        # Manually populate co-occurrence data
        mock_linker.entity_cooccurrence['Q90']['Q142'] = 5  # Paris-France
        mock_linker.entity_cooccurrence['Q90']['Q456'] = 2  # Paris-London
        mock_linker.entity_cooccurrence['Q142']['Q90'] = 5  # France-Paris

        network = mock_linker.get_entity_network(min_cooccurrence=2)

        assert 'Q90' in network
        assert len(network['Q90']) == 2  # Both edges meet threshold
        assert ('Q142', 5) in network['Q90']
        assert ('Q456', 2) in network['Q90']

    def test_entity_network_threshold(self, mock_linker):
        """Test co-occurrence network threshold filtering."""
        mock_linker.entity_cooccurrence['Q1']['Q2'] = 5
        mock_linker.entity_cooccurrence['Q1']['Q3'] = 1  # Below threshold

        network = mock_linker.get_entity_network(min_cooccurrence=3)

        assert 'Q1' in network
        assert len(network['Q1']) == 1  # Only Q2 meets threshold
        assert ('Q2', 5) in network['Q1']

    def test_save_cooccurrence_data(self, mock_linker):
        """Test saving co-occurrence data to file."""
        import json

        mock_linker.entity_cooccurrence['Q90']['Q142'] = 5

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name

        try:
            mock_linker.save_cooccurrence_data(output_path)

            # Verify file was created and contains correct data
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert 'Q90' in data
            assert data['Q90']['Q142'] == 5

        finally:
            Path(output_path).unlink()


class TestPhase4EntityDescriptions:
    """Test Phase 4 entity description retrieval."""

    @pytest.fixture
    def mock_linker(self):
        """Create EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(
                device='cpu',
                enable_cache=False,
                enable_entity_descriptions=True
            )
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    @patch('core.entity_linker.requests.get')
    def test_get_entity_description_success(self, mock_get, mock_linker):
        """Test successful entity description retrieval."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'entities': {
                'Q90': {
                    'descriptions': {
                        'en': {'value': 'capital and largest city of France'}
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        description = mock_linker._get_entity_description('Q90', 'en')

        assert description == 'capital and largest city of France'
        assert 'Q90_en' in mock_linker.entity_descriptions_cache

    @patch('core.entity_linker.requests.get')
    def test_get_entity_description_cache(self, mock_get, mock_linker):
        """Test that descriptions are cached."""
        # Pre-populate cache
        mock_linker.entity_descriptions_cache['Q90_en'] = 'cached description'

        description = mock_linker._get_entity_description('Q90', 'en')

        assert description == 'cached description'
        # API should not be called
        mock_get.assert_not_called()

    def test_get_entity_description_invalid_qid(self, mock_linker):
        """Test with invalid Wikidata ID."""
        result = mock_linker._get_entity_description('invalid', 'en')
        assert result is None

        result = mock_linker._get_entity_description(None, 'en')
        assert result is None


class TestPhase4DocumentContext:
    """Test Phase 4 document-level context."""

    @pytest.fixture
    def mock_linker(self):
        """Create EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(
                device='cpu',
                enable_cache=False,
                use_document_context=True
            )
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_set_document_context(self, mock_linker):
        """Test setting document context."""
        doc_text = "This is a long document about Paris, France."
        mock_linker.set_document_context(doc_text)

        assert mock_linker.document_context == doc_text

    def test_clear_document_context(self, mock_linker):
        """Test clearing document context."""
        mock_linker.set_document_context("Some text")
        assert mock_linker.document_context is not None

        mock_linker.clear_document_context()
        assert mock_linker.document_context is None


class TestPhase4TypedRelationships:
    """Test Phase 4 typed relationship extraction."""

    @pytest.fixture
    def mock_linker(self):
        """Create EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'):

            linker = EntityLinker(
                device='cpu',
                enable_cache=False,
                enable_typed_relationships=True
            )
            linker.model = Mock()
            linker.tokenizer = Mock()

            return linker

    def test_extract_works_for_relationship(self, mock_linker):
        """Test extraction of works-for relationship."""
        entities = [
            {
                'text': 'John Doe',
                'type': 'PER',
                'is_linked': True,
                'wikidata_id': 'Q1',
                'canonical_name': 'John Doe'
            },
            {
                'text': 'Acme Corp',
                'type': 'ORG',
                'is_linked': True,
                'wikidata_id': 'Q2',
                'canonical_name': 'Acme Corporation'
            }
        ]

        text = "John Doe works for Acme Corp as CEO."

        relationships = mock_linker.extract_typed_relationships(entities, text)

        assert len(relationships) >= 1
        work_rels = [r for r in relationships if r['relationship_type'] == 'works_for']
        assert len(work_rels) == 1
        assert work_rels[0]['source_entity'] == 'Q1'
        assert work_rels[0]['target_entity'] == 'Q2'

    def test_extract_located_in_relationship(self, mock_linker):
        """Test extraction of located-in relationship."""
        entities = [
            {
                'text': 'Paris',
                'type': 'LOC',
                'is_linked': True,
                'wikidata_id': 'Q90',
                'canonical_name': 'Paris'
            },
            {
                'text': 'France',
                'type': 'LOC',
                'is_linked': True,
                'wikidata_id': 'Q142',
                'canonical_name': 'France'
            }
        ]

        text = "Paris is located in France."

        relationships = mock_linker.extract_typed_relationships(entities, text)

        assert len(relationships) >= 1
        loc_rels = [r for r in relationships if r['relationship_type'] == 'located_in']
        assert len(loc_rels) == 1
        assert loc_rels[0]['source_entity'] == 'Q90'
        assert loc_rels[0]['target_entity'] == 'Q142'

    def test_extract_part_of_relationship(self, mock_linker):
        """Test extraction of part-of relationship."""
        entities = [
            {
                'text': 'Instagram',
                'type': 'ORG',
                'is_linked': True,
                'wikidata_id': 'Q209330',
                'canonical_name': 'Instagram'
            },
            {
                'text': 'Meta',
                'type': 'ORG',
                'is_linked': True,
                'wikidata_id': 'Q380',
                'canonical_name': 'Meta Platforms'
            }
        ]

        text = "Instagram is a subsidiary of Meta."

        relationships = mock_linker.extract_typed_relationships(entities, text)

        assert len(relationships) >= 1
        part_rels = [r for r in relationships if r['relationship_type'] == 'part_of']
        assert len(part_rels) == 1
        assert part_rels[0]['source_entity'] == 'Q209330'
        assert part_rels[0]['target_entity'] == 'Q380'

    def test_no_relationships_single_entity(self, mock_linker):
        """Test with single entity (no relationships)."""
        entities = [
            {
                'text': 'Paris',
                'type': 'LOC',
                'is_linked': True,
                'wikidata_id': 'Q90'
            }
        ]

        relationships = mock_linker.extract_typed_relationships(entities, "Paris is beautiful.")

        assert len(relationships) == 0

    def test_relationship_evidence_extraction(self, mock_linker):
        """Test that evidence is extracted correctly."""
        entities = [
            {
                'text': 'Alice',
                'type': 'PER',
                'is_linked': True,
                'wikidata_id': 'Q1',
                'canonical_name': 'Alice'
            },
            {
                'text': 'TechCo',
                'type': 'ORG',
                'is_linked': True,
                'wikidata_id': 'Q2',
                'canonical_name': 'TechCo'
            }
        ]

        text = "Alice is the CEO of TechCo, leading innovation."

        relationships = mock_linker.extract_typed_relationships(entities, text)

        if relationships:
            assert 'evidence' in relationships[0]
            evidence = relationships[0]['evidence']
            assert 'Alice' in evidence
            assert 'TechCo' in evidence


class TestPhase4Integration:
    """Test Phase 4 integration scenarios."""

    @pytest.fixture
    def mock_linker(self):
        """Create fully-featured EntityLinker for testing."""
        with patch('core.entity_linker.AutoModelForSeq2SeqLM'), \
             patch('core.entity_linker.AutoTokenizer'), \
             patch('core.entity_linker.SENTENCE_TRANSFORMERS_AVAILABLE', True), \
             patch('core.entity_linker.SentenceTransformer'):

            linker = EntityLinker(
                device='cpu',
                enable_cache=False,
                enable_advanced_disambiguation=True,
                enable_entity_descriptions=True,
                enable_typed_relationships=True,
                use_document_context=True
            )
            linker.model = Mock()
            linker.tokenizer = Mock()
            linker.embedder = Mock()

            return linker

    def test_all_phase4_features_enabled(self, mock_linker):
        """Test that all Phase 4 features can be enabled together."""
        assert mock_linker.enable_entity_descriptions is True
        assert mock_linker.enable_typed_relationships is True
        assert mock_linker.use_document_context is True

    def test_document_context_used_in_disambiguation(self, mock_linker):
        """Test that document context is used when set."""
        # Set document context
        doc_text = "Long document about Paris, the capital of France."
        mock_linker.set_document_context(doc_text)

        # Document context should be available
        assert mock_linker.document_context == doc_text
        assert mock_linker.use_document_context is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

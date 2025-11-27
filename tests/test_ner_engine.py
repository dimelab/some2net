"""
Unit tests for NER Engine module.

Note: These tests require the model to be downloaded, which happens on first run.
Some tests may be slow due to model loading.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.core.ner_engine import NEREngine


@pytest.fixture(scope="module")
def ner_engine():
    """
    Create NER engine instance for testing.
    Note: This downloads the model on first run, which may take a few minutes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = NEREngine(
            enable_cache=True,
            cache_dir=tmpdir,
            confidence_threshold=0.80  # Lower for testing
        )
        yield engine


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestNEREngineInitialization:
    """Test NER Engine initialization."""

    def test_initialization_cpu(self, temp_cache_dir):
        """Test initialization with CPU."""
        engine = NEREngine(
            device='cpu',
            cache_dir=temp_cache_dir,
            enable_cache=True
        )
        assert engine is not None
        assert engine.device == -1  # CPU
        assert engine.cache is not None

    def test_initialization_no_cache(self):
        """Test initialization without caching."""
        engine = NEREngine(enable_cache=False)
        assert engine is not None
        assert engine.cache is None

    def test_default_parameters(self, temp_cache_dir):
        """Test default parameter values."""
        engine = NEREngine(cache_dir=temp_cache_dir)
        assert engine.model_name == "Davlan/xlm-roberta-base-ner-hrl"
        assert engine.confidence_threshold == 0.85


class TestEntityExtraction:
    """Test entity extraction functionality."""

    def test_extract_single_entity(self, ner_engine):
        """Test extracting entities from single text."""
        text = "John Smith works at Microsoft in Seattle."
        entities = ner_engine.extract_entities(text)

        assert isinstance(entities, list)
        # Should find at least John Smith (PER) and Microsoft (ORG)
        assert len(entities) > 0

        # Check entity structure
        for entity in entities:
            assert 'text' in entity
            assert 'type' in entity
            assert 'score' in entity
            assert entity['type'] in ['PER', 'LOC', 'ORG']

    def test_extract_person_entity(self, ner_engine):
        """Test extracting person entities."""
        text = "Barack Obama met with Angela Merkel in Berlin."
        entities = ner_engine.extract_entities(text)

        # Should find persons
        person_entities = [e for e in entities if e['type'] == 'PER']
        assert len(person_entities) > 0

        # Check for expected names
        entity_texts = [e['text'].lower() for e in person_entities]
        assert any('obama' in text.lower() for text in entity_texts)

    def test_extract_organization_entity(self, ner_engine):
        """Test extracting organization entities."""
        text = "Microsoft and Google are competing with Apple."
        entities = ner_engine.extract_entities(text)

        # Should find organizations
        org_entities = [e for e in entities if e['type'] == 'ORG']
        assert len(org_entities) > 0

    def test_extract_location_entity(self, ner_engine):
        """Test extracting location entities."""
        text = "I visited Paris, London, and Copenhagen last summer."
        entities = ner_engine.extract_entities(text)

        # Should find locations
        loc_entities = [e for e in entities if e['type'] == 'LOC']
        assert len(loc_entities) > 0

    def test_empty_text(self, ner_engine):
        """Test extraction with empty text."""
        entities = ner_engine.extract_entities("")
        assert entities == []

    def test_no_entities(self, ner_engine):
        """Test text with no named entities."""
        text = "This is a simple sentence without any names."
        entities = ner_engine.extract_entities(text)
        # May or may not find entities depending on model, just check it doesn't crash
        assert isinstance(entities, list)


class TestBatchProcessing:
    """Test batch entity extraction."""

    def test_batch_extraction_basic(self, ner_engine):
        """Test basic batch extraction."""
        texts = [
            "John Smith works at Microsoft.",
            "Angela Merkel visited Paris.",
            "Apple released a new iPhone."
        ]

        results, languages = ner_engine.extract_entities_batch(
            texts,
            show_progress=False,
            detect_languages=True
        )

        assert len(results) == 3
        assert len(languages) == 3
        assert all(isinstance(r, list) for r in results)

    def test_batch_empty_list(self, ner_engine):
        """Test batch extraction with empty list."""
        results, languages = ner_engine.extract_entities_batch(
            [],
            show_progress=False
        )

        assert results == []
        assert languages == []

    def test_batch_single_item(self, ner_engine):
        """Test batch extraction with single item."""
        texts = ["Elon Musk founded SpaceX."]
        results, languages = ner_engine.extract_entities_batch(
            texts,
            show_progress=False
        )

        assert len(results) == 1
        assert isinstance(results[0], list)

    def test_batch_different_sizes(self, ner_engine):
        """Test batch processing with different batch sizes."""
        texts = [
            "Person one in Location one.",
            "Person two at Organization two.",
            "Person three visited Location three.",
            "Person four works at Organization four.",
            "Person five in Location five."
        ]

        # Process with different batch sizes
        for batch_size in [1, 2, 10]:
            results, _ = ner_engine.extract_entities_batch(
                texts,
                batch_size=batch_size,
                show_progress=False
            )
            assert len(results) == len(texts)


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_detect_english(self, ner_engine):
        """Test English language detection."""
        text = "This is an English sentence with some words."
        lang = ner_engine.detect_language(text)
        assert lang == 'en'

    def test_detect_danish(self, ner_engine):
        """Test Danish language detection."""
        text = "Dette er en dansk sætning med nogle ord."
        lang = ner_engine.detect_language(text)
        assert lang == 'da'

    def test_detect_spanish(self, ner_engine):
        """Test Spanish language detection."""
        text = "Esta es una oración en español con algunas palabras."
        lang = ner_engine.detect_language(text)
        assert lang in ['es', 'pt']  # Sometimes confused

    def test_detect_empty_text(self, ner_engine):
        """Test language detection with empty text."""
        lang = ner_engine.detect_language("")
        assert lang == 'unknown'

    def test_detect_short_text(self, ner_engine):
        """Test language detection with very short text."""
        lang = ner_engine.detect_language("Hi")
        # May or may not detect correctly, just check it doesn't crash
        assert isinstance(lang, str)

    def test_batch_with_language_detection(self, ner_engine):
        """Test batch processing with language detection."""
        texts = [
            "English text here.",
            "Dette er dansk.",
            "Esto es español."
        ]

        results, languages = ner_engine.extract_entities_batch(
            texts,
            detect_languages=True,
            show_progress=False
        )

        assert len(languages) == 3
        assert languages[0] == 'en'
        assert languages[1] == 'da'


class TestCaching:
    """Test caching functionality."""

    def test_cache_hit(self, temp_cache_dir):
        """Test cache hit on repeated text."""
        engine = NEREngine(
            enable_cache=True,
            cache_dir=temp_cache_dir,
            confidence_threshold=0.80
        )

        text = "John Smith works at Microsoft."

        # First call - cache miss
        result1 = engine.extract_entities(text)
        stats1 = engine.get_cache_stats()

        # Second call - should hit cache
        result2 = engine.extract_entities(text)
        stats2 = engine.get_cache_stats()

        # Results should be identical
        assert result1 == result2

        # Cache should have same size (no new entries)
        assert stats1['size'] == stats2['size']

    def test_cache_clear(self, temp_cache_dir):
        """Test clearing cache."""
        engine = NEREngine(
            enable_cache=True,
            cache_dir=temp_cache_dir
        )

        text = "Test entity extraction."
        engine.extract_entities(text)

        # Cache should have entries
        stats_before = engine.get_cache_stats()
        assert stats_before['size'] > 0

        # Clear cache
        engine.clear_cache()

        # Cache should be empty
        stats_after = engine.get_cache_stats()
        assert stats_after['size'] == 0

    def test_cache_disabled(self):
        """Test engine with caching disabled."""
        engine = NEREngine(enable_cache=False)

        text = "John Smith at Microsoft."
        result = engine.extract_entities(text)

        # Should work without caching
        assert isinstance(result, list)

        # Cache stats should show zero
        stats = engine.get_cache_stats()
        assert stats['size'] == 0

    def test_batch_caching(self, temp_cache_dir):
        """Test caching in batch processing."""
        engine = NEREngine(
            enable_cache=True,
            cache_dir=temp_cache_dir
        )

        texts = [
            "First text with entities.",
            "Second text with entities.",
            "First text with entities."  # Duplicate
        ]

        # Process batch
        results, _ = engine.extract_entities_batch(
            texts,
            show_progress=False
        )

        # Should have 3 results
        assert len(results) == 3

        # First and third should be identical (cached)
        assert results[0] == results[2]


class TestConfidenceThreshold:
    """Test confidence threshold filtering."""

    def test_high_threshold(self, temp_cache_dir):
        """Test with high confidence threshold."""
        engine_high = NEREngine(
            enable_cache=False,
            confidence_threshold=0.95  # Very high
        )

        text = "John Smith works at Microsoft in Seattle."
        entities = engine_high.extract_entities(text)

        # High threshold may filter out some entities
        assert isinstance(entities, list)

    def test_low_threshold(self, temp_cache_dir):
        """Test with low confidence threshold."""
        engine_low = NEREngine(
            enable_cache=False,
            confidence_threshold=0.50  # Low
        )

        text = "John Smith works at Microsoft in Seattle."
        entities = engine_low.extract_entities(text)

        # Low threshold should find more entities
        assert isinstance(entities, list)
        assert len(entities) > 0


class TestEntityCleaning:
    """Test entity cleaning and filtering."""

    def test_entity_types_filtered(self, ner_engine):
        """Test that only PER, LOC, ORG entities are returned."""
        text = "Barack Obama visited the United Nations in New York."
        entities = ner_engine.extract_entities(text)

        # All entities should be PER, LOC, or ORG (no MISC)
        for entity in entities:
            assert entity['type'] in ['PER', 'LOC', 'ORG']

    def test_entity_structure(self, ner_engine):
        """Test entity dictionary structure."""
        text = "John Smith at Microsoft."
        entities = ner_engine.extract_entities(text)

        if entities:  # If any found
            entity = entities[0]
            assert 'text' in entity
            assert 'type' in entity
            assert 'score' in entity
            assert 'start' in entity
            assert 'end' in entity

            # Check types
            assert isinstance(entity['text'], str)
            assert isinstance(entity['type'], str)
            assert isinstance(entity['score'], float)
            assert isinstance(entity['start'], int)
            assert isinstance(entity['end'], int)


class TestMultilingual:
    """Test multilingual entity extraction."""

    def test_english_entities(self, ner_engine):
        """Test English text processing."""
        text = "Joe Biden met with Boris Johnson in London."
        entities = ner_engine.extract_entities(text)

        assert len(entities) > 0
        # Should find persons and location

    def test_danish_entities(self, ner_engine):
        """Test Danish text processing."""
        text = "Mette Frederiksen mødtes med Angela Merkel i København."
        entities = ner_engine.extract_entities(text)

        # Should find entities in Danish text
        assert isinstance(entities, list)

    def test_mixed_language_batch(self, ner_engine):
        """Test batch with mixed languages."""
        texts = [
            "Barack Obama in Washington.",  # English
            "Angela Merkel in Berlin.",      # English/German name
            "Dronningen besøgte København."  # Danish
        ]

        results, languages = ner_engine.extract_entities_batch(
            texts,
            detect_languages=True,
            show_progress=False
        )

        assert len(results) == 3
        assert len(languages) == 3

        # All should return some results
        for result in results:
            assert isinstance(result, list)


class TestErrorHandling:
    """Test error handling."""

    def test_malformed_text(self, ner_engine):
        """Test with unusual characters."""
        text = "Test 🎉 with 表情 and symbols ñ ü ø"
        entities = ner_engine.extract_entities(text)

        # Should not crash
        assert isinstance(entities, list)

    def test_very_long_text(self, ner_engine):
        """Test with very long text (exceeds model max length)."""
        # Create text longer than 512 tokens
        text = " ".join(["word"] * 1000)
        entities = ner_engine.extract_entities(text)

        # Should not crash (will be truncated by model)
        assert isinstance(entities, list)

    def test_special_characters(self, ner_engine):
        """Test text with special characters."""
        text = "Person @ Company & Location | etc."
        entities = ner_engine.extract_entities(text)

        # Should not crash
        assert isinstance(entities, list)


@pytest.mark.integration
class TestIntegration:
    """Integration tests with real data."""

    def test_with_sample_data(self, ner_engine):
        """Test with actual sample data from examples."""
        texts = [
            "John Smith announced that Microsoft will open a new office in Copenhagen.",
            "Angela Merkel visited Paris to meet with Emmanuel Macron.",
            "Apple Inc. released a new product in California."
        ]

        results, languages = ner_engine.extract_entities_batch(
            texts,
            show_progress=False
        )

        # Should process all texts
        assert len(results) == 3

        # Each should have some entities
        for i, result in enumerate(results):
            assert len(result) > 0, f"No entities found in text {i}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

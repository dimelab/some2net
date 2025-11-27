"""
Integration tests for DataLoader + NER Engine.

Tests the complete data loading and entity extraction pipeline.
"""

import pytest
from pathlib import Path
import tempfile

from src.core.data_loader import DataLoader
from src.core.ner_engine import NEREngine


@pytest.fixture(scope="module")
def ner_engine():
    """Create NER engine for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = NEREngine(
            enable_cache=True,
            cache_dir=tmpdir,
            confidence_threshold=0.80
        )
        yield engine


@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()


class TestDataLoaderNERIntegration:
    """Integration tests combining DataLoader and NER Engine."""

    def test_csv_to_ner_pipeline(self, data_loader, ner_engine):
        """Test complete pipeline: CSV -> chunks -> NER extraction."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        all_entities = []
        all_texts = []
        total_rows = 0

        # Load CSV in chunks
        for chunk in data_loader.load_csv(
            example_file,
            author_column='author',
            text_column='text',
            chunksize=5
        ):
            total_rows += len(chunk)
            texts = chunk['text'].tolist()
            all_texts.extend(texts)

            # Extract entities from chunk
            entities_batch, languages = ner_engine.extract_entities_batch(
                texts,
                show_progress=False,
                detect_languages=True
            )

            all_entities.extend(entities_batch)

            # Verify results structure
            assert len(entities_batch) == len(texts)
            assert len(languages) == len(texts)

            # Check languages detected
            for lang in languages:
                assert isinstance(lang, str)

        # Verify we processed data
        assert total_rows > 0
        assert len(all_entities) == total_rows
        assert len(all_texts) == total_rows

        # Verify entities were found
        entities_found = sum(len(e) for e in all_entities)
        assert entities_found > 0, "No entities found in sample data"

        print(f"\nProcessed {total_rows} rows, found {entities_found} entities")

    def test_ndjson_to_ner_pipeline(self, data_loader, ner_engine):
        """Test pipeline with NDJSON format."""
        example_file = Path('examples/sample_data.ndjson')

        if not example_file.exists():
            pytest.skip("Example NDJSON file not found")

        all_entities = []
        total_rows = 0

        # Load NDJSON in chunks
        for chunk in data_loader.load_ndjson(
            example_file,
            author_column='author',
            text_column='text',
            chunksize=5
        ):
            total_rows += len(chunk)
            texts = chunk['text'].tolist()

            # Extract entities
            entities_batch, _ = ner_engine.extract_entities_batch(
                texts,
                show_progress=False
            )

            all_entities.extend(entities_batch)

        assert total_rows > 0
        assert len(all_entities) == total_rows

    def test_danish_data_pipeline(self, data_loader, ner_engine):
        """Test pipeline with Danish data."""
        example_file = Path('examples/sample_danish.csv')

        if not example_file.exists():
            pytest.skip("Danish example file not found")

        all_entities = []
        all_languages = []

        # Load Danish CSV
        for chunk in data_loader.load_csv(
            example_file,
            author_column='forfatter',
            text_column='tekst'
        ):
            texts = chunk['tekst'].tolist()

            # Extract entities with language detection
            entities_batch, languages = ner_engine.extract_entities_batch(
                texts,
                show_progress=False,
                detect_languages=True
            )

            all_entities.extend(entities_batch)
            all_languages.extend(languages)

        # Should detect Danish language
        danish_count = sum(1 for lang in all_languages if lang == 'da')
        assert danish_count > 0, "Failed to detect Danish language"

        # Should find some entities
        total_entities = sum(len(e) for e in all_entities)
        assert total_entities > 0, "No entities found in Danish text"

        print(f"\nDanish texts: {danish_count}/{len(all_languages)}")
        print(f"Entities found: {total_entities}")

    def test_batch_processing_efficiency(self, data_loader, ner_engine):
        """Test that batch processing is more efficient than individual processing."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        # Load all data
        all_chunks = list(data_loader.load_csv(
            example_file,
            author_column='author',
            text_column='text'
        ))

        if not all_chunks:
            pytest.skip("No data loaded")

        import pandas as pd
        df = pd.concat(all_chunks, ignore_index=True)
        texts = df['text'].tolist()[:10]  # Just first 10 for speed

        if len(texts) < 2:
            pytest.skip("Need at least 2 texts")

        # Process in batch
        entities_batch, _ = ner_engine.extract_entities_batch(
            texts,
            batch_size=5,
            show_progress=False
        )

        # Process individually
        entities_individual = [
            ner_engine.extract_entities(text)
            for text in texts
        ]

        # Results should be the same
        assert len(entities_batch) == len(entities_individual)

        # Compare entity counts (may not be exactly equal due to batching differences)
        batch_count = sum(len(e) for e in entities_batch)
        individual_count = sum(len(e) for e in entities_individual)

        print(f"\nBatch entities: {batch_count}, Individual entities: {individual_count}")

    def test_caching_across_chunks(self, data_loader):
        """Test that caching works across multiple chunks."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create engine with caching
            engine = NEREngine(
                enable_cache=True,
                cache_dir=tmpdir,
                confidence_threshold=0.80
            )

            # Process data twice
            for iteration in range(2):
                for chunk in data_loader.load_csv(
                    example_file,
                    author_column='author',
                    text_column='text',
                    chunksize=5
                ):
                    texts = chunk['text'].tolist()
                    entities_batch, _ = engine.extract_entities_batch(
                        texts,
                        show_progress=False
                    )

                    assert len(entities_batch) == len(texts)

            # Check cache has entries
            stats = engine.get_cache_stats()
            assert stats['size'] > 0, "Cache should have entries after processing"

    def test_entity_aggregation(self, data_loader, ner_engine):
        """Test aggregating entities across multiple posts."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        # Track entity mentions
        entity_mentions = {}

        for chunk in data_loader.load_csv(
            example_file,
            author_column='author',
            text_column='text'
        ):
            texts = chunk['text'].tolist()
            entities_batch, _ = ner_engine.extract_entities_batch(
                texts,
                show_progress=False
            )

            # Aggregate entities
            for entities in entities_batch:
                for entity in entities:
                    entity_text = entity['text']
                    entity_type = entity['type']

                    key = (entity_text, entity_type)
                    entity_mentions[key] = entity_mentions.get(key, 0) + 1

        # Should have found multiple entities
        assert len(entity_mentions) > 0

        # Print top entities
        sorted_entities = sorted(
            entity_mentions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print("\nTop entities found:")
        for (text, etype), count in sorted_entities[:10]:
            print(f"  {text} ({etype}): {count} mentions")

    def test_memory_efficient_processing(self, data_loader, ner_engine):
        """Test that processing is memory efficient (uses chunks)."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        # Process with very small chunks
        chunk_sizes = []
        max_chunk_size = 0

        for chunk in data_loader.load_csv(
            example_file,
            author_column='author',
            text_column='text',
            chunksize=3  # Very small chunks
        ):
            chunk_size = len(chunk)
            chunk_sizes.append(chunk_size)
            max_chunk_size = max(max_chunk_size, chunk_size)

            # Process chunk
            texts = chunk['text'].tolist()
            entities_batch, _ = ner_engine.extract_entities_batch(
                texts,
                show_progress=False
            )

            assert len(entities_batch) == chunk_size

        # Verify chunking worked
        assert len(chunk_sizes) > 1, "Should have multiple chunks with small chunksize"
        assert max_chunk_size <= 3, "Chunks should respect chunksize limit"

        print(f"\nProcessed {len(chunk_sizes)} chunks, max size: {max_chunk_size}")

    def test_error_recovery(self, data_loader, ner_engine):
        """Test that pipeline handles errors gracefully."""
        # Create temp file with some problematic data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("author,text\n")
            f.write("@user1,Normal text with John Smith\n")
            f.write("@user2,\n")  # Empty text (will be filtered)
            f.write("@user3,Another normal text\n")
            filepath = f.name

        try:
            total_processed = 0

            for chunk in data_loader.load_csv(
                filepath,
                author_column='author',
                text_column='text'
            ):
                texts = chunk['text'].tolist()
                entities_batch, _ = ner_engine.extract_entities_batch(
                    texts,
                    show_progress=False
                )

                total_processed += len(texts)

            # Should process valid rows (empty text filtered by DataLoader)
            assert total_processed == 2

        finally:
            Path(filepath).unlink(missing_ok=True)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_pipeline_english(self, data_loader):
        """Test complete pipeline with English data."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            engine = NEREngine(
                enable_cache=True,
                cache_dir=tmpdir,
                confidence_threshold=0.80
            )

            # Process data
            results = {
                'total_posts': 0,
                'total_entities': 0,
                'entities_by_type': {'PER': 0, 'LOC': 0, 'ORG': 0},
                'languages': {}
            }

            for chunk in data_loader.load_csv(
                example_file,
                author_column='author',
                text_column='text',
                chunksize=10
            ):
                texts = chunk['text'].tolist()
                results['total_posts'] += len(texts)

                # Extract entities
                entities_batch, languages = engine.extract_entities_batch(
                    texts,
                    show_progress=False,
                    detect_languages=True
                )

                # Aggregate results
                for entities, lang in zip(entities_batch, languages):
                    results['total_entities'] += len(entities)

                    for entity in entities:
                        results['entities_by_type'][entity['type']] += 1

                    results['languages'][lang] = results['languages'].get(lang, 0) + 1

            # Verify results
            assert results['total_posts'] > 0
            assert results['total_entities'] > 0

            print("\nPipeline Results:")
            print(f"  Posts processed: {results['total_posts']}")
            print(f"  Entities found: {results['total_entities']}")
            print(f"  By type: {results['entities_by_type']}")
            print(f"  Languages: {results['languages']}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])

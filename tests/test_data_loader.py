"""
Unit tests for DataLoader module.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.core.data_loader import DataLoader


@pytest.fixture
def data_loader():
    """Create DataLoader instance for testing."""
    return DataLoader()


@pytest.fixture
def sample_csv_file():
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("author,text,post_id\n")
        f.write("@user1,John Smith works at Microsoft,1\n")
        f.write("@user2,Meeting in Copenhagen tomorrow,2\n")
        f.write("@user3,Apple released new product,3\n")
        filepath = f.name

    yield filepath

    # Cleanup
    Path(filepath).unlink(missing_ok=True)


@pytest.fixture
def sample_ndjson_file():
    """Create temporary NDJSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False, encoding='utf-8') as f:
        f.write('{"author": "@user1", "text": "John Smith works at Microsoft", "post_id": "1"}\n')
        f.write('{"author": "@user2", "text": "Meeting in Copenhagen tomorrow", "post_id": "2"}\n')
        f.write('{"author": "@user3", "text": "Apple released new product", "post_id": "3"}\n')
        filepath = f.name

    yield filepath

    # Cleanup
    Path(filepath).unlink(missing_ok=True)


@pytest.fixture
def large_csv_file():
    """Create large CSV file for chunking tests."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("author,text,post_id\n")
        for i in range(100):
            f.write(f"@user{i % 10},This is test post number {i},post_{i}\n")
        filepath = f.name

    yield filepath

    Path(filepath).unlink(missing_ok=True)


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_initialization(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader is not None
        assert '.csv' in data_loader.supported_formats
        assert '.ndjson' in data_loader.supported_formats

    def test_detect_encoding(self, data_loader, sample_csv_file):
        """Test encoding detection."""
        encoding = data_loader.detect_encoding(sample_csv_file)
        assert encoding is not None
        assert isinstance(encoding, str)

    def test_validate_columns_success(self, data_loader):
        """Test successful column validation."""
        df = pd.DataFrame({
            'author': ['@user1', '@user2'],
            'text': ['Hello', 'World'],
            'extra': [1, 2]
        })

        result = data_loader.validate_columns(df, 'author', 'text')
        assert result is True

    def test_validate_columns_missing(self, data_loader):
        """Test column validation with missing columns."""
        df = pd.DataFrame({
            'author': ['@user1', '@user2'],
            'content': ['Hello', 'World']  # Wrong column name
        })

        with pytest.raises(ValueError) as exc_info:
            data_loader.validate_columns(df, 'author', 'text')

        assert 'Missing required columns' in str(exc_info.value)

    def test_load_csv_basic(self, data_loader, sample_csv_file):
        """Test basic CSV loading."""
        chunks = list(data_loader.load_csv(
            sample_csv_file,
            author_column='author',
            text_column='text',
            chunksize=10
        ))

        assert len(chunks) > 0
        assert isinstance(chunks[0], pd.DataFrame)
        assert 'author' in chunks[0].columns
        assert 'text' in chunks[0].columns

    def test_load_csv_chunking(self, data_loader, large_csv_file):
        """Test CSV chunking with large file."""
        chunk_size = 25
        chunks = list(data_loader.load_csv(
            large_csv_file,
            author_column='author',
            text_column='text',
            chunksize=chunk_size
        ))

        # Should have multiple chunks
        assert len(chunks) > 1

        # Each chunk should be <= chunk_size
        for chunk in chunks[:-1]:  # All except potentially smaller last chunk
            assert len(chunk) <= chunk_size

    def test_load_ndjson_basic(self, data_loader, sample_ndjson_file):
        """Test basic NDJSON loading."""
        chunks = list(data_loader.load_ndjson(
            sample_ndjson_file,
            author_column='author',
            text_column='text',
            chunksize=10
        ))

        assert len(chunks) > 0
        assert isinstance(chunks[0], pd.DataFrame)
        assert 'author' in chunks[0].columns
        assert 'text' in chunks[0].columns

    def test_load_file_auto_csv(self, data_loader, sample_csv_file):
        """Test automatic format detection for CSV."""
        chunks = list(data_loader.load_file(
            sample_csv_file,
            author_column='author',
            text_column='text'
        ))

        assert len(chunks) > 0
        assert isinstance(chunks[0], pd.DataFrame)

    def test_load_file_auto_ndjson(self, data_loader, sample_ndjson_file):
        """Test automatic format detection for NDJSON."""
        chunks = list(data_loader.load_file(
            sample_ndjson_file,
            author_column='author',
            text_column='text'
        ))

        assert len(chunks) > 0
        assert isinstance(chunks[0], pd.DataFrame)

    def test_get_column_names_csv(self, data_loader, sample_csv_file):
        """Test getting column names from CSV."""
        columns = data_loader.get_column_names(sample_csv_file)

        assert isinstance(columns, list)
        assert 'author' in columns
        assert 'text' in columns
        assert 'post_id' in columns

    def test_get_column_names_ndjson(self, data_loader, sample_ndjson_file):
        """Test getting column names from NDJSON."""
        columns = data_loader.get_column_names(sample_ndjson_file)

        assert isinstance(columns, list)
        assert 'author' in columns
        assert 'text' in columns
        assert 'post_id' in columns

    def test_file_not_found(self, data_loader):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            list(data_loader.load_csv(
                'nonexistent_file.csv',
                author_column='author',
                text_column='text'
            ))

    def test_unsupported_format(self, data_loader):
        """Test handling of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(ValueError) as exc_info:
                list(data_loader.load_file(
                    filepath,
                    author_column='author',
                    text_column='text'
                ))

            assert 'Unsupported file format' in str(exc_info.value)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_empty_text_filtering(self, data_loader):
        """Test that rows with empty text are filtered out."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("author,text\n")
            f.write("@user1,Valid text\n")
            f.write("@user2,\n")  # Empty text
            f.write("@user3,Another valid text\n")
            filepath = f.name

        try:
            chunks = list(data_loader.load_csv(
                filepath,
                author_column='author',
                text_column='text'
            ))

            # Concatenate all chunks
            df = pd.concat(chunks, ignore_index=True)

            # Should only have 2 rows (empty text filtered)
            assert len(df) == 2
            assert '@user2' not in df['author'].values
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_malformed_json_handling(self, data_loader):
        """Test handling of malformed JSON in NDJSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False, encoding='utf-8') as f:
            f.write('{"author": "@user1", "text": "Valid JSON"}\n')
            f.write('{invalid json}\n')  # Malformed
            f.write('{"author": "@user3", "text": "Valid again"}\n')
            filepath = f.name

        try:
            chunks = list(data_loader.load_ndjson(
                filepath,
                author_column='author',
                text_column='text'
            ))

            df = pd.concat(chunks, ignore_index=True)

            # Should have 2 valid rows (malformed JSON skipped)
            assert len(df) == 2
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests with real example files."""

    def test_load_example_csv(self, data_loader):
        """Test loading actual example CSV file."""
        example_file = Path('examples/sample_data.csv')

        if not example_file.exists():
            pytest.skip("Example file not found")

        chunks = list(data_loader.load_csv(
            example_file,
            author_column='author',
            text_column='text'
        ))

        assert len(chunks) > 0
        df = pd.concat(chunks, ignore_index=True)
        assert len(df) > 0

    def test_load_example_ndjson(self, data_loader):
        """Test loading actual example NDJSON file."""
        example_file = Path('examples/sample_data.ndjson')

        if not example_file.exists():
            pytest.skip("Example file not found")

        chunks = list(data_loader.load_ndjson(
            example_file,
            author_column='author',
            text_column='text'
        ))

        assert len(chunks) > 0
        df = pd.concat(chunks, ignore_index=True)
        assert len(df) > 0

    def test_load_danish_example(self, data_loader):
        """Test loading Danish example file."""
        example_file = Path('examples/sample_danish.csv')

        if not example_file.exists():
            pytest.skip("Danish example file not found")

        chunks = list(data_loader.load_csv(
            example_file,
            author_column='forfatter',
            text_column='tekst'
        ))

        assert len(chunks) > 0
        df = pd.concat(chunks, ignore_index=True)
        assert len(df) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

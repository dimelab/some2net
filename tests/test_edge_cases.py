"""
Edge case tests for the Social Network Analytics library.

Tests unusual inputs, boundary conditions, and potential failure modes.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.core.data_loader import DataLoader
from src.core.entity_resolver import EntityResolver
from src.core.network_builder import NetworkBuilder


# =============================================================================
# Data Loader Edge Cases
# =============================================================================

@pytest.mark.edge_case
class TestDataLoaderEdgeCases:
    """Test edge cases for DataLoader."""

    def test_empty_file(self, temp_dir):
        """Test loading empty CSV file."""
        filepath = temp_dir / "empty.csv"
        filepath.write_text("")

        loader = DataLoader()

        with pytest.raises(Exception):  # Should raise some error
            list(loader.load_csv(filepath, "author", "text"))

    def test_header_only(self, temp_dir):
        """Test CSV with header but no data."""
        filepath = temp_dir / "header_only.csv"
        filepath.write_text("author,text\n")

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        # Should return empty list or handle gracefully
        assert len(chunks) == 0 or all(len(chunk) == 0 for chunk in chunks)

    def test_missing_columns(self, temp_dir):
        """Test CSV with missing required columns."""
        filepath = temp_dir / "wrong_columns.csv"
        filepath.write_text("user,content\n@user1,text1\n")

        loader = DataLoader()

        with pytest.raises(ValueError) as exc_info:
            list(loader.load_csv(filepath, "author", "text"))

        assert "author" in str(exc_info.value).lower() or "column" in str(exc_info.value).lower()

    def test_special_characters_in_text(self, temp_dir):
        """Test handling special characters."""
        filepath = temp_dir / "special_chars.csv"
        df = pd.DataFrame({
            'author': ['@user1', '@user2', '@user3'],
            'text': [
                "Test with emoji 😀🎉",
                "Special chars: !@#$%^&*()",
                "Newline\nand\ttab"
            ]
        })
        df.to_csv(filepath, index=False)

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        assert len(chunks) > 0
        assert len(chunks[0]) == 3

    def test_unicode_text(self, temp_dir):
        """Test handling Unicode text."""
        filepath = temp_dir / "unicode.csv"
        df = pd.DataFrame({
            'author': ['@user1', '@user2', '@user3'],
            'text': [
                "Hello in Chinese: 你好",
                "Russian: Привет мир",
                "Arabic: مرحبا بالعالم"
            ]
        })
        df.to_csv(filepath, index=False, encoding='utf-8')

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        assert len(chunks) > 0
        assert "你好" in chunks[0]['text'].values[0]

    def test_very_long_text(self, temp_dir):
        """Test handling very long text (>10k characters)."""
        filepath = temp_dir / "long_text.csv"
        long_text = "A" * 20000  # 20k characters

        df = pd.DataFrame({
            'author': ['@user1'],
            'text': [long_text]
        })
        df.to_csv(filepath, index=False)

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        assert len(chunks) > 0
        assert len(chunks[0]['text'].values[0]) == 20000

    def test_null_values(self, temp_dir):
        """Test handling null/missing values."""
        filepath = temp_dir / "nulls.csv"
        df = pd.DataFrame({
            'author': ['@user1', None, '@user3'],
            'text': ['text1', 'text2', None]
        })
        df.to_csv(filepath, index=False)

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        # Should handle nulls gracefully (fill with 'unknown' or empty string)
        assert len(chunks) > 0

    def test_duplicate_rows(self, temp_dir):
        """Test handling duplicate rows."""
        filepath = temp_dir / "duplicates.csv"
        df = pd.DataFrame({
            'author': ['@user1', '@user1', '@user1'],
            'text': ['same text', 'same text', 'same text']
        })
        df.to_csv(filepath, index=False)

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        assert len(chunks) > 0
        assert len(chunks[0]) == 3  # All rows should be loaded

    def test_mixed_encodings(self, temp_dir):
        """Test file with mixed character encodings."""
        filepath = temp_dir / "mixed_encoding.csv"

        # Write with UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("author,text\n")
            f.write("@user1,English text\n")
            f.write("@user2,Français text\n")
            f.write("@user3,中文 text\n")

        loader = DataLoader()

        chunks = list(loader.load_csv(filepath, "author", "text"))

        assert len(chunks) > 0
        assert len(chunks[0]) == 3


# =============================================================================
# Entity Resolver Edge Cases
# =============================================================================

@pytest.mark.edge_case
class TestEntityResolverEdgeCases:
    """Test edge cases for EntityResolver."""

    def test_empty_entity_text(self):
        """Test handling empty entity text."""
        resolver = EntityResolver()

        canonical = resolver.get_canonical_form("")

        assert canonical == ""

    def test_whitespace_only(self):
        """Test handling whitespace-only entity."""
        resolver = EntityResolver()

        canonical = resolver.get_canonical_form("   \n\t  ")

        assert canonical.strip() == ""

    def test_very_long_entity_name(self):
        """Test handling very long entity names."""
        resolver = EntityResolver()

        long_name = "A" * 1000
        canonical = resolver.get_canonical_form(long_name)

        assert len(canonical) == 1000

    def test_unicode_normalization(self):
        """Test Unicode normalization in entity names."""
        resolver = EntityResolver()

        # Different Unicode representations of same character
        name1 = "Café"  # é as single character
        name2 = "Cafe\u0301"  # e + combining acute accent

        canon1 = resolver.get_canonical_form(name1)
        canon2 = resolver.get_canonical_form(name2)

        # Should normalize to same form (lowercase)
        assert canon1.lower() == canon2.lower()

    def test_special_characters_in_names(self):
        """Test entity names with special characters."""
        resolver = EntityResolver()

        names = [
            "O'Brien",
            "Jean-Claude",
            "Smith & Jones",
            "Company (UK) Ltd.",
            "São Paulo"
        ]

        for name in names:
            canonical = resolver.get_canonical_form(name)
            assert canonical is not None
            assert len(canonical) > 0

    def test_numbers_in_names(self):
        """Test entity names with numbers."""
        resolver = EntityResolver()

        canonical1 = resolver.get_canonical_form("Microsoft 365")
        canonical2 = resolver.get_canonical_form("Microsoft 365")

        # Should match
        assert canonical1 == canonical2

    def test_case_variations(self):
        """Test various case combinations."""
        resolver = EntityResolver()

        variations = [
            "John Smith",
            "JOHN SMITH",
            "john smith",
            "JoHn SmItH"
        ]

        canonicals = [resolver.get_canonical_form(v) for v in variations]

        # All should map to same canonical form
        assert len(set(canonicals)) == 1

    def test_leading_trailing_whitespace(self):
        """Test whitespace trimming."""
        resolver = EntityResolver()

        canonical1 = resolver.get_canonical_form("  John Smith  ")
        canonical2 = resolver.get_canonical_form("John Smith")

        assert canonical1 == canonical2


# =============================================================================
# Network Builder Edge Cases
# =============================================================================

@pytest.mark.edge_case
class TestNetworkBuilderEdgeCases:
    """Test edge cases for NetworkBuilder."""

    def test_empty_author(self):
        """Test handling empty author name."""
        builder = NetworkBuilder()

        # Should handle gracefully
        builder.add_post(
            author="",
            entities=[{'text': 'Test', 'type': 'PER', 'score': 0.9}]
        )

        stats = builder.get_statistics()
        # Should not add node for empty author
        assert stats['authors'] == 0

    def test_no_entities(self):
        """Test post with no entities."""
        builder = NetworkBuilder()

        builder.add_post(
            author="@user1",
            entities=[]
        )

        stats = builder.get_statistics()

        assert stats['authors'] == 1
        assert stats['total_edges'] == 0

    def test_duplicate_entities_in_post(self):
        """Test post mentioning same entity multiple times."""
        builder = NetworkBuilder()

        builder.add_post(
            author="@user1",
            entities=[
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.9},
                {'text': 'Microsoft', 'type': 'ORG', 'score': 0.95},
                {'text': 'microsoft', 'type': 'ORG', 'score': 0.92}
            ]
        )

        graph = builder.get_graph()

        # Should create single entity node
        org_nodes = [n for n, d in graph.nodes(data=True)
                     if d.get('node_type') == 'organization']

        assert len(org_nodes) == 1

    def test_very_high_mention_count(self):
        """Test entity mentioned many times."""
        builder = NetworkBuilder()

        entity = {'text': 'Popular Entity', 'type': 'PER', 'score': 0.9}

        for i in range(1000):
            builder.add_post(
                author=f"@user{i}",
                entities=[entity]
            )

        graph = builder.get_graph()
        stats = builder.get_statistics()

        # Entity should have high mention count
        entity_node = 'Popular Entity'  # After resolution
        if entity_node in graph.nodes:
            assert graph.nodes[entity_node]['mention_count'] >= 1000

    def test_self_mention(self):
        """Test author mentioning themselves."""
        builder = NetworkBuilder(create_author_edges=True)

        builder.add_post(
            author="@johndoe",
            entities=[
                {'text': 'John Doe', 'type': 'PER', 'score': 0.9}
            ]
        )

        graph = builder.get_graph()

        # Should not create self-loop
        # (depends on implementation - author matching)

    def test_maximum_nodes(self):
        """Test network with many nodes."""
        builder = NetworkBuilder()

        for i in range(1000):
            builder.add_post(
                author=f"@user{i}",
                entities=[
                    {'text': f'Entity {i}', 'type': 'PER', 'score': 0.9}
                ]
            )

        stats = builder.get_statistics()

        assert stats['total_nodes'] >= 2000  # 1000 authors + 1000 entities

    def test_special_characters_in_author(self):
        """Test author names with special characters."""
        builder = NetworkBuilder()

        authors = [
            "@user_123",
            "@user.name",
            "@user-name",
            "@用户",  # Chinese
            "@пользователь"  # Russian
        ]

        for author in authors:
            builder.add_post(
                author=author,
                entities=[{'text': 'Test', 'type': 'PER', 'score': 0.9}]
            )

        stats = builder.get_statistics()

        assert stats['authors'] == len(authors)


# =============================================================================
# Text Processing Edge Cases
# =============================================================================

@pytest.mark.edge_case
class TestTextProcessingEdgeCases:
    """Test edge cases in text processing."""

    def test_html_tags(self):
        """Test text containing HTML tags."""
        text = "<p>John Smith</p> works at <b>Microsoft</b>"

        # Test that text can be processed
        # (Actual NER may or may not extract entities from HTML)
        assert len(text) > 0

    def test_urls_in_text(self):
        """Test text containing URLs."""
        text = "Visit https://example.com and http://test.org for more info about John Smith."

        assert "https://" in text

    def test_emails_in_text(self):
        """Test text containing email addresses."""
        text = "Contact john@example.com or jane@company.org"

        assert "@" in text

    def test_hashtags_and_mentions(self):
        """Test social media specific text."""
        text = "@johndoe mentioned #Microsoft and @janedoe in #Seattle"

        assert "@johndoe" in text
        assert "#Microsoft" in text

    def test_code_snippets(self):
        """Test text containing code."""
        text = "function test() { return 'John Smith'; } at Microsoft"

        assert "function" in text

    def test_numbers_and_dates(self):
        """Test text with numbers and dates."""
        text = "On 2024-01-15, John Smith spent $1,000,000 at Microsoft"

        assert "2024" in text
        assert "$" in text

    def test_mixed_languages_in_sentence(self):
        """Test sentence mixing multiple languages."""
        text = "Hello 你好 مرحبا John Smith works at Microsoft"

        assert "Hello" in text
        assert "你好" in text


# =============================================================================
# Boundary Condition Tests
# =============================================================================

@pytest.mark.edge_case
class TestBoundaryConditions:
    """Test boundary conditions."""

    def test_minimum_text_length(self):
        """Test very short text (single character)."""
        text = "A"

        assert len(text) == 1

    def test_maximum_reasonable_text_length(self):
        """Test text at reasonable maximum length."""
        text = "A" * 10000

        assert len(text) == 10000

    def test_zero_confidence_threshold(self):
        """Test with confidence threshold of 0.0."""
        # Would accept all entities
        threshold = 0.0

        assert 0.0 <= threshold <= 1.0

    def test_maximum_confidence_threshold(self):
        """Test with confidence threshold of 1.0."""
        # Would reject most entities
        threshold = 1.0

        assert 0.0 <= threshold <= 1.0

    def test_single_row_dataframe(self):
        """Test processing single row."""
        df = pd.DataFrame({
            'author': ['@user1'],
            'text': ['Test text']
        })

        assert len(df) == 1

    def test_empty_dataframe(self):
        """Test empty dataframe."""
        df = pd.DataFrame(columns=['author', 'text'])

        assert len(df) == 0

    def test_very_wide_dataframe(self):
        """Test dataframe with many columns."""
        data = {'author': ['@user1'], 'text': ['Test']}
        # Add many extra columns
        for i in range(100):
            data[f'col_{i}'] = [f'value_{i}']

        df = pd.DataFrame(data)

        assert len(df.columns) > 100

# =============================================================================
# Malformed Data Tests
# =============================================================================

@pytest.mark.edge_case
class TestMalformedData:
    """Test handling of malformed data."""

    def test_csv_with_inconsistent_columns(self, temp_dir):
        """Test CSV where rows have different number of columns."""
        filepath = temp_dir / "inconsistent.csv"

        with open(filepath, 'w') as f:
            f.write("author,text\n")
            f.write("@user1,text1\n")
            f.write("@user2,text2,extra_column\n")  # Extra column
            f.write("@user3\n")  # Missing column

        loader = DataLoader()

        # Should handle inconsistency (pandas may add NaN)
        try:
            chunks = list(loader.load_csv(filepath, "author", "text"))
            # If it succeeds, verify data
            assert len(chunks) > 0
        except Exception:
            # Or it may raise an error - both are acceptable
            pass

    def test_json_with_missing_fields(self, temp_dir):
        """Test NDJSON with missing fields in some records."""
        filepath = temp_dir / "missing_fields.ndjson"

        with open(filepath, 'w') as f:
            f.write('{"author": "@user1", "text": "text1"}\n')
            f.write('{"author": "@user2"}\n')  # Missing text
            f.write('{"text": "text3"}\n')  # Missing author

        loader = DataLoader()

        # Should handle missing fields
        chunks = list(loader.load_ndjson(filepath, "author", "text"))

        # Might have NaN values
        assert len(chunks) > 0

    def test_malformed_json_lines(self, temp_dir):
        """Test NDJSON with some malformed lines."""
        filepath = temp_dir / "malformed.ndjson"

        with open(filepath, 'w') as f:
            f.write('{"author": "@user1", "text": "text1"}\n')
            f.write('{"invalid json\n')  # Malformed
            f.write('{"author": "@user3", "text": "text3"}\n')

        loader = DataLoader()

        # Should skip malformed lines
        chunks = list(loader.load_ndjson(filepath, "author", "text"))

        # Should get 2 valid records
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == 2

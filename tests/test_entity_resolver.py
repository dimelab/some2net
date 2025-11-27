"""
Unit tests for Entity Resolver module.

Tests simple normalized matching (no fuzzy matching).
"""

import pytest
from src.core.entity_resolver import EntityResolver


@pytest.fixture
def resolver():
    """Create fresh EntityResolver instance for each test."""
    return EntityResolver()


class TestNormalization:
    """Test text normalization functionality."""

    def test_lowercase_normalization(self, resolver):
        """Test that text is normalized to lowercase."""
        assert resolver.normalize_text("John Smith") == "john smith"
        assert resolver.normalize_text("JOHN SMITH") == "john smith"
        assert resolver.normalize_text("JoHn SmItH") == "john smith"

    def test_whitespace_normalization(self, resolver):
        """Test that extra whitespace is collapsed."""
        assert resolver.normalize_text("John  Smith") == "john smith"
        assert resolver.normalize_text("John   Smith") == "john smith"
        assert resolver.normalize_text("  John Smith  ") == "john smith"
        assert resolver.normalize_text("John\t\nSmith") == "john smith"

    def test_punctuation_stripping(self, resolver):
        """Test that leading/trailing punctuation is removed."""
        assert resolver.normalize_text("John Smith.") == "john smith"
        assert resolver.normalize_text("'John Smith'") == "john smith"
        assert resolver.normalize_text('"John Smith"') == "john smith"
        assert resolver.normalize_text("John Smith!") == "john smith"
        assert resolver.normalize_text("John Smith?") == "john smith"

    def test_combined_normalization(self, resolver):
        """Test combination of normalizations."""
        assert resolver.normalize_text("  JOHN  SMITH  ") == "john smith"
        assert resolver.normalize_text("'John Smith'.") == "john smith"
        assert resolver.normalize_text("  'JOHN  SMITH'!  ") == "john smith"

    def test_empty_string(self, resolver):
        """Test normalization of empty string."""
        assert resolver.normalize_text("") == ""
        assert resolver.normalize_text("   ") == ""

    def test_single_word(self, resolver):
        """Test normalization of single word."""
        assert resolver.normalize_text("Microsoft") == "microsoft"
        assert resolver.normalize_text("  MICROSOFT  ") == "microsoft"


class TestCanonicalForm:
    """Test canonical form resolution."""

    def test_first_occurrence_preserved(self, resolver):
        """Test that first occurrence's capitalization is preserved."""
        canonical1 = resolver.get_canonical_form("John Smith")
        assert canonical1 == "John Smith"

        # Second occurrence with different capitalization
        canonical2 = resolver.get_canonical_form("john smith")
        assert canonical2 == "John Smith"  # Should return first occurrence

        # Third occurrence
        canonical3 = resolver.get_canonical_form("JOHN SMITH")
        assert canonical3 == "John Smith"

    def test_whitespace_variations(self, resolver):
        """Test that whitespace variations map to same entity."""
        canonical1 = resolver.get_canonical_form("John Smith")
        canonical2 = resolver.get_canonical_form("John  Smith")  # Extra space
        canonical3 = resolver.get_canonical_form("  John Smith  ")  # Leading/trailing

        assert canonical1 == canonical2 == canonical3 == "John Smith"

    def test_punctuation_variations(self, resolver):
        """Test that punctuation variations map to same entity."""
        canonical1 = resolver.get_canonical_form("John Smith")
        canonical2 = resolver.get_canonical_form("John Smith.")
        canonical3 = resolver.get_canonical_form("'John Smith'")

        assert canonical1 == canonical2 == canonical3 == "John Smith"

    def test_different_entities(self, resolver):
        """Test that different entities remain different."""
        john = resolver.get_canonical_form("John Smith")
        jane = resolver.get_canonical_form("Jane Doe")
        microsoft = resolver.get_canonical_form("Microsoft")

        assert john != jane
        assert john != microsoft
        assert jane != microsoft

    def test_case_insensitive_matching(self, resolver):
        """Test case-insensitive matching."""
        # All these should map to the same canonical form
        forms = [
            "john smith",
            "John Smith",
            "JOHN SMITH",
            "JoHn SmItH",
            "John smith",
            "john SMITH"
        ]

        canonical_forms = [resolver.get_canonical_form(f) for f in forms]

        # All should be the same (the first one)
        assert len(set(canonical_forms)) == 1
        assert canonical_forms[0] == forms[0]  # First occurrence preserved

    def test_entity_map_update(self, resolver):
        """Test that entity map is updated correctly."""
        assert len(resolver.entity_map) == 0

        resolver.get_canonical_form("John Smith")
        assert len(resolver.entity_map) == 1

        resolver.get_canonical_form("john smith")  # Same entity
        assert len(resolver.entity_map) == 1

        resolver.get_canonical_form("Jane Doe")  # Different entity
        assert len(resolver.entity_map) == 2

    def test_multiple_words(self, resolver):
        """Test entities with multiple words."""
        canonical = resolver.get_canonical_form("New York City")
        assert canonical == "New York City"

        # Verify variations match
        assert resolver.get_canonical_form("new york city") == "New York City"
        assert resolver.get_canonical_form("NEW YORK CITY") == "New York City"


class TestAuthorMatching:
    """Test author-entity matching functionality."""

    def test_exact_match(self, resolver):
        """Test exact matches after normalization."""
        assert resolver.is_author_mention("johndoe", "John Doe") == True
        assert resolver.is_author_mention("John Smith", "john smith") == True
        assert resolver.is_author_mention("ALICE", "alice") == True

    def test_handle_matching(self, resolver):
        """Test matching with @ handles."""
        assert resolver.is_author_mention("@johndoe", "John Doe") == True
        assert resolver.is_author_mention("@johnsmith", "John Smith") == True
        assert resolver.is_author_mention("@alice", "Alice") == True

    def test_substring_matching(self, resolver):
        """Test substring matching."""
        # Entity contains author
        assert resolver.is_author_mention("John", "John Smith") == True

        # Author contains entity
        assert resolver.is_author_mention("John Smith", "John") == True
        assert resolver.is_author_mention("John Smith", "Smith") == True

    def test_handle_substring_matching(self, resolver):
        """Test substring matching with handles."""
        assert resolver.is_author_mention("@johnsmith", "John") == True
        assert resolver.is_author_mention("@johnsmith", "Smith") == True
        assert resolver.is_author_mention("@alice_wonder", "Alice") == True
        assert resolver.is_author_mention("@bob_the_builder", "Bob") == True

    def test_word_part_matching(self, resolver):
        """Test matching when entity is part of author name."""
        assert resolver.is_author_mention("Jane Smith", "Smith") == True
        assert resolver.is_author_mention("John Peter Doe", "Peter") == True
        assert resolver.is_author_mention("Mary Jane Watson", "Jane") == True

    def test_no_match(self, resolver):
        """Test cases that should not match."""
        assert resolver.is_author_mention("@johndoe", "Jane Smith") == False
        assert resolver.is_author_mention("@alice", "Bob") == False
        assert resolver.is_author_mention("@user123", "Microsoft") == False

    def test_short_word_no_match(self, resolver):
        """Test that very short words don't match loosely."""
        # Short words (<=3 chars) should only match if they're exact parts
        assert resolver.is_author_mention("@bob123", "Bo") == False
        assert resolver.is_author_mention("@alice", "Al") == False

    def test_partial_handle_match(self, resolver):
        """Test matching parts of handles."""
        # Should match if entity is substantial part of handle
        assert resolver.is_author_mention("@johnsmith99", "John") == True
        assert resolver.is_author_mention("@johnsmith99", "Smith") == True

    def test_case_insensitive_author_match(self, resolver):
        """Test case-insensitive author matching."""
        assert resolver.is_author_mention("JOHN SMITH", "john smith") == True
        assert resolver.is_author_mention("john smith", "JOHN SMITH") == True
        assert resolver.is_author_mention("@JohnDoe", "john doe") == True

    def test_whitespace_in_author_match(self, resolver):
        """Test author matching with whitespace variations."""
        assert resolver.is_author_mention("John  Smith", "John Smith") == True
        assert resolver.is_author_mention("  John Smith  ", "john smith") == True


class TestStatistics:
    """Test statistics functionality."""

    def test_initial_statistics(self, resolver):
        """Test statistics with no entities."""
        stats = resolver.get_statistics()
        assert stats['unique_entities'] == 0
        assert stats['normalized_forms'] == []

    def test_statistics_after_additions(self, resolver):
        """Test statistics after adding entities."""
        resolver.get_canonical_form("John Smith")
        resolver.get_canonical_form("Jane Doe")
        resolver.get_canonical_form("Microsoft")

        stats = resolver.get_statistics()
        assert stats['unique_entities'] == 3
        assert len(stats['normalized_forms']) == 3

    def test_statistics_with_duplicates(self, resolver):
        """Test that duplicates don't increase count."""
        resolver.get_canonical_form("John Smith")
        resolver.get_canonical_form("john smith")  # Duplicate
        resolver.get_canonical_form("JOHN SMITH")  # Duplicate

        stats = resolver.get_statistics()
        assert stats['unique_entities'] == 1

    def test_statistics_preview_limit(self, resolver):
        """Test that normalized_forms is limited to 10 items."""
        # Add more than 10 entities
        for i in range(15):
            resolver.get_canonical_form(f"Entity {i}")

        stats = resolver.get_statistics()
        assert stats['unique_entities'] == 15
        assert len(stats['normalized_forms']) <= 10


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_map(self, resolver):
        """Test that reset clears entity map."""
        resolver.get_canonical_form("John Smith")
        resolver.get_canonical_form("Jane Doe")

        assert len(resolver.entity_map) == 2

        resolver.reset()

        assert len(resolver.entity_map) == 0

    def test_after_reset_new_canonical(self, resolver):
        """Test that after reset, new canonical forms are created."""
        canonical1 = resolver.get_canonical_form("John Smith")
        assert canonical1 == "John Smith"

        resolver.reset()

        # After reset, lowercase version should become canonical
        canonical2 = resolver.get_canonical_form("john smith")
        assert canonical2 == "john smith"  # New canonical form


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_canonical(self, resolver):
        """Test canonical form of empty string."""
        canonical = resolver.get_canonical_form("")
        assert canonical == ""

        # Second empty string should map to same
        canonical2 = resolver.get_canonical_form("")
        assert canonical2 == ""

    def test_single_character(self, resolver):
        """Test single character entities."""
        canonical = resolver.get_canonical_form("A")
        assert canonical == "A"

        assert resolver.get_canonical_form("a") == "A"

    def test_numbers_in_entities(self, resolver):
        """Test entities containing numbers."""
        canonical = resolver.get_canonical_form("Boeing 747")
        assert canonical == "Boeing 747"

        assert resolver.get_canonical_form("boeing 747") == "Boeing 747"
        assert resolver.get_canonical_form("BOEING 747") == "Boeing 747"

    def test_special_characters(self, resolver):
        """Test entities with special characters."""
        # Hyphens, apostrophes, etc. in middle should be preserved
        canonical1 = resolver.get_canonical_form("O'Brien")
        canonical2 = resolver.get_canonical_form("o'brien")
        assert canonical2 == "O'Brien"

        canonical3 = resolver.get_canonical_form("Hewlett-Packard")
        canonical4 = resolver.get_canonical_form("hewlett-packard")
        assert canonical4 == "Hewlett-Packard"

    def test_unicode_characters(self, resolver):
        """Test entities with Unicode characters."""
        canonical1 = resolver.get_canonical_form("Zürich")
        canonical2 = resolver.get_canonical_form("zürich")
        assert canonical2 == "Zürich"

        canonical3 = resolver.get_canonical_form("São Paulo")
        canonical4 = resolver.get_canonical_form("são paulo")
        assert canonical4 == "São Paulo"

    def test_very_long_entity(self, resolver):
        """Test very long entity names."""
        long_name = "The International Business Machines Corporation"
        canonical = resolver.get_canonical_form(long_name)
        assert canonical == long_name

        assert resolver.get_canonical_form(long_name.lower()) == long_name


class TestRealisticScenarios:
    """Test realistic use cases."""

    def test_news_article_entities(self, resolver):
        """Test with entities from a news article."""
        entities = [
            "Barack Obama",  # First mention
            "President Obama",  # Different form
            "barack obama",  # Different case
            "Obama",  # Short form
        ]

        # Get canonical forms
        canonicals = [resolver.get_canonical_form(e) for e in entities]

        # Barack Obama and barack obama should match
        assert canonicals[0] == canonicals[2] == "Barack Obama"

        # But "President Obama" and "Obama" are different entities
        assert canonicals[1] == "President Obama"
        assert canonicals[3] == "Obama"

    def test_company_mentions(self, resolver):
        """Test with company names."""
        companies = [
            "Microsoft Corporation",
            "microsoft corporation",
            "MICROSOFT CORPORATION",
            "Microsoft",  # Short form - different entity
        ]

        canonicals = [resolver.get_canonical_form(c) for c in companies]

        # First three should match
        assert canonicals[0] == canonicals[1] == canonicals[2] == "Microsoft Corporation"

        # Short form is different
        assert canonicals[3] == "Microsoft"
        assert canonicals[3] != canonicals[0]

    def test_location_variations(self, resolver):
        """Test with location names."""
        locations = [
            "New York City",
            "new york city",
            "NEW YORK CITY",
            "New  York  City",  # Extra spaces
        ]

        canonicals = [resolver.get_canonical_form(loc) for loc in locations]

        # All should map to same entity
        assert len(set(canonicals)) == 1
        assert canonicals[0] == "New York City"

    def test_social_media_mentions(self, resolver):
        """Test author matching in social media context."""
        # User posts about themselves
        assert resolver.is_author_mention("@johnsmith", "John Smith") == True
        assert resolver.is_author_mention("@jane_doe", "Jane Doe") == True

        # User mentions another user
        assert resolver.is_author_mention("@johnsmith", "Jane Doe") == False

        # User with name in handle
        assert resolver.is_author_mention("@alice_cooper", "Alice") == True
        assert resolver.is_author_mention("@bob_dylan", "Dylan") == True


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_batch_entity_resolution(self, resolver):
        """Test resolving a batch of entities."""
        entities = [
            "John Smith", "jane doe", "JOHN SMITH", "Microsoft",
            "Jane Doe", "microsoft", "John  Smith", "MICROSOFT"
        ]

        canonicals = [resolver.get_canonical_form(e) for e in entities]

        # Count unique canonical forms
        unique = set(canonicals)
        assert len(unique) == 3  # John Smith, jane doe, Microsoft

        # Verify specific mappings
        assert canonicals[0] == canonicals[2] == canonicals[6] == "John Smith"
        assert canonicals[1] == canonicals[4] == "jane doe"
        assert canonicals[3] == canonicals[5] == canonicals[7] == "Microsoft"

    def test_combined_with_author_matching(self, resolver):
        """Test entity resolution combined with author matching."""
        # Resolve entities
        entity1 = resolver.get_canonical_form("John Smith")
        entity2 = resolver.get_canonical_form("jane doe")

        # Check author matching
        assert resolver.is_author_mention("@johnsmith", entity1) == True
        assert resolver.is_author_mention("@janedoe", entity2) == True
        assert resolver.is_author_mention("@johnsmith", entity2) == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

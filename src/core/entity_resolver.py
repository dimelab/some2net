"""Entity resolution with simple normalized matching (no fuzzy matching)."""
from typing import Dict, Set
import re


class EntityResolver:
    """Resolve and deduplicate named entities using simple normalized matching."""

    # Cross-language entity mappings (common locations and organizations)
    # Maps variants to canonical English form
    ENTITY_TRANSLATIONS = {
        # Countries
        'danmark': 'denmark',
        'tyskland': 'germany',
        'deutschland': 'germany',
        'frankrig': 'france',
        'sverige': 'sweden',
        'norge': 'norway',
        'finland': 'finland',
        'island': 'iceland',
        'spanien': 'spain',
        'italien': 'italy',
        'holland': 'netherlands',
        'nederlandene': 'netherlands',
        'belgien': 'belgium',
        'østrig': 'austria',
        'schweiz': 'switzerland',
        'polen': 'poland',
        'rusland': 'russia',
        'kina': 'china',
        'japan': 'japan',
        'indien': 'india',
        'usa': 'united states',
        'amerika': 'united states',
        'storbritannien': 'united kingdom',
        'england': 'england',
        'skotland': 'scotland',

        # Cities
        'københavn': 'copenhagen',
        'kobenhavn': 'copenhagen',
        'århus': 'aarhus',
        'aarhus': 'aarhus',
        'odense': 'odense',
        'berlin': 'berlin',
        'paris': 'paris',
        'london': 'london',
        'stockholm': 'stockholm',
        'oslo': 'oslo',
        'bruxelles': 'brussels',
        'bryssel': 'brussels',
        'wien': 'vienna',
        'moskva': 'moscow',

        # Regions
        'europa': 'europe',
        'eu': 'european union',
        'norden': 'nordic countries',
        'skandinavien': 'scandinavia',
    }

    def __init__(self):
        """Initialize resolver with simple matching only."""
        self.entity_map: Dict[str, str] = {}  # normalized -> canonical
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for matching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text (lowercase, collapsed whitespace)
        """
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation at start/end
        text = text.strip('.,!?;:\'"')
        
        return text
    
    def get_canonical_form(self, entity_text: str) -> str:
        """
        Get canonical form of entity using simple normalized matching with cross-language support.
        "John Smith" in post 1 = "john smith" in post 2 = "JOHN SMITH" in post 3
        "Danmark" = "Denmark" (cross-language)

        Args:
            entity_text: Entity text to resolve

        Returns:
            Canonical form (first occurrence with that normalized form)
        """
        normalized = self.normalize_text(entity_text)

        # Check for cross-language translation first
        if normalized in self.ENTITY_TRANSLATIONS:
            translated = self.ENTITY_TRANSLATIONS[normalized]
            # Use the translated form to look up or create canonical
            if translated in self.entity_map:
                return self.entity_map[translated]
            else:
                # First time seeing this entity - use capitalized English form
                canonical = translated.title()
                self.entity_map[translated] = canonical
                return canonical

        # Check for exact normalized match
        if normalized in self.entity_map:
            return self.entity_map[normalized]

        # New entity - use original text as canonical form
        # This preserves the original capitalization from first occurrence
        self.entity_map[normalized] = entity_text
        return entity_text
    
    def is_author_mention(self, author_name: str, entity_text: str) -> bool:
        """
        Check if entity matches an author name using stricter matching.

        Args:
            author_name: Author name/handle
            entity_text: Extracted entity text

        Returns:
            True if entity likely refers to author
        """
        # Normalize both
        norm_author = self.normalize_text(author_name)
        norm_entity = self.normalize_text(entity_text)

        # Check if entity has @ symbol (explicit @-mention)
        entity_has_at = entity_text.strip().startswith('@')

        # Remove @ symbol from handles for comparison
        norm_author = norm_author.lstrip('@')
        norm_entity = norm_entity.lstrip('@')

        # If entity has @, it's an explicit mention - check for exact match
        if entity_has_at:
            return norm_author == norm_entity

        # Exact match after normalization (without @)
        if norm_author == norm_entity:
            return True

        # STRICT: Only match if entity is the FULL author name
        # This prevents partial matches like "Liberal" matching "Liberal Alliance"
        # Author-to-author edges should only be created for explicit mentions

        # Check if author name is at start/end with word boundaries
        # e.g., "@johndoe" mentioned as "johndoe" or "John Doe"
        if norm_author.startswith(norm_entity + ' ') or norm_author.endswith(' ' + norm_entity):
            # Only if entity is substantial (>4 chars to avoid short false positives)
            if len(norm_entity) > 4:
                return True

        return False
    
    def reset(self):
        """Clear all cached entity mappings."""
        self.entity_map.clear()
    
    def get_statistics(self) -> Dict:
        """
        Get entity resolution statistics.
        
        Returns:
            Dictionary with resolution stats
        """
        return {
            'unique_entities': len(self.entity_map),
            'normalized_forms': list(self.entity_map.keys())[:10]  # First 10 for preview
        }


# Example usage
if __name__ == "__main__":
    resolver = EntityResolver()
    
    # Test normalization - same entity across posts
    print("=== Testing Simple Normalized Matching ===")
    entity1 = resolver.get_canonical_form("John Smith")
    print(f"Post 1: 'John Smith' -> '{entity1}'")
    
    entity2 = resolver.get_canonical_form("john smith")
    print(f"Post 2: 'john smith' -> '{entity2}'")
    
    entity3 = resolver.get_canonical_form("JOHN SMITH")
    print(f"Post 3: 'JOHN SMITH' -> '{entity3}'")
    
    entity4 = resolver.get_canonical_form("John  Smith")  # Extra space
    print(f"Post 4: 'John  Smith' -> '{entity4}'")
    
    # All should return the same canonical form
    print(f"\nAll are same entity: {entity1 == entity2 == entity3 == entity4}")
    
    # Test different entities
    print("\n=== Testing Different Entities ===")
    jane = resolver.get_canonical_form("Jane Doe")
    print(f"'Jane Doe' -> '{jane}'")
    print(f"Is same as John Smith: {jane == entity1}")
    
    # Test author matching
    print("\n=== Testing Author Matching ===")
    test_cases = [
        ("@johndoe", "John Doe", True),
        ("@johnsmith", "John Smith", True),
        ("Jane Smith", "Smith", True),
        ("@alice_wonder", "Alice", True),
        ("@bob123", "Robert", False),
        ("@charlie", "Charles Brown", False),
    ]
    
    for author, entity, expected in test_cases:
        result = resolver.is_author_mention(author, entity)
        status = "✅" if result == expected else "❌"
        print(f"{status} Author: '{author}', Entity: '{entity}' -> {result}")
    
    # Get statistics
    print("\n=== Statistics ===")
    stats = resolver.get_statistics()
    print(f"Unique entities: {stats['unique_entities']}")
    print(f"Sample normalized forms: {stats['normalized_forms']}")

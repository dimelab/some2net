"""
Test URL field detection functionality for finding expanded URLs.
"""

import pandas as pd
import pytest


def _find_expanded_url_field(df, current_text_col):
    """
    Find a field containing expanded/non-shortened URLs.
    (Copy of function from app.py for testing)
    """
    import re

    # Common URL shortener domains to detect
    url_shorteners = [
        't.co', 'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'short.link', 'tiny.cc', 'cli.gs',
        'pic.twitter.com', 'youtu.be', 'fb.me', 'amzn.to'
    ]

    # Pattern to match URLs
    url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)

    # Find candidate URL fields
    url_field_candidates = []

    for col in df.columns:
        col_lower = str(col).lower()

        # Skip the current text column (we want to find a BETTER one)
        if col == current_text_col:
            continue

        # Look for fields with 'url' in the name
        if 'url' in col_lower or 'link' in col_lower or 'expanded' in col_lower:
            url_field_candidates.append(col)

    # If no obvious candidates, check all fields
    if not url_field_candidates:
        url_field_candidates = [col for col in df.columns if col != current_text_col]

    # Score each candidate
    best_field = None
    best_score = -1

    # Sample up to 100 rows for efficiency
    sample_df = df.head(100)

    for col in url_field_candidates:
        try:
            # Get non-null values and convert to string
            values = sample_df[col].dropna().astype(str)

            if len(values) == 0:
                continue

            # Count URLs and shortened URLs
            total_urls = 0
            shortened_urls = 0
            total_url_length = 0

            for val in values:
                urls = url_pattern.findall(val)
                if urls:
                    total_urls += len(urls)
                    for url in urls:
                        total_url_length += len(url)
                        # Check if it's a shortened URL
                        if any(shortener in url.lower() for shortener in url_shorteners):
                            shortened_urls += 1

            if total_urls == 0:
                continue

            # Calculate score based on:
            # 1. Ratio of non-shortened URLs (higher is better)
            # 2. Average URL length (longer is better for expanded URLs)
            # 3. Presence of 'url' in field name (bonus)

            non_shortened_ratio = (total_urls - shortened_urls) / total_urls
            avg_url_length = total_url_length / total_urls

            # Normalize avg_url_length (typical shortened URL ~20-30 chars, expanded ~60-200 chars)
            length_score = min(avg_url_length / 100.0, 1.0)

            # Field name bonus
            col_lower = str(col).lower()
            name_bonus = 0.2 if any(word in col_lower for word in ['expand', 'full', 'unshorten']) else 0.1 if 'url' in col_lower else 0

            # Combined score
            score = (non_shortened_ratio * 0.5) + (length_score * 0.4) + name_bonus

            if score > best_score:
                best_score = score
                best_field = col

        except Exception:
            continue

    # Only return if we found a significantly better field (score > 0.3)
    if best_score > 0.3:
        return best_field

    return None


class TestURLFieldDetection:
    """Test URL field detection functionality."""

    def test_detects_expanded_url_field(self):
        """Test that function detects field with expanded URLs."""
        df = pd.DataFrame({
            'text': [
                'Check this out https://t.co/abc123',
                'Another link https://t.co/def456',
                'See https://t.co/ghi789'
            ],
            'expanded_url': [
                'https://www.example.com/article/full-title-here',
                'https://www.example.com/another/long/path/to/content',
                'https://www.example.com/news/story/details'
            ],
            'author': ['user1', 'user2', 'user3']
        })

        result = _find_expanded_url_field(df, 'text')
        assert result == 'expanded_url'

    def test_prefers_field_with_url_in_name(self):
        """Test that function prefers fields with 'url' in the name."""
        df = pd.DataFrame({
            'text': ['https://t.co/abc'],
            'some_field': ['https://www.example.com/long/path/here'],
            'url_field': ['https://www.example.com/another/long/path']
        })

        result = _find_expanded_url_field(df, 'text')
        # Should pick url_field due to name bonus
        assert result == 'url_field'

    def test_returns_none_when_no_better_field(self):
        """Test that function returns None when no better field exists."""
        df = pd.DataFrame({
            'text': ['https://www.example.com/full/url'],
            'author': ['user1'],
            'timestamp': ['2024-01-01']
        })

        result = _find_expanded_url_field(df, 'text')
        assert result is None

    def test_ignores_current_text_column(self):
        """Test that function doesn't return the current text column."""
        df = pd.DataFrame({
            'text': ['https://www.example.com/full/url'],
            'text2': ['https://www.example.com/another/url']
        })

        result = _find_expanded_url_field(df, 'text')
        # Should not return 'text' even if it has good URLs
        assert result != 'text'

    def test_handles_mixed_shortened_and_expanded(self):
        """Test that function prefers field with more expanded URLs."""
        df = pd.DataFrame({
            'text': [
                'https://t.co/abc',
                'https://t.co/def',
                'https://www.example.com/path'
            ],
            'urls': [
                'https://www.example.com/article/one',
                'https://www.example.com/article/two',
                'https://www.example.com/article/three'
            ]
        })

        result = _find_expanded_url_field(df, 'text')
        assert result == 'urls'

    def test_handles_numeric_columns_gracefully(self):
        """Test that function handles numeric columns gracefully (converts to string)."""
        df = pd.DataFrame({
            'text': ['https://t.co/abc'],
            'expanded_urls': ['https://www.example.com/long/path'],
            'numeric_col': [123],
            'bool_col': [True]
        })

        result = _find_expanded_url_field(df, 'text')
        # Should still find expanded_urls despite numeric columns present
        assert result == 'expanded_urls'

    def test_handles_empty_or_null_values(self):
        """Test that function handles empty or null values gracefully."""
        df = pd.DataFrame({
            'text': ['https://t.co/abc', 'https://t.co/def', 'https://t.co/ghi', None, ''],
            'expanded_url': [
                'https://www.example.com/path/to/article',
                'https://www.example.com/path2/content',
                'https://www.example.com/path3/story',
                None,
                None
            ]
        })

        result = _find_expanded_url_field(df, 'text')
        assert result == 'expanded_url'

    def test_detects_common_url_shorteners(self):
        """Test that function recognizes common URL shorteners."""
        shorteners = ['t.co', 'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly']

        for shortener in shorteners:
            df = pd.DataFrame({
                'text': [f'https://{shortener}/abc123'],
                'full_url': ['https://www.example.com/full/path/here']
            })

            result = _find_expanded_url_field(df, 'text')
            assert result == 'full_url', f"Failed to detect {shortener} as shortened URL"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

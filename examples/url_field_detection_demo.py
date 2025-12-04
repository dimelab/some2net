"""
Demo: URL Field Detection for Domain Extraction

This demonstrates how the automatic URL field detection works.
It finds fields containing expanded URLs instead of shortened ones (e.g., t.co links).
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the function from CLI
import re


def _find_expanded_url_field(df, current_text_col):
    """Find a field containing expanded/non-shortened URLs."""
    # Common URL shortener domains to detect
    url_shorteners = [
        't.co', 'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'short.link', 'tiny.cc', 'cli.gs',
        'pic.twitter.com', 'youtu.be', 'fb.me', 'amzn.to'
    ]

    url_pattern = re.compile(r'https?://[^\s]+', re.IGNORECASE)
    url_field_candidates = []

    for col in df.columns:
        col_lower = str(col).lower()
        if col == current_text_col:
            continue
        if 'url' in col_lower or 'link' in col_lower or 'expanded' in col_lower:
            url_field_candidates.append(col)

    if not url_field_candidates:
        url_field_candidates = [col for col in df.columns if col != current_text_col]

    best_field = None
    best_score = -1
    sample_df = df.head(100)

    for col in url_field_candidates:
        try:
            values = sample_df[col].dropna().astype(str)
            if len(values) == 0:
                continue

            total_urls = 0
            shortened_urls = 0
            total_url_length = 0

            for val in values:
                urls = url_pattern.findall(val)
                if urls:
                    total_urls += len(urls)
                    for url in urls:
                        total_url_length += len(url)
                        if any(shortener in url.lower() for shortener in url_shorteners):
                            shortened_urls += 1

            if total_urls == 0:
                continue

            non_shortened_ratio = (total_urls - shortened_urls) / total_urls
            avg_url_length = total_url_length / total_urls
            length_score = min(avg_url_length / 100.0, 1.0)
            col_lower = str(col).lower()
            name_bonus = 0.2 if any(word in col_lower for word in ['expand', 'full', 'unshorten']) else 0.1 if 'url' in col_lower else 0
            score = (non_shortened_ratio * 0.5) + (length_score * 0.4) + name_bonus

            if score > best_score:
                best_score = score
                best_field = col

        except Exception:
            continue

    if best_score > 0.3:
        return best_field
    return None


def main():
    """Demonstrate URL field detection."""

    print("=" * 70)
    print("URL FIELD DETECTION DEMO")
    print("=" * 70)

    # Example 1: Twitter data with shortened URLs
    print("\n📋 Example 1: Twitter data with t.co links")
    print("-" * 70)

    twitter_df = pd.DataFrame({
        'author': ['@user1', '@user2', '@user3'],
        'text': [
            'Check this out https://t.co/abc123def',
            'Great article https://t.co/xyz789ghi',
            'Must read https://t.co/qwe456rty'
        ],
        'expanded_urls': [
            'https://www.nytimes.com/2024/03/15/technology/ai-breakthrough-research.html',
            'https://www.scientificamerican.com/article/climate-change-impact-2024/',
            'https://www.nature.com/articles/nature-study-quantum-computing'
        ],
        'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03']
    })

    print("\nDataFrame columns:")
    for col in twitter_df.columns:
        print(f"  - {col}")

    print("\nCurrent text column: 'text'")
    print("Sample text field values (shortened URLs):")
    for val in twitter_df['text'][:2]:
        print(f"  • {val}")

    result = _find_expanded_url_field(twitter_df, 'text')

    if result:
        print(f"\n✅ Auto-detected URL field: '{result}'")
        print(f"\nSample {result} field values (expanded URLs):")
        for val in twitter_df[result][:2]:
            print(f"  • {val}")
    else:
        print("\n⚠️ No better URL field found")

    # Example 2: Data with multiple URL fields
    print("\n\n📋 Example 2: Multiple URL fields")
    print("-" * 70)

    multi_url_df = pd.DataFrame({
        'author': ['@user1', '@user2'],
        'text': ['Post with link https://t.co/abc', 'Another post https://t.co/def'],
        'short_url': ['https://bit.ly/xyz123', 'https://bit.ly/abc456'],
        'full_url': [
            'https://www.example.com/blog/post/full-article-title-here',
            'https://www.example.com/news/story/detailed-content-path'
        ],
        'metadata': ['info1', 'info2']
    })

    print("\nDataFrame columns:")
    for col in multi_url_df.columns:
        print(f"  - {col}")

    result = _find_expanded_url_field(multi_url_df, 'text')

    if result:
        print(f"\n✅ Auto-detected best URL field: '{result}'")
        print("(Prefers 'full_url' over 'short_url' due to longer URLs)")
    else:
        print("\n⚠️ No better URL field found")

    # Example 3: Data already has expanded URLs in text field
    print("\n\n📋 Example 3: Text field already contains expanded URLs")
    print("-" * 70)

    expanded_df = pd.DataFrame({
        'author': ['@user1', '@user2'],
        'text': [
            'Great article https://www.longdomainname.com/path/to/article',
            'Check out https://www.example.com/blog/post/title-here'
        ],
        'other_field': ['data1', 'data2']
    })

    result = _find_expanded_url_field(expanded_df, 'text')

    if result:
        print(f"\n✅ Found alternative field: '{result}'")
    else:
        print("\n✅ Current 'text' field already contains expanded URLs")
        print("No need to switch fields!")

    # Summary
    print("\n" + "=" * 70)
    print("HOW IT WORKS")
    print("=" * 70)
    print("""
The URL field detection algorithm:

1. 🔍 Searches for fields with 'url', 'link', or 'expanded' in the name
2. 📊 Analyzes URL content:
   - Detects common URL shorteners (t.co, bit.ly, etc.)
   - Measures average URL length
   - Counts ratio of expanded vs shortened URLs
3. 🎯 Scores each field based on:
   - Non-shortened URL ratio (50% weight)
   - Average URL length (40% weight)
   - Field name quality (10% weight)
4. ✅ Selects the field with the highest score (if > 0.3)

Benefits:
• Works with files that have 200+ columns
• Automatically finds the best URL field
• Only switches if a significantly better field exists
• Falls back gracefully if no better field is found
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()

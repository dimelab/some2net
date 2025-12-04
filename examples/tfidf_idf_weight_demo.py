"""
Example: Demonstrating TF-IDF IDF Weight Coefficient

This example shows how the IDF weight coefficient affects keyword ranking:
- Low IDF weight (0.0): Emphasizes frequent terms (pure TF)
- Standard IDF weight (1.0): Balanced approach (standard TF-IDF)
- High IDF weight (>1.0): Emphasizes distinctive/rare terms

The IDF weight coefficient controls the formula: TF * (IDF ^ weight)
"""

from src.core.extractors import KeywordExtractor


def main():
    """Demonstrate IDF weight coefficient effects."""

    print("=" * 70)
    print("TF-IDF IDF WEIGHT COEFFICIENT DEMONSTRATION")
    print("=" * 70)

    # Sample data: Two developers with different specializations
    # User1 uses "python" frequently (distinctive to them)
    # Both users use "programming" and "code" (common terms)

    python_dev_texts = [
        "python python python programming language",
        "python code development tools",
        "python programming best practices",
        "python web frameworks django flask"
    ]

    java_dev_texts = [
        "java programming language features",
        "java code implementation patterns",
        "java enterprise development",
        "javascript web programming"
    ]

    print("\nData:")
    print("  Python Dev: Frequently mentions 'python' (distinctive)")
    print("  Java Dev:   Frequently mentions 'java' (distinctive)")
    print("  Both:       Use 'programming', 'code', 'development' (common)")

    # Test different IDF weight values
    idf_weights = [
        (0.0, "Pure TF", "Emphasizes frequent terms, ignores distinctiveness"),
        (0.5, "Low IDF", "Slight emphasis on distinctive terms"),
        (1.0, "Standard", "Balanced TF-IDF (default)"),
        (1.5, "High IDF", "Strong emphasis on distinctive terms"),
        (2.0, "Very High", "Maximum emphasis on distinctive terms")
    ]

    for weight, name, description in idf_weights:
        print(f"\n{'=' * 70}")
        print(f"IDF Weight: {weight} ({name})")
        print(f"{description}")
        print(f"{'=' * 70}")

        extractor = KeywordExtractor(
            method='tfidf',
            min_keywords=5,
            max_keywords=8,
            include_bigrams=False,
            idf_weight=weight
        )

        extractor.collect_texts("@python_dev", python_dev_texts)
        extractor.collect_texts("@java_dev", java_dev_texts)

        results = extractor.extract_all_authors(show_progress=False)

        # Show results for Python developer
        print("\nPython Developer's top keywords:")
        for i, kw in enumerate(results['@python_dev'], 1):
            word_type = "DISTINCTIVE" if kw['text'] in ['python', 'django', 'flask'] else "COMMON" if kw['text'] in ['programming', 'code', 'development'] else "NEUTRAL"
            print(f"  {i}. {kw['text']:15s} (score: {kw['score']:.3f}) [{word_type}]")

        print("\nJava Developer's top keywords:")
        for i, kw in enumerate(results['@java_dev'], 1):
            word_type = "DISTINCTIVE" if kw['text'] in ['java', 'javascript', 'enterprise'] else "COMMON" if kw['text'] in ['programming', 'code', 'development'] else "NEUTRAL"
            print(f"  {i}. {kw['text']:15s} (score: {kw['score']:.3f}) [{word_type}]")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("\nHow to choose IDF weight:")
    print("  • Use 0.0 when you want the most frequent terms")
    print("    (Good for: finding what each author talks about most)")
    print()
    print("  • Use 1.0 for standard TF-IDF (default, balanced)")
    print("    (Good for: general-purpose keyword extraction)")
    print()
    print("  • Use 1.5-2.0 when you want distinctive/unique terms")
    print("    (Good for: finding what makes each author different)")
    print()
    print("Key insight:")
    print("  - Low weight → Common frequent terms rank higher")
    print("  - High weight → Rare distinctive terms rank higher")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

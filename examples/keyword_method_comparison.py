"""
Example: Comparing RAKE and TF-IDF Keyword Extraction Methods

This example demonstrates the difference between the two keyword extraction methods:
- RAKE: Extracts multi-word phrases (e.g., "machine learning", "deep learning")
- TF-IDF: Extracts individual words weighted by importance

Use RAKE when you want to capture compound concepts and phrases.
Use TF-IDF when you want individual keywords with clear statistical significance.
"""

from src.core.extractors import KeywordExtractor


def main():
    """Compare RAKE and TF-IDF keyword extraction."""

    print("=" * 70)
    print("KEYWORD EXTRACTION METHOD COMPARISON")
    print("=" * 70)

    # Sample data about two different topics
    ml_texts = [
        "Machine learning algorithms are transforming data science and analytics",
        "Deep learning neural networks excel at pattern recognition tasks",
        "Natural language processing enables computers to understand human text",
        "Machine learning requires quality training data and feature engineering"
    ]

    climate_texts = [
        "Climate change poses serious environmental challenges globally",
        "Renewable energy sources reduce carbon emissions significantly",
        "Sustainable development balances economic and environmental needs",
        "Global warming affects weather patterns and sea levels"
    ]

    # Test both methods (skip RAKE if not installed)
    methods_to_test = []

    # Check if RAKE is available
    try:
        test_extractor = KeywordExtractor(method='rake', min_keywords=1, max_keywords=1)
        methods_to_test.append('rake')
        del test_extractor
    except ImportError:
        print("\nNote: RAKE method skipped (rake-nltk not installed)")
        print("Install with: pip install rake-nltk\n")

    # TF-IDF is always available
    methods_to_test.append('tfidf')

    for method in methods_to_test:
        print(f"\n{'=' * 70}")
        print(f"METHOD: {method.upper()}")
        print(f"{'=' * 70}")

        if method == 'rake':
            print("\nRAKE extracts multi-word phrases based on co-occurrence patterns.")
            print("Good for: Capturing compound concepts, domain-specific terminology")
            extractor = KeywordExtractor(
                method='rake',
                min_keywords=5,
                max_keywords=8,
                max_phrase_length=3,  # Allow up to 3-word phrases
                use_tfidf=True        # Apply TF-IDF weighting for distinctiveness
            )
        else:
            print("\nTF-IDF extracts individual words weighted by statistical importance.")
            print("Good for: Finding distinctive terms, cross-author comparison")
            extractor = KeywordExtractor(
                method='tfidf',
                min_keywords=5,
                max_keywords=8
            )

        # Collect texts for two authors
        extractor.collect_texts("@ml_researcher", ml_texts)
        extractor.collect_texts("@climate_scientist", climate_texts)

        # Extract keywords
        results = extractor.extract_all_authors(show_progress=False)

        # Display results
        for author, keywords in results.items():
            print(f"\n{author}:")
            for i, kw in enumerate(keywords, 1):
                print(f"  {i}. {kw['text']:35s} (score: {kw['score']:.3f})")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("\nRAKE:")
    print("  ✓ Extracts meaningful phrases (2-3 words)")
    print("  ✓ Better for understanding topics and concepts")
    print("  ✓ More intuitive for humans to read")
    print("  ✓ Can capture compound terms ('machine learning', 'climate change')")
    print("  ✗ Requires rake-nltk package")

    print("\nTF-IDF:")
    print("  ✓ Extracts individual words")
    print("  ✓ Statistically sound (classic IR approach)")
    print("  ✓ Better for finding distinctive/unique terms per author")
    print("  ✓ No external dependencies (built-in)")
    print("  ✗ Misses compound concepts")

    print("\nRecommendations:")
    print("  - Use RAKE for topic modeling and phrase extraction")
    print("  - Use TF-IDF for author profiling and comparative analysis")
    print("  - Use RAKE with use_tfidf=True for best of both worlds")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()

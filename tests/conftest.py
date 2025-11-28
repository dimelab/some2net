"""
Pytest configuration and shared fixtures.

This file contains fixtures that are available to all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import json


# =============================================================================
# Temporary Directories
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    tmpdir = tempfile.mkdtemp(prefix="sna_cache_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    tmpdir = tempfile.mkdtemp(prefix="sna_output_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# Test Data - English
# =============================================================================

@pytest.fixture
def sample_texts_english():
    """Sample English texts for testing."""
    return [
        "John Smith works at Microsoft in Seattle.",
        "Angela Merkel visited Paris last week.",
        "Apple Inc. announced a new product in California.",
        "The meeting will be held in New York City.",
        "Google opened a new office in London."
    ]


@pytest.fixture
def sample_csv_data_english():
    """Sample CSV data in English."""
    return {
        'author': ['@user1', '@user2', '@user3', '@user4', '@user5'],
        'text': [
            "John Smith works at Microsoft in Seattle.",
            "Angela Merkel visited Paris last week.",
            "Apple Inc. announced a new product in California.",
            "The meeting will be held in New York City.",
            "Google opened a new office in London."
        ],
        'post_id': ['1', '2', '3', '4', '5'],
        'timestamp': [
            '2024-01-01', '2024-01-02', '2024-01-03',
            '2024-01-04', '2024-01-05'
        ]
    }


# =============================================================================
# Test Data - Danish
# =============================================================================

@pytest.fixture
def sample_texts_danish():
    """Sample Danish texts for testing."""
    return [
        "Statsministeren mødtes med embedsmænd i København.",
        "Mette Frederiksen besøgte Aarhus i går.",
        "Danske Bank åbnede en ny filial i Odense.",
        "Konferencen afholdes i Aalborg næste uge.",
        "Carlsberg producerer øl i Danmark."
    ]


@pytest.fixture
def sample_csv_data_danish():
    """Sample CSV data in Danish."""
    return {
        'author': ['@bruger1', '@bruger2', '@bruger3', '@bruger4', '@bruger5'],
        'text': [
            "Statsministeren mødtes med embedsmænd i København.",
            "Mette Frederiksen besøgte Aarhus i går.",
            "Danske Bank åbnede en ny filial i Odense.",
            "Konferencen afholdes i Aalborg næste uge.",
            "Carlsberg producerer øl i Danmark."
        ],
        'post_id': ['1', '2', '3', '4', '5'],
        'timestamp': [
            '2024-01-01', '2024-01-02', '2024-01-03',
            '2024-01-04', '2024-01-05'
        ]
    }


# =============================================================================
# Test Data - Multilingual
# =============================================================================

@pytest.fixture
def sample_csv_data_multilingual():
    """Sample CSV data with mixed languages."""
    return {
        'author': ['@user1', '@user2', '@user3', '@user4', '@user5', '@user6'],
        'text': [
            "John Smith works at Microsoft in Seattle.",  # English
            "Statsministeren mødtes med embedsmænd i København.",  # Danish
            "Angela Merkel besuchte Paris letzte Woche.",  # German
            "Apple Inc. lancerede et nyt produkt i Californien.",  # Danish
            "Google åbnede et nyt kontor i London.",  # Danish
            "The Prime Minister of Denmark spoke in Brussels."  # English
        ],
        'post_id': ['1', '2', '3', '4', '5', '6'],
        'timestamp': [
            '2024-01-01', '2024-01-02', '2024-01-03',
            '2024-01-04', '2024-01-05', '2024-01-06'
        ]
    }


# =============================================================================
# Test Data - Edge Cases
# =============================================================================

@pytest.fixture
def edge_case_texts():
    """Edge case texts for testing."""
    return {
        'empty': "",
        'whitespace': "   \n\t  ",
        'special_chars': "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        'unicode': "Héllo Wörld 你好世界 مرحبا العالم",
        'very_long': "A" * 10000,
        'html': "<p>John Smith</p> works at <b>Microsoft</b>",
        'urls': "Check out https://example.com and http://test.org",
        'emails': "Contact john@example.com or jane@test.org",
        'numbers_only': "123 456 789 000",
        'mixed': "User@123 mentioned CEO#456 at Company$789"
    }


@pytest.fixture
def malformed_csv_data():
    """Malformed CSV data for error testing."""
    return [
        "author,text\n",  # Header only
        "author,text\n@user1,",  # Missing text
        "author,text\n,Some text without author",  # Missing author
        "wrong_col1,wrong_col2\n@user1,text",  # Wrong columns
        "author,text\n@user1,\"Unclosed quote",  # Malformed quote
    ]


# =============================================================================
# File Fixtures
# =============================================================================

@pytest.fixture
def create_csv_file(temp_dir):
    """Factory fixture to create CSV files."""
    def _create(data, filename="test.csv"):
        filepath = temp_dir / filename
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath
    return _create


@pytest.fixture
def create_ndjson_file(temp_dir):
    """Factory fixture to create NDJSON files."""
    def _create(data, filename="test.ndjson"):
        filepath = temp_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            # Assume data is a dict with lists
            num_records = len(data[list(data.keys())[0]])
            for i in range(num_records):
                record = {key: data[key][i] for key in data.keys()}
                f.write(json.dumps(record) + '\n')
        return filepath
    return _create


@pytest.fixture
def sample_csv_file(temp_dir, sample_csv_data_english, create_csv_file):
    """Pre-created CSV file with English data."""
    return create_csv_file(sample_csv_data_english)


@pytest.fixture
def sample_ndjson_file(temp_dir, sample_csv_data_english, create_ndjson_file):
    """Pre-created NDJSON file with English data."""
    return create_ndjson_file(sample_csv_data_english)


@pytest.fixture
def sample_danish_csv(temp_dir, sample_csv_data_danish, create_csv_file):
    """Pre-created CSV file with Danish data."""
    return create_csv_file(sample_csv_data_danish, "danish.csv")


@pytest.fixture
def sample_multilingual_csv(temp_dir, sample_csv_data_multilingual, create_csv_file):
    """Pre-created CSV file with multilingual data."""
    return create_csv_file(sample_csv_data_multilingual, "multilingual.csv")


# =============================================================================
# Large Data Fixtures
# =============================================================================

@pytest.fixture
def large_csv_file(temp_dir):
    """Create large CSV file for performance testing."""
    filepath = temp_dir / "large.csv"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("author,text,post_id\n")
        for i in range(1000):
            author = f"@user{i % 100}"
            text = f"This is test post number {i} mentioning John Smith and Microsoft."
            f.write(f"{author},{text},post_{i}\n")

    return filepath


@pytest.fixture
def very_large_csv_file(temp_dir):
    """Create very large CSV file for stress testing."""
    filepath = temp_dir / "very_large.csv"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("author,text,post_id\n")
        for i in range(10000):
            author = f"@user{i % 500}"
            text = f"Post {i}: Meeting in Copenhagen with Jane Doe from Google."
            f.write(f"{author},{text},post_{i}\n")

    return filepath


# =============================================================================
# Expected Results
# =============================================================================

@pytest.fixture
def expected_entities_english():
    """Expected entities from English sample texts."""
    return {
        0: [
            {'text': 'John Smith', 'type': 'PER'},
            {'text': 'Microsoft', 'type': 'ORG'},
            {'text': 'Seattle', 'type': 'LOC'}
        ],
        1: [
            {'text': 'Angela Merkel', 'type': 'PER'},
            {'text': 'Paris', 'type': 'LOC'}
        ],
        2: [
            {'text': 'Apple Inc.', 'type': 'ORG'},
            {'text': 'California', 'type': 'LOC'}
        ]
        # Note: Actual results may vary based on model and confidence threshold
    }


# =============================================================================
# Mock Objects
# =============================================================================

@pytest.fixture
def mock_ner_results():
    """Mock NER results for testing without model."""
    return [
        [
            {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
            {'text': 'Microsoft', 'type': 'ORG', 'score': 0.92}
        ],
        [
            {'text': 'Angela Merkel', 'type': 'PER', 'score': 0.96},
            {'text': 'Paris', 'type': 'LOC', 'score': 0.89}
        ],
        [
            {'text': 'Apple Inc.', 'type': 'ORG', 'score': 0.94},
            {'text': 'California', 'type': 'LOC', 'score': 0.88}
        ]
    ]


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Pytest configuration hook."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "large" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark tests requiring NER model
        if "ner_engine" in str(item.fspath):
            item.add_marker(pytest.mark.requires_model)


# =============================================================================
# Session-scoped Fixtures (Expensive Resources)
# =============================================================================

@pytest.fixture(scope="session")
def ner_engine_session():
    """
    Session-scoped NER engine to avoid reloading model for each test.
    Only use for tests that don't modify the engine state.
    """
    import tempfile
    from src.core.ner_engine import NEREngine

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = NEREngine(
            enable_cache=False,  # Disable cache for testing
            confidence_threshold=0.80
        )
        yield engine

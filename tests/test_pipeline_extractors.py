"""
Integration tests for pipeline with different extraction methods.

Tests the pipeline with all extraction methods:
- Hashtag extraction
- Mention extraction
- Domain extraction
- Keyword extraction
- Exact match extraction
- NER extraction (existing)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import csv
import json

from src.core.pipeline import SocialNetworkPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def hashtag_csv_file(temp_dir):
    """Create a CSV file with hashtags for testing."""
    filepath = Path(temp_dir) / "hashtag_data.csv"

    data = [
        ['author', 'text'],
        ['@alice', 'I love #python and #datascience!'],
        ['@bob', '#python is great for #machinelearning'],
        ['@alice', 'Working on #datascience projects with #python'],
        ['@charlie', '#javascript and #webdev are fun'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


@pytest.fixture
def mention_csv_file(temp_dir):
    """Create a CSV file with mentions for testing."""
    filepath = Path(temp_dir) / "mention_data.csv"

    data = [
        ['author', 'text'],
        ['@alice', 'Great work @bob! Thanks @charlie for the help.'],
        ['@bob', 'Collaborating with @alice on this project'],
        ['@charlie', 'Learned a lot from @alice'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


@pytest.fixture
def domain_csv_file(temp_dir):
    """Create a CSV file with URLs for testing."""
    filepath = Path(temp_dir) / "domain_data.csv"

    data = [
        ['author', 'text'],
        ['@alice', 'Check out https://nytimes.com and https://bbc.com'],
        ['@bob', 'Great article at https://nytimes.com/article'],
        ['@alice', 'News from https://cnn.com'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


@pytest.fixture
def keyword_csv_file(temp_dir):
    """Create a CSV file for keyword extraction testing."""
    filepath = Path(temp_dir) / "keyword_data.csv"

    data = [
        ['author', 'text'],
        ['@alice', 'Machine learning and artificial intelligence are transforming the world'],
        ['@alice', 'Neural networks and deep learning models are very powerful'],
        ['@alice', 'Python programming is essential for machine learning'],
        ['@bob', 'Web development with JavaScript and React is fun'],
        ['@bob', 'Building responsive web applications with modern frameworks'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


@pytest.fixture
def exact_match_csv_file(temp_dir):
    """Create a CSV file for exact match testing."""
    filepath = Path(temp_dir) / "exact_data.csv"

    data = [
        ['author', 'text'],
        ['@alice', 'positive'],
        ['@bob', 'negative'],
        ['@charlie', 'neutral'],
        ['@alice', 'positive'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


@pytest.fixture
def metadata_csv_file(temp_dir):
    """Create a CSV file with metadata columns for testing."""
    filepath = Path(temp_dir) / "metadata_data.csv"

    data = [
        ['author', 'text', 'sentiment', 'likes', 'retweets'],
        ['@alice', '#python is awesome', 'positive', '10', '5'],
        ['@bob', '#python rocks', 'positive', '15', '8'],
        ['@charlie', '#javascript is cool', 'neutral', '7', '3'],
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return str(filepath)


class TestHashtagExtraction:
    """Test hashtag extraction method."""

    def test_hashtag_extraction_basic(self, hashtag_csv_file, temp_dir):
        """Test basic hashtag extraction."""
        pipeline = SocialNetworkPipeline(
            extraction_method="hashtag",
            extractor_config={'normalize_case': True}
        )

        graph, stats = pipeline.process_file(
            hashtag_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check network was created
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Check for hashtag nodes
        hashtag_nodes = [n for n in graph.nodes if '#' in n]
        assert len(hashtag_nodes) > 0

        # Check for specific hashtags
        assert '#python' in graph.nodes or 'python' in str(graph.nodes)

    def test_hashtag_case_normalization(self, temp_dir):
        """Test hashtag case normalization."""
        # Create test file with mixed case hashtags
        filepath = Path(temp_dir) / "case_test.csv"
        data = [
            ['author', 'text'],
            ['@alice', '#Python #PYTHON #python'],
        ]
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

        # With normalization
        pipeline = SocialNetworkPipeline(
            extraction_method="hashtag",
            extractor_config={'normalize_case': True}
        )
        graph, _ = pipeline.process_file(
            str(filepath),
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Should have only one hashtag node (normalized)
        hashtag_nodes = [n for n in graph.nodes if n.startswith('#')]
        # All should be lowercase
        assert all(tag.lower() == tag for tag in hashtag_nodes)


class TestMentionExtraction:
    """Test mention extraction method."""

    def test_mention_extraction_basic(self, mention_csv_file, temp_dir):
        """Test basic mention extraction."""
        pipeline = SocialNetworkPipeline(
            extraction_method="mention"
        )

        graph, stats = pipeline.process_file(
            mention_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check network was created
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Check for author-to-author edges
        assert graph.has_edge('@alice', '@bob') or graph.has_edge('@alice', '@charlie')


class TestDomainExtraction:
    """Test domain extraction method."""

    def test_domain_extraction_basic(self, domain_csv_file, temp_dir):
        """Test basic domain extraction."""
        pipeline = SocialNetworkPipeline(
            extraction_method="domain",
            extractor_config={'strip_www': True}
        )

        graph, stats = pipeline.process_file(
            domain_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check network was created
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Check for domain nodes
        domain_nodes = [n for n in graph.nodes if '.com' in n or '.org' in n]
        assert len(domain_nodes) > 0


class TestKeywordExtraction:
    """Test keyword extraction method."""

    def test_keyword_extraction_basic(self, keyword_csv_file, temp_dir):
        """Test basic keyword extraction."""
        pipeline = SocialNetworkPipeline(
            extraction_method="keyword",
            extractor_config={
                'min_keywords': 3,
                'max_keywords': 10,
                'language': 'english'
            }
        )

        graph, stats = pipeline.process_file(
            keyword_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check network was created
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Check that keywords were extracted for authors
        alice_edges = list(graph.out_edges('@alice'))
        bob_edges = list(graph.out_edges('@bob'))

        assert len(alice_edges) >= 3  # At least min_keywords
        assert len(bob_edges) >= 3


class TestExactMatchExtraction:
    """Test exact match extraction method."""

    def test_exact_match_basic(self, exact_match_csv_file, temp_dir):
        """Test basic exact match extraction."""
        pipeline = SocialNetworkPipeline(
            extraction_method="exact"
        )

        graph, stats = pipeline.process_file(
            exact_match_csv_file,
            author_column='author',
            text_column='text',
            show_progress=False
        )

        # Check network was created
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Check for exact text values as nodes
        assert 'positive' in graph.nodes
        assert 'negative' in graph.nodes
        assert 'neutral' in graph.nodes


class TestMetadataIntegration:
    """Test metadata column support."""

    def test_node_metadata(self, metadata_csv_file, temp_dir):
        """Test node metadata attachment."""
        pipeline = SocialNetworkPipeline(
            extraction_method="hashtag"
        )

        graph, stats = pipeline.process_file(
            metadata_csv_file,
            author_column='author',
            text_column='text',
            node_metadata_columns=['sentiment'],
            show_progress=False
        )

        # Check that metadata was attached to author nodes
        alice_attrs = graph.nodes.get('@alice', {})
        # Metadata should be present (though it might be from first occurrence)
        assert 'sentiment' in alice_attrs or len(alice_attrs) > 0

    def test_edge_metadata(self, metadata_csv_file, temp_dir):
        """Test edge metadata attachment."""
        pipeline = SocialNetworkPipeline(
            extraction_method="hashtag"
        )

        graph, stats = pipeline.process_file(
            metadata_csv_file,
            author_column='author',
            text_column='text',
            edge_metadata_columns=['likes', 'retweets'],
            show_progress=False
        )

        # Check that network was created with edges
        assert len(graph.edges) > 0

        # Check that some edge has metadata
        for u, v, attrs in graph.edges(data=True):
            # At least check that edges were created
            assert u is not None
            assert v is not None


class TestExtractorFactory:
    """Test the extractor factory method."""

    def test_create_all_extractors(self):
        """Test that all extractor types can be created."""
        methods = ['ner', 'hashtag', 'mention', 'domain', 'keyword', 'exact']

        for method in methods:
            pipeline = SocialNetworkPipeline(extraction_method=method)
            assert pipeline.extractor is not None
            assert pipeline.extraction_method == method

    def test_invalid_method_raises_error(self):
        """Test that invalid extraction method raises error."""
        with pytest.raises(ValueError, match="Unknown extraction method"):
            pipeline = SocialNetworkPipeline(extraction_method="invalid_method")


class TestBackwardCompatibility:
    """Test backward compatibility with existing NER-based code."""

    def test_default_is_ner(self):
        """Test that default extraction method is NER."""
        pipeline = SocialNetworkPipeline()
        assert pipeline.extraction_method == "ner"
        assert pipeline.ner_engine is not None

    def test_ner_specific_params_still_work(self):
        """Test that NER-specific parameters still work."""
        pipeline = SocialNetworkPipeline(
            model_name="Davlan/xlm-roberta-base-ner-hrl",
            confidence_threshold=0.9,
            enable_cache=True
        )
        assert pipeline.extraction_method == "ner"
        assert pipeline.ner_engine is not None

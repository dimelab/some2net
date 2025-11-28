"""
Performance benchmark tests.

Tests processing speed, memory usage, and scalability.
Run with: pytest tests/test_performance.py -v -m performance
"""

import pytest
import time
import tempfile
from pathlib import Path
import pandas as pd

from src.core.data_loader import DataLoader
from src.core.network_builder import NetworkBuilder


# =============================================================================
# Performance Benchmarks
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestDataLoaderPerformance:
    """Performance tests for DataLoader."""

    def test_load_1000_rows(self, temp_dir):
        """Benchmark loading 1,000 rows."""
        filepath = temp_dir / "perf_1k.csv"

        # Create test file
        with open(filepath, 'w') as f:
            f.write("author,text,post_id\n")
            for i in range(1000):
                f.write(f"@user{i % 100},This is test post number {i},post_{i}\n")

        loader = DataLoader()

        start_time = time.time()
        total_rows = 0

        for chunk in loader.load_csv(filepath, "author", "text"):
            total_rows += len(chunk)

        elapsed = time.time() - start_time

        assert total_rows == 1000
        assert elapsed < 5.0  # Should complete in under 5 seconds
        print(f"\n1,000 rows loaded in {elapsed:.2f}s ({1000/elapsed:.0f} rows/sec)")

    def test_load_10000_rows(self, temp_dir):
        """Benchmark loading 10,000 rows."""
        filepath = temp_dir / "perf_10k.csv"

        # Create test file
        with open(filepath, 'w') as f:
            f.write("author,text,post_id\n")
            for i in range(10000):
                f.write(f"@user{i % 500},This is test post number {i} about Microsoft and Google.,post_{i}\n")

        loader = DataLoader()

        start_time = time.time()
        total_rows = 0

        for chunk in loader.load_csv(filepath, "author", "text", chunksize=1000):
            total_rows += len(chunk)

        elapsed = time.time() - start_time

        assert total_rows == 10000
        assert elapsed < 10.0  # Should complete in under 10 seconds
        print(f"\n10,000 rows loaded in {elapsed:.2f}s ({10000/elapsed:.0f} rows/sec)")

    def test_chunking_performance(self, temp_dir):
        """Test chunking performance with different chunk sizes."""
        filepath = temp_dir / "perf_chunking.csv"

        # Create test file
        with open(filepath, 'w') as f:
            f.write("author,text,post_id\n")
            for i in range(5000):
                f.write(f"@user{i},Test post {i},post_{i}\n")

        loader = DataLoader()

        chunk_sizes = [100, 500, 1000, 2000, 5000]
        times = {}

        for chunk_size in chunk_sizes:
            start_time = time.time()
            total_rows = 0

            for chunk in loader.load_csv(filepath, "author", "text", chunksize=chunk_size):
                total_rows += len(chunk)

            elapsed = time.time() - start_time
            times[chunk_size] = elapsed

            assert total_rows == 5000

        print(f"\nChunking performance (5,000 rows):")
        for size, elapsed in times.items():
            print(f"  Chunk size {size:5d}: {elapsed:.3f}s ({5000/elapsed:.0f} rows/sec)")


@pytest.mark.performance
@pytest.mark.slow
class TestNetworkBuilderPerformance:
    """Performance tests for NetworkBuilder."""

    def test_build_network_1000_posts(self):
        """Benchmark building network from 1,000 posts."""
        builder = NetworkBuilder()

        # Simulate 1,000 posts with entities
        start_time = time.time()

        for i in range(1000):
            entities = [
                {'text': f'Person {i % 50}', 'type': 'PER', 'score': 0.9},
                {'text': f'Company {i % 20}', 'type': 'ORG', 'score': 0.85},
            ]

            builder.add_post(
                author=f"@user{i % 100}",
                entities=entities,
                post_id=f"post_{i}"
            )

        elapsed = time.time() - start_time
        stats = builder.get_statistics()

        assert stats['posts_processed'] == 1000
        assert elapsed < 5.0  # Should complete in under 5 seconds
        print(f"\n1,000 posts processed in {elapsed:.2f}s ({1000/elapsed:.0f} posts/sec)")
        print(f"Network: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    def test_build_large_network(self):
        """Benchmark building larger network (5,000 posts)."""
        builder = NetworkBuilder()

        start_time = time.time()

        for i in range(5000):
            # Variable number of entities per post
            num_entities = (i % 5) + 1
            entities = [
                {'text': f'Entity {(i + j) % 200}', 'type': 'PER', 'score': 0.9}
                for j in range(num_entities)
            ]

            builder.add_post(
                author=f"@user{i % 500}",
                entities=entities,
                post_id=f"post_{i}"
            )

        elapsed = time.time() - start_time
        stats = builder.get_statistics()

        assert stats['posts_processed'] == 5000
        assert elapsed < 15.0  # Should complete in under 15 seconds
        print(f"\n5,000 posts processed in {elapsed:.2f}s ({5000/elapsed:.0f} posts/sec)")
        print(f"Network: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    def test_statistics_calculation_performance(self):
        """Benchmark statistics calculation on large network."""
        builder = NetworkBuilder()

        # Build network
        for i in range(2000):
            entities = [
                {'text': f'Entity {i % 100}', 'type': 'PER', 'score': 0.9}
            ]
            builder.add_post(f"@user{i % 200}", entities)

        # Benchmark statistics calculation
        start_time = time.time()

        stats = builder.get_statistics()

        elapsed = time.time() - start_time

        assert stats is not None
        assert elapsed < 1.0  # Should be very fast
        print(f"\nStatistics calculated in {elapsed:.4f}s")


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory efficiency."""

    def test_chunked_loading_memory(self, temp_dir):
        """Verify chunked loading doesn't load entire file."""
        filepath = temp_dir / "large.csv"

        # Create larger file
        with open(filepath, 'w') as f:
            f.write("author,text,post_id\n")
            for i in range(10000):
                f.write(f"@user{i},{'A' * 100},post_{i}\n")

        loader = DataLoader()

        # Process in small chunks
        chunk_count = 0
        for chunk in loader.load_csv(filepath, "author", "text", chunksize=100):
            chunk_count += 1
            # Verify chunk size is limited
            assert len(chunk) <= 100

        assert chunk_count == 100  # 10,000 rows / 100 per chunk


# =============================================================================
# Scalability Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestScalability:
    """Test scaling behavior."""

    def test_linear_scaling_data_loading(self, temp_dir):
        """Test that data loading scales linearly."""
        loader = DataLoader()

        sizes = [100, 500, 1000]
        times = {}

        for size in sizes:
            filepath = temp_dir / f"scale_{size}.csv"

            with open(filepath, 'w') as f:
                f.write("author,text\n")
                for i in range(size):
                    f.write(f"@user{i},Test post {i}\n")

            start_time = time.time()

            total_rows = 0
            for chunk in loader.load_csv(filepath, "author", "text"):
                total_rows += len(chunk)

            elapsed = time.time() - start_time
            times[size] = elapsed

            assert total_rows == size

        print(f"\nData loading scalability:")
        for size, elapsed in times.items():
            rate = size / elapsed if elapsed > 0 else 0
            print(f"  {size:4d} rows: {elapsed:.3f}s ({rate:.0f} rows/sec)")

        # Check scaling is roughly linear
        # Time for 1000 should be ~10x time for 100
        if times[100] > 0:
            ratio = times[1000] / times[100]
            assert 5 < ratio < 15  # Allow for some variance

    def test_network_growth(self):
        """Test network builder performance as network grows."""
        builder = NetworkBuilder()

        times = []
        sizes = [100, 500, 1000, 2000]

        for target_size in sizes:
            start_time = time.time()

            for i in range(target_size):
                entities = [
                    {'text': f'Entity {i % 50}', 'type': 'PER', 'score': 0.9}
                ]
                builder.add_post(f"@user{i % 20}", entities)

            elapsed = time.time() - start_time
            times.append(elapsed)

            # Reset for next iteration
            builder.reset()

        print(f"\nNetwork builder scalability:")
        for size, elapsed in zip(sizes, times):
            rate = size / elapsed if elapsed > 0 else 0
            print(f"  {size:4d} posts: {elapsed:.3f}s ({rate:.0f} posts/sec)")


# =============================================================================
# Throughput Tests
# =============================================================================

@pytest.mark.performance
class TestThroughput:
    """Test processing throughput."""

    def test_data_loading_throughput(self, very_large_csv_file):
        """Measure data loading throughput."""
        if not very_large_csv_file.exists():
            pytest.skip("Large test file not available")

        loader = DataLoader()

        start_time = time.time()
        total_rows = 0

        for chunk in loader.load_csv(
            very_large_csv_file,
            "author",
            "text",
            chunksize=1000
        ):
            total_rows += len(chunk)

        elapsed = time.time() - start_time
        throughput = total_rows / elapsed

        print(f"\nData loading throughput: {throughput:.0f} rows/second")
        assert throughput > 100  # Should process at least 100 rows/sec

    def test_network_building_throughput(self):
        """Measure network building throughput."""
        builder = NetworkBuilder()

        num_posts = 3000
        start_time = time.time()

        for i in range(num_posts):
            entities = [
                {'text': f'Entity {i % 100}', 'type': 'PER', 'score': 0.9},
                {'text': f'Org {i % 30}', 'type': 'ORG', 'score': 0.85}
            ]

            builder.add_post(f"@user{i % 300}", entities)

        elapsed = time.time() - start_time
        throughput = num_posts / elapsed

        print(f"\nNetwork building throughput: {throughput:.0f} posts/second")
        assert throughput > 100  # Should process at least 100 posts/sec


# =============================================================================
# Stress Tests
# =============================================================================

@pytest.mark.performance
@pytest.mark.slow
class TestStressConditions:
    """Stress test under extreme conditions."""

    def test_many_small_chunks(self, temp_dir):
        """Test processing with very small chunk size."""
        filepath = temp_dir / "stress_small_chunks.csv"

        with open(filepath, 'w') as f:
            f.write("author,text\n")
            for i in range(1000):
                f.write(f"@user{i},Test {i}\n")

        loader = DataLoader()

        # Use tiny chunk size
        start_time = time.time()
        total_rows = 0

        for chunk in loader.load_csv(filepath, "author", "text", chunksize=10):
            total_rows += len(chunk)

        elapsed = time.time() - start_time

        assert total_rows == 1000
        print(f"\n1,000 rows with 10-row chunks: {elapsed:.2f}s")

    def test_many_unique_entities(self):
        """Test network with many unique entities."""
        builder = NetworkBuilder()

        num_posts = 1000
        start_time = time.time()

        for i in range(num_posts):
            # Each post has unique entities
            entities = [
                {'text': f'Unique Entity {i}', 'type': 'PER', 'score': 0.9}
            ]

            builder.add_post(f"@user{i}", entities)

        elapsed = time.time() - start_time
        stats = builder.get_statistics()

        # Should have ~1000 entity nodes + 1000 author nodes
        assert stats['total_nodes'] >= 1800

        print(f"\n{num_posts} posts with unique entities: {elapsed:.2f}s")
        print(f"Network: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    def test_dense_network(self):
        """Test building very dense network (many connections)."""
        builder = NetworkBuilder()

        num_authors = 100
        num_shared_entities = 20

        shared_entities = [
            {'text': f'Shared Entity {i}', 'type': 'PER', 'score': 0.9}
            for i in range(num_shared_entities)
        ]

        start_time = time.time()

        # Each author mentions all shared entities
        for i in range(num_authors):
            builder.add_post(f"@user{i}", shared_entities)

        elapsed = time.time() - start_time
        stats = builder.get_statistics()

        # Should have dense connections
        expected_edges = num_authors * num_shared_entities
        assert stats['total_edges'] == expected_edges

        print(f"\nDense network ({num_authors} x {num_shared_entities}): {elapsed:.2f}s")
        print(f"Density: {stats['density']:.4f}")


# =============================================================================
# Comparison Benchmarks
# =============================================================================

@pytest.mark.performance
class TestComparisons:
    """Compare different configuration options."""

    def test_entity_resolver_impact(self):
        """Compare performance with/without entity resolver."""
        num_posts = 1000
        entities_per_post = [
            {'text': 'john smith', 'type': 'PER', 'score': 0.9},
            {'text': 'John Smith', 'type': 'PER', 'score': 0.95},
        ]

        # With resolver
        builder_with = NetworkBuilder(use_entity_resolver=True)
        start_with = time.time()

        for i in range(num_posts):
            builder_with.add_post(f"@user{i}", entities_per_post)

        time_with = time.time() - start_with
        stats_with = builder_with.get_statistics()

        # Without resolver
        builder_without = NetworkBuilder(use_entity_resolver=False)
        start_without = time.time()

        for i in range(num_posts):
            builder_without.add_post(f"@user{i}", entities_per_post)

        time_without = time.time() - start_without
        stats_without = builder_without.get_statistics()

        print(f"\nEntity resolver impact ({num_posts} posts):")
        print(f"  With resolver:    {time_with:.3f}s, {stats_with['persons']} person nodes")
        print(f"  Without resolver: {time_without:.3f}s, {stats_without['persons']} person nodes")

        # With resolver should deduplicate
        assert stats_with['persons'] < stats_without['persons']

    def test_chunk_size_comparison(self, temp_dir):
        """Compare different chunk sizes."""
        filepath = temp_dir / "chunk_comparison.csv"

        with open(filepath, 'w') as f:
            f.write("author,text\n")
            for i in range(5000):
                f.write(f"@user{i},Test post {i}\n")

        loader = DataLoader()
        chunk_sizes = [100, 500, 1000, 5000]
        times = {}

        for chunk_size in chunk_sizes:
            start_time = time.time()

            total_rows = 0
            for chunk in loader.load_csv(filepath, "author", "text", chunksize=chunk_size):
                total_rows += len(chunk)

            times[chunk_size] = time.time() - start_time

        print(f"\nChunk size comparison (5,000 rows):")
        for size in chunk_sizes:
            print(f"  {size:5d}: {times[size]:.3f}s")

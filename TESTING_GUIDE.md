# Testing Guide

Complete guide to testing the Social Network Analytics library.

## Overview

The library includes **400+ tests** organized into multiple categories with comprehensive coverage of all modules, error scenarios, edge cases, and performance benchmarks.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Fixtures](#fixtures)
- [Best Practices](#best-practices)
- [Continuous Integration](#continuous-integration)

---

## Quick Start

### Installation

```bash
# Install test dependencies
pip install pytest pytest-cov

# Or install with development dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing tests/
```

### Run Specific Tests

```bash
# Run single test file
pytest tests/test_error_handling.py -v

# Run single test class
pytest tests/test_error_handling.py::TestCustomExceptions -v

# Run single test function
pytest tests/test_error_handling.py::TestCustomExceptions::test_file_not_found_error -v
```

---

## Test Structure

```
tests/
├── conftest.py                   # Shared fixtures (450+ lines)
├── test_data_loader.py          # DataLoader tests (~30 tests)
├── test_ner_engine.py           # NER Engine tests (~40 tests)
├── test_entity_resolver.py      # Entity resolution tests (~45 tests)
├── test_network_builder.py      # Network builder tests (~50 tests)
├── test_exporters.py            # Export functionality tests (~55 tests)
├── test_pipeline.py             # Pipeline tests (~40 tests)
├── test_integration.py          # Integration tests (~35 tests)
├── test_error_handling.py       # Error handling tests (42 tests)
├── test_edge_cases.py           # Edge case tests (48 tests)
└── test_performance.py          # Performance benchmarks (16 tests)
```

**Total: 400+ tests**

---

## Test Categories

### Unit Tests

Individual module testing with isolated components.

```bash
# Run all unit tests
pytest -m unit -v
```

**Coverage:**
- `test_data_loader.py` - CSV/NDJSON loading, encoding, validation
- `test_ner_engine.py` - Model loading, entity extraction, caching
- `test_entity_resolver.py` - Deduplication, fuzzy matching
- `test_network_builder.py` - Graph construction, statistics
- `test_exporters.py` - All export formats

### Integration Tests

End-to-end pipeline testing.

```bash
# Run integration tests
pytest -m integration -v
```

**Coverage:**
- Complete data pipeline (load → NER → network → export)
- Multi-module interactions
- Error propagation
- Progress callbacks

### Edge Case Tests

Boundary conditions and unusual inputs.

```bash
# Run edge case tests
pytest -m edge_case -v
```

**48 tests covering:**
- Empty files and data
- Unicode and special characters
- Very long text (10K+ characters)
- Malformed CSV/JSON
- Missing columns
- Null values
- Extreme confidence thresholds
- Maximum node counts

### Error Handling Tests

Exception and error tracking functionality.

```bash
# Run error handling tests
pytest tests/test_error_handling.py -v
```

**42 tests covering:**
- All custom exception classes
- Error message formatting
- ErrorTracker functionality
- Error context manager
- Logger setup
- Error report export

### Performance Tests

Benchmarks and scalability validation.

```bash
# Run performance tests
pytest -m performance -v
```

**16 tests covering:**
- Data loading speed (1K, 10K rows)
- Network building throughput
- Statistics calculation
- Memory efficiency
- Scalability (linear scaling)
- Stress conditions

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/ -v

# Run with short traceback
pytest tests/ --tb=short

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Show local variables in traceback
pytest tests/ -l
```

### By Marker

```bash
# Run only unit tests
pytest -m unit -v

# Run integration tests
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v

# Run edge case tests
pytest -m edge_case -v

# Run performance benchmarks
pytest -m performance -v

# Run tests requiring model download
pytest -m requires_model -v

# Run GPU tests (if available)
pytest -m requires_gpu -v
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# Terminal coverage report
pytest --cov=src --cov-report=term-missing tests/

# Fail if coverage below threshold
pytest --cov=src --cov-fail-under=80 tests/

# Coverage for specific module
pytest --cov=src.core.data_loader tests/test_data_loader.py
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (auto-detect CPUs)
pytest -n auto tests/

# Run on 4 cores
pytest -n 4 tests/
```

### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and re-run tests
ptw tests/
```

---

## Writing Tests

### Test File Template

```python
"""
Tests for ModuleName.

Description of what this test file covers.
"""

import pytest
from src.module import ClassName


@pytest.fixture
def instance():
    """Create instance for testing."""
    return ClassName()


class TestClassName:
    """Test cases for ClassName."""

    def test_initialization(self, instance):
        """Test object initialization."""
        assert instance is not None

    def test_method_name(self, instance):
        """Test specific method."""
        result = instance.method()
        assert result == expected_value

    def test_error_case(self, instance):
        """Test error handling."""
        with pytest.raises(ValueError) as exc_info:
            instance.method(invalid_input)

        assert "expected error message" in str(exc_info.value)
```

### Using Fixtures

```python
def test_with_csv_file(sample_csv_file):
    """Test using pre-created CSV file."""
    loader = DataLoader()

    chunks = list(loader.load_csv(sample_csv_file, "author", "text"))

    assert len(chunks) > 0
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("test", "TEST"),
    ("Hello", "HELLO"),
    ("123", "123"),
])
def test_uppercase(input, expected):
    """Test uppercase conversion with multiple inputs."""
    assert input.upper() == expected
```

### Markers

```python
@pytest.mark.slow
def test_large_dataset():
    """This test takes a long time."""
    # Test implementation

@pytest.mark.edge_case
def test_empty_input():
    """Test edge case with empty input."""
    # Test implementation

@pytest.mark.requires_model
def test_ner_extraction():
    """This test requires NER model download."""
    # Test implementation
```

---

## Fixtures

### Available Fixtures

See `tests/conftest.py` for all fixtures. Key fixtures:

#### Temporary Directories

```python
def test_with_temp_dir(temp_dir):
    """Use temporary directory."""
    filepath = temp_dir / "test.csv"
    # ... test code
    # Directory cleaned up automatically
```

#### Test Data

```python
def test_with_english_data(sample_csv_data_english):
    """Use English test data."""
    data = sample_csv_data_english
    assert len(data['author']) == 5

def test_with_danish_data(sample_csv_data_danish):
    """Use Danish test data."""
    data = sample_csv_data_danish
    assert "København" in data['text'][0]
```

#### File Creators

```python
def test_create_csv(create_csv_file, sample_csv_data_english, temp_dir):
    """Create CSV file dynamically."""
    filepath = create_csv_file(sample_csv_data_english, "custom.csv")
    assert filepath.exists()
```

#### Edge Case Data

```python
def test_edge_cases(edge_case_texts):
    """Test with edge case text."""
    unicode_text = edge_case_texts['unicode']
    empty_text = edge_case_texts['empty']
    # ... test code
```

---

## Best Practices

### 1. Test Organization

✅ **Good:**
```python
class TestDataLoader:
    """Group related tests together."""

    def test_load_csv(self):
        """Test CSV loading."""
        pass

    def test_load_ndjson(self):
        """Test NDJSON loading."""
        pass
```

❌ **Bad:**
```python
def test_1():
    """Unclear test name."""
    pass

def test_2():
    """Another unclear test."""
    pass
```

### 2. Test Names

✅ **Good:**
```python
def test_load_csv_with_missing_column_raises_error():
    """Clear, descriptive test name."""
    pass
```

❌ **Bad:**
```python
def test_csv():
    """Vague test name."""
    pass
```

### 3. Assertions

✅ **Good:**
```python
assert len(result) == 5, f"Expected 5 items, got {len(result)}"
assert "John Smith" in entities, "Expected entity not found"
```

❌ **Bad:**
```python
assert len(result)  # What are we checking?
assert result  # Too vague
```

### 4. Test Independence

✅ **Good:**
```python
def test_feature_a():
    """Independent test."""
    instance = create_instance()
    # ... test code

def test_feature_b():
    """Another independent test."""
    instance = create_instance()
    # ... test code
```

❌ **Bad:**
```python
global_instance = None

def test_feature_a():
    """Depends on global state."""
    global global_instance
    global_instance = create_instance()

def test_feature_b():
    """Depends on test_feature_a."""
    assert global_instance is not None
```

### 5. Use Fixtures

✅ **Good:**
```python
@pytest.fixture
def loader():
    return DataLoader()

def test_with_fixture(loader):
    result = loader.load_csv("test.csv")
```

❌ **Bad:**
```python
def test_without_fixture():
    loader = DataLoader()  # Repeated in every test
    result = loader.load_csv("test.csv")
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest tests/ -m "not slow" --tb=short || exit 1
```

---

## Test Coverage Goals

- **Overall Coverage**: 80%+
- **Core Modules**: 90%+
- **Error Paths**: 100%
- **Public APIs**: 100%

### Check Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing tests/

# View HTML report
pytest --cov=src --cov-report=html tests/
open htmlcov/index.html
```

---

## Troubleshooting Tests

### Tests Fail on First Run

Some tests require NER model download. Run once to download:

```bash
pytest tests/test_ner_engine.py -v
```

### Slow Tests

Skip slow tests during development:

```bash
pytest -m "not slow" tests/
```

### Memory Issues

Run tests sequentially:

```bash
pytest -n 0 tests/
```

### GPU Tests Fail

Skip GPU-required tests if no GPU available:

```bash
pytest -m "not requires_gpu" tests/
```

---

## Performance Benchmarks

Expected performance on modern hardware:

| Test | Expected Time | Threshold |
|------|---------------|-----------|
| 1,000 rows | < 5 seconds | PASS if < 5s |
| 10,000 rows | < 10 seconds | PASS if < 10s |
| Network (1K posts) | < 5 seconds | PASS if < 5s |
| Network (5K posts) | < 15 seconds | PASS if < 15s |
| Statistics | < 1 second | PASS if < 1s |

---

## Test Maintenance

### Adding New Tests

1. Create test file or add to existing
2. Use appropriate markers
3. Add fixtures to `conftest.py` if reusable
4. Update this guide if needed
5. Run full test suite before committing

### Updating Tests

1. Update test when changing functionality
2. Ensure backward compatibility tests exist
3. Add deprecation tests for removed features
4. Update documentation

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/example/markers.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Last Updated**: 2025-11-28
**Status**: Complete
**Coverage**: 80%+ estimated

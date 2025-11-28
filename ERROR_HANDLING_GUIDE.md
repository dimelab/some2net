# Error Handling Guide

## Overview

The Social Network Analytics (SNA) library includes a comprehensive error handling system designed to:

1. **Provide clear, user-friendly error messages**
2. **Track and log errors for debugging**
3. **Gracefully recover from recoverable errors**
4. **Generate detailed error reports**
5. **Maintain processing continuity when possible**

---

## Error Hierarchy

### Base Exception: `SNAException`

All custom exceptions inherit from `SNAException`, which provides:
- User-friendly `message`
- Technical `details` for debugging
- Consistent error formatting

```python
from src.core.exceptions import SNAException

try:
    # ... code ...
except SNAException as e:
    print(e.message)        # User-friendly message
    print(e.details)        # Technical details
```

### Error Categories

```
SNAException (base)
├── UserError (user input issues)
│   ├── FileNotFoundError
│   ├── InvalidFileFormatError
│   ├── ColumnNotFoundError
│   ├── EmptyDataError
│   ├── EncodingError
│   └── ValidationError
│       ├── ThresholdValidationError
│       └── BatchSizeValidationError
│
├── ProcessingError (recoverable errors)
│   ├── NERProcessingError
│   ├── EntityResolutionError
│   ├── NetworkConstructionError
│   └── ExportError
│
├── CriticalError (system errors)
│   ├── ModelLoadError
│   ├── GPUError
│   ├── OutOfMemoryError
│   └── DiskSpaceError
│
├── NetworkError (network/download issues)
│   ├── ModelDownloadError
│   └── CacheError
│
└── ConfigurationError (configuration issues)
    ├── InvalidConfigError
    └── MissingConfigError
```

---

## Exception Classes Reference

### User Errors (Input Issues)

#### FileNotFoundError
```python
from src.core.exceptions import FileNotFoundError

raise FileNotFoundError("/path/to/missing/file.csv")
# Message: File not found: /path/to/missing/file.csv
# Details: Please check the file path and ensure the file exists.
```

#### ColumnNotFoundError
```python
from src.core.exceptions import ColumnNotFoundError

raise ColumnNotFoundError("username", ["user", "text", "timestamp"])
# Message: Column 'username' not found in data
# Details: Available columns: 'user', 'text', 'timestamp'
```

#### InvalidFileFormatError
```python
from src.core.exceptions import InvalidFileFormatError

raise InvalidFileFormatError("data.xlsx", [".csv", ".ndjson"])
# Message: Invalid file format: data.xlsx
# Details: Supported formats: .csv, .ndjson
```

### Processing Errors (Recoverable)

#### NERProcessingError
```python
from src.core.exceptions import NERProcessingError

raise NERProcessingError("Model inference failed", batch_index=5)
# Message: Model inference failed
# Details: Batch index: 5
```

#### NetworkConstructionError
```python
from src.core.exceptions import NetworkConstructionError

raise NetworkConstructionError("Failed to add edge", post_id="12345")
# Message: Failed to add edge
# Details: Post ID: 12345
```

### Critical Errors (System Issues)

#### OutOfMemoryError
```python
from src.core.exceptions import OutOfMemoryError

raise OutOfMemoryError("NER processing", "Try reducing batch size to 16")
# Message: Out of memory during NER processing
# Details: Try reducing batch size to 16
```

#### GPUError
```python
from src.core.exceptions import GPUError

raise GPUError("CUDA out of memory", fallback_available=True)
# Message: CUDA out of memory
# Details: Falling back to CPU processing
```

---

## Logging System

### Setup Logger

```python
from src.utils.logger import setup_logger

# Basic setup
logger = setup_logger("sna", level="INFO")

# Custom configuration
logger = setup_logger(
    name="sna",
    level="DEBUG",
    log_dir="./logs",
    console_output=True,
    file_output=True
)
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (potential issues)
- **ERROR**: Error messages (recoverable)
- **CRITICAL**: Critical errors (may cause failure)

### Example Usage

```python
logger.debug("Processing chunk 5 with 10000 rows")
logger.info("Successfully loaded data file")
logger.warning("Low confidence in entity detection")
logger.error("Failed to process post #12345")
logger.critical("GPU out of memory, terminating")
```

---

## Error Tracking

### ErrorTracker Class

Track and accumulate errors during processing for later analysis.

```python
from src.utils.logger import ErrorTracker

# Initialize tracker
tracker = ErrorTracker(max_errors=1000)

# Add errors
tracker.add_error(
    error=ValueError("Invalid value"),
    context="Data validation",
    post_id="12345",
    chunk_num=5,
    severity="ERROR"
)

# Get error summary
summary = tracker.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"By severity: {summary['by_severity']}")
print(f"By type: {summary['by_type']}")

# Export error reports
tracker.export_to_json("./output/error_report.json")
tracker.export_to_text("./output/error_report.txt")
```

### Error Context Manager

Automatically track errors in specific contexts:

```python
from src.utils.logger import error_context, ErrorTracker

tracker = ErrorTracker()
logger = setup_logger("sna")

# Errors within context are automatically tracked
with error_context(tracker, "NER processing", chunk_num=5, logger=logger):
    # This error will be caught and tracked
    process_batch(data)
```

---

## Error Handling Patterns

### Pattern 1: Try-Catch with Custom Exceptions

```python
from src.core.exceptions import FileNotFoundError, ColumnNotFoundError

def load_data(filepath, column):
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(filepath)

        df = pd.read_csv(filepath)

        if column not in df.columns:
            raise ColumnNotFoundError(column, df.columns.tolist())

        return df

    except SNAException as e:
        logger.error(f"Error loading data: {e.message}")
        raise
```

### Pattern 2: Error Conversion

```python
from src.core.exceptions import handle_error

try:
    # Code that may raise standard exceptions
    file = open("data.csv")

except Exception as e:
    # Convert to custom exception
    custom_exc = handle_error(e, logger=logger, context="file loading")
    raise custom_exc
```

### Pattern 3: Graceful Recovery

```python
from src.core.exceptions import ProcessingError
from src.utils.logger import ErrorTracker

tracker = ErrorTracker()

for post in posts:
    try:
        process_post(post)
    except ProcessingError as e:
        # Track error but continue processing
        tracker.add_error(e, context="post processing", post_id=post.id)
        logger.warning(f"Skipping post {post.id}: {e.message}")
        continue

# After processing
if tracker.has_errors():
    tracker.export_to_text("./output/processing_errors.txt")
    print(f"Completed with {len(tracker)} errors. See error report.")
```

### Pattern 4: Critical Error Handling

```python
from src.core.exceptions import OutOfMemoryError, GPUError

try:
    results = process_with_gpu(data, batch_size=64)

except OutOfMemoryError:
    # Try again with smaller batch
    logger.warning("GPU OOM, reducing batch size")
    results = process_with_gpu(data, batch_size=16)

except GPUError as e:
    # Fall back to CPU
    if e.fallback_available:
        logger.warning("GPU error, falling back to CPU")
        results = process_with_cpu(data)
    else:
        logger.critical("GPU error with no fallback")
        raise
```

---

## Error Reports

### JSON Error Report Format

```json
{
  "generated_at": "2025-11-28T10:30:00",
  "summary": {
    "total_errors": 25,
    "by_severity": {
      "WARNING": 10,
      "ERROR": 13,
      "CRITICAL": 2
    },
    "by_type": {
      "ValueError": 8,
      "NERProcessingError": 12,
      "OutOfMemoryError": 2,
      "FileNotFoundError": 3
    },
    "truncated": false
  },
  "errors": [
    {
      "timestamp": "2025-11-28T10:25:34",
      "type": "NERProcessingError",
      "message": "Model inference failed",
      "severity": "ERROR",
      "context": "NER batch processing",
      "post_id": "12345",
      "chunk_num": 5,
      "traceback": null
    }
  ]
}
```

### Text Error Report Format

```
================================================================================
ERROR REPORT
================================================================================
Generated: 2025-11-28 10:30:00
Total Errors: 25

SUMMARY BY SEVERITY:
----------------------------------------
  WARNING: 10
  ERROR: 13
  CRITICAL: 2

SUMMARY BY TYPE:
----------------------------------------
  ValueError: 8
  NERProcessingError: 12
  OutOfMemoryError: 2
  FileNotFoundError: 3

================================================================================
DETAILED ERRORS
================================================================================

Error #1
----------------------------------------
Timestamp: 2025-11-28T10:25:34
Severity: ERROR
Type: NERProcessingError
Message: Model inference failed
Context: NER batch processing
Post ID: 12345
Chunk: 5
```

---

## Integration Examples

### Example 1: Data Loader with Error Handling

```python
from src.core.data_loader import DataLoader
from src.core.exceptions import FileNotFoundError, EncodingError, handle_error
from src.utils.logger import setup_logger, ErrorTracker

logger = setup_logger("data_loader")
tracker = ErrorTracker()

try:
    loader = DataLoader()

    for chunk in loader.load_csv("data.csv", "author", "text"):
        # Process chunk
        pass

except FileNotFoundError as e:
    logger.error(f"{e.message}\n{e.details}")
    print(format_error_for_user(e, include_details=True))

except EncodingError as e:
    logger.error(f"Encoding issue: {e.message}")
    print("Try saving the file as UTF-8 and retry")

except Exception as e:
    custom_exc = handle_error(e, logger=logger, context="data loading")
    print(format_error_for_user(custom_exc))
```

### Example 2: Pipeline with Complete Error Tracking

```python
from src.core.pipeline import SocialNetworkPipeline
from src.utils.logger import setup_logger, ErrorTracker
from src.core.exceptions import format_error_for_user

# Setup
logger = setup_logger("pipeline", level="INFO")
tracker = ErrorTracker()

pipeline = SocialNetworkPipeline()

try:
    # Process with error tracking
    graph, stats = pipeline.process_file(
        filepath="data.csv",
        author_column="user",
        text_column="text"
    )

    # Check for errors during processing
    if pipeline.get_processing_metadata()['errors']:
        print("⚠️  Processing completed with errors:")
        for error in pipeline.get_processing_metadata()['errors']:
            print(f"  - {error}")

        # Export error report
        tracker.export_to_text("./output/error_report.txt")

except Exception as e:
    logger.critical(f"Pipeline failed: {e}", exc_info=True)
    print(format_error_for_user(e, include_details=True))
    tracker.export_to_text("./output/critical_error.txt")
```

### Example 3: CLI with Error Handling

```python
import sys
from src.core import process_social_media_data
from src.core.exceptions import format_error_for_user, UserError, CriticalError
from src.utils.logger import setup_logger

logger = setup_logger("cli")

try:
    graph, stats, files = process_social_media_data(
        "data.csv",
        author_column="user",
        text_column="text"
    )

    print("✅ Processing complete!")
    sys.exit(0)

except UserError as e:
    # User input error - show friendly message
    print(format_error_for_user(e, include_details=True))
    sys.exit(1)

except CriticalError as e:
    # System error - show error and suggest action
    print(format_error_for_user(e, include_details=True))
    logger.critical(str(e), exc_info=True)
    sys.exit(2)

except KeyboardInterrupt:
    print("\n⚠️  Processing interrupted by user")
    sys.exit(130)

except Exception as e:
    # Unexpected error
    print(f"❌ Unexpected error: {e}")
    logger.critical("Unexpected error", exc_info=True)
    sys.exit(1)
```

---

## Best Practices

### 1. Use Specific Exceptions

❌ **Bad:**
```python
raise Exception("File not found")
```

✅ **Good:**
```python
from src.core.exceptions import FileNotFoundError
raise FileNotFoundError(filepath)
```

### 2. Provide Context

❌ **Bad:**
```python
raise ValueError("Invalid value")
```

✅ **Good:**
```python
tracker.add_error(
    ValueError("Invalid confidence threshold: 1.5"),
    context="Configuration validation",
    severity="ERROR"
)
```

### 3. Log Appropriately

❌ **Bad:**
```python
print(f"Error: {e}")
```

✅ **Good:**
```python
logger.error(f"Failed to process post {post_id}: {e}")
```

### 4. Handle Gracefully

❌ **Bad:**
```python
for post in posts:
    process(post)  # Fails on first error
```

✅ **Good:**
```python
for post in posts:
    try:
        process(post)
    except ProcessingError as e:
        tracker.add_error(e, post_id=post.id)
        continue  # Process remaining posts
```

### 5. Export Error Reports

```python
# At end of processing
if tracker.has_errors():
    tracker.export_to_json("./output/error_report.json")
    tracker.export_to_text("./output/error_report.txt")

    summary = tracker.get_error_summary()
    print(f"Completed with {summary['total_errors']} errors")
    print("Error report saved to ./output/")
```

---

## Testing Error Handling

### Test Error Raising

```python
import pytest
from src.core.exceptions import ColumnNotFoundError

def test_column_not_found():
    with pytest.raises(ColumnNotFoundError) as exc_info:
        raise ColumnNotFoundError("missing_col", ["col1", "col2"])

    assert exc_info.value.column_name == "missing_col"
    assert "col1" in str(exc_info.value)
```

### Test Error Tracking

```python
from src.utils.logger import ErrorTracker

def test_error_tracker():
    tracker = ErrorTracker()

    tracker.add_error(ValueError("test"), context="testing")

    assert len(tracker) == 1
    assert tracker.has_errors()

    summary = tracker.get_error_summary()
    assert summary['total_errors'] == 1
    assert 'ValueError' in summary['by_type']
```

---

## Error Message Guidelines

### User-Facing Messages

- **Clear and concise**: Explain what went wrong
- **Actionable**: Tell user what they can do
- **Friendly tone**: Avoid technical jargon

Example:
```
❌ Error: File not found: data.csv
   Please check the file path and ensure the file exists.
```

### Technical Details (Logs)

- **Precise**: Include exact error type and location
- **Contextual**: Add file names, line numbers, variables
- **Stack traces**: For critical errors only

Example:
```
2025-11-28 10:30:45 - sna - ERROR - load_csv:156 - File not found: /path/to/data.csv
Details: Attempted to load CSV file for processing. File does not exist at specified path.
```

---

## Troubleshooting Common Errors

### FileNotFoundError

**Cause**: Input file doesn't exist at specified path

**Solutions**:
- Check file path is correct
- Use absolute paths instead of relative
- Ensure file has correct extension (.csv, .ndjson)

### ColumnNotFoundError

**Cause**: Specified column name not in data

**Solutions**:
- Check column name spelling (case-sensitive)
- Check for extra spaces in column names
- Load file in spreadsheet to verify column names

### EncodingError

**Cause**: File encoding not supported

**Solutions**:
- Resave file as UTF-8
- Try specifying encoding explicitly: `encoding="latin-1"`
- Use text editor to check file encoding

### OutOfMemoryError

**Cause**: Insufficient RAM or GPU memory

**Solutions**:
- Reduce `batch_size` (try 16 or 8)
- Reduce `chunk_size` (try 5000)
- Close other applications
- Use CPU instead of GPU

### ModelLoadError

**Cause**: Cannot load NER model

**Solutions**:
- Check internet connection (first download)
- Clear model cache: `rm -rf ./models/*`
- Try different model name
- Check disk space

---

## Configuration

### Logging Configuration

Create `config.yaml`:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_dir: "./logs"
  console_output: true
  file_output: true

error_tracking:
  max_errors: 1000
  export_on_complete: true
  export_formats: ["json", "text"]
```

### Error Handling Settings

```python
# In pipeline initialization
pipeline = SocialNetworkPipeline(
    # ... other params ...
    error_handling={
        'track_errors': True,
        'continue_on_error': True,
        'max_errors': 1000,
        'export_error_report': True
    }
)
```

---

## API Reference

### Functions

#### `format_error_for_user(error, include_details=False)`
Format exception for user display

#### `handle_error(error, logger=None, context=None)`
Convert standard exception to custom exception

#### `setup_logger(name, level, log_dir, ...)`
Configure logging

### Classes

#### `ErrorTracker`
- `add_error()` - Add error to tracker
- `get_errors()` - Get tracked errors
- `get_error_summary()` - Get summary statistics
- `export_to_json()` - Export to JSON
- `export_to_text()` - Export to text
- `clear()` - Clear all errors

#### `error_context` (Context Manager)
- Automatically track errors in code block
- Optionally suppress non-critical errors
- Log errors to logger

---

## Changelog

### Version 0.1.0 (2025-11-28)

**Added:**
- Custom exception hierarchy
- `ErrorTracker` class for error accumulation
- Logging utility with file and console handlers
- Error report export (JSON and text formats)
- `error_context` context manager
- Integration with all core modules

**Features:**
- User-friendly error messages
- Technical details for debugging
- Automatic error tracking
- Graceful error recovery
- Comprehensive error reports

---

## Support

For issues or questions about error handling:

1. Check this guide
2. Review error reports in `./logs/`
3. Check application logs
4. Search existing GitHub issues
5. Create new issue with error report attached

---

**Last Updated**: 2025-11-28
**Version**: 0.1.0
**Status**: Production Ready

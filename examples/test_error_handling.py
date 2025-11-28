"""
Test Error Handling System

Demonstrates and tests the error handling functionality in the SNA library.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from src
from src.core.exceptions import (
    FileNotFoundError,
    ColumnNotFoundError,
    NERProcessingError,
    OutOfMemoryError,
    format_error_for_user,
    handle_error
)
from src.utils.logger import setup_logger, ErrorTracker, error_context


def test_custom_exceptions():
    """Test custom exception classes."""
    print("=" * 80)
    print("TEST 1: Custom Exceptions")
    print("=" * 80)

    # Test FileNotFoundError
    try:
        raise FileNotFoundError("/path/to/missing/file.csv")
    except Exception as e:
        print(f"\n{format_error_for_user(e, include_details=True)}\n")

    # Test ColumnNotFoundError
    try:
        raise ColumnNotFoundError("username", ["user", "text", "timestamp"])
    except Exception as e:
        print(f"{format_error_for_user(e, include_details=True)}\n")

    # Test NERProcessingError
    try:
        raise NERProcessingError("Model inference failed", batch_index=5)
    except Exception as e:
        print(f"{format_error_for_user(e, include_details=True)}\n")

    # Test OutOfMemoryError
    try:
        raise OutOfMemoryError("GPU processing", "Try reducing batch size to 16")
    except Exception as e:
        print(f"{format_error_for_user(e, include_details=True)}\n")


def test_error_conversion():
    """Test converting standard exceptions to custom exceptions."""
    print("=" * 80)
    print("TEST 2: Error Conversion")
    print("=" * 80)

    logger = setup_logger("test", level="INFO", file_output=False)

    # Test converting FileNotFoundError
    try:
        open("/nonexistent/file.txt")
    except Exception as e:
        custom_exc = handle_error(e, logger=logger, context="file loading")
        print(f"\nOriginal: {type(e).__name__}")
        print(f"Converted: {type(custom_exc).__name__}")
        print(f"{format_error_for_user(custom_exc, include_details=True)}\n")


def test_error_tracker():
    """Test ErrorTracker functionality."""
    print("=" * 80)
    print("TEST 3: Error Tracker")
    print("=" * 80)

    tracker = ErrorTracker(max_errors=100)

    # Add various errors
    tracker.add_error(
        ValueError("Invalid confidence threshold: 1.5"),
        context="Configuration validation",
        severity="ERROR"
    )

    tracker.add_error(
        FileNotFoundError("data.csv"),
        context="File loading",
        severity="ERROR"
    )

    tracker.add_error(
        NERProcessingError("Batch failed", batch_index=3),
        context="NER processing",
        chunk_num=5,
        severity="WARNING"
    )

    tracker.add_error(
        OutOfMemoryError("GPU inference"),
        context="Model processing",
        severity="CRITICAL"
    )

    # Print summary
    print(f"\nTotal errors tracked: {len(tracker)}")

    summary = tracker.get_error_summary()
    print(f"\nSummary:")
    print(f"  By Severity: {summary['by_severity']}")
    print(f"  By Type: {summary['by_type']}")

    # Export reports
    output_dir = Path("./output/error_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "error_report.json"
    text_path = output_dir / "error_report.txt"

    tracker.export_to_json(str(json_path))
    tracker.export_to_text(str(text_path))

    print(f"\nError reports exported:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {text_path}")


def test_error_context_manager():
    """Test error_context context manager."""
    print("\n" + "=" * 80)
    print("TEST 4: Error Context Manager")
    print("=" * 80)

    logger = setup_logger("test", level="INFO", file_output=False)
    tracker = ErrorTracker()

    # Test 1: Error caught and tracked
    print("\nTest 4.1: Catching and tracking error")
    with error_context(tracker, "Test operation", logger=logger):
        raise ValueError("This error will be caught and tracked")

    print(f"Errors tracked: {len(tracker)}")

    # Test 2: Multiple errors
    print("\nTest 4.2: Multiple errors in different contexts")
    for i in range(3):
        with error_context(tracker, f"Batch {i}", chunk_num=i, logger=logger):
            if i % 2 == 0:
                raise ValueError(f"Error in batch {i}")

    print(f"Errors tracked: {len(tracker)}")

    # Test 3: Critical error (should be re-raised)
    print("\nTest 4.3: Critical error (re-raised)")
    try:
        with error_context(tracker, "Critical test", logger=logger, raise_on_critical=True):
            raise MemoryError("Out of memory!")
    except MemoryError:
        print("Critical error was re-raised as expected")

    print(f"Total errors tracked: {len(tracker)}")


def test_logging():
    """Test logging system."""
    print("\n" + "=" * 80)
    print("TEST 5: Logging System")
    print("=" * 80)

    # Create logger
    logger = setup_logger(
        "test_logger",
        level="DEBUG",
        log_dir="./logs/test",
        console_output=True,
        file_output=True
    )

    # Test different log levels
    print("\nLogging at different levels:")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    try:
        raise ValueError("Test exception")
    except Exception:
        logger.exception("This is an EXCEPTION message with traceback")

    print("\nLogs written to: ./logs/test/")


def test_integration():
    """Test integration with actual processing."""
    print("\n" + "=" * 80)
    print("TEST 6: Integration Example")
    print("=" * 80)

    logger = setup_logger("integration_test", level="INFO", file_output=False)
    tracker = ErrorTracker()

    # Simulate processing with error handling
    print("\nSimulating batch processing with error handling:")

    posts = [
        {"id": "1", "text": "Valid post"},
        {"id": "2", "text": ""},  # Empty text - will cause error
        {"id": "3", "text": "Another valid post"},
        {"id": "4"},  # Missing text field - will cause error
        {"id": "5", "text": "Valid post"},
    ]

    processed = 0
    for post in posts:
        try:
            # Simulate processing
            if not post.get("text"):
                raise ValueError("Empty or missing text")

            # Processing logic here
            processed += 1
            logger.debug(f"Processed post {post['id']}")

        except Exception as e:
            tracker.add_error(
                e,
                context="Post processing",
                post_id=post.get("id"),
                severity="WARNING"
            )
            logger.warning(f"Skipping post {post.get('id')}: {e}")
            continue

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}/{len(posts)} posts")
    print(f"  Errors: {len(tracker)}")

    if tracker.has_errors():
        print(f"\nError summary:")
        summary = tracker.get_error_summary()
        for error_type, count in summary['by_type'].items():
            print(f"  {error_type}: {count}")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "ERROR HANDLING SYSTEM TEST SUITE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    try:
        test_custom_exceptions()
        test_error_conversion()
        test_error_tracker()
        test_error_context_manager()
        test_logging()
        test_integration()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✅")
        print("=" * 80)
        print("\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

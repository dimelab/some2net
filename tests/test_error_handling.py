"""
Unit tests for error handling system.

Tests custom exceptions, error tracking, and logging functionality.
"""

import pytest
from pathlib import Path
import tempfile

from src.core.exceptions import (
    SNAException,
    UserError,
    ProcessingError,
    CriticalError,
    FileNotFoundError,
    ColumnNotFoundError,
    InvalidFileFormatError,
    NERProcessingError,
    OutOfMemoryError,
    handle_error,
    format_error_for_user
)
from src.utils.logger import (
    setup_logger,
    ErrorTracker,
    error_context,
    get_log_files,
    cleanup_old_logs
)


# =============================================================================
# Exception Tests
# =============================================================================

class TestCustomExceptions:
    """Test custom exception classes."""

    def test_sna_exception_base(self):
        """Test base SNAException class."""
        exc = SNAException("Test message", "Test details")

        assert exc.message == "Test message"
        assert exc.details == "Test details"
        assert "Test message" in str(exc)

    def test_file_not_found_error(self):
        """Test FileNotFoundError exception."""
        exc = FileNotFoundError("/path/to/missing.csv")

        assert exc.filepath == "/path/to/missing.csv"
        assert "missing.csv" in exc.message
        assert "not found" in exc.message.lower()

    def test_column_not_found_error(self):
        """Test ColumnNotFoundError exception."""
        exc = ColumnNotFoundError("username", ["user", "text", "id"])

        assert exc.column_name == "username"
        assert exc.available_columns == ["user", "text", "id"]
        assert "username" in exc.message
        assert "user" in exc.details

    def test_invalid_file_format_error(self):
        """Test InvalidFileFormatError exception."""
        exc = InvalidFileFormatError("data.xlsx", [".csv", ".ndjson"])

        assert exc.filepath == "data.xlsx"
        assert exc.expected_formats == [".csv", ".ndjson"]
        assert "xlsx" in exc.message
        assert ".csv" in exc.details

    def test_ner_processing_error(self):
        """Test NERProcessingError exception."""
        exc = NERProcessingError("Batch failed", batch_index=5)

        assert exc.batch_index == 5
        assert "failed" in exc.message.lower()

    def test_out_of_memory_error(self):
        """Test OutOfMemoryError exception."""
        exc = OutOfMemoryError("GPU processing", "Try reducing batch size")

        assert exc.operation == "GPU processing"
        assert "batch size" in exc.details

    def test_user_error_inheritance(self):
        """Test that user errors inherit from UserError."""
        assert issubclass(FileNotFoundError, UserError)
        assert issubclass(ColumnNotFoundError, UserError)
        assert issubclass(InvalidFileFormatError, UserError)

    def test_processing_error_inheritance(self):
        """Test that processing errors inherit from ProcessingError."""
        assert issubclass(NERProcessingError, ProcessingError)

    def test_critical_error_inheritance(self):
        """Test that critical errors inherit from CriticalError."""
        assert issubclass(OutOfMemoryError, CriticalError)


# =============================================================================
# Error Utility Tests
# =============================================================================

class TestErrorUtilities:
    """Test error handling utility functions."""

    def test_format_error_for_user_simple(self):
        """Test formatting error without details."""
        exc = FileNotFoundError("test.csv")
        formatted = format_error_for_user(exc, include_details=False)

        assert "❌" in formatted
        assert "test.csv" in formatted
        assert exc.details not in formatted

    def test_format_error_for_user_with_details(self):
        """Test formatting error with details."""
        exc = ColumnNotFoundError("username", ["user", "text"])
        formatted = format_error_for_user(exc, include_details=True)

        assert "❌" in formatted
        assert "username" in formatted
        assert "user" in formatted  # Details included

    def test_handle_error_file_not_found(self):
        """Test converting FileNotFoundError."""
        original = FileNotFoundError("test.txt")
        converted = handle_error(original)

        assert isinstance(converted, SNAException)

    def test_handle_error_unicode_decode(self):
        """Test converting UnicodeDecodeError."""
        original = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
        converted = handle_error(original, context="file loading")

        assert isinstance(converted, SNAException)
        assert "file loading" in str(converted) or hasattr(converted, 'details')

    def test_handle_error_memory_error(self):
        """Test converting MemoryError."""
        original = MemoryError("Out of memory")
        converted = handle_error(original)

        assert isinstance(converted, SNAException)

    def test_handle_error_with_logger(self):
        """Test error handling with logger."""
        import logging
        logger = logging.getLogger("test")

        original = ValueError("Test error")
        converted = handle_error(original, logger=logger, context="testing")

        assert isinstance(converted, SNAException)


# =============================================================================
# Error Tracker Tests
# =============================================================================

class TestErrorTracker:
    """Test ErrorTracker functionality."""

    def test_initialization(self):
        """Test ErrorTracker initialization."""
        tracker = ErrorTracker(max_errors=100)

        assert len(tracker) == 0
        assert not tracker.has_errors()
        assert not tracker.has_critical_errors()

    def test_add_error(self):
        """Test adding errors to tracker."""
        tracker = ErrorTracker()

        tracker.add_error(
            ValueError("Test error"),
            context="Testing",
            severity="ERROR"
        )

        assert len(tracker) == 1
        assert tracker.has_errors()

    def test_add_multiple_errors(self):
        """Test adding multiple errors."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError("Error 1"), severity="WARNING")
        tracker.add_error(TypeError("Error 2"), severity="ERROR")
        tracker.add_error(MemoryError("Error 3"), severity="CRITICAL")

        assert len(tracker) == 3
        assert tracker.has_critical_errors()

    def test_get_error_summary(self):
        """Test getting error summary."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError("Error 1"), severity="WARNING")
        tracker.add_error(TypeError("Error 2"), severity="ERROR")
        tracker.add_error(TypeError("Error 3"), severity="ERROR")

        summary = tracker.get_error_summary()

        assert summary['total_errors'] == 3
        assert summary['by_severity']['WARNING'] == 1
        assert summary['by_severity']['ERROR'] == 2
        assert summary['by_type']['ValueError'] == 1
        assert summary['by_type']['TypeError'] == 2

    def test_get_errors_filtered(self):
        """Test getting filtered errors."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError("Error 1"), severity="WARNING")
        tracker.add_error(TypeError("Error 2"), severity="ERROR")
        tracker.add_error(MemoryError("Error 3"), severity="CRITICAL")

        critical_errors = tracker.get_errors(severity="CRITICAL")

        assert len(critical_errors) == 1
        assert critical_errors[0]['type'] == 'MemoryError'

    def test_export_to_json(self, temp_dir):
        """Test exporting errors to JSON."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError("Test error"), context="Testing")

        json_path = temp_dir / "errors.json"
        tracker.export_to_json(str(json_path))

        assert json_path.exists()
        assert json_path.stat().st_size > 0

    def test_export_to_text(self, temp_dir):
        """Test exporting errors to text."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError("Test error"), context="Testing")

        text_path = temp_dir / "errors.txt"
        tracker.export_to_text(str(text_path))

        assert text_path.exists()
        assert text_path.stat().st_size > 0

        # Check content
        content = text_path.read_text()
        assert "ERROR REPORT" in content
        assert "Test error" in content

    def test_max_errors_limit(self):
        """Test that max_errors limit is enforced."""
        tracker = ErrorTracker(max_errors=5)

        for i in range(10):
            tracker.add_error(ValueError(f"Error {i}"))

        assert len(tracker) == 5  # Should not exceed max

    def test_clear_errors(self):
        """Test clearing errors."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError("Error 1"))
        tracker.add_error(TypeError("Error 2"))

        assert len(tracker) == 2

        tracker.clear()

        assert len(tracker) == 0
        assert not tracker.has_errors()

    def test_error_with_context(self):
        """Test error tracking with context."""
        tracker = ErrorTracker()

        tracker.add_error(
            ValueError("Test"),
            context="Data processing",
            post_id="12345",
            chunk_num=3,
            severity="ERROR"
        )

        errors = tracker.get_errors()

        assert errors[0]['context'] == "Data processing"
        assert errors[0]['post_id'] == "12345"
        assert errors[0]['chunk_num'] == 3


# =============================================================================
# Error Context Manager Tests
# =============================================================================

class TestErrorContext:
    """Test error_context context manager."""

    def test_error_caught_and_tracked(self):
        """Test that errors are caught and tracked."""
        tracker = ErrorTracker()

        with error_context(tracker, "Test context"):
            raise ValueError("Test error")

        assert len(tracker) == 1
        assert tracker.get_errors()[0]['context'] == "Test context"

    def test_multiple_contexts(self):
        """Test multiple error contexts."""
        tracker = ErrorTracker()

        for i in range(3):
            with error_context(tracker, f"Context {i}"):
                if i % 2 == 0:
                    raise ValueError(f"Error {i}")

        assert len(tracker) == 2  # Only errors from i=0 and i=2

    def test_critical_error_reraised(self):
        """Test that critical errors are re-raised."""
        tracker = ErrorTracker()

        with pytest.raises(MemoryError):
            with error_context(tracker, "Test", raise_on_critical=True):
                raise MemoryError("Out of memory")

        assert len(tracker) == 1

    def test_non_critical_suppressed(self):
        """Test that non-critical errors are suppressed."""
        tracker = ErrorTracker()

        # Should not raise
        with error_context(tracker, "Test"):
            raise ValueError("Non-critical error")

        assert len(tracker) == 1

    def test_with_logger(self):
        """Test error context with logger."""
        import logging
        logger = logging.getLogger("test")
        tracker = ErrorTracker()

        with error_context(tracker, "Test context", logger=logger):
            raise ValueError("Test error")

        assert len(tracker) == 1


# =============================================================================
# Logger Tests
# =============================================================================

class TestLogger:
    """Test logging functionality."""

    def test_setup_logger_basic(self, temp_dir):
        """Test basic logger setup."""
        logger = setup_logger(
            "test_logger",
            level="INFO",
            log_dir=str(temp_dir),
            console_output=False,
            file_output=True
        )

        assert logger is not None
        assert logger.name == "test_logger"

        # Test logging
        logger.info("Test message")

        # Check log file created
        log_files = list(temp_dir.glob("*.log"))
        assert len(log_files) > 0

    def test_logger_levels(self, temp_dir):
        """Test different log levels."""
        logger = setup_logger(
            "test_levels",
            level="DEBUG",
            log_dir=str(temp_dir),
            console_output=False
        )

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # All should be logged
        log_files = list(temp_dir.glob("*.log"))
        assert len(log_files) > 0

    def test_error_log_separation(self, temp_dir):
        """Test that errors go to separate error log."""
        logger = setup_logger(
            "test_errors",
            level="DEBUG",
            log_dir=str(temp_dir),
            file_output=True
        )

        logger.info("Info message")
        logger.error("Error message")

        # Should have both regular log and errors.log
        error_log = temp_dir / "errors.log"
        assert error_log.exists()

    def test_console_output_disabled(self, temp_dir, capsys):
        """Test logger with console output disabled."""
        logger = setup_logger(
            "test_no_console",
            level="INFO",
            log_dir=str(temp_dir),
            console_output=False
        )

        logger.info("Test message")

        captured = capsys.readouterr()
        assert "Test message" not in captured.out


# =============================================================================
# Integration Tests
# =============================================================================

class TestErrorHandlingIntegration:
    """Integration tests for error handling system."""

    def test_complete_error_flow(self, temp_dir):
        """Test complete error handling flow."""
        logger = setup_logger(
            "integration_test",
            log_dir=str(temp_dir),
            console_output=False
        )
        tracker = ErrorTracker()

        # Simulate processing with errors
        posts = [
            {"id": "1", "text": "Valid post"},
            {"id": "2", "text": ""},  # Empty - will cause error
            {"id": "3", "text": "Valid post"},
            {"id": "4"},  # Missing text - will cause error
        ]

        for post in posts:
            try:
                if not post.get("text"):
                    raise ValueError("Empty or missing text")
                # Processing...
            except Exception as e:
                tracker.add_error(
                    e,
                    context="Post processing",
                    post_id=post.get("id"),
                    severity="WARNING"
                )
                logger.warning(f"Skipping post {post.get('id')}: {e}")

        # Verify tracking
        assert len(tracker) == 2
        assert tracker.has_errors()

        # Export report
        report_path = temp_dir / "error_report.json"
        tracker.export_to_json(str(report_path))

        assert report_path.exists()

    def test_error_recovery_pattern(self):
        """Test graceful error recovery pattern."""
        tracker = ErrorTracker()

        processed = []
        failed = []

        data = [1, 2, "invalid", 4, "error", 6]

        for item in data:
            try:
                # Simulate processing
                result = int(item) * 2
                processed.append(result)
            except ValueError as e:
                tracker.add_error(
                    e,
                    context="Data conversion",
                    severity="WARNING"
                )
                failed.append(item)

        # Should have processed valid items and tracked errors
        assert len(processed) == 4  # 1, 2, 4, 6
        assert len(failed) == 2  # "invalid", "error"
        assert len(tracker) == 2


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.edge_case
class TestErrorHandlingEdgeCases:
    """Test edge cases in error handling."""

    def test_empty_error_message(self):
        """Test handling empty error messages."""
        tracker = ErrorTracker()

        tracker.add_error(ValueError(""), context="Test")

        assert len(tracker) == 1

    def test_very_long_error_message(self):
        """Test handling very long error messages."""
        tracker = ErrorTracker()

        long_message = "A" * 10000
        tracker.add_error(ValueError(long_message), context="Test")

        assert len(tracker) == 1

    def test_unicode_in_errors(self):
        """Test handling Unicode in error messages."""
        tracker = ErrorTracker()

        tracker.add_error(
            ValueError("Ошибка: 错误 مخطئ"),
            context="Unicode test"
        )

        assert len(tracker) == 1

        # Export should handle Unicode
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unicode_errors.json"
            tracker.export_to_json(str(path))
            assert path.exists()

    def test_nested_exceptions(self):
        """Test handling nested exceptions."""
        tracker = ErrorTracker()

        try:
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise RuntimeError("Outer error") from e
        except Exception as e:
            tracker.add_error(e, context="Nested test")

        assert len(tracker) == 1

    def test_concurrent_error_tracking(self):
        """Test error tracking with simulated concurrent access."""
        tracker = ErrorTracker()

        # Simulate multiple sources adding errors
        for i in range(10):
            tracker.add_error(
                ValueError(f"Error {i}"),
                context=f"Thread {i % 3}"
            )

        assert len(tracker) == 10

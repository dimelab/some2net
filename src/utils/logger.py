"""
Logging Utility Module

Provides centralized logging configuration and error tracking
for the Social Network Analytics application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import json
import traceback


# ============================================================================
# Logger Setup
# ============================================================================

def setup_logger(
    name: str = "sna",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up and configure logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional specific log file name
        log_dir: Directory for log files
        console_output: Whether to output to console
        file_output: Whether to output to file

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Generate log filename if not provided
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            log_file = f"sna_{timestamp}.log"

        file_path = log_path / log_file

        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Error file handler (separate file for errors only)
    if file_output:
        error_file = log_path / "errors.log"
        error_handler = logging.FileHandler(error_file, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    logger.info(f"Logger '{name}' initialized at level {level}")
    if file_output:
        logger.debug(f"Logging to file: {file_path}")

    return logger


# ============================================================================
# Error Tracker
# ============================================================================

class ErrorTracker:
    """
    Track and accumulate errors during processing for later reporting.
    """

    def __init__(self, max_errors: int = 1000):
        """
        Initialize error tracker.

        Args:
            max_errors: Maximum number of errors to track (prevents memory issues)
        """
        self.errors: List[Dict] = []
        self.max_errors = max_errors
        self.error_counts: Dict[str, int] = {}

    def add_error(
        self,
        error: Exception,
        context: str = None,
        post_id: str = None,
        chunk_num: int = None,
        severity: str = "ERROR"
    ):
        """
        Add an error to the tracker.

        Args:
            error: Exception object
            context: Context where error occurred
            post_id: Optional post ID where error occurred
            chunk_num: Optional chunk number
            severity: Error severity ('WARNING', 'ERROR', 'CRITICAL')
        """
        # Don't track more than max_errors
        if len(self.errors) >= self.max_errors:
            return

        error_type = type(error).__name__
        error_msg = str(error)

        # Track error counts by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Create error record
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_msg,
            'severity': severity,
            'context': context,
            'post_id': post_id,
            'chunk_num': chunk_num,
            'traceback': traceback.format_exc() if severity == 'CRITICAL' else None
        }

        self.errors.append(error_record)

    def get_errors(self, severity: str = None) -> List[Dict]:
        """
        Get tracked errors, optionally filtered by severity.

        Args:
            severity: Optional severity filter

        Returns:
            List of error records
        """
        if severity:
            return [e for e in self.errors if e['severity'] == severity]
        return self.errors

    def get_error_summary(self) -> Dict:
        """
        Get summary of tracked errors.

        Returns:
            Dictionary with error summary
        """
        severity_counts = {}
        for error in self.errors:
            sev = error['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            'total_errors': len(self.errors),
            'by_severity': severity_counts,
            'by_type': self.error_counts.copy(),
            'truncated': len(self.errors) >= self.max_errors
        }

    def has_errors(self) -> bool:
        """Check if any errors were tracked."""
        return len(self.errors) > 0

    def has_critical_errors(self) -> bool:
        """Check if any critical errors were tracked."""
        return any(e['severity'] == 'CRITICAL' for e in self.errors)

    def export_to_json(self, filepath: str):
        """
        Export errors to JSON file.

        Args:
            filepath: Path to output file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_error_summary(),
            'errors': self.errors
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def export_to_text(self, filepath: str):
        """
        Export errors to human-readable text file.

        Args:
            filepath: Path to output file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Errors: {len(self.errors)}\n")
            f.write("\n")

            # Summary
            summary = self.get_error_summary()
            f.write("SUMMARY BY SEVERITY:\n")
            f.write("-" * 40 + "\n")
            for severity, count in summary['by_severity'].items():
                f.write(f"  {severity}: {count}\n")
            f.write("\n")

            f.write("SUMMARY BY TYPE:\n")
            f.write("-" * 40 + "\n")
            for error_type, count in summary['by_type'].items():
                f.write(f"  {error_type}: {count}\n")
            f.write("\n")

            # Individual errors
            f.write("=" * 80 + "\n")
            f.write("DETAILED ERRORS\n")
            f.write("=" * 80 + "\n\n")

            for i, error in enumerate(self.errors, 1):
                f.write(f"Error #{i}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Timestamp: {error['timestamp']}\n")
                f.write(f"Severity: {error['severity']}\n")
                f.write(f"Type: {error['type']}\n")
                f.write(f"Message: {error['message']}\n")

                if error.get('context'):
                    f.write(f"Context: {error['context']}\n")
                if error.get('post_id'):
                    f.write(f"Post ID: {error['post_id']}\n")
                if error.get('chunk_num'):
                    f.write(f"Chunk: {error['chunk_num']}\n")

                if error.get('traceback'):
                    f.write(f"\nTraceback:\n{error['traceback']}\n")

                f.write("\n")

    def clear(self):
        """Clear all tracked errors."""
        self.errors.clear()
        self.error_counts.clear()

    def __len__(self):
        """Return number of tracked errors."""
        return len(self.errors)

    def __bool__(self):
        """Return True if errors exist."""
        return self.has_errors()


# ============================================================================
# Context Manager for Error Tracking
# ============================================================================

class error_context:
    """
    Context manager for tracking errors in a specific context.

    Usage:
        with error_context(tracker, "NER processing", chunk_num=5):
            # ... code that might raise errors ...
    """

    def __init__(
        self,
        tracker: ErrorTracker,
        context: str,
        post_id: str = None,
        chunk_num: int = None,
        logger: logging.Logger = None,
        raise_on_critical: bool = True
    ):
        """
        Initialize error context.

        Args:
            tracker: ErrorTracker instance
            context: Description of the context
            post_id: Optional post ID
            chunk_num: Optional chunk number
            logger: Optional logger for logging errors
            raise_on_critical: Whether to re-raise critical errors
        """
        self.tracker = tracker
        self.context = context
        self.post_id = post_id
        self.chunk_num = chunk_num
        self.logger = logger
        self.raise_on_critical = raise_on_critical

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Determine severity
            severity = "ERROR"
            if isinstance(exc_val, (MemoryError, KeyboardInterrupt)):
                severity = "CRITICAL"

            # Add to tracker
            self.tracker.add_error(
                exc_val,
                context=self.context,
                post_id=self.post_id,
                chunk_num=self.chunk_num,
                severity=severity
            )

            # Log if logger provided
            if self.logger:
                self.logger.error(
                    f"Error in {self.context}: {exc_val}",
                    exc_info=severity == "CRITICAL"
                )

            # Re-raise critical errors
            if severity == "CRITICAL" and self.raise_on_critical:
                return False

            # Suppress non-critical errors
            return True

        return False


# ============================================================================
# Utility Functions
# ============================================================================

def get_log_files(log_dir: str = "./logs") -> List[Path]:
    """
    Get list of log files in log directory.

    Args:
        log_dir: Log directory path

    Returns:
        List of log file paths
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    return list(log_path.glob("*.log"))


def cleanup_old_logs(log_dir: str = "./logs", days_to_keep: int = 30):
    """
    Remove log files older than specified number of days.

    Args:
        log_dir: Log directory path
        days_to_keep: Number of days to keep logs
    """
    import time

    log_path = Path(log_dir)
    if not log_path.exists():
        return

    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    removed_count = 0

    for log_file in log_path.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            removed_count += 1

    if removed_count > 0:
        logger = logging.getLogger("sna")
        logger.info(f"Cleaned up {removed_count} old log files")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("sna", level="DEBUG", log_dir="./logs")

    # Test logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test error tracker
    tracker = ErrorTracker()

    # Simulate some errors
    tracker.add_error(
        ValueError("Test error 1"),
        context="Data loading",
        chunk_num=1,
        severity="WARNING"
    )

    tracker.add_error(
        FileNotFoundError("test.csv"),
        context="File validation",
        severity="ERROR"
    )

    tracker.add_error(
        MemoryError("Out of memory"),
        context="NER processing",
        chunk_num=5,
        severity="CRITICAL"
    )

    # Print summary
    print("\nError Summary:")
    print(json.dumps(tracker.get_error_summary(), indent=2))

    # Export errors
    tracker.export_to_text("./logs/error_report.txt")
    tracker.export_to_json("./logs/error_report.json")
    print("\nError reports exported")

    # Test context manager
    print("\nTesting context manager:")
    with error_context(tracker, "Test context", logger=logger):
        # This error will be caught and tracked
        raise ValueError("Test error in context")

    print(f"Errors tracked: {len(tracker)}")

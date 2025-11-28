"""
Custom Exception Classes

Provides a hierarchy of custom exceptions for better error handling
and user-friendly error messages throughout the application.
"""


# ============================================================================
# Base Exceptions
# ============================================================================

class SNAException(Exception):
    """Base exception for all Social Network Analytics errors."""

    def __init__(self, message: str, details: str = None):
        """
        Initialize exception with message and optional details.

        Args:
            message: User-friendly error message
            details: Technical details for logging/debugging
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


# ============================================================================
# Input/Validation Errors - User-facing errors
# ============================================================================

class UserError(SNAException):
    """Base class for user-facing errors (invalid input, missing files, etc.)."""
    pass


class FileNotFoundError(UserError):
    """File does not exist."""

    def __init__(self, filepath: str):
        message = f"File not found: {filepath}"
        details = "Please check the file path and ensure the file exists."
        super().__init__(message, details)
        self.filepath = filepath


class InvalidFileFormatError(UserError):
    """File format is not supported."""

    def __init__(self, filepath: str, expected_formats: list):
        formats_str = ", ".join(expected_formats)
        message = f"Invalid file format: {filepath}"
        details = f"Supported formats: {formats_str}"
        super().__init__(message, details)
        self.filepath = filepath
        self.expected_formats = expected_formats


class ColumnNotFoundError(UserError):
    """Required column not found in data."""

    def __init__(self, column_name: str, available_columns: list):
        message = f"Column '{column_name}' not found in data"
        cols_str = ", ".join([f"'{c}'" for c in available_columns[:10]])
        if len(available_columns) > 10:
            cols_str += f" (and {len(available_columns) - 10} more)"
        details = f"Available columns: {cols_str}"
        super().__init__(message, details)
        self.column_name = column_name
        self.available_columns = available_columns


class EmptyDataError(UserError):
    """Data file is empty or contains no valid records."""

    def __init__(self, filepath: str):
        message = f"No valid data found in file: {filepath}"
        details = "The file appears to be empty or contains no processable records."
        super().__init__(message, details)
        self.filepath = filepath


class EncodingError(UserError):
    """File encoding issues."""

    def __init__(self, filepath: str, attempted_encodings: list = None):
        message = f"Unable to read file with supported encodings: {filepath}"
        if attempted_encodings:
            enc_str = ", ".join(attempted_encodings)
            details = f"Tried encodings: {enc_str}"
        else:
            details = "File encoding could not be determined."
        super().__init__(message, details)
        self.filepath = filepath


# ============================================================================
# Processing Errors - Recoverable errors
# ============================================================================

class ProcessingError(SNAException):
    """Base class for processing errors that can be recovered from."""
    pass


class NERProcessingError(ProcessingError):
    """Error during NER extraction."""

    def __init__(self, message: str, batch_index: int = None):
        details = f"Batch index: {batch_index}" if batch_index is not None else None
        super().__init__(message, details)
        self.batch_index = batch_index


class EntityResolutionError(ProcessingError):
    """Error during entity resolution/deduplication."""

    def __init__(self, entity_text: str, reason: str):
        message = f"Could not resolve entity: {entity_text}"
        details = f"Reason: {reason}"
        super().__init__(message, details)
        self.entity_text = entity_text


class NetworkConstructionError(ProcessingError):
    """Error during network graph construction."""

    def __init__(self, message: str, post_id: str = None):
        details = f"Post ID: {post_id}" if post_id else None
        super().__init__(message, details)
        self.post_id = post_id


class ExportError(ProcessingError):
    """Error during network export."""

    def __init__(self, format_name: str, reason: str):
        message = f"Failed to export to {format_name} format"
        details = f"Reason: {reason}"
        super().__init__(message, details)
        self.format_name = format_name


# ============================================================================
# System Errors - Critical errors
# ============================================================================

class CriticalError(SNAException):
    """Base class for critical system errors."""
    pass


class ModelLoadError(CriticalError):
    """Failed to load NER model."""

    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load NER model: {model_name}"
        details = f"Reason: {reason}"
        super().__init__(message, details)
        self.model_name = model_name


class GPUError(CriticalError):
    """GPU/CUDA error."""

    def __init__(self, message: str, fallback_available: bool = True):
        details = "Falling back to CPU processing" if fallback_available else "No fallback available"
        super().__init__(message, details)
        self.fallback_available = fallback_available


class OutOfMemoryError(CriticalError):
    """System out of memory."""

    def __init__(self, operation: str, suggestion: str = None):
        message = f"Out of memory during {operation}"
        details = suggestion or "Try reducing batch size or chunk size"
        super().__init__(message, details)
        self.operation = operation


class DiskSpaceError(CriticalError):
    """Insufficient disk space."""

    def __init__(self, required_mb: float = None):
        message = "Insufficient disk space for operation"
        if required_mb:
            details = f"Required: ~{required_mb:.1f} MB"
        else:
            details = "Free up disk space and try again"
        super().__init__(message, details)


# ============================================================================
# Network/Download Errors
# ============================================================================

class NetworkError(SNAException):
    """Base class for network-related errors."""
    pass


class ModelDownloadError(NetworkError):
    """Failed to download model from HuggingFace."""

    def __init__(self, model_name: str, reason: str):
        message = f"Failed to download model: {model_name}"
        details = f"Reason: {reason}\nCheck internet connection and try again."
        super().__init__(message, details)
        self.model_name = model_name


class CacheError(NetworkError):
    """Error with caching system."""

    def __init__(self, operation: str, reason: str):
        message = f"Cache error during {operation}"
        details = f"Reason: {reason}"
        super().__init__(message, details)


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(SNAException):
    """Base class for configuration errors."""
    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value."""

    def __init__(self, param_name: str, value, expected: str):
        message = f"Invalid configuration for '{param_name}': {value}"
        details = f"Expected: {expected}"
        super().__init__(message, details)
        self.param_name = param_name
        self.value = value


class MissingConfigError(ConfigurationError):
    """Required configuration missing."""

    def __init__(self, param_name: str):
        message = f"Missing required configuration: {param_name}"
        details = "Please provide this parameter or update config file"
        super().__init__(message, details)
        self.param_name = param_name


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(UserError):
    """Base class for validation errors."""
    pass


class ThresholdValidationError(ValidationError):
    """Invalid threshold value."""

    def __init__(self, param_name: str, value: float, min_val: float = 0.0, max_val: float = 1.0):
        message = f"Invalid {param_name}: {value}"
        details = f"Must be between {min_val} and {max_val}"
        super().__init__(message, details)
        self.param_name = param_name
        self.value = value


class BatchSizeValidationError(ValidationError):
    """Invalid batch size."""

    def __init__(self, value: int):
        message = f"Invalid batch size: {value}"
        details = "Batch size must be a positive integer (typically 1-128)"
        super().__init__(message, details)
        self.value = value


# ============================================================================
# Helper Functions
# ============================================================================

def handle_error(error: Exception, logger = None, context: str = None) -> SNAException:
    """
    Convert standard exceptions to custom exceptions with better messages.

    Args:
        error: Original exception
        logger: Optional logger for logging
        context: Optional context string

    Returns:
        SNAException or subclass
    """
    import os

    # Add context to message if provided
    ctx_msg = f" [{context}]" if context else ""

    # File errors
    if isinstance(error, FileNotFoundError):
        exc = FileNotFoundError(str(error.filename) if hasattr(error, 'filename') else str(error))

    # Encoding errors
    elif isinstance(error, UnicodeDecodeError):
        exc = EncodingError("Unknown file", attempted_encodings=['utf-8', 'latin-1', 'cp1252'])

    # Memory errors
    elif isinstance(error, MemoryError):
        exc = OutOfMemoryError("data processing")

    # Disk space errors
    elif isinstance(error, OSError) and error.errno == 28:  # No space left
        exc = DiskSpaceError()

    # Network errors
    elif "Connection" in str(error) or "HTTP" in str(error):
        exc = ModelDownloadError("unknown", str(error))

    # Generic processing error
    elif isinstance(error, (ValueError, TypeError, KeyError)):
        exc = ProcessingError(f"Processing error{ctx_msg}: {str(error)}")

    # Already a custom exception
    elif isinstance(error, SNAException):
        exc = error

    # Unknown error
    else:
        exc = SNAException(f"Unexpected error{ctx_msg}: {str(error)}",
                          details=f"Type: {type(error).__name__}")

    # Log if logger provided
    if logger:
        logger.error(str(exc))
        if hasattr(exc, 'details') and exc.details:
            logger.debug(f"Error details: {exc.details}")

    return exc


def format_error_for_user(error: Exception, include_details: bool = False) -> str:
    """
    Format error message for display to user.

    Args:
        error: Exception to format
        include_details: Whether to include technical details

    Returns:
        Formatted error message
    """
    if isinstance(error, SNAException):
        msg = f"❌ Error: {error.message}"
        if include_details and error.details:
            msg += f"\n   {error.details}"
        return msg
    else:
        return f"❌ Error: {str(error)}"


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: User errors
    try:
        raise ColumnNotFoundError("username", ["user", "text", "timestamp"])
    except SNAException as e:
        print(format_error_for_user(e, include_details=True))

    print()

    # Example 2: Processing errors
    try:
        raise NERProcessingError("Model inference failed", batch_index=5)
    except SNAException as e:
        print(format_error_for_user(e, include_details=True))

    print()

    # Example 3: Critical errors
    try:
        raise OutOfMemoryError("NER processing", "Try reducing batch size to 16")
    except SNAException as e:
        print(format_error_for_user(e, include_details=True))

    print()

    # Example 4: Error conversion
    try:
        open("nonexistent.txt")
    except Exception as e:
        custom_exc = handle_error(e, context="file loading")
        print(format_error_for_user(custom_exc, include_details=True))

"""
Data Loader Module

Handles loading of CSV and NDJSON files with:
- Chunked reading for memory efficiency
- Automatic encoding detection
- Column validation
- Support for large files
"""

import pandas as pd
import json
from typing import Iterator, Optional, List, Union
from pathlib import Path
import chardet
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads social media data from CSV or NDJSON files.

    Features:
    - Chunked reading for large files
    - Automatic encoding detection
    - Column validation
    - Memory-efficient streaming
    """

    def __init__(self):
        """Initialize DataLoader."""
        self.supported_formats = ['.csv', '.ndjson', '.jsonl']

    def detect_encoding(self, filepath: Union[str, Path], sample_size: int = 10000) -> str:
        """
        Detect file encoding using chardet.

        Args:
            filepath: Path to file
            sample_size: Number of bytes to sample for detection

        Returns:
            Detected encoding string (e.g., 'utf-8', 'latin-1')
        """
        filepath = Path(filepath)

        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']

                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

                # Default to utf-8 if detection is uncertain
                if confidence < 0.7:
                    logger.warning(f"Low confidence encoding detection. Defaulting to utf-8")
                    return 'utf-8'

                return encoding

        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}. Defaulting to utf-8")
            return 'utf-8'

    def validate_columns(
        self,
        df: pd.DataFrame,
        author_column: str,
        text_column: str
    ) -> bool:
        """
        Validate that required columns exist in dataframe.

        Args:
            df: DataFrame to validate
            author_column: Name of author column
            text_column: Name of text column

        Returns:
            True if valid, raises ValueError if not

        Raises:
            ValueError: If required columns are missing
        """
        available_columns = df.columns.tolist()

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        available_columns = df.columns.tolist()

        missing_columns = []

        if author_column not in available_columns:
            missing_columns.append(author_column)

        if text_column not in available_columns:
            missing_columns.append(text_column)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_columns}"
            )

        logger.info(f"Column validation passed. Using: author='{author_column}', text='{text_column}'")
        return True

    def load_csv(
        self,
        filepath: Union[str, Path],
        author_column: str,
        text_column: str,
        chunksize: int = 10000,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Load CSV file in chunks for memory efficiency.

        Args:
            filepath: Path to CSV file
            author_column: Name of column containing author information
            text_column: Name of column containing text content
            chunksize: Number of rows per chunk (default: 10000)
            encoding: File encoding (auto-detected if None)
            **kwargs: Additional arguments passed to pd.read_csv

        Yields:
            DataFrame chunks with validated columns

        Example:
            >>> loader = DataLoader()
            >>> for chunk in loader.load_csv('data.csv', 'author', 'text'):
            ...     print(f"Processing {len(chunk)} rows")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect encoding if not provided
        if encoding is None:
            encoding = self.detect_encoding(filepath)

        # Try multiple encodings if the first fails
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for enc in encodings_to_try:
            try:
                logger.info(f"Attempting to read CSV with encoding: {enc}")

                # Read first chunk to validate columns
                first_chunk = pd.read_csv(
                    filepath,
                    encoding=enc,
                    nrows=chunksize,
                    on_bad_lines='warn',  # Skip malformed rows with warning
                    **kwargs
                )

                # Handle duplicate column names by keeping only the first occurrence
                if first_chunk.columns.duplicated().any():
                    logger.warning(f"Duplicate column names detected. Keeping only first occurrence.")
                    # Get mask of duplicate columns
                    dup_mask = first_chunk.columns.duplicated(keep='first')
                    # Keep only non-duplicate columns
                    first_chunk = first_chunk.loc[:, ~dup_mask]

                # Validate columns exist
                self.validate_columns(first_chunk, author_column, text_column)

                # Now read full file in chunks
                chunk_iterator = pd.read_csv(
                    filepath,
                    encoding=enc,
                    chunksize=chunksize,
                    on_bad_lines='warn',  # Skip malformed rows with warning
                    **kwargs
                )

                chunk_count = 0
                for chunk in chunk_iterator:
                    # Strip whitespace from column names
                    chunk.columns = chunk.columns.str.strip()

                    # Handle duplicate column names by keeping only the first occurrence
                    if chunk.columns.duplicated().any():
                        dup_mask = chunk.columns.duplicated(keep='first')
                        chunk = chunk.loc[:, ~dup_mask]

                    # Handle missing values and ensure string type
                    chunk[author_column] = chunk[author_column].fillna('unknown').astype(str)
                    chunk[text_column] = chunk[text_column].fillna('').astype(str)

                    # Remove rows with empty text after converting to string
                    chunk = chunk[chunk[text_column].str.strip().str.len() > 0]

                    chunk_count += 1
                    logger.debug(f"Yielding chunk {chunk_count} with {len(chunk)} rows")

                    yield chunk

                logger.info(f"Successfully processed {chunk_count} chunks from CSV")
                return  # Success, exit the function

            except UnicodeDecodeError:
                logger.warning(f"Failed to read with encoding {enc}, trying next...")
                continue

            except Exception as e:
                if enc == encodings_to_try[-1]:  # Last encoding attempt
                    logger.error(f"Failed to read CSV with all encodings: {e}")
                    raise
                continue

        raise ValueError(f"Could not read CSV file with any supported encoding")

    def load_ndjson(
        self,
        filepath: Union[str, Path],
        author_column: str,
        text_column: str,
        chunksize: int = 10000,
        encoding: Optional[str] = None
    ) -> Iterator[pd.DataFrame]:
        """
        Load NDJSON (newline-delimited JSON) file in chunks.

        Args:
            filepath: Path to NDJSON file
            author_column: Name of field containing author information
            text_column: Name of field containing text content
            chunksize: Number of lines per chunk (default: 10000)
            encoding: File encoding (auto-detected if None)

        Yields:
            DataFrame chunks with validated columns

        Example:
            >>> loader = DataLoader()
            >>> for chunk in loader.load_ndjson('data.ndjson', 'author', 'text'):
            ...     print(f"Processing {len(chunk)} rows")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect encoding if not provided
        if encoding is None:
            encoding = self.detect_encoding(filepath)

        # Try multiple encodings
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252']

        for enc in encodings_to_try:
            try:
                logger.info(f"Attempting to read NDJSON with encoding: {enc}")

                chunk_data = []
                chunk_count = 0

                with open(filepath, 'r', encoding=enc) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()

                        if not line:  # Skip empty lines
                            continue

                        try:
                            record = json.loads(line)
                            chunk_data.append(record)

                            # Yield chunk when it reaches chunksize
                            if len(chunk_data) >= chunksize:
                                chunk_df = pd.DataFrame(chunk_data)

                                # Validate columns on first chunk
                                if chunk_count == 0:
                                    self.validate_columns(chunk_df, author_column, text_column)

                                # Handle missing values and ensure string type
                                chunk_df[author_column] = chunk_df[author_column].fillna('unknown').astype(str)
                                chunk_df[text_column] = chunk_df[text_column].fillna('').astype(str)

                                # Remove rows with empty text after converting to string
                                chunk_df = chunk_df[chunk_df[text_column].str.strip().str.len() > 0]

                                chunk_count += 1
                                logger.debug(f"Yielding chunk {chunk_count} with {len(chunk_df)} rows")

                                yield chunk_df
                                chunk_data = []  # Reset for next chunk

                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                            continue

                    # Yield remaining data as final chunk
                    if chunk_data:
                        chunk_df = pd.DataFrame(chunk_data)

                        # Validate columns if this is the only chunk
                        if chunk_count == 0:
                            self.validate_columns(chunk_df, author_column, text_column)

                        chunk_df[author_column] = chunk_df[author_column].fillna('unknown').astype(str)
                        chunk_df[text_column] = chunk_df[text_column].fillna('').astype(str)
                        chunk_df = chunk_df[chunk_df[text_column].str.strip().str.len() > 0]

                        chunk_count += 1
                        logger.debug(f"Yielding final chunk {chunk_count} with {len(chunk_df)} rows")

                        yield chunk_df

                logger.info(f"Successfully processed {chunk_count} chunks from NDJSON")
                return  # Success

            except UnicodeDecodeError:
                logger.warning(f"Failed to read with encoding {enc}, trying next...")
                continue

            except Exception as e:
                if enc == encodings_to_try[-1]:
                    logger.error(f"Failed to read NDJSON with all encodings: {e}")
                    raise
                continue

        raise ValueError(f"Could not read NDJSON file with any supported encoding")

    def load_file(
        self,
        filepath: Union[str, Path],
        author_column: str,
        text_column: str,
        chunksize: int = 10000,
        encoding: Optional[str] = None,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Auto-detect file format and load appropriately.

        Args:
            filepath: Path to data file
            author_column: Name of column/field containing author
            text_column: Name of column/field containing text
            chunksize: Number of rows per chunk
            encoding: File encoding (auto-detected if None)
            **kwargs: Additional arguments for CSV reader

        Yields:
            DataFrame chunks

        Raises:
            ValueError: If file format is not supported
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == '.csv':
            yield from self.load_csv(
                filepath, author_column, text_column, chunksize, encoding, **kwargs
            )

        elif suffix in ['.ndjson', '.jsonl']:
            yield from self.load_ndjson(
                filepath, author_column, text_column, chunksize, encoding
            )

        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.supported_formats}"
            )

    def get_column_names(
        self,
        filepath: Union[str, Path],
        encoding: Optional[str] = None
    ) -> List[str]:
        """
        Get list of column names from file (useful for UI column selection).

        Args:
            filepath: Path to data file
            encoding: File encoding (auto-detected if None)

        Returns:
            List of column names
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if encoding is None:
            encoding = self.detect_encoding(filepath)

        if suffix == '.csv':
            # Read just the header
            df = pd.read_csv(filepath, encoding=encoding, nrows=0)
            # Strip whitespace
            df.columns = df.columns.str.strip()
            # Handle duplicate column names by keeping only first occurrence
            if df.columns.duplicated().any():
                logger.warning(f"Duplicate column names detected in header. Keeping only first occurrence.")
                dup_mask = df.columns.duplicated(keep='first')
                df = df.loc[:, ~dup_mask]
            return df.columns.tolist()

        elif suffix in ['.ndjson', '.jsonl']:
            # Read first valid JSON line
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            return list(record.keys())
                        except json.JSONDecodeError:
                            continue

            raise ValueError("No valid JSON records found in NDJSON file")

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

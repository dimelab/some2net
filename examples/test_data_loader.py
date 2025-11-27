"""
Example script to test DataLoader functionality.

Run this after installing dependencies:
    pip install -r requirements.txt
"""

from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import DataLoader


def test_csv_loading():
    """Test loading CSV file."""
    print("=" * 60)
    print("Testing CSV Loading")
    print("=" * 60)

    loader = DataLoader()
    csv_file = Path(__file__).parent / 'sample_data.csv'

    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return

    print(f"Loading: {csv_file}")
    print()

    total_rows = 0
    chunk_count = 0

    for chunk in loader.load_csv(csv_file, 'author', 'text', chunksize=5):
        chunk_count += 1
        total_rows += len(chunk)
        print(f"Chunk {chunk_count}: {len(chunk)} rows")
        print(f"  Authors: {chunk['author'].unique().tolist()}")
        print(f"  Sample text: {chunk['text'].iloc[0][:80]}...")
        print()

    print(f"Total: {total_rows} rows in {chunk_count} chunks")
    print()


def test_ndjson_loading():
    """Test loading NDJSON file."""
    print("=" * 60)
    print("Testing NDJSON Loading")
    print("=" * 60)

    loader = DataLoader()
    ndjson_file = Path(__file__).parent / 'sample_data.ndjson'

    if not ndjson_file.exists():
        print(f"Error: {ndjson_file} not found")
        return

    print(f"Loading: {ndjson_file}")
    print()

    total_rows = 0
    chunk_count = 0

    for chunk in loader.load_ndjson(ndjson_file, 'author', 'text', chunksize=5):
        chunk_count += 1
        total_rows += len(chunk)
        print(f"Chunk {chunk_count}: {len(chunk)} rows")
        print(f"  Authors: {chunk['author'].unique().tolist()}")
        print()

    print(f"Total: {total_rows} rows in {chunk_count} chunks")
    print()


def test_danish_data():
    """Test loading Danish CSV file."""
    print("=" * 60)
    print("Testing Danish Data Loading")
    print("=" * 60)

    loader = DataLoader()
    danish_file = Path(__file__).parent / 'sample_danish.csv'

    if not danish_file.exists():
        print(f"Error: {danish_file} not found")
        return

    print(f"Loading: {danish_file}")
    print()

    # Get column names first
    columns = loader.get_column_names(danish_file)
    print(f"Columns: {columns}")
    print()

    total_rows = 0
    for chunk in loader.load_csv(danish_file, 'forfatter', 'tekst'):
        total_rows += len(chunk)
        print(f"Sample Danish text: {chunk['tekst'].iloc[0]}")
        break  # Just show first chunk

    print(f"\nTotal rows: {total_rows}")
    print()


def test_auto_detection():
    """Test automatic format detection."""
    print("=" * 60)
    print("Testing Auto Format Detection")
    print("=" * 60)

    loader = DataLoader()
    files = [
        Path(__file__).parent / 'sample_data.csv',
        Path(__file__).parent / 'sample_data.ndjson',
    ]

    for filepath in files:
        if not filepath.exists():
            continue

        print(f"Auto-detecting format for: {filepath.name}")

        total_rows = 0
        for chunk in loader.load_file(filepath, 'author', 'text', chunksize=100):
            total_rows += len(chunk)

        print(f"  Loaded {total_rows} rows")
        print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DataLoader Test Suite")
    print("=" * 60 + "\n")

    try:
        test_csv_loading()
        test_ndjson_loading()
        test_danish_data()
        test_auto_detection()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

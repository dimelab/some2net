"""
Enhanced Streamlit Web Interface for Social Network Analytics

Features:
- File upload (CSV/NDJSON)
- Column selection
- Entity type filtering (PER, LOC, ORG)
- Progress tracking with ETA
- Pipeline integration
- Force Atlas 2 visualization
- Multi-format downloads
- Cache management
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
import sys
import time
from collections import Counter
import json
import networkx as nx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import SocialNetworkPipeline
from core.data_loader import DataLoader
from utils.exporters import export_all_formats
from utils.visualizer import NetworkVisualizer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Social Network Analytics",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: #1f77b4;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">🔗 Social Network Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract social networks from social media posts using multilingual NER</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Extraction method selection
        st.subheader("🔍 Extraction Method")
        extraction_method = st.selectbox(
            "Choose Extraction Method",
            [
                "NER (Named Entities)",
                "Hashtags",
                "Mentions (@username)",
                "URL Domains",
                "Keywords (RAKE)",
                "Exact Match"
            ],
            help="Choose how to extract items from text to build the network"
        )

        # Map display names to method identifiers
        method_map = {
            "NER (Named Entities)": "ner",
            "Hashtags": "hashtag",
            "Mentions (@username)": "mention",
            "URL Domains": "domain",
            "Keywords (RAKE)": "keyword",
            "Exact Match": "exact"
        }
        extraction_method_id = method_map[extraction_method]

        st.divider()

        # Model selection (only for NER)
        if extraction_method_id == "ner":
            model_name = st.selectbox(
                "NER Model",
                [
                    "Davlan/xlm-roberta-base-ner-hrl",
                    "Babelscape/wikineural-multilingual-ner"
                ],
                help="Multilingual NER model for entity extraction"
            )
        else:
            model_name = None  # Not needed for other methods

        # NER-specific settings (only show for NER method)
        if extraction_method_id == "ner":
            # Confidence threshold
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.85,
                step=0.05,
                help="Minimum confidence score for extracted entities"
            )

            # Batch size
            batch_size = st.number_input(
                "Batch Size",
                min_value=8,
                max_value=128,
                value=32,
                step=8,
                help="Number of texts to process at once (higher = faster with more GPU memory)"
            )
        else:
            # Default values for non-NER methods
            confidence = 0.85
            batch_size = 32

        # Chunk size (applies to all methods)
        chunksize = st.number_input(
            "Chunk Size",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Number of rows per chunk (for memory management)"
        )

        st.divider()

        # Method-specific configuration
        st.subheader("🛠️ Method Configuration")
        extractor_config = {}

        if extraction_method_id == "ner":
            # Entity type selection for NER
            st.caption("Select entity types to extract:")
            extract_persons = st.checkbox("✅ Persons (PER)", value=True)
            extract_locations = st.checkbox("✅ Locations (LOC)", value=True)
            extract_organizations = st.checkbox("✅ Organizations (ORG)", value=True)

            entity_types_to_extract = []
            if extract_persons:
                entity_types_to_extract.extend(['PER', 'PERSON'])
            if extract_locations:
                entity_types_to_extract.extend(['LOC', 'LOCATION'])
            if extract_organizations:
                entity_types_to_extract.extend(['ORG', 'ORGANIZATION'])

        elif extraction_method_id == "hashtag":
            normalize_case = st.checkbox(
                "Normalize case",
                value=True,
                help="Convert #Python to #python for consistency"
            )
            extractor_config['normalize_case'] = normalize_case
            entity_types_to_extract = ['HASHTAG']

        elif extraction_method_id == "mention":
            normalize_case = st.checkbox(
                "Normalize case",
                value=True,
                help="Convert @User to @user for consistency"
            )
            extractor_config['normalize_case'] = normalize_case
            entity_types_to_extract = ['MENTION']

        elif extraction_method_id == "domain":
            strip_www = st.checkbox(
                "Strip 'www.' prefix",
                value=True,
                help="Convert www.example.com to example.com"
            )
            extractor_config['strip_www'] = strip_www
            entity_types_to_extract = ['DOMAIN']

        elif extraction_method_id == "keyword":
            min_keywords = st.slider(
                "Min keywords per author",
                min_value=1,
                max_value=10,
                value=5,
                help="Minimum number of keywords to extract per author"
            )
            max_keywords = st.slider(
                "Max keywords per author",
                min_value=5,
                max_value=50,
                value=20,
                help="Maximum number of keywords to extract per author"
            )
            language = st.selectbox(
                "Language for stopwords",
                ["english", "danish", "spanish", "french", "german", "italian", "portuguese"],
                help="Language for stopword filtering"
            )
            max_phrase_length = st.slider(
                "Max phrase length (words)",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum number of words in a keyword phrase"
            )
            extractor_config['min_keywords'] = min_keywords
            extractor_config['max_keywords'] = max_keywords
            extractor_config['language'] = language
            extractor_config['max_phrase_length'] = max_phrase_length
            entity_types_to_extract = ['KEYWORD']

            st.info("ℹ️ Keyword extraction uses two-pass processing and may take longer")

        elif extraction_method_id == "exact":
            entity_types_to_extract = ['EXACT']
            st.info("ℹ️ Exact match will use the text value as-is without extraction")

        st.divider()

        # Advanced options (NER-specific and general)
        with st.expander("🔧 Advanced Options"):
            # NER-specific options
            if extraction_method_id == "ner":
                enable_cache = st.checkbox("Enable NER Cache", value=True,
                                          help="Cache NER results to speed up reprocessing")
                detect_language = st.checkbox("Detect Languages", value=True,
                                             help="Automatically detect language of each post")
                create_author_edges = st.checkbox("Author-to-Author Edges", value=True,
                                                 help="Create edges when authors mention each other")
                use_entity_resolver = st.checkbox("Entity Deduplication", value=True,
                                                 help="Deduplicate entities (case-insensitive matching)")

                st.divider()

                # Entity Linking Options (Phase 2)
                st.subheader("🔗 Entity Linking (Phase 2)")
                enable_entity_linking = st.checkbox(
                    "Enable Wikipedia/Wikidata Linking",
                    value=False,
                    help="Link entities to Wikipedia/Wikidata for enhanced cross-language resolution"
                )

                if enable_entity_linking:
                    st.info("ℹ️ Entity linking will connect entities like 'København', 'Copenhagen', and 'Copenhague' to the same Wikipedia/Wikidata entry (Q1748)")

                    linking_confidence = st.slider(
                        "Linking Confidence Threshold",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.7,
                        step=0.05,
                        help="Minimum confidence for entity linking"
                    )

                    linking_cache = st.checkbox(
                        "Cache Linking Results",
                        value=True,
                        help="Cache entity linking results for faster reprocessing"
                    )
                else:
                    linking_confidence = 0.7
                    linking_cache = True

                st.divider()

                if enable_cache:
                    if st.button("🗑️ Clear NER Cache", use_container_width=True):
                        st.session_state['clear_cache'] = True
                        st.warning("⚠️ Cache will be cleared on next processing. Click 'Process Data' to reprocess with updated settings.")
            else:
                # Default values for non-NER methods
                enable_cache = False
                detect_language = False
                create_author_edges = False
                use_entity_resolver = True  # Keep this enabled for other methods
                enable_entity_linking = False
                linking_confidence = 0.7
                linking_cache = True

            # General visualization option (applies to all methods)
            st.divider()
            layout_iterations = st.slider("Visualization Quality", 50, 200, 100, 10,
                                         help="Force Atlas iterations (higher = better layout, slower)")

    # Main content area
    st.header("1️⃣ Upload Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV or NDJSON file",
        type=['csv', 'ndjson', 'jsonl'],
        help="Upload your social media data file (.csv or .ndjson)"
    )

    if uploaded_file is not None:
        # Generate file identifier
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # Clear previous results ONLY when a DIFFERENT file is uploaded
        # This prevents session state from becoming too large (causes 413 errors on subsequent uploads)
        # But allows viewing results after processing the same file
        if 'current_file_id' not in st.session_state:
            st.session_state.current_file_id = None

        if st.session_state.current_file_id != current_file_id:
            # Different file - clear old results
            if 'graph' in st.session_state and st.session_state.graph is not None:
                st.session_state.graph = None
                st.session_state.stats = None
                st.session_state.processed = False
            st.session_state.current_file_id = current_file_id

        # Save uploaded file temporarily
        temp_dir = Path("/tmp/sna_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name

        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Auto-detect file type
        file_type = 'ndjson' if uploaded_file.name.endswith(('.ndjson', '.jsonl')) else 'csv'

        # Preview data
        st.subheader("📊 Data Preview")

        # CSV parsing options (for malformed files)
        csv_error_handling = None
        if file_type == 'csv':
            with st.expander("⚙️ CSV Parsing Options (click if you see errors)"):
                st.markdown("""
                If your CSV has issues (inconsistent columns, quotes, etc.), try these options:
                """)
                csv_error_handling = st.radio(
                    "Error Handling Strategy",
                    options=['default', 'skip_bad_lines', 'fill_missing'],
                    index=0,
                    help="""
                    - **default**: Strict parsing (fails on malformed rows)
                    - **skip_bad_lines**: Skip rows with wrong number of columns
                    - **fill_missing**: Fill missing columns with empty values
                    """
                )

        try:
            if file_type == 'csv':
                # Try different parsing strategies based on user selection
                if csv_error_handling == 'skip_bad_lines':
                    preview_df = pd.read_csv(
                        temp_path,
                        nrows=10,
                        on_bad_lines='skip',  # Skip malformed rows
                        engine='python'
                    )
                    st.info("ℹ️ Using 'skip_bad_lines' mode - malformed rows will be skipped during processing")
                elif csv_error_handling == 'fill_missing':
                    preview_df = pd.read_csv(
                        temp_path,
                        nrows=10,
                        on_bad_lines='warn',  # Warn but continue
                        engine='python'
                    )
                    st.info("ℹ️ Using 'fill_missing' mode - missing columns will be filled with NaN")
                else:
                    # Default strict parsing
                    preview_df = pd.read_csv(temp_path, nrows=10)
            else:
                # Use json_normalize to flatten nested structures (matches actual loading)
                import json
                records = []
                with open(temp_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 10:  # Only read first 10 rows
                            break
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                preview_df = pd.json_normalize(records, sep='.')

            # Handle duplicate column names
            if preview_df.columns.duplicated().any():
                st.warning("⚠️ Duplicate column names detected. Keeping only the first occurrence of each column.")
                dup_mask = preview_df.columns.duplicated(keep='first')
                preview_df = preview_df.loc[:, ~dup_mask]

            st.dataframe(preview_df, use_container_width=True)

            # File info
            col1, col2, col3 = st.columns(3)
            with col1:
                file_size_kb = uploaded_file.size / 1024
                file_size_mb = file_size_kb / 1024
                size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_kb:.1f} KB"
                st.metric("📁 File Size", size_str)

            with col2:
                # Count total rows efficiently
                try:
                    if file_type == 'csv':
                        # Use pandas to count rows for large files
                        loader = DataLoader()
                        total_rows = sum(len(chunk) for chunk in loader.load_csv(
                            str(temp_path),
                            list(preview_df.columns)[0],
                            list(preview_df.columns)[0],
                            chunksize=10000
                        ))
                    else:
                        total_rows = sum(1 for _ in open(temp_path))
                    st.metric("📝 Total Rows", f"{total_rows:,}")
                    st.session_state['total_rows'] = total_rows
                except:
                    st.metric("📝 Total Rows", "Unknown")
                    st.session_state['total_rows'] = None

            with col3:
                st.metric("📊 Columns", len(preview_df.columns))

        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error reading file: {error_msg}")

            # Provide specific help for common CSV errors
            if "Expected" in error_msg and "fields" in error_msg and "saw" in error_msg:
                st.warning("""
                **CSV Parsing Error Detected!**

                Your CSV file has inconsistent column counts (some rows have more/fewer columns than the header).

                **Solutions:**

                1. **Try the parsing options above** ⬆️
                   - Expand "⚙️ CSV Parsing Options"
                   - Select "skip_bad_lines" or "fill_missing"

                2. **Fix your CSV file:**
                   - Open in Excel and re-save as CSV UTF-8
                   - Use a CSV validator tool
                   - Check for unescaped commas or quotes in your data

                3. **Use NDJSON format instead:**
                   - More robust for messy data
                   - Each line is independent
                """)

                # Offer to try with error handling automatically
                st.info("""
                💡 **Tip:** The processing will automatically skip bad rows using the DataLoader's built-in error handling.
                You can proceed even if the preview fails!
                """)

                # Store error flag
                st.session_state['csv_parse_error'] = True
            else:
                st.info("Try using NDJSON format if CSV continues to have issues.")

            return

        # Column selection
        st.header("2️⃣ Select Columns")

        # Smart column detection (only on first file upload for this file)
        # Use file name + size as unique identifier
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # Check if this is a new file (different from last processed)
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != file_id:
            # New file - run smart detection
            st.session_state.last_file_id = file_id

            # Try to find author column (supports nested fields like data.user.unique_id or datauserunique_id)
            author_default_index = 0
            author_patterns = ['author', 'username', 'user', 'unique_id', 'userid', 'user_id', 'nickname', 'datauserunique_id']
            for col_idx, col_name in enumerate(preview_df.columns):
                col_lower = col_name.lower()
                # Check if any pattern appears in the column name
                # Prioritize exact matches, then partial matches
                if col_lower in author_patterns:
                    author_default_index = col_idx
                    break
                elif any(pattern in col_lower for pattern in author_patterns):
                    author_default_index = col_idx
                    # Don't break - keep looking for exact match
            st.session_state.author_col_selection = author_default_index

            # Try to find text column (supports nested fields like data.text or datatext)
            text_default_index = 0
            text_patterns = ['text', 'body', 'content', 'message', 'post', 'desc', 'description', 'comment', 'datatext']
            for col_idx, col_name in enumerate(preview_df.columns):
                col_lower = col_name.lower()
                # Check if any pattern appears in the column name
                # Prioritize exact matches, then partial matches
                if col_lower in text_patterns:
                    text_default_index = col_idx
                    break
                elif any(pattern in col_lower for pattern in text_patterns):
                    text_default_index = col_idx
                    # Don't break - keep looking for exact match
            st.session_state.text_col_selection = text_default_index

        # Ensure selections exist (fallback for edge cases)
        if 'author_col_selection' not in st.session_state:
            st.session_state.author_col_selection = 0
        if 'text_col_selection' not in st.session_state:
            st.session_state.text_col_selection = 0

        col1, col2 = st.columns(2)

        with col1:
            author_col = st.selectbox(
                "👤 Author Column",
                options=preview_df.columns,
                index=st.session_state.author_col_selection,
                help="Column containing post authors/usernames",
                key="author_column_select"
            )

        with col2:
            text_col = st.selectbox(
                "💬 Text Column",
                options=preview_df.columns,
                index=st.session_state.text_col_selection,
                help="Column containing post text/content",
                key="text_column_select"
            )

        # Update session state with current user selection
        # This allows user changes to persist across reruns
        st.session_state.author_col_selection = list(preview_df.columns).index(author_col)
        st.session_state.text_col_selection = list(preview_df.columns).index(text_col)

        # Show sample data with selected columns
        st.caption("Sample data with selected columns:")
        sample_display = preview_df[[author_col, text_col]].head(5)
        st.dataframe(sample_display, use_container_width=True)

        # Metadata column selection
        st.subheader("📊 Metadata Columns (Optional)")
        st.caption("Attach additional columns as metadata to nodes and edges")

        # Get all available columns
        all_available_cols = list(preview_df.columns)

        col1, col2 = st.columns(2)

        with col1:
            node_metadata_cols = st.multiselect(
                "Node Metadata Columns",
                all_available_cols,
                help="Attach these columns as attributes to author nodes (aggregated per author). You can include the text or author columns here."
            )

        with col2:
            edge_metadata_cols = st.multiselect(
                "Edge Metadata Columns",
                all_available_cols,
                help="Attach these columns as attributes to edges (per post/mention). You can include the text or author columns here."
            )

        # Check for empty text values in preview
        empty_text_count = preview_df[text_col].isna().sum()
        total_preview = len(preview_df)
        non_empty_count = total_preview - empty_text_count

        if empty_text_count > 0:
            empty_pct = (empty_text_count / total_preview) * 100
            st.warning(f"⚠️ {empty_text_count}/{total_preview} ({empty_pct:.1f}%) preview rows have empty text. Only {non_empty_count} rows will be processed.")
        else:
            st.success(f"✅ All {total_preview} preview rows have text content!")

        # Show sample of actual text content
        non_empty_texts = preview_df[preview_df[text_col].notna()][text_col].head(3)
        if len(non_empty_texts) > 0:
            with st.expander("📝 Sample text content", expanded=False):
                for i, text in enumerate(non_empty_texts, 1):
                    text_str = str(text)[:200]
                    st.text(f"{i}. {text_str}{'...' if len(str(text)) > 200 else ''}")

        # Validation (only for NER method)
        if extraction_method_id == "ner" and not entity_types_to_extract:
            st.warning("⚠️ Please select at least one entity type to extract in the sidebar")
            return

        # Process button
        st.header("3️⃣ Process Data")

        col1, col2 = st.columns([3, 1])
        with col1:
            process_button = st.button(
                "🚀 Start Processing" if not st.session_state.processed else "🔄 Reprocess",
                type="primary",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        if process_button:
            # Clear previous results before reprocessing
            st.session_state.processed = False
            st.session_state.graph = None
            st.session_state.stats = None

            process_data_with_pipeline(
                str(temp_path),
                file_type,
                author_col,
                text_col,
                model_name,
                confidence,
                batch_size,
                chunksize,
                entity_types_to_extract,
                enable_cache,
                detect_language,
                create_author_edges,
                use_entity_resolver,
                layout_iterations,
                enable_entity_linking,
                linking_confidence,
                linking_cache,
                extraction_method_id,
                extractor_config,
                node_metadata_cols,
                edge_metadata_cols
            )

        # Display results if already processed
        if st.session_state.processed and st.session_state.graph is not None:
            display_results(
                st.session_state.graph,
                st.session_state.stats,
                layout_iterations,
                enable_entity_linking
            )


def process_data_with_pipeline(
    filepath: str,
    file_type: str,
    author_col: str,
    text_col: str,
    model_name: str,
    confidence: float,
    batch_size: int,
    chunksize: int,
    entity_types: list,
    enable_cache: bool,
    detect_language: bool,
    create_author_edges: bool,
    use_entity_resolver: bool,
    layout_iterations: int,
    enable_entity_linking: bool = False,
    linking_confidence: float = 0.7,
    linking_cache: bool = True,
    extraction_method: str = "ner",
    extractor_config: dict = None,
    node_metadata_columns: list = None,
    edge_metadata_columns: list = None
):
    """Process data using the integrated pipeline."""

    if extractor_config is None:
        extractor_config = {}
    if node_metadata_columns is None:
        node_metadata_columns = []
    if edge_metadata_columns is None:
        edge_metadata_columns = []

    start_time = time.time()

    try:
        # Create pipeline
        method_names = {
            "ner": "NER model",
            "hashtag": "Hashtag extractor",
            "mention": "Mention extractor",
            "domain": "Domain extractor",
            "keyword": "Keyword extractor (RAKE)",
            "exact": "Exact match extractor"
        }

        spinner_msg = f"🔄 Initializing pipeline and loading {method_names.get(extraction_method, 'extractor')}..."
        if extraction_method == "ner" and enable_entity_linking:
            spinner_msg = "🔄 Initializing pipeline and loading NER + Entity Linking models..."

        with st.spinner(spinner_msg):
            # Configure entity linking (only for NER)
            entity_linking_config = None
            if extraction_method == "ner" and enable_entity_linking:
                entity_linking_config = {
                    'confidence_threshold': linking_confidence,
                    'enable_cache': linking_cache
                }

            # Initialize pipeline with extraction method
            pipeline = SocialNetworkPipeline(
                extraction_method=extraction_method,
                extractor_config=extractor_config,
                model_name=model_name if extraction_method == "ner" else None,
                confidence_threshold=confidence if extraction_method == "ner" else None,
                enable_cache=enable_cache if extraction_method == "ner" else False,
                use_entity_resolver=use_entity_resolver,
                create_author_edges=create_author_edges,
                enable_entity_linking=enable_entity_linking if extraction_method == "ner" else False,
                entity_linking_config=entity_linking_config
            )

            # Clear cache if requested (only for NER)
            if st.session_state.get('clear_cache', False) and extraction_method == "ner":
                if hasattr(pipeline, 'ner_engine') and pipeline.ner_engine:
                    pipeline.ner_engine.clear_cache()
                if enable_entity_linking and hasattr(pipeline, 'entity_linker') and pipeline.entity_linker:
                    pipeline.entity_linker.clear_cache()
                st.session_state['clear_cache'] = False
                st.info("🗑️ Cache cleared! Processing with fresh extraction...")

        st.success(f"✅ Pipeline initialized with {method_names.get(extraction_method, 'extractor')}!")

        # Show cache stats (only for NER)
        if enable_cache and extraction_method == "ner":
            try:
                if hasattr(pipeline, 'ner_engine') and pipeline.ner_engine:
                    cache_stats = pipeline.ner_engine.get_cache_stats()
                    st.info(f"💾 Cache: {cache_stats['size']} entries, {cache_stats['size_bytes'] / 1024:.1f} KB")
            except:
                pass

        # Progress tracking
        st.subheader("⏳ Processing Progress")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        eta_text = st.empty()
        stats_cols = st.columns(4)

        # Counters
        processed_posts = 0
        total_rows = st.session_state.get('total_rows', None)
        chunk_times = []

        # Custom progress callback
        def progress_callback(current, total, status_msg):
            nonlocal processed_posts
            processed_posts = current

            # Update progress bar
            if total_rows:
                progress = min(current / total_rows, 1.0)
                progress_bar.progress(progress)

                # Calculate ETA
                if chunk_times and current < total_rows:
                    avg_time_per_chunk = sum(chunk_times) / len(chunk_times)
                    remaining_posts = total_rows - current
                    eta_seconds = (remaining_posts / chunksize) * avg_time_per_chunk
                    eta_text.text(f"⏱️ Estimated time remaining: {eta_seconds:.0f}s")

            status_text.text(f"📊 {status_msg}")

        # Process file
        chunk_start = time.time()

        st.info(f"🔄 Processing file with {chunksize} rows per chunk, batch size {batch_size}")

        graph, stats = pipeline.process_file(
            filepath=filepath,
            author_column=author_col,
            text_column=text_col,
            file_format=file_type,
            chunksize=chunksize,
            batch_size=batch_size,
            detect_languages=detect_language,
            show_progress=False,
            progress_callback=progress_callback,
            node_metadata_columns=node_metadata_columns,
            edge_metadata_columns=edge_metadata_columns
        )

        # Debug: Check what was processed
        st.info(f"""
        🔍 **Processing Debug Info:**
        - Posts processed: {stats.get('processing_metadata', {}).get('total_posts', 0)}
        - Chunks processed: {stats.get('processing_metadata', {}).get('total_chunks', 0)}
        - Entities extracted: {stats.get('processing_metadata', {}).get('entities_extracted', 0)}
        - Graph nodes: {graph.number_of_nodes()}
        - Graph edges: {graph.number_of_edges()}
        - Author column used: `{author_col}`
        - Text column used: `{text_col}`
        """)

        # Filter entities by selected types if needed
        # (This is now handled in the NER engine or can be done post-processing)

        # Complete
        elapsed_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        eta_text.text(f"⏱️ Total time: {elapsed_time:.1f}s")

        # Store in session state
        st.session_state.graph = graph
        st.session_state.stats = stats
        st.session_state.processed = True
        st.session_state.elapsed_time = elapsed_time
        st.session_state.model_name = model_name
        st.session_state.confidence = confidence

        # Show success message
        st.success(f"""
        ✅ **Processing Complete!**
        - Posts processed: {stats['processing_metadata']['total_posts']:,}
        - Entities extracted: {stats['processing_metadata']['entities_extracted']:,}
        - Time elapsed: {elapsed_time:.1f}s
        - Speed: {stats['processing_metadata']['total_posts'] / elapsed_time:.1f} posts/second
        """)

        # Note: Results will be displayed by the session state check after rerun
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        with st.expander("🔍 Show error details"):
            import traceback
            st.code(traceback.format_exc())


def display_results(graph, stats, layout_iterations, enable_entity_linking=False):
    """Display processing results, statistics, and visualization."""

    st.header("4️⃣ Results")

    # Debug: Check actual graph object
    actual_nodes = graph.number_of_nodes()
    actual_edges = graph.number_of_edges()
    st.info(f"🔍 Debug - Actual graph object: {actual_nodes} nodes, {actual_edges} edges")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔵 Total Nodes", f"{stats.get('total_nodes', 0):,}")
    with col2:
        st.metric("➡️ Total Edges", f"{stats.get('total_edges', 0):,}")
    with col3:
        st.metric("👥 Authors", f"{stats.get('authors', 0):,}")
    with col4:
        total_entities = stats.get('persons', 0) + stats.get('locations', 0) + stats.get('organizations', 0)
        st.metric("🏷️ Entities", f"{total_entities:,}")

    # Detailed entity metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Persons", f"{stats.get('persons', 0):,}")
    with col2:
        st.metric("📍 Locations", f"{stats.get('locations', 0):,}")
    with col3:
        st.metric("🏢 Organizations", f"{stats.get('organizations', 0):,}")
    with col4:
        st.metric("🔗 Density", f"{stats.get('density', 0.0):.4f}")

    # Processing metadata
    with st.expander("📊 Processing Details"):
        metadata = stats.get('processing_metadata', {})

        # Display processing metadata with entity linking stats if available
        if metadata.get('entities_linked', 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Posts Processed", f"{metadata.get('total_posts', 0):,}")
            with col2:
                st.metric("Chunks Processed", f"{metadata.get('total_chunks', 0):,}")
            with col3:
                st.metric("Entities Extracted", f"{metadata.get('entities_extracted', 0):,}")
            with col4:
                entities_extracted = metadata.get('entities_extracted', 0)
                entities_linked = metadata.get('entities_linked', 0)
                linked_pct = (entities_linked / entities_extracted * 100) if entities_extracted > 0 else 0
                st.metric("Entities Linked", f"{entities_linked:,}", f"{linked_pct:.1f}%")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Posts Processed", f"{metadata.get('total_posts', 0):,}")
            with col2:
                st.metric("Chunks Processed", f"{metadata.get('total_chunks', 0):,}")
            with col3:
                st.metric("Entities Extracted", f"{metadata.get('entities_extracted', 0):,}")

        if metadata.get('errors'):
            st.warning(f"⚠️ {len(metadata['errors'])} errors encountered during processing")
            with st.expander("View errors"):
                for i, error in enumerate(metadata['errors'][:10], 1):
                    st.text(f"{i}. {error}")
                if len(metadata['errors']) > 10:
                    st.text(f"... and {len(metadata['errors']) - 10} more")

    # Top mentioned entities
    st.subheader("🏆 Top Mentioned Entities")
    if stats.get('top_entities'):
        top_df = pd.DataFrame(stats['top_entities'][:20])
        top_df.columns = ['Entity', 'Mentions', 'Type']

        # Add Wikidata/Wikipedia links if entity linking was enabled
        if enable_entity_linking:
            # Get Wikidata IDs and Wikipedia URLs from graph nodes
            wikidata_ids = []
            wikipedia_urls = []
            for entity_name in top_df['Entity']:
                if graph.has_node(entity_name):
                    node_data = graph.nodes[entity_name]
                    wikidata_id = node_data.get('wikidata_id', '')
                    wikipedia_url = node_data.get('wikipedia_url', '')
                    wikidata_ids.append(wikidata_id if wikidata_id else '-')
                    wikipedia_urls.append(f'[Link]({wikipedia_url})' if wikipedia_url else '-')
                else:
                    wikidata_ids.append('-')
                    wikipedia_urls.append('-')

            top_df['Wikidata ID'] = wikidata_ids
            top_df['Wikipedia'] = wikipedia_urls

        # Color code by type
        def highlight_type(row):
            if row['Type'] == 'person':
                return ['background-color: #e3f2fd'] * len(row)
            elif row['Type'] == 'location':
                return ['background-color: #fff3e0'] * len(row)
            elif row['Type'] == 'organization':
                return ['background-color: #f3e5f5'] * len(row)
            return [''] * len(row)

        st.dataframe(
            top_df.style.apply(highlight_type, axis=1),
            use_container_width=True
        )

    # Network Visualization
    st.header("5️⃣ Network Visualization")

    # Filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Choose which part of the network to visualize:")
    with col2:
        show_giant_component = st.checkbox(
            "🔍 Giant Component Only",
            value=False,
            help="Show only the largest connected component of the network"
        )

    try:
        with st.spinner("🎨 Preparing interactive Force Atlas 2 visualization..."):
            viz = NetworkVisualizer()

            # Start with full graph
            display_graph = graph

            # Limit visualization for very large networks (BEFORE giant component filter)
            if display_graph.number_of_nodes() > 1000:
                st.warning(f"⚠️ Network has {display_graph.number_of_nodes():,} nodes. Showing top 500 most connected nodes for performance.")
                # Get top nodes by degree
                degree_dict = dict(display_graph.degree())
                top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:500]
                top_node_ids = [n[0] for n in top_nodes]
                display_graph = display_graph.subgraph(top_node_ids).copy()

            # Filter to giant component if requested (AFTER size filtering)
            if show_giant_component:
                # Get weakly connected components (for directed graphs)
                if display_graph.is_directed():
                    connected_components = list(nx.weakly_connected_components(display_graph))
                else:
                    connected_components = list(nx.connected_components(display_graph))

                if connected_components:
                    # Get the largest component
                    giant_component = max(connected_components, key=len)
                    display_graph = display_graph.subgraph(giant_component).copy()

                    st.info(f"🔍 Showing giant component: {len(giant_component):,} nodes "
                           f"({len(giant_component)/display_graph.number_of_nodes()*100:.1f}% of filtered network)")
                else:
                    st.warning("No connected components found")

            # Export graph data for Sigma.js
            graph_data = viz.export_for_sigma(display_graph)

            # Check JSON size and reduce further if needed (avoid 413 error)
            graph_json = json.dumps(graph_data)
            json_size_mb = len(graph_json.encode('utf-8')) / (1024 * 1024)

            # If JSON is too large (> 10MB), reduce further
            if json_size_mb > 10 and display_graph.number_of_nodes() > 100:
                st.warning(f"⚠️ Visualization data is large ({json_size_mb:.1f}MB). Reducing to top 200 nodes to avoid browser issues.")
                degree_dict = dict(display_graph.degree())
                top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:200]
                top_node_ids = [n[0] for n in top_nodes]
                display_graph = display_graph.subgraph(top_node_ids).copy()

                # Re-export with smaller graph
                graph_data = viz.export_for_sigma(display_graph)
                graph_json = json.dumps(graph_data)
                json_size_mb = len(graph_json.encode('utf-8')) / (1024 * 1024)

            # Debug info
            st.info(f"📊 Graph data: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('edges', []))} edges (Size: {json_size_mb:.1f}MB)")

            # Load HTML template
            template_path = Path(__file__).parent / 'templates' / 'sigma_viewer.html'
            with open(template_path, 'r') as f:
                html_template = f.read()

            # Inject graph data
            html_content = html_template.replace(
                '{{GRAPH_DATA}}',
                graph_json
            )

            # Display in Streamlit
            components.html(html_content, height=850, scrolling=False)

            st.caption("""
            **How to interact:**
            - 🖱️ Hover over nodes to see details
            - 🔍 Zoom with scroll wheel
            - 🖐️ Pan by clicking and dragging
            - ▶️ Use controls on the right to adjust the layout in real-time
            - 🔍 Toggle "Giant Component Only" above to focus on the main connected network
            - 🎨 Node colors: Blue=Authors, Orange=Persons, Green=Locations, Red=Organizations
            - 💡 The layout is computed in your browser using Force Atlas 2
            """)
    except Exception as e:
        st.error(f"❌ Error creating visualization: {e}")
        with st.expander("Show error details"):
            import traceback
            st.code(traceback.format_exc())

    # Composition chart
    try:
        st.subheader("📊 Network Composition")
        comp_fig = viz.create_simple_stats_plot(stats)
        st.plotly_chart(comp_fig, use_container_width=True)
    except:
        pass

    # Export options
    st.header("6️⃣ Download Results")

    output_dir = Path("/tmp/sna_output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Export all formats
    with st.spinner("📦 Preparing download files..."):
        files = export_all_formats(
            graph,
            stats,
            str(output_dir),
            "network"
        )

    st.success(f"✅ Exported {len(files)} formats")

    # Primary download (GEXF)
    st.subheader("Primary Export Format")
    if 'gexf' in files:
        with open(files['gexf'], 'rb') as f:
            st.download_button(
                label="📥 Download GEXF (Primary - for Gephi)",
                data=f.read(),
                file_name="network.gexf",
                mime="application/xml",
                use_container_width=True,
                type="primary",
                key="download_gexf"
            )

    # Secondary formats
    st.subheader("Additional Formats")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'graphml' in files:
            with open(files['graphml'], 'rb') as f:
                st.download_button(
                    label="📥 GraphML",
                    data=f.read(),
                    file_name="network.graphml",
                    mime="application/xml",
                    use_container_width=True,
                    key="download_graphml"
                )

    with col2:
        if 'json' in files:
            with open(files['json'], 'rb') as f:
                st.download_button(
                    label="📥 JSON (D3.js)",
                    data=f.read(),
                    file_name="network.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_json"
                )

    with col3:
        if 'edgelist' in files:
            with open(files['edgelist'], 'rb') as f:
                st.download_button(
                    label="📥 Edge List CSV",
                    data=f.read(),
                    file_name="network_edgelist.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_edgelist"
                )

    # Statistics
    if 'statistics' in files:
        with open(files['statistics'], 'rb') as f:
            st.download_button(
                label="📊 Statistics (JSON)",
                data=f.read(),
                file_name="statistics.json",
                mime="application/json",
                use_container_width=True,
                key="download_statistics"
            )

    # Processing summary
    st.divider()

    if hasattr(st.session_state, 'elapsed_time'):
        metadata = stats['processing_metadata']
        st.info(f"""
        **📋 Processing Summary:**
        - ✅ Posts processed: {metadata['total_posts']:,}
        - ⏱️ Time elapsed: {st.session_state.elapsed_time:.1f}s
        - ⚡ Processing speed: {metadata['total_posts'] / st.session_state.elapsed_time:.1f} posts/second
        - 🤖 Model: {st.session_state.model_name}
        - 📊 Confidence threshold: {st.session_state.confidence}
        - 🔗 Network nodes: {stats['total_nodes']:,}
        - ➡️ Network edges: {stats['total_edges']:,}
        """)


if __name__ == "__main__":
    main()

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

        # Model selection
        model_name = st.selectbox(
            "NER Model",
            [
                "Davlan/xlm-roberta-base-ner-hrl",
                "Babelscape/wikineural-multilingual-ner"
            ],
            help="Multilingual NER model for entity extraction"
        )

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

        # Chunk size
        chunksize = st.number_input(
            "Chunk Size",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Number of rows per chunk (for memory management)"
        )

        st.divider()

        # Entity type selection
        st.subheader("🏷️ Entity Types")
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

        st.divider()

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            enable_cache = st.checkbox("Enable NER Cache", value=True,
                                      help="Cache NER results to speed up reprocessing")
            detect_language = st.checkbox("Detect Languages", value=True,
                                         help="Automatically detect language of each post")
            create_author_edges = st.checkbox("Author-to-Author Edges", value=True,
                                             help="Create edges when authors mention each other")
            use_entity_resolver = st.checkbox("Entity Deduplication", value=True,
                                             help="Deduplicate entities (case-insensitive matching)")
            layout_iterations = st.slider("Visualization Quality", 50, 200, 100, 10,
                                         help="Force Atlas iterations (higher = better layout, slower)")

            st.divider()

            if enable_cache:
                if st.button("🗑️ Clear NER Cache", use_container_width=True):
                    st.session_state['clear_cache'] = True
                    st.success("✅ Cache will be cleared on next processing")

    # Main content area
    st.header("1️⃣ Upload Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV or NDJSON file",
        type=['csv', 'ndjson', 'jsonl'],
        help="Upload your social media data file (.csv or .ndjson)"
    )

    if uploaded_file is not None:
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
        try:
            if file_type == 'csv':
                preview_df = pd.read_csv(temp_path, nrows=10)
            else:
                preview_df = pd.read_json(temp_path, lines=True, nrows=10)

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
            st.error(f"❌ Error reading file: {e}")
            return

        # Column selection
        st.header("2️⃣ Select Columns")

        col1, col2 = st.columns(2)

        with col1:
            author_col = st.selectbox(
                "👤 Author Column",
                options=preview_df.columns,
                help="Column containing post authors/usernames"
            )

        with col2:
            text_col = st.selectbox(
                "💬 Text Column",
                options=preview_df.columns,
                help="Column containing post text/content"
            )

        # Show sample data with selected columns
        st.caption("Sample data with selected columns:")
        sample_display = preview_df[[author_col, text_col]].head(3)
        st.dataframe(sample_display, use_container_width=True)

        # Validation
        if not entity_types_to_extract:
            st.warning("⚠️ Please select at least one entity type to extract in the sidebar")
            return

        # Process button
        st.header("3️⃣ Process Data")

        col1, col2 = st.columns([3, 1])
        with col1:
            process_button = st.button(
                "🚀 Start Processing",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processed
            )
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        if process_button:
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
                layout_iterations
            )

        # Display results if already processed
        if st.session_state.processed and st.session_state.graph is not None:
            display_results(
                st.session_state.graph,
                st.session_state.stats,
                layout_iterations
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
    layout_iterations: int
):
    """Process data using the integrated pipeline."""

    start_time = time.time()

    try:
        # Create pipeline
        with st.spinner("🔄 Initializing pipeline and loading NER model..."):
            pipeline = SocialNetworkPipeline(
                model_name=model_name,
                confidence_threshold=confidence,
                enable_cache=enable_cache,
                use_entity_resolver=use_entity_resolver,
                create_author_edges=create_author_edges
            )

            # Clear cache if requested
            if st.session_state.get('clear_cache', False):
                pipeline.ner_engine.clear_cache()
                st.session_state['clear_cache'] = False
                st.success("✅ Cache cleared!")

        st.success("✅ Pipeline initialized and model loaded!")

        # Show cache stats
        if enable_cache:
            try:
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
            progress_callback=progress_callback
        )

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

        # Display results
        display_results(graph, stats, layout_iterations)

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        with st.expander("🔍 Show error details"):
            import traceback
            st.code(traceback.format_exc())


def display_results(graph, stats, layout_iterations):
    """Display processing results, statistics, and visualization."""

    st.header("4️⃣ Results")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔵 Total Nodes", f"{stats['total_nodes']:,}")
    with col2:
        st.metric("➡️ Total Edges", f"{stats['total_edges']:,}")
    with col3:
        st.metric("👥 Authors", f"{stats['authors']:,}")
    with col4:
        total_entities = stats['persons'] + stats['locations'] + stats['organizations']
        st.metric("🏷️ Entities", f"{total_entities:,}")

    # Detailed entity metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Persons", f"{stats['persons']:,}")
    with col2:
        st.metric("📍 Locations", f"{stats['locations']:,}")
    with col3:
        st.metric("🏢 Organizations", f"{stats['organizations']:,}")
    with col4:
        st.metric("🔗 Density", f"{stats['density']:.4f}")

    # Processing metadata
    with st.expander("📊 Processing Details"):
        metadata = stats['processing_metadata']

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posts Processed", f"{metadata['total_posts']:,}")
        with col2:
            st.metric("Chunks Processed", f"{metadata['total_chunks']:,}")
        with col3:
            st.metric("Entities Extracted", f"{metadata['entities_extracted']:,}")

        if metadata['errors']:
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

    try:
        with st.spinner("🎨 Preparing interactive Force Atlas 2 visualization..."):
            viz = NetworkVisualizer()

            # Limit visualization for very large networks
            display_graph = graph
            if graph.number_of_nodes() > 1000:
                st.warning(f"⚠️ Network has {graph.number_of_nodes():,} nodes. Showing top 500 most connected nodes for performance.")
                # Get top nodes by degree
                degree_dict = dict(graph.degree())
                top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:500]
                top_node_ids = [n[0] for n in top_nodes]
                display_graph = graph.subgraph(top_node_ids).copy()

            # Export graph data for Sigma.js
            graph_data = viz.export_for_sigma(display_graph)

            # Load HTML template
            template_path = Path(__file__).parent / 'templates' / 'sigma_viewer.html'
            with open(template_path, 'r') as f:
                html_template = f.read()

            # Inject graph data
            html_content = html_template.replace(
                '{{GRAPH_DATA}}',
                json.dumps(graph_data)
            )

            # Display in Streamlit
            components.html(html_content, height=850, scrolling=False)

            st.caption("""
            **How to interact:**
            - 🖱️ Hover over nodes to see details
            - 🔍 Zoom with scroll wheel
            - 🖐️ Pan by clicking and dragging
            - ▶️ Use controls on the right to adjust the layout in real-time
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
                type="primary"
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
                    use_container_width=True
                )

    with col2:
        if 'json' in files:
            with open(files['json'], 'rb') as f:
                st.download_button(
                    label="📥 JSON (D3.js)",
                    data=f.read(),
                    file_name="network.json",
                    mime="application/json",
                    use_container_width=True
                )

    with col3:
        if 'edgelist' in files:
            with open(files['edgelist'], 'rb') as f:
                st.download_button(
                    label="📥 Edge List CSV",
                    data=f.read(),
                    file_name="network_edgelist.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Statistics
    if 'statistics' in files:
        with open(files['statistics'], 'rb') as f:
            st.download_button(
                label="📊 Statistics (JSON)",
                data=f.read(),
                file_name="statistics.json",
                mime="application/json",
                use_container_width=True
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

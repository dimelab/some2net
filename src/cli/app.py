"""Enhanced Streamlit web interface with visualization and entity filtering."""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import time
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import DataLoader
from core.ner_engine import NEREngine
from core.network_builder import NetworkBuilder
from utils.exporters import export_gexf, export_graphml, export_json, export_statistics
from utils.visualizer import NetworkVisualizer


def main():
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
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🔗 Social Network Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract social networks from social media posts using multilingual NER</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_name = st.selectbox(
            "NER Model",
            [
                "Davlan/xlm-roberta-base-ner-hrl",
                "Babelscape/wikineural-multilingual-ner"
            ],
            help="Multilingual NER model for entity extraction"
        )
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Minimum confidence score for extracted entities"
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=8,
            max_value=128,
            value=32,
            step=8,
            help="Number of texts to process at once (higher = faster with more GPU memory)"
        )
        
        st.divider()
        
        # Entity type selection
        st.subheader("🏷️ Entity Types")
        extract_persons = st.checkbox("Persons (PER)", value=True)
        extract_locations = st.checkbox("Locations (LOC)", value=True)
        extract_organizations = st.checkbox("Organizations (ORG)", value=True)
        
        entity_types_to_extract = []
        if extract_persons:
            entity_types_to_extract.append('PER')
        if extract_locations:
            entity_types_to_extract.append('LOC')
        if extract_organizations:
            entity_types_to_extract.append('ORG')
        
        st.divider()
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            enable_cache = st.checkbox("Enable NER Cache", value=True, 
                                      help="Cache NER results to speed up reprocessing")
            detect_language = st.checkbox("Detect Languages", value=True,
                                         help="Automatically detect language of each post")
            layout_iterations = st.slider("Layout Quality", 50, 200, 100, 10,
                                         help="Force Atlas iterations (higher = better layout, slower)")
            
            if enable_cache:
                if st.button("🗑️ Clear Cache"):
                    # Clear cache (will be implemented in processing)
                    st.session_state['clear_cache'] = True
                    st.success("Cache will be cleared on next processing")
    
    # Main content area
    st.header("1️⃣ Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or NDJSON file",
        type=['csv', 'ndjson'],
        help="Upload your social media data file"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Detect file type
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'ndjson'
        
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
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                # Get total rows
                if file_type == 'csv':
                    total_rows = sum(1 for _ in open(temp_path)) - 1
                else:
                    total_rows = sum(1 for _ in open(temp_path))
                st.metric("Total Rows", f"{total_rows:,}")
            with col3:
                st.metric("Columns", len(preview_df.columns))
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return
        
        # Column selection
        st.header("2️⃣ Select Columns")
        col1, col2 = st.columns(2)
        
        with col1:
            author_col = st.selectbox(
                "Author Column",
                options=preview_df.columns,
                help="Column containing post authors"
            )
        
        with col2:
            text_col = st.selectbox(
                "Text Column",
                options=preview_df.columns,
                help="Column containing post text/content"
            )
        
        # Validation
        if not entity_types_to_extract:
            st.warning("⚠️ Please select at least one entity type to extract")
            return
        
        # Process button
        st.header("3️⃣ Process Data")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        if process_button:
            process_data(
                temp_path,
                file_type,
                author_col,
                text_col,
                model_name,
                confidence,
                batch_size,
                entity_types_to_extract,
                enable_cache,
                detect_language,
                layout_iterations,
                total_rows
            )


def process_data(
    filepath: str,
    file_type: str,
    author_col: str,
    text_col: str,
    model_name: str,
    confidence: float,
    batch_size: int,
    entity_types: list,
    enable_cache: bool,
    detect_language: bool,
    layout_iterations: int,
    total_rows: int
):
    """Process uploaded data and build network with visualization."""
    
    start_time = time.time()
    
    try:
        # Initialize components
        with st.spinner("🔄 Loading NER model..."):
            engine = NEREngine(
                model_name=model_name,
                confidence_threshold=confidence,
                enable_cache=enable_cache
            )
            
            # Clear cache if requested
            if st.session_state.get('clear_cache', False):
                engine.clear_cache()
                st.session_state['clear_cache'] = False
        
        st.success("✅ Model loaded!")
        
        # Show cache stats
        if enable_cache:
            cache_stats = engine.get_cache_stats()
            st.info(f"💾 Cache: {cache_stats['size']} entries, {cache_stats['size_bytes'] / 1024:.1f} KB")
        
        loader = DataLoader(chunk_size=10000)
        builder = NetworkBuilder()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_placeholder = st.empty()
        
        # Language tracking
        language_counter = Counter()
        
        # Load and process data
        processed_posts = 0
        
        # Get data chunks
        if file_type == 'csv':
            chunks = loader.load_csv(filepath, author_col, text_col)
        else:
            chunks = loader.load_ndjson(filepath, author_col, text_col)
        
        # Process chunks
        for chunk_idx, chunk in enumerate(chunks):
            # Extract texts
            texts = chunk[text_col].tolist()
            authors = chunk[author_col].tolist()
            post_ids = chunk.get('post_id', range(len(chunk))).tolist()
            
            # Update status
            status_text.text(f"🔍 Processing chunk {chunk_idx + 1} ({len(texts)} posts)...")
            
            # Run NER with language detection
            entities_batch, languages = engine.extract_entities_batch(
                texts,
                batch_size=batch_size,
                show_progress=False,
                detect_languages=detect_language
            )
            
            # Track languages
            if detect_language:
                language_counter.update(languages)
            
            # Filter entities by selected types
            filtered_entities_batch = []
            for entities in entities_batch:
                filtered = [e for e in entities if e['type'] in entity_types]
                filtered_entities_batch.append(filtered)
            
            # Build network
            for author, entities, post_id in zip(authors, filtered_entities_batch, post_ids):
                builder.add_post(author, entities, str(post_id))
            
            # Update progress
            processed_posts += len(chunk)
            progress = min(processed_posts / total_rows, 1.0)
            progress_bar.progress(progress)
            
            # Show live stats
            current_stats = builder.get_statistics()
            stats_placeholder.metric(
                "Entities Extracted",
                f"{current_stats['entities_extracted']:,}",
                f"+{len([e for entities in filtered_entities_batch for e in entities])}"
            )
        
        # Complete
        elapsed_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Processing complete in {elapsed_time:.1f}s!")
        
        # Display results
        st.header("4️⃣ Results")
        
        # Get statistics
        stats = builder.get_statistics()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Nodes", f"{stats['total_nodes']:,}")
        with col2:
            st.metric("Total Edges", f"{stats['total_edges']:,}")
        with col3:
            st.metric("Authors", f"{stats['authors']:,}")
        with col4:
            st.metric("Entities", f"{stats['entities_extracted']:,}")
        
        # Detailed stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Persons", f"{stats['persons']:,}")
        with col2:
            st.metric("Locations", f"{stats['locations']:,}")
        with col3:
            st.metric("Organizations", f"{stats['organizations']:,}")
        with col4:
            st.metric("Network Density", f"{stats['density']:.4f}")
        
        # Language distribution
        if detect_language and language_counter:
            st.subheader("🌍 Language Distribution")
            lang_df = pd.DataFrame(
                language_counter.most_common(10),
                columns=['Language', 'Count']
            )
            st.dataframe(lang_df, use_container_width=True)
        
        # Top mentioned entities
        st.subheader("🏆 Top Mentioned Entities")
        if stats['top_mentioned']:
            top_df = pd.DataFrame(
                stats['top_mentioned'],
                columns=['Entity', 'Mentions']
            )
            st.dataframe(top_df, use_container_width=True)
        
        # Network Visualization
        st.header("5️⃣ Network Visualization")
        
        with st.spinner(f"🎨 Creating Force Atlas 2 visualization ({layout_iterations} iterations)..."):
            viz = NetworkVisualizer()
            graph = builder.get_graph()
            
            # Limit visualization for very large networks
            if graph.number_of_nodes() > 1000:
                st.warning(f"⚠️ Network has {graph.number_of_nodes():,} nodes. Showing top 500 most connected nodes.")
                # Get top nodes by degree
                degree_dict = dict(graph.degree())
                top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:500]
                top_node_ids = [n[0] for n in top_nodes]
                graph = graph.subgraph(top_node_ids).copy()
            
            fig = viz.create_interactive_plot(
                graph,
                title="Social Network - Force Atlas 2 Layout",
                width=1200,
                height=800,
                layout_iterations=layout_iterations
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Composition chart
        st.subheader("📊 Network Composition")
        comp_fig = viz.create_simple_stats_plot(stats)
        st.plotly_chart(comp_fig, use_container_width=True)
        
        # Export options
        st.header("6️⃣ Download Results")
        
        output_dir = Path("/tmp/sna_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export all formats
        graph = builder.get_graph()
        
        # GEXF (Primary)
        gexf_path = output_dir / "network.gexf"
        export_gexf(graph, str(gexf_path))
        
        with open(gexf_path, 'rb') as f:
            st.download_button(
                label="📥 Download GEXF (Primary - for Gephi)",
                data=f,
                file_name="network.gexf",
                mime="application/xml",
                use_container_width=True
            )
        
        col1, col2 = st.columns(2)
        
        # GraphML (Secondary)
        with col1:
            graphml_path = output_dir / "network.graphml"
            export_graphml(graph, str(graphml_path))
            
            with open(graphml_path, 'rb') as f:
                st.download_button(
                    label="📥 Download GraphML",
                    data=f,
                    file_name="network.graphml",
                    mime="application/xml",
                    use_container_width=True
                )
        
        # JSON
        with col2:
            json_path = output_dir / "network.json"
            export_json(graph, str(json_path))
            
            with open(json_path, 'rb') as f:
                st.download_button(
                    label="📥 Download JSON (for D3.js)",
                    data=f,
                    file_name="network.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Statistics
        stats_with_language = stats.copy()
        if language_counter:
            stats_with_language['language_distribution'] = dict(language_counter)
        
        stats_path = output_dir / "statistics.json"
        export_statistics(stats_with_language, str(stats_path))
        
        with open(stats_path, 'rb') as f:
            st.download_button(
                label="📊 Download Statistics (JSON)",
                data=f,
                file_name="statistics.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Processing info
        st.divider()
        st.info(f"""
        **Processing Summary:**
        - Posts processed: {processed_posts:,}
        - Time elapsed: {elapsed_time:.1f}s
        - Processing speed: {processed_posts / elapsed_time:.1f} posts/second
        - Model: {model_name}
        - Confidence threshold: {confidence}
        """)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        with st.expander("Show error details"):
            st.exception(e)


if __name__ == "__main__":
    main()

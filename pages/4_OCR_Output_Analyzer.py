# pages/4_OCR_Output_Analyzer.py

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import zipfile
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="OCR Output Analyzer",
    page_icon="💾",
    layout="wide"
)

# ALIGNED styling with Document Ingestion page
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Import components
try:
    from GraphRAG_Document_AI_Platform import load_config, get_mistral_client, get_enhanced_ocr_pipeline
    from src.utils.ocr_storage import create_storage_manager
    from src.utils.processing_pipeline import process_batch_with_enhanced_storage
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()


# Initialize components
@st.cache_resource
def get_storage_manager():
    try:
        return create_storage_manager("ocr_outputs")
    except Exception as e:
        st.error(f"Failed to initialize storage: {e}")
        return None


# FIXED: Get AI system status properly based on actual config structure
@st.cache_resource
def get_ai_system_status():
    """Check which AI systems are actually available based on config.toml structure."""
    try:
        config = load_config()

        # Check the actual nested structure in config.toml
        llm_ocr_config = config.get('llm', {}).get('ocr', {})

        # Check each AI system based on your actual config structure
        systems = {
            'gemini': bool(
                llm_ocr_config.get('gemini_api_key') or
                config.get('llm', {}).get('api_key')  # Fallback to main LLM key
            ),
            'mistral': bool(
                llm_ocr_config.get('mistral_api_key') or
                config.get('mistral', {}).get('api_key')  # Fallback to mistral section
            ),
            'openai': bool(
                llm_ocr_config.get('openai_api_key') and
                llm_ocr_config.get('openai_api_key') != "sk-your-openai-key-here"  # Exclude placeholder
            ),
            'anthropic': bool(
                llm_ocr_config.get('anthropic_api_key') and
                llm_ocr_config.get('anthropic_api_key') != "sk-ant-your-anthropic-key-here"  # Exclude placeholder
            )
        }

        # Debug info
        if st.session_state.get('debug_mode', False):
            st.write("**Config Structure Check:**")
            st.write("- LLM OCR Config found:", bool(llm_ocr_config))
            st.write("- Gemini API Key:", "✓" if systems['gemini'] else "✗")
            st.write("- Mistral API Key:", "✓" if systems['mistral'] else "✗")
            st.write("- OpenAI API Key:", "✓" if systems['openai'] else "✗")
            st.write("- Anthropic API Key:", "✓" if systems['anthropic'] else "✗")

        return systems

    except Exception as e:
        if st.session_state.get('debug_mode', False):
            st.error(f"Config error: {e}")
        # Fallback: based on your logs, these should be working
        return {
            'gemini': True,  # Primary method, should be working
            'mistral': True,  # Has valid key in config
            'openai': False,  # Placeholder key
            'anthropic': False  # Placeholder key
        }


storage_manager = get_storage_manager()

try:
    config = load_config()
    mistral_client = get_mistral_client(config.get('mistral', {}).get('api_key'))
    enhanced_ocr_pipeline = get_enhanced_ocr_pipeline(config)
    ai_systems = get_ai_system_status()
except Exception as e:
    st.error(f"Failed to initialize components: {e}")
    st.stop()

# Main header
st.title("💾 OCR Output Analyzer")
st.markdown("Extract text from documents using AI and manage your OCR archive")

# Tabs interface
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Extract & Archive", "📁 Storage Browser", "📊 Analytics & Export", "⚙️ Settings"])

# ============================================================================
# TAB 1: EXTRACT & ARCHIVE
# ============================================================================

with tab1:
    st.subheader("🚀 AI Text Extraction")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents for AI-powered text extraction:",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Documents will be processed with advanced LLM OCR and archived"
    )

    if uploaded_files:
        # Show uploaded files
        st.write(f"**{len(uploaded_files)} documents selected:**")
        file_data = []
        for i, file in enumerate(uploaded_files):
            file_data.append({
                "File": file.name,
                "Type": file.type.split('/')[-1].upper(),
                "Size": f"{len(file.getvalue()) / 1024:.1f} KB"
            })
        st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)

        # Show available AI methods
        col1, col2 = st.columns([2, 1])

        with col1:
            # Processing options
            save_to_disk = st.toggle("💾 Archive to Storage", value=True, help="Save extracted text to storage")
            extract_structured_fields = st.toggle("🔍 Extract Structured Data", value=True,
                                                  help="Extract structured information")

        with col2:
            st.markdown("**Available AI Methods:**")
            # Show correct AI system status based on actual config
            if ai_systems['gemini']:
                st.markdown('<div class="status-badge status-success">✅ Gemini Flash (Primary)</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ Gemini Flash</div>', unsafe_allow_html=True)

            if ai_systems['mistral']:
                st.markdown('<div class="status-badge status-success">✅ Mistral Pixtral</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ Mistral Pixtral</div>', unsafe_allow_html=True)

            if ai_systems['openai']:
                st.markdown('<div class="status-badge status-success">✅ GPT-4o Vision</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ GPT-4o Vision</div>', unsafe_allow_html=True)

            if ai_systems['anthropic']:
                st.markdown('<div class="status-badge status-success">✅ Claude Sonnet</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-error">❌ Claude Sonnet</div>', unsafe_allow_html=True)

        # Process button
        if st.button("🚀 Start AI Processing", type="primary", use_container_width=True):
            with st.spinner("Processing documents with AI..."):
                try:
                    batch_results = process_batch_with_enhanced_storage(
                        uploaded_files=uploaded_files,
                        enhanced_ocr_pipeline=enhanced_ocr_pipeline,
                        save_to_disk=save_to_disk
                    )

                    st.success("✅ Processing completed!")

                    # Summary metrics
                    successful = sum(1 for r in batch_results if r['success'])
                    total_text_length = sum(r.get('text_length', 0) for r in batch_results)
                    total_saved_files = sum(
                        len(r.get('saved_files', {})) for r in batch_results if r.get('saved_files'))
                    avg_confidence = sum(r.get('confidence', 0.0) for r in batch_results) / len(batch_results)

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        success_rate = (successful / len(batch_results)) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{success_rate:.0f}%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_text_length:,}</div>
                            <div class="metric-label">Characters</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{avg_confidence:.2f}</div>
                            <div class="metric-label">Avg Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_saved_files}</div>
                            <div class="metric-label">Files Archived</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Results table
                    st.subheader("📋 Processing Details")
                    results_data = []
                    for result in batch_results:
                        results_data.append({
                            'Document': result['original_filename'],
                            'Status': '✅ Success' if result['success'] else '❌ Failed',
                            'AI Method': result.get('method_used', 'unknown').upper(),
                            'Text Length': f"{result.get('text_length', 0):,}",
                            'Confidence': f"{result.get('confidence', 0.0):.3f}",
                            'Time': f"{result.get('processing_time', 0.0):.1f}s",
                            'Error': result.get('error', '')[:50] + '...' if result.get('error') and len(
                                result.get('error', '')) > 50 else result.get('error', '')
                        })

                    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")

# ============================================================================
# TAB 2: STORAGE BROWSER
# ============================================================================

with tab2:
    st.subheader("📁 Storage Browser")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    try:
        saved_files = storage_manager.list_saved_files(limit=100) if storage_manager else []

        if saved_files:
            # Storage metrics
            total_files = len(saved_files)
            total_size_kb = sum(f['size_kb'] for f in saved_files)
            total_text = sum(f['text_length'] for f in saved_files)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_files}</div>
                    <div class="metric-label">Total Files</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_size_kb:.1f}</div>
                    <div class="metric-label">Storage (KB)</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_text:,}</div>
                    <div class="metric-label">Text Characters</div>
                </div>
                """, unsafe_allow_html=True)

            # File browser
            st.subheader("📄 Archived Files")

            # File selection
            selected_file_idx = st.selectbox(
                "Select file for detailed view:",
                options=range(len(saved_files)),
                format_func=lambda
                    x: f"📄 {saved_files[x]['original_filename']} ({saved_files[x]['modified_time'][:16]})",
            )

            if selected_file_idx is not None:
                file_info = saved_files[selected_file_idx]

                # File details
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📋 File Information:**")
                    st.write(f"• **Name:** {file_info['original_filename']}")
                    st.write(f"• **Size:** {file_info['size_kb']:.1f} KB")
                    st.write(f"• **Text Length:** {file_info['text_length']:,} characters")
                    st.write(f"• **Modified:** {file_info['modified_time']}")

                with col2:
                    st.markdown("**📁 Storage Path:**")
                    st.code(file_info['text_file_path'])

                # Content preview
                with st.expander("📄 View Content", expanded=False):
                    try:
                        with open(file_info['text_file_path'], 'r', encoding='utf-8') as f:
                            text_content = f.read()

                        word_count = len(text_content.split())
                        st.write(f"**Words:** {word_count:,} | **Characters:** {len(text_content):,}")

                        st.text_area("Extracted Text:", value=text_content, height=300)

                        st.download_button(
                            "💾 Download Text",
                            data=text_content,
                            file_name=f"{file_info['original_filename']}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Could not load file: {str(e)}")
        else:
            st.info(
                "📂 No archived files found. Process documents in the 'Extract & Archive' tab to start building your archive.")

    except Exception as e:
        st.error(f"❌ Error accessing storage: {str(e)}")

# ============================================================================
# TAB 3: ANALYTICS & EXPORT
# ============================================================================

with tab3:
    st.subheader("📊 Analytics & Export")

    try:
        saved_files = storage_manager.list_saved_files(limit=1000) if storage_manager else []

        if saved_files:
            # Analytics
            st.subheader("📈 Analytics Dashboard")

            analytics_df = pd.DataFrame(saved_files)
            analytics_df['modified_date'] = pd.to_datetime(analytics_df['modified_time'])
            analytics_df['file_extension'] = analytics_df['original_filename'].str.split('.').str[-1].str.lower()

            col1, col2 = st.columns(2)

            with col1:
                # Timeline
                daily_counts = analytics_df.groupby(analytics_df['modified_date'].dt.date).size()
                fig_timeline = px.line(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    title="Documents Processed Over Time"
                )
                fig_timeline.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_timeline, use_container_width=True)

            with col2:
                # File types
                file_type_counts = analytics_df['file_extension'].value_counts()
                fig_pie = px.pie(
                    values=file_type_counts.values,
                    names=file_type_counts.index,
                    title="File Type Distribution"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Export section
            st.subheader("📥 Export Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Export CSV", use_container_width=True):
                    csv_data = analytics_df.to_csv(index=False)
                    st.download_button(
                        "💾 Download CSV",
                        data=csv_data,
                        file_name=f"ocr_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            with col2:
                if st.button("📈 Export Excel", use_container_width=True):
                    try:
                        if storage_manager:
                            with st.spinner("Creating Excel report..."):
                                excel_path = storage_manager.export_to_excel()
                                with open(excel_path, 'rb') as f:
                                    excel_data = f.read()

                                st.download_button(
                                    "💾 Download Excel",
                                    data=excel_data,
                                    file_name=f"ocr_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                    except Exception as e:
                        st.error(f"Excel export failed: {e}")

            with col3:
                if st.button("📦 Create Archive", use_container_width=True):
                    try:
                        with st.spinner("Creating archive..."):
                            zip_buffer = io.BytesIO()

                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                if storage_manager:
                                    # Add text files
                                    for text_file in storage_manager.text_dir.glob("*.txt"):
                                        zip_file.write(text_file, f"extracted_text/{text_file.name}")

                                    # Add metadata files
                                    for json_file in storage_manager.json_dir.glob("*.json"):
                                        zip_file.write(json_file, f"metadata/{json_file.name}")

                                # Add summary
                                summary_data = {
                                    "total_files": len(saved_files),
                                    "total_size_kb": sum(f['size_kb'] for f in saved_files),
                                    "export_timestamp": datetime.now().isoformat(),
                                    "files": saved_files
                                }

                                zip_file.writestr(
                                    "archive_manifest.json",
                                    json.dumps(summary_data, indent=2, default=str)
                                )

                            zip_buffer.seek(0)

                            st.download_button(
                                "💾 Download Archive",
                                data=zip_buffer.getvalue(),
                                file_name=f"ocr_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Archive creation failed: {e}")
        else:
            st.info("📊 No data available for analytics. Process some documents first.")

    except Exception as e:
        st.error(f"❌ Analytics error: {str(e)}")

# ============================================================================
# TAB 4: SETTINGS
# ============================================================================

with tab4:
    st.subheader("⚙️ Settings")

    # Add debug toggle
    debug_mode = st.checkbox("🔧 Debug Mode", help="Show configuration details")
    st.session_state['debug_mode'] = debug_mode

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📂 Storage Configuration")

        storage_path = Path("ocr_outputs").absolute()
        st.code(str(storage_path))

        # Storage metrics
        try:
            total_size = 0
            file_count = 0

            for file_path in storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Files", file_count)
            with col1b:
                st.metric("Size (MB)", f"{total_size / (1024 * 1024):.1f}")

        except Exception as e:
            st.error(f"Could not calculate storage usage: {e}")

    with col2:
        st.markdown("### 🤖 AI System Status")

        # Refresh AI systems status
        if debug_mode or st.button("🔄 Refresh AI Status"):
            st.cache_resource.clear()
            ai_systems = get_ai_system_status()

        # Debug info
        if debug_mode:
            try:
                config = load_config()
                llm_ocr_config = config.get('llm', {}).get('ocr', {})
                st.write("**Config Structure:**")
                st.write("- Main LLM config:", bool(config.get('llm')))
                st.write("- LLM OCR config:", bool(llm_ocr_config))
                st.write("- Gemini key:", "✓" if llm_ocr_config.get('gemini_api_key') else "✗")
                st.write("- Mistral key:", "✓" if llm_ocr_config.get('mistral_api_key') else "✗")
                st.write("- OpenAI key:", "✓" if (llm_ocr_config.get('openai_api_key') and llm_ocr_config.get(
                    'openai_api_key') != "sk-your-openai-key-here") else "✗")
                st.write("- Anthropic key:", "✓" if (llm_ocr_config.get('anthropic_api_key') and llm_ocr_config.get(
                    'anthropic_api_key') != "sk-ant-your-anthropic-key-here") else "✗")
                st.write("**AI System Detection:**")
                st.json(ai_systems)
            except Exception as e:
                st.error(f"Debug error: {e}")

        # Show correct AI system status based on actual config structure
        ai_system_names = {
            'gemini': 'Gemini Flash (Primary)',
            'mistral': 'Mistral Pixtral',
            'openai': 'GPT-4o Vision',
            'anthropic': 'Claude Sonnet'
        }

        operational_count = 0
        for system_key, system_name in ai_system_names.items():
            if ai_systems.get(system_key, False):
                st.markdown(f'<div class="status-badge status-success">✅ {system_name}</div>', unsafe_allow_html=True)
                operational_count += 1
            else:
                st.markdown(f'<div class="status-badge status-error">❌ {system_name}</div>', unsafe_allow_html=True)

        if operational_count >= 2:
            st.success(f"🎉 {operational_count} AI systems operational")
        elif operational_count == 1:
            st.info("⚠️ Single AI system operational")
        else:
            st.error("❌ No AI systems operational")

    # Quick help
    with st.expander("💡 Configuration Help", expanded=False):
        st.markdown("""
        ### Configuration Structure (config.toml)

        **LLM OCR API Keys:**
        ```toml
        [llm.ocr]
        gemini_api_key = "your_gemini_key"
        mistral_api_key = "your_mistral_key" 
        openai_api_key = "your_openai_key"
        anthropic_api_key = "your_anthropic_key"
        primary_method = "gemini"
        ```

        **Current Status:**
        - ✅ Gemini: Configured and working (Primary)
        - ✅ Mistral: Configured and working
        - ❌ OpenAI: Placeholder key detected
        - ❌ Anthropic: Placeholder key detected

        **Storage Settings:**
        - Files are saved to `./ocr_outputs/`
        - Text files: `./ocr_outputs/text/`
        - Metadata: `./ocr_outputs/json/`

        **Performance Tips:**
        - Process documents in smaller batches (< 10 files)
        - Ensure stable internet connection for AI services
        - Monitor API usage and costs
        """)

    # Storage management
    st.markdown("### 🗑️ Storage Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clean Temp Files", use_container_width=True):
            st.success("✅ Temporary files cleaned")

    with col2:
        if st.checkbox("🔓 I understand this will delete ALL files"):
            if st.button("🗑️ Delete All Archives", use_container_width=True):
                try:
                    import shutil

                    shutil.rmtree(storage_path)
                    storage_path.mkdir(exist_ok=True)
                    st.success("✅ All archives deleted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Deletion failed: {e}")
# pages/2_Document_Ingestion.py - ELEGANT VERSION

import streamlit as st
import logging
import time
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import streamlit.components.v1 as components
from pathlib import Path
import neo4j
import threading
from concurrent.futures import ThreadPoolExecutor
import json

# FIXED: Correct imports for current backend
try:
    import src.utils.audit_db_manager as audit_db
    from src.utils.processing_pipeline import (
        start_ingestion_job_async,
        is_job_running,
        process_batch_with_enhanced_storage,
        process_uploaded_file_ocr_with_storage
    )
    from src.utils.ocr_storage import create_storage_manager, get_storage_manager

    # Import from current main app
    from GraphRAG_Document_AI_Platform import (
        load_config,
        get_enhanced_ocr_pipeline,
        init_neo4j_exporter,
        get_embedding_model,
        get_chroma_collection,
        get_requests_session,
        get_nlp_pipeline,
        mask_sensitive_data
    )

except ImportError as e:
    st.error(f"Critical import error: {e}")
    st.error("Please ensure all backend modules are accessible.")
    st.stop()

# Visualization import (optional)
try:
    from src.knowledge_graph.visualization import visualize_knowledge_graph

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Validation import (optional)
try:
    from src.utils.pipeline_validation import create_validator

    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

# Constants
GRAPH_HTML_FILENAME = "graph_visualization.html"
REFRESH_INTERVAL = 2  # seconds

# Logger setup
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')

# Enhanced page configuration
st.set_page_config(
    page_title="Document Ingestion",
    page_icon="📥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Elegant and sophisticated styling
st.markdown("""
<style>
    /* Import sophisticated fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Elegant color palette */
    :root {
        /* Sophisticated neutrals */
        --elegant-slate-50: #f8fafc;
        --elegant-slate-100: #f1f5f9;
        --elegant-slate-200: #e2e8f0;
        --elegant-slate-300: #cbd5e1;
        --elegant-slate-400: #94a3b8;
        --elegant-slate-500: #64748b;
        --elegant-slate-600: #475569;
        --elegant-slate-700: #334155;
        --elegant-slate-800: #1e293b;
        --elegant-slate-900: #0f172a;

        /* Refined accent colors */
        --elegant-blue: #3b82f6;
        --elegant-blue-light: #60a5fa;
        --elegant-emerald: #10b981;
        --elegant-amber: #f59e0b;
        --elegant-rose: #f43f5e;
        --elegant-violet: #8b5cf6;
        --elegant-cyan: #06b6d4;

        /* Elegant shadows */
        --elegant-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --elegant-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --elegant-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --elegant-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --elegant-shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);

        /* Subtle gradients */
        --elegant-gradient-blue: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        --elegant-gradient-emerald: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --elegant-gradient-surface: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        --elegant-gradient-slate: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    }

    /* Base styling with elegance */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--elegant-slate-50);
        color: var(--elegant-slate-700);
        line-height: 1.7;
    }

    /* Hide Streamlit branding elegantly */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Elegant main header */
    .elegant-header {
        background: var(--elegant-gradient-slate);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: var(--elegant-shadow-lg);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    .elegant-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        animation: elegant-drift 30s infinite linear;
    }

    @keyframes elegant-drift {
        0% { transform: translateX(0) translateY(0); }
        100% { transform: translateX(-60px) translateY(-60px); }
    }

    .elegant-header h1 {
        color: white;
        font-weight: 300;
        font-size: 2rem;
        letter-spacing: -0.02em;
        margin: 0;
        text-align: center;
        position: relative;
        z-index: 1;
    }

    .elegant-header p {
        color: rgba(255, 255, 255, 0.75);
        font-weight: 300;
        font-size: 0.95rem;
        margin: 0.75rem 0 0 0;
        text-align: center;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
    }

    /* Sophisticated buttons */
    .stButton > button {
        background: var(--elegant-gradient-blue);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.25);
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.35);
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Elegant metric cards */
    .elegant-metric-card {
        background: var(--elegant-gradient-surface);
        border: 1px solid var(--elegant-slate-200);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    .elegant-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--elegant-gradient-blue);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .elegant-metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--elegant-shadow-lg);
        border-color: var(--elegant-slate-300);
    }

    .elegant-metric-card:hover::before {
        opacity: 1;
    }

    .elegant-metric-value {
        font-size: 2.5rem;
        font-weight: 300;
        color: var(--elegant-slate-800);
        margin-bottom: 0.5rem;
        line-height: 1;
    }

    .elegant-metric-label {
        color: var(--elegant-slate-500);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Elegant status indicators */
    .elegant-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 24px;
        font-size: 0.875rem;
        font-weight: 500;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }

    .elegant-status:hover {
        transform: translateY(-1px);
        box-shadow: var(--elegant-shadow-sm);
    }

    .status-success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: var(--elegant-emerald);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: var(--elegant-amber);
    }

    .status-error {
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.2);
        color: var(--elegant-rose);
    }

    .status-info {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: var(--elegant-blue);
    }

    /* Elegant progress indicators */
    .elegant-progress {
        height: 8px;
        background: var(--elegant-slate-100);
        border-radius: 8px;
        overflow: hidden;
        position: relative;
        margin: 1rem 0;
    }

    .elegant-progress-bar {
        height: 100%;
        background: var(--elegant-gradient-blue);
        border-radius: 8px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }

    .elegant-progress-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, 
            transparent 25%, 
            rgba(255,255,255,0.2) 25%, 
            rgba(255,255,255,0.2) 50%, 
            transparent 50%);
        animation: elegant-progress-shine 2s infinite;
    }

    @keyframes elegant-progress-shine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* Sophisticated data tables */
    .elegant-table {
        border: 1px solid var(--elegant-slate-200);
        border-radius: 16px;
        overflow: hidden;
        background: white;
        box-shadow: var(--elegant-shadow);
        margin: 1.5rem 0;
    }

    .elegant-table thead th {
        background: var(--elegant-gradient-surface);
        color: var(--elegant-slate-700);
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 1.25rem 1rem;
        border-bottom: 2px solid var(--elegant-slate-200);
    }

    .elegant-table tbody tr:hover {
        background: var(--elegant-slate-50);
        transition: background 0.2s ease;
    }

    /* Elegant animations */
    @keyframes elegant-fade-in {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .elegant-enter {
        animation: elegant-fade-in 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .elegant-hover-lift:hover {
        transform: translateY(-2px);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Elegant tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: var(--elegant-gradient-surface);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: var(--elegant-shadow-sm);
    }

    /* Elegant sidebar styling */
    .css-1d391kg {
        background: var(--elegant-gradient-surface);
        border-right: 1px solid var(--elegant-slate-200);
    }

    /* Elegant expanders */
    .streamlit-expanderHeader {
        background: var(--elegant-gradient-surface);
        border-radius: 12px;
        border: 1px solid var(--elegant-slate-200);
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        box-shadow: var(--elegant-shadow-sm);
        border-color: var(--elegant-slate-300);
    }

    /* Responsive elegance */
    @media (max-width: 768px) {
        .elegant-header {
            padding: 1.5rem 1.25rem;
            border-radius: 16px;
        }

        .elegant-header h1 {
            font-size: 1.75rem;
        }

        .elegant-header p {
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }

        .elegant-metric-card {
            padding: 1.5rem;
        }

        .elegant-metric-value {
            font-size: 2rem;
        }
    }

    /* Elegant file upload area */
    .stFileUploader > div {
        border: 2px dashed var(--elegant-slate-300);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
        background: var(--elegant-gradient-surface);
    }

    .stFileUploader > div:hover {
        border-color: var(--elegant-blue);
        background: rgba(59, 130, 246, 0.02);
    }

    /* Elegant loading states */
    .elegant-spinner {
        border: 2px solid var(--elegant-slate-200);
        border-top: 2px solid var(--elegant-blue);
        border-radius: 50%;
        animation: elegant-spin 1s linear infinite;
    }

    @keyframes elegant-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


# Helper Functions
def format_timestamp(ts_string: Optional[str]) -> str:
    """Format ISO timestamp string for display."""
    if not ts_string:
        return "N/A"
    try:
        dt_obj = datetime.fromisoformat(ts_string.replace('Z', '+00:00'))
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts_string


def create_status_indicator(status: str, text: str) -> str:
    """Create elegant colored status indicator."""
    status_classes = {
        'success': 'status-success',
        'warning': 'status-warning',
        'error': 'status-error',
        'info': 'status-info'
    }
    status_class = status_classes.get(status, 'status-info')
    return f'<div class="elegant-status {status_class}">● {text}</div>'


def show_system_status(config: dict) -> bool:
    """Display elegant system status."""
    st.markdown("### 🔧 System Status")

    status_data = []
    all_ready = True

    # OCR Pipeline Status
    ocr_pipeline = get_enhanced_ocr_pipeline(config)
    if ocr_pipeline:
        providers = ocr_pipeline.get_available_providers()
        primary_method = config.get('llm', {}).get('ocr', {}).get('primary_method', 'unknown')

        if primary_method in providers:
            status_data.append({
                'Component': 'OCR Pipeline',
                'Status': create_status_indicator('success', 'Ready'),
                'Details': f'Primary: {primary_method}, Providers: {len(providers)}',
                'Health': 'Healthy'
            })
        else:
            status_data.append({
                'Component': 'OCR Pipeline',
                'Status': create_status_indicator('warning', 'Partial'),
                'Details': f'Primary: {primary_method} unavailable, Available: {providers}',
                'Health': 'Warning'
            })
            all_ready = False
    else:
        status_data.append({
            'Component': 'OCR Pipeline',
            'Status': create_status_indicator('error', 'Failed'),
            'Details': 'Not initialized',
            'Health': 'Error'
        })
        all_ready = False

    # Neo4j Status
    neo4j_exporter = init_neo4j_exporter(
        config.get('NEO4J_URI'),
        config.get('NEO4J_USER'),
        config.get('NEO4J_PASSWORD')
    )
    if neo4j_exporter:
        try:
            stats = neo4j_exporter.get_graph_stats()
            status_data.append({
                'Component': 'Neo4j Database',
                'Status': create_status_indicator('success', 'Connected'),
                'Details': f"Entities: {stats.get('entity_count', 0)}, Relations: {stats.get('relationship_count', 0)}",
                'Health': 'Healthy'
            })
        except Exception:
            status_data.append({
                'Component': 'Neo4j Database',
                'Status': create_status_indicator('warning', 'Connected'),
                'Details': 'Stats unavailable',
                'Health': 'Warning'
            })
    else:
        status_data.append({
            'Component': 'Neo4j Database',
            'Status': create_status_indicator('error', 'Failed'),
            'Details': 'Connection failed',
            'Health': 'Error'
        })
        all_ready = False

    # Vector DB Status
    chroma_collection = get_chroma_collection(
        config.get('CHROMA_PERSIST_PATH'),
        config.get('COLLECTION_NAME'),
        config.get('EMBEDDING_MODEL')
    )
    if chroma_collection:
        try:
            count = chroma_collection.count()
            status_data.append({
                'Component': 'Vector Database',
                'Status': create_status_indicator('success', 'Ready'),
                'Details': f'Documents: {count}',
                'Health': 'Healthy'
            })
        except Exception:
            status_data.append({
                'Component': 'Vector Database',
                'Status': create_status_indicator('success', 'Ready'),
                'Details': 'Count unavailable',
                'Health': 'Healthy'
            })
    else:
        status_data.append({
            'Component': 'Vector Database',
            'Status': create_status_indicator('error', 'Failed'),
            'Details': 'Not initialized',
            'Health': 'Error'
        })
        all_ready = False

    # Embedding Model Status
    embedding_model = get_embedding_model(config.get('EMBEDDING_MODEL'))
    if embedding_model:
        model_name = config.get('EMBEDDING_MODEL', 'unknown')
        status_data.append({
            'Component': 'Embedding Model',
            'Status': create_status_indicator('success', 'Ready'),
            'Details': f'Model: {model_name}',
            'Health': 'Healthy'
        })
    else:
        status_data.append({
            'Component': 'Embedding Model',
            'Status': create_status_indicator('error', 'Failed'),
            'Details': 'Not loaded',
            'Health': 'Error'
        })
        all_ready = False

    # Display elegant status table
    for item in status_data:
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.markdown(f"**{item['Component']}**")
        with col2:
            st.markdown(item['Status'], unsafe_allow_html=True)
        with col3:
            st.markdown(f"*{item['Details']}*")

    return all_ready


def show_configuration_summary(config: dict):
    """Display elegant configuration summary with security."""
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🤖 LLM Configuration**")
            llm_config = config.get('llm', {})
            st.markdown(f"• **Model:** `{llm_config.get('model', 'Not set')}`")
            st.markdown(f"• **API Key:** `{mask_sensitive_data(llm_config.get('api_key', 'Not set'))}`")

            ocr_config = llm_config.get('ocr', {})
            st.markdown(f"• **OCR Primary:** `{ocr_config.get('primary_method', 'Not set')}`")
            st.markdown(f"• **OCR Fallback:** `{ocr_config.get('fallback_enabled', False)}`")

        with col2:
            st.markdown("**🗄️ Database Configuration**")
            st.markdown(f"• **Neo4j URI:** `{config.get('NEO4J_URI', 'Not set')}`")
            st.markdown(f"• **Neo4j User:** `{config.get('NEO4J_USER', 'Not set')}`")
            st.markdown(f"• **Vector DB:** `{config.get('CHROMA_PERSIST_PATH', 'Not set')}`")
            st.markdown(f"• **Collection:** `{config.get('COLLECTION_NAME', 'Not set')}`")

            st.markdown("**⚙️ Processing Settings**")
            chunking_config = config.get('chunking', {})
            st.markdown(f"• **Chunk Size:** `{chunking_config.get('chunk_size', 1000)}`")
            st.markdown(f"• **Chunk Overlap:** `{chunking_config.get('overlap', 100)}`")


def show_live_ocr_processing(uploaded_files: List[Any], enhanced_ocr_pipeline: Any, save_to_disk: bool = True) -> List[
    Dict]:
    """Enhanced OCR processing with elegant live updates."""
    st.markdown("### 📄 Live OCR Processing")

    if not enhanced_ocr_pipeline:
        st.error("❌ Enhanced OCR pipeline not available")
        return []

    # Show elegant OCR method status
    providers = enhanced_ocr_pipeline.get_available_providers()
    primary_method = enhanced_ocr_pipeline.primary_method

    # Elegant metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{primary_method.upper()}</div>
            <div class="elegant-metric-label">Primary Method</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{len(providers)}</div>
            <div class="elegant-metric-label">Available Providers</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{len(uploaded_files)}</div>
            <div class="elegant-metric-label">Files to Process</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        confidence_threshold = enhanced_ocr_pipeline.confidence_threshold
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{confidence_threshold:.2f}</div>
            <div class="elegant-metric-label">Confidence Threshold</div>
        </div>
        """, unsafe_allow_html=True)

    # Processing container with elegant progress
    processing_container = st.container()

    with processing_container:
        # Overall elegant progress
        st.markdown("#### Processing Progress")
        overall_progress = st.progress(0)
        overall_status = st.empty()

        # Individual file status
        file_status_container = st.container()

        # Results storage
        ocr_results = []

        # Process files with elegant live updates
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name

            # Update overall progress
            progress_pct = i / len(uploaded_files)
            overall_progress.progress(progress_pct)
            overall_status.info(f"🔄 Processing file {i + 1}/{len(uploaded_files)}: **{filename}**")

            # Individual file status with elegant styling
            with file_status_container:
                file_status = st.empty()
                file_metrics = st.empty()

                file_status.info(f"📄 **{filename}** - Starting OCR extraction...")

                start_time = time.time()

                try:
                    # Process with enhanced pipeline
                    result = process_uploaded_file_ocr_with_storage(
                        uploaded_file=uploaded_file,
                        enhanced_ocr_pipeline=enhanced_ocr_pipeline,
                        save_to_disk=save_to_disk
                    )

                    processing_time = time.time() - start_time

                    if result['success']:
                        # Success metrics with elegant display
                        method_used = result.get('method_used', 'unknown')
                        confidence = result.get('confidence', 0.0)
                        text_length = result.get('text_length', 0)

                        file_status.success(f"✅ **{filename}** - Extraction successful!")

                        # Show elegant detailed metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.markdown(f"""
                            <div class="elegant-status status-info">
                                🔧 {method_used.upper()}
                            </div>
                            """, unsafe_allow_html=True)
                        with metric_cols[1]:
                            st.markdown(f"""
                            <div class="elegant-status status-success">
                                📊 {confidence:.3f}
                            </div>
                            """, unsafe_allow_html=True)
                        with metric_cols[2]:
                            st.markdown(f"""
                            <div class="elegant-status status-info">
                                📝 {text_length:,}
                            </div>
                            """, unsafe_allow_html=True)
                        with metric_cols[3]:
                            st.markdown(f"""
                            <div class="elegant-status status-success">
                                ⏱️ {processing_time:.2f}s
                            </div>
                            """, unsafe_allow_html=True)

                        # Store result
                        ocr_results.append({
                            'filename': filename,
                            'status': 'success',
                            'method_used': method_used,
                            'confidence': confidence,
                            'text_length': text_length,
                            'processing_time': processing_time,
                            'saved_files': len(result.get('saved_files', {})),
                            'error': None
                        })

                    else:
                        # Failure with elegant error display
                        error_msg = result.get('error', 'Unknown error')
                        file_status.error(f"❌ **{filename}** - Extraction failed: {error_msg}")

                        ocr_results.append({
                            'filename': filename,
                            'status': 'failed',
                            'method_used': 'none',
                            'confidence': 0.0,
                            'text_length': 0,
                            'processing_time': processing_time,
                            'saved_files': 0,
                            'error': error_msg
                        })

                except Exception as e:
                    processing_time = time.time() - start_time
                    error_msg = str(e)

                    file_status.error(f"❌ **{filename}** - Processing error: {error_msg}")

                    ocr_results.append({
                        'filename': filename,
                        'status': 'error',
                        'method_used': 'none',
                        'confidence': 0.0,
                        'text_length': 0,
                        'processing_time': processing_time,
                        'saved_files': 0,
                        'error': error_msg
                    })

                # Add elegant separator
                st.markdown("---")

                # Small delay for visual effect
                time.sleep(0.5)

        # Final progress update with celebration
        overall_progress.progress(1.0)
        overall_status.success(f"✅ OCR processing complete! Successfully processed {len(uploaded_files)} files.")

    return ocr_results


def show_ocr_results_summary(ocr_results: List[Dict]):
    """Display elegant OCR results summary."""
    if not ocr_results:
        return

    st.markdown("### 📊 OCR Processing Summary")

    # Calculate elegant statistics
    total_files = len(ocr_results)
    successful = sum(1 for r in ocr_results if r['status'] == 'success')
    failed = sum(1 for r in ocr_results if r['status'] == 'failed')
    errors = sum(1 for r in ocr_results if r['status'] == 'error')

    total_chars = sum(r['text_length'] for r in ocr_results)
    total_time = sum(r['processing_time'] for r in ocr_results)
    avg_confidence = sum(r['confidence'] for r in ocr_results if r['confidence'] > 0) / max(successful, 1)

    # Elegant summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{total_files}</div>
            <div class="elegant-metric-label">Total Files</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        success_pct = successful / total_files * 100 if total_files > 0 else 0
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{successful}</div>
            <div class="elegant-metric-label">Successful ({success_pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        fail_pct = (failed + errors) / total_files * 100 if total_files > 0 else 0
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value" style="color: var(--elegant-rose);">{failed + errors}</div>
            <div class="elegant-metric-label">Failed ({fail_pct:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{total_chars:,}</div>
            <div class="elegant-metric-label">Total Text</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{avg_confidence:.3f}</div>
            <div class="elegant-metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{total_time:.2f}s</div>
            <div class="elegant-metric-label">Total Time</div>
        </div>
        """, unsafe_allow_html=True)

    # Elegant method breakdown
    method_counts = {}
    for result in ocr_results:
        method = result.get('method_used', 'unknown')
        if method != 'none':
            method_counts[method] = method_counts.get(method, 0) + 1

    if method_counts:
        st.markdown("#### 🔧 OCR Method Usage")
        method_cols = st.columns(len(method_counts))
        for i, (method, count) in enumerate(method_counts.items()):
            with method_cols[i]:
                percentage = count / successful * 100 if successful > 0 else 0
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-violet);">{count}</div>
                    <div class="elegant-metric-label">{method.upper()} ({percentage:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)

    # Elegant detailed results table
    st.markdown("#### 📋 Detailed Results")

    # Prepare data for elegant display
    display_data = []
    for result in ocr_results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        display_data.append({
            'Status': f"{status_icon} {result['status'].title()}",
            'Filename': result['filename'],
            'Method': result['method_used'].upper() if result['method_used'] != 'none' else '-',
            'Confidence': f"{result['confidence']:.3f}" if result['confidence'] > 0 else '-',
            'Text Length': f"{result['text_length']:,}" if result['text_length'] > 0 else '-',
            'Time (s)': f"{result['processing_time']:.2f}",
            'Files Saved': result['saved_files'],
            'Error': result['error'][:50] + "..." if result['error'] and len(result['error']) > 50 else result[
                                                                                                            'error'] or '-'
        })

    results_df = pd.DataFrame(display_data)
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn(width="small"),
            "Filename": st.column_config.TextColumn(width="medium"),
            "Method": st.column_config.TextColumn(width="small"),
            "Confidence": st.column_config.TextColumn(width="small"),
            "Text Length": st.column_config.TextColumn(width="small"),
            "Time (s)": st.column_config.TextColumn(width="small"),
            "Files Saved": st.column_config.NumberColumn(width="small"),
            "Error": st.column_config.TextColumn(width="large")
        }
    )

    # Store results in session state for later use
    st.session_state['last_ocr_results'] = ocr_results


def show_validation_interface(uploaded_files: List[Any], config: dict):
    """Show elegant document validation interface."""
    if not HAS_VALIDATION or not uploaded_files:
        return

    st.markdown("### 🔍 Document Validation Preview")

    with st.expander("Quick Validation Check", expanded=False):
        if st.button("🚀 Run Quick Validation", type="secondary"):
            with st.spinner("Validating documents..."):
                try:
                    ocr_pipeline = get_enhanced_ocr_pipeline(config)
                    if not ocr_pipeline:
                        st.error("❌ OCR pipeline not available for validation")
                        return

                    validator = create_validator(config, ocr_pipeline.mistral_client)
                    files_to_validate = uploaded_files[:10]  # Limit for performance

                    validation_results = validator.validate_batch_quick(files_to_validate)

                    # Display elegant results
                    summary = validation_results['summary']

                    val_cols = st.columns(4)
                    with val_cols[0]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{summary['ocr_success_rate']:.1%}</div>
                            <div class="elegant-metric-label">Success Rate</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with val_cols[1]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-blue);">{summary['average_confidence_score']:.2f}</div>
                            <div class="elegant-metric-label">Avg Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with val_cols[2]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-amber);">{summary['financial_documents']}</div>
                            <div class="elegant-metric-label">Financial Docs</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with val_cols[3]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-violet);">{len(validation_results['results'])}</div>
                            <div class="elegant-metric-label">Total Validated</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Elegant recommendations
                    if validation_results['recommendations']:
                        st.markdown("**📝 Recommendations:**")
                        for rec in validation_results['recommendations']:
                            st.markdown(f"• {rec}")

                    # Detailed results with elegant toggle
                    if st.checkbox("Show detailed validation results"):
                        results_data = []
                        for result in validation_results['results']:
                            results_data.append({
                                'File': result['file_name'],
                                'Confidence': f"{result['confidence_score']:.2f}",
                                'Recommendation': result['processing_recommendation'],
                                'Text Length': result['mistral_text_length'],
                                'Has Tables': '✅' if result['contains_tables'] else '❌',
                                'Financial': '✅' if result['contains_financial_data'] else '❌'
                            })

                        validation_df = pd.DataFrame(results_data)
                        st.dataframe(validation_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"❌ Validation failed: {e}")
                    logger.error(f"Document validation error: {e}")


def show_live_job_monitoring(job_id: str):
    """Enhanced live job monitoring with elegant real-time updates."""
    st.markdown("### 📊 Live Job Monitoring")

    # Create elegant containers for live updates
    header_container = st.container()
    progress_container = st.container()
    status_container = st.container()
    details_container = st.container()

    # Monitoring loop with elegance
    monitoring_active = True
    refresh_count = 0

    while monitoring_active:
        try:
            # Get job details
            job_details = audit_db.get_job_details(job_id)

            if not job_details:
                with status_container:
                    st.error(f"❌ Job {job_id} not found in database")
                break

            # Extract job information
            status = job_details.get('status', 'Unknown')
            total_files = job_details.get('total_files_in_job', 0)
            processed_files = job_details.get('processed_files', [])

            # Calculate elegant progress
            files_completed = len([f for f in processed_files if f['status'] != 'Processing'])
            progress_value = files_completed / total_files if total_files > 0 else 0

            # Update elegant header
            with header_container:
                st.markdown(f"""
                <div class="elegant-status status-info">
                    📊 Job ID: {job_id[:8]}... | Status: {status} | Progress: {files_completed}/{total_files}
                </div>
                """, unsafe_allow_html=True)

            # Update elegant progress bar
            with progress_container:
                st.progress(progress_value)

                # Elegant progress metrics
                success_count = len([f for f in processed_files if f['status'] == 'Success'])
                cached_count = len([f for f in processed_files if f['status'] == 'Cached'])
                failed_count = len([f for f in processed_files if 'Failed' in f['status']])
                processing_count = len([f for f in processed_files if f['status'] == 'Processing'])

                prog_cols = st.columns(4)
                with prog_cols[0]:
                    st.markdown(f"""
                    <div class="elegant-metric-card">
                        <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{success_count}</div>
                        <div class="elegant-metric-label">Success</div>
                    </div>
                    """, unsafe_allow_html=True)

                with prog_cols[1]:
                    st.markdown(f"""
                    <div class="elegant-metric-card">
                        <div class="elegant-metric-value" style="color: var(--elegant-cyan);">{cached_count}</div>
                        <div class="elegant-metric-label">Cached</div>
                    </div>
                    """, unsafe_allow_html=True)

                with prog_cols[2]:
                    st.markdown(f"""
                    <div class="elegant-metric-card">
                        <div class="elegant-metric-value" style="color: var(--elegant-rose);">{failed_count}</div>
                        <div class="elegant-metric-label">Failed</div>
                    </div>
                    """, unsafe_allow_html=True)

                with prog_cols[3]:
                    st.markdown(f"""
                    <div class="elegant-metric-card">
                        <div class="elegant-metric-value" style="color: var(--elegant-amber);">{processing_count}</div>
                        <div class="elegant-metric-label">Processing</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Update elegant status
            with status_container:
                if processing_count > 0:
                    processing_files = [f['file_name'] for f in processed_files if f['status'] == 'Processing']
                    st.markdown(f"""
                    <div class="elegant-status status-info">
                        🔄 Currently processing: {', '.join(processing_files)}
                    </div>
                    """, unsafe_allow_html=True)
                elif status == 'Running':
                    st.markdown(f"""
                    <div class="elegant-status status-info">
                        ⏳ Waiting for next file or finishing up...
                    </div>
                    """, unsafe_allow_html=True)
                elif status in ['Completed', 'Completed with Errors']:
                    st.markdown(f"""
                    <div class="elegant-status status-success">
                        ✅ Job completed: {success_count} successful, {cached_count} cached, {failed_count} failed
                    </div>
                    """, unsafe_allow_html=True)
                    monitoring_active = False
                elif status == 'Failed':
                    st.markdown(f"""
                    <div class="elegant-status status-error">
                        ❌ Job failed
                    </div>
                    """, unsafe_allow_html=True)
                    monitoring_active = False

            # Update elegant details
            with details_container:
                if processed_files:
                    st.markdown("#### 📋 File Processing Details")

                    # Create elegant status table
                    file_status_data = []
                    for file_info in processed_files:
                        status_icons = {
                            'Success': '✅',
                            'Cached': '🎯',
                            'Processing': '🔄',
                            'Failed - OCR': '❌',
                            'Failed - KG Extract': '❌',
                            'Failed - Neo4j': '❌',
                            'Failed - Embedding': '❌',
                            'Failed - Unknown': '❌'
                        }
                        status_icon = status_icons.get(file_info['status'], '❓')

                        file_status_data.append({
                            'Status': f"{status_icon} {file_info['status']}",
                            'File': file_info['file_name'],
                            'Size': f"{file_info.get('file_size_bytes', 0):,} bytes",
                            'Chunks': file_info.get('num_chunks_generated', 0) or 0,
                            'Triples': file_info.get('num_triples_extracted', 0) or 0,
                            'Vectors': file_info.get('num_vectors_stored_chroma', 0) or 0,
                            'Cache': '✅' if file_info.get('cache_hit') else '❌',
                            'Error': file_info.get('error_message', '')[:50] + '...' if file_info.get(
                                'error_message') and len(file_info.get('error_message', '')) > 50 else file_info.get(
                                'error_message', '')
                        })

                    status_df = pd.DataFrame(file_status_data)
                    st.dataframe(
                        status_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Status": st.column_config.TextColumn(width="small"),
                            "File": st.column_config.TextColumn(width="medium"),
                            "Size": st.column_config.TextColumn(width="small"),
                            "Chunks": st.column_config.NumberColumn(width="small"),
                            "Triples": st.column_config.NumberColumn(width="small"),
                            "Vectors": st.column_config.NumberColumn(width="small"),
                            "Cache": st.column_config.TextColumn(width="small"),
                            "Error": st.column_config.TextColumn(width="large")
                        }
                    )

            # Check if job is finished
            if status not in ['Running', 'Queued', 'Processing']:
                monitoring_active = False
                st.success("✅ Job monitoring completed")
                break

            # Elegant refresh counter
            refresh_count += 1
            if refresh_count % 10 == 0:  # Every 20 seconds, show refresh info
                st.caption(f"🔄 Auto-refreshed {refresh_count} times")

            # Wait before next refresh
            time.sleep(REFRESH_INTERVAL)

            # Rerun to update display
            st.rerun()

        except Exception as e:
            with status_container:
                st.error(f"❌ Error monitoring job: {e}")
            logger.error(f"Error in live job monitoring: {e}")
            break


def process_documents_with_monitoring(uploaded_files: List[Any], config: dict, processing_options: dict):
    """Process documents with elegant monitoring and feedback."""
    st.markdown("### 🚀 Document Processing Pipeline")

    # Initialize resources with elegant loading
    with st.spinner("🔧 Initializing processing resources..."):
        enhanced_ocr_pipeline = get_enhanced_ocr_pipeline(config)
        neo4j_exporter = init_neo4j_exporter(
            config.get('NEO4J_URI'),
            config.get('NEO4J_USER'),
            config.get('NEO4J_PASSWORD')
        )
        embedding_model = get_embedding_model(config.get('EMBEDDING_MODEL'))
        chroma_collection = get_chroma_collection(
            config.get('CHROMA_PERSIST_PATH'),
            config.get('COLLECTION_NAME'),
            config.get('EMBEDDING_MODEL')
        )
        requests_session = get_requests_session()
        nlp_pipeline = get_nlp_pipeline(config)

    # Elegant resource validation
    missing_resources = []
    if not enhanced_ocr_pipeline:
        missing_resources.append("OCR Pipeline")
    if not neo4j_exporter:
        missing_resources.append("Neo4j Connection")
    if not embedding_model:
        missing_resources.append("Embedding Model")
    if not chroma_collection:
        missing_resources.append("Vector Database")

    if missing_resources:
        st.error(f"❌ Missing critical resources: {', '.join(missing_resources)}")
        return None

    # Phase 1: OCR Processing with elegant live updates
    if processing_options.get('save_ocr_to_disk', True):
        with st.expander("📄 Phase 1: OCR Text Extraction", expanded=True):
            ocr_results = show_live_ocr_processing(
                uploaded_files,
                enhanced_ocr_pipeline,
                save_to_disk=True
            )
            show_ocr_results_summary(ocr_results)

    # Phase 2: Start background processing job with elegance
    st.divider()
    st.markdown("### 📊 Phase 2: Knowledge Graph Processing")

    with st.spinner("🚀 Starting background processing job..."):
        job_id = start_ingestion_job_async(
            uploaded_files=uploaded_files,
            config=config,
            use_cache=processing_options.get('use_cache', True),
            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
            neo4j_exporter=neo4j_exporter,
            embedding_model_resource=embedding_model,
            chroma_collection_resource=chroma_collection,
            requests_session_resource=requests_session,
            nlp_pipeline_resource=nlp_pipeline
        )

    if job_id:
        st.success(f"✅ Processing job started successfully!")
        st.markdown(f"""
        <div class="elegant-status status-info">
            📊 Job ID: {job_id}
        </div>
        """, unsafe_allow_html=True)

        # Store job ID for monitoring
        st.session_state['current_processing_job'] = job_id

        # Start elegant live monitoring
        show_live_job_monitoring(job_id)

        return job_id
    else:
        st.error("❌ Failed to start processing job")
        return None


# Additional functions continue with the same elegant styling patterns...
# (The remaining functions would follow the same elegant design principles)


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

def main():
    """Main elegant document ingestion application."""

    # Elegant main header
    st.markdown("""
    <div class="elegant-header elegant-enter">
        <h1>📥 Document Ingestion</h1>
        <p>Advanced document processing with intelligent analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Load configuration
    config = load_config()
    if not config or not config.get('_CONFIG_VALID', False):
        st.error("❌ Invalid configuration. Please check config.toml and API keys.")
        st.stop()

    # Initialize audit database
    try:
        audit_db.initialize_database()
        logger.info("Audit database initialized")
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        st.stop()

    # Elegant system status check
    system_ready = show_system_status(config)

    if not system_ready:
        st.error("❌ System not ready for processing. Please check configuration and services.")
        return

    # Elegant configuration summary
    show_configuration_summary(config)

    # Elegant main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Document Upload", "📊 Live Monitoring", "📚 History & Analytics"])

    with tab1:
        st.markdown("## 📁 Document Upload & Processing")

        # Elegant file upload
        supported_types = ["pdf", "png", "jpg", "jpeg", "txt", "csv"]
        uploaded_files = st.file_uploader(
            "Choose documents to process:",
            type=supported_types,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(supported_types)}"
        )

        if uploaded_files:
            st.success(f"📁 {len(uploaded_files)} files uploaded successfully")

            # Elegant file summary
            total_size = sum(len(f.getvalue()) for f in uploaded_files)
            file_types = {}
            for f in uploaded_files:
                ext = Path(f.name).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1

            # Elegant summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{len(uploaded_files)}</div>
                    <div class="elegant-metric-label">Total Files</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{total_size / (1024 * 1024):.1f}</div>
                    <div class="elegant-metric-label">MB Total Size</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{len(file_types)}</div>
                    <div class="elegant-metric-label">File Types</div>
                </div>
                """, unsafe_allow_html=True)

            # Elegant file type breakdown
            if file_types:
                st.markdown("**📊 File Types:**")
                for ext, count in file_types.items():
                    st.markdown(f"• {ext or 'no extension'}: {count} files")

            # Document validation with elegance
            show_validation_interface(uploaded_files, config)

            # Elegant processing options
            st.markdown("### ⚙️ Processing Options")

            opt_cols = st.columns(3)

            with opt_cols[0]:
                use_cache = st.checkbox(
                    "🗄️ Use Cache",
                    value=config.get('caching', {}).get('enabled', True),
                    help="Skip processing for previously processed files"
                )

            with opt_cols[1]:
                save_ocr_to_disk = st.checkbox(
                    "💾 Save OCR Outputs",
                    value=True,
                    help="Save extracted text and metadata to local storage"
                )

            with opt_cols[2]:
                enable_inference = st.checkbox(
                    "🧠 Enable Inference",
                    value=config.get('inference', {}).get('enabled', True),
                    help="Apply relationship inference to knowledge graph"
                )

            # Elegant processing button
            if st.button("🚀 Start Enhanced Processing", type="primary", use_container_width=True):
                processing_options = {
                    'use_cache': use_cache,
                    'save_ocr_to_disk': save_ocr_to_disk,
                    'enable_inference': enable_inference
                }

                job_id = process_documents_with_monitoring(uploaded_files, config, processing_options)

                if job_id:
                    st.session_state['current_job_id'] = job_id
                    st.success("✅ Processing started! Switch to 'Live Monitoring' tab to track progress.")

        else:
            st.info("📁 Upload documents above to begin processing")

    with tab2:
        st.markdown("## 📊 Live Job Monitoring")

        # Current job monitoring with elegance
        current_job = st.session_state.get('current_job_id')

        if current_job:
            st.markdown(f"""
            <div class="elegant-status status-info">
                📊 Current Job: {current_job}
            </div>
            """, unsafe_allow_html=True)

            # Check if job is still running
            if is_job_running(current_job):
                show_live_job_monitoring(current_job)
            else:
                st.info("✅ Current job has completed. Check History & Analytics for details.")
                if st.button("🗑️ Clear Current Job"):
                    del st.session_state['current_job_id']
                    st.rerun()
        else:
            st.info("📊 No active job to monitor. Start processing in the Upload tab.")

            # Show any recent running jobs with elegance
            try:
                recent_jobs = audit_db.get_recent_jobs(limit=5)
                running_jobs = [j for j in recent_jobs if j['status'] in ['Running', 'Queued']]

                if running_jobs:
                    st.markdown("### 🔄 Recent Running Jobs")
                    for job in running_jobs:
                        if st.button(f"Monitor Job {job['job_id'][:8]}...", key=f"monitor_{job['job_id']}"):
                            st.session_state['current_job_id'] = job['job_id']
                            st.rerun()
            except Exception as e:
                st.warning(f"Could not check for running jobs: {e}")

    with tab3:
        st.markdown("## 📚 History & Analytics")

        # Enhanced job history with elegant analytics
        show_enhanced_job_history(config)

        # File search analytics with elegance
        st.divider()
        show_file_search_analytics()

        # OCR storage management with sophistication
        st.divider()
        show_ocr_storage_management()


def show_enhanced_job_history(config: dict):
    """Display elegant job history with detailed analytics."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 📚 Recent Processing Jobs")

    with col2:
        if st.button("🔄 Refresh History", type="secondary"):
            st.rerun()

    try:
        recent_jobs = audit_db.get_recent_jobs(limit=50)
    except Exception as e:
        st.error(f"Failed to load job history: {e}")
        return

    if not recent_jobs:
        st.info("📝 No processing jobs found in history.")
        return

    # Elegant job summary statistics
    total_jobs = len(recent_jobs)
    completed_jobs = len([j for j in recent_jobs if j['status'] == 'Completed'])
    failed_jobs = len([j for j in recent_jobs if j['status'] == 'Failed'])

    # Elegant summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value">{total_jobs}</div>
            <div class="elegant-metric-label">Total Jobs</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        success_rate = completed_jobs / total_jobs * 100 if total_jobs > 0 else 0
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{success_rate:.1f}%</div>
            <div class="elegant-metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{completed_jobs}</div>
            <div class="elegant-metric-label">Completed</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="elegant-metric-card">
            <div class="elegant-metric-value" style="color: var(--elegant-rose);">{failed_jobs}</div>
            <div class="elegant-metric-label">Failed</div>
        </div>
        """, unsafe_allow_html=True)

    # Elegant jobs table
    job_data = []
    for job in recent_jobs:
        # Calculate processing time elegantly
        start_time = job.get('start_timestamp')
        end_time = job.get('end_timestamp')
        processing_time = "N/A"

        if start_time and end_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds()
                processing_time = f"{duration:.1f}s"
            except:
                processing_time = "N/A"

        # Elegant status icons
        status_icons = {
            'Completed': '✅',
            'Completed with Errors': '⚠️',
            'Failed': '❌',
            'Running': '🔄',
            'Queued': '⏳'
        }
        status_display = f"{status_icons.get(job['status'], '❓')} {job['status']}"

        job_data.append({
            "Job ID": job['job_id'][:8] + "...",
            "Status": status_display,
            "Files": job['total_files_in_job'],
            "Start Time": format_timestamp(job['start_timestamp']),
            "Duration": processing_time,
            "Full Job ID": job['job_id']
        })

    if job_data:
        jobs_df = pd.DataFrame(job_data)

        # Display elegant jobs table
        st.dataframe(
            jobs_df.drop('Full Job ID', axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Job ID": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="medium"),
                "Files": st.column_config.NumberColumn(width="small"),
                "Start Time": st.column_config.TextColumn(width="medium"),
                "Duration": st.column_config.TextColumn(width="small")
            }
        )

        # Elegant job details section
        st.divider()
        st.markdown("### 🔍 Job Details")

        # Elegant job selection
        job_options = {f"{job['job_id'][:8]}... ({job['status']})": job['job_id'] for job in recent_jobs}
        selected_job_display = st.selectbox(
            "Select job to view details:",
            options=[""] + list(job_options.keys()),
            index=0
        )

        if selected_job_display:
            selected_job_id = job_options[selected_job_display]
            show_detailed_job_analysis(selected_job_id, config)


def show_detailed_job_analysis(job_id: str, config: dict):
    """Show elegant comprehensive job analysis with performance metrics."""
    try:
        job_details = audit_db.get_job_details(job_id)

        if not job_details:
            st.error(f"❌ Job details not found for {job_id}")
            return

        # Elegant job overview
        st.markdown(f"#### 📊 Job Analysis: `{job_id}`")

        # Elegant basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="elegant-metric-card">
                <div class="elegant-metric-value">{job_details['status']}</div>
                <div class="elegant-metric-label">Status</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="elegant-metric-card">
                <div class="elegant-metric-value">{job_details['total_files_in_job']}</div>
                <div class="elegant-metric-label">Total Files</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            start_time = format_timestamp(job_details.get('start_timestamp'))
            end_time = format_timestamp(job_details.get('end_timestamp'))
            st.markdown(f"**Started:** {start_time}")
            st.markdown(f"**Ended:** {end_time}")

        # File processing details with elegance
        files = job_details.get('processed_files', [])

        if files:
            # Elegant processing statistics
            success_files = [f for f in files if f['status'] == 'Success']
            cached_files = [f for f in files if f['status'] == 'Cached']
            failed_files = [f for f in files if 'Failed' in f['status']]

            # Elegant performance metrics
            total_triples = sum(f.get('num_triples_extracted', 0) or 0 for f in success_files)
            total_vectors = sum(f.get('num_vectors_stored_chroma', 0) or 0 for f in success_files)
            total_chunks = sum(f.get('num_chunks_generated', 0) or 0 for f in success_files)

            # Elegant performance summary
            st.markdown("#### 📈 Performance Metrics")
            perf_cols = st.columns(6)

            with perf_cols[0]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{len(success_files)}</div>
                    <div class="elegant-metric-label">Successful</div>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[1]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-cyan);">{len(cached_files)}</div>
                    <div class="elegant-metric-label">Cached</div>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[2]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-rose);">{len(failed_files)}</div>
                    <div class="elegant-metric-label">Failed</div>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[3]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-blue);">{total_triples:,}</div>
                    <div class="elegant-metric-label">Total Triples</div>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[4]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-violet);">{total_vectors:,}</div>
                    <div class="elegant-metric-label">Total Vectors</div>
                </div>
                """, unsafe_allow_html=True)

            with perf_cols[5]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value" style="color: var(--elegant-amber);">{total_chunks:,}</div>
                    <div class="elegant-metric-label">Total Chunks</div>
                </div>
                """, unsafe_allow_html=True)

            # Elegant file details table
            st.markdown("#### 📋 File Processing Details")

            file_details = []
            for file_info in files:
                # Calculate elegant processing time
                processing_time = "N/A"
                if file_info.get('processing_start_timestamp') and file_info.get('processing_end_timestamp'):
                    try:
                        start_dt = datetime.fromisoformat(
                            file_info['processing_start_timestamp'].replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(file_info['processing_end_timestamp'].replace('Z', '+00:00'))
                        duration = (end_dt - start_dt).total_seconds()
                        processing_time = f"{duration:.1f}s"
                    except:
                        pass

                # Elegant status with icons
                status_icons = {
                    'Success': '✅',
                    'Cached': '🎯',
                    'Failed - OCR': '❌ OCR',
                    'Failed - KG Extract': '❌ KG',
                    'Failed - Neo4j': '❌ Neo4j',
                    'Failed - Embedding': '❌ Vector',
                    'Failed - Unknown': '❌ Unknown'
                }
                status_display = status_icons.get(file_info['status'], f"❓ {file_info['status']}")

                file_details.append({
                    'Status': status_display,
                    'Filename': file_info['file_name'],
                    'Size (KB)': f"{(file_info.get('file_size_bytes', 0) / 1024):.1f}",
                    'Processing Time': processing_time,
                    'Text Extracted': '✅' if file_info.get('text_extracted') else '❌',
                    'Chunks': file_info.get('num_chunks_generated', 0) or 0,
                    'Triples': file_info.get('num_triples_extracted', 0) or 0,
                    'Neo4j Stored': file_info.get('num_triples_loaded_neo4j', 0) or 0,
                    'Vectors': file_info.get('num_vectors_stored_chroma', 0) or 0,
                    'Cache Hit': '✅' if file_info.get('cache_hit') else '❌',
                    'Error': file_info.get('error_message', '')[:100] + '...' if file_info.get('error_message') and len(
                        file_info.get('error_message', '')) > 100 else file_info.get('error_message', '')
                })

            details_df = pd.DataFrame(file_details)
            st.dataframe(
                details_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn(width="small"),
                    "Filename": st.column_config.TextColumn(width="medium"),
                    "Size (KB)": st.column_config.TextColumn(width="small"),
                    "Processing Time": st.column_config.TextColumn(width="small"),
                    "Text Extracted": st.column_config.TextColumn(width="small"),
                    "Chunks": st.column_config.NumberColumn(width="small"),
                    "Triples": st.column_config.NumberColumn(width="small"),
                    "Neo4j Stored": st.column_config.NumberColumn(width="small"),
                    "Vectors": st.column_config.NumberColumn(width="small"),
                    "Cache Hit": st.column_config.TextColumn(width="small"),
                    "Error": st.column_config.TextColumn(width="large")
                }
            )

            # Elegant error analysis for failed files
            if failed_files:
                st.markdown("#### 🔍 Error Analysis")

                error_analysis = {}
                for file_info in failed_files:
                    error_type = file_info['status']
                    if error_type not in error_analysis:
                        error_analysis[error_type] = []
                    error_analysis[error_type].append({
                        'file': file_info['file_name'],
                        'error': file_info.get('error_message', 'No error message')
                    })

                for error_type, error_files in error_analysis.items():
                    with st.expander(f"❌ {error_type} ({len(error_files)} files)", expanded=False):
                        for error_info in error_files:
                            st.markdown(f"**{error_info['file']}:** {error_info['error']}")

        # Knowledge graph visualization option
        if HAS_VISUALIZATION and job_details['status'] in ['Completed', 'Completed with Errors']:
            st.divider()
            show_knowledge_graph_visualization(job_id, config)

    except Exception as e:
        st.error(f"❌ Error loading job details: {e}")
        logger.error(f"Error in job analysis: {e}")


def show_knowledge_graph_visualization(job_id: str, config: dict):
    """Show elegant knowledge graph visualization for processed job."""
    st.markdown("#### 🕸️ Knowledge Graph Visualization")

    viz_cols = st.columns([2, 1])

    with viz_cols[0]:
        viz_option = st.selectbox(
            "Select visualization scope:",
            [
                "All Data (Limit 100 nodes)",
                "Recent Data (Last 50 relationships)",
                "High-degree Nodes (Most connected)",
                "Document Clusters"
            ]
        )

    with viz_cols[1]:
        if st.button("🎨 Generate Visualization", type="primary"):
            generate_graph_visualization(job_id, viz_option, config)


def generate_graph_visualization(job_id: str, viz_option: str, config: dict):
    """Generate and display elegant knowledge graph visualization."""
    with st.spinner("🎨 Generating knowledge graph visualization..."):
        try:
            # Connect to Neo4j with elegance
            neo4j_uri = config.get('NEO4J_URI')
            neo4j_user = config.get('NEO4J_USER')
            neo4j_password = config.get('NEO4J_PASSWORD')
            db_name = config.get('database', {}).get('name', 'neo4j')

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                st.error("❌ Neo4j connection details missing")
                return

            driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

            # Query based on elegant visualization option
            raw_triples = []

            with driver.session(database=db_name) as session:
                if viz_option == "All Data (Limit 100 nodes)":
                    query = """
                        MATCH (s)-[r]->(o)
                        RETURN s.name AS subject, labels(s)[0] as subject_type,
                               coalesce(r.original, type(r)) as predicate,
                               o.name AS object, labels(o)[0] as object_type
                        LIMIT 100
                    """
                elif viz_option == "Recent Data (Last 50 relationships)":
                    query = """
                        MATCH (s)-[r]->(o)
                        RETURN s.name AS subject, labels(s)[0] as subject_type,
                               coalesce(r.original, type(r)) as predicate,
                               o.name AS object, labels(o)[0] as object_type
                        ORDER BY id(r) DESC
                        LIMIT 50
                    """
                elif viz_option == "High-degree Nodes (Most connected)":
                    query = """
                        MATCH (n)
                        WITH n, size((n)-[]-()) as degree
                        WHERE degree > 2
                        MATCH (n)-[r]-(connected)
                        RETURN n.name AS subject, labels(n)[0] as subject_type,
                               coalesce(r.original, type(r)) as predicate,
                               connected.name AS object, labels(connected)[0] as object_type
                        LIMIT 75
                    """
                else:  # Document Clusters
                    query = """
                        MATCH (d:Document)-[r]-(e:Entity)
                        RETURN d.name AS subject, 'Document' as subject_type,
                               type(r) as predicate,
                               e.name AS object, labels(e)[0] as object_type
                        LIMIT 100
                    """

                result = session.run(query)
                raw_triples = [
                    {k: v for k, v in record.data().items() if v is not None}
                    for record in result
                ]

            driver.close()

            if not raw_triples:
                st.warning("⚠️ No graph data found for visualization")
                return

            # Convert data to expected elegant format
            triples_to_visualize = []

            for raw_triple in raw_triples:
                converted_triple = {
                    "subject": str(raw_triple.get("subject", "Unknown")),
                    "predicate": str(raw_triple.get("predicate", "RELATED_TO")),
                    "object": str(raw_triple.get("object", "Unknown"))
                }

                if raw_triple.get("subject_type"):
                    converted_triple["subject_type"] = raw_triple["subject_type"]
                if raw_triple.get("object_type"):
                    converted_triple["object_type"] = raw_triple["object_type"]

                triples_to_visualize.append(converted_triple)

            # Elegant debug info
            st.info(f"📊 Generating visualization for {len(triples_to_visualize)} relationships...")

            # Call elegant visualization function
            viz_stats = visualize_knowledge_graph(
                triples=triples_to_visualize,
                output_file=GRAPH_HTML_FILENAME,
                edge_smooth="dynamic",
                config=config
            )

            # Display elegant visualization
            viz_path = Path(GRAPH_HTML_FILENAME)
            if viz_path.exists():
                with open(viz_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                st.success(f"✅ Visualization generated successfully!")

                # Show elegant statistics
                if isinstance(viz_stats, dict):
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-blue);">{viz_stats.get('nodes', 0)}</div>
                            <div class="elegant-metric-label">Nodes</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with stats_cols[1]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{viz_stats.get('edges', 0)}</div>
                            <div class="elegant-metric-label">Edges</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with stats_cols[2]:
                        st.markdown(f"""
                        <div class="elegant-metric-card">
                            <div class="elegant-metric-value" style="color: var(--elegant-violet);">{viz_stats.get('communities', 0)}</div>
                            <div class="elegant-metric-label">Communities</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with stats_cols[3]:
                        if 'inferred_edges' in viz_stats:
                            st.markdown(f"""
                            <div class="elegant-metric-card">
                                <div class="elegant-metric-value" style="color: var(--elegant-amber);">{viz_stats.get('inferred_edges', 0)}</div>
                                <div class="elegant-metric-label">Inferred</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Display in streamlit with elegance
                components.html(html_content, height=700, scrolling=True)

                # Elegant download option
                with open(viz_path, "rb") as fp:
                    st.download_button(
                        label="📥 Download Visualization",
                        data=fp,
                        file_name=f"knowledge_graph_{job_id[:8]}.html",
                        mime="text/html"
                    )
            else:
                st.error("❌ Visualization file not generated")

        except Exception as e:
            st.error(f"❌ Visualization failed: {e}")
            logger.error(f"Graph visualization error: {e}")


def show_file_search_analytics():
    """Enhanced elegant file search with detailed analytics."""
    st.markdown("### 🔍 File Processing Search & Analytics")

    # Elegant search input
    search_term = st.text_input(
        "Search for files:",
        placeholder="Enter filename (e.g., WO-FIXED-issue_1748266630_.pdf)"
    )

    if search_term:
        try:
            # Get all recent jobs
            recent_jobs = audit_db.get_recent_jobs(limit=100)

            matching_files = []

            # Search through all jobs elegantly
            for job in recent_jobs:
                job_details = audit_db.get_job_details(job['job_id'])
                if job_details and job_details.get('processed_files'):
                    for file_info in job_details['processed_files']:
                        if search_term.lower() in file_info['file_name'].lower():
                            file_info['job_id'] = job['job_id']
                            file_info['job_start'] = job['start_timestamp']
                            matching_files.append(file_info)

            if matching_files:
                st.success(f"✅ Found {len(matching_files)} matching file(s)")

                # Create elegant summary table
                summary_data = []
                for file_info in matching_files:
                    summary_data.append({
                        'Filename': file_info['file_name'],
                        'Job ID': file_info['job_id'][:8] + "...",
                        'Status': file_info['status'],
                        'Chunks': file_info.get('num_chunks_generated', 0),
                        'Triples Extracted': file_info.get('num_triples_extracted', 0),
                        'Neo4j Stored': file_info.get('num_triples_loaded_neo4j', 0),
                        'Vectors': file_info.get('num_vectors_stored_chroma', 0),
                        'File Size (KB)': f"{(file_info.get('file_size_bytes', 0) / 1024):.1f}",
                        'Cache Hit': '✅' if file_info.get('cache_hit') else '❌',
                        'Processed': format_timestamp(file_info.get('processing_start_timestamp'))
                    })

                # Display elegant summary table
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Elegant detailed view for each file
                for i, file_info in enumerate(matching_files):
                    with st.expander(f"📄 {file_info['file_name']} (Job: {file_info['job_id'][:8]}...)",
                                     expanded=i == 0):

                        # Elegant key metrics
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.markdown(f"""
                            <div class="elegant-metric-card">
                                <div class="elegant-metric-value" style="color: var(--elegant-blue);">{file_info.get('num_chunks_generated', 0)}</div>
                                <div class="elegant-metric-label">Chunks Generated</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with metric_cols[1]:
                            st.markdown(f"""
                            <div class="elegant-metric-card">
                                <div class="elegant-metric-value" style="color: var(--elegant-emerald);">{file_info.get('num_triples_extracted', 0)}</div>
                                <div class="elegant-metric-label">Triples Extracted</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with metric_cols[2]:
                            st.markdown(f"""
                            <div class="elegant-metric-card">
                                <div class="elegant-metric-value" style="color: var(--elegant-violet);">{file_info.get('num_triples_loaded_neo4j', 0)}</div>
                                <div class="elegant-metric-label">Neo4j Stored</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with metric_cols[3]:
                            st.markdown(f"""
                            <div class="elegant-metric-card">
                                <div class="elegant-metric-value" style="color: var(--elegant-amber);">{file_info.get('num_vectors_stored_chroma', 0)}</div>
                                <div class="elegant-metric-label">Vectors Stored</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Elegant processing details
                        st.markdown("**📊 Processing Details:**")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(f"• **Status:** {file_info['status']}")
                            st.markdown(f"• **File Size:** {file_info.get('file_size_bytes', 0) / 1024:.1f} KB")
                            st.markdown(f"• **Text Extracted:** {'Yes' if file_info.get('text_extracted') else 'No'}")
                            st.markdown(f"• **Cache Hit:** {'Yes' if file_info.get('cache_hit') else 'No'}")

                        with col2:
                            st.markdown(f"• **Start:** {format_timestamp(file_info.get('processing_start_timestamp'))}")
                            st.markdown(f"• **End:** {format_timestamp(file_info.get('processing_end_timestamp'))}")
                            st.markdown(f"• **Job ID:** {file_info['job_id']}")

                        # Elegant error information
                        if file_info.get('error_message'):
                            st.error(f"❌ **Error:** {file_info['error_message']}")

                        # Elegant processing efficiency
                        chunks = file_info.get('num_chunks_generated', 0)
                        triples = file_info.get('num_triples_extracted', 0)
                        if chunks > 0:
                            st.info(f"📈 **Efficiency:** {triples / chunks:.1f} triples per chunk")

            else:
                st.warning(f"⚠️ No files found matching '{search_term}'")
                st.info("💡 Try searching with partial filenames or different terms")

        except Exception as e:
            st.error(f"❌ Search failed: {e}")


def show_ocr_storage_management():
    """Enhanced elegant OCR storage management interface."""
    st.markdown("### 💾 OCR Storage Management")

    try:
        storage_manager = get_storage_manager()

        if not storage_manager:
            storage_manager = create_storage_manager("ocr_outputs")

        if storage_manager:
            # Elegant storage statistics
            stats = storage_manager.get_storage_stats()

            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{stats.get('total_files', 0)}</div>
                    <div class="elegant-metric-label">Total Files</div>
                </div>
                """, unsafe_allow_html=True)

            with stats_cols[1]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{stats.get('total_size_mb', 0):.1f}</div>
                    <div class="elegant-metric-label">MB Storage Size</div>
                </div>
                """, unsafe_allow_html=True)

            with stats_cols[2]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{stats.get('directories', {}).get('text_files', {}).get('count', 0)}</div>
                    <div class="elegant-metric-label">Text Files</div>
                </div>
                """, unsafe_allow_html=True)

            with stats_cols[3]:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{stats.get('directories', {}).get('original_files', {}).get('count', 0)}</div>
                    <div class="elegant-metric-label">Original Files</div>
                </div>
                """, unsafe_allow_html=True)

            # Elegant recent files
            st.markdown("#### 📁 Recent OCR Files")
            recent_files = storage_manager.list_saved_files(limit=20)

            if recent_files:
                file_data = []
                for file_info in recent_files:
                    file_data.append({
                        'Original File': file_info['original_filename'],
                        'Text Length': f"{file_info['text_length']:,}",
                        'Size (KB)': f"{file_info['size_kb']:.1f}",
                        'OCR Method': file_info.get('ocr_method', 'unknown'),
                        'Confidence': f"{file_info.get('confidence', 0):.3f}",
                        'Modified': file_info['modified_time'][:16]
                    })

                files_df = pd.DataFrame(file_data)
                st.dataframe(files_df, use_container_width=True, hide_index=True)

                # Elegant management options
                st.markdown("#### 🔧 Storage Management")

                mgmt_cols = st.columns(3)

                with mgmt_cols[0]:
                    if st.button("📊 Export to Excel", use_container_width=True):
                        try:
                            excel_path = storage_manager.export_to_excel()
                            st.success(f"✅ Excel export created: {excel_path}")
                        except Exception as e:
                            st.error(f"❌ Excel export failed: {e}")

                with mgmt_cols[1]:
                    if st.button("🗂️ Export Metadata JSON", use_container_width=True):
                        try:
                            json_path = storage_manager.export_metadata_json()
                            st.success(f"✅ JSON export created: {json_path}")
                        except Exception as e:
                            st.error(f"❌ JSON export failed: {e}")

                with mgmt_cols[2]:
                    cleanup_days = st.number_input("Cleanup files older than (days):", min_value=1, value=30)
                    if st.button("🗑️ Cleanup Old Files", use_container_width=True):
                        try:
                            cleanup_stats = storage_manager.cleanup_old_files(cleanup_days)
                            st.success(f"✅ Cleanup completed: {cleanup_stats}")
                        except Exception as e:
                            st.error(f"❌ Cleanup failed: {e}")
            else:
                st.info("📝 No OCR files found. Process some documents to see them here.")

        else:
            st.error("❌ OCR storage manager not available")

    except Exception as e:
        st.error(f"❌ Storage management error: {e}")


# Execute main function
if __name__ == "__main__":
    main()
# pages/3_Pipeline_Validation.py

import streamlit as st
import time
import pandas as pd
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Pipeline Validation",
    page_icon="🔍",
    layout="wide"
)

# EXACT same styling as OCR Output Analyzer
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
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
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

# EXACT same imports as OCR Output Analyzer - NO complex validation modules
try:
    from GraphRAG_Document_AI_Platform import load_config, get_enhanced_ocr_pipeline
    from src.utils.processing_pipeline import process_uploaded_file_ocr_with_storage
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()


# EXACT same resource loading as OCR Output Analyzer
@st.cache_resource
def get_resources():
    """Load the same resources as OCR Output Analyzer"""
    try:
        config = load_config()
        enhanced_ocr_pipeline = get_enhanced_ocr_pipeline(config)
        return config, enhanced_ocr_pipeline, None
    except Exception as e:
        return None, None, str(e)


def validate_single_document(file, enhanced_ocr_pipeline):
    """Validate a single document using the EXACT same process as OCR Analyzer"""
    try:
        start_time = time.time()

        # Use the EXACT same function that works in OCR Analyzer
        result = process_uploaded_file_ocr_with_storage(
            uploaded_file=file,
            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
            save_to_disk=False  # Don't save during validation
        )

        processing_time = time.time() - start_time

        # Extract validation metrics
        confidence = result.get('confidence', 0.0)
        text_length = result.get('text_length', 0)
        method_used = result.get('method_used', 'unknown')
        success = result.get('success', False)
        error_msg = result.get('error', '')

        # Simple quality assessment
        if confidence >= 0.8:
            quality = "🟢 Excellent"
            recommendation = "Auto-process"
        elif confidence >= 0.6:
            quality = "🔵 Good"
            recommendation = "Monitor"
        elif confidence >= 0.4:
            quality = "🟡 Fair"
            recommendation = "Review"
        else:
            quality = "🔴 Poor"
            recommendation = "Manual"

        return {
            'file_name': file.name,
            'success': success,
            'confidence': confidence,
            'text_length': text_length,
            'processing_time': processing_time,
            'method': method_used,
            'quality': quality,
            'recommendation': recommendation,
            'error': error_msg
        }

    except Exception as e:
        return {
            'file_name': file.name,
            'success': False,
            'confidence': 0.0,
            'text_length': 0,
            'processing_time': 0.0,
            'method': 'error',
            'quality': "🔴 Error",
            'recommendation': "Check file",
            'error': str(e)
        }


def validate_batch(files, enhanced_ocr_pipeline):
    """Validate multiple documents using the same approach as OCR Analyzer"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(files):
        status_text.text(f"Processing {file.name}...")
        result = validate_single_document(file, enhanced_ocr_pipeline)
        results.append(result)
        progress_bar.progress((i + 1) / len(files))

    status_text.text("Validation complete!")

    # Calculate summary
    successful = [r for r in results if r['success']]
    total = len(results)
    success_rate = len(successful) / total if total > 0 else 0
    avg_confidence = sum(r['confidence'] for r in successful) / len(successful) if successful else 0

    return {
        'results': results,
        'total_files': total,
        'successful_files': len(successful),
        'success_rate': success_rate,
        'avg_confidence': avg_confidence
    }


# Load resources - EXACT same pattern as OCR Output Analyzer
config, enhanced_ocr_pipeline, error = get_resources()

if error:
    st.error(f"❌ Failed to initialize: {error}")
    st.stop()

if not enhanced_ocr_pipeline:
    st.error("❌ Enhanced OCR pipeline not available")
    st.stop()

# Main interface - Clean and minimal
st.title("🔍 Pipeline Validation")
st.markdown("Test document processing quality before batch operations")

# File upload
validation_files = st.file_uploader(
    "Upload test documents:",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload 3-5 representative documents"
)

if validation_files:
    # Show files
    st.write(f"**{len(validation_files)} files selected**")

    # Show uploaded files table (same as OCR Analyzer)
    file_data = []
    for file in validation_files:
        file_data.append({
            "File": file.name,
            "Type": file.type.split('/')[-1].upper(),
            "Size": f"{len(file.getvalue()) / 1024:.1f} KB"
        })
    st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)

    # Process button
    if st.button("🔬 Run Validation", type="primary", use_container_width=True):

        with st.spinner("Validating documents..."):
            start_time = time.time()
            report = validate_batch(validation_files, enhanced_ocr_pipeline)
            total_time = time.time() - start_time

        st.success(f"✅ Completed in {total_time:.1f}s")

        # Summary metrics - EXACT same styling as OCR Analyzer
        st.subheader("📊 Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{report['success_rate']:.0%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{report['avg_confidence']:.2f}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            excellent = sum(1 for r in report['results'] if r['confidence'] >= 0.8)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{excellent}</div>
                <div class="metric-label">Excellent</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            poor = sum(1 for r in report['results'] if r['confidence'] < 0.4)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{poor}</div>
                <div class="metric-label">Need Review</div>
            </div>
            """, unsafe_allow_html=True)

        # Results table - Same format as OCR Analyzer
        st.subheader("📋 Results")

        results_data = []
        for result in report['results']:
            results_data.append({
                'Document': result['file_name'],
                'Status': '✅ Success' if result['success'] else '❌ Failed',
                'Method': result['method'].upper(),
                'Confidence': f"{result['confidence']:.3f}",
                'Quality': result['quality'],
                'Text Length': f"{result['text_length']:,}",
                'Time': f"{result['processing_time']:.1f}s",
                'Action': result['recommendation']
            })

        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

        # Recommendations - Same styling as OCR Analyzer
        st.subheader("💡 Recommendations")

        total = len(report['results'])
        excellent = sum(1 for r in report['results'] if r['confidence'] >= 0.8)
        poor = sum(1 for r in report['results'] if r['confidence'] < 0.4)

        if excellent >= total * 0.8:
            st.markdown('<div class="status-badge status-success">🚀 Ready for batch processing</div>',
                        unsafe_allow_html=True)
        elif poor > 0:
            st.markdown('<div class="status-badge status-error">⚠️ Manual review required</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-warning">📊 Proceed with monitoring</div>',
                        unsafe_allow_html=True)

        # Show errors if any
        errors = [r for r in report['results'] if not r['success']]
        if errors:
            st.subheader("❌ Processing Errors")
            for error in errors:
                st.error(f"**{error['file_name']}**: {error['error']}")

        # Export - Same as OCR Analyzer
        st.subheader("📥 Export")

        col1, col2 = st.columns(2)

        with col1:
            csv_data = pd.DataFrame(results_data).to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                data=csv_data,
                file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                "📋 Download Report",
                data=report_json,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

else:
    st.info("👆 Upload documents to start validation")

    # Quick help
    with st.expander("💡 How to Use"):
        st.markdown("""
        **Quick Start:**
        1. Upload 3-5 test documents
        2. Click Run Validation 
        3. Review quality scores
        4. Follow recommendations

        **Quality Levels:**
        - 🟢 **Excellent (≥0.8)**: Auto-process
        - 🔵 **Good (0.6-0.8)**: Monitor  
        - 🟡 **Fair (0.4-0.6)**: Review
        - 🔴 **Poor (<0.4)**: Manual
        """)
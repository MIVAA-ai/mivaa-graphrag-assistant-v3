# pages/3_Data_Extraction_Validation.py

import streamlit as st
import time
import logging
import pandas as pd
from typing import List, Any, Dict
st.set_page_config(page_title="Data Extraction Validation")

# Import your existing components
try:
    from GraphRAG_Document_AI_Platform import load_config, get_mistral_client
    from src.utils.pipeline_validation import create_validator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Setup logging
logger = logging.getLogger(__name__)

st.title("🔍 Pipeline Validation & Quality Check")
st.write("Test your document processing pipeline accuracy before running full ingestion jobs.")


# Load configuration and resources
@st.cache_resource
def get_validation_resources():
    """Load and cache validation resources."""
    try:
        config = load_config()
        mistral_client = get_mistral_client(config.get('MISTRAL_API_KEY'))
        validator = create_validator(config, mistral_client)
        return config, validator, None
    except Exception as e:
        return None, None, str(e)


config, validator, error = get_validation_resources()

if error:
    st.error(f"Failed to initialize validation resources: {error}")
    st.stop()

if not validator:
    st.error("Validator not initialized. Check your configuration.")
    st.stop()

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Document Quality Assessment")

with col2:
    st.info("💡 **Tip**: Test 3-5 representative documents to assess batch quality")

# File uploader
validation_files = st.file_uploader(
    "Upload documents to test pipeline accuracy:",
    type=["pdf", "png", "jpg", "jpeg", "txt"],
    accept_multiple_files=True,
    help="Upload representative documents from your batch to test processing quality"
)

# Show file info if files uploaded
if validation_files:
    with st.expander(f"📁 Uploaded Files ({len(validation_files)})", expanded=False):
        for i, file in enumerate(validation_files):
            st.write(f"{i + 1}. **{file.name}** ({file.type}, {len(file.getvalue()):,} bytes)")

# Validation controls
if validation_files:
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_validation = st.button("🧪 Run Quick Validation", type="primary", use_container_width=True)

    with col2:
        if len(validation_files) > 5:
            st.warning(
                f"⚠️ {len(validation_files)} files selected. Consider testing with 3-5 representative files first.")

    # Run validation
    if run_validation:
        st.header("📊 Validation Results")

        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Track timing
            start_time = time.time()

            # Process files
            try:
                status_text.text("Starting validation...")

                # Run batch validation
                with st.spinner("Validating documents..."):
                    report = validator.validate_batch_quick(validation_files)

                processing_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.text(f"Validation completed in {processing_time:.1f}s")

                # Display results
                st.success("Validation completed successfully!")

                # Summary metrics
                st.subheader("📈 Summary Metrics")
                summary = report['summary']

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    ocr_rate = summary['ocr_success_rate']
                    st.metric(
                        "OCR Success Rate",
                        f"{ocr_rate:.1%}",
                        delta=f"{summary['total_files']} files tested"
                    )

                with metric_col2:
                    confidence = summary['average_confidence_score']
                    st.metric(
                        "Avg Confidence",
                        f"{confidence:.2f}",
                        delta="Higher is better"
                    )

                with metric_col3:
                    financial = summary['financial_documents']
                    st.metric(
                        "Business Documents",
                        f"{financial}/{summary['total_files']}",
                        delta="Contains financial data"
                    )

                with metric_col4:
                    avg_time = summary['total_validation_time'] / summary['total_files']
                    st.metric(
                        "Avg Processing Time",
                        f"{avg_time:.1f}s",
                        delta="Per document"
                    )

                # Results table
                st.subheader("📄 File-by-File Results")

                # Prepare data for display
                results_data = []
                for result in report['results']:
                    results_data.append({
                        'File Name': result['file_name'],
                        'Type': result['file_type'],
                        'Size': f"{result['file_size']:,} bytes",
                        'OCR Success': '✅' if result['mistral_ocr_success'] else '❌',
                        'Text Length': f"{result['mistral_text_length']:,}" if result[
                                                                                   'mistral_text_length'] > 0 else '0',
                        'Confidence': f"{result['confidence_score']:.2f}",
                        'Quality Score': f"{result['text_quality_score']:.2f}",
                        'Chunks': str(result['num_chunks']),
                        'Recommendation': result['processing_recommendation'],
                        'Has Tables': '📊' if result['contains_tables'] else '',
                        'Has Handwriting': '✍️' if result['contains_handwriting_indicators'] else '',
                        'Processing Time': f"{result['mistral_extraction_time']:.1f}s"
                    })

                df = pd.DataFrame(results_data)


                # Color-code the confidence column
                def style_confidence(df):
                    """Apply color styling to the confidence column."""
                    # Create a style dataframe with empty strings
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)

                    # Apply styling only to the Confidence column
                    if 'Confidence' in df.columns:
                        confidence_col = df['Confidence']
                        for idx in df.index:
                            try:
                                score = float(confidence_col[idx])
                                if score >= 0.8:
                                    styles.loc[idx, 'Confidence'] = 'background-color: #d4edda'  # Light green
                                elif score >= 0.6:
                                    styles.loc[idx, 'Confidence'] = 'background-color: #fff3cd'  # Light yellow
                                elif score >= 0.4:
                                    styles.loc[idx, 'Confidence'] = 'background-color: #f8d7da'  # Light red
                                else:
                                    styles.loc[idx, 'Confidence'] = 'background-color: #f5c6cb'  # Darker red
                            except (ValueError, TypeError):
                                continue

                    return styles


                st.dataframe(
                    df.style.apply(style_confidence, axis=None),
                    use_container_width=True,
                    height=400
                )

                # Recommendations section
                st.subheader("💡 Processing Recommendations")

                recommendations = report['recommendations']
                for rec in recommendations:
                    if rec.startswith('🚀'):
                        st.success(rec)
                    elif rec.startswith('✅'):
                        st.success(rec)
                    elif rec.startswith('⚠️'):
                        st.warning(rec)
                    elif rec.startswith('❌'):
                        st.error(rec)
                    else:
                        st.info(rec)

                # Detailed analysis
                with st.expander("🔍 Detailed Analysis", expanded=False):

                    # File categorization
                    excellent_files = [r for r in report['results'] if r['confidence_score'] >= 0.8]
                    good_files = [r for r in report['results'] if 0.6 <= r['confidence_score'] < 0.8]
                    fair_files = [r for r in report['results'] if 0.4 <= r['confidence_score'] < 0.6]
                    poor_files = [r for r in report['results'] if r['confidence_score'] < 0.4]

                    cat_col1, cat_col2, cat_col3, cat_col4 = st.columns(4)

                    with cat_col1:
                        st.metric("Excellent (≥0.8)", len(excellent_files), "Auto-process")
                    with cat_col2:
                        st.metric("Good (0.6-0.8)", len(good_files), "Monitor")
                    with cat_col3:
                        st.metric("Fair (0.4-0.6)", len(fair_files), "Review")
                    with cat_col4:
                        st.metric("Poor (<0.4)", len(poor_files), "Manual")

                    # Show problematic files
                    if poor_files:
                        st.write("**Files needing attention:**")
                        for file in poor_files:
                            error_msg = file.get('ocr_error_message', 'Quality issues detected')
                            st.write(f"• **{file['file_name']}**: {error_msg}")

                # Export results
                st.subheader("📥 Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Download detailed report
                    import json

                    report_json = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        "📊 Download Detailed Report (JSON)",
                        data=report_json,
                        file_name=f"validation_report_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    # Download summary CSV
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📋 Download Summary (CSV)",
                        data=csv_data,
                        file_name=f"validation_summary_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Validation failed: {str(e)}")
                logger.error(f"Validation error: {e}", exc_info=True)

else:
    # Show help when no files uploaded
    st.info("👆 Upload some documents to start validation testing")

    # Example section
    st.subheader("📖 How to Use This Tool")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Step 1: Upload Test Documents**")
        st.write("• Select 3-5 representative documents")
        st.write("• Choose different types (scanned, digital, handwritten)")
        st.write("• Mix of good and potentially problematic files")

        st.write("**Step 3: Review Results**")
        st.write("• Check confidence scores for each file")
        st.write("• Identify files needing manual review")
        st.write("• Note processing time estimates")

    with col2:
        st.write("**Step 2: Run Validation**")
        st.write("• Click 'Run Quick Validation'")
        st.write("• Wait for OCR and quality analysis")
        st.write("• Review success rates and recommendations")

        st.write("**Step 4: Optimize Pipeline**")
        st.write("• Adjust settings based on results")
        st.write("• Set up manual review workflows")
        st.write("• Process with confidence")

# Help section
with st.expander("ℹ️ Understanding Results"):
    st.write("""
    **Confidence Score Meaning:**
    - **0.8+ (Excellent)**: Process automatically, high success expected
    - **0.6-0.8 (Good)**: Process with monitoring, mostly successful
    - **0.4-0.6 (Fair)**: Manual review recommended before processing
    - **0.0-0.4 (Poor)**: Manual processing needed, automation likely to fail

    **Quality Indicators:**
    - **📊 Has Tables**: Document contains structured tabular data
    - **✍️ Has Handwriting**: Potential handwritten content detected
    - **OCR Success**: Whether text extraction worked
    - **Text Length**: Amount of text successfully extracted

    **When to Use This Tool:**
    - Before processing new document batches
    - When document quality/sources change
    - If extraction results seem poor
    - For regular quality audits

    **Cost Optimization:**
    - Identify documents that will fail before spending API credits
    - Route low-quality documents to manual review
    - Optimize batch processing based on confidence scores
    """)

# Footer
st.markdown("---")
st.caption("💡 Pro tip: Run validation on a sample before processing large batches to save time and API costs.")
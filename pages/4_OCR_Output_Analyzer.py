# pages/4_OCR_Output_Analyzer.py

import streamlit as st
import pandas as pd
from pathlib import Path
import json
import zipfile
import io
from datetime import datetime
st.set_page_config(page_title="Processed Files Manager")

# Import your components
try:
    from GraphRAG_Document_AI_Platform import load_config, get_mistral_client, get_enhanced_ocr_pipeline
    from src.utils.ocr_storage import create_storage_manager
    from src.utils.processing_pipeline import process_uploaded_file_ocr_with_storage, process_batch_with_enhanced_storage
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.title("💾 OCR Output Storage & Management")
st.write("Save, manage, and export your OCR extraction results to local storage.")


# Initialize storage manager
@st.cache_resource
def get_storage_manager():
    return create_storage_manager("ocr_outputs")


storage_manager = get_storage_manager()

# Load config and initialize components
config = load_config()
mistral_client = get_mistral_client(config.get('MISTRAL_API_KEY'))

# ✅ FIX: Get the enhanced OCR pipeline
enhanced_ocr_pipeline = get_enhanced_ocr_pipeline(config)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Extract & Save", "📁 Browse Saved Files", "📊 Export Data", "⚙️ Settings"])

with tab1:
    st.header("Extract Text & Save to Disk")

    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents to extract text and save locally:",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Files will be processed with OCR and saved to your local drive"
        )

    with col2:
        st.info("**Storage Location:**\n`./ocr_outputs/`")

        if st.button("📂 Open Storage Folder"):
            storage_path = Path("ocr_outputs").absolute()
            st.code(f"Storage location:\n{storage_path}")

    if uploaded_files:
        col1, col2 = st.columns(2)

        with col1:
            save_to_disk = st.checkbox("💾 Save to Local Disk", value=True,
                                       help="Save OCR output and metadata to local files")

        with col2:
            extract_fields = st.checkbox("🔍 Extract Structured Fields", value=True,
                                         help="Extract invoice numbers, dates, amounts, etc.")

        if st.button("🚀 Process Files & Save", type="primary"):

            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()

            with st.spinner("Processing files..."):

                try:
                    # ✅ FIX: Now enhanced_ocr_pipeline is properly defined
                    batch_results = process_batch_with_enhanced_storage(
                        uploaded_files=uploaded_files,
                        enhanced_ocr_pipeline=enhanced_ocr_pipeline,
                        save_to_disk=save_to_disk
                    )

                    progress_bar.progress(1.0)
                    status_text.success(f"Processed {len(batch_results)} files successfully!")

                    # Display results
                    with results_container:
                        st.subheader("📋 Processing Results")

                        # Summary metrics
                        successful = sum(1 for r in batch_results if r['success'])
                        total_text_length = sum(r.get('text_length', 0) for r in batch_results)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files Processed", f"{successful}/{len(batch_results)}")
                        with col2:
                            st.metric("Total Text Extracted", f"{total_text_length:,} chars")
                        with col3:
                            if save_to_disk:
                                saved_files = sum(
                                    len(r.get('saved_files', {})) for r in batch_results if r.get('saved_files'))
                                st.metric("Files Saved", saved_files)

                        # Results table
                        results_data = []
                        for result in batch_results:
                            results_data.append({
                                'Filename': result['original_filename'],
                                'Status': 'Success' if result['success'] else 'Failed',
                                'Text Length': f"{result.get('text_length', 0):,}",
                                'Saved Files': len(result.get('saved_files', {})) if save_to_disk else 'Not saved',
                                'Error': result.get('error', result.get('save_error', ''))[:50] if not result[
                                    'success'] else ''
                            })

                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)

                        # Show file paths for successful saves
                        if save_to_disk:
                            with st.expander("📁 Saved File Locations"):
                                for result in batch_results:
                                    if result.get('saved_files'):
                                        st.write(f"**{result['original_filename']}:**")
                                        for file_type, path in result['saved_files'].items():
                                            st.code(f"{file_type}: {path}")
                                        st.write("---")

                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")

with tab2:
    st.header("📁 Browse Saved OCR Files")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Recently Saved Files")

    with col2:
        if st.button("🔄 Refresh List"):
            st.rerun()

    try:
        # Get list of saved files
        saved_files = storage_manager.list_saved_files(limit=50)

        if saved_files:
            # Display as table
            df = pd.DataFrame(saved_files)
            df['modified_time'] = pd.to_datetime(df['modified_time']).dt.strftime('%Y-%m-%d %H:%M')

            # File selection
            selected_file = st.selectbox(
                "Select a file to view:",
                options=range(len(saved_files)),
                format_func=lambda x: f"{saved_files[x]['original_filename']} ({saved_files[x]['modified_time']})"
            )

            # Display file details and content
            if selected_file is not None:
                file_info = saved_files[selected_file]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**File Information:**")
                    st.write(f"• **Original:** {file_info['original_filename']}")
                    st.write(f"• **Size:** {file_info['size_kb']} KB")
                    st.write(f"• **Modified:** {file_info['modified_time']}")
                    st.write(f"• **Text Length:** {file_info['text_length']:,} characters")

                with col2:
                    # Load and display text content
                    try:
                        with open(file_info['text_file_path'], 'r', encoding='utf-8') as f:
                            text_content = f.read()

                        # Download button
                        st.download_button(
                            "💾 Download Text File",
                            data=text_content,
                            file_name=f"{file_info['filename']}",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Could not load text file: {e}")

                # Display text content
                with st.expander("📄 View Extracted Text", expanded=False):
                    try:
                        with open(file_info['text_file_path'], 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        st.text_area("Extracted Text:", value=text_content, height=300)
                    except Exception as e:
                        st.error(f"Could not display text: {e}")

        else:
            st.info("📝 No saved OCR files found. Process some documents in the 'Extract & Save' tab first.")

    except Exception as e:
        st.error(f"Error loading saved files: {e}")

with tab3:
    st.header("📊 Export Saved Data")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 CSV Export")
        if st.button("📥 Export Summary to CSV"):
            try:
                saved_files = storage_manager.list_saved_files(limit=1000)
                if saved_files:
                    df = pd.DataFrame(saved_files)
                    csv_data = df.to_csv(index=False)

                    st.download_button(
                        "💾 Download CSV",
                        data=csv_data,
                        file_name=f"ocr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
            except Exception as e:
                st.error(f"CSV export failed: {e}")

    with col2:
        st.subheader("📊 Excel Export")
        if st.button("📥 Export All Data to Excel"):
            try:
                with st.spinner("Creating Excel file..."):
                    excel_path = storage_manager.export_to_excel()

                    # Read the created Excel file for download
                    with open(excel_path, 'rb') as f:
                        excel_data = f.read()

                    st.download_button(
                        "💾 Download Excel File",
                        data=excel_data,
                        file_name=f"ocr_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.success(f"Excel file created: {excel_path}")

            except ImportError:
                st.error("📦 Please install pandas and openpyxl for Excel export:\n```pip install pandas openpyxl```")
            except Exception as e:
                st.error(f"Excel export failed: {e}")

    # Bulk download option
    st.subheader("📦 Bulk Download")
    if st.button("📦 Create ZIP Archive of All Files"):
        try:
            with st.spinner("Creating ZIP archive..."):
                # Create ZIP file in memory
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add all text files
                    for text_file in storage_manager.text_dir.glob("*.txt"):
                        zip_file.write(text_file, f"text_files/{text_file.name}")

                    # Add all JSON files
                    for json_file in storage_manager.json_dir.glob("*.json"):
                        zip_file.write(json_file, f"metadata/{json_file.name}")

                zip_buffer.seek(0)

                st.download_button(
                    "💾 Download ZIP Archive",
                    data=zip_buffer.getvalue(),
                    file_name=f"ocr_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

                st.success("ZIP archive created successfully!")

        except Exception as e:
            st.error(f"ZIP creation failed: {e}")

with tab4:
    st.header("⚙️ Storage Settings")

    # Storage location info
    storage_path = Path("ocr_outputs").absolute()
    st.subheader("📂 Storage Location")
    st.code(str(storage_path))

    # Storage usage
    st.subheader("💽 Storage Usage")
    try:
        total_size = 0
        file_count = 0

        for file_path in storage_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", file_count)
        with col2:
            st.metric("Storage Used", f"{total_size / (1024 * 1024):.1f} MB")
        with col3:
            st.metric("Text Files", len(list(storage_manager.text_dir.glob("*.txt"))))

    except Exception as e:
        st.error(f"Could not calculate storage usage: {e}")

    # Cleanup options
    st.subheader("🧹 Cleanup")
    st.warning("⚠️ Cleanup operations cannot be undone!")

    if st.button("🗑️ Clear All Saved Files", type="secondary"):
        if st.checkbox("I understand this will delete all saved OCR files"):
            try:
                import shutil

                shutil.rmtree(storage_path)
                storage_path.mkdir(exist_ok=True)
                st.success("All files cleared successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Cleanup failed: {e}")

# Footer
st.markdown("---")
st.caption("💡 All OCR outputs are saved locally in the `ocr_outputs/` directory relative to your app.")
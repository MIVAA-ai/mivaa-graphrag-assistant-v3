# pages/2_Document_Ingestion.py

import streamlit as st
import logging
import time
import pandas as pd  # For displaying history nicely
from datetime import datetime, timezone  # For timestamp conversion
from typing import List, Dict, Any, Optional, Tuple
import streamlit.components.v1 as components
from pathlib import Path
import neo4j  # Import Neo4j driver to query data

try:
    import src.utils.audit_db_manager  # Needs access to the audit DB functions
    import src.utils.processing_pipeline  # Needs access to start the pipeline thread
    from src.utils.ocr_storage import create_storage_manager
    # Import the functions that provide cached resources from the main app script
    from GraphRAG_Document_AI_Platform import (
        load_config,
        get_mistral_client,
        init_neo4j_exporter,
        get_embedding_model,
        get_chroma_collection,
        get_requests_session,
        get_nlp_pipeline,
        get_enhanced_ocr_pipeline  # Add the enhanced OCR pipeline import
    )
    # Import the enhanced OCR compatibility functions
    from src.utils.processing_pipeline import process_batch_with_enhanced_storage
    from src.utils.processing_pipeline import process_uploaded_file_ocr_with_storage
    from enhanced_ocr_pipeline import EnhancedOCRPipeline, create_enhanced_config
except ImportError as e:
    st.error(
        f"Error importing project modules in Data Ingestion page: {e}. Ensure GraphRAG_Document_AI_Platform.py, audit_db_manager.py, and processing_pipeline.py are accessible.")
    st.stop()

# Import the visualization function (adjust path if needed)
try:
    from src.knowledge_graph.visualization import visualize_knowledge_graph
    # You might also need the function to get a sync Neo4j driver/session
    # Or adapt visualize_knowledge_graph to accept config/credentials
except ImportError as e:
    st.error(f"Failed to import visualization function: {e}")
    visualize_knowledge_graph = None  # Disable feature if import fails

# Define the output filename constant (can be same as in GraphRAG_Document_AI_Platform.py)
GRAPH_HTML_FILENAME = "graph_visualization.html"

# Logger setup
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():  # Basic config if needed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')


# --- Helper Function ---
def format_timestamp(ts_string: Optional[str]) -> str:
    """Formats ISO timestamp string for display."""
    if not ts_string:
        return "N/A"
    try:
        # Parse ISO string (handling potential Z for UTC)
        dt_obj = datetime.fromisoformat(ts_string.replace('Z', '+00:00'))
        # Convert to local timezone for display (optional)
        # dt_obj = dt_obj.astimezone(tz=None)
        return dt_obj.strftime("%Y-%m-%d %H:%M:%S %Z")  # Adjust format as needed
    except (ValueError, TypeError):
        return ts_string  # Return original if parsing fails


# --- Streamlit Page Logic ---

st.title("📄 Data Ingestion & Audit Trail")

# --- Load Config and Initialize Resources ---
# We need resources required by the processing pipeline thread
try:
    config = load_config()
    if not config or not config.get('_CONFIG_VALID'):
        st.error("App configuration is invalid. Cannot proceed.")
        st.stop()

    # Initialize resources with enhanced LLM OCR
    enhanced_ocr_pipeline = get_enhanced_ocr_pipeline(config)
    neo4j_exporter = init_neo4j_exporter(config.get('NEO4J_URI'), config.get('NEO4J_USER'),
                                         config.get('NEO4J_PASSWORD'))
    embedding_model = get_embedding_model(config.get('EMBEDDING_MODEL'))
    chroma_collection = get_chroma_collection(config.get('CHROMA_PERSIST_PATH'), config.get('COLLECTION_NAME'),
                                              config.get('EMBEDDING_MODEL'))
    requests_session = get_requests_session()
    nlp_pipeline = get_nlp_pipeline(config)

    # Check LLM OCR availability
    ocr_available = enhanced_ocr_pipeline and (
            enhanced_ocr_pipeline.mistral_client or
            hasattr(enhanced_ocr_pipeline, 'gemini_client') or
            hasattr(enhanced_ocr_pipeline, 'openai_client') or
            hasattr(enhanced_ocr_pipeline, 'anthropic_client')
    )
    if not ocr_available:
        st.warning("No LLM OCR method available. Only TXT upload will work.", icon="⚠️")

    # Initialize OCR storage manager
    try:
        ocr_storage = create_storage_manager("ocr_outputs")
        logger.info("OCR storage manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OCR storage: {e}")
        ocr_storage = None

    # Check essential resources needed for processing
    processing_possible = True
    if not neo4j_exporter:
        st.warning("Neo4j Exporter not initialized. Triples won't be stored.", icon="⚠️")
        # processing_possible = False # Decide if this is critical enough to disable processing
    if not embedding_model or not chroma_collection:
        st.warning("Embedding Model or Chroma Collection not available. Embeddings won't be stored.", icon="⚠️")
        # processing_possible = False

except Exception as e:
    logger.error(f"Error initializing resources for ingestion page: {e}", exc_info=True)
    st.error(f"Failed to initialize necessary resources: {e}")
    st.stop()

# --- File Upload Section ---
st.header("1. Upload Documents")

# Define allowed file types based on whether LLM OCR is available
file_types = ["pdf", "png", "jpg", "jpeg", "txt"]
if not ocr_available:
    st.warning("LLM OCR not available. Only TXT upload supported.", icon="ℹ️")
    file_types = ["txt"]

uploaded_files = st.file_uploader(
    "Select documents to process:",
    type=file_types,
    accept_multiple_files=True,
    key="ingestion_uploader"
)

with st.expander("Processing Options", expanded=False):
    use_cache = st.toggle(
        "Use Processing Cache",
        value=config.get('CACHE_ENABLED', True),
        key="ingestion_use_cache",
        help="If enabled, avoids re-extracting KG triples for files that haven't changed since last processing."
    )

    save_ocr_to_disk = st.toggle(
        "💾 Save OCR Output to Disk",
        value=True,
        key="save_ocr_option",
        help="Save extracted text and metadata to local storage for backup and analysis."
    )

    if save_ocr_to_disk and ocr_storage:
        st.info(f"📁 OCR outputs will be saved to: `{ocr_storage.base_dir.absolute()}`")

process_button_disabled = not uploaded_files  # Disable if no files uploaded

if st.button("🚀 Start Ingestion Job", disabled=process_button_disabled, use_container_width=True):
    if uploaded_files and processing_possible:

        # Enhanced LLM OCR Pre-processing
        if save_ocr_to_disk and enhanced_ocr_pipeline:
            with st.expander("📄 Enhanced LLM OCR Pre-Processing", expanded=True):
                st.info(f"Extracting text using Enhanced LLM OCR Pipeline for {len(uploaded_files)} files...")

                # Show LLM OCR methods being used
                available_methods = []
                if enhanced_ocr_pipeline.mistral_client:
                    available_methods.append("☁️ Mistral Pixtral")
                if hasattr(enhanced_ocr_pipeline, 'gemini_client') and enhanced_ocr_pipeline.gemini_client:
                    available_methods.append("🔥 Gemini 1.5 Flash")
                if hasattr(enhanced_ocr_pipeline, 'openai_client') and enhanced_ocr_pipeline.openai_client:
                    available_methods.append("🚀 GPT-4o Vision")
                if hasattr(enhanced_ocr_pipeline, 'anthropic_client') and enhanced_ocr_pipeline.anthropic_client:
                    available_methods.append("🤖 Claude 3.5 Sonnet")

                if available_methods:
                    st.success(f"Available LLM OCR Methods: {', '.join(available_methods)}")
                else:
                    st.warning("⚠️ No LLM OCR methods available")

                progress_bar = st.progress(0)
                ocr_results = []

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text = st.empty()
                    status_text.info(f"Processing {uploaded_file.name}...")

                    try:
                        # Use the enhanced LLM OCR pipeline compatibility function
                        result = process_uploaded_file_ocr_with_storage(
                            uploaded_file=uploaded_file,
                            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
                            save_to_disk=True
                        )

                        if result['success']:
                            ocr_results.append({
                                'filename': uploaded_file.name,
                                'status': f'{result.get("method_used", "unknown").upper()}',
                                'confidence': f"{result.get('confidence', 0.0):.3f}",
                                'text_length': result.get('text_length', 0),
                                'processing_time': f"{result.get('processing_time', 0.0):.2f}s",
                                'saved_files': len(result.get('saved_files', {}))
                            })
                            status_text.success(
                                f"{uploaded_file.name}: {result.get('text_length', 0)} chars via {result.get('method_used', 'unknown')}")
                        else:
                            ocr_results.append({
                                'filename': uploaded_file.name,
                                'status': 'Failed',
                                'confidence': '0.000',
                                'text_length': 0,
                                'processing_time': '0.00s',
                                'saved_files': 0
                            })
                            status_text.error(f"{uploaded_file.name}: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        ocr_results.append({
                            'filename': uploaded_file.name,
                            'status': f'Error: {str(e)[:50]}',
                            'confidence': '0.000',
                            'text_length': 0,
                            'processing_time': '0.00s',
                            'saved_files': 0
                        })
                        status_text.error(f"{uploaded_file.name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(uploaded_files))

                # Show LLM OCR results summary
                if ocr_results:
                    st.subheader("📋 Enhanced LLM OCR Extraction Summary")
                    ocr_df = pd.DataFrame(ocr_results)
                    st.dataframe(ocr_df, use_container_width=True)

                    # Show successful extractions with enhanced metrics
                    successful = sum(1 for r in ocr_results if r['status'] != 'Failed')
                    total_chars = sum(r['text_length'] for r in ocr_results)
                    total_saved = sum(r['saved_files'] for r in ocr_results)
                    avg_confidence = sum(
                        float(r['confidence']) for r in ocr_results if r['confidence'] != '0.000') / max(successful, 1)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Successful", f"{successful}/{len(uploaded_files)}")
                    with col2:
                        st.metric("Total Text", f"{total_chars:,} chars")
                    with col3:
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    with col4:
                        st.metric("Files Saved", total_saved)

        # Continue with normal ingestion using Enhanced LLM OCR Pipeline
        st.info(f"Starting ingestion job for {len(uploaded_files)} file(s) in the background...")
        # Call the function that starts the background thread
        job_id = src.utils.processing_pipeline.start_ingestion_job_async(
            uploaded_files=uploaded_files,
            config=config,
            use_cache=use_cache,
            # Use enhanced LLM OCR pipeline
            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
            neo4j_exporter=neo4j_exporter,
            embedding_model_resource=embedding_model,
            chroma_collection_resource=chroma_collection,
            requests_session_resource=requests_session,
            nlp_pipeline_resource=nlp_pipeline
        )

        if job_id:
            st.success(
                f"Ingestion Job '{job_id}' started successfully in the background. See history below for progress.")
            # Store the running job ID to potentially show live status (optional future feature)
            st.session_state['running_ingestion_job_id'] = job_id
            # Give the thread a moment to start and update status
            time.sleep(1)
            st.rerun()  # Rerun to refresh the history table
        else:
            st.error("Failed to start ingestion job. Check logs for details.")
    elif not processing_possible:
        st.error("Cannot start ingestion due to missing critical resources (check warnings above).")

# --- Display Currently Running Job Status (Polling DB) ---
st.divider()
st.subheader("Current Job Status")
running_job_id = st.session_state.get('running_ingestion_job_id')

if running_job_id:
    # Create placeholders for progress bar and text
    progress_bar_placeholder = st.empty()
    status_text_placeholder = st.empty()
    status_text_placeholder.info(f"⏳ Monitoring ingestion job `{running_job_id}`...")

    try:
        # Poll the database for job status
        job_details = src.utils.audit_db_manager.get_job_details(running_job_id)

        if job_details:
            status = job_details.get('status', 'Unknown')
            total_files = job_details.get('total_files_in_job', 0)
            processed_files = job_details.get('processed_files', [])
            files_done_count = len(processed_files)

            # Calculate progress
            progress_value = (files_done_count / total_files) if total_files > 0 else 0

            # Determine current activity (find the last file still processing or first failed)
            current_activity = ""
            failed_files_summary = []
            success_count = 0
            cached_count = 0
            processing_now = None
            for f in processed_files:
                if f['status'] == 'Processing':
                    processing_now = f['file_name']
                elif f['status'] == 'Success':
                    success_count += 1
                elif f['status'] == 'Cached':
                    cached_count += 1
                elif 'Failed' in f['status']:
                    failed_files_summary.append(f"'{f['file_name']}' ({f['status']})")

            if processing_now:
                current_activity = f"Processing '{processing_now}'..."
            elif failed_files_summary:
                current_activity = f"Encountered errors ({len(failed_files_summary)} failed)."
            elif status == 'Running':
                current_activity = "Waiting for next file or finishing up..."
            else:
                current_activity = f"Job Status: {status}"

            # Update placeholders
            progress_bar_placeholder.progress(progress_value)
            status_text_placeholder.info(
                f"Job `{running_job_id}`: {files_done_count}/{total_files} files attempted. {current_activity}")

            # Check if job is finished based on DB status
            if status not in ['Running', 'Queued']:
                logger.info(f"Job {running_job_id} finished with DB status: {status}. Stopping UI monitor.")
                st.session_state['running_ingestion_job_id'] = None  # Clear the running job ID
                # Display final status message
                if status == 'Completed':
                    st.success(
                        f"Job `{running_job_id}` completed successfully ({success_count} processed, {cached_count} cached).")
                elif status == 'Completed with Errors':
                    st.warning(
                        f"⚠️ Job `{running_job_id}` completed with errors. Processed: {success_count}, Cached: {cached_count}, Failed: {len(failed_files_summary)}. Details: {', '.join(failed_files_summary)}")
                else:  # Failed
                    st.error(f"Job `{running_job_id}` failed.")
                # No rerun needed here, state change will trigger it if needed by other interactions

            else:
                # If still running, schedule a rerun to poll again
                time.sleep(3)  # Poll every 3 seconds
                st.rerun()

        else:
            # Job ID was in session state, but not found in DB (shouldn't happen often)
            status_text_placeholder.warning(
                f"Could not find details for job `{running_job_id}` in database. Clearing status.")
            if running_job_id in st.session_state:
                del st.session_state['running_ingestion_job_id']
            st.rerun()  # Rerun to clear spinner

    except Exception as poll_e:
        status_text_placeholder.error(f"Error checking job status: {poll_e}")
        logger.error(f"Error polling job status for {running_job_id}", exc_info=True)
        # Consider clearing running_job_id here too
        # if running_job_id in st.session_state: del st.session_state['running_ingestion_job_id']
else:
    st.info("No ingestion job currently running.")

# --- Ingestion History Section ---
st.header("2. Ingestion History")
refresh_button = st.button("🔄 Refresh History")

try:
    recent_jobs = src.utils.audit_db_manager.get_recent_jobs(limit=100)  # Get recent jobs
except Exception as e:
    st.error(f"Failed to load ingestion history from database: {e}")
    logger.error("Failed to load ingestion history", exc_info=True)
    recent_jobs = []

if not recent_jobs:
    st.info("No ingestion jobs found in the history.")
else:
    # Prepare data for display
    job_data = []
    for job in recent_jobs:
        job_data.append({
            "Job ID": job['job_id'],
            "Start Time": format_timestamp(job['start_timestamp']),
            "End Time": format_timestamp(job.get('end_timestamp')),  # Use .get for potentially null end time
            "Status": job['status'],
            "Files": job['total_files_in_job']
        })

    if job_data:  # Check if job_data was populated
        df_jobs = pd.DataFrame(job_data)

        # --- FIX: Force ALL columns to string dtype as a precaution ---
        try:
            for col in df_jobs.columns:
                df_jobs[col] = df_jobs[col].astype(str)
            logger.debug("Forced job summary DataFrame columns to string dtype.")
        except Exception as e:
            logger.warning(f"Could not force job summary columns to string dtype: {e}")
        # --- End Fix ---

        st.dataframe(
            df_jobs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Job ID": st.column_config.TextColumn(width="small"),
                "Status": st.column_config.TextColumn(width="small"),
                "Files": st.column_config.NumberColumn(width="small", format="%d"),
                "Start Time": st.column_config.TextColumn(width="medium"),
                "End Time": st.column_config.TextColumn(width="medium"),
            }
        )

    st.subheader("View Job Details")
    job_ids = [job['job_id'] for job in recent_jobs]  # Get IDs for selectbox
    selected_job_id = st.selectbox(
        "Select Job ID to view file details:",
        options=[""] + job_ids,  # Add empty option
        index=0,
        key="job_detail_select"
    )

    if selected_job_id:
        details = src.utils.audit_db_manager.get_job_details(selected_job_id)
        if details:
            st.write(f"**Overall Status:** {details['status']}")
            st.write(
                f"**Started:** {format_timestamp(details['start_timestamp'])} | **Ended:** {format_timestamp(details.get('end_timestamp'))}")

            files = details.get('processed_files', [])
            if files:
                st.write(f"**Files Processed ({len(files)}):**")
                for file_rec in files:
                    with st.expander(f"📄 {file_rec['file_name']} (Status: {file_rec['status']})", expanded=False):
                        # vvv ENSURE EVERY ITEM HERE IS WRAPPED IN str() vvv
                        display_values = [
                            str(file_rec.get('file_processing_id', 'N/A')),
                            str(file_rec.get('file_hash', 'N/A')),
                            f"{file_rec.get('file_size_bytes', 0):,} bytes",
                            str(file_rec.get('file_type', 'N/A')),
                            format_timestamp(file_rec.get('processing_start_timestamp')),
                            format_timestamp(file_rec.get('processing_end_timestamp')),
                            str(file_rec.get('cache_hit', False)),
                            str(file_rec.get('text_extracted', False)),
                            str(file_rec.get('num_chunks', 'N/A')),
                            str(file_rec.get('num_triples_extracted', 'N/A')),
                            str(file_rec.get('num_triples_loaded', 'N/A')),
                            str(file_rec.get('num_vectors_stored', 'N/A')),
                            str(file_rec.get('error_message', 'None'))
                        ]

                        file_df_data = {
                            "Detail": [
                                "File Processing ID", "File Hash", "Size", "Type",
                                "Processing Start", "Processing End", "Cache Hit?",
                                "Text Extracted?", "Chunks Generated", "Triples Extracted",
                                "Triples Loaded (Neo4j)", "Vectors Stored (Chroma)", "Error Message"
                            ],
                            "Value": display_values
                        }
                        # Create the DataFrame
                        details_df = pd.DataFrame(file_df_data)

                        # --- FIX: Explicitly force the 'Value' column to string dtype ---
                        try:
                            details_df['Value'] = details_df['Value'].astype(str)
                        except Exception as e:
                            logger.warning(
                                f"Could not force 'Value' column to string dtype for file '{file_rec['file_name']}': {e}")
                        # --- End Fix ---

                        # Pass the potentially modified DataFrame to Streamlit
                        st.dataframe(details_df, hide_index=True, use_container_width=True)

            # --- Add Graph Visualization Button ---
            st.markdown("---")
            st.subheader("Knowledge Graph Visualization (Experimental)")

            # Provide options for what to visualize
            viz_option = st.selectbox(
                "Select data to visualize:",
                # Add more options later (e.g., "Data for Specific File")
                ["All Data (Limit 100)", "Data Related to this Job (Placeholder)"],
                key=f"viz_option_{selected_job_id}"
            )

            if st.button("📊 Generate & Show Graph", key=f"viz_button_{selected_job_id}"):
                if visualize_knowledge_graph:  # Check if import succeeded
                    triples_to_visualize = []
                    status_placeholder = st.empty()
                    status_placeholder.info("Querying graph data for visualization...")
                    driver = None  # Use sync driver for query here
                    try:
                        # Get Neo4j connection details from config
                        neo4j_uri = config.get('NEO4J_URI')
                        neo4j_user = config.get('NEO4J_USER')
                        neo4j_password = config.get('NEO4J_PASSWORD')
                        db_name = config.get('DB_NAME', 'neo4j')

                        if not all([neo4j_uri, neo4j_user, neo4j_password]):
                            raise ValueError("Neo4j connection details missing in config.")

                        driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                        with driver.session(database=db_name) as session:
                            if viz_option == "All Data (Limit 100)":
                                # Simple query to get a sample of the graph
                                result = session.run("""
                                            MATCH (s)-[r]->(o)
                                            RETURN s.name AS subject, labels(s)[0] as subject_type,
                                                   coalesce(r.original, type(r)) as predicate,
                                                   o.name AS object, labels(o)[0] as object_type
                                            LIMIT 100
                                        """)
                                triples_to_visualize = [
                                    {k: v for k, v in record.items() if v is not None}
                                    for record in result
                                ]
                            elif viz_option == "Data Related to this Job (Placeholder)":
                                # !! Placeholder !!
                                # This requires linking job_id/file_hash to nodes/rels in Neo4j during ingestion
                                # OR querying based on source_document property if stored on nodes/chunks
                                st.warning("Querying by job ID is not yet implemented. Showing sample instead.")
                                result = session.run("""
                                            MATCH (s)-[r]->(o) RETURN s.name AS subject, labels(s)[0] as subject_type,
                                                   coalesce(r.original, type(r)) as predicate,
                                                   o.name AS object, labels(o)[0] as object_type
                                            LIMIT 50
                                        """)
                                triples_to_visualize = [
                                    {k: v for k, v in record.items() if v is not None}
                                    for record in result
                                ]
                            # Add logic here for other visualization options

                        if not triples_to_visualize:
                            status_placeholder.warning("No graph data found for the selected option.")
                        else:
                            status_placeholder.info(
                                f"Generating visualization for {len(triples_to_visualize)} relationships...")
                            with st.spinner("Generating graph HTML..."):
                                try:
                                    # Call the visualization function (ensure it takes triples list)
                                    viz_stats = visualize_knowledge_graph(
                                        triples_list=triples_to_visualize,
                                        output_file=GRAPH_HTML_FILENAME,
                                        config=config  # Pass config if needed by viz function
                                    )
                                    logger.info(f"Graph viz generated for job {selected_job_id}: {viz_stats}")

                                    # Read and display HTML
                                    viz_path = Path(GRAPH_HTML_FILENAME)
                                    if viz_path.is_file():
                                        with open(viz_path, 'r', encoding='utf-8') as f:
                                            html_content = f.read()
                                        status_placeholder.empty()  # Remove status message
                                        components.html(html_content, height=800, scrolling=True)
                                        # Add download button for the generated graph
                                        with open(viz_path, "rb") as fp:
                                            st.download_button(
                                                label="Download Graph HTML",
                                                data=fp,
                                                file_name=f"graph_job_{selected_job_id}.html",  # Job specific name
                                                mime="text/html"
                                            )
                                    else:
                                        status_placeholder.error("Graph HTML file was not generated.")

                                except Exception as viz_e:
                                    status_placeholder.error(f"Error generating graph visualization: {viz_e}")
                                    logger.error(f"Error generating graph viz for job {selected_job_id}: {viz_e}",
                                                 exc_info=True)

                    except Exception as db_e:
                        status_placeholder.error(f"Error querying Neo4j for visualization: {db_e}")
                        logger.error(f"Error querying Neo4j for viz data (Job {selected_job_id}): {db_e}",
                                     exc_info=True)
                    finally:
                        if driver:
                            driver.close()

                else:
                    st.error("Graph visualization function not available (import failed).")

        else:
            st.info("No file processing details found for this job.")
    else:
        # Message when no job is selected
        st.info("Select a job from the dropdown above to view file processing details.")

# --- OCR Storage Management Section ---
if ocr_storage:
    st.markdown("---")
    st.header("3. LLM OCR Storage Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📁 Recently Saved LLM OCR Files")

    with col2:
        if st.button("📂 Open Storage Manager", key="open_storage_manager"):
            st.info("💡 Go to the 'Processed Files Manager' page for full management features.")

    try:
        # Show recent OCR files
        recent_ocr_files = ocr_storage.list_saved_files(limit=10)

        if recent_ocr_files:
            # Display as compact table
            ocr_display_data = []
            for file_info in recent_ocr_files:
                ocr_display_data.append({
                    'Original File': file_info['original_filename'],
                    'Size (KB)': file_info['size_kb'],
                    'Text Length': f"{file_info['text_length']:,}",
                    'Modified': file_info['modified_time'][:16]  # Truncate timestamp
                })

            ocr_df = pd.DataFrame(ocr_display_data)
            st.dataframe(ocr_df, use_container_width=True, hide_index=True)

            # Quick stats
            total_files = len(recent_ocr_files)
            total_size = sum(f['size_kb'] for f in recent_ocr_files)
            total_text = sum(f['text_length'] for f in recent_ocr_files)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recent Files", total_files)
            with col2:
                st.metric("Total Size", f"{total_size:.1f} KB")
            with col3:
                st.metric("Total Text", f"{total_text:,}")
        else:
            st.info("📝 No LLM OCR files saved yet. Enable 'Save OCR Output to Disk' above to start saving.")

    except Exception as e:
        st.error(f"Error loading LLM OCR storage info: {e}")

# --- Enhanced LLM OCR Status Display ---
if enhanced_ocr_pipeline:
    st.markdown("---")
    st.header("4. Enhanced LLM OCR Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if enhanced_ocr_pipeline.mistral_client:
            st.success("☁️ **Mistral Pixtral Ready**")
            st.write("Specialized OCR & Vision")
        else:
            st.error("**Mistral Pixtral Not Available**")

    with col2:
        if hasattr(enhanced_ocr_pipeline, 'gemini_client') and enhanced_ocr_pipeline.gemini_client:
            st.success("🔥 **Gemini 1.5 Flash Ready**")
            st.write("High-quality vision processing")
        else:
            st.warning("**Gemini 1.5 Flash Not Available**")

    with col3:
        # Count available LLM methods
        available_count = 0
        if enhanced_ocr_pipeline.mistral_client:
            available_count += 1
        if hasattr(enhanced_ocr_pipeline, 'gemini_client') and enhanced_ocr_pipeline.gemini_client:
            available_count += 1
        if hasattr(enhanced_ocr_pipeline, 'openai_client') and enhanced_ocr_pipeline.openai_client:
            available_count += 1
        if hasattr(enhanced_ocr_pipeline, 'anthropic_client') and enhanced_ocr_pipeline.anthropic_client:
            available_count += 1

        st.info(f"🎯 **LLM OCR Methods**")
        st.write(f"Available: {available_count}")
        st.write(f"Fallback: {'✅' if available_count > 1 else '❌'}")

    # LLM OCR Method Recommendation
    st.subheader("💡 LLM OCR Method Recommendation")

    if available_count >= 2:
        st.success(
            "🎉 **Optimal Setup**: Multiple LLM OCR methods available. You have maximum accuracy with intelligent fallback!")
    elif available_count == 1:
        st.info(
            "🔥 **Good Setup**: One LLM OCR method available. Consider adding more API keys for fallback redundancy.")
    else:
        st.error(
            "**No LLM OCR Available**: Please add at least one LLM API key to config.toml (Gemini, Mistral, OpenAI, or Anthropic)")

# --- Configuration Help Section ---
with st.expander("⚙️ LLM OCR Configuration Help", expanded=False):
    st.markdown("""
    ### Enhanced LLM OCR Configuration

    **Add this to your `config.toml` file:**

    ```toml
    [llm]
    # Primary LLM for text processing
    model = "gemini-1.5-flash-latest"
    api_key = "your_gemini_api_key_here"

    # OCR-specific LLM settings
    [llm.ocr]
    mistral_api_key = "your_mistral_api_key_here"  # For Pixtral OCR
    gemini_api_key = "your_gemini_api_key_here"   # For Gemini Vision
    openai_api_key = "your_openai_api_key_here"   # For GPT-4o Vision
    anthropic_api_key = "your_claude_api_key_here" # For Claude Vision

    # OCR processing settings
    [ocr]
    primary_method = "gemini"  # gemini, mistral, openai, anthropic
    fallback_enabled = true    # Use other methods if primary fails
    confidence_threshold = 0.7 # Minimum confidence for results
    ```

    **Environment Variables (override config.toml):**
    - `GEMINI_API_KEY=your_api_key_here`
    - `MISTRAL_API_KEY=your_api_key_here`
    - `OPENAI_API_KEY=your_api_key_here`
    - `ANTHROPIC_API_KEY=your_api_key_here`

    **Performance Tips:**
    - 🔥 **Gemini 1.5 Flash**: Fast and cost-effective for most documents
    - ☁️ **Mistral Pixtral**: Specialized for complex layouts and tables
    - 🚀 **GPT-4o Vision**: Highest accuracy for challenging documents
    - 🤖 **Claude 3.5 Sonnet**: Excellent for structured document analysis
    - 💾 **Save to Disk**: Enable for backup and analysis
    - 🎯 **Multiple Methods**: Use fallback for maximum reliability
    """)

# --- Performance Metrics Display ---
if st.session_state.get('last_ocr_results'):
    st.markdown("---")
    st.header("5. Last LLM OCR Performance")

    last_results = st.session_state['last_ocr_results']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Files Processed", len(last_results))
    with col2:
        successful = sum(1 for r in last_results if r.get('success', False))
        st.metric("Success Rate", f"{successful / len(last_results) * 100:.1f}%")
    with col3:
        avg_confidence = sum(r.get('confidence', 0) for r in last_results) / len(last_results)
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    with col4:
        total_time = sum(r.get('processing_time', 0) for r in last_results)
        st.metric("Total Time", f"{total_time:.1f}s")

    # Method breakdown
    st.subheader("LLM OCR Method Usage")
    method_counts = {}
    for result in last_results:
        method = result.get('method_used', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1

    method_df = pd.DataFrame([
        {'Method': method.upper(), 'Count': count, 'Percentage': f"{count / len(last_results) * 100:.1f}%"}
        for method, count in method_counts.items()
    ])
    st.dataframe(method_df, use_container_width=True, hide_index=True)
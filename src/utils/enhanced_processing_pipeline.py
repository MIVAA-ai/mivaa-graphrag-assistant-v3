# src/utils/enhanced_processing_pipeline.py - ENHANCED PROCESSING WITH REAL-TIME UPDATES

import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid
from pathlib import Path

# Import existing modules
from src.utils.processing_pipeline import (
    run_ingestion_pipeline_thread,
    start_ingestion_job_async as original_start_job,
    is_job_running,
    extract_knowledge_graph,
    store_chunks_and_embeddings,
    get_file_hash,
    load_triples_from_cache,
    save_triples_to_cache
)

# Import progress tracking
from src.utils.realtime_progress import (
    get_progress_tracker,
    update_file_progress,
    ProcessingStage,
    ProgressUpdate
)

import src.utils.audit_db_manager as audit_db

logger = logging.getLogger(__name__)

# Thread tracking for enhanced monitoring
_enhanced_job_threads: Dict[str, threading.Thread] = {}
_job_progress_data: Dict[str, Dict] = {}


def enhanced_start_ingestion_job_async(
        uploaded_files: List[Any],
        config: Dict[str, Any],
        use_cache: bool,
        enhanced_ocr_pipeline: Optional[Any],
        neo4j_exporter: Optional[Any],
        embedding_model_resource: Optional[Any],
        chroma_collection_resource: Optional[Any],
        requests_session_resource: Optional[Any],
        nlp_pipeline_resource: Optional[Any],
        enable_realtime_progress: bool = True
) -> Optional[str]:
    """
    Enhanced ingestion job starter with real-time progress tracking.

    This replaces the original start_ingestion_job_async but adds:
    - Real-time progress updates
    - Enhanced monitoring
    - Stage-by-stage tracking
    - Performance metrics
    """
    if not uploaded_files:
        logger.warning("No files provided for enhanced ingestion.")
        return None

    # Create the initial job record
    job_id = audit_db.create_ingestion_job(total_files=len(uploaded_files))
    if not job_id:
        logger.error("Failed to create enhanced ingestion job record.")
        return None

    # Initialize progress tracking
    if enable_realtime_progress:
        _job_progress_data[job_id] = {
            'total_files': len(uploaded_files),
            'start_time': datetime.now().isoformat(),
            'files': {f.name: {'status': 'queued', 'progress': 0} for f in uploaded_files}
        }

    # Create and start enhanced background thread
    thread = threading.Thread(
        target=enhanced_ingestion_pipeline_thread,
        kwargs={
            "job_id": job_id,
            "uploaded_files": uploaded_files,
            "config": config,
            "use_cache": use_cache,
            "enhanced_ocr_pipeline": enhanced_ocr_pipeline,
            "neo4j_exporter": neo4j_exporter,
            "embedding_model_resource": embedding_model_resource,
            "chroma_collection_resource": chroma_collection_resource,
            "requests_session": requests_session_resource,
            "nlp_pipeline_resource": nlp_pipeline_resource,
            "enable_realtime_progress": enable_realtime_progress
        },
        daemon=True
    )

    _enhanced_job_threads[job_id] = thread
    thread.start()

    logger.info(f"Enhanced ingestion job {job_id} started with real-time tracking")
    return job_id


def enhanced_ingestion_pipeline_thread(
        job_id: str,
        uploaded_files: List[Any],
        config: Dict[str, Any],
        use_cache: bool,
        enhanced_ocr_pipeline: Optional[Any],
        neo4j_exporter: Optional[Any],
        embedding_model_resource: Optional[Any],
        chroma_collection_resource: Optional[Any],
        requests_session: Optional[Any],
        nlp_pipeline_resource: Optional[Any],
        enable_realtime_progress: bool = True
):
    """
    Enhanced ingestion pipeline with real-time progress tracking.

    This wraps the original pipeline but adds detailed progress updates
    for each stage of processing.
    """
    logger.info(f"[Enhanced Job {job_id}] Starting pipeline thread for {len(uploaded_files)} files")

    # Track overall job progress
    files_processed_successfully = 0
    files_failed = 0
    files_cached = 0

    # Process each file with enhanced tracking
    for i, uploaded_file in enumerate(uploaded_files):
        file_name = getattr(uploaded_file, 'name', f'file_{i}')
        file_processing_id = None

        # Update progress: Starting file
        if enable_realtime_progress:
            update_file_progress(
                job_id=job_id,
                file_name=file_name,
                stage=ProcessingStage.UPLOAD,
                progress=0,
                status="starting",
                message=f"Starting processing of {file_name}"
            )

        try:
            # Stage 1: File Upload and Preparation
            logger.info(f"[Enhanced Job {job_id}] Processing file {i + 1}/{len(uploaded_files)}: '{file_name}'")

            # Get file details
            try:
                file_content_bytes = uploaded_file.getvalue()
            except AttributeError:
                if isinstance(uploaded_file, bytes):
                    file_content_bytes = uploaded_file
                else:
                    raise ValueError(f"Cannot extract content from {type(uploaded_file)}")

            file_hash = get_file_hash(file_content_bytes)
            file_size = len(file_content_bytes)
            file_type = getattr(uploaded_file, 'type', 'unknown')

            # Create audit record
            file_processing_id = audit_db.start_file_processing(
                job_id=job_id,
                file_name=file_name,
                file_size=file_size,
                file_type=file_type,
                file_hash=file_hash
            )

            if not file_processing_id:
                raise Exception("Failed to create audit record")

            # Update progress: Upload complete
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.UPLOAD,
                    progress=10,
                    status="processing",
                    message="File uploaded and prepared"
                )

            # Stage 2: Check Cache
            cached_data = load_triples_from_cache(file_hash) if use_cache else None
            cache_hit = bool(cached_data)

            if cache_hit:
                logger.info(f"[Enhanced Job {job_id}] Cache hit for '{file_name}'")

                extracted_triples, text_chunks = cached_data
                num_chunks = len(text_chunks) if text_chunks else 0
                num_triples_extracted = len(extracted_triples) if extracted_triples else 0

                # Update progress: Cache hit
                if enable_realtime_progress:
                    update_file_progress(
                        job_id=job_id,
                        file_name=file_name,
                        stage=ProcessingStage.COMPLETION,
                        progress=100,
                        status="cached",
                        message=f"Retrieved from cache: {num_triples_extracted} triples, {num_chunks} chunks"
                    )

                # Update audit record for cached file
                audit_db.update_file_status(
                    file_processing_id=file_processing_id,
                    status='Cached',
                    cache_hit=True,
                    text_extracted=True,
                    num_chunks=num_chunks,
                    num_triples_extracted=num_triples_extracted,
                    num_triples_loaded=num_triples_extracted,
                    num_vectors_stored=num_chunks
                )

                files_cached += 1
                continue

            # Stage 3: OCR Text Extraction
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.OCR_EXTRACTION,
                    progress=15,
                    status="processing",
                    message="Extracting text with OCR..."
                )

            text_content = None
            if file_type == "text/plain":
                text_content = file_content_bytes.decode('utf-8', errors='replace')
            elif file_type in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
                if enhanced_ocr_pipeline:
                    from src.utils.processing_pipeline import process_uploaded_file_ocr
                    text_content = process_uploaded_file_ocr(uploaded_file, enhanced_ocr_pipeline)
                else:
                    raise ValueError("Enhanced OCR pipeline not available")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            if not text_content:
                raise ValueError("Text extraction failed or yielded no content")

            # Update progress: OCR complete
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.OCR_EXTRACTION,
                    progress=30,
                    status="processing",
                    message=f"Text extracted: {len(text_content)} characters"
                )

            # Stage 4: Text Sanitization
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.TEXT_SANITIZATION,
                    progress=35,
                    status="processing",
                    message="Sanitizing and structuring text..."
                )

            # Apply text sanitization if available
            try:
                from src.knowledge_graph.text_sanitization import sanitize_and_structure_text
                from src.utils.processing_pipeline import detect_document_type

                document_type = detect_document_type(file_name, text_content)

                sanitization_config = {
                    "text_sanitization": {
                        "max_tokens_per_chunk": 3000,
                        "overlap_tokens": 200,
                        "min_chunk_tokens": 100,
                        "max_tokens": 4000,
                        "temperature": 0.1
                    },
                    "llm": {
                        "model": config.get('LLM_MODEL'),
                        "api_key": config.get('LLM_API_KEY'),
                        "base_url": config.get('LLM_BASE_URL'),
                        "max_tokens": 4000,
                        "temperature": 0.1
                    }
                }

                sanitized_text = sanitize_and_structure_text(
                    text=text_content,
                    config=sanitization_config,
                    document_type=document_type,
                    requests_session=requests_session
                )

                if enable_realtime_progress:
                    update_file_progress(
                        job_id=job_id,
                        file_name=file_name,
                        stage=ProcessingStage.TEXT_SANITIZATION,
                        progress=40,
                        status="processing",
                        message="Text sanitization completed"
                    )

            except ImportError:
                logger.info(f"Text sanitization not available, using original text")
                sanitized_text = text_content
            except Exception as e:
                logger.warning(f"Text sanitization failed: {e}, using original text")
                sanitized_text = text_content

            # Stage 5: Coreference Resolution
            if config.get('COREFERENCE_RESOLUTION_ENABLED', False) and nlp_pipeline_resource:
                if enable_realtime_progress:
                    update_file_progress(
                        job_id=job_id,
                        file_name=file_name,
                        stage=ProcessingStage.TEXT_SANITIZATION,
                        progress=45,
                        status="processing",
                        message="Resolving coreferences..."
                    )

                from src.knowledge_graph.text_utils import resolve_coreferences_spacy
                resolved_text = resolve_coreferences_spacy(sanitized_text, nlp_pipeline_resource)
                if len(resolved_text) != len(sanitized_text):
                    sanitized_text = resolved_text

            # Stage 6: Text Chunking
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.CHUNKING,
                    progress=50,
                    status="processing",
                    message="Creating text chunks..."
                )

            from src.knowledge_graph.text_utils import chunk_text
            text_chunks = chunk_text(
                sanitized_text,
                chunk_size=config.get('CHUNK_SIZE', 1000),
                chunk_overlap=config.get('CHUNK_OVERLAP', 100)
            )

            num_chunks = len(text_chunks)
            if num_chunks == 0:
                raise ValueError("Text chunking resulted in zero chunks")

            # Update progress: Chunking complete
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.CHUNKING,
                    progress=55,
                    status="processing",
                    message=f"Created {num_chunks} text chunks"
                )

            # Stage 7: Knowledge Graph Extraction
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.KNOWLEDGE_EXTRACTION,
                    progress=60,
                    status="processing",
                    message="Extracting knowledge graph..."
                )

            extracted_triples, _, num_triples_extracted = extract_knowledge_graph(
                text_content=sanitized_text,
                config=config,
                requests_session=requests_session
            )

            # Update progress: KG extraction complete
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.KNOWLEDGE_EXTRACTION,
                    progress=75,
                    status="processing",
                    message=f"Extracted {num_triples_extracted} knowledge triples"
                )

            # Stage 8: Standardization (if enabled)
            num_triples_after_standardization = len(extracted_triples)
            if config.get('STANDARDIZATION_ENABLED', False) and extracted_triples:
                if enable_realtime_progress:
                    update_file_progress(
                        job_id=job_id,
                        file_name=file_name,
                        stage=ProcessingStage.STANDARDIZATION,
                        progress=80,
                        status="processing",
                        message="Standardizing entities..."
                    )

                try:
                    from src.knowledge_graph.entity_standardization import standardize_entities
                    standardized_result = standardize_entities(extracted_triples, config)
                    if standardized_result is not None:
                        extracted_triples = standardized_result
                        num_triples_after_standardization = len(extracted_triples)
                except Exception as e:
                    logger.error(f"Standardization failed: {e}")

            # Stage 9: Inference (if enabled)
            if config.get('INFERENCE_ENABLED', False) and extracted_triples:
                if enable_realtime_progress:
                    update_file_progress(
                        job_id=job_id,
                        file_name=file_name,
                        stage=ProcessingStage.INFERENCE,
                        progress=85,
                        status="processing",
                        message="Applying relationship inference..."
                    )

                try:
                    from src.knowledge_graph.entity_standardization import infer_relationships
                    inferred_result = infer_relationships(extracted_triples, config)
                    if inferred_result is not None:
                        extracted_triples = inferred_result
                except Exception as e:
                    logger.error(f"Inference failed: {e}")

            # Stage 10: Neo4j Storage
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.NEO4J_STORAGE,
                    progress=90,
                    status="processing",
                    message="Storing triples in Neo4j..."
                )

            num_triples_loaded = 0
            if extracted_triples and neo4j_exporter:
                neo4j_success, num_triples_loaded = neo4j_exporter.store_triples(extracted_triples)
                if not neo4j_success:
                    logger.warning(f"Neo4j storage had issues, loaded {num_triples_loaded} triples")
            else:
                logger.info(f"Skipping Neo4j storage (no triples or exporter unavailable)")

            # Stage 11: Vector Storage
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.VECTOR_STORAGE,
                    progress=95,
                    status="processing",
                    message="Storing embeddings in vector database..."
                )

            num_vectors_stored = 0
            if text_chunks and embedding_model_resource and chroma_collection_resource:
                embedding_success, num_vectors_stored = store_chunks_and_embeddings(
                    text_chunks=text_chunks,
                    embedding_model=embedding_model_resource,
                    chroma_collection=chroma_collection_resource,
                    current_doc_name=file_name,
                    config=config
                )
                if not embedding_success:
                    logger.warning(f"Embedding storage had issues, stored {num_vectors_stored} vectors")
            else:
                logger.info(f"Skipping embedding storage (no chunks or resources unavailable)")

            # Stage 12: Cache Storage
            if not cache_hit and extracted_triples and text_chunks:
                cache_success = save_triples_to_cache(file_hash, extracted_triples, text_chunks)
                if cache_success:
                    logger.info(f"Successfully saved to cache for future use")

            # Stage 13: Completion
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.COMPLETION,
                    progress=100,
                    status="success",
                    message=f"Processing complete: {num_triples_loaded} triples, {num_vectors_stored} vectors"
                )

            # Update audit record
            audit_db.update_file_status(
                file_processing_id=file_processing_id,
                status='Success',
                cache_hit=cache_hit,
                text_extracted=True,
                num_chunks=num_chunks,
                num_triples_extracted=num_triples_extracted,
                num_triples_loaded=num_triples_loaded,
                num_vectors_stored=num_vectors_stored
            )

            files_processed_successfully += 1
            logger.info(f"[Enhanced Job {job_id}] Successfully processed '{file_name}'")

        except Exception as e:
            logger.error(f"[Enhanced Job {job_id}] Failed processing '{file_name}': {e}", exc_info=True)
            files_failed += 1

            # Update progress: Failed
            if enable_realtime_progress:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=ProcessingStage.COMPLETION,
                    progress=100,
                    status="failed",
                    message=f"Processing failed: {str(e)[:100]}",
                    error=str(e)
                )

            # Determine failure type
            error_str = str(e).lower()
            if "ocr" in error_str:
                final_status = 'Failed - OCR'
            elif "kg extract" in error_str or "knowledge" in error_str:
                final_status = 'Failed - KG Extract'
            elif "neo4j" in error_str:
                final_status = 'Failed - Neo4j'
            elif "embedding" in error_str or "vector" in error_str:
                final_status = 'Failed - Embedding'
            else:
                final_status = 'Failed - Unknown'

            # Update audit record for failure
            if file_processing_id:
                audit_db.update_file_status(
                    file_processing_id=file_processing_id,
                    status=final_status,
                    error_message=str(e)[:500]
                )

    # Update overall job status
    if files_failed == 0:
        final_job_status = 'Completed'
    elif files_processed_successfully > 0 or files_cached > 0:
        final_job_status = 'Completed with Errors'
    else:
        final_job_status = 'Failed'

    audit_db.update_job_status(job_id=job_id, status=final_job_status)

    # Clean up progress data
    if job_id in _job_progress_data:
        _job_progress_data[job_id]['end_time'] = datetime.now().isoformat()
        _job_progress_data[job_id]['final_status'] = final_job_status

    logger.info(
        f"[Enhanced Job {job_id}] Pipeline completed. Success: {files_processed_successfully}, Failed: {files_failed}, Cached: {files_cached}")


def get_enhanced_job_progress(job_id: str) -> Optional[Dict[str, Any]]:
    """Get real-time progress data for an enhanced job."""
    return _job_progress_data.get(job_id)


def is_enhanced_job_running(job_id: str) -> bool:
    """Check if an enhanced job is still running."""
    thread = _enhanced_job_threads.get(job_id)
    if thread:
        return thread.is_alive()
    return False


def get_all_enhanced_jobs() -> Dict[str, Dict]:
    """Get progress data for all enhanced jobs."""
    return _job_progress_data.copy()


def cleanup_completed_jobs():
    """Clean up completed job threads and data."""
    completed_jobs = []

    for job_id, thread in _enhanced_job_threads.items():
        if not thread.is_alive():
            completed_jobs.append(job_id)

    for job_id in completed_jobs:
        del _enhanced_job_threads[job_id]
        # Keep progress data for analytics, but mark as completed
        if job_id in _job_progress_data:
            _job_progress_data[job_id]['cleaned_up'] = True


def get_job_performance_metrics(job_id: str) -> Optional[Dict[str, Any]]:
    """Get performance metrics for a completed job."""
    progress_data = _job_progress_data.get(job_id)
    if not progress_data:
        return None

    # Calculate metrics
    start_time = progress_data.get('start_time')
    end_time = progress_data.get('end_time')

    if start_time and end_time:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        total_duration = (end_dt - start_dt).total_seconds()
    else:
        total_duration = None

    files_data = progress_data.get('files', {})
    total_files = len(files_data)
    successful_files = len([f for f in files_data.values() if f.get('status') == 'success'])
    failed_files = len([f for f in files_data.values() if f.get('status') == 'failed'])
    cached_files = len([f for f in files_data.values() if f.get('status') == 'cached'])

    return {
        'job_id': job_id,
        'total_files': total_files,
        'successful_files': successful_files,
        'failed_files': failed_files,
        'cached_files': cached_files,
        'success_rate': successful_files / total_files if total_files > 0 else 0,
        'total_duration_seconds': total_duration,
        'files_per_second': total_files / total_duration if total_duration and total_duration > 0 else None,
        'final_status': progress_data.get('final_status', 'Unknown')
    }


# Integration with existing systems

def enhanced_process_uploaded_file_ocr_with_storage(uploaded_file, enhanced_ocr_pipeline, save_to_disk=True,
                                                    job_id=None):
    """Enhanced OCR processing with optional progress tracking."""
    filename = getattr(uploaded_file, 'name', 'uploaded_file')

    # Update progress if job_id provided
    if job_id:
        update_file_progress(
            job_id=job_id,
            file_name=filename,
            stage=ProcessingStage.OCR_EXTRACTION,
            progress=0,
            status="processing",
            message="Starting OCR extraction..."
        )

    try:
        # Use existing processing function
        from src.utils.processing_pipeline import process_uploaded_file_ocr_with_storage
        result = process_uploaded_file_ocr_with_storage(uploaded_file, enhanced_ocr_pipeline, save_to_disk)

        # Update progress on completion
        if job_id and result['success']:
            update_file_progress(
                job_id=job_id,
                file_name=filename,
                stage=ProcessingStage.OCR_EXTRACTION,
                progress=100,
                status="completed",
                message=f"OCR completed: {result.get('text_length', 0)} characters extracted"
            )
        elif job_id:
            update_file_progress(
                job_id=job_id,
                file_name=filename,
                stage=ProcessingStage.OCR_EXTRACTION,
                progress=100,
                status="failed",
                message="OCR extraction failed",
                error=result.get('error', 'Unknown error')
            )

        return result

    except Exception as e:
        if job_id:
            update_file_progress(
                job_id=job_id,
                file_name=filename,
                stage=ProcessingStage.OCR_EXTRACTION,
                progress=100,
                status="error",
                message="OCR processing error",
                error=str(e)
            )
        raise


# Backward compatibility functions

def start_ingestion_job_async(*args, **kwargs):
    """Backward compatible wrapper that uses enhanced processing by default."""
    # Add enable_realtime_progress=True by default
    kwargs.setdefault('enable_realtime_progress', True)

    # Use enhanced version
    return enhanced_start_ingestion_job_async(*args, **kwargs)


def get_job_status_enhanced(job_id: str) -> Dict[str, Any]:
    """Get comprehensive job status including real-time progress."""
    # Get basic audit data
    try:
        job_details = audit_db.get_job_details(job_id)
    except Exception:
        job_details = None

    # Get enhanced progress data
    progress_data = get_enhanced_job_progress(job_id)

    # Get performance metrics
    performance_metrics = get_job_performance_metrics(job_id)

    # Check if job is running
    is_running = is_enhanced_job_running(job_id)

    return {
        'job_id': job_id,
        'is_running': is_running,
        'audit_details': job_details,
        'progress_data': progress_data,
        'performance_metrics': performance_metrics,
        'has_realtime_data': bool(progress_data)
    }


# Monitoring and maintenance functions

def monitor_all_jobs() -> Dict[str, Any]:
    """Monitor all active and recent jobs."""
    active_jobs = []
    completed_jobs = []

    for job_id, thread in _enhanced_job_threads.items():
        if thread.is_alive():
            active_jobs.append({
                'job_id': job_id,
                'progress_data': get_enhanced_job_progress(job_id),
                'performance_metrics': get_job_performance_metrics(job_id)
            })
        else:
            completed_jobs.append({
                'job_id': job_id,
                'progress_data': get_enhanced_job_progress(job_id),
                'performance_metrics': get_job_performance_metrics(job_id)
            })

    return {
        'active_jobs': active_jobs,
        'completed_jobs': completed_jobs,
        'total_active': len(active_jobs),
        'total_completed': len(completed_jobs)
    }


def export_job_analytics(job_id: str, output_path: Optional[str] = None) -> str:
    """Export comprehensive job analytics to JSON."""
    if not output_path:
        output_path = f"job_analytics_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    analytics_data = {
        'job_id': job_id,
        'export_timestamp': datetime.now().isoformat(),
        'job_status': get_job_status_enhanced(job_id),
        'progress_data': get_enhanced_job_progress(job_id),
        'performance_metrics': get_job_performance_metrics(job_id)
    }

    # Get detailed progress updates from tracker
    from src.utils.realtime_progress import get_progress_tracker
    tracker = get_progress_tracker()
    analytics_data['detailed_updates'] = [
        {
            'file_name': update.file_name,
            'stage': update.stage.value,
            'progress': update.progress_percent,
            'status': update.status,
            'message': update.message,
            'timestamp': update.timestamp,
            'error': update.error
        }
        for update in tracker.get_job_updates(job_id)
    ]

    # Save to file
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analytics_data, f, indent=2, default=str)

    logger.info(f"Job analytics exported to: {output_path}")
    return output_path


# Example usage and testing

if __name__ == "__main__":
    # Example of how to use the enhanced pipeline
    print("Enhanced Processing Pipeline with Real-Time Progress")
    print("=" * 60)

    # Monitor all jobs
    all_jobs = monitor_all_jobs()
    print(f"Active jobs: {all_jobs['total_active']}")
    print(f"Completed jobs: {all_jobs['total_completed']}")

    # Cleanup completed jobs
    cleanup_completed_jobs()
    print("Cleaned up completed job threads")

    print("\nEnhanced processing pipeline ready for integration!")
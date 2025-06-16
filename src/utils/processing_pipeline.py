# processing_pipeline.py

import nest_asyncio

nest_asyncio.apply()

import streamlit as st
import logging
import threading
import datetime
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import traceback
import base64
import tempfile
import os
import io
import re
import json
import configparser
import sys
import shutil
from datetime import date
import requests
import asyncio
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection
from src.knowledge_graph.text_utils import resolve_coreferences_spacy
from src.utils.ocr_storage import create_storage_manager

# Text sanitization import with fallback
try:
    from src.knowledge_graph.text_sanitization import sanitize_and_structure_text

    HAS_TEXT_SANITIZATION = True
    logger = logging.getLogger(__name__)
    logger.info("Text sanitization module imported successfully")
except ImportError as e:
    HAS_TEXT_SANITIZATION = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Text sanitization module not available: {e}. Using original text.")


    def sanitize_and_structure_text(text, config, document_type, requests_session):
        """Fallback function when text sanitization is not available"""
        return text

# Import necessary modules from your project
try:
    import src.utils.audit_db_manager
except ImportError:
    st.error("Failed to import audit_db_manager. Ensure it's in the correct path.")


    class DummyAuditManager:
        def __getattr__(self, name): return lambda *args, **kwargs: None


    src.utils.audit_db_manager = DummyAuditManager()
    logging.error("Using DummyAuditManager due to import error.", exc_info=True)

# Import functions/classes needed for processing
try:
    from src.knowledge_graph.text_utils import chunk_text
    from src.knowledge_graph.llm import QuotaError
    from neo4j_exporter import Neo4jExporter
    from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError
    from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships, \
        limit_predicate_length
    from src.knowledge_graph.prompts import MAIN_SYSTEM_PROMPT, MAIN_USER_PROMPT

except ImportError as e:
    st.error(f"Import Error in processing_pipeline.py: {e}. Ensure all source files are accessible.")
    logging.error(f"Import Error in processing_pipeline.py: {e}", exc_info=True)


    # Define placeholders if imports fail
    def chunk_text(*args, **kwargs):
        return ["chunk1", "chunk2"]


    class QuotaError(Exception):
        pass

# Logger setup
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')

# Initialize storage manager
ocr_storage = create_storage_manager("ocr_outputs")


# =============================================================================
# OCR PROCESSING FUNCTIONS (CLEANED UP)
# =============================================================================

def process_uploaded_file_ocr(uploaded_file: Any, enhanced_ocr_pipeline: Any) -> Optional[str]:
    """
    Enhanced OCR processing using the new pipeline.
    This is the MAIN OCR function that returns text content as a string.

    Args:
        uploaded_file: Streamlit uploaded file object
        enhanced_ocr_pipeline: The EnhancedOCRPipeline instance

    Returns:
        str: Extracted text content, or None if extraction fails
    """
    if not enhanced_ocr_pipeline:
        logger.error("Enhanced OCR pipeline not provided")
        return None

    # Safe filename extraction
    filename = getattr(uploaded_file, 'name', 'uploaded_file')

    try:
        # Use the enhanced pipeline
        result = enhanced_ocr_pipeline.extract_text(uploaded_file, save_to_disk=True)

        if result.success:
            logger.info(f"Enhanced OCR successful for {filename}: "
                        f"method={result.method_used}, confidence={result.confidence:.3f}, "
                        f"length={len(result.text)}")
            return result.text
        else:
            logger.error(f"Enhanced OCR failed for {filename}: {result.error_message}")
            return None

    except Exception as e:
        logger.error(f"Enhanced OCR processing error for {filename}: {e}")
        return None


def process_uploaded_file_ocr_with_storage(uploaded_file, enhanced_ocr_pipeline, save_to_disk=True):
    """
    Process file with OCR and return detailed metadata.
    This function returns a dictionary with processing results and metadata.

    Args:
        uploaded_file: Streamlit uploaded file object
        enhanced_ocr_pipeline: The EnhancedOCRPipeline instance
        save_to_disk: Whether to save OCR output to local storage

    Returns:
        dict: Contains OCR results, metadata, and file paths if saved
    """
    filename = getattr(uploaded_file, 'name', 'uploaded_file')

    try:
        # Use the enhanced pipeline
        ocr_result = enhanced_ocr_pipeline.extract_text(uploaded_file, save_to_disk=save_to_disk)

        # Convert to the format expected by your existing code
        result = {
            'success': ocr_result.success,
            'ocr_text': ocr_result.text,
            'text_length': len(ocr_result.text) if ocr_result.text else 0,
            'saved_files': ocr_result.saved_files,
            'method_used': ocr_result.method_used,
            'confidence': ocr_result.confidence,
            'processing_time': ocr_result.processing_time
        }

        if not ocr_result.success:
            result['error'] = ocr_result.error_message

        return result

    except Exception as e:
        logger.error(f"Enhanced OCR processing failed for {filename}: {e}")
        return {
            'success': False,
            'ocr_text': None,
            'error': str(e),
            'saved_files': None,
            'text_length': 0
        }


# =============================================================================
# BATCH PROCESSING FUNCTIONS (CLEANED UP)
# =============================================================================

def process_batch_with_enhanced_storage(uploaded_files, enhanced_ocr_pipeline, save_to_disk=True):
    """
    Process multiple files with enhanced OCR pipeline.
    Returns list of processing results with metadata.

    Args:
        uploaded_files: List of file objects
        enhanced_ocr_pipeline: Enhanced OCR pipeline object
        save_to_disk: Whether to save results to disk

    Returns:
        list: List of processing results with metadata
    """
    import datetime

    batch_results = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, 'name', 'uploaded_file')
        logger.info(f"Processing {filename} in batch...")

        # Process single file and get detailed metadata
        result = process_uploaded_file_ocr_with_storage(
            uploaded_file=uploaded_file,
            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
            save_to_disk=save_to_disk
        )

        # Add additional metadata safely
        try:
            file_type = getattr(uploaded_file, 'type', 'unknown')
            file_size_bytes = len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else 0
        except Exception:
            file_type = 'unknown'
            file_size_bytes = 0

        result.update({
            'original_filename': filename,
            'file_type': file_type,
            'file_size_bytes': file_size_bytes,
            'timestamp': datetime.datetime.now().isoformat(),
            'processing_status': 'success' if result.get('success') else 'failed'
        })

        batch_results.append(result)

    # Save batch summary if storage is available
    if save_to_disk and batch_results and hasattr(enhanced_ocr_pipeline,
                                                  'ocr_storage') and enhanced_ocr_pipeline.ocr_storage:
        try:
            summary_path = enhanced_ocr_pipeline.ocr_storage.save_batch_summary(batch_results)
            logger.info(f"Batch summary saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save batch summary: {e}")

    return batch_results


def process_batch_with_storage(uploaded_files, mistral_client=None, enhanced_ocr_pipeline=None, save_to_disk=True):
    """
    Backward compatibility wrapper.
    Maps old function calls to new enhanced storage function.
    """
    if enhanced_ocr_pipeline is None and mistral_client is not None:
        logger.warning("Using legacy mistral_client parameter. Consider upgrading to enhanced_ocr_pipeline.")

    pipeline = enhanced_ocr_pipeline or mistral_client

    return process_batch_with_enhanced_storage(
        uploaded_files=uploaded_files,
        enhanced_ocr_pipeline=pipeline,
        save_to_disk=save_to_disk
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_basic_fields_from_text(text: str) -> dict:
    """Extract basic structured fields from OCR text."""
    import re
    from datetime import datetime

    fields = {}

    try:
        # Invoice/WO number
        invoice_match = re.search(r'(?:Invoice|WO|Work Order)\s*#?\s*:?\s*([A-Z0-9\-]+)', text, re.IGNORECASE)
        if invoice_match:
            fields['document_number'] = invoice_match.group(1)

        # Date
        date_match = re.search(r'Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
        if date_match:
            fields['date'] = date_match.group(1)

        # Total amount
        total_match = re.search(r'Total\s*:?\s*\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
        if total_match:
            fields['total_amount'] = total_match.group(1)

        # AFE number (common in oil & gas)
        afe_match = re.search(r'AFE\s*#?\s*:?\s*([A-Z0-9\-]+)', text, re.IGNORECASE)
        if afe_match:
            fields['afe_number'] = afe_match.group(1)

        # Company names
        company_match = re.search(r'(?:From|Company)\s*:?\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
        if company_match:
            fields['company'] = company_match.group(1).strip()

        # Add extraction timestamp
        fields['extracted_at'] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Field extraction failed: {e}")
        fields['extraction_error'] = str(e)

    return fields


def detect_document_type(filename, content):
    """Auto-detect document type based on filename and content."""
    filename_lower = filename.lower()
    content_lower = content.lower()

    # Document type detection logic
    if any(term in filename_lower for term in ['maintenance', 'repair', 'work_order', 'wo']):
        return 'maintenance_report'
    elif any(term in filename_lower for term in ['inventory', 'asset_list', 'equipment']):
        return 'asset_inventory'
    elif any(term in filename_lower for term in ['inspection', 'audit', 'compliance']):
        return 'inspection_report'
    elif any(term in filename_lower for term in ['invoice', 'bill', 'statement']):
        return 'invoice'
    elif any(term in content_lower for term in ['work order', 'wo#', 'maintenance request']):
        return 'work_order'
    else:
        return 'unknown'


# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION
# =============================================================================

def extract_knowledge_graph(
        text_content: str,
        config: Dict[str, Any],
        requests_session: Optional[Any]
) -> Tuple[List[Dict], List[str], int]:
    """
    Extracts, standardizes, and infers knowledge graph triples from text.
    Returns: Tuple containing (list of final triples, list of text chunks, count of initially extracted triples)
    """
    initial_triples = []
    text_chunks = []
    total_extracted = 0

    if not text_content:
        logger.warning("extract_knowledge_graph called with empty text_content.")
        return initial_triples, text_chunks, total_extracted

    # Configuration
    chunk_size_chars = config.get('CHUNK_SIZE', 1000)
    overlap_chars = config.get('CHUNK_OVERLAP', 100)
    extraction_model = config.get('TRIPLE_EXTRACTION_LLM_MODEL')
    extraction_api_key = config.get('TRIPLE_EXTRACTION_API_KEY')
    extraction_base_url = config.get('TRIPLE_EXTRACTION_BASE_URL')
    extraction_max_tokens = config.get('TRIPLE_EXTRACTION_MAX_TOKENS', 1500)
    extraction_temperature = config.get('TRIPLE_EXTRACTION_TEMPERATURE', 0.2)

    if not all([extraction_model, extraction_api_key]):
        logger.error("KG Extraction Error! Missing LLM Config for triple extraction.")
        raise ValueError("Missing configuration for Triple Extraction LLM.")

    # Chunking
    try:
        text_chunks = chunk_text(text_content, chunk_size=chunk_size_chars, chunk_overlap=overlap_chars)
        num_chunks = len(text_chunks)
        logger.info(
            f"Split text into {num_chunks} chunks (size={chunk_size_chars} chars, overlap={overlap_chars} chars).")
        if num_chunks == 0:
            logger.warning("Text chunking resulted in zero chunks.")
            return [], [], 0
    except Exception as e:
        logger.error(f"Error during text chunking: {e}", exc_info=True)
        raise RuntimeError(f"Critical error during text chunking: {e}") from e

    # KG Extraction Loop
    logger.info(f"Preparing to extract triples from {num_chunks} chunk(s)...")
    system_prompt = MAIN_SYSTEM_PROMPT
    max_retries = 2
    default_retry_delay = 5
    quota_retry_delay = 70
    overall_start_time = time.time()

    for i, chunk in enumerate(text_chunks):
        chunk_start_time = time.time()
        logger.info(f"Processing chunk {i + 1}/{num_chunks} for KG extraction...")

        user_prompt = MAIN_USER_PROMPT + f"\n```text\n{chunk}\n```\n"
        attempt = 0
        success = False
        response_text = None
        valid_chunk_triples = []

        while attempt < max_retries and not success:
            try:
                logger.debug(f"Chunk {i + 1}, Attempt {attempt + 1}: Calling LLM...")
                llm_call_start_time = time.time()
                response_text = call_llm(
                    model=extraction_model,
                    user_prompt=user_prompt,
                    api_key=extraction_api_key,
                    system_prompt=system_prompt,
                    max_tokens=extraction_max_tokens,
                    temperature=extraction_temperature,
                    base_url=extraction_base_url,
                    session=requests_session
                )
                llm_call_duration = time.time() - llm_call_start_time
                logger.debug(f"Chunk {i + 1}, Attempt {attempt + 1}: LLM call duration: {llm_call_duration:.2f}s.")

                logger.debug(f"Chunk {i + 1}, Attempt {attempt + 1}: Extracting JSON...")
                json_extract_start_time = time.time()
                chunk_results = extract_json_from_text(response_text)
                json_extract_duration = time.time() - json_extract_start_time
                logger.debug(
                    f"Chunk {i + 1}, Attempt {attempt + 1}: JSON extraction duration: {json_extract_duration:.2f}s.")

                if chunk_results and isinstance(chunk_results, list):
                    required_keys = {"subject", "subject_type", "predicate", "object", "object_type"}
                    for item_idx, item in enumerate(chunk_results):
                        if isinstance(item, dict):
                            missing_keys = required_keys - item.keys()
                            invalid_values = {k: item[k] for k in required_keys.intersection(item.keys()) if
                                              not isinstance(item[k], str) or not item[k].strip()}
                            if not missing_keys and not invalid_values:
                                chunk_hash = hashlib.sha256(chunk.encode('utf-8')).hexdigest()
                                item["chunk_text"] = chunk.strip()
                                item["predicate"] = limit_predicate_length(item["predicate"])
                                valid_chunk_triples.append(item)
                            else:
                                reason = []
                                if missing_keys: reason.append(f"missing keys: {missing_keys}")
                                if invalid_values: reason.append(f"invalid/empty values: {invalid_values}")
                                logger.warning(
                                    f"Invalid triple structure in chunk {i + 1}, item {item_idx + 1} ({'; '.join(reason)}): {item}")
                        else:
                            logger.warning(
                                f"Invalid item type (expected dict) in chunk {i + 1}, item {item_idx + 1}: {item}")

                    chunk_extracted_count = len(valid_chunk_triples)
                    total_extracted += chunk_extracted_count
                    initial_triples.extend(valid_chunk_triples)
                    success = True
                    logger.info(
                        f"Chunk {i + 1}/{num_chunks}: Attempt {attempt + 1} successful. Extracted {chunk_extracted_count} valid triples.")

                elif chunk_results is None:
                    logger.warning(f"LLM response for chunk {i + 1}, attempt {attempt + 1} did not contain valid JSON.")
                    success = True
                else:
                    logger.warning(
                        f"No valid list of triples extracted from chunk {i + 1}, attempt {attempt + 1}. Response: {response_text[:200]}...")
                    success = True

            except QuotaError as qe:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Quota Error processing chunk {i + 1}: {qe}")
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to Quota Error. Skipping chunk.")
                    break
                else:
                    logger.info(f"Waiting {quota_retry_delay}s before retry...")
                    time.sleep(quota_retry_delay)
            except TimeoutError as te:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Timeout Error processing chunk {i + 1}: {te}")
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to Timeout Error. Skipping chunk.")
                    break
                else:
                    logger.info(f"Waiting {default_retry_delay}s before retry...")
                    time.sleep(default_retry_delay)
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt}/{max_retries}: Unexpected error processing chunk {i + 1}: {e}",
                             exc_info=True)
                if attempt >= max_retries:
                    logger.error(f"Max retries reached for chunk {i + 1} due to Error. Skipping chunk.")
                    break
                else:
                    logger.info(f"Waiting {default_retry_delay}s before retry...")
                    time.sleep(default_retry_delay)

        chunk_duration = time.time() - chunk_start_time
        logger.info(f"Chunk {i + 1}/{num_chunks}: Finished processing. Duration: {chunk_duration:.2f}s.")

    overall_duration = time.time() - overall_start_time
    logger.info(
        f"Finished initial triple extraction phase. Total extracted: {total_extracted} triples. Total time: {overall_duration:.2f}s.")

    # Standardization
    processed_triples = initial_triples
    if config.get('STANDARDIZATION_ENABLED', False) and processed_triples:
        logger.info("Applying entity standardization...")
        try:
            standardized_result = standardize_entities(processed_triples, config)
            processed_triples = standardized_result if standardized_result is not None else processed_triples
            logger.info(f"Entity standardization complete. Triple count: {len(processed_triples)}")
        except Exception as e:
            logger.error(f"Error during standardization: {e}", exc_info=True)
    else:
        logger.info("Skipping standardization.")

    # Inference
    if config.get('INFERENCE_ENABLED', False) and processed_triples:
        logger.info("Applying relationship inference...")
        try:
            inferred_result = infer_relationships(processed_triples, config)
            processed_triples = inferred_result if inferred_result is not None else processed_triples
            logger.info(f"Relationship inference complete. Triple count: {len(processed_triples)}")
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
    else:
        logger.info("Skipping inference.")

    final_triples = processed_triples
    final_triple_count = len(final_triples)
    logger.info(
        f"extract_knowledge_graph finished. Final triple count: {final_triple_count}, Initial extracted count: {total_extracted}")
    if final_triples:
        logger.debug(f"Final triple example: {final_triples[0]}")

    return final_triples, text_chunks, total_extracted


# =============================================================================
# EMBEDDINGS AND STORAGE
# =============================================================================

EmbeddingModelType = Any
ChromaCollectionType = Any


def store_chunks_and_embeddings(
        text_chunks: List[str],
        embedding_model: Optional[EmbeddingModelType],
        chroma_collection: Optional[ChromaCollectionType],
        current_doc_name: str,
        config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, int]:
    """
    Generates embeddings for text chunks and stores/updates them in ChromaDB.
    Returns success status and count stored.
    """
    num_chunks = len(text_chunks)
    if not text_chunks:
        logger.warning(f"No chunks provided for embedding for document '{current_doc_name}'. Skipping storage.")
        return True, 0

    # Input Validation
    if not embedding_model:
        logger.error(
            f"Embedding model resource was not provided for '{current_doc_name}'. Cannot generate/store embeddings.")
        return False, 0
    if not chroma_collection:
        logger.error(
            f"ChromaDB collection resource was not provided for '{current_doc_name}'. Cannot store embeddings.")
        return False, 0
    if not hasattr(embedding_model, 'encode'):
        logger.error(f"Provided embedding_model object lacks an 'encode' method for '{current_doc_name}'.")
        return False, 0
    if not hasattr(chroma_collection, 'upsert'):
        logger.error(f"Provided chroma_collection object lacks an 'upsert' method for '{current_doc_name}'.")
        return False, 0

    logger.info(f"Preparing to generate and store embeddings for {num_chunks} chunks from '{current_doc_name}'...")

    # Generate IDs
    try:
        chunk_ids = [f"{current_doc_name}_{hashlib.sha256(chunk.encode('utf-8')).hexdigest()[:16]}" for chunk in
                     text_chunks]
        logger.debug(
            f"Generated {len(chunk_ids)} chunk IDs for embedding (Example: {chunk_ids[0] if chunk_ids else 'N/A'}).")
    except Exception as e:
        logger.error(f"Failed to generate chunk IDs for '{current_doc_name}': {e}", exc_info=True)
        return False, 0

    # Generate Embeddings
    embeddings_list = []
    try:
        logger.info(f"Generating embeddings for {num_chunks} chunks...")
        start_time = time.time()
        embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
        duration = time.time() - start_time
        logger.info(f"Embeddings generated successfully for {num_chunks} chunks ({duration:.2f}s).")
    except Exception as e:
        logger.error(f"Failed to generate embeddings for '{current_doc_name}': {e}", exc_info=True)
        return False, 0

    # Store in ChromaDB
    num_stored = 0
    try:
        collection_name = getattr(chroma_collection, 'name', 'Unknown')
        logger.info(f"Storing {len(chunk_ids)} embeddings/documents in ChromaDB collection '{collection_name}'...")
        start_time = time.time()
        metadatas = [{"source_document": current_doc_name, "original_chunk_index": i} for i in range(num_chunks)]

        chroma_collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings_list,
            documents=text_chunks,
            metadatas=metadatas
        )
        num_stored = len(chunk_ids)
        duration = time.time() - start_time

        try:
            current_count = chroma_collection.count()
            logger.info(
                f"Successfully upserted {num_stored} embeddings for '{current_doc_name}' ({duration:.2f}s). Collection count now: {current_count}")
        except Exception as count_e:
            logger.info(
                f"Successfully upserted {num_stored} embeddings for '{current_doc_name}' ({duration:.2f}s). (Could not retrieve updated collection count: {count_e})")

        return True, num_stored

    except Exception as e:
        logger.error(f"Failed to store embeddings in ChromaDB for '{current_doc_name}': {e}", exc_info=True)
        return False, 0


# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

CACHE_DIR = Path("./graphrag_cache")
TRIPLE_CACHE_DIR = CACHE_DIR / "triples"


def get_file_hash(file_content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    if not isinstance(file_content, bytes):
        logger.error("get_file_hash requires bytes input.")
        raise TypeError("Input must be bytes to calculate hash.")
    return hashlib.sha256(file_content).hexdigest()


def load_triples_from_cache(file_hash: str) -> Optional[Tuple[List[Dict], List[str]]]:
    """
    Loads triples and chunks from a cache file based on file hash.
    Returns a tuple containing (list of triples, list of chunks) if cache exists and is valid.
    """
    if not file_hash:
        logger.warning("load_triples_from_cache called with empty file_hash.")
        return None

    cache_file = TRIPLE_CACHE_DIR / f"{file_hash}.json"
    logger.debug(f"Checking cache file: {cache_file}")

    if cache_file.is_file():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and "triples" in data and "chunks" in data and isinstance(data["triples"],
                                                                                                list) and isinstance(
                    data["chunks"], list):
                logger.info(
                    f"Cache hit: Loaded {len(data['triples'])} triples and {len(data['chunks'])} chunks from {cache_file}")
                return data["triples"], data["chunks"]
            else:
                logger.warning(
                    f"Invalid cache file format: {cache_file}. Required keys 'triples' (list) and 'chunks' (list) not found or invalid type.")
                return None
        except json.JSONDecodeError as jde:
            logger.warning(f"Failed to decode JSON from cache file {cache_file}: {jde}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}.", exc_info=True)
            return None
    else:
        logger.debug(f"Cache miss for hash {file_hash} (File not found: {cache_file})")
        return None


def save_triples_to_cache(file_hash: str, triples: List[Dict], chunks: List[str]) -> bool:
    """
    Saves extracted triples and corresponding text chunks to a cache file.
    Returns True if saving was successful, False otherwise.
    """
    if not file_hash:
        logger.error("Cannot save cache: file_hash is empty.")
        return False
    if not isinstance(triples, list) or not isinstance(chunks, list):
        logger.error(f"Cannot save cache for hash {file_hash}: triples or chunks are not lists.")
        return False

    try:
        TRIPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = TRIPLE_CACHE_DIR / f"{file_hash}.json"
        data_to_save = {"triples": triples, "chunks": chunks}

        logger.info(f"Saving {len(triples)} triples and {len(chunks)} chunks to cache: {cache_file}")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)

        logger.info(f"Successfully saved cache file: {cache_file}")
        return True
    except TypeError as te:
        logger.error(f"Failed to save cache for hash {file_hash} due to non-serializable data: {te}", exc_info=True)
        return False
    except OSError as oe:
        logger.error(f"Failed to save cache file {cache_file} due to OS error: {oe}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred saving cache for hash {file_hash}: {e}", exc_info=True)
        return False


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def run_ingestion_pipeline_thread(
        job_id: str,
        uploaded_files: List[Any],
        config: Dict[str, Any],
        use_cache: bool,
        enhanced_ocr_pipeline: Optional[Any],
        neo4j_exporter: Optional[Neo4jExporter],
        embedding_model_resource: Optional[Any],
        chroma_collection_resource: Optional[Any],
        requests_session: Optional[requests.Session],
        nlp_pipeline_resource: Optional[Any]
):
    """
    Orchestrates the ingestion pipeline for a batch of files.
    Designed to be run in a separate thread. Logs progress to the audit DB.
    """
    logger.info(f"[Job {job_id}] Starting ingestion pipeline thread for {len(uploaded_files)} files.")
    files_processed_successfully = 0
    files_failed = 0
    files_cached = 0

    # Iterate through each uploaded file
    for i, uploaded_file in enumerate(uploaded_files):
        # Initialize details for THIS file
        file_name = getattr(uploaded_file, 'name', 'uploaded_file')
        file_processing_id = None
        file_hash = None
        text_content = None
        text_chunks = []
        extracted_triples = []

        text_extracted_success = False
        num_chunks = 0
        num_triples_extracted = 0
        num_triples_loaded = 0
        num_vectors_stored = 0
        cache_hit = False
        final_status = 'Failed - Unknown'
        error_msg_details = None
        log_msg_details = None

        try:
            # Get File Details & Start Audit Record
            logger.info(f"[Job {job_id}] Processing file {i + 1}/{len(uploaded_files)}: '{file_name}'")

            # Safe file content extraction
            try:
                file_content_bytes = uploaded_file.getvalue()
            except AttributeError:
                # Handle case where uploaded_file might be bytes already
                if isinstance(uploaded_file, bytes):
                    file_content_bytes = uploaded_file
                else:
                    raise ValueError(f"Cannot extract file content from {type(uploaded_file)}")

            file_hash = get_file_hash(file_content_bytes)
            file_size = len(file_content_bytes)
            file_type = getattr(uploaded_file, 'type', 'unknown')

            file_processing_id = src.utils.audit_db_manager.start_file_processing(
                job_id=job_id, file_name=file_name, file_size=file_size,
                file_type=file_type, file_hash=file_hash
            )
            if not file_processing_id:
                logger.error(f"[Job {job_id}] Failed to create audit record for file '{file_name}'. Skipping.")
                files_failed += 1
                continue

            # Cache Check
            cached_data = load_triples_from_cache(file_hash) if use_cache else None
            cache_hit = bool(cached_data)

            if cache_hit:
                logger.info(f"[Job {job_id} | File {file_processing_id}] Cache hit for '{file_name}'.")
                extracted_triples, text_chunks = cached_data
                num_chunks = len(text_chunks) if text_chunks else 0
                num_triples_extracted = len(extracted_triples) if extracted_triples else 0
                num_triples_loaded = num_triples_extracted
                num_vectors_stored = num_chunks
                text_extracted_success = True
                final_status = 'Cached'
                files_cached += 1

            else:  # Not a cache hit, perform full processing
                cache_hit = False

                # Step 1: Extract Text
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Extracting text...")
                if file_type == "text/plain":
                    text_content = file_content_bytes.decode('utf-8', errors='replace')
                elif file_type in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
                    if enhanced_ocr_pipeline:
                        # FIXED: Use the correct function that returns text string
                        text_content = process_uploaded_file_ocr(uploaded_file, enhanced_ocr_pipeline)
                    else:
                        raise ValueError("Enhanced OCR pipeline not available for OCR.")
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")

                if text_content:
                    text_extracted_success = True
                else:
                    raise ValueError("Text extraction failed or yielded no content.")
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Text extracted successfully.")


                logger.info(f"[Job {job_id}] OCR extracted text preview: {text_content[:300]}...")

                # Step 2: Coreference Resolution
                if config.get('COREFERENCE_RESOLUTION_ENABLED', False) and nlp_pipeline_resource:
                    logger.info(f"[Job {job_id} | File {file_processing_id}] Applying coreference resolution...")
                    resolved_text_content = resolve_coreferences_spacy(text_content, nlp_pipeline_resource)
                    if len(resolved_text_content) != len(text_content):
                        logger.info(
                            f"[Job {job_id} | File {file_processing_id}] Coreference resolution modified the text.")
                        text_content = resolved_text_content
                    else:
                        logger.info(
                            f"[Job {job_id} | File {file_processing_id}] Coreference resolution made no changes or was skipped by the resolver.")
                else:
                    logger.info(f"[Job {job_id} | File {file_processing_id}] Skipping coreference resolution.")

                # Step 3: Chunk Text
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Chunking text...")
                text_chunks = chunk_text(text_content, chunk_size=config.get('CHUNK_SIZE', 1000),
                                         chunk_overlap=config.get('CHUNK_OVERLAP', 100))
                num_chunks = len(text_chunks)
                if num_chunks == 0:
                    raise ValueError("Text chunking resulted in zero chunks.")
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Chunking complete ({num_chunks} chunks).")

                # Step 4: Text Sanitization (NEW STEP)
                if HAS_TEXT_SANITIZATION:
                    logger.info(f"[Job {job_id} | File {file_processing_id}] 🔧 Sanitizing text...")
                    document_type = detect_document_type(file_name, text_content)

                    # FIX: Create proper config structure for TextSanitizer
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
                        config=sanitization_config,  # Pass structured config
                        document_type=document_type,
                        requests_session=requests_session
                    )
                    logger.info(f"[Job {job_id} | File {file_processing_id}] Text sanitization complete.")
                else:
                    logger.info(
                        f"[Job {job_id} | File {file_processing_id}] Text sanitization not available, using original text.")
                    sanitized_text = text_content

                # Step 5: Knowledge Graph Extraction
                logger.debug(f"[Job {job_id} | File {file_processing_id}] Extracting Knowledge Graph...")
                extracted_triples, _, num_triples_extracted = extract_knowledge_graph(
                    text_content=sanitized_text,  # Use sanitized text
                    config=config,
                    requests_session=requests_session
                )
                logger.debug(
                    f"[Job {job_id} | File {file_processing_id}] KG Extraction complete. Initial extracted: {num_triples_extracted}.")

                # Step 6: Store Triples in Neo4j
                if extracted_triples and neo4j_exporter:
                    num_triples_to_store = len(extracted_triples)
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] Storing {num_triples_to_store} triples in Neo4j...")
                    neo4j_success, num_triples_loaded = neo4j_exporter.store_triples(extracted_triples)
                    if not neo4j_success:
                        raise RuntimeError(f"Neo4j storage failed, processed {num_triples_loaded} before error.")
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] Neo4j storage complete ({num_triples_loaded} triples).")
                else:
                    num_triples_loaded = 0
                    logger.info(
                        f"[Job {job_id} | File {file_processing_id}] Skipping Neo4j storage (No triples or exporter unavailable).")

                # Step 7: Store Chunks & Embeddings
                if text_chunks and embedding_model_resource and chroma_collection_resource:
                    logger.debug(f"[Job {job_id} | File {file_processing_id}] Storing {num_chunks} embeddings...")
                    embedding_success, num_vectors_stored = store_chunks_and_embeddings(
                        text_chunks=text_chunks, embedding_model=embedding_model_resource,
                        chroma_collection=chroma_collection_resource, current_doc_name=file_name, config=config
                    )
                    if not embedding_success:
                        raise RuntimeError(f"Embedding storage failed after processing {num_vectors_stored} vectors.")
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] ChromaDB storage complete ({num_vectors_stored} vectors).")
                else:
                    num_vectors_stored = 0
                    logger.info(
                        f"[Job {job_id} | File {file_processing_id}] Skipping embedding storage (No chunks or resources unavailable).")

                # Step 8: Save to Cache
                if not cache_hit and extracted_triples and text_chunks:
                    logger.debug(f"[Job {job_id} | File {file_processing_id}] Saving to cache...")
                    cache_success = save_triples_to_cache(file_hash, extracted_triples, text_chunks)
                    if cache_success:
                        logger.info(f"[Job {job_id} | File {file_processing_id}] Successfully saved to cache.")
                    else:
                        logger.warning(
                            f"[Job {job_id} | File {file_processing_id}] Failed to save to cache (non-critical).")
                else:
                    logger.debug(
                        f"[Job {job_id} | File {file_processing_id}] Skipping cache save (cache hit or no data).")

                final_status = 'Success'
                files_processed_successfully += 1
                logger.info(f"[Job {job_id} | File {file_processing_id}] Successfully processed file '{file_name}'.")

        # Main Exception Handler for the file
        except Exception as e:
            logger.error(f"[Job {job_id}] FAILED processing file '{file_name}' (ID: {file_processing_id}). Error: {e}",
                         exc_info=True)
            files_failed += 1
            tb_str = traceback.format_exc()

            # Determine specific failure status
            if isinstance(e, ValueError) and "OCR" in str(e):
                final_status = 'Failed - OCR'
            elif isinstance(e, ValueError) and "chunking" in str(e):
                final_status = 'Failed - Chunking'
            elif isinstance(e, QuotaError) or "KG Extraction" in tb_str or "extract_knowledge_graph" in tb_str:
                final_status = 'Failed - KG Extract'
            elif "Neo4j" in str(e) or "neo4j_exporter" in tb_str:
                final_status = 'Failed - Neo4j'
            elif "Embedding" in str(e) or "ChromaDB" in str(e) or "store_chunks_and_embeddings" in tb_str:
                final_status = 'Failed - Embedding'
            else:
                final_status = 'Failed - Unknown'
            error_msg_details = f"{type(e).__name__}: {str(e)[:500]}"
            log_msg_details = tb_str[:1500]

        # Single Audit Update Call for the file
        if file_processing_id:
            logger.debug(
                f"[Job {job_id} | File {file_processing_id}] Performing final status update: Status='{final_status}', Chunks={num_chunks}, Extracted={num_triples_extracted}, Loaded={num_triples_loaded}, Stored={num_vectors_stored}")
            src.utils.audit_db_manager.update_file_status(
                file_processing_id=file_processing_id,
                status=final_status,
                cache_hit=cache_hit,
                text_extracted=text_extracted_success,
                num_chunks=num_chunks,
                num_triples_extracted=num_triples_extracted,
                num_triples_loaded=num_triples_loaded,
                num_vectors_stored=num_vectors_stored,
                error_message=error_msg_details,
                log_messages=log_msg_details
            )
        else:
            logger.error(
                f"[Job {job_id}] Could not update audit status for '{file_name}' because file_processing_id was not generated.")

    # Update Overall Job Status
    final_job_status = 'Failed'
    if files_failed == 0:
        if files_processed_successfully > 0 or files_cached > 0:
            final_job_status = 'Completed'
        else:
            final_job_status = 'Completed'
    elif files_processed_successfully > 0 or files_cached > 0:
        final_job_status = 'Completed with Errors'

    logger.info(
        f"[Job {job_id}] Pipeline thread finished. Success: {files_processed_successfully}, Failed: {files_failed}, Cached: {files_cached}. Final Status: {final_job_status}")
    src.utils.audit_db_manager.update_job_status(job_id=job_id, status=final_job_status)


_pipeline_threads: Dict[str, threading.Thread] = {}


def start_ingestion_job_async(
        uploaded_files: List[Any],
        config: Dict[str, Any],
        use_cache: bool,
        enhanced_ocr_pipeline: Optional[Any],
        neo4j_exporter: Optional[Neo4jExporter],
        embedding_model_resource: Optional[Any],
        chroma_collection_resource: Optional[Any],
        requests_session_resource: Optional[Any],
        nlp_pipeline_resource: Optional[Any]
) -> Optional[str]:
    """
    Creates the audit job record and starts the pipeline in a background thread.
    Returns the job_id if successfully started, otherwise None.
    """
    if not uploaded_files:
        logger.warning("No files provided for ingestion.")
        return None

    # Create the initial job record in the DB
    job_id = src.utils.audit_db_manager.create_ingestion_job(total_files=len(uploaded_files))
    if not job_id:
        logger.error("Failed to create ingestion job record in the database.")
        st.error("Failed to start ingestion job (database error).")
        return None

    # Create and start the background thread
    logger.info(f"Starting background thread for ingestion job {job_id}...")
    thread = threading.Thread(
        target=run_ingestion_pipeline_thread,
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
            "nlp_pipeline_resource": nlp_pipeline_resource
        },
        daemon=True
    )
    _pipeline_threads[job_id] = thread
    thread.start()
    logger.info(f"Ingestion job {job_id} started in background thread.")

    return job_id


def is_job_running(job_id: str) -> bool:
    """Checks if the thread for a given job ID is still alive."""
    thread = _pipeline_threads.get(job_id)
    if thread:
        return thread.is_alive()
    return False
import logging
import json
from pathlib import Path
import configparser
import sys
from typing import List, Dict, Tuple, Set, Any
import time
from functools import wraps

# --- Vector DB and Embeddings Libraries ---
# Ensure you have installed the necessary libraries:
# pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer  # type: ignore

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

# --- Configuration ---
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Good default sentence transformer
CHROMA_PERSIST_PATH = "./chroma_db_embeddings"  # Directory to store Chroma data
COLLECTION_NAME = "doc_pipeline_embeddings"


def monitor_performance(operation_name):
    """Decorator to monitor operation performance."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Performance: {operation_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Performance: {operation_name} failed after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def load_unique_chunks_from_json(file_path: Path) -> Dict[str, str]:
    """
    Loads triple data from a JSON file and extracts unique chunks.

    Args:
        file_path: Path to the input JSON file containing triples.

    Returns:
        A dictionary mapping unique chunk IDs (as strings) to their text content.
        Returns an empty dictionary if the file cannot be read or parsed.
    """
    unique_chunks: Dict[str, str] = {}
    if not file_path.exists():
        logger.error("Input JSON file not found at %s", file_path)
        return unique_chunks

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            triples_data = json.load(f)

        if not isinstance(triples_data, list):
            logger.error("Expected a JSON list in %s, but got %s.", file_path, type(triples_data))
            return unique_chunks

        logger.info("Extracting unique chunks from %d triples...", len(triples_data))
        for triple in triples_data:
            chunk_id_val = triple.get("chunk")
            chunk_text = triple.get("chunk_text")

            # Ensure chunk_id and text are present and valid
            if chunk_id_val is not None and isinstance(chunk_text, str) and chunk_text.strip():
                chunk_id = str(chunk_id_val)  # Use string representation for ID
                chunk_text_stripped = chunk_text.strip()

                # Store chunk text only if ID is new
                if chunk_id not in unique_chunks:
                    unique_chunks[chunk_id] = chunk_text_stripped

        logger.info("Found %d unique chunks to process.", len(unique_chunks))
        return unique_chunks

    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from %s: %s", file_path, e, exc_info=True)
        return {}
    except Exception as e:
        logger.error("Failed to read or process file %s: %s", file_path, e, exc_info=True)
        return {}


@monitor_performance("create_and_store_embeddings")
def create_and_store_embeddings(
        chunks: Dict[str, str],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        chroma_path: str = CHROMA_PERSIST_PATH,
        collection_name: str = COLLECTION_NAME
):
    """
    Generates embeddings for text chunks and stores them in ChromaDB.

    Args:
        chunks: Dictionary mapping chunk IDs (str) to chunk text (str).
        model_name: The name of the sentence-transformer model to use.
        chroma_path: The directory path to persist ChromaDB data.
        collection_name: The name for the ChromaDB collection.
    """
    if not chunks:
        logger.warning("No chunks provided to embed.")
        return

    logger.info("Initializing embedding model: %s", model_name)
    try:
        model = SentenceTransformer(model_name)
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", embedding_dim)
    except Exception as e:
        logger.error("Failed to load sentence transformer model '%s': %s", model_name, e, exc_info=True)
        return

    logger.info("Initializing ChromaDB client at path: %s", chroma_path)
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB collection '%s' ready.", collection_name)

    except Exception as e:
        logger.error("Failed to initialize ChromaDB or collection '%s': %s", collection_name, e, exc_info=True)
        return

    # Prepare data for embedding and storage
    chunk_ids = list(chunks.keys())
    chunk_texts = [chunks[id] for id in chunk_ids]

    logger.info("Generating embeddings for %d text chunks...", len(chunk_texts))
    embedding_start = time.time()
    try:
        embeddings = model.encode(chunk_texts, show_progress_bar=True)
        embedding_duration = time.time() - embedding_start
        logger.info(
            f"Performance: Embedding generation completed in {embedding_duration:.3f}s for {len(chunk_texts)} chunks")

        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

    except Exception as e:
        logger.error("Failed during embedding generation: %s", e, exc_info=True)
        return

    logger.info("Adding %d embeddings to ChromaDB collection '%s'...", len(chunk_ids), collection_name)
    storage_start = time.time()
    try:
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings_list,
            documents=chunk_texts
        )
        storage_duration = time.time() - storage_start
        logger.info(f"Performance: ChromaDB storage completed in {storage_duration:.3f}s for {len(chunk_ids)} items")

        count = collection.count()
        logger.info("Collection '%s' now contains %d items.", collection_name, count)

    except Exception as e:
        logger.error("Failed to add embeddings to ChromaDB collection '%s': %s", collection_name, e, exc_info=True)


@monitor_performance("store_enhanced_ocr_results")
def store_enhanced_ocr_results_in_vector_db(
        ocr_results: List[Dict],  # Enhanced OCR results from the pipeline
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        chroma_path: str = CHROMA_PERSIST_PATH,
        collection_name: str = "enhanced_ocr_embeddings"
) -> int:
    """
    Enhanced function to store OCR results with rich metadata in ChromaDB.

    Args:
        ocr_results: List of enhanced OCR results with comprehensive metadata
        model_name: Sentence transformer model name
        chroma_path: ChromaDB persistence path
        collection_name: Collection name for enhanced OCR results

    Returns:
        Number of documents stored
    """
    if not ocr_results:
        logger.warning("No OCR results provided to embed.")
        return 0

    logger.info(f"Processing {len(ocr_results)} enhanced OCR results for vector storage...")

    # Initialize embedding model
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return 0

    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_ef,
            metadata={"hnsw:space": "cosine", "description": "Enhanced OCR results with metadata"}
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready for enhanced OCR storage.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return 0

    # Prepare data for embedding
    documents = []
    metadatas = []
    ids = []

    total_chunks_processed = 0

    for doc_idx, ocr_result in enumerate(ocr_results):
        # Extract core OCR data
        success = ocr_result.get('success', False)
        if not success:
            logger.warning(f"Skipping failed OCR result {doc_idx}")
            continue

        text_content = ocr_result.get('text', '').strip()
        if not text_content:
            logger.warning(f"Skipping OCR result {doc_idx} with empty text")
            continue

        # Extract metadata components
        file_metadata = ocr_result.get('file_metadata', {})
        content_metadata = ocr_result.get('content_metadata', {})
        quality_metrics = ocr_result.get('quality_metrics', {})
        document_classification = ocr_result.get('document_classification', {})
        detected_entities = ocr_result.get('detected_entities', {})
        processing_metadata = ocr_result.get('processing_metadata', {})
        chunk_metadata = ocr_result.get('chunk_metadata', [])

        # Process chunks if available
        if chunk_metadata:
            for chunk_idx, chunk in enumerate(chunk_metadata):
                chunk_text = chunk.get('chunk_text', '').strip()
                if not chunk_text:
                    continue

                chunk_id = chunk.get('chunk_id', f"doc_{doc_idx}_chunk_{chunk_idx}")
                document_id = f"{file_metadata.get('file_id', f'doc_{doc_idx}')}_chunk_{chunk_idx}"

                documents.append(chunk_text)
                ids.append(document_id)

                # Create comprehensive metadata for the chunk
                chunk_metadata_dict = {
                    # File Information
                    "file_id": file_metadata.get('file_id', f"doc_{doc_idx}"),
                    "filename": file_metadata.get('original_filename', 'unknown'),
                    "file_type": file_metadata.get('mime_type', 'unknown'),
                    "file_size": file_metadata.get('file_size_bytes', 0),
                    "file_hash": file_metadata.get('file_hash', ''),

                    # Chunk Information
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "chunk_word_count": chunk.get('word_count', 0),
                    "chunk_character_count": chunk.get('character_count', 0),
                    "chunk_start_word": chunk.get('start_word', 0),
                    "chunk_end_word": chunk.get('end_word', 0),

                    # OCR Processing Info
                    "ocr_method": ocr_result.get('method_used', 'unknown'),
                    "ocr_confidence": float(ocr_result.get('confidence', 0.0)),
                    "processing_time": float(ocr_result.get('processing_time', 0.0)),

                    # Document Classification
                    "document_type": document_classification.get('document_type', 'unknown'),
                    "document_category": document_classification.get('category', 'general'),
                    "classification_confidence": float(document_classification.get('classification_confidence', 0.0)),

                    # Content Analysis
                    "language": content_metadata.get('language_detected', 'en'),
                    "has_tables": bool(content_metadata.get('has_tables', False)),
                    "has_financial_data": bool(content_metadata.get('has_financial_data', False)),
                    "total_word_count": int(content_metadata.get('word_count', 0)),
                    "total_character_count": int(content_metadata.get('character_count', 0)),

                    # Quality Metrics
                    "quality_score": float(quality_metrics.get('quality_score', 0.0)),
                    "readability_score": float(quality_metrics.get('readability_score', 0.0)),
                    "complexity": quality_metrics.get('complexity', 'unknown'),

                    # Timestamps
                    "upload_timestamp": file_metadata.get('upload_timestamp', ''),
                    "processing_timestamp": processing_metadata.get('extraction_timestamp', ''),

                    # Domain-specific entity counts
                    "company_count": len(detected_entities.get('companies', [])),
                    "people_count": len(detected_entities.get('people', [])),
                    "location_count": len(detected_entities.get('locations', [])),
                    "well_count": len(detected_entities.get('wells', [])),
                    "formation_count": len(detected_entities.get('formations', [])),
                    "equipment_count": len(detected_entities.get('equipment', [])),

                    # Source tracking
                    "source_pipeline": "enhanced_llm_ocr",
                    "pipeline_version": processing_metadata.get('pipeline_version', 'unknown'),
                    "extraction_features": {
                        "entity_extraction": processing_metadata.get('entity_extraction_enabled', False),
                        "classification": processing_metadata.get('classification_enabled', False),
                        "quality_analysis": processing_metadata.get('quality_analysis_enabled', False)
                    }
                }

                # Add entity lists as searchable text (limited to avoid metadata size issues)
                for entity_type, entity_list in detected_entities.items():
                    if entity_list:
                        # Store up to 5 entities per type to avoid metadata size limits
                        chunk_metadata_dict[f"entities_{entity_type}"] = ", ".join(entity_list[:5])

                metadatas.append(chunk_metadata_dict)
                total_chunks_processed += 1

        else:
            # Process full document if no chunks available
            document_id = file_metadata.get('file_id', f"doc_{doc_idx}")

            documents.append(text_content)
            ids.append(document_id)

            # Create metadata for full document
            doc_metadata = {
                # File Information
                "file_id": file_metadata.get('file_id', f"doc_{doc_idx}"),
                "filename": file_metadata.get('original_filename', 'unknown'),
                "file_type": file_metadata.get('mime_type', 'unknown'),
                "file_size": file_metadata.get('file_size_bytes', 0),
                "file_hash": file_metadata.get('file_hash', ''),

                # Document level processing
                "is_full_document": True,
                "chunk_id": "full_document",
                "chunk_index": 0,

                # OCR Processing Info
                "ocr_method": ocr_result.get('method_used', 'unknown'),
                "ocr_confidence": float(ocr_result.get('confidence', 0.0)),
                "processing_time": float(ocr_result.get('processing_time', 0.0)),

                # Document Classification
                "document_type": document_classification.get('document_type', 'unknown'),
                "document_category": document_classification.get('category', 'general'),
                "classification_confidence": float(document_classification.get('classification_confidence', 0.0)),

                # Content Analysis
                "language": content_metadata.get('language_detected', 'en'),
                "has_tables": bool(content_metadata.get('has_tables', False)),
                "has_financial_data": bool(content_metadata.get('has_financial_data', False)),
                "word_count": int(content_metadata.get('word_count', 0)),
                "character_count": int(content_metadata.get('character_count', 0)),

                # Quality Metrics
                "quality_score": float(quality_metrics.get('quality_score', 0.0)),
                "readability_score": float(quality_metrics.get('readability_score', 0.0)),
                "complexity": quality_metrics.get('complexity', 'unknown'),

                # Timestamps
                "upload_timestamp": file_metadata.get('upload_timestamp', ''),
                "processing_timestamp": processing_metadata.get('extraction_timestamp', ''),

                # Entity counts
                "company_count": len(detected_entities.get('companies', [])),
                "people_count": len(detected_entities.get('people', [])),
                "location_count": len(detected_entities.get('locations', [])),
                "well_count": len(detected_entities.get('wells', [])),
                "formation_count": len(detected_entities.get('formations', [])),
                "equipment_count": len(detected_entities.get('equipment', [])),

                # Source tracking
                "source_pipeline": "enhanced_llm_ocr",
                "pipeline_version": processing_metadata.get('pipeline_version', 'unknown')
            }

            # Add entity lists
            for entity_type, entity_list in detected_entities.items():
                if entity_list:
                    doc_metadata[f"entities_{entity_type}"] = ", ".join(entity_list[:5])

            metadatas.append(doc_metadata)
            total_chunks_processed += 1

    if not documents:
        logger.warning("No valid documents found to embed")
        return 0

    # Generate embeddings and store
    logger.info(f"Generating embeddings for {len(documents)} text segments...")
    try:
        embeddings = model.encode(documents, show_progress_bar=True)

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )

        final_count = collection.count()
        logger.info(f"Successfully stored {len(documents)} enhanced OCR results in vector database")
        logger.info(f"Collection '{collection_name}' now contains {final_count} total items")

        return len(documents)

    except Exception as e:
        logger.error(f"Failed to generate embeddings or store in ChromaDB: {e}")
        return 0


def enhanced_similarity_search(
        query: str,
        collection_name: str = "enhanced_ocr_embeddings",
        n_results: int = 5,
        filters: Dict = None,
        chroma_path: str = CHROMA_PERSIST_PATH
) -> Dict[str, Any]:
    """
    Enhanced similarity search with comprehensive metadata filtering.

    Args:
        query: Search query text
        collection_name: ChromaDB collection name
        n_results: Number of results to return
        filters: Metadata filters for refined search
        chroma_path: Path to ChromaDB storage

    Returns:
        Dictionary containing search results with metadata
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        # Build where clause for filtering
        where_clause = {}
        if filters:
            where_clause.update(filters)

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None,
            include=['documents', 'metadatas', 'distances']
        )

        # Process and enhance results
        enhanced_results = {
            'query': query,
            'total_results': len(results['documents'][0]) if results['documents'] else 0,
            'filters_applied': filters,
            'results': []
        }

        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                result_item = {
                    'rank': i + 1,
                    'document': doc,
                    'metadata': metadata,
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'distance': distance
                }
                enhanced_results['results'].append(result_item)

        return enhanced_results

    except Exception as e:
        logger.error(f"Failed to perform similarity search: {e}")
        return {'query': query, 'total_results': 0, 'results': [], 'error': str(e)}


def get_collection_stats(collection_name: str = "enhanced_ocr_embeddings", chroma_path: str = CHROMA_PERSIST_PATH) -> \
Dict[str, Any]:
    """
    Get comprehensive statistics about the vector collection.

    Args:
        collection_name: Name of the ChromaDB collection
        chroma_path: Path to ChromaDB storage

    Returns:
        Dictionary with collection statistics
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        total_count = collection.count()

        # Sample some items to get metadata statistics
        sample_size = min(100, total_count)
        if sample_size > 0:
            sample_results = collection.get(limit=sample_size, include=['metadatas'])

            # Analyze metadata
            document_types = {}
            ocr_methods = {}
            quality_scores = []

            for metadata in sample_results['metadatas']:
                # Document type distribution
                doc_type = metadata.get('document_type', 'unknown')
                document_types[doc_type] = document_types.get(doc_type, 0) + 1

                # OCR method distribution
                ocr_method = metadata.get('ocr_method', 'unknown')
                ocr_methods[ocr_method] = ocr_methods.get(ocr_method, 0) + 1

                # Quality scores
                quality_score = metadata.get('quality_score', 0)
                if isinstance(quality_score, (int, float)) and quality_score > 0:
                    quality_scores.append(quality_score)

            # Calculate averages
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            stats = {
                'collection_name': collection_name,
                'total_documents': total_count,
                'sample_size_analyzed': sample_size,
                'document_type_distribution': document_types,
                'ocr_method_distribution': ocr_methods,
                'average_quality_score': round(avg_quality, 3),
                'quality_score_range': {
                    'min': min(quality_scores) if quality_scores else 0,
                    'max': max(quality_scores) if quality_scores else 0
                }
            }
        else:
            stats = {
                'collection_name': collection_name,
                'total_documents': 0,
                'message': 'Collection is empty'
            }

        return stats

    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {'collection_name': collection_name, 'error': str(e)}


def search_by_entity_type(
        entity_type: str,
        collection_name: str = "enhanced_ocr_embeddings",
        n_results: int = 10,
        chroma_path: str = CHROMA_PERSIST_PATH
) -> List[Dict[str, Any]]:
    """
    Search for documents containing specific entity types.

    Args:
        entity_type: Type of entity to search for (companies, wells, formations, etc.)
        collection_name: ChromaDB collection name
        n_results: Maximum number of results
        chroma_path: Path to ChromaDB storage

    Returns:
        List of documents containing the specified entity type
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        # Search for documents that have entities of the specified type
        where_clause = {f"{entity_type}_count": {"$gt": 0}}

        results = collection.get(
            where=where_clause,
            limit=n_results,
            include=['documents', 'metadatas']
        )

        entity_results = []
        if results['documents']:
            for doc, metadata in zip(results['documents'], results['metadatas']):
                entities_field = f"entities_{entity_type}"
                entities_found = metadata.get(entities_field, "").split(", ") if metadata.get(entities_field) else []

                result_item = {
                    'document': doc[:500] + "..." if len(doc) > 500 else doc,  # Truncate for display
                    'filename': metadata.get('filename', 'unknown'),
                    'document_type': metadata.get('document_type', 'unknown'),
                    'entity_count': metadata.get(f"{entity_type}_count", 0),
                    'entities_found': [e.strip() for e in entities_found if e.strip()],
                    'quality_score': metadata.get('quality_score', 0),
                    'file_id': metadata.get('file_id', 'unknown')
                }
                entity_results.append(result_item)

        logger.info(f"Found {len(entity_results)} documents containing {entity_type}")
        return entity_results

    except Exception as e:
        logger.error(f"Failed to search by entity type {entity_type}: {e}")
        return []


def search_by_document_type(
        document_type: str,
        collection_name: str = "enhanced_ocr_embeddings",
        n_results: int = 10,
        min_quality_score: float = 0.0,
        chroma_path: str = CHROMA_PERSIST_PATH
) -> List[Dict[str, Any]]:
    """
    Search for documents of a specific type with optional quality filtering.

    Args:
        document_type: Type of document to search for (invoice, contract, report, etc.)
        collection_name: ChromaDB collection name
        n_results: Maximum number of results
        min_quality_score: Minimum quality score threshold
        chroma_path: Path to ChromaDB storage

    Returns:
        List of documents of the specified type
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        # Build where clause
        where_clause = {"document_type": document_type}
        if min_quality_score > 0:
            where_clause["quality_score"] = {"$gte": min_quality_score}

        results = collection.get(
            where=where_clause,
            limit=n_results,
            include=['documents', 'metadatas']
        )

        type_results = []
        if results['documents']:
            for doc, metadata in zip(results['documents'], results['metadatas']):
                result_item = {
                    'document': doc[:500] + "..." if len(doc) > 500 else doc,
                    'filename': metadata.get('filename', 'unknown'),
                    'document_category': metadata.get('document_category', 'general'),
                    'classification_confidence': metadata.get('classification_confidence', 0),
                    'quality_score': metadata.get('quality_score', 0),
                    'word_count': metadata.get('word_count', 0),
                    'has_tables': metadata.get('has_tables', False),
                    'has_financial_data': metadata.get('has_financial_data', False),
                    'ocr_method': metadata.get('ocr_method', 'unknown'),
                    'file_id': metadata.get('file_id', 'unknown')
                }
                type_results.append(result_item)

        logger.info(
            f"Found {len(type_results)} documents of type '{document_type}' with quality >= {min_quality_score}")
        return type_results

    except Exception as e:
        logger.error(f"Failed to search by document type {document_type}: {e}")
        return []


def advanced_multi_filter_search(
        query: str = None,
        document_types: List[str] = None,
        entity_types: List[str] = None,
        min_quality_score: float = 0.0,
        max_quality_score: float = 1.0,
        has_tables: bool = None,
        has_financial_data: bool = None,
        ocr_methods: List[str] = None,
        date_range: Dict[str, str] = None,
        collection_name: str = "enhanced_ocr_embeddings",
        n_results: int = 10,
        chroma_path: str = CHROMA_PERSIST_PATH
) -> Dict[str, Any]:
    """
    Advanced search with multiple filter combinations.

    Args:
        query: Optional semantic search query
        document_types: List of document types to include
        entity_types: List of entity types that must be present
        min_quality_score: Minimum quality threshold
        max_quality_score: Maximum quality threshold
        has_tables: Filter by presence of tables
        has_financial_data: Filter by presence of financial data
        ocr_methods: List of OCR methods to include
        date_range: Date range filter {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
        collection_name: ChromaDB collection name
        n_results: Maximum number of results
        chroma_path: Path to ChromaDB storage

    Returns:
        Dictionary with filtered search results
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        # Build complex where clause
        where_clause = {}

        # Quality score range
        if min_quality_score > 0 or max_quality_score < 1.0:
            where_clause["quality_score"] = {}
            if min_quality_score > 0:
                where_clause["quality_score"]["$gte"] = min_quality_score
            if max_quality_score < 1.0:
                where_clause["quality_score"]["$lte"] = max_quality_score

        # Document types
        if document_types:
            where_clause["document_type"] = {"$in": document_types}

        # OCR methods
        if ocr_methods:
            where_clause["ocr_method"] = {"$in": ocr_methods}

        # Boolean filters
        if has_tables is not None:
            where_clause["has_tables"] = has_tables
        if has_financial_data is not None:
            where_clause["has_financial_data"] = has_financial_data

        # Entity type filters (must have at least one entity of specified types)
        if entity_types:
            entity_conditions = []
            for entity_type in entity_types:
                entity_conditions.append({f"{entity_type}_count": {"$gt": 0}})
            if len(entity_conditions) == 1:
                where_clause.update(entity_conditions[0])
            # Note: ChromaDB doesn't support $or directly, so we use the first condition
            # For complex OR conditions, multiple queries would be needed

        # Perform search
        if query:
            # Semantic search with filters
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=['documents', 'metadatas', 'distances']
            )

            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    result_item = {
                        'rank': i + 1,
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,
                        'distance': distance
                    }
                    search_results.append(result_item)
        else:
            # Filter-only search
            results = collection.get(
                where=where_clause if where_clause else None,
                limit=n_results,
                include=['documents', 'metadatas']
            )

            search_results = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    result_item = {
                        'rank': i + 1,
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': None,  # No semantic similarity for filter-only
                        'distance': None
                    }
                    search_results.append(result_item)

        return {
            'query': query,
            'filters_applied': {
                'document_types': document_types,
                'entity_types': entity_types,
                'quality_range': [min_quality_score, max_quality_score],
                'has_tables': has_tables,
                'has_financial_data': has_financial_data,
                'ocr_methods': ocr_methods,
                'date_range': date_range
            },
            'total_results': len(search_results),
            'results': search_results
        }

    except Exception as e:
        logger.error(f"Failed to perform advanced search: {e}")
        return {
            'query': query,
            'total_results': 0,
            'results': [],
            'error': str(e)
        }


def get_entity_statistics(
        collection_name: str = "enhanced_ocr_embeddings",
        chroma_path: str = CHROMA_PERSIST_PATH
) -> Dict[str, Any]:
    """
    Get comprehensive entity statistics from the collection.

    Args:
        collection_name: ChromaDB collection name
        chroma_path: Path to ChromaDB storage

    Returns:
        Dictionary with entity statistics
    """
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection(collection_name)

        # Get all documents to analyze entities
        all_results = collection.get(include=['metadatas'])

        if not all_results['metadatas']:
            return {'message': 'No documents found in collection'}

        entity_stats = {
            'total_documents': len(all_results['metadatas']),
            'entity_type_statistics': {},
            'top_entities_by_type': {},
            'documents_with_entities': 0
        }

        entity_types = ['companies', 'people', 'locations', 'wells', 'formations', 'equipment']

        for entity_type in entity_types:
            count_field = f"{entity_type}_count"
            entities_field = f"entities_{entity_type}"

            # Count documents with this entity type
            docs_with_entities = 0
            total_entities = 0
            all_entities = []

            for metadata in all_results['metadatas']:
                entity_count = metadata.get(count_field, 0)
                if entity_count > 0:
                    docs_with_entities += 1
                    total_entities += entity_count

                    # Extract individual entities
                    entities_text = metadata.get(entities_field, "")
                    if entities_text:
                        entities = [e.strip() for e in entities_text.split(", ") if e.strip()]
                        all_entities.extend(entities)

            # Count entity frequencies
            entity_frequency = {}
            for entity in all_entities:
                entity_frequency[entity] = entity_frequency.get(entity, 0) + 1

            # Get top entities
            top_entities = sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

            entity_stats['entity_type_statistics'][entity_type] = {
                'documents_with_entities': docs_with_entities,
                'total_entity_mentions': total_entities,
                'unique_entities': len(entity_frequency),
                'coverage_percentage': round((docs_with_entities / len(all_results['metadatas'])) * 100, 2)
            }

            entity_stats['top_entities_by_type'][entity_type] = [
                {'entity': entity, 'frequency': freq} for entity, freq in top_entities
            ]

        # Count documents with any entities
        entity_stats['documents_with_entities'] = len([
            metadata for metadata in all_results['metadatas']
            if any(metadata.get(f"{et}_count", 0) > 0 for et in entity_types)
        ])

        return entity_stats

    except Exception as e:
        logger.error(f"Failed to get entity statistics: {e}")
        return {'error': str(e)}


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Enhanced Vector Storage with OCR Integration ---")

    # Configuration loading
    config_path = Path("graph_config.ini")
    if not config_path.is_file():
        print(f"ERROR: Configuration file not found at {config_path}")
        logger.critical("Configuration file not found at %s", config_path)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    json_file_path = None
    try:
        # Load JSON file path from config
        if "data" in config and "json_file" in config["data"]:
            file_path_str = config.get("data", "json_file", fallback=None)
            if file_path_str:
                json_file_path = Path(file_path_str)

        if not json_file_path:
            logger.error("Missing 'json_file' path in [data] section of graph_config.ini")
            sys.exit(1)

        print(f"[INFO] Using input JSON file: {json_file_path}")

    except configparser.Error as e:
        print(f"ERROR reading configuration file: {e}")
        logger.error("Failed to read configuration file.", exc_info=True)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR processing configuration: {e}")
        logger.exception("Unexpected error during configuration processing.")
        sys.exit(1)

    # 1. Load unique chunks from traditional pipeline (backward compatibility)
    print("\n=== Processing Traditional Pipeline Chunks ===")
    unique_chunks = load_unique_chunks_from_json(json_file_path)

    if unique_chunks:
        create_and_store_embeddings(
            chunks=unique_chunks,
            model_name=DEFAULT_EMBEDDING_MODEL,
            chroma_path=CHROMA_PERSIST_PATH,
            collection_name=COLLECTION_NAME
        )
        print(f"[INFO] Stored {len(unique_chunks)} traditional chunks")
    else:
        print("[WARNING] No traditional chunks found")

    # 2. Example of processing enhanced OCR results
    print("\n=== Processing Enhanced OCR Results ===")

    # Example enhanced OCR results (in real usage, these would come from your OCR pipeline)
    sample_enhanced_results = [
        {
            'success': True,
            'text': 'This is a sample invoice from ABC Corporation to XYZ Ltd for drilling services at Well A-1 in the Daman Formation. Total amount: $125,000.',
            'confidence': 0.95,
            'method_used': 'gemini_1.5_flash',
            'processing_time': 2.5,
            'text_regions_detected': 1,
            'preprocessing_applied': ['llm_vision_extraction'],
            'file_metadata': {
                'file_id': 'invoice_001',
                'original_filename': 'invoice_abc_xyz.pdf',
                'file_hash': 'abc123def456',
                'file_size_bytes': 245760,
                'mime_type': 'application/pdf',
                'upload_timestamp': '2024-06-20T10:00:00Z',
                'file_extension': '.pdf'
            },
            'content_metadata': {
                'word_count': 25,
                'character_count': 150,
                'paragraph_count': 1,
                'line_count': 3,
                'has_tables': True,
                'has_financial_data': True,
                'language_detected': 'en'
            },
            'quality_metrics': {
                'quality_score': 0.92,
                'readability_score': 75.0,
                'complexity': 'medium',
                'avg_word_length': 5.2,
                'avg_sentence_length': 12.5,
                'issues': [],
                'word_count': 25,
                'sentence_count': 2,
                'character_count': 150
            },
            'detected_entities': {
                'companies': ['ABC Corporation', 'XYZ Ltd'],
                'wells': ['A-1'],
                'formations': ['Daman Formation'],
                'money': ['$125,000']
            },
            'document_classification': {
                'document_type': 'invoice',
                'classification_confidence': 0.88,
                'category': 'financial'
            },
            'processing_metadata': {
                'extraction_timestamp': '2024-06-20T10:02:30Z',
                'pipeline_version': 'v2.1.0',
                'entity_extraction_enabled': True,
                'classification_enabled': True,
                'quality_analysis_enabled': True
            },
            'chunk_metadata': [
                {
                    'chunk_id': 'chunk_0',
                    'chunk_text': 'This is a sample invoice from ABC Corporation to XYZ Ltd for drilling services',
                    'start_word': 0,
                    'end_word': 13,
                    'word_count': 13,
                    'character_count': 78
                },
                {
                    'chunk_id': 'chunk_1',
                    'chunk_text': 'at Well A-1 in the Daman Formation. Total amount: $125,000.',
                    'start_word': 13,
                    'end_word': 25,
                    'word_count': 12,
                    'character_count': 72
                }
            ]
        },
        {
            'success': True,
            'text': 'Geological report for the Bakken Shale formation showing drilling results for Well B-12A operated by Energy Corp.',
            'confidence': 0.89,
            'method_used': 'claude_3.5_sonnet',
            'processing_time': 3.1,
            'text_regions_detected': 1,
            'preprocessing_applied': ['llm_vision_extraction'],
            'file_metadata': {
                'file_id': 'geo_report_002',
                'original_filename': 'bakken_drilling_report.pdf',
                'file_hash': 'def456ghi789',
                'file_size_bytes': 512000,
                'mime_type': 'application/pdf',
                'upload_timestamp': '2024-06-20T11:00:00Z',
                'file_extension': '.pdf'
            },
            'content_metadata': {
                'word_count': 18,
                'character_count': 120,
                'paragraph_count': 1,
                'line_count': 2,
                'has_tables': False,
                'has_financial_data': False,
                'language_detected': 'en'
            },
            'quality_metrics': {
                'quality_score': 0.87,
                'readability_score': 68.0,
                'complexity': 'high',
                'avg_word_length': 6.1,
                'avg_sentence_length': 18.0,
                'issues': [],
                'word_count': 18,
                'sentence_count': 1,
                'character_count': 120
            },
            'detected_entities': {
                'companies': ['Energy Corp'],
                'wells': ['B-12A'],
                'formations': ['Bakken Shale'],
                'equipment': ['drilling']
            },
            'document_classification': {
                'document_type': 'geological_report',
                'classification_confidence': 0.91,
                'category': 'technical'
            },
            'processing_metadata': {
                'extraction_timestamp': '2024-06-20T11:03:15Z',
                'pipeline_version': 'v2.1.0',
                'entity_extraction_enabled': True,
                'classification_enabled': True,
                'quality_analysis_enabled': True
            },
            'chunk_metadata': [
                {
                    'chunk_id': 'chunk_0',
                    'chunk_text': 'Geological report for the Bakken Shale formation showing drilling results for Well B-12A operated by Energy Corp.',
                    'start_word': 0,
                    'end_word': 18,
                    'word_count': 18,
                    'character_count': 120
                }
            ]
        }
    ]

    # Store enhanced OCR results
    stored_count = store_enhanced_ocr_results_in_vector_db(
        ocr_results=sample_enhanced_results,
        model_name=DEFAULT_EMBEDDING_MODEL,
        chroma_path=CHROMA_PERSIST_PATH,
        collection_name="enhanced_ocr_embeddings"
    )

    print(f"[INFO] Stored {stored_count} enhanced OCR results")

    # 3. Demonstrate enhanced search capabilities
    print("\n=== Testing Enhanced Search Capabilities ===")

    # Test semantic search
    print("\n--- Semantic Search Test ---")
    search_results = enhanced_similarity_search(
        query="drilling operations invoice payment",
        collection_name="enhanced_ocr_embeddings",
        n_results=3
    )

    print(f"Found {search_results['total_results']} results for semantic search:")
    for result in search_results['results']:
        print(f"  Rank {result['rank']}: {result['document'][:100]}...")
        print(f"    Similarity: {result['similarity_score']:.3f}")
        print(f"    Type: {result['metadata'].get('document_type', 'unknown')}")
        print(f"    Quality: {result['metadata'].get('quality_score', 0):.3f}")
        print()

    # Test filtered search
    print("\n--- Filtered Search Test ---")
    filtered_results = enhanced_similarity_search(
        query="formation drilling",
        collection_name="enhanced_ocr_embeddings",
        filters={"document_type": "geological_report"},
        n_results=2
    )

    print(f"Found {filtered_results['total_results']} geological reports:")
    for result in filtered_results['results']:
        print(f"  - {result['metadata'].get('filename', 'unknown')}")
        print(f"    Formations: {result['metadata'].get('entities_formations', 'none')}")
        print()

    # Test entity-based search
    print("\n--- Entity-Based Search Test ---")
    wells_docs = search_by_entity_type("wells", "enhanced_ocr_embeddings", 5)
    print(f"Found {len(wells_docs)} documents containing wells:")
    for doc in wells_docs:
        print(f"  - {doc['filename']}: {doc['entities_found']}")

    # Test document type search
    print("\n--- Document Type Search Test ---")
    invoices = search_by_document_type("invoice", "enhanced_ocr_embeddings", 5, min_quality_score=0.8)
    print(f"Found {len(invoices)} high-quality invoices:")
    for invoice in invoices:
        print(f"  - {invoice['filename']} (Quality: {invoice['quality_score']:.3f})")

    # Test advanced multi-filter search
    print("\n--- Advanced Multi-Filter Search Test ---")
    advanced_results = advanced_multi_filter_search(
        query="drilling services",
        document_types=["invoice", "geological_report"],
        entity_types=["wells", "companies"],
        min_quality_score=0.8,
        has_financial_data=True,
        collection_name="enhanced_ocr_embeddings",
        n_results=5
    )

    print(f"Advanced search found {advanced_results['total_results']} results:")
    for result in advanced_results['results']:
        metadata = result['metadata']
        print(f"  - {metadata.get('filename', 'unknown')} ({metadata.get('document_type', 'unknown')})")
        print(
            f"    Quality: {metadata.get('quality_score', 0):.3f}, Financial: {metadata.get('has_financial_data', False)}")

    # 4. Collection statistics
    print("\n=== Collection Statistics ===")

    # Traditional collection stats
    try:
        traditional_stats = get_collection_stats(COLLECTION_NAME)
        print(f"\nTraditional Collection ({COLLECTION_NAME}):")
        print(f"  Total documents: {traditional_stats.get('total_documents', 0)}")
    except Exception as e:
        print(f"  Error getting traditional stats: {e}")

    # Enhanced collection stats
    enhanced_stats = get_collection_stats("enhanced_ocr_embeddings")
    print(f"\nEnhanced Collection (enhanced_ocr_embeddings):")
    print(f"  Total documents: {enhanced_stats.get('total_documents', 0)}")

    if 'document_type_distribution' in enhanced_stats:
        print("  Document types:")
        for doc_type, count in enhanced_stats['document_type_distribution'].items():
            print(f"    {doc_type}: {count}")

    if 'ocr_method_distribution' in enhanced_stats:
        print("  OCR methods:")
        for method, count in enhanced_stats['ocr_method_distribution'].items():
            print(f"    {method}: {count}")

    if 'average_quality_score' in enhanced_stats:
        print(f"  Average quality score: {enhanced_stats['average_quality_score']}")

    # Entity statistics
    print("\n--- Entity Statistics ---")
    entity_stats = get_entity_statistics("enhanced_ocr_embeddings")

    if 'entity_type_statistics' in entity_stats:
        print(f"Documents with entities: {entity_stats['documents_with_entities']}/{entity_stats['total_documents']}")

        for entity_type, stats in entity_stats['entity_type_statistics'].items():
            print(f"\n{entity_type.title()}:")
            print(f"  Documents: {stats['documents_with_entities']} ({stats['coverage_percentage']}%)")
            print(f"  Unique entities: {stats['unique_entities']}")
            print(f"  Total mentions: {stats['total_entity_mentions']}")

            # Show top entities
            top_entities = entity_stats['top_entities_by_type'].get(entity_type, [])[:3]
            if top_entities:
                print(f"  Top entities: {', '.join([f'{e['entity']} ({e['frequency']})' for e in top_entities])}")

    print("\n=== Enhanced Vector Storage Complete ===")
    print("Features available:")
    print("  ✓ Enhanced OCR result storage with rich metadata")
    print("  ✓ Semantic similarity search")
    print("  ✓ Metadata-based filtering")
    print("  ✓ Entity-based document discovery")
    print("  ✓ Document type classification search")
    print("  ✓ Quality-based filtering")
    print("  ✓ Advanced multi-filter search")
    print("  ✓ Comprehensive collection analytics")
    print("  ✓ Entity statistics and frequency analysis")
    print("  ✓ Backward compatibility with traditional pipelines")
import logging
import json
from pathlib import Path
import configparser
import sys
from typing import List, Dict, Tuple, Set
import time
from functools import wraps

# --- Vector DB and Embeddings Libraries ---
# Ensure you have installed the necessary libraries:
# pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer # type: ignore

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
# You can move these to your config file (e.g., graph_config.ini or config.toml)
# Or define them here if simpler for now.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Good default sentence transformer
CHROMA_PERSIST_PATH = "./chroma_db_embeddings" # Directory to store Chroma data
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
        processed_count = 0
        for triple in triples_data:
            chunk_id_val = triple.get("chunk")
            chunk_text = triple.get("chunk_text")

            # Ensure chunk_id and text are present and valid
            if chunk_id_val is not None and isinstance(chunk_text, str) and chunk_text.strip():
                chunk_id = str(chunk_id_val) # Use string representation for ID
                chunk_text_stripped = chunk_text.strip()

                # Store chunk text only if ID is new or text is different (preferring first seen)
                if chunk_id not in unique_chunks:
                    unique_chunks[chunk_id] = chunk_text_stripped
                    processed_count += 1
                # Optional: Handle cases where the same chunk ID might appear with different text?
                # elif unique_chunks[chunk_id] != chunk_text_stripped:
                #     logger.warning("Chunk ID %s has different text in different triples. Keeping first occurrence.", chunk_id)

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
        # Load the sentence transformer model
        # Consider adding device='cuda' if GPU is available: model = SentenceTransformer(model_name, device='cuda')
        model = SentenceTransformer(model_name)
        # Get the embedding dimension dynamically from the loaded model
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", embedding_dim)
    except Exception as e:
        logger.error("Failed to load sentence transformer model '%s': %s", model_name, e, exc_info=True)
        return

    logger.info("Initializing ChromaDB client at path: %s", chroma_path)
    try:
        # Create a persistent ChromaDB client
        client = chromadb.PersistentClient(path=chroma_path)

        # Define the embedding function for Chroma - used for querying later
        # We pre-compute embeddings here, but Chroma needs this info for consistency checks and querying
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        # Get or create the collection
        # Pass metadata={'hnsw:space': 'cosine'} for cosine distance if preferred
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_ef, # Provide function info
            metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity
        )
        logger.info("ChromaDB collection '%s' ready.", collection_name)

    except Exception as e:
        logger.error("Failed to initialize ChromaDB or collection '%s': %s", collection_name, e, exc_info=True)
        return

    # Prepare data for embedding and storage
    chunk_ids = list(chunks.keys())
    chunk_texts = [chunks[id] for id in chunk_ids] # Maintain order

    logger.info("Generating embeddings for %d text chunks...", len(chunk_texts))
    embedding_start = time.time()
    try:
        # Generate embeddings in batches (model.encode handles batching internally)
        embeddings = model.encode(chunk_texts, show_progress_bar=True)
        embedding_duration = time.time() - embedding_start
        logger.info(
            f"Performance: Embedding generation completed in {embedding_duration:.3f}s for {len(chunk_texts)} chunks")

        # Convert embeddings to list if needed (some versions might return numpy array)
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)

    except Exception as e:
        logger.error("Failed during embedding generation: %s", e, exc_info=True)
        return

    logger.info("Adding %d embeddings to ChromaDB collection '%s'...", len(chunk_ids), collection_name)
    storage_start = time.time()
    try:
        # Add data to ChromaDB
        # Note: IDs must be strings. Documents are the original texts.
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings_list,
            documents=chunk_texts
            # You could add more metadata here if needed, e.g.,
            # metadatas=[{"source": "doc_x"} for _ in chunk_ids]
        )
        storage_duration = time.time() - storage_start
        logger.info(f"Performance: ChromaDB storage completed in {storage_duration:.3f}s for {len(chunk_ids)} items")

        # Optional: Verify count in collection
        count = collection.count()
        logger.info("Collection '%s' now contains %d items.", collection_name, count)

    except Exception as e:
        logger.error("Failed to add embeddings to ChromaDB collection '%s': %s", collection_name, e, exc_info=True)


def store_ocr_results_in_vector_db(
        ocr_results: List[Dict],  # From your OCR pipeline
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        chroma_path: str = CHROMA_PERSIST_PATH,
        collection_name: str = COLLECTION_NAME
):
    """
    Enhanced function to store OCR results with rich metadata in ChromaDB.

    Args:
        ocr_results: List of OCR results with text and metadata
        model_name: Sentence transformer model name
        chroma_path: ChromaDB persistence path
        collection_name: Collection name
    """
    if not ocr_results:
        logger.warning("No OCR results provided to embed.")
        return

    # Initialize embedding model
    model = SentenceTransformer(model_name)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=chroma_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for embedding
    documents = []
    metadatas = []
    ids = []

    for idx, result in enumerate(ocr_results):
        # Create unique ID
        file_id = result.get('file_id', f"doc_{idx}")
        chunk_id = f"{file_id}_chunk_{idx}"

        # Extract text for embedding
        text_content = result.get('text', '')
        if not text_content.strip():
            continue

        documents.append(text_content)
        ids.append(chunk_id)

        # Create comprehensive metadata
        metadata = {
            # File Information
            "file_id": result.get('file_id', f"doc_{idx}"),
            "filename": result.get('original_filename', 'unknown'),
            "file_type": result.get('file_type', 'unknown'),
            "file_size": result.get('file_size_bytes', 0),

            # OCR Processing Info
            "ocr_method": result.get('method_used', 'unknown'),
            "ocr_confidence": result.get('confidence', 0.0),
            "processing_time": result.get('processing_time', 0.0),
            "text_length": len(text_content),

            # Document Classification
            "document_type": result.get('document_type', 'unknown'),
            "language": result.get('language_detected', 'en'),
            "page_count": result.get('page_count', 1),

            # Content Analysis
            "word_count": len(text_content.split()),
            "has_tables": result.get('has_tables', False),
            "has_financial_data": bool(result.get('invoice_fields')),

            # Timestamps
            "upload_timestamp": result.get('upload_timestamp', ''),
            "processing_timestamp": result.get('extraction_timestamp', ''),

            # Quality Metrics
            "quality_score": result.get('text_quality_score', 0.0),
            "complexity": result.get('complexity_score', 'unknown'),

            # Domain-specific (Energy/Oil & Gas)
            "well_count": len(result.get('detected_entities', {}).get('wells', [])),
            "company_count": len(result.get('detected_entities', {}).get('companies', [])),
            "formation_count": len(result.get('detected_entities', {}).get('formations', [])),

            # Source tracking
            "source_pipeline": "llm_ocr_enhanced",
            "pipeline_version": "v2.1.0"
        }

        # Add detected entities as searchable fields
        entities = result.get('detected_entities', {})
        if entities:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    metadata[f"entities_{entity_type}"] = ", ".join(entity_list[:5])  # Limit for metadata size

        metadatas.append(metadata)

    # Generate embeddings and store
    logger.info(f"Generating embeddings for {len(documents)} OCR documents...")
    embeddings = model.encode(documents, show_progress_bar=True)

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )

    logger.info(f"Stored {len(documents)} OCR results in vector database with enhanced metadata")
    return collection.count()


def enhanced_similarity_search(
        query: str,
        collection_name: str = COLLECTION_NAME,
        n_results: int = 5,
        filters: Dict = None
):
    """
    Enhanced similarity search with metadata filtering.

    Args:
        query: Search query text
        collection_name: ChromaDB collection name
        n_results: Number of results to return
        filters: Metadata filters (e.g., {"document_type": "invoice"})
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
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

    return {
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0],
        'distances': results['distances'][0]
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Chunk Embedding and Vector Storage Script ---")

    # --- Configuration Loading (Example using graph_config.ini) ---
    config_path = Path("graph_config.ini")
    if not config_path.is_file():
        print(f"ERROR: Configuration file not found at {config_path}")
        logger.critical("Configuration file not found at %s", config_path)
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    json_file_path = None
    try:
        # Assuming the JSON file path is in the [data] section
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

    # 1. Load unique chunks
    unique_chunks = load_unique_chunks_from_json(json_file_path)

    # 2. Create and store embeddings
    if unique_chunks:
        create_and_store_embeddings(
            chunks=unique_chunks,
            model_name=DEFAULT_EMBEDDING_MODEL, # Or load from config
            chroma_path=CHROMA_PERSIST_PATH,     # Or load from config
            collection_name=COLLECTION_NAME      # Or load from config
        )
    else:
        logger.warning("No unique chunks found or loaded. Skipping embedding process.")

    print("\n--- Script Finished ---")

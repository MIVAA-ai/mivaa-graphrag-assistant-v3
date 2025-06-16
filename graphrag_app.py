# graphrag_app.py (Main App Entry Point - FINALIZED)

import nest_asyncio
nest_asyncio.apply() # Apply early for asyncio compatibility

import streamlit as st
import os
import logging
import sys
from pathlib import Path
import spacy
import configparser
import requests # Keep for get_requests_session
import asyncio # Keep for running async QA call if needed anywhere else? (Likely only needed in chat page)
from typing import Dict, Optional, Any # Keep necessary base types
st.set_page_config(
        page_title="GraphRAG Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded" # Keep sidebar open initially
    )
from enhanced_ocr_pipeline import EnhancedOCRPipeline
# Use tomli if available (standardized as tomllib)
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        logging.critical("No TOML parser found. Please install 'tomli' or use Python 3.11+.")
        # Avoid direct st calls at module level if possible
        print("FATAL: No TOML parser found.", file=sys.stderr)
        sys.exit(1)

# --- Core Logic Imports for Resource Initialization ---
try:
    from neo4j_exporter import Neo4jExporter
    from graph_rag_qa import GraphRAGQA
    # Import LlamaIndex LLM class needed for get_correction_llm
    from llama_index.llms.gemini import Gemini # Example for Gemini
    # Import Vector DB & Embeddings classes needed for resource init
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    # Import OCR client if needed for resource init
    from mistralai import Mistral
    # Import Neo4j driver if needed for resource init
    import neo4j
    # Import the new audit manager
    import src.utils.audit_db_manager # Assuming it's in the root or src path
except ImportError as e:
    logging.critical(f"Fatal Import Error in graphrag_app.py: {e}. Ensure project structure and dependencies.", exc_info=True)
    # Display error in Streamlit if it loads this far
    st_available = 'streamlit' in sys.modules
    if st_available:
        st.error(f"Fatal Import Error: {e}. Cannot start application. Check logs and dependencies.")
        st.stop()
    else:
        print(f"Fatal Import Error: {e}. Cannot start application.", file=sys.stderr)
        sys.exit(1)

# --- Define Custom CSS ---
st.markdown("""
<style>
    /* --- General --- */
    .stApp {
        /* background-color: #f4f6f8; */ /* Slightly off-white background */
    }
    /* Use a more modern sans-serif font */
    /* html, body, [class*="st-"], button, input, select, textarea {
        font-family: 'Inter', 'system-ui', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    } */

    /* --- Headers --- */
    h1 { /* Main App Title */
        color: #1E293B; /* Dark Slate Gray */
        text-align: center;
        padding-bottom: 0.5rem;
        /* border-bottom: 2px solid #64748B; */ /* Optional underline */
    }
    h2 { /* Page Titles / Major Sections */
        color: #334155; /* Medium Slate Gray */
        border-bottom: 1px solid #CBD5E1; /* Subtle underline */
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
     h3 { /* Sub-Sections */
        color: #475569; /* Lighter Slate Gray */
        margin-top: 1rem;
    }

    /* --- Chat Interface --- */
    [data-testid="stChatMessage"] {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border: 1px solid #E2E8F0; /* Light border */
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    [data-testid="stChatMessage"] p { /* Paragraphs within chat */
        line-height: 1.65;
        color: #333;
    }
    /* Assistant Message Bubble */
    [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #F8FAFC; /* Very light gray/blue */
    }
     /* User Message Bubble */
    [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
        background-color: #EFF6FF; /* Light blue */
    }

    /* --- Expanders --- */
    .stExpander {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background-color: #ffffff;
        margin-bottom: 0.5rem; /* Space below expanders */
    }
    .stExpander header {
        font-weight: 500; /* Slightly bolder header */
        color: #475569;
        padding: 8px 12px; /* Adjust padding */
        border-bottom: 1px solid #E2E8F0;
    }
    .stExpander div[data-testid="stExpanderDetails"] {
         background-color: #F8FAFC; /* Slightly different background for content */
         padding: 12px;
         border: none; /* Remove inner border */
         font-size: 0.95rem; /* Slightly smaller font inside */
    }
    .stExpander pre { /* Code blocks inside expanders */
        background-color: #E2E8F0;
        border: none;
    }

    /* --- Last Response Info Box --- */
    /* Target the container created in 1_Chat_Assistant.py */
    /* You might need to inspect element to get the exact selector if this changes */
    /* This targets a block that contains columns, which often wraps the info section */
     div[data-testid="stVerticalBlock"]:has(>div.stColumns) {
        /* Add specific styles here if st.container(border=True) isn't used/sufficient */
        /* Example: Add a top border */
        /* border-top: 1px dashed #CBD5E1; */
        /* padding-top: 1rem; */
        /* margin-bottom: 1rem; */
     }


     /* --- Buttons --- */
     .stButton>button {
         border-radius: 6px;
         padding: 8px 16px;
         font-weight: 500;
     }

     /* --- Dataframes --- */
     /* Styling dataframes deeply can be tricky, basic header style */
     /* .stDataFrame thead th {
          background-color: #E2E8F0;
          color: #1E293B;
          font-weight: 600;
      } */

</style>
""", unsafe_allow_html=True)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
logger = logging.getLogger(__name__)
# Optional: Prevent duplicate handlers if logger might be configured elsewhere
# if len(logger.handlers) > 1: logger.handlers = [logger.handlers[0]]
# logger.propagate = False

# --- Constants (Minimal - Others moved with functions) ---
# Define paths needed by resource loaders if not passed via config
# Example: If get_chroma_collection doesn't get path from config
# CHROMA_DEFAULT_PATH = "./chroma_db_pipeline" # Example
# --- End Constants ---

@st.cache_resource
def get_enhanced_ocr_pipeline(config):
    """Initialize the enhanced OCR pipeline with EasyOCR + Mistral"""
    logger.info("Initializing Enhanced OCR Pipeline...")
    try:
        pipeline = EnhancedOCRPipeline(config)

        # Set Mistral client if available
        mistral_client = get_mistral_client(config.get('MISTRAL_API_KEY'))
        if mistral_client:
            pipeline.set_mistral_client(mistral_client)

        logger.info(f"Enhanced OCR Pipeline initialized: "
                    f"EasyOCR={'✓' if pipeline.easyocr_reader else '✗'}, "
                    f"Mistral={'✓' if pipeline.mistral_client else '✗'}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced OCR Pipeline: {e}")
        return None

# --- Configuration Loading Function ---
@st.cache_data # Cache the loaded configuration dictionary
def load_config():
    """
    Loads configuration from files and environment variables with validation.
    Order of precedence: Environment Variables > config.toml > graph_config.ini > Defaults
    Returns the configuration as a dictionary. Sets '_CONFIG_VALID' flag.
    """
    config = {}
    logger.info("Loading configuration...")
    try:
        # 1. Load from config.toml (Primary Source)
        toml_config_path = Path("config.toml")
        config_toml = {}
        if tomllib and toml_config_path.is_file():
            with open(toml_config_path, "rb") as f:
                config_toml = dict(tomllib.load(f))
            logger.info("Loaded config from config.toml")

            llm_config = config_toml.get("llm", {})
            config['LLM_MODEL'] = llm_config.get("model")
            config['LLM_API_KEY'] = llm_config.get("api_key")
            config['LLM_BASE_URL'] = llm_config.get("base_url")
            config['LLM_EXTRA_PARAMS'] = llm_config.get("parameters", {})

            triple_llm_config = config_toml.get("llm", {}).get("triple_extraction", llm_config)
            config['TRIPLE_EXTRACTION_LLM_MODEL'] = triple_llm_config.get("model", config.get('LLM_MODEL'))
            config['TRIPLE_EXTRACTION_API_KEY'] = triple_llm_config.get("api_key", config.get('LLM_API_KEY'))
            config['TRIPLE_EXTRACTION_BASE_URL'] = triple_llm_config.get("base_url", config.get('LLM_BASE_URL'))
            config['TRIPLE_EXTRACTION_MAX_TOKENS'] = triple_llm_config.get("max_tokens", 1500)
            config['TRIPLE_EXTRACTION_TEMPERATURE'] = triple_llm_config.get("temperature", 0.2)

            config['MISTRAL_API_KEY'] = config_toml.get("llm", {}).get("ocr", {}).get("mistral_api_key")
            config['EMBEDDING_MODEL'] = config_toml.get("embeddings", {}).get("model_name", "all-MiniLM-L6-v2")
            config['CHUNK_SIZE'] = config_toml.get("chunking", {}).get("chunk_size", 1000)
            config['CHUNK_OVERLAP'] = config_toml.get("chunking", {}).get("overlap", 100)
            config['CHROMA_PERSIST_PATH'] = config_toml.get("vector_db", {}).get('persist_directory', "./chroma_db_pipeline")
            config['COLLECTION_NAME'] = config_toml.get("vector_db", {}).get('collection_name', "doc_pipeline_embeddings")
            config['STANDARDIZATION_ENABLED'] = config_toml.get("standardization", {}).get("enabled", False)
            config['INFERENCE_ENABLED'] = config_toml.get("inference", {}).get("enabled", False)
            config['CACHE_ENABLED'] = config_toml.get("caching", {}).get("enabled", True)
            config['DB_NAME'] = config_toml.get("database", {}).get("name", "neo4j")

            # Add OCR configuration section
            ocr_config = config_toml.get("ocr", {})
            config['EASYOCR_ENABLED'] = ocr_config.get("easyocr_enabled", True)
            config['EASYOCR_GPU'] = ocr_config.get("gpu_enabled", True)
            config['EASYOCR_LANGUAGES'] = ocr_config.get("languages", ["en"])
            config['OCR_PRIMARY_METHOD'] = ocr_config.get("primary_method", "easyocr")
            config['OCR_FALLBACK_ENABLED'] = ocr_config.get("fallback_enabled", True)
            config['OCR_CONFIDENCE_THRESHOLD'] = ocr_config.get("confidence_threshold", 0.5)

            # Override with environment variables
            config['EASYOCR_ENABLED'] = os.getenv('EASYOCR_ENABLED',
                                                  str(config.get('EASYOCR_ENABLED', True))).lower() == 'true'
            config['EASYOCR_GPU'] = os.getenv('EASYOCR_GPU', str(config.get('EASYOCR_GPU', True))).lower() == 'true'

            config['standardization'] = config_toml.get("standardization", {})
            config['inference'] = config_toml.get("inference", {})
            config['llm_full_config'] = config_toml.get("llm", {})

            # Load NLP specific settings
            nlp_config = config_toml.get("nlp", {})  # Get the [nlp] section
            config['COREFERENCE_RESOLUTION_ENABLED'] = nlp_config.get("COREFERENCE_RESOLUTION_ENABLED", False)  # Default to False if key not in section
            config['SPACY_MODEL_NAME'] = nlp_config.get("SPACY_MODEL_NAME", "en_core_web_trf")  # Default model
        else:
            logger.warning("config.toml not found or tomllib not available.")

        # At the top
        import configparser
        print("--- Configparser imported at module level:", configparser)  # Diagnostic print

        # 2. Load from graph_config.ini (Fallback/Supplementary)
        config_path_ini = Path("graph_config.ini")
        if config_path_ini.is_file():
            if 'configparser' not in sys.modules: import configparser # Import only if needed
            neo4j_config_parser = configparser.ConfigParser(); neo4j_config_parser.read(config_path_ini)
            config.setdefault('NEO4J_URI', neo4j_config_parser.get("neo4j", "uri", fallback=None))
            config.setdefault('NEO4J_USER', neo4j_config_parser.get("neo4j", "user", fallback=None))
            config.setdefault('NEO4J_PASSWORD', neo4j_config_parser.get("neo4j", "password", fallback=None))
            config.setdefault('CHROMA_PERSIST_PATH', neo4j_config_parser.get("vector_db", "chroma_path", fallback=config.get('CHROMA_PERSIST_PATH')))
            config.setdefault('COLLECTION_NAME', neo4j_config_parser.get("vector_db", "collection_name", fallback=config.get('COLLECTION_NAME')))
            config.setdefault('DB_NAME', neo4j_config_parser.get("database", "name", fallback=config.get('DB_NAME', 'neo4j')))
            logger.info("Loaded/updated config from graph_config.ini")
        else:
            logger.warning("graph_config.ini not found.")

        # 3. Override with Environment Variables (Highest Priority)
        config['NEO4J_URI'] = os.getenv('NEO4J_URI', config.get('NEO4J_URI'))
        config['NEO4J_USER'] = os.getenv('NEO4J_USER', config.get('NEO4J_USER'))
        config['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD', config.get('NEO4J_PASSWORD'))
        config['LLM_API_KEY'] = os.getenv('LLM_API_KEY', os.getenv('GOOGLE_API_KEY', config.get('LLM_API_KEY')))
        config['TRIPLE_EXTRACTION_API_KEY'] = os.getenv('TRIPLE_EXTRACTION_API_KEY', config.get('TRIPLE_EXTRACTION_API_KEY', config.get('LLM_API_KEY')))
        config['MISTRAL_API_KEY'] = os.getenv('MISTRAL_API_KEY', config.get('MISTRAL_API_KEY'))
        config['LLM_MODEL'] = os.getenv('LLM_MODEL', config.get('LLM_MODEL'))
        config['TRIPLE_EXTRACTION_LLM_MODEL'] = os.getenv('TRIPLE_EXTRACTION_LLM_MODEL', config.get('TRIPLE_EXTRACTION_LLM_MODEL'))
        config['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL', config.get('EMBEDDING_MODEL'))
        config['CHROMA_PERSIST_PATH'] = os.getenv('CHROMA_PERSIST_PATH', config.get('CHROMA_PERSIST_PATH'))
        config['COLLECTION_NAME'] = os.getenv('COLLECTION_NAME', config.get('COLLECTION_NAME'))
        config['LLM_BASE_URL'] = os.getenv('LLM_BASE_URL', config.get('LLM_BASE_URL'))
        config['TRIPLE_EXTRACTION_BASE_URL'] = os.getenv('TRIPLE_EXTRACTION_BASE_URL', config.get('TRIPLE_EXTRACTION_BASE_URL', config.get('LLM_BASE_URL')))
        config['STANDARDIZATION_ENABLED'] = os.getenv('STANDARDIZATION_ENABLED', str(config.get('STANDARDIZATION_ENABLED', False))).lower() == 'true'
        config['INFERENCE_ENABLED'] = os.getenv('INFERENCE_ENABLED', str(config.get('INFERENCE_ENABLED', False))).lower() == 'true'
        config['CACHE_ENABLED'] = os.getenv('CACHE_ENABLED', str(config.get('CACHE_ENABLED', True))).lower() == 'true'
        config['DB_NAME'] = os.getenv('DB_NAME', config.get('DB_NAME'))
        config['CHUNK_SIZE'] = int(os.getenv('CHUNK_SIZE', config.get('CHUNK_SIZE', 1000)))
        config['CHUNK_OVERLAP'] = int(os.getenv('CHUNK_OVERLAP', config.get('CHUNK_OVERLAP', 100)))
        config['COREFERENCE_RESOLUTION_ENABLED'] = os.getenv('COREFERENCE_RESOLUTION_ENABLED',str(config.get('COREFERENCE_RESOLUTION_ENABLED',False))).lower() == 'true'
        config['SPACY_MODEL_NAME'] = os.getenv('SPACY_MODEL_NAME', config.get('SPACY_MODEL_NAME'))

        # 4. Final Validation
        required_for_qa = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'LLM_MODEL', 'LLM_API_KEY', 'EMBEDDING_MODEL', 'CHROMA_PERSIST_PATH', 'COLLECTION_NAME', 'DB_NAME']
        missing_keys = [k for k in required_for_qa if not config.get(k)]
        if missing_keys:
            error_message = f"Missing required config/secrets for core Q&A: {', '.join(missing_keys)}"
            logger.error(error_message)
            config['_CONFIG_VALID'] = False
        else:
             config['_CONFIG_VALID'] = True

        logger.info("Configuration loading process complete.")
        logger.debug(f"Final Config Summary: LLM_MODEL={config.get('LLM_MODEL')}, EMBEDDING_MODEL={config.get('EMBEDDING_MODEL')}, NEO4J_URI={config.get('NEO4J_URI')}")
        return config

    except Exception as e:
        logger.exception("Critical error during configuration loading.")
        return {'_CONFIG_VALID': False}


# --- Resource Initialization (Cached) ---
# These functions load resources once per session and cache them.
# They can be imported and called from different page files.

@st.cache_resource
def get_requests_session():
    """Creates and returns a requests.Session object."""
    logger.info("Initializing requests.Session resource...")
    session = requests.Session()
    logger.info("requests.Session resource initialized.")
    return session

@st.cache_resource
def get_correction_llm(config):
    """Initializes and returns the LlamaIndex LLM needed for correction."""
    if not config or not config.get('_CONFIG_VALID', False):
         logger.warning("Skipping correction LLM initialization: Invalid base config.")
         return None
    model_name = config.get('LLM_MODEL')
    api_key = config.get('LLM_API_KEY')
    if not model_name or not api_key:
        logger.warning("Correction LLM model/API key missing. Correction disabled.")
        return None
    logger.info(f"Initializing LlamaIndex LLM '{model_name}' for correction...")
    try:
        if "gemini" in model_name.lower():
            llm = Gemini(model_name=model_name, api_key=api_key)
        else:
            logger.error(f"Unsupported LLM provider for correction model: {model_name}.")
            llm = None
        if llm: logger.info("Correction LLM initialized successfully.")
        return llm
    except ImportError as e:
        logger.error(f"ImportError for LlamaIndex LLM class {model_name}: {e}. Correction disabled.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex LLM for correction: {e}", exc_info=True)
        return None

@st.cache_resource
def get_mistral_client(api_key):
    """Initializes and returns a Mistral client."""
    if not api_key:
        logger.warning("Mistral API Key not provided. OCR will be disabled.")
        return None
    logger.info("Initializing Mistral client...")
    try:
        client = Mistral(api_key=api_key)
        logger.info("Mistral client initialized.")
        return client
    except Exception as e:
        logger.error(f"Mistral client initialization failed: {e}", exc_info=True)
        return None

@st.cache_resource
def get_embedding_model(model_name):
    """Loads and returns a SentenceTransformer embedding model."""
    if not model_name:
        logger.error("Embedding model name not provided.")
        return None
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, device=None)
        logger.info("Embedding model loaded.")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        return None

@st.cache_resource
@st.cache_resource
def get_chroma_collection(chroma_path, collection_name, embedding_model_name):
    """Enhanced ChromaDB collection with consistent settings."""
    if not all([chroma_path, collection_name, embedding_model_name]):
        logger.error("Missing ChromaDB path, collection name, or embedding model name.")
        return None

    logger.info(f"Initializing ChromaDB connection at {chroma_path}")
    try:
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        # Use consistent settings across all ChromaDB connections
        settings = chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )

        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=settings  # Add consistent settings
        )

        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_ef,
            metadata={"hnsw:space": "cosine"}
        )

        count = collection.count()
        logger.info(f"ChromaDB collection '{collection_name}' ready. Count: {count}")
        return collection

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection '{collection_name}': {e}", exc_info=True)
        return None

@st.cache_resource
def init_neo4j_exporter(uri, user, password):
    """Initializes and returns a Neo4jExporter instance."""
    if not all([uri, user, password]):
        logger.error("Missing Neo4j URI, user, or password for exporter.")
        return None
    logger.info("Initializing Neo4jExporter resource...")
    try:
        exporter = Neo4jExporter(uri=uri, user=user, password=password)
        logger.info("Neo4jExporter resource initialized.")
        return exporter
    except Exception as e:
        logger.error(f"Neo4jExporter initialization failed: {e}", exc_info=True)
        return None

@st.cache_resource
def load_qa_engine(config, _correction_llm):
    """Initializes and returns the GraphRAGQA engine."""
    logger.info("Initializing GraphRAGQA Engine resource...")
    if not config or not config.get('_CONFIG_VALID', False):
         logger.error("Config invalid, cannot initialize GraphRAGQA engine.")
         return None
    try:
        engine = GraphRAGQA(
            neo4j_uri=config['NEO4J_URI'],
            neo4j_user=config['NEO4J_USER'],
            neo4j_password=config['NEO4J_PASSWORD'],
            llm_instance_for_correction=_correction_llm,
            llm_model=config['LLM_MODEL'],
            llm_api_key=config['LLM_API_KEY'],
            llm_base_url=config.get('LLM_BASE_URL'),
            embedding_model_name=config['EMBEDDING_MODEL'],
            chroma_path=config['CHROMA_PERSIST_PATH'],
            collection_name=config['COLLECTION_NAME'],
            db_name=config['DB_NAME'],
            llm_config_extra=config.get('LLM_EXTRA_PARAMS', {}),
            max_cypher_retries=config.get('max_cypher_retries', 1)
        )
        logger.info(f"GraphRAGQA Engine resource initialized. Ready: {engine.is_ready()}")
        # Display warning in UI if engine isn't ready
        if not engine.is_ready() and 'streamlit' in sys.modules:
            st.warning("Q&A Engine initialized but may not be fully ready (check Neo4j/LLM status).", icon="⚠️")
        return engine
    except Exception as e:
        logger.error(f"GraphRAGQA Engine initialization failed: {e}", exc_info=True)
        if 'streamlit' in sys.modules:
            st.error(f"Failed to initialize Q&A Engine: {e}")
        return None

@st.cache_resource
def get_nlp_pipeline(config): # Renamed for broader NLP tasks
    """Loads and returns a spaCy NLP pipeline."""

    logger.info(f"Current config object: {config}")  # Or a more specific part
    logger.info(f"Value of COREFERENCE_RESOLUTION_ENABLED from config: {config.get('COREFERENCE_RESOLUTION_ENABLED')}")
    logger.info(
        f"Type of COREFERENCE_RESOLUTION_ENABLED from config: {type(config.get('COREFERENCE_RESOLUTION_ENABLED'))}")


    if not config.get('COREFERENCE_RESOLUTION_ENABLED', False): # Check master toggle
        logger.info("Coreference resolution (and NLP pipeline) is disabled in config.")
        return None

    model_name = config.get('SPACY_MODEL_NAME', "en_core_web_trf") # e.g., "en_core_web_trf"
    logger.info(f"Loading spaCy NLP pipeline: {model_name}...")
    try:
        nlp = spacy.load(model_name)
        # Depending on the model, coreference might be a default component
        # or you might need to add a pipe if using a specific coref component
        # e.g., if you found a specific component to add:
        # if "coref_component_name" not in nlp.pipe_names:
        #    nlp.add_pipe("coref_component_name", config={"device": -1}) # -1 CPU, 0 GPU
        logger.info(f"SpaCy NLP pipeline '{model_name}' loaded successfully.")
        return nlp
    except OSError as e:
        logger.error(f"Could not load spaCy model '{model_name}'. Is it downloaded? (python -m spacy download {model_name}). Error: {e}", exc_info=True)
        # Avoid st.error here; main/page will handle None return
        return None
    except Exception as e:
        logger.error(f"Error loading spaCy NLP pipeline '{model_name}': {e}", exc_info=True)
        return None

# --- Helper Functions REMOVED ---
# Definitions for load_chat_history, save_chat_history, display_pdf,
# process_uploaded_file_ocr, extract_knowledge_graph, store_chunks_and_embeddings,
# get_file_hash, load_triples_from_cache, save_triples_to_cache
# have been removed from this file.
# They should now reside in appropriate page files (like pages/1_Chat_Assistant.py)
# or utility/pipeline modules (like processing_pipeline.py).
# -----------------------------


# --- Streamlit App Main Logic (Minimal Entry Point) ---
def main():
    """Sets up the main app configuration and landing page."""

    # Removed the centered markdown title, use standard title
    st.title("📄 Document GraphRAG Assistant")

    # Load configuration early - essential for everything else
    config = load_config()
    if not config or not config.get('_CONFIG_VALID', False):
        # Error message displayed by load_config now
        logger.critical("Halting app start due to invalid configuration.")
        st.stop() # Stop execution if config is bad

    # Initialize Audit Database (essential for ingestion page)
    try:
        # Assuming audit_db_manager is imported correctly at the top
        src.utils.audit_db_manager.initialize_database()
        logger.info("Audit database initialized successfully.")
    except NameError:
        st.error("Fatal Error: `audit_db_manager` not imported correctly.")
        logger.critical("Fatal: Audit DB Manager import failed.", exc_info=True)
        st.stop()
    except Exception as db_init_e:
        st.error(f"Fatal Error: Could not initialize Audit Database: {db_init_e}")
        logger.critical("Fatal: Audit DB Initialization failed.", exc_info=True)
        st.stop() # Stop execution if audit DB fails

    # --- Initialize session state defaults (Minimal) ---
    # Pages will manage their own specific state loading as needed
    st.session_state.setdefault("running_ingestion_job_id", None) # Track background job

    # vvv ADD DEFAULTS FOR LAST RESPONSE INFO vvv
    st.session_state.setdefault("last_response_sources", None)
    st.session_state.setdefault("last_response_cypher", None)
    st.session_state.setdefault("last_response_error", None)
    st.session_state.setdefault("last_response_info", None)
    # ^^^ END ADDED DEFAULTS ^^^

    # --- Landing Page Content ---
    st.info("Select an option from the sidebar to get started:")
    # Use new page links (ensure filenames in 'pages/' directory are correct)
    st.page_link("pages/2_Data_Ingestion.py", label="Process New Documents", icon="📥")
    st.page_link("pages/1_Chat_Assistant.py", label="Ask Questions (Chat)", icon="💬")

    st.markdown("---")
    st.markdown("""
    **Welcome!**

    * Use **Data Ingestion** to upload documents (PDF, TXT, images via OCR). The system will extract information, build a knowledge graph, and create vector embeddings.
    * Use **Chat Assistant** to ask questions about the information contained within your processed documents.
    """)
    # Optional: Pre-load some resources if it improves perceived performance on page load
    # with st.spinner("Initializing resources..."):
    #    get_requests_session()
    #    get_embedding_model(config.get('EMBEDDING_MODEL'))
    #    # etc. - use cautiously as it slows down initial app load

# --- Entry Point ---
if __name__ == "__main__":
    # Ensure nest_asyncio is applied (should be at the very top)
    # nest_asyncio.apply() # Already at top

    # Basic logging setup (can be more sophisticated)
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s')
    main()
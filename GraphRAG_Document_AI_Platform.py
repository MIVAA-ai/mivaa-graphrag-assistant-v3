# GraphRAG_Document_AI_Platform.py (Main App Entry Point - FIXED VERSION)

import nest_asyncio

nest_asyncio.apply()

import streamlit as st
import os
import logging
import sys
from pathlib import Path
import spacy
import configparser
import requests
import re
from typing import Dict, Optional, Any

st.set_page_config(
    page_title="Document AI Assistant",  # Updated page title
    page_icon="🤖",  # Updated icon to be more AI-focused
    layout="wide",
    initial_sidebar_state="expanded"
)

from enhanced_ocr_pipeline import EnhancedOCRPipeline

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        logging.critical("No TOML parser found. Please install 'tomli' or use Python 3.11+.")
        print("FATAL: No TOML parser found.", file=sys.stderr)
        sys.exit(1)

# Core Logic Imports
try:
    from neo4j_exporter import Neo4jExporter
    from graph_rag_qa import GraphRAGQA
    from llama_index.llms.gemini import Gemini
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    from mistralai import Mistral
    import neo4j
    import src.utils.audit_db_manager
except ImportError as e:
    logging.critical(f"Fatal Import Error in graphrag_app.py: {e}")
    st.error(f"Fatal Import Error: {e}. Cannot start application.")
    st.stop()


# FIXED: API Key masking utility functions
def mask_sensitive_data(value: str, mask_char: str = "*", show_chars: int = 4) -> str:
    """Mask sensitive data like API keys and passwords for logging."""
    if not isinstance(value, str) or len(value) <= show_chars:
        return mask_char * 8  # Default mask for short values

    return value[:show_chars] + mask_char * (len(value) - show_chars)


def get_masked_config_for_logging(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a copy of config with sensitive data masked for safe logging."""
    sensitive_keys = {
        'api_key', 'password', 'secret', 'token', 'key',
        'LLM_API_KEY', 'TRIPLE_EXTRACTION_API_KEY', 'MISTRAL_API_KEY',
        'NEO4J_PASSWORD', 'mistral_api_key'
    }

    def mask_recursive(obj):
        if isinstance(obj, dict):
            return {
                k: mask_sensitive_data(str(v)) if any(sens in k.lower() for sens in sensitive_keys)
                else mask_recursive(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [mask_recursive(item) for item in obj]
        else:
            return obj

    return mask_recursive(config)


# Enhanced Custom CSS
st.markdown("""
<style>
    .stApp {
        /* background-color: #f4f6f8; */
    }

    h1 {
        color: #1E293B;
        text-align: center;
        padding-bottom: 0.5rem;
    }
    h2 {
        color: #334155;
        border-bottom: 1px solid #CBD5E1;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
    }
    h3 {
        color: #475569;
        margin-top: 1rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #F8FAFC;
    }

    [data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {
        background-color: #EFF6FF;
    }

    .stExpander {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background-color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .stButton>button {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# FIXED: Logger Setup with API key masking
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
logger = logging.getLogger(__name__)


@st.cache_resource
def get_enhanced_ocr_pipeline(config):
    """Initialize the enhanced OCR pipeline with EasyOCR + Mistral"""
    logger.info("Initializing Enhanced OCR Pipeline...")
    try:
        pipeline = EnhancedOCRPipeline(config)

        # FIXED: Get Mistral API key from the centralized TOML structure
        mistral_api_key = (
                config.get('MISTRAL_API_KEY') or
                config.get('mistral_api_key') or
                config.get('mistral', {}).get('api_key') or
                config.get('llm', {}).get('ocr', {}).get('mistral_api_key')
        )

        mistral_client = get_mistral_client(mistral_api_key)
        if mistral_client:
            pipeline.set_mistral_client(mistral_client)

        # FIXED: Use masked logging
        easyocr_status = '✓' if pipeline.easyocr_reader else '✗'
        mistral_status = '✓' if pipeline.mistral_client else '✗'
        logger.info(f"Enhanced OCR Pipeline initialized: EasyOCR={easyocr_status}, Mistral={mistral_status}")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced OCR Pipeline: {e}")
        return None


@st.cache_data
def load_config():
    """
    FIXED: Loads configuration with API key masking and centralized settings.
    """
    config = {}
    logger.info("Loading configuration...")
    try:
        # 1. Load from config.toml (Primary Source - CENTRALIZED)
        toml_config_path = Path("config.toml")
        config_toml = {}
        if tomllib and toml_config_path.is_file():
            with open(toml_config_path, "rb") as f:
                config_toml = dict(tomllib.load(f))
            logger.info("Loaded config from config.toml")

            # CENTRALIZED: Load all configuration from TOML
            llm_config = config_toml.get("llm", {})
            config['LLM_MODEL'] = llm_config.get("model")
            config['LLM_API_KEY'] = llm_config.get("api_key")
            config['LLM_BASE_URL'] = llm_config.get("base_url")
            config['LLM_EXTRA_PARAMS'] = llm_config.get("parameters", {})

            # Triple extraction (uses main LLM config by default)
            triple_config = config_toml.get("triple_extraction", llm_config)
            config['TRIPLE_EXTRACTION_LLM_MODEL'] = triple_config.get("model", config.get('LLM_MODEL'))
            config['TRIPLE_EXTRACTION_API_KEY'] = triple_config.get("api_key", config.get('LLM_API_KEY'))
            config['TRIPLE_EXTRACTION_BASE_URL'] = triple_config.get("base_url", config.get('LLM_BASE_URL'))
            config['TRIPLE_EXTRACTION_MAX_TOKENS'] = triple_config.get("max_tokens", 2000)
            config['TRIPLE_EXTRACTION_TEMPERATURE'] = triple_config.get("temperature", 0.1)

            # FIXED: Get Mistral API key from centralized location
            config['MISTRAL_API_KEY'] = llm_config.get("ocr", {}).get("mistral_api_key")

            # CENTRALIZED: Database configuration
            neo4j_config = config_toml.get("neo4j", {})
            config['NEO4J_URI'] = neo4j_config.get("uri")
            config['NEO4J_USER'] = neo4j_config.get("user")
            config['NEO4J_PASSWORD'] = neo4j_config.get("password")
            config['DB_NAME'] = neo4j_config.get("database", "neo4j")

            # CENTRALIZED: Vector DB configuration
            vector_config = config_toml.get("vector_db", {})
            config['CHROMA_PERSIST_PATH'] = vector_config.get("persist_directory", "./chroma_db_pipeline")
            config['COLLECTION_NAME'] = vector_config.get("collection_name", "doc_pipeline_embeddings")

            # CENTRALIZED: Embeddings configuration
            embedding_config = config_toml.get("embeddings", {})
            config['EMBEDDING_MODEL'] = embedding_config.get("model_name", "all-MiniLM-L6-v2")

            # CENTRALIZED: Chunking configuration
            chunk_config = config_toml.get("chunking", {})
            config['CHUNK_SIZE'] = chunk_config.get("chunk_size", 1000)
            config['CHUNK_OVERLAP'] = chunk_config.get("overlap", 100)

            # CENTRALIZED: Feature flags
            config['STANDARDIZATION_ENABLED'] = config_toml.get("standardization", {}).get("enabled", True)
            config['INFERENCE_ENABLED'] = config_toml.get("inference", {}).get("enabled", True)
            config['CACHE_ENABLED'] = config_toml.get("caching", {}).get("enabled", True)

            # CENTRALIZED: OCR configuration
            ocr_config = config_toml.get("ocr", {})
            config['EASYOCR_ENABLED'] = ocr_config.get("easyocr_enabled", True)
            config['EASYOCR_GPU'] = ocr_config.get("easyocr_gpu", True)
            config['EASYOCR_LANGUAGES'] = ocr_config.get("easyocr_languages", ["en"])
            config['OCR_PRIMARY_METHOD'] = ocr_config.get("ocr_primary_method", "easyocr")
            config['OCR_FALLBACK_ENABLED'] = ocr_config.get("ocr_fallback_enabled", True)
            config['OCR_CONFIDENCE_THRESHOLD'] = ocr_config.get("confidence_threshold", 0.5)

            # CENTRALIZED: NLP configuration
            nlp_config = config_toml.get("nlp", {})
            config['COREFERENCE_RESOLUTION_ENABLED'] = nlp_config.get("coreference_resolution_enabled", False)
            config['SPACY_MODEL_NAME'] = nlp_config.get("spacy_model_name", "en_core_web_trf")

            # Pass full config sections for modules that need them
            config['standardization'] = config_toml.get("standardization", {})
            config['inference'] = config_toml.get("inference", {})
            config['llm_full_config'] = config_toml.get("llm", {})

            # FIXED: Include relationship inference configurations
            config['relationship_inference'] = config_toml.get("relationship_inference", {})
            config['within_community_inference'] = config_toml.get("within_community_inference", {})

        else:
            logger.warning("config.toml not found or tomllib not available")

        # 2. DEPRECATED: graph_config.ini as fallback only (will be phased out)
        config_path_ini = Path("graph_config.ini")
        if config_path_ini.is_file():
            neo4j_config_parser = configparser.ConfigParser()
            neo4j_config_parser.read(config_path_ini)

            # Only use as fallback if not already set
            config.setdefault('NEO4J_URI', neo4j_config_parser.get("neo4j", "uri", fallback=None))
            config.setdefault('NEO4J_USER', neo4j_config_parser.get("neo4j", "user", fallback=None))
            config.setdefault('NEO4J_PASSWORD', neo4j_config_parser.get("neo4j", "password", fallback=None))

            logger.info("Loaded fallback config from graph_config.ini (DEPRECATED)")

        # 3. Environment Variables (Highest Priority)
        config['NEO4J_URI'] = os.getenv('NEO4J_URI', config.get('NEO4J_URI'))
        config['NEO4J_USER'] = os.getenv('NEO4J_USER', config.get('NEO4J_USER'))
        config['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD', config.get('NEO4J_PASSWORD'))
        config['LLM_API_KEY'] = os.getenv('LLM_API_KEY', os.getenv('GOOGLE_API_KEY', config.get('LLM_API_KEY')))
        config['MISTRAL_API_KEY'] = os.getenv('MISTRAL_API_KEY', config.get('MISTRAL_API_KEY'))
        config['EASYOCR_ENABLED'] = os.getenv('EASYOCR_ENABLED',
                                              str(config.get('EASYOCR_ENABLED', True))).lower() == 'true'
        config['EASYOCR_GPU'] = os.getenv('EASYOCR_GPU', str(config.get('EASYOCR_GPU', True))).lower() == 'true'

        # 4. Final Validation
        required_for_qa = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'LLM_MODEL', 'LLM_API_KEY', 'EMBEDDING_MODEL',
                           'CHROMA_PERSIST_PATH', 'COLLECTION_NAME', 'DB_NAME']
        missing_keys = [k for k in required_for_qa if not config.get(k)]
        if missing_keys:
            error_message = f"Missing required config/secrets: {', '.join(missing_keys)}"
            logger.error(error_message)
            config['_CONFIG_VALID'] = False
        else:
            config['_CONFIG_VALID'] = True

        logger.info("Configuration loading process complete")

        # FIXED: Log configuration summary with masked sensitive data
        masked_config = get_masked_config_for_logging(config)
        logger.debug(
            f"Config Summary: LLM_MODEL={masked_config.get('LLM_MODEL')}, NEO4J_URI={masked_config.get('NEO4J_URI')}")

        return config

    except Exception as e:
        logger.exception("Critical error during configuration loading")
        return {'_CONFIG_VALID': False}


@st.cache_resource
def get_requests_session():
    """Creates and returns a requests.Session object."""
    logger.info("Initializing requests.Session resource...")
    session = requests.Session()
    logger.info("requests.Session resource initialized")
    return session


@st.cache_resource
def get_correction_llm(config):
    """Initializes and returns the LlamaIndex LLM needed for correction."""
    if not config or not config.get('_CONFIG_VALID', False):
        logger.warning("Skipping correction LLM initialization: Invalid base config")
        return None

    model_name = config.get('LLM_MODEL')
    api_key = config.get('LLM_API_KEY')
    if not model_name or not api_key:
        logger.warning("Correction LLM model/API key missing. Correction disabled")
        return None

    # FIXED: Mask API key in logs
    masked_key = mask_sensitive_data(api_key)
    logger.info(f"Initializing LlamaIndex LLM '{model_name}' for correction... (key: {masked_key})")

    try:
        if "gemini" in model_name.lower():
            llm = Gemini(model_name=model_name, api_key=api_key)
        else:
            logger.error(f"Unsupported LLM provider for correction model: {model_name}")
            llm = None

        if llm:
            logger.info("Correction LLM initialized successfully")
        return llm
    except ImportError as e:
        logger.error(f"ImportError for LlamaIndex LLM class {model_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex LLM for correction: {e}")
        return None


@st.cache_resource
def get_mistral_client(api_key):
    """Initializes and returns a Mistral client."""
    if not api_key:
        logger.warning("Mistral API Key not provided. OCR will be disabled")
        return None

    # FIXED: Mask API key in logs
    masked_key = mask_sensitive_data(api_key)
    logger.info(f"Initializing Mistral client... (key: {masked_key})")

    try:
        client = Mistral(api_key=api_key)
        logger.info("Mistral client initialized")
        return client
    except Exception as e:
        logger.error(f"Mistral client initialization failed: {e}")
        return None


@st.cache_resource
def get_embedding_model(model_name):
    """Loads and returns a SentenceTransformer embedding model."""
    if not model_name:
        logger.error("Embedding model name not provided")
        return None

    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, device=None)
        logger.info("Embedding model loaded")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        return None


@st.cache_resource
def get_chroma_collection(chroma_path, collection_name, embedding_model_name):
    """Enhanced ChromaDB collection with consistent settings."""
    if not all([chroma_path, collection_name, embedding_model_name]):
        logger.error("Missing ChromaDB path, collection name, or embedding model name")
        return None

    logger.info(f"Initializing ChromaDB connection at {chroma_path}")
    try:
        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        settings = chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )

        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=settings
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
        logger.error(f"Failed to initialize ChromaDB collection '{collection_name}': {e}")
        return None


@st.cache_resource
def init_neo4j_exporter(uri, user, password):
    """Initializes and returns a Neo4jExporter instance."""
    if not all([uri, user, password]):
        logger.error("Missing Neo4j URI, user, or password for exporter")
        return None

    # FIXED: Mask password in logs
    masked_password = mask_sensitive_data(password)
    logger.info(f"Initializing Neo4jExporter resource... (password: {masked_password})")

    try:
        exporter = Neo4jExporter(uri=uri, user=user, password=password)
        logger.info("Neo4jExporter resource initialized")
        return exporter
    except Exception as e:
        logger.error(f"Neo4jExporter initialization failed: {e}")
        return None


@st.cache_resource
def load_qa_engine(config, _correction_llm):
    """Initializes and returns the GraphRAGQA engine."""
    logger.info("Initializing GraphRAGQA Engine resource...")
    if not config or not config.get('_CONFIG_VALID', False):
        logger.error("Config invalid, cannot initialize GraphRAGQA engine")
        return None

    try:
        # FIXED: Mask sensitive data in logs
        masked_password = mask_sensitive_data(config['NEO4J_PASSWORD'])
        masked_api_key = mask_sensitive_data(config['LLM_API_KEY'])
        logger.info(f"Initializing QA engine with Neo4j password: {masked_password}, LLM key: {masked_api_key}")

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

        if not engine.is_ready() and 'streamlit' in sys.modules:
            st.warning("Q&A Engine initialized but may not be fully ready (check Neo4j/LLM status)", icon="⚠️")
        return engine
    except Exception as e:
        logger.error(f"GraphRAGQA Engine initialization failed: {e}")
        if 'streamlit' in sys.modules:
            st.error(f"Failed to initialize Q&A Engine: {e}")
        return None


@st.cache_resource
def get_nlp_pipeline(config):
    """Loads and returns a spaCy NLP pipeline."""

    # FIXED: Use consistent config key names
    coreference_enabled = config.get('COREFERENCE_RESOLUTION_ENABLED', False)

    logger.info(f"Coreference resolution enabled: {coreference_enabled}")

    if not coreference_enabled:
        logger.info("Coreference resolution (and NLP pipeline) is disabled in config")
        return None

    model_name = config.get('SPACY_MODEL_NAME', "en_core_web_trf")
    logger.info(f"Loading spaCy NLP pipeline: {model_name}...")

    try:
        nlp = spacy.load(model_name)
        logger.info(f"SpaCy NLP pipeline '{model_name}' loaded successfully")
        return nlp
    except OSError as e:
        logger.error(f"Could not load spaCy model '{model_name}'. Is it downloaded? Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading spaCy NLP pipeline '{model_name}': {e}")
        return None


def main():
    """Sets up the main app configuration and landing page."""

    # Updated main title
    st.title("🤖 Document AI Assistant")

    # Load configuration early
    config = load_config()
    if not config or not config.get('_CONFIG_VALID', False):
        logger.critical("Halting app start due to invalid configuration")
        st.stop()

    # Initialize Audit Database
    try:
        src.utils.audit_db_manager.initialize_database()
        logger.info("Audit database initialized successfully")
    except Exception as db_init_e:
        st.error(f"Fatal Error: Could not initialize Audit Database: {db_init_e}")
        logger.critical("Fatal: Audit DB Initialization failed", exc_info=True)
        st.stop()

    # Initialize session state defaults
    st.session_state.setdefault("running_ingestion_job_id", None)
    st.session_state.setdefault("last_response_sources", None)
    st.session_state.setdefault("last_response_cypher", None)
    st.session_state.setdefault("last_response_error", None)
    st.session_state.setdefault("last_response_info", None)

    # Landing Page Content with Updated Display Names
    st.info("Select an option from the sidebar to get started:")

    # Updated page links with new display names (keeping file names unchanged)
    st.page_link("pages/2_Document_Ingestion.py", label="Document Ingestion", icon="📥")
    st.page_link("pages/1_Knowledge_Chat_Assistant.py", label="Knowledge Chat Assistant", icon="💬")
    st.page_link("pages/3_Data_Extraction_Validation.py", label="Data Extraction Validation", icon="📊")
    st.page_link("pages/4_OCR_Output_Analyzer.py", label="Processed Files Manager", icon="🗂️")

    st.markdown("---")
    st.markdown("""
    **Welcome to your AI-powered Document Assistant!**

    * Use **Document Ingestion** to upload documents (PDF, TXT, images via OCR). The AI will extract information, build a knowledge graph, and create vector embeddings.
    * Use **Knowledge Chat Assistant** to ask questions about the information contained within your processed documents.
    * Use **Data Extraction Validation** to monitor AI extraction quality and performance metrics.
    * Use **Processed Files Manager** to browse and manage your document archive.

    **Security Note**: All API keys and passwords are automatically masked in application logs for security.
    """)


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s')
    main()
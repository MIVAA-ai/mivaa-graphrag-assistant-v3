# GraphRAG_Document_AI_Platform.py - ENHANCED WITH MULTI-PROVIDER LLM SUPPORT

import nest_asyncio

nest_asyncio.apply()

import streamlit as st

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import logging
import sys
from pathlib import Path
import spacy
import configparser
import requests
import re
from typing import Dict, Optional, Any, List, Union
from dotenv import load_dotenv
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
    from enhanced_graph_rag_qa import EnhancedGraphRAGQA

    # ENHANCED: Import new LLM system
    from src.knowledge_graph.llm import (
        LLMProviderFactory,
        LLMManager,
        LLMConfig,
        LLMProvider,
        create_llm_config_from_env,
        LLMConfigurationError,
        LLMProviderError
    )

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


# ENHANCED: LLM Factory and Management Classes
class LLMConfigurationManager:
    """Manages LLM configurations and provider instantiation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.llm_managers = {}

    def _get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get provider configuration from config."""
        providers_config = self.config.get('llm', {}).get('providers', {})
        provider_config = providers_config.get(provider_name, {})

        if not provider_config:
            raise LLMConfigurationError(f"Provider '{provider_name}' not found in configuration")

        if not provider_config.get('enabled', False):
            raise LLMConfigurationError(f"Provider '{provider_name}' is disabled in configuration")

        return provider_config

    def create_provider(self, provider_name: str, task_overrides: Dict[str, Any] = None) -> 'BaseLLMProvider':
        """Create a provider instance with optional task-specific overrides."""
        provider_config = self._get_provider_config(provider_name)

        # Apply task-specific overrides
        if task_overrides:
            provider_config = {**provider_config, **task_overrides}

        # Create LLM config
        config_dict = {
            'provider': provider_name,
            'model': provider_config.get('model'),
            'api_key': provider_config.get('api_key'),
            'base_url': provider_config.get('base_url'),
            'max_tokens': provider_config.get('max_tokens', 2000),
            'temperature': provider_config.get('temperature', 0.1),
            'timeout': provider_config.get('timeout', 60),
            'extra_params': {k: v for k, v in provider_config.items()
                             if k not in ['provider', 'model', 'api_key', 'base_url', 'max_tokens', 'temperature',
                                          'timeout']}
        }

        return LLMProviderFactory.create_from_dict(config_dict)

    def create_task_llm_manager(self, task_name: str) -> LLMManager:
        """Create an LLM manager for a specific task with fallback support."""
        # Get task-specific configuration
        task_config = self.config.get(task_name, {})

        # Determine primary provider
        primary_provider_name = task_config.get('provider') or self.config.get('llm', {}).get('primary_provider',
                                                                                              'gemini')

        # Create task-specific overrides
        task_overrides = {}
        if task_config.get('model'):
            task_overrides['model'] = task_config['model']
        if task_config.get('max_tokens'):
            task_overrides['max_tokens'] = task_config['max_tokens']
        if task_config.get('temperature') is not None:
            task_overrides['temperature'] = task_config['temperature']

        try:
            # Create primary provider
            primary_provider = self.create_provider(primary_provider_name, task_overrides)

            # Create fallback providers
            fallback_providers = []

            # Check if fallback is enabled
            enable_fallback = self.config.get('llm', {}).get('enable_fallback', True)
            if enable_fallback:
                # Get fallback chain
                fallback_chain = self._get_fallback_chain(primary_provider_name)

                for fallback_name in fallback_chain[:2]:  # Limit to 2 fallback providers
                    try:
                        fallback_provider = self.create_provider(fallback_name, task_overrides)
                        fallback_providers.append(fallback_provider)
                    except Exception as e:
                        logger.warning(f"Could not create fallback provider {fallback_name}: {e}")

            # Create and cache LLM manager
            manager = LLMManager(primary_provider, fallback_providers)
            self.llm_managers[task_name] = manager

            logger.info(
                f"Created LLM manager for {task_name}: primary={primary_provider_name}, fallbacks={len(fallback_providers)}")
            return manager

        except Exception as e:
            logger.error(f"Failed to create LLM manager for {task_name}: {e}")
            raise LLMConfigurationError(f"Could not create LLM manager for {task_name}: {e}")

    def _get_fallback_chain(self, provider_name: str) -> List[str]:
        """Get fallback chain for a provider."""
        # First check provider policies
        policies = self.config.get('provider_policies', {})
        fallback_chains = policies.get('fallback_chains', {})

        if provider_name in fallback_chains:
            return fallback_chains[provider_name]

        # Default fallback chain
        default_fallbacks = {
            'gemini': ['openai', 'claude'],
            'openai': ['gemini', 'claude'],
            'claude': ['gemini', 'openai'],
            'mistral': ['gemini', 'openai'],
            'anthropic': ['gemini', 'openai']
        }

        return default_fallbacks.get(provider_name, ['gemini', 'openai'])

    def get_llm_manager(self, task_name: str) -> LLMManager:
        """Get or create LLM manager for a task."""
        if task_name not in self.llm_managers:
            return self.create_task_llm_manager(task_name)
        return self.llm_managers[task_name]

    def get_available_providers(self) -> List[str]:
        """Get list of available and enabled providers."""
        providers_config = self.config.get('llm', {}).get('providers', {})
        return [name for name, config in providers_config.items() if config.get('enabled', False)]


# ENHANCED: Global LLM Configuration Manager
_llm_config_manager = None


def get_llm_config_manager(config: Dict[str, Any]) -> LLMConfigurationManager:
    """Get or create global LLM configuration manager."""
    global _llm_config_manager
    if _llm_config_manager is None:
        _llm_config_manager = LLMConfigurationManager(config)
    return _llm_config_manager


# ENHANCED: Task-specific LLM getters
@st.cache_resource
def get_triple_extraction_llm(config: Dict[str, Any]):
    """Get LLM manager for triple extraction."""
    try:
        llm_manager = get_llm_config_manager(config)
        return llm_manager.get_llm_manager('triple_extraction')
    except Exception as e:
        logger.error(f"Failed to create triple extraction LLM: {e}")
        return None


@st.cache_resource
def get_relationship_inference_llm(config: Dict[str, Any]):
    """Get LLM manager for relationship inference."""
    try:
        llm_manager = get_llm_config_manager(config)
        return llm_manager.get_llm_manager('relationship_inference')
    except Exception as e:
        logger.error(f"Failed to create relationship inference LLM: {e}")
        return None


@st.cache_resource
def get_text_sanitization_llm(config: Dict[str, Any]):
    """Get LLM manager for text sanitization."""
    try:
        llm_manager = get_llm_config_manager(config)
        return llm_manager.get_llm_manager('text_sanitization')
    except Exception as e:
        logger.error(f"Failed to create text sanitization LLM: {e}")
        return None


@st.cache_resource
def get_cypher_correction_llm(config: Dict[str, Any]):
    """Get LLM manager for Cypher correction."""
    try:
        llm_manager = get_llm_config_manager(config)
        return llm_manager.get_llm_manager('cypher_correction')
    except Exception as e:
        logger.error(f"Failed to create Cypher correction LLM: {e}")
        return None


# ENHANCED: Generic LLM caller for backward compatibility
def call_task_llm(task_name: str, user_prompt: str, config: Dict[str, Any], system_prompt: str = None, **kwargs) -> str:
    """Generic function to call LLM for any task."""
    try:
        llm_manager = get_llm_config_manager(config)
        task_llm = llm_manager.get_llm_manager(task_name)
        return task_llm.call_llm(user_prompt, system_prompt, **kwargs)
    except Exception as e:
        logger.error(f"Failed to call LLM for task {task_name}: {e}")
        raise


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
        'NEO4J_PASSWORD', 'mistral_api_key', 'gemini_api_key', 'openai_api_key', 'anthropic_api_key'
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


# Enhanced Custom CSS (unchanged)
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
    """Initialize the enhanced LLM OCR pipeline with FIXED configuration mapping"""
    load_dotenv()

    logger.info("Initializing Enhanced LLM OCR Pipeline...")
    try:
        # FIXED: Proper config mapping for OCR settings with environment variable priority
        ocr_config = {
            # Map TOML structure to expected format
            'LLM_OCR_PRIMARY_METHOD': config.get('llm', {}).get('ocr', {}).get('primary_method', 'gemini'),
            'LLM_OCR_FALLBACK_ENABLED': config.get('llm', {}).get('ocr', {}).get('fallback_enabled', True),
            'LLM_OCR_CONFIDENCE_THRESHOLD': config.get('llm', {}).get('ocr', {}).get('confidence_threshold', 0.7),
            'LLM_OCR_TIMEOUT': config.get('llm', {}).get('ocr', {}).get('timeout_seconds', 60),
            'LLM_OCR_MAX_RETRIES': config.get('llm', {}).get('ocr', {}).get('max_retries', 2),

            # Enhanced metadata settings
            'EXTRACT_ENTITIES': config.get('metadata', {}).get('extract_entities', True),
            'CLASSIFY_DOCUMENTS': config.get('metadata', {}).get('classify_documents', True),
            'ANALYZE_QUALITY': config.get('metadata', {}).get('analyze_quality', True),
            'CHUNK_SIZE': config.get('metadata', {}).get('chunk_size', 1000),

            # API Keys - UPDATED: Environment variables first, then config fallback
            'mistral_api_key': os.getenv('MISTRAL_API_KEY') or config.get('llm', {}).get('ocr', {}).get(
                'mistral_api_key'),
            'gemini_api_key': os.getenv('GOOGLE_API_KEY') or config.get('llm', {}).get('ocr', {}).get(
                'gemini_api_key') or config.get('llm', {}).get('api_key'),
            'openai_api_key': os.getenv('OPENAI_API_KEY') or config.get('llm', {}).get('ocr', {}).get('openai_api_key'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY') or config.get('llm', {}).get('ocr', {}).get(
                'anthropic_api_key'),

            # Pass full config sections
            'llm': config.get('llm', {}),
            'metadata': config.get('metadata', {}),
            'file_processing': config.get('file_processing', {}),
            'storage': config.get('storage', {})
        }

        pipeline = EnhancedOCRPipeline(ocr_config)

        # Initialize LLM clients with environment variable priority
        mistral_api_key = os.getenv('MISTRAL_API_KEY') or ocr_config.get('mistral_api_key')
        if mistral_api_key:
            mistral_client = get_mistral_client(mistral_api_key)
            if mistral_client:
                pipeline.set_mistral_client(mistral_client)
                logger.info("✓ Mistral client initialized for OCR")

        # Log primary method configuration
        primary_method = ocr_config.get('LLM_OCR_PRIMARY_METHOD', 'unknown')
        available_methods = pipeline.get_available_providers()

        logger.info(f"🎯 Enhanced OCR Pipeline: Primary={primary_method}, Available={available_methods}")

        # Verify primary method is available
        if primary_method not in available_methods:
            logger.warning(f"⚠️ Primary method '{primary_method}' not available. Available: {available_methods}")
        else:
            logger.info(f"✓ Primary OCR method '{primary_method}' is correctly configured and available")

        return pipeline

    except Exception as e:
        logger.error(f"Failed to initialize Enhanced LLM OCR Pipeline: {e}")
        return None


@st.cache_data
def load_config():
    """UPDATED: Load configuration with proper TOML structure mapping"""
    load_dotenv()

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

            # FIXED: Direct TOML structure preservation
            config = config_toml.copy()

            # BACKWARD COMPATIBILITY: Also create flat keys for legacy code
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

            # Database configuration
            neo4j_config = config_toml.get("neo4j", {})
            config['NEO4J_URI'] = neo4j_config.get("uri")
            config['NEO4J_USER'] = neo4j_config.get("user")
            config['NEO4J_PASSWORD'] = neo4j_config.get("password")
            config['DB_NAME'] = neo4j_config.get("database", "neo4j")

            # Vector DB configuration
            vector_config = config_toml.get("vector_db", {})
            config['CHROMA_PERSIST_PATH'] = vector_config.get("persist_directory", "./chroma_db_pipeline")
            config['COLLECTION_NAME'] = vector_config.get("collection_name", "doc_pipeline_embeddings")

            # Embeddings configuration
            embedding_config = config_toml.get("embeddings", {})
            config['EMBEDDING_MODEL'] = embedding_config.get("model_name", "all-MiniLM-L6-v2")

            # Chunking configuration
            chunk_config = config_toml.get("chunking", {})
            config['CHUNK_SIZE'] = chunk_config.get("chunk_size", 1000)
            config['CHUNK_OVERLAP'] = chunk_config.get("overlap", 100)

            # Feature flags
            config['STANDARDIZATION_ENABLED'] = config_toml.get("standardization", {}).get("enabled", True)
            config['INFERENCE_ENABLED'] = config_toml.get("inference", {}).get("enabled", True)
            config['CACHE_ENABLED'] = config_toml.get("caching", {}).get("enabled", True)

            # NLP configuration
            nlp_config = config_toml.get("nlp", {})
            config['COREFERENCE_RESOLUTION_ENABLED'] = nlp_config.get("coreference_resolution_enabled", False)
            config['SPACY_MODEL_NAME'] = nlp_config.get("spacy_model_name", "en_core_web_trf")

        else:
            logger.warning("config.toml not found or tomllib not available")

        # 2. Environment Variables (Highest Priority) - UPDATED ORDER
        config['NEO4J_URI'] = os.getenv('NEO4J_URI') or config.get('NEO4J_URI')
        config['NEO4J_USER'] = os.getenv('NEO4J_USER') or config.get('NEO4J_USER')
        config['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD') or config.get('NEO4J_PASSWORD')
        config['LLM_API_KEY'] = os.getenv('GOOGLE_API_KEY') or os.getenv('LLM_API_KEY') or config.get('LLM_API_KEY')

        # Also override nested config API keys
        if config.get('llm') and isinstance(config['llm'], dict):
            config['llm']['api_key'] = os.getenv('GOOGLE_API_KEY') or os.getenv('LLM_API_KEY') or config['llm'].get(
                'api_key')

        # ENHANCED: Override provider API keys from environment
        if config.get('llm', {}).get('providers'):
            providers = config['llm']['providers']

            # Gemini provider
            if 'gemini' in providers:
                providers['gemini']['api_key'] = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or \
                                                 providers['gemini'].get('api_key')

            # OpenAI provider
            if 'openai' in providers:
                providers['openai']['api_key'] = os.getenv('OPENAI_API_KEY') or providers['openai'].get('api_key')

            # Anthropic provider
            if 'anthropic' in providers:
                providers['anthropic']['api_key'] = os.getenv('ANTHROPIC_API_KEY') or providers['anthropic'].get(
                    'api_key')

            # Mistral provider
            if 'mistral' in providers:
                providers['mistral']['api_key'] = os.getenv('MISTRAL_API_KEY') or providers['mistral'].get('api_key')

        # Update task-specific configs with environment variables
        for task_name in ['triple_extraction', 'relationship_inference', 'within_community_inference',
                          'text_sanitization', 'cypher_correction']:
            if config.get(task_name) and isinstance(config[task_name], dict):
                task_config = config[task_name]
                # Inherit from primary if not specified
                if not task_config.get('api_key'):
                    task_config['api_key'] = os.getenv('GOOGLE_API_KEY') or os.getenv('LLM_API_KEY') or config.get(
                        'LLM_API_KEY')
                # Set derived flat keys for backward compatibility
                if task_name == 'triple_extraction':
                    config['TRIPLE_EXTRACTION_API_KEY'] = task_config['api_key']

        # 3. Final Validation
        required_for_qa = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'LLM_MODEL', 'LLM_API_KEY', 'EMBEDDING_MODEL',
                           'CHROMA_PERSIST_PATH', 'COLLECTION_NAME', 'DB_NAME']
        missing_keys = [k for k in required_for_qa if not config.get(k)]
        if missing_keys:
            error_message = f"Missing required config/secrets: {', '.join(missing_keys)}"
            logger.error(error_message)
            config['_CONFIG_VALID'] = False
        else:
            config['_CONFIG_VALID'] = True

        logger.info("✅ Configuration loading complete with proper TOML structure preservation")

        # ENHANCED: Log LLM provider status
        if config.get('llm', {}).get('providers'):
            llm_mgr = get_llm_config_manager(config)
            available_providers = llm_mgr.get_available_providers()
            primary_provider = config.get('llm', {}).get('primary_provider', 'gemini')
            logger.info(f"🎯 LLM Configuration: Primary={primary_provider}, Available={available_providers}")

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


# ENHANCED: Updated correction LLM to use new system
@st.cache_resource
def get_correction_llm(config):
    """Initializes and returns the LLM for correction using new provider system."""
    load_dotenv()

    if not config or not config.get('_CONFIG_VALID', False):
        logger.warning("Skipping correction LLM initialization: Invalid base config")
        return None

    try:
        # Try to use the new multi-provider system
        llm_manager = get_cypher_correction_llm(config)
        if llm_manager:
            logger.info("✅ Correction LLM initialized using new multi-provider system")
            return llm_manager
    except Exception as e:
        logger.warning(f"Could not initialize correction LLM with new system: {e}")

    # Fallback to legacy system for backward compatibility
    try:
        # Import after trying new system to avoid circular imports
        from llama_index.llms.gemini import Gemini

        model_name = config.get('LLM_MODEL')
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('LLM_API_KEY') or config.get('LLM_API_KEY')

        if not model_name or not api_key:
            logger.warning("Correction LLM model/API key missing. Correction disabled")
            return None

        masked_key = mask_sensitive_data(api_key)
        logger.info(f"Initializing legacy LlamaIndex LLM '{model_name}' for correction... (key: {masked_key})")

        if "gemini" in model_name.lower():
            llm = Gemini(model_name=model_name, api_key=api_key)
            logger.info("✅ Legacy correction LLM initialized successfully")
            return llm
        else:
            logger.error(f"Unsupported LLM provider for correction model: {model_name}")
            return None

    except ImportError as e:
        logger.error(f"ImportError for LlamaIndex LLM class: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize correction LLM: {e}")
        return None


@st.cache_resource
def get_mistral_client(api_key):
    """Initializes and returns a Mistral client."""
    load_dotenv()

    # UPDATED: Check environment variable first if no API key provided
    if not api_key:
        api_key = os.getenv('MISTRAL_API_KEY')

    if not api_key:
        logger.warning("Mistral API Key not provided. Mistral OCR will be disabled")
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
    load_dotenv()

    # UPDATED: Check environment variables first if parameters not provided
    if not uri:
        uri = os.getenv('NEO4J_URI')
    if not user:
        user = os.getenv('NEO4J_USER')
    if not password:
        password = os.getenv('NEO4J_PASSWORD')

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


# ENHANCED: Updated QA engine to use new LLM system
@st.cache_resource
def load_qa_engine(config, _correction_llm):
    """Initializes and returns the Enhanced GraphRAGQA engine with multi-provider LLM support."""
    logger.info("Initializing Enhanced GraphRAGQA Engine resource with multi-provider LLM support...")
    if not config or not config.get('_CONFIG_VALID', False):
        logger.error("Config invalid, cannot initialize Enhanced GraphRAGQA engine")
        return None

    try:
        # ENHANCED: Extract universal settings from config
        universal_config = config.get('universal', {})
        query_config = config.get('query_engine', {})

        # Get LLM configuration manager for the QA engine
        try:
            llm_config_manager = get_llm_config_manager(config)

            # Create primary LLM for QA (could be different from correction LLM)
            primary_provider = config.get('llm', {}).get('primary_provider', 'gemini')
            qa_llm_manager = llm_config_manager.get_llm_manager('llm')  # Uses primary LLM config

            logger.info(f"QA Engine using primary LLM provider: {primary_provider}")

        except Exception as llm_e:
            logger.warning(f"Could not initialize new LLM system for QA engine: {llm_e}, falling back to legacy")
            qa_llm_manager = None

        # Mask sensitive data for logging
        masked_password = mask_sensitive_data(config['NEO4J_PASSWORD'])
        masked_api_key = mask_sensitive_data(config['LLM_API_KEY'])
        logger.info(
            f"Initializing Enhanced QA engine with Neo4j password: {masked_password}, LLM key: {masked_api_key}")

        # UPDATED: Use EnhancedGraphRAGQA with optional new LLM system
        engine_params = {
            # Your existing parameters (unchanged)
            'neo4j_uri': config['NEO4J_URI'],
            'neo4j_user': config['NEO4J_USER'],
            'neo4j_password': config['NEO4J_PASSWORD'],
            'llm_instance_for_correction': _correction_llm,
            'llm_model': config['LLM_MODEL'],
            'llm_api_key': config['LLM_API_KEY'],
            'llm_base_url': config.get('LLM_BASE_URL'),
            'embedding_model_name': config['EMBEDDING_MODEL'],
            'chroma_path': config['CHROMA_PERSIST_PATH'],
            'collection_name': config['COLLECTION_NAME'],
            'db_name': config['DB_NAME'],
            'llm_config_extra': config.get('LLM_EXTRA_PARAMS', {}),
            'max_cypher_retries': config.get('max_cypher_retries', 2),

            # NEW ENHANCED PARAMETERS
            'enable_universal_patterns': universal_config.get('enable_universal_patterns', True),
            'manual_industry': universal_config.get('manual_industry', None),
            'pattern_confidence_threshold': universal_config.get('confidence_threshold', 0.6),
            'fuzzy_threshold': query_config.get('entity_linking_fuzzy_threshold', 70),
            'enable_query_caching': query_config.get('enable_query_caching', True)
        }

        # ENHANCED: Add LLM manager if available
        if qa_llm_manager:
            engine_params['llm_manager'] = qa_llm_manager

        engine = EnhancedGraphRAGQA(**engine_params)

        logger.info(f"Enhanced GraphRAGQA Engine resource initialized. Ready: {engine.is_ready()}")

        # Enhanced status checking
        if not engine.is_ready() and 'streamlit' in sys.modules:
            st.warning("Enhanced Q&A Engine initialized but may not be fully ready (check Neo4j/LLM status)", icon="⚠️")

        # Log enhancement status
        if hasattr(engine, 'is_enhanced') and engine.is_enhanced():
            industry_info = engine.get_industry_info()
            detected_industry = industry_info.get('detected_industry', 'unknown')
            logger.info(f"✅ Universal enhancements active. Detected industry: {detected_industry}")
        else:
            logger.info("ℹ️ Running in base mode (universal enhancements disabled or failed)")

        return engine
    except Exception as e:
        logger.error(f"Enhanced GraphRAGQA Engine initialization failed: {e}")
        if 'streamlit' in sys.modules:
            st.error(f"Failed to initialize Enhanced Q&A Engine: {e}")
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


# ENHANCED: New function to show LLM provider status
def show_llm_provider_status(config):
    """Display current LLM provider configuration and status."""
    try:
        llm_config_manager = get_llm_config_manager(config)

        # Get provider information
        available_providers = llm_config_manager.get_available_providers()
        primary_provider = config.get('llm', {}).get('primary_provider', 'gemini')
        fallback_provider = config.get('llm', {}).get('fallback_provider', 'openai')

        st.markdown("### 🤖 LLM Provider Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Primary Provider**")
            if primary_provider in available_providers:
                st.success(f"✅ {primary_provider.title()}")
            else:
                st.error(f"❌ {primary_provider.title()} (Not Available)")

        with col2:
            st.markdown("**Fallback Provider**")
            if fallback_provider in available_providers:
                st.success(f"✅ {fallback_provider.title()}")
            else:
                st.warning(f"⚠️ {fallback_provider.title()} (Not Available)")

        with col3:
            st.markdown("**Available Providers**")
            st.info(f"🎯 {len(available_providers)} providers ready")

        # Show detailed provider status
        if st.expander("📋 Detailed Provider Status", expanded=False):
            providers_config = config.get('llm', {}).get('providers', {})

            for provider_name, provider_config in providers_config.items():
                enabled = provider_config.get('enabled', False)
                model = provider_config.get('model', 'Unknown')

                # Check if API key is available
                api_key_available = False
                if provider_name == 'gemini':
                    api_key_available = bool(os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'))
                elif provider_name == 'openai':
                    api_key_available = bool(os.getenv('OPENAI_API_KEY'))
                elif provider_name == 'anthropic':
                    api_key_available = bool(os.getenv('ANTHROPIC_API_KEY'))
                elif provider_name == 'mistral':
                    api_key_available = bool(os.getenv('MISTRAL_API_KEY'))
                elif provider_name == 'ollama':
                    api_key_available = True  # Local model

                status_icon = "✅" if (enabled and api_key_available) else "❌"
                st.markdown(
                    f"{status_icon} **{provider_name.title()}**: {model} {'(Enabled)' if enabled else '(Disabled)'}")

                if enabled and not api_key_available:
                    st.warning(f"⚠️ API key missing for {provider_name}")

        return True

    except Exception as e:
        logger.error(f"Error showing LLM provider status: {e}")
        st.error(f"❌ Could not load LLM provider status: {e}")
        return False


def process_documents_batch(uploaded_files, enhanced_ocr_pipeline, save_to_disk=True):
    """
    MISSING FUNCTION IMPLEMENTATION: Process multiple documents with enhanced OCR pipeline.
    This function was imported but didn't exist - now implemented properly.

    Args:
        uploaded_files: List of Streamlit uploaded file objects
        enhanced_ocr_pipeline: The EnhancedOCRPipeline instance
        save_to_disk: Whether to save results to local storage

    Returns:
        List of processing results with comprehensive metadata
    """
    logger.info(f"🔄 Processing batch of {len(uploaded_files)} documents")

    # Use the existing batch processing function from processing_pipeline
    from src.utils.processing_pipeline import process_batch_with_enhanced_storage

    try:
        batch_results = process_batch_with_enhanced_storage(
            uploaded_files=uploaded_files,
            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
            save_to_disk=save_to_disk
        )

        # Add summary statistics
        successful = sum(1 for r in batch_results if r.get('success', False))
        total_text_length = sum(r.get('text_length', 0) for r in batch_results)

        logger.info(
            f"✅ Batch processing complete: {successful}/{len(uploaded_files)} successful, {total_text_length} total characters extracted")

        return batch_results

    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        # Return error results for each file
        return [
            {
                'success': False,
                'ocr_text': None,
                'error': str(e),
                'saved_files': None,
                'text_length': 0,
                'original_filename': getattr(f, 'name', 'unknown'),
                'file_type': getattr(f, 'type', 'unknown'),
                'processing_status': 'failed'
            }
            for f in uploaded_files
        ]


def main():
    """Sets up the main app configuration and landing page with enhanced LLM status."""

    # ADDED: Load .env file first
    load_dotenv()

    # Updated main title
    st.title("🤖 Document AI Assistant")

    # Load configuration early
    config = load_config()
    if not config or not config.get('_CONFIG_VALID', False):
        logger.critical("Halting app start due to invalid configuration")
        st.stop()

    # ENHANCED: Show LLM provider status
    show_llm_provider_status(config)

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

    * Use **Document Ingestion** to upload documents (PDF, TXT, images via LLM OCR). The AI will extract information, build a knowledge graph, and create vector embeddings.
    * Use **Knowledge Chat Assistant** to ask questions about the information contained within your processed documents.
    * Use **Data Extraction Validation** to monitor AI extraction quality and performance metrics.
    * Use **Processed Files Manager** to browse and manage your document archive.

    **Enhanced Multi-Provider LLM Support**: Configure any combination of Gemini, OpenAI, Claude, Mistral, or local Ollama models with automatic fallback support.

    **Security Note**: All API keys and passwords are automatically loaded from .env file and masked in application logs for security.
    """)


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s')
    main()
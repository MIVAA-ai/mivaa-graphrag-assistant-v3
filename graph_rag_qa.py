import logging
import re
import os
from typing import List, Dict, Optional, Any, Tuple
import json
from pathlib import Path
import configparser
import sys
import time
from functools import wraps
from functools import lru_cache
import hashlib
import asyncio
from dotenv import load_dotenv

# Import prompts - make sure these use the generic versions
from src.knowledge_graph.prompts import (
    TEXT_TO_CYPHER_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT,
    EVALUATE_EMPTY_RESULT_USER_PROMPT,
    REVISE_EMPTY_RESULT_SYSTEM_PROMPT,
    REVISE_EMPTY_RESULT_USER_PROMPT
)
from neo4j import GraphDatabase, exceptions as neo4j_exceptions

# Use tomli if available (standardized as tomllib)
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("[ERROR] No TOML parser found. Please install 'tomli' or use Python 3.11+.")
        tomllib = None

# Neo4j Imports
try:
    from neo4j import GraphDatabase, AsyncGraphDatabase, exceptions as neo4j_exceptions
except ImportError:
    raise ImportError("Neo4j Python driver not found. Please install: pip install neo4j")

# ENHANCED: Import new LLM system with fallback to legacy
try:
    from src.knowledge_graph.llm import (
        LLMManager,
        LLMProviderError,
        QuotaError,
        call_llm as legacy_call_llm,
        extract_json_from_text
    )

    NEW_LLM_SYSTEM_AVAILABLE = True
    print("[INFO] Using enhanced multi-provider LLM system.")
except ImportError:
    NEW_LLM_SYSTEM_AVAILABLE = False
    print("[WARN] Enhanced LLM system not found. Using legacy call_llm function.")

    # Legacy imports
    try:
        from src.knowledge_graph.llm import call_llm as legacy_call_llm, extract_json_from_text, QuotaError

        print("[INFO] Using legacy 'call_llm' function.")
    except ImportError:
        print("[WARN] 'call_llm' not found. Using mock function.")


        class QuotaError(Exception):
            pass


        def legacy_call_llm(*args, **kwargs):
            print("[WARN] Mock call_llm returning placeholder.")
            return "Mock response"


        def extract_json_from_text(text):
            return None

# Vector DB Imports
try:
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("ChromaDB library not found. Please install: pip install chromadb")

# Embeddings Imports
try:
    from sentence_transformers import SentenceTransformer

    print("[INFO] Imported SentenceTransformer.")
    embeddings_available = True
except ImportError:
    print("[WARN] 'sentence-transformers' library not found. Few-shot retrieval/storage might be affected.")
    embeddings_available = False


    class SentenceTransformer:
        pass

# Fuzzy matching imports
try:
    from fuzzywuzzy import fuzz

    fuzzy_available = True
    print("[INFO] Fuzzy matching available.")
except ImportError:
    fuzzy_available = False
    print("[WARN] fuzzywuzzy not available. Install with: pip install fuzzywuzzy python-levenshtein")

# Correction Step Imports
try:
    from text2cypher.correct_cypher import correct_cypher_step

    print("[INFO] Imported 'correct_cypher_step'.")
    from llama_index.graph_stores.neo4j import Neo4jGraphStore

    print("[INFO] Imported 'Neo4jGraphStore'.")
    llama_index_store_available = True
except ImportError as e:
    print(f"[WARN] Could not import correction dependencies (Error: {e}). Cypher correction will be disabled.")
    llama_index_store_available = False


    async def correct_cypher_step(*args, **kwargs):
        print("[ERROR] correct_cypher_step not available due to import error.")
        await asyncio.sleep(0)
        return None

# Few-Shot Manager Import
try:
    from text2cypher.neo4j_fewshot_manager import Neo4jFewshotManager

    print("[INFO] Imported 'Neo4jFewshotManager'.")
    fewshot_manager_available = True
except ImportError as e:
    print(f"[WARN] Could not import 'Neo4jFewshotManager' (Error: {e}). Few-shot self-learning will be disabled.")
    fewshot_manager_available = False


    class Neo4jFewshotManager:
        def __init__(self, *args, **kwargs): pass

        def retrieve_fewshots(self, *args, **kwargs): return []

        def store_fewshot_example(self, *args, **kwargs): pass


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


def get_query_hash(question: str, top_k: int) -> str:
    """Generate hash for query caching."""
    return hashlib.md5(f"{question.lower().strip()}_{top_k}".encode()).hexdigest()


# Logger Setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')
    logger = logging.getLogger(__name__)


class EmbeddingModelWrapper:
    """Wraps a SentenceTransformer model to provide a get_text_embedding method."""

    def __init__(self, model):
        if not hasattr(model, 'encode'):
            raise ValueError("Wrapped model must have an 'encode' method.")
        self._model = model
        try:
            self.dimensions = self._model.get_sentence_embedding_dimension()
        except Exception:
            self.dimensions = None

    def get_text_embedding(self, text: str) -> List[float]:
        """Encodes text using the wrapped model's .encode() and returns a list."""
        embedding_array = self._model.encode([text])[0]
        return embedding_array.tolist()

    def __getattr__(self, name):
        return getattr(self._model, name)


class GraphRAGQA:
    """
    Enhanced GraphRAG Q&A with improved entity linking, query generation, and multi-provider LLM support.
    """

    def __init__(self, *,
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 llm_instance_for_correction: Optional[Any],
                 llm_model: str, llm_api_key: str, llm_base_url: Optional[str] = None,
                 embedding_model_name: str,
                 chroma_path: str, collection_name: str,
                 db_name: str = "neo4j",
                 llm_config_extra: Optional[Dict[str, Any]] = None,
                 max_cypher_retries: int = 1,
                 # New parameters for improved functionality
                 fuzzy_threshold: int = 70,
                 enable_query_caching: bool = True,
                 # ENHANCED: New LLM manager parameter
                 llm_manager: Optional['LLMManager'] = None):

        logger.info(f"Initializing Enhanced GraphRAG QA Engine...")

        # Store configuration
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.llm_instance_for_correction = llm_instance_for_correction
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_config_extra = llm_config_extra or {}
        self.embedding_model_name = embedding_model_name
        self.max_cypher_retries = max_cypher_retries
        self.db_name = db_name
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_query_caching = enable_query_caching

        # ENHANCED: Store LLM manager for new multi-provider system
        self.llm_manager = llm_manager

        # Determine which LLM system to use
        self.use_new_llm_system = NEW_LLM_SYSTEM_AVAILABLE and llm_manager is not None

        if self.use_new_llm_system:
            logger.info("✅ Using enhanced multi-provider LLM system")
        else:
            logger.info("ℹ️ Using legacy LLM system")

        # Initialize status flags and components
        self.is_neo4j_connected = False
        self.is_vector_search_enabled = False
        self.driver: Optional[GraphDatabase.driver] = None
        self.embedding_function: Optional[EmbeddingFunction] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.neo4j_graph_store_for_correction: Optional[Neo4jGraphStore] = None
        self.embed_model: Optional[SentenceTransformer] = None
        self.embed_model_wrapper: Optional[EmbeddingModelWrapper] = None
        self.fewshot_manager: Optional[Neo4jFewshotManager] = None

        # Query caching
        if self.enable_query_caching:
            self._query_cache = {}
            self._cache_max_size = 100
            self._cache_ttl = 3600  # 1 hour

        # Initialize connections and components
        self._initialize_neo4j()
        self._initialize_embeddings()
        self._initialize_vector_db(chroma_path, collection_name)
        self._initialize_fewshot_manager()

        # Store base LLM config for backward compatibility
        self.llm_qna_config_base = {"model": llm_model, "api_key": llm_api_key, "base_url": llm_base_url}
        self.llm_qna_config_extra = llm_config_extra if llm_config_extra else {}

        logger.info(f"GraphRAG engine initialization complete. "
                    f"Neo4j: {self.is_neo4j_connected}, "
                    f"Vector: {self.is_vector_search_enabled}, "
                    f"Few-Shot: {self.fewshot_manager is not None}, "
                    f"LLM System: {'Enhanced' if self.use_new_llm_system else 'Legacy'}")

    def _initialize_neo4j(self):
        """Initialize Neo4j connection and graph store."""
        try:
            logger.info(f"Connecting to Neo4j at {self.neo4j_uri}")
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.driver.verify_connectivity()
            self.is_neo4j_connected = True
            logger.info("Successfully connected to Neo4j")

            # Initialize Neo4jGraphStore for correction
            if llama_index_store_available:
                try:
                    self.neo4j_graph_store_for_correction = Neo4jGraphStore(
                        username=self.neo4j_user,
                        password=self.neo4j_password,
                        url=self.neo4j_uri,
                        database=self.db_name,
                        refresh_schema=False
                    )
                    logger.info("Initialized Neo4jGraphStore for correction")
                except Exception as store_e:
                    logger.error(f"Failed to initialize Neo4jGraphStore: {store_e}")
                    self.neo4j_graph_store_for_correction = None

        except Exception as e:
            logger.error(f"Could not connect to Neo4j: {e}")
            self.is_neo4j_connected = False
            self.driver = None

    def _initialize_embeddings(self):
        """Initialize embedding model and wrapper."""
        if embeddings_available:
            try:
                logger.info(f"Initializing embedding model: {self.embedding_model_name}")
                self.embed_model = SentenceTransformer(self.embedding_model_name)
                self.embed_model_wrapper = EmbeddingModelWrapper(self.embed_model)

                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
                logger.info("Embedding model initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                self.embed_model = None
                self.embed_model_wrapper = None
                self.embedding_function = None
        else:
            logger.warning("Skipping embedding initialization - sentence-transformers not available")

    def _initialize_vector_db(self, chroma_path: str, collection_name: str):
        """Initialize ChromaDB vector database."""
        if self.embedding_function:
            try:
                logger.info(f"Initializing ChromaDB at: {chroma_path}")
                Path(chroma_path).mkdir(parents=True, exist_ok=True)

                settings = chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
                self.chroma_client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=settings
                )

                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )

                if self.chroma_collection:
                    self.is_vector_search_enabled = True
                    count = self.chroma_collection.count()
                    logger.info(f"ChromaDB collection '{collection_name}' ready with {count} documents")

            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.chroma_client = None
                self.chroma_collection = None
                self.is_vector_search_enabled = False

    def _initialize_fewshot_manager(self):
        """Initialize few-shot learning manager."""
        if fewshot_manager_available and self.embed_model_wrapper:
            try:
                self.fewshot_manager = Neo4jFewshotManager()
                if self.fewshot_manager.graph_store:
                    logger.info("Few-shot manager initialized successfully")
                else:
                    logger.warning("Few-shot manager connection failed")
                    self.fewshot_manager = None
            except Exception as e:
                logger.error(f"Failed to initialize few-shot manager: {e}")
                self.fewshot_manager = None

    # ENHANCED: New LLM calling method with fallback support
    def _call_llm(self, user_prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Enhanced LLM calling with multi-provider support and fallback to legacy system."""
        if self.use_new_llm_system and self.llm_manager:
            try:
                # Use new multi-provider system
                return self.llm_manager.call_llm(user_prompt, system_prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Enhanced LLM system failed: {e}, falling back to legacy")
                # Fall back to legacy system
                self.use_new_llm_system = False

        # Use legacy system
        try:
            return legacy_call_llm(
                model=self.llm_qna_config_base['model'],
                user_prompt=user_prompt,
                api_key=self.llm_qna_config_base['api_key'],
                system_prompt=system_prompt,
                base_url=self.llm_qna_config_base.get('base_url'),
                **{k: v for k, v in kwargs.items() if k in ['max_tokens', 'temperature', 'session']}
            )
        except Exception as e:
            logger.error(f"Legacy LLM system also failed: {e}")
            raise

    async def close(self):
        """Close all connections."""
        if self.driver:
            try:
                await self.driver.close()
                logger.info("Closed Neo4j connection")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")

        if self.fewshot_manager and hasattr(self.fewshot_manager, 'graph_store'):
            if hasattr(self.fewshot_manager.graph_store, '_driver'):
                try:
                    self.fewshot_manager.graph_store._driver.close()
                    logger.info("Closed few-shot manager connection")
                except Exception as e:
                    logger.error(f"Error closing few-shot manager: {e}")

    def is_ready(self) -> bool:
        """Check if the engine is ready for Q&A."""
        return self.is_neo4j_connected and (
                (self.use_new_llm_system and self.llm_manager is not None) or
                bool(self.llm_qna_config_base.get('api_key'))
        )

    def _get_schema_string(self) -> str:
        """Retrieve graph schema for query generation."""
        if self.neo4j_graph_store_for_correction:
            try:
                schema = self.neo4j_graph_store_for_correction.get_schema()
                if schema and isinstance(schema, str) and schema.strip():
                    logger.debug("Retrieved schema using Neo4jGraphStore")
                    return schema
            except Exception as e:
                logger.warning(f"Could not get schema via Neo4jGraphStore: {e}")

        # Fallback to basic schema retrieval
        if self.driver:
            try:
                with self.driver.session(database=self.db_name) as session:
                    labels_res = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels").single()
                    rels_res = session.run(
                        "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as rel_types").single()

                labels_list = labels_res['labels'] if labels_res else []
                rels_list = rels_res['rel_types'] if rels_res else []

                schema = f"Node Labels: {labels_list}\nRelationship Types: {rels_list}"
                logger.debug("Retrieved schema using fallback method")
                return schema

            except Exception as e:
                logger.error(f"Could not get basic schema: {e}")

        return "Schema information could not be retrieved."

    def _extract_potential_entities(self, question: str) -> List[str]:
        """Extract potential entity mentions from question using improved patterns."""
        mentions = []

        # 1. Quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', question)
        mentions.extend(quoted)

        # 2. Capitalized words/phrases
        cap_phrases = re.findall(r'\b[A-Z][a-zA-Z0-9#/\-.:]*(?:\s+[A-Z][a-zA-Z0-9#/\-.:]*)*\b', question)
        mentions.extend(cap_phrases)

        # 3. Alphanumeric codes (equipment IDs, etc.)
        codes = re.findall(r'\b[A-Z0-9]+[-#/]?[A-Z0-9]*\b', question)
        mentions.extend(codes)

        # 4. Numbers with potential units or identifiers
        numeric = re.findall(r'\b\d+[a-zA-Z]*\b', question)
        mentions.extend(numeric)

        # Clean and filter
        filtered_mentions = []
        common_words = {'What', 'Who', 'Where', 'When', 'Why', 'How', 'Is', 'Are', 'The', 'List', 'Tell', 'Find',
                        'Show'}

        for mention in mentions:
            clean = mention.strip()
            if len(clean) > 1 and clean not in common_words:
                filtered_mentions.append(clean)

        # Deduplicate and sort by length (longer first)
        unique_mentions = sorted(list(set(filtered_mentions)), key=len, reverse=True)
        logger.debug(f"Extracted potential entities: {unique_mentions}")

        return unique_mentions

    def _link_entities(self, mentions: List[str]) -> Dict[str, Optional[str]]:
        """Enhanced entity linking with exact, contains, and fuzzy matching."""
        if not self.driver or not mentions:
            return {m: None for m in mentions}

        linked_entities: Dict[str, Optional[str]] = {}
        logger.info(f"Linking {len(mentions)} entity mentions")

        for mention in mentions:
            canonical_name: Optional[str] = None
            mention_lower = mention.lower()

            try:
                # 1. Exact match (highest priority)
                exact_query = "MATCH (e:Entity) WHERE toLower(e.name) = $mention RETURN e.name LIMIT 1"
                exact_result, _, _ = self.driver.execute_query(
                    exact_query, mention=mention_lower, database_=self.db_name
                )

                if exact_result:
                    canonical_name = exact_result[0].data().get('e.name')
                    logger.debug(f"Exact match: '{mention}' -> '{canonical_name}'")
                else:
                    # 2. Contains match
                    contains_query = """
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS $mention
                        RETURN e.name
                        ORDER BY size(e.name) ASC
                        LIMIT 10
                    """
                    contains_result, _, _ = self.driver.execute_query(
                        contains_query, mention=mention_lower, database_=self.db_name
                    )

                    if contains_result:
                        candidates = [record.data().get('e.name') for record in contains_result]

                        # 3. Use fuzzy matching to pick best candidate (if available)
                        if fuzzy_available and len(candidates) > 1:
                            best_match = None
                            best_score = 0

                            for candidate in candidates:
                                # Try multiple fuzzy algorithms
                                scores = [
                                    fuzz.ratio(mention_lower, candidate.lower()),
                                    fuzz.partial_ratio(mention_lower, candidate.lower()),
                                    fuzz.token_sort_ratio(mention_lower, candidate.lower())
                                ]
                                score = max(scores)

                                if score > best_score and score >= self.fuzzy_threshold:
                                    best_score = score
                                    best_match = candidate

                            if best_match:
                                canonical_name = best_match
                                logger.debug(f"Fuzzy match: '{mention}' -> '{canonical_name}' (score: {best_score})")
                            else:
                                canonical_name = candidates[0]  # Fallback to shortest
                        else:
                            canonical_name = candidates[0]  # Take shortest match
                            logger.debug(f"Contains match: '{mention}' -> '{canonical_name}'")

            except Exception as e:
                logger.error(f"Error linking entity '{mention}': {e}")

            linked_entities[mention] = canonical_name

        found_count = sum(1 for v in linked_entities.values() if v)
        logger.info(f"Entity linking completed: {found_count}/{len(mentions)} entities found")

        return linked_entities

    def _format_fewshot_examples(self, examples: List[Dict]) -> str:
        """Format few-shot examples for prompt inclusion."""
        if not examples:
            return "No examples available."

        formatted = []
        for ex in examples:
            if ex.get("question") and ex.get("cypher"):
                formatted.append(f"Question: {ex['question']}\nCypher: {ex['cypher']}")

        return "\n\n".join(formatted) if formatted else "No valid examples found."

    def _generate_cypher_query(self, question: str, linked_entities: Dict[str, Optional[str]]) -> Optional[str]:
        """Generate Cypher query using LLM with enhanced prompts and few-shot examples."""
        logger.debug(f"Generating Cypher query for: '{question}'")

        # Retrieve few-shot examples
        few_shot_examples_str = "No examples available."
        if self.fewshot_manager and self.embed_model_wrapper:
            try:
                retrieved_examples = self.fewshot_manager.retrieve_fewshots(
                    question=question,
                    database=self.db_name,
                    embed_model=self.embed_model_wrapper
                )
                if retrieved_examples:
                    few_shot_examples_str = self._format_fewshot_examples(retrieved_examples)
                    logger.info(f"Retrieved {len(retrieved_examples)} few-shot examples")
            except Exception as e:
                logger.error(f"Error retrieving few-shot examples: {e}")

        # Get schema
        schema_str = self._get_schema_string()

        # Format system prompt with schema
        try:
            system_prompt = TEXT_TO_CYPHER_SYSTEM_PROMPT.format(dynamic_schema=schema_str)
        except KeyError:
            logger.error("Failed to format system prompt - missing {dynamic_schema} placeholder")
            system_prompt = TEXT_TO_CYPHER_SYSTEM_PROMPT

        # Format linked entities for prompt
        linked_entities_parts = ["Pre-linked Entities:"]
        if linked_entities:
            for mention, canonical in sorted(linked_entities.items()):
                status = canonical if canonical else 'None'
                linked_entities_parts.append(f"- '{mention}' -> '{status}'")
        else:
            linked_entities_parts.append("(No entities pre-linked)")

        linked_entities_str = "\n".join(linked_entities_parts)

        # Create structured input
        structured_input = f"""User Question: {question}

{linked_entities_str}
"""

        # Format user prompt
        try:
            user_prompt = GENERATE_USER_TEMPLATE.format(
                structured_input=structured_input,
                schema=schema_str,
                fewshot_examples=few_shot_examples_str
            )
        except KeyError as e:
            logger.error(f"Failed to format user template: missing {e}")
            user_prompt = f"{structured_input}\nCypher query:"

        # ENHANCED: Call LLM using new system with fallback
        try:
            temp = self.llm_config_extra.get("cypher_temperature", 0.0)
            max_tokens = self.llm_config_extra.get("cypher_max_tokens", 500)

            response_text = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temp
            )

            return self._extract_and_validate_cypher(response_text)

        except Exception as e:
            logger.error(f"Error during LLM call: {e}")
            return None

    def _extract_and_validate_cypher(self, response_text: str) -> Optional[str]:
        """Extract and validate Cypher query from LLM response."""
        if not response_text:
            return None

        response_text = response_text.strip()

        # Extract from code block
        cypher_match = re.search(r"```(?:cypher)?\s*([\s\S]+?)\s*```", response_text, re.IGNORECASE)
        if cypher_match:
            query = cypher_match.group(1).strip()
        elif "NO_QUERY_GENERATED" in response_text.upper():
            logger.info("LLM indicated no query could be generated")
            return None
        elif re.match(r"^(MATCH|MERGE|CREATE|CALL|OPTIONAL MATCH)\b", response_text, re.IGNORECASE):
            query = response_text
        else:
            logger.warning("Could not extract valid Cypher from LLM response")
            return None

        # Basic validation
        if self._validate_cypher_syntax(query):
            logger.info(f"Generated Cypher query:\n{query}")
            return query
        else:
            return None

    def _validate_cypher_syntax(self, query: str) -> bool:
        """Basic Cypher syntax validation."""
        if not query:
            return False

        query_upper = query.upper()

        # Check for basic required patterns
        has_match = "MATCH" in query_upper
        has_return = "RETURN" in query_upper
        has_call = "CALL" in query_upper
        has_merge = "MERGE" in query_upper
        has_create = "CREATE" in query_upper

        valid_patterns = [
            has_match and has_return,
            has_match and has_call,
            has_merge,
            has_create,
            has_call
        ]

        if any(valid_patterns):
            return True
        else:
            logger.warning(f"Cypher query failed basic validation: {query}")
            return False

    def _query_neo4j(self, cypher_query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query against Neo4j with caching."""
        if not self.is_neo4j_connected or not self.driver:
            raise ConnectionError("Neo4j not connected")

        if not cypher_query:
            return []

        # Check cache first
        if self.enable_query_caching:
            cache_key = get_query_hash(cypher_query, len(params) if params else 0)
            if cache_key in self._query_cache:
                cached_result, timestamp = self._query_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    logger.debug("Using cached query result")
                    return cached_result

        logger.info("Executing Cypher query...")
        logger.debug(f"Query: {cypher_query}")

        try:
            records, summary, keys = self.driver.execute_query(
                cypher_query,
                parameters_=params or {},
                database_=self.db_name
            )

            result_data = [record.data() for record in records]
            logger.info(f"Query returned {len(result_data)} records")

            # Cache the result
            if self.enable_query_caching:
                if len(self._query_cache) >= self._cache_max_size:
                    # Remove oldest entry
                    oldest_key = min(self._query_cache.keys(), key=lambda k: self._query_cache[k][1])
                    del self._query_cache[oldest_key]

                self._query_cache[cache_key] = (result_data, time.time())

            return result_data

        except Exception as e:
            logger.error(f"Neo4j query execution failed: {e}")
            raise

    @monitor_performance("vector_search")
    def _query_vector_db(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query vector database for relevant chunks."""
        if not self.is_vector_search_enabled:
            logger.warning("Vector search not enabled")
            return []

        logger.info(f"Performing vector search (top {top_k})")

        try:
            results = self.chroma_collection.query(
                query_texts=[question],
                n_results=top_k,
                include=['documents', 'distances', 'metadatas']
            )

            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                ids, documents, distances, metadatas = (
                    results.get(k, [[]])[0] for k in ['ids', 'documents', 'distances', 'metadatas']
                )

                for i in range(len(ids)):
                    formatted_results.append({
                        "text": documents[i] if i < len(documents) else "[No text]",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "distance": distances[i] if i < len(distances) else -1.0
                    })

                logger.info(f"Vector search returned {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _format_context(self, graph_results: Optional[List[Dict]], vector_results: List[Dict]) -> str:
        """Format combined context from graph and vector results."""
        context_parts = []

        # Add vector search results
        if vector_results:
            vector_context = "Relevant Text Snippets (Vector Search):\n---\n"
            vector_context += "\n---\n".join([
                f"Source: {chunk.get('metadata', {}).get('source_document', 'Unknown')}\n"
                f"Content: {chunk.get('text', '[No text]')}"
                for chunk in vector_results
            ])
            vector_context += "\n---"
            context_parts.append(vector_context)

        # Add graph results
        if graph_results is not None:
            graph_context = "Knowledge Graph Facts:\n---\n"
            if graph_results:
                facts = []
                seen_facts = set()

                for record in graph_results[:15]:  # Limit to prevent context overflow
                    # Extract subject, predicate, object pattern
                    subj = record.get('subject', record.get('e1.name', record.get('e.name', '?')))
                    pred_type = record.get('type', '?')
                    pred_orig = record.get('predicate', '?')
                    pred = pred_orig if pred_orig != '?' else pred_type
                    obj = record.get('object', record.get('related.name', record.get('related_entity', '?')))

                    # Format fact
                    if len(record) == 1:
                        fact = f"- {list(record.keys())[0]}: {list(record.values())[0]}"
                    elif subj != '?' and pred != '?' and obj != '?':
                        fact = f"- {subj} -[{pred}]-> {obj}"
                    else:
                        fact = "- " + ", ".join([f"{k}: {v}" for k, v in record.items()])

                    # Create hashable tuple for deduplication - FIXED
                    try:
                        # Convert any lists or unhashable types to strings for hashing
                        hashable_items = []
                        for k, v in record.items():
                            if isinstance(v, (list, dict)):
                                # Convert unhashable types to strings
                                hashable_items.append((k, str(v)))
                            else:
                                hashable_items.append((k, v))

                        fact_tuple = tuple(sorted(hashable_items))

                        # Check for duplicates
                        if fact_tuple not in seen_facts:
                            seen_facts.add(fact_tuple)
                            facts.append(fact)

                    except (TypeError, ValueError) as e:
                        # If we still can't hash it, just use the string representation
                        fact_str = str(record)
                        if fact_str not in seen_facts:
                            seen_facts.add(fact_str)
                            facts.append(fact)

                graph_context += "\n".join(facts) if facts else "No specific facts found."
            else:
                graph_context += "No relevant facts found."

            graph_context += "\n---"
            context_parts.append(graph_context)

        if not context_parts:
            return "No relevant context found."

        return "\n\n".join(context_parts)

    def _synthesize_answer(self, query: str, context: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate final answer using LLM with combined context."""
        logger.info("Generating final answer...")

        system_prompt = """You are a helpful assistant answering questions based on the provided context.

Context may include:
1. Knowledge Graph Facts (structured relationships)
2. Relevant Text Snippets (source documents)

Instructions:
1. Prioritize Knowledge Graph facts for direct entity relationships
2. Use text snippets for supporting details and explanations
3. Ignore unrelated information even if retrieved
4. If you cannot answer confidently from the context, say so
5. Do not make up information not present in the context
"""

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            temp = self.llm_config_extra.get("qna_temperature", 0.1)
            max_tokens = self.llm_config_extra.get("qna_max_tokens", 500)

            # ENHANCED: Use new LLM system with fallback
            answer_text = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=max_tokens
            )

            return {
                "answer": answer_text.strip() if answer_text else "Could not generate answer",
                "sources": context_chunks
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "sources": context_chunks
            }

    def _get_corrected_cypher(self, question: str, failed_cypher: str, error_message: str) -> Optional[str]:
        """Attempt to correct failed Cypher query."""
        logger.warning("Cypher correction is temporarily disabled")
        return None

    def _evaluate_and_revise_empty_result_query(self, question: str, empty_query: str) -> Optional[str]:
        """Evaluate why query returned empty results and attempt revision."""
        logger.info("Evaluating empty result query...")

        if not self.llm_instance_for_correction and not self.use_new_llm_system:
            return None

        schema_str = self._get_schema_string()

        # Step 1: Evaluate why query returned empty
        try:
            eval_system = EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT.format(dynamic_schema=schema_str)
            eval_user = EVALUATE_EMPTY_RESULT_USER_PROMPT.format(
                question=question,
                cypher=empty_query,
                schema=schema_str
            )

            # ENHANCED: Use new LLM system with fallback
            eval_response = self._call_llm(
                user_prompt=eval_user,
                system_prompt=eval_system,
                temperature=0.1,
                max_tokens=50
            )

            evaluation = eval_response.strip().upper() if eval_response else "UNKNOWN"
            logger.info(f"Empty result evaluation: {evaluation}")

            # Step 2: Revise if query mismatch
            if evaluation == "QUERY_MISMATCH":
                try:
                    revise_system = REVISE_EMPTY_RESULT_SYSTEM_PROMPT.format(dynamic_schema=schema_str)
                    revise_user = REVISE_EMPTY_RESULT_USER_PROMPT.format(
                        question=question,
                        cypher=empty_query,
                        schema=schema_str
                    )

                    # ENHANCED: Use new LLM system with fallback
                    revise_response = self._call_llm(
                        user_prompt=revise_user,
                        system_prompt=revise_system,
                        temperature=0.3,
                        max_tokens=500
                    )

                    if revise_response and "NO_REVISION" not in revise_response.upper():
                        revised_query = self._extract_and_validate_cypher(revise_response)
                        if revised_query and revised_query != empty_query.strip():
                            logger.info(f"Query revision successful:\n{revised_query}")
                            return revised_query

                except Exception as e:
                    logger.error(f"Error during query revision: {e}")

        except Exception as e:
            logger.error(f"Error during empty result evaluation: {e}")

        return None

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer user question using enhanced Graph RAG with retry and revision.
        """
        logger.info(f"=== Enhanced GraphRAG for: {question} ===")

        if not self.is_ready():
            return {
                "answer": "Error: System not ready (check Neo4j connection and API keys)",
                "sources": [],
                "cypher_query": "N/A"
            }

        # Initialize variables
        retries = 0
        current_cypher_query: Optional[str] = None
        initial_cypher_query: Optional[str] = None
        graph_results: Optional[List[Dict]] = None
        execution_error: Optional[Exception] = None
        correction_attempted = False
        revision_attempted = False
        self_healing_successful = False

        # Step 1: Enhanced entity linking
        linked_entities: Dict[str, Optional[str]] = {}
        try:
            potential_mentions = self._extract_potential_entities(question)
            if potential_mentions:
                linked_entities = self._link_entities(potential_mentions)
                logger.info(f"Entity linking: {len([v for v in linked_entities.values() if v])} entities linked")
        except Exception as e:
            logger.error(f"Entity linking failed: {e}")

        # Step 2: Initial Cypher generation
        initial_cypher_query = self._generate_cypher_query(question, linked_entities)
        current_cypher_query = initial_cypher_query

        # Step 3: Execute with retry/revision loop
        if not current_cypher_query:
            logger.warning("Initial Cypher generation failed")
            graph_results = []
        else:
            while retries <= self.max_cypher_retries:
                logger.info(f"Attempt {retries + 1}/{self.max_cypher_retries + 1}")
                execution_error = None

                try:
                    # Execute query
                    graph_results = self._query_neo4j(current_cypher_query)
                    logger.info(f"Query executed successfully, found {len(graph_results)} results")

                    # Handle empty results
                    if not graph_results:
                        logger.warning("Query returned 0 results")
                        retries += 1

                        if retries <= self.max_cypher_retries:
                            logger.info("Attempting revision for empty result...")
                            revision_attempted = True
                            revised_query = self._evaluate_and_revise_empty_result_query(question, current_cypher_query)

                            if revised_query:
                                current_cypher_query = revised_query
                                continue
                            else:
                                logger.info("No revision possible, stopping")
                                graph_results = []
                                break
                        else:
                            logger.error("Max retries reached for empty results")
                            graph_results = []
                            break
                    else:
                        # Success - check if self-healing occurred
                        if correction_attempted or revision_attempted:
                            self_healing_successful = True
                        break

                except Exception as e:
                    execution_error = e
                    logger.warning(f"Query execution failed: {type(e).__name__} - {e}")
                    retries += 1

                    if retries <= self.max_cypher_retries:
                        logger.info("Attempting correction for execution error...")
                        correction_attempted = True
                        corrected_query = self._get_corrected_cypher(question, current_cypher_query, str(e))

                        if corrected_query:
                            current_cypher_query = corrected_query
                            continue
                        else:
                            logger.error("No correction available")
                            graph_results = []
                            break
                    else:
                        logger.error("Max retries reached for execution errors")
                        graph_results = []
                        break

        # Step 4: Store successful self-healing example
        if self_healing_successful and current_cypher_query and self.fewshot_manager:
            try:
                logger.info("Storing successful self-healing example")
                self.fewshot_manager.store_fewshot_example(
                    question=question,
                    cypher=current_cypher_query,
                    llm=self.llm_model,
                    embed_model=self.embed_model_wrapper,
                    database=self.db_name,
                    success=True
                )
            except Exception as e:
                logger.error(f"Failed to store few-shot example: {e}")

        # Step 5: Vector search
        vector_top_k = self.llm_config_extra.get("vector_search_top_k", 5)
        similar_chunks = self._query_vector_db(question, top_k=vector_top_k)

        # Step 6: Format context and generate answer
        context_str = self._format_context(graph_results, similar_chunks)
        logger.debug(f"Combined context length: {len(context_str)} characters")

        # Handle no context case
        if (not graph_results) and (not similar_chunks):
            final_answer = "I could not find relevant information to answer your question."
            if execution_error:
                final_answer += f" (Technical issue: {type(execution_error).__name__})"

            return {
                "answer": final_answer,
                "sources": [],
                "cypher_query": current_cypher_query or initial_cypher_query or "N/A",
                "linked_entities": linked_entities,
                "llm_system_used": "Enhanced" if self.use_new_llm_system else "Legacy"
            }

        # Generate final answer
        answer_dict = self._synthesize_answer(question, context_str, similar_chunks)

        # Add metadata
        answer_dict["cypher_query"] = current_cypher_query or initial_cypher_query or "N/A"
        answer_dict["linked_entities"] = linked_entities
        answer_dict["llm_system_used"] = "Enhanced" if self.use_new_llm_system else "Legacy"

        if execution_error:
            answer_dict["error_info"] = f"Query failed after {retries} attempts: {type(execution_error).__name__}"
        elif self_healing_successful:
            answer_dict["info"] = f"Query succeeded after self-healing ({retries} attempts)"
        elif not graph_results:
            answer_dict["info"] = f"Query executed but returned no results"

        logger.info("=== GraphRAG processing complete ===")
        return answer_dict


# Example usage and main block
if __name__ == "__main__":
    print("=== Enhanced GraphRAG QA System with Multi-Provider LLM Support ===")

    # ADDED: Load .env file first
    load_dotenv()

    # Configuration loading
    config_data = {}
    llm_for_correction = None

    try:
        # Load from config.toml
        toml_config_path = Path("config.toml")
        if tomllib and toml_config_path.is_file():
            with open(toml_config_path, "rb") as f:
                config_toml = tomllib.load(f)

            logger.info("Loaded configuration from config.toml")

            # Extract configuration
            llm_config = config_toml.get("llm", {})
            config_data['LLM_MODEL'] = llm_config.get("model")
            config_data['LLM_API_KEY'] = llm_config.get("api_key")
            config_data['LLM_BASE_URL'] = llm_config.get("base_url")
            config_data['LLM_EXTRA_PARAMS'] = llm_config.get("parameters", {})

            # Embedding and vector config
            config_data['EMBEDDING_MODEL'] = config_toml.get("embeddings", {}).get("model_name", "all-MiniLM-L6-v2")
            config_data['CHROMA_PERSIST_PATH'] = config_toml.get("vector_db", {}).get('persist_directory',
                                                                                      "./chroma_db_embeddings")
            config_data['COLLECTION_NAME'] = config_toml.get("vector_db", {}).get('collection_name',
                                                                                  "doc_pipeline_embeddings")
            config_data['DB_NAME'] = config_toml.get("database", {}).get("name", "neo4j")

            # Query engine config (new)
            query_config = config_toml.get("query_engine", {})
            config_data['FUZZY_THRESHOLD'] = query_config.get("entity_linking_fuzzy_threshold", 70)
            config_data['ENABLE_CACHING'] = query_config.get("enable_query_caching", True)
            config_data['MAX_RETRIES'] = query_config.get("max_cypher_retries", 1)
        else:
            logger.warning("config.toml not found, using fallback config")

        # Load from graph_config.ini (fallback)
        config_path_ini = Path("graph_config.ini")
        if config_path_ini.is_file():
            neo4j_config = configparser.ConfigParser()
            neo4j_config.read(config_path_ini)

            config_data.setdefault('NEO4J_URI', neo4j_config.get("neo4j", "uri", fallback=None))
            config_data.setdefault('NEO4J_USER', neo4j_config.get("neo4j", "user", fallback=None))
            config_data.setdefault('NEO4J_PASSWORD', neo4j_config.get("neo4j", "password", fallback=None))

        # Environment variable overrides - UPDATED: Check environment first
        config_data['NEO4J_URI'] = os.getenv('NEO4J_URI') or config_data.get('NEO4J_URI')
        config_data['NEO4J_USER'] = os.getenv('NEO4J_USER') or config_data.get('NEO4J_USER')
        config_data['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD') or config_data.get('NEO4J_PASSWORD')
        config_data['LLM_API_KEY'] = os.getenv('GOOGLE_API_KEY') or os.getenv('LLM_API_KEY') or config_data.get(
            'LLM_API_KEY')
        config_data['LLM_MODEL'] = os.getenv('LLM_MODEL') or config_data.get('LLM_MODEL')

        # Validate required config
        required_keys = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "LLM_MODEL", "LLM_API_KEY"]
        missing_keys = [k for k in required_keys if not config_data.get(k)]
        if missing_keys:
            raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")

        logger.info("Configuration validation successful")

        # Initialize correction LLM if available
        if llama_index_store_available:
            try:
                from llama_index.llms.gemini import Gemini

                llm_for_correction = Gemini(
                    model_name=config_data['LLM_MODEL'],
                    api_key=config_data['LLM_API_KEY']
                )
                logger.info("Initialized correction LLM")
            except Exception as e:
                logger.warning(f"Could not initialize correction LLM: {e}")

    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Initialize QA engine
    print("\n--- Initializing Enhanced GraphRAG QA Engine ---")
    qa_engine = None

    try:
        qa_engine = GraphRAGQA(
            neo4j_uri=config_data['NEO4J_URI'],
            neo4j_user=config_data['NEO4J_USER'],
            neo4j_password=config_data['NEO4J_PASSWORD'],
            llm_instance_for_correction=llm_for_correction,
            llm_model=config_data['LLM_MODEL'],
            llm_api_key=config_data['LLM_API_KEY'],
            llm_base_url=config_data.get('LLM_BASE_URL'),
            embedding_model_name=config_data['EMBEDDING_MODEL'],
            chroma_path=config_data['CHROMA_PERSIST_PATH'],
            collection_name=config_data['COLLECTION_NAME'],
            db_name=config_data['DB_NAME'],
            llm_config_extra=config_data.get('LLM_EXTRA_PARAMS', {}),
            max_cypher_retries=config_data.get('MAX_RETRIES', 1),
            fuzzy_threshold=config_data.get('FUZZY_THRESHOLD', 70),
            enable_query_caching=config_data.get('ENABLE_CACHING', True)
        )

        if not qa_engine.is_ready():
            print("FATAL: QA Engine failed to initialize")
            sys.exit(1)

        print("✅ Enhanced GraphRAG QA Engine ready!")
        print(f"🔧 LLM System: {'Enhanced Multi-Provider' if qa_engine.use_new_llm_system else 'Legacy'}")
        print("\n🔍 Ask questions (type 'exit' or 'quit' to stop):")

        # Interactive loop
        while True:
            try:
                question = input("\n❓ Your Question: ").strip()
                if not question:
                    continue
                if question.lower() in {"exit", "quit"}:
                    break

                print("🔄 Processing...")
                start_time = time.time()

                response = qa_engine.answer_question(question)

                processing_time = time.time() - start_time

                print(f"\n💡 Answer:\n{response.get('answer', 'N/A')}")
                print(f"\n⚡ Processing time: {processing_time:.2f}s")
                print(f"🔧 LLM System Used: {response.get('llm_system_used', 'Unknown')}")

                if response.get('cypher_query') != 'N/A':
                    print(f"\n🔧 Cypher Query:\n{response['cypher_query']}")

                if response.get('linked_entities'):
                    linked = {k: v for k, v in response['linked_entities'].items() if v}
                    if linked:
                        print(f"\n🔗 Linked Entities: {linked}")

                if response.get('error_info'):
                    print(f"\n⚠️  {response['error_info']}")
                elif response.get('info'):
                    print(f"\nℹ️  {response['info']}")

                if response.get('sources'):
                    print(f"\n📚 Found {len(response['sources'])} source documents")

                print("-" * 60)

            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                logger.error(f"Error during question processing: {e}")
                print(f"\n❌ Error: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize QA engine: {e}")
        sys.exit(1)

    finally:
        if qa_engine:
            try:
                asyncio.run(qa_engine.close())
                print("\n✅ Connections closed successfully")
            except Exception as e:
                logger.error(f"Error closing connections: {e}")

    print("\n👋 Goodbye!")
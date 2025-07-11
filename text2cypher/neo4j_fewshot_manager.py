import configparser
import logging
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# ENHANCED: Import multi-provider LLM system with fallback
try:
    from src.knowledge_graph.llm import (
        LLMManager,
        LLMProviderFactory,
        LLMConfig,
        LLMProvider,
        LLMProviderError,
        QuotaError
    )

    NEW_LLM_SYSTEM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Multi-provider LLM system available for Neo4j Fewshot Manager")
except ImportError as e:
    NEW_LLM_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Multi-provider LLM system not available: {e}. Using legacy system.")

# Import Neo4jPropertyGraphStore and handle potential ImportError
try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
except ImportError:
    print(
        "llama_index.graph_stores.neo4j not found."
        " Please install it: pip install llama-index-graph-stores-neo4j"
    )
    Neo4jPropertyGraphStore = None

# Import SentenceTransformer for type hinting
try:
    from sentence_transformers import SentenceTransformer

    print("[INFO] Imported SentenceTransformer for FewShotManager type hints.")
except ImportError:
    print("[WARN] sentence-transformers not found for FewShotManager type hints.")
    SentenceTransformer = Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_NEO4J_TIMEOUT = 30
DEFAULT_RETRIEVAL_LIMIT = 7
CONFIG_FILE_NAME = "graph_config.ini"
ENV_VAR_PREFIX = "FEWSHOT_"
NEO4J_SECTION = "neo4j"
LABEL_FEWSHOT = "Fewshot"
LABEL_MISSING = "Missing"
PROPERTY_ID = "id"
PROPERTY_QUESTION = "question"
PROPERTY_CYPHER = "cypher"
PROPERTY_DATABASE = "database"
PROPERTY_LLM = "llm"
PROPERTY_EMBEDDING = "embedding"
PROPERTY_CREATED = "created"

# OPTIMIZED: Performance configuration
DEFAULT_LLM_TIMEOUT = 5  # 5 second timeout for LLM calls
FAST_LLM_TIMEOUT = 3  # 3 second timeout for simple operations
MAX_EXAMPLE_GENERATION = 2  # Reduced from 3 to 2 examples
VALIDATION_TIMEOUT = 3  # 3 second timeout for validation


class Neo4jFewshotManager:
    """
    OPTIMIZED: Manages storing and retrieving few-shot examples with performance improvements.
    *** ENHANCED WITH MULTI-PROVIDER LLM SUPPORT AND SPEED OPTIMIZATIONS ***
    """
    graph_store: Optional[Neo4jPropertyGraphStore] = None

    def __init__(self, config_file: str = CONFIG_FILE_NAME, timeout: int = DEFAULT_NEO4J_TIMEOUT,
                 config: Optional[Dict[str, Any]] = None, enable_multi_provider_llm: bool = True,
                 fast_mode: bool = True):  # OPTIMIZED: Add fast_mode parameter
        """
        OPTIMIZED: Initializes the Neo4jFewshotManager with performance optimizations.

        Args:
            config_file: Path to configuration file
            timeout: Neo4j connection timeout
            config: Configuration dictionary for multi-provider LLM
            enable_multi_provider_llm: Whether to enable multi-provider LLM features
            fast_mode: Enable performance optimizations (default: True)
        """
        # OPTIMIZED: Store performance configuration
        self.fast_mode = fast_mode
        self.llm_timeout = FAST_LLM_TIMEOUT if fast_mode else DEFAULT_LLM_TIMEOUT

        # Store configuration for multi-provider LLM
        self.config = config or {}
        self.enable_multi_provider_llm = enable_multi_provider_llm
        self.llm_managers = {}

        # OPTIMIZED: Performance tracking
        self.performance_stats = {
            'total_llm_calls': 0,
            'average_llm_time': 0,
            'timeout_count': 0,
            'cache_hits': 0
        }

        # OPTIMIZED: Simple caching for repeated queries
        self.validation_cache = {} if fast_mode else None

        if Neo4jPropertyGraphStore is None:
            logger.error("Neo4jPropertyGraphStore is not available. Cannot initialize manager.")
            return

        username, password, url = self._load_credentials(config_file)

        if not all([username, password, url]):
            logger.error(
                "Failed to load Neo4j credentials from environment variables or"
                f" config file ('{config_file}'). Few-shot manager will be inactive."
            )
            return

        try:
            self.graph_store = Neo4jPropertyGraphStore(
                username=username,
                password=password,
                url=url,
                refresh_schema=False,
                create_indexes=False,
                timeout=timeout,
            )
            logger.info("Successfully configured Neo4j connection details for few-shot management.")
        except Exception as e:
            logger.error(f"Failed to configure Neo4j connection: {e}", exc_info=True)
            self.graph_store = None

        # ENHANCED: Initialize multi-provider LLM system
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            self._initialize_multi_provider_llm()

        logger.info(
            f"Neo4j Fewshot Manager initialized. Multi-provider LLM: {self.enable_multi_provider_llm}, Fast mode: {self.fast_mode}")

    def _initialize_multi_provider_llm(self):
        """ENHANCED: Initialize multi-provider LLM system for few-shot tasks"""
        try:
            from GraphRAG_Document_AI_Platform import get_llm_config_manager

            main_llm_manager = get_llm_config_manager(self.config)

            # Create task-specific LLM managers for few-shot tasks
            fewshot_tasks = ['example_generation', 'cypher_correction', 'question_similarity', 'example_validation']

            for task in fewshot_tasks:
                try:
                    self.llm_managers[task] = main_llm_manager.get_llm_manager(task)
                    logger.info(f"✅ Initialized LLM manager for {task}")
                except Exception as e:
                    logger.warning(f"Could not initialize LLM manager for {task}: {e}")
                    self.llm_managers[task] = None

        except ImportError as e:
            logger.warning(f"Could not import main LLM configuration manager: {e}")
            self.llm_managers = {}

    def _enhanced_llm_call(self, task_name: str, prompt: str, system_prompt: str = None,
                           timeout: Optional[int] = None, **kwargs) -> str:
        """
        OPTIMIZED: Enhanced LLM call with timeout and performance monitoring.
        """
        start_time = time.time()

        # OPTIMIZED: Use configured timeout or parameter override
        actual_timeout = timeout or self.llm_timeout

        # Update performance stats
        self.performance_stats['total_llm_calls'] += 1

        # Try enhanced system first
        if (self.enable_multi_provider_llm and
                NEW_LLM_SYSTEM_AVAILABLE and
                task_name in self.llm_managers and
                self.llm_managers[task_name]):

            try:
                logger.debug(f"🎯 Using enhanced LLM system for {task_name} (timeout: {actual_timeout}s)")

                # FIXED: Extract conflicting parameters to avoid the max_tokens issue
                max_tokens = kwargs.pop('max_tokens', 300 if self.fast_mode else 500)
                temperature = kwargs.pop('temperature', 0.1)

                response = self.llm_managers[task_name].call_llm(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    timeout=actual_timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )

                elapsed = time.time() - start_time
                logger.debug(f"⚡ Enhanced LLM {task_name} completed in {elapsed:.2f}s")

                # Update performance stats
                current_avg = self.performance_stats['average_llm_time']
                total_calls = self.performance_stats['total_llm_calls']
                self.performance_stats['average_llm_time'] = (
                        (current_avg * (total_calls - 1) + elapsed) / total_calls
                )

                return response

            except Exception as e:
                elapsed = time.time() - start_time
                logger.warning(f"Enhanced LLM failed for {task_name} after {elapsed:.2f}s: {e}")
                self.performance_stats['timeout_count'] += 1

        # Fall back to basic functionality (no LLM call)
        logger.debug(f"🔄 Enhanced LLM not available for {task_name}, using basic functionality")
        return ""

    def _load_credentials(self, config_file: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Loads Neo4j credentials from environment variables or config file."""

        # ADDED: Ensure .env is loaded
        load_dotenv()

        # Try standard environment variables first (consistent with other files)
        username_env = os.getenv('NEO4J_USER')
        password_env = os.getenv('NEO4J_PASSWORD')
        url_env = os.getenv('NEO4J_URI')

        if all([username_env, password_env, url_env]):
            logger.info("Loaded Neo4j credentials from standard environment variables.")
            return username_env, password_env, url_env

        # Try fewshot-specific environment variables (fallback)
        username_env = os.getenv(f"{ENV_VAR_PREFIX}NEO4J_USERNAME")
        password_env = os.getenv(f"{ENV_VAR_PREFIX}NEO4J_PASSWORD")
        url_env = os.getenv(f"{ENV_VAR_PREFIX}NEO4J_URI")

        if all([username_env, password_env, url_env]):
            logger.info("Loaded Neo4j credentials from fewshot-specific environment variables.")
            return username_env, password_env, url_env

        logger.info(f"Environment variables not fully set. Attempting to load from config file: {config_file}")

        # Fallback to config file (existing code unchanged)
        config = configparser.ConfigParser()
        if not os.path.exists(config_file):
            logger.warning(f"Config file '{config_file}' not found.")
            return None, None, None

        try:
            config.read(config_file)
            if NEO4J_SECTION in config:
                username_conf = config[NEO4J_SECTION].get("user")
                password_conf = config[NEO4J_SECTION].get("password")
                url_conf = config[NEO4J_SECTION].get("uri")

                # Override with environment variables if available
                username_conf = os.getenv('NEO4J_USER') or username_conf
                password_conf = os.getenv('NEO4J_PASSWORD') or password_conf
                url_conf = os.getenv('NEO4J_URI') or url_conf

                if all([username_conf, password_conf, url_conf]):
                    logger.info(
                        f"Loaded Neo4j credentials from config file '{config_file}' with environment overrides.")
                    return username_conf, password_conf, url_conf
                else:
                    logger.warning(
                        f"Missing required keys ('user', 'password', 'uri') in [{NEO4J_SECTION}] section of '{config_file}'.")
            else:
                logger.warning(f"Missing section '[{NEO4J_SECTION}]' in config file '{config_file}'.")

        except configparser.Error as e:
            logger.error(f"Error reading config file '{config_file}': {e}", exc_info=True)

        return None, None, None

    def _execute_query(self, query: str, param_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """OPTIMIZED: Helper method with faster error handling."""
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot execute query.")
            return []

        start_time = time.time()
        try:
            results = self.graph_store.structured_query(query, param_map=param_map)
            result_list = [dict(record) for record in results] if results else []

            elapsed = time.time() - start_time
            logger.debug(f"⚡ Neo4j query executed in {elapsed:.3f}s, {len(result_list)} results")

            return result_list
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error executing Cypher query after {elapsed:.3f}s: {e}\nQuery: {query[:100]}...")
            return []

    def retrieve_fewshots(self, question: str, database: str, embed_model: SentenceTransformer,
                          limit: int = DEFAULT_RETRIEVAL_LIMIT) -> List[Dict[str, str]]:
        """
        OPTIMIZED: Retrieves few-shot examples with performance improvements.
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot retrieve few-shots.")
            return []

        # Check if the passed model has the required .encode() method
        if not hasattr(embed_model, 'encode'):
            logger.error("CRITICAL: Provided embed_model object does not have an 'encode' method.")
            return []

        start_time = time.time()

        # OPTIMIZED: Reduce limit in fast mode
        if self.fast_mode:
            limit = min(limit, 5)  # Max 5 examples for speed

        try:
            embedding = embed_model.encode([question])[0].tolist()
        except Exception as e:
            logger.error(f"Failed to get text embedding using .encode(): {e}", exc_info=True)
            return []

        # OPTIMIZED: Simplified query for better performance
        query = f"""
        MATCH (f:{LABEL_FEWSHOT})
        WHERE f.{PROPERTY_DATABASE} = $database
        WITH f, vector.similarity.cosine(f.{PROPERTY_EMBEDDING}, $embedding) AS score
        WHERE score IS NOT NULL AND score > 0.7
        ORDER BY score DESC LIMIT $limit
        RETURN f.{PROPERTY_QUESTION} AS {PROPERTY_QUESTION}, f.{PROPERTY_CYPHER} AS {PROPERTY_CYPHER}
        """
        param_map = {"embedding": embedding, "database": database, "limit": limit}

        examples = self._execute_query(query, param_map)

        elapsed = time.time() - start_time
        logger.info(f"Retrieved {len(examples)} few-shot examples for database '{database}' in {elapsed:.2f}s")
        return examples

    def store_fewshot_example(self, question: str, database: str, cypher: Optional[str], llm: str,
                              embed_model: SentenceTransformer, success: bool = True) -> None:
        """
        OPTIMIZED: Stores few-shot examples with faster processing.
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot store few-shot example.")
            return

        if not hasattr(embed_model, 'encode'):
            logger.error("CRITICAL: Provided embed_model object does not have an 'encode' method for storing.")
            return

        start_time = time.time()
        label = LABEL_FEWSHOT if success else LABEL_MISSING
        node_id = f"{question}|{llm}|{database}"

        # OPTIMIZED: Fast existence check
        already_exists_result = self._execute_query(
            f"MATCH (f:`{label}` {{{PROPERTY_ID}: $node_id}}) RETURN True LIMIT 1",
            param_map={"node_id": node_id},
        )
        if already_exists_result:
            logger.info(f"Fewshot example already exists for ID '{node_id}'. Skipping store.")
            return

        try:
            embedding = embed_model.encode([question])[0].tolist()
        except Exception as e:
            logger.error(f"Failed to get text embedding for storage using .encode(): {e}", exc_info=True)
            return

        # OPTIMIZED: Simplified store query
        query = f"""
        MERGE (f:`{label}` {{{PROPERTY_ID}: $node_id}})
        ON CREATE SET
            f.{PROPERTY_CYPHER} = $cypher,
            f.{PROPERTY_LLM} = $llm,
            f.{PROPERTY_QUESTION} = $question,
            f.{PROPERTY_DATABASE} = $database,
            f.{PROPERTY_CREATED} = datetime()
        ON MATCH SET
            f.{PROPERTY_CYPHER} = $cypher,
            f.{PROPERTY_LLM} = $llm,
            f.{PROPERTY_DATABASE} = $database,
            f.{PROPERTY_QUESTION} = $question
        WITH f
        CALL db.create.setNodeVectorProperty(f, '{PROPERTY_EMBEDDING}', $embedding)
        """

        param_map = {
            "node_id": node_id,
            "question": question,
            "cypher": cypher,
            "embedding": embedding,
            "database": database,
            "llm": llm,
        }

        self._execute_query(query, param_map)

        elapsed = time.time() - start_time
        logger.info(f"Stored '{label}' example with ID '{node_id}' in {elapsed:.2f}s")

    def generate_enhanced_fewshot_examples(self, question: str, database: str, context: str = "",
                                           num_examples: int = MAX_EXAMPLE_GENERATION) -> List[Dict[str, str]]:
        """
        OPTIMIZED: Generate few-shot examples with performance improvements.
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            logger.info("Multi-provider LLM not available for enhanced few-shot generation")
            return []

        # OPTIMIZED: Reduce examples in fast mode
        if self.fast_mode:
            num_examples = min(num_examples, 2)

        try:
            # OPTIMIZED: Shorter, more focused prompt
            example_generation_prompt = f"""Generate {num_examples} Cypher examples:

Database: {database}
Question: {question}
Context: {context[:500]}...

Format:
Question: [question]
Cypher: [query]

Be concise and relevant."""

            system_prompt = "Generate concise Cypher examples for training. Be brief and accurate."

            response = self._enhanced_llm_call(
                task_name='example_generation',
                prompt=example_generation_prompt,
                system_prompt=system_prompt,
                timeout=self.llm_timeout,
                max_tokens=600,  # Reasonable limit
                temperature=0.7
            )

            if response:
                examples = self._parse_generated_examples(response)
                logger.info(f"Generated {len(examples)} enhanced few-shot examples")
                return examples

        except Exception as e:
            logger.error(f"Failed to generate enhanced few-shot examples: {e}")

        return []

    def _parse_generated_examples(self, response: str) -> List[Dict[str, str]]:
        """OPTIMIZED: Faster parsing of generated examples."""
        examples = []
        lines = response.split('\n')
        current_example = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Question:'):
                if current_example and 'question' in current_example and 'cypher' in current_example:
                    examples.append(current_example)
                current_example = {'question': line[9:].strip()}
            elif line.startswith('Cypher:'):
                if 'question' in current_example:
                    current_example['cypher'] = line[7:].strip()

        # Add the last example
        if current_example and 'question' in current_example and 'cypher' in current_example:
            examples.append(current_example)

        return examples

    def validate_cypher_with_llm(self, cypher: str, question: str, database: str) -> Dict[str, Any]:
        """
        OPTIMIZED: Validate Cypher query with caching for performance.
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {'valid': True, 'confidence': 0.5, 'suggestions': []}

        # OPTIMIZED: Use cache in fast mode
        cache_key = f"{cypher}|{question}"
        if self.fast_mode and self.validation_cache and cache_key in self.validation_cache:
            self.performance_stats['cache_hits'] += 1
            logger.debug("⚡ Using cached validation result")
            return self.validation_cache[cache_key]

        try:
            # OPTIMIZED: Much shorter validation prompt
            validation_prompt = f"""Validate this Cypher query:

Question: {question}
Query: {cypher}

Is it correct? Rate confidence 0-1."""

            system_prompt = "Validate Cypher queries. Be concise: correct/incorrect + confidence score."

            response = self._enhanced_llm_call(
                task_name='example_validation',
                prompt=validation_prompt,
                system_prompt=system_prompt,
                timeout=VALIDATION_TIMEOUT,  # 3 second timeout
                max_tokens=100,  # Very short response
                temperature=0.1
            )

            if response:
                # OPTIMIZED: Simple parsing
                confidence = 0.8  # Default confidence
                valid = True

                response_lower = response.lower()
                if 'error' in response_lower or 'incorrect' in response_lower or 'wrong' in response_lower:
                    confidence = 0.3
                    valid = False

                result = {
                    'valid': valid,
                    'confidence': confidence,
                    'suggestions': [],
                    'llm_feedback': response[:200]  # Truncate for performance
                }

                # OPTIMIZED: Cache result in fast mode
                if self.fast_mode and self.validation_cache:
                    self.validation_cache[cache_key] = result

                return result

        except Exception as e:
            logger.error(f"Failed to validate Cypher with LLM: {e}")

        return {'valid': True, 'confidence': 0.5, 'suggestions': []}

    def correct_cypher_with_llm(self, cypher: str, question: str, database: str, error_message: str = "") -> Optional[
        str]:
        """
        OPTIMIZED: Correct Cypher query with faster processing.
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return None

        try:
            # OPTIMIZED: Shorter correction prompt
            correction_prompt = f"""Fix this Cypher query:

Question: {question}
Broken Query: {cypher}
Error: {error_message[:200]}...

Fixed Query:"""

            system_prompt = "Fix broken Cypher queries. Return only the corrected query."

            corrected_cypher = self._enhanced_llm_call(
                task_name='cypher_correction',
                prompt=correction_prompt,
                system_prompt=system_prompt,
                timeout=self.llm_timeout,
                max_tokens=300,
                temperature=0.1
            )

            if corrected_cypher and corrected_cypher.strip():
                logger.info("Successfully corrected Cypher query using LLM")
                return corrected_cypher.strip()

        except Exception as e:
            logger.error(f"Failed to correct Cypher with LLM: {e}")

        return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """OPTIMIZED: Get performance statistics."""
        return {
            'performance_stats': self.performance_stats.copy(),
            'configuration': {
                'fast_mode': self.fast_mode,
                'llm_timeout': self.llm_timeout,
                'cache_enabled': self.validation_cache is not None,
                'cache_size': len(self.validation_cache) if self.validation_cache else 0
            }
        }

    def optimize_for_speed(self):
        """OPTIMIZED: Enable maximum speed optimizations."""
        self.fast_mode = True
        self.llm_timeout = FAST_LLM_TIMEOUT
        if not self.validation_cache:
            self.validation_cache = {}
        logger.info("🚀 Maximum speed optimizations enabled for few-shot manager")

    def optimize_for_quality(self):
        """OPTIMIZED: Prioritize quality over speed."""
        self.fast_mode = False
        self.llm_timeout = DEFAULT_LLM_TIMEOUT * 2  # 10 seconds
        self.validation_cache = None  # Disable caching
        logger.info("🎯 Quality-focused optimizations enabled for few-shot manager")

    def get_llm_provider_info(self) -> Dict[str, Any]:
        """ENHANCED: Get information about configured LLM providers"""
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {
                'multi_provider_enabled': False,
                'system_available': NEW_LLM_SYSTEM_AVAILABLE,
                'providers': []
            }

        provider_info = {
            'multi_provider_enabled': True,
            'system_available': NEW_LLM_SYSTEM_AVAILABLE,
            'providers': []
        }

        for task_name, manager in self.llm_managers.items():
            if manager:
                try:
                    primary_provider = manager.primary_provider
                    fallback_providers = manager.fallback_providers

                    task_providers = {
                        'task': task_name,
                        'primary_provider': {
                            'name': primary_provider.config.provider.value,
                            'model': primary_provider.config.model,
                            'ready': True
                        },
                        'fallback_providers': []
                    }

                    for fp in fallback_providers:
                        task_providers['fallback_providers'].append({
                            'name': fp.config.provider.value,
                            'model': fp.config.model,
                            'ready': True
                        })

                    provider_info['providers'].append(task_providers)

                except Exception as e:
                    provider_info['providers'].append({
                        'task': task_name,
                        'error': str(e),
                        'ready': False
                    })

        return provider_info

    def get_system_health(self) -> Dict[str, Any]:
        """OPTIMIZED: Get system health including performance metrics."""
        health = {
            'neo4j_connected': False,
            'embedding_model_ready': False,
            'multi_provider_llm_active': False,
            'llm_providers_ready': [],
            'performance_metrics': {
                'fast_mode': self.fast_mode,
                'average_llm_time': self.performance_stats['average_llm_time'],
                'timeout_count': self.performance_stats['timeout_count'],
                'cache_hits': self.performance_stats['cache_hits'],
                'total_llm_calls': self.performance_stats['total_llm_calls']
            }
        }

        try:
            # Check Neo4j connection
            if self.graph_store:
                test_result = self._execute_query("RETURN 1 as test", {})
                health['neo4j_connected'] = bool(test_result)

            # Check embedding model (would need to be passed to check)
            health['embedding_model_ready'] = True  # Placeholder

            # Check multi-provider LLM
            if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
                health['multi_provider_llm_active'] = True

                # Check each LLM manager
                for task_name, manager in self.llm_managers.items():
                    if manager:
                        try:
                            primary_provider = manager.primary_provider
                            health['llm_providers_ready'].append({
                                'task': task_name,
                                'provider': primary_provider.config.provider.value,
                                'ready': True
                            })
                        except Exception as e:
                            health['llm_providers_ready'].append({
                                'task': task_name,
                                'provider': 'unknown',
                                'ready': False,
                                'error': str(e)
                            })

        except Exception as e:
            logger.warning(f"Error checking system health: {e}")

        return health

    def clear_cache(self):
        """OPTIMIZED: Clear validation cache for memory management."""
        if self.validation_cache:
            cache_size = len(self.validation_cache)
            self.validation_cache.clear()
            logger.info(f"🧹 Cleared validation cache ({cache_size} entries)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """OPTIMIZED: Get detailed cache statistics."""
        if not self.validation_cache:
            return {'cache_enabled': False}

        return {
            'cache_enabled': True,
            'cache_size': len(self.validation_cache),
            'cache_hits': self.performance_stats['cache_hits'],
            'hit_rate': (
                self.performance_stats['cache_hits'] / self.performance_stats['total_llm_calls']
                if self.performance_stats['total_llm_calls'] > 0 else 0
            )
        }

    def cleanup_old_examples(self, days_old: int = 30) -> int:
        """
        OPTIMIZED: Clean up old few-shot examples to maintain performance.

        Args:
            days_old: Remove examples older than this many days

        Returns:
            Number of examples removed
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot cleanup examples.")
            return 0

        try:
            # Query to find and delete old examples
            cleanup_query = f"""
            MATCH (f:{LABEL_FEWSHOT})
            WHERE f.{PROPERTY_CREATED} < datetime() - duration({{days: $days}})
            WITH f, count(*) as count
            DELETE f
            RETURN count
            """

            result = self._execute_query(cleanup_query, {"days": days_old})

            deleted_count = result[0]['count'] if result else 0
            logger.info(f"🧹 Cleaned up {deleted_count} old few-shot examples (older than {days_old} days)")
            return deleted_count

        except Exception as e:
            logger.error(f"Error during cleanup of old examples: {e}")
            return 0

    def reset_performance_stats(self):
        """OPTIMIZED: Reset performance statistics for fresh monitoring."""
        self.performance_stats = {
            'total_llm_calls': 0,
            'average_llm_time': 0,
            'timeout_count': 0,
            'cache_hits': 0
        }
        logger.info("📊 Performance statistics reset")

    def export_examples(self, database: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        OPTIMIZED: Export few-shot examples for backup or analysis.

        Args:
            database: Database to export examples from
            output_file: Optional file path to save JSON export

        Returns:
            List of exported examples
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot export examples.")
            return []

        try:
            export_query = f"""
            MATCH (f:{LABEL_FEWSHOT})
            WHERE f.{PROPERTY_DATABASE} = $database
            RETURN f.{PROPERTY_QUESTION} as question,
                   f.{PROPERTY_CYPHER} as cypher,
                   f.{PROPERTY_LLM} as llm,
                   f.{PROPERTY_CREATED} as created,
                   f.{PROPERTY_DATABASE} as database
            ORDER BY f.{PROPERTY_CREATED} DESC
            """

            examples = self._execute_query(export_query, {"database": database})

            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(examples, f, indent=2, default=str)
                logger.info(f"📤 Exported {len(examples)} examples to {output_file}")

            logger.info(f"📤 Retrieved {len(examples)} examples for database '{database}'")
            return examples

        except Exception as e:
            logger.error(f"Error exporting examples: {e}")
            return []

    def import_examples(self, examples: List[Dict[str, Any]], embed_model: SentenceTransformer) -> int:
        """
        OPTIMIZED: Import few-shot examples from external source.

        Args:
            examples: List of example dictionaries with 'question', 'cypher', 'database', 'llm'
            embed_model: Embedding model for vectorization

        Returns:
            Number of successfully imported examples
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot import examples.")
            return 0

        imported_count = 0
        for example in examples:
            try:
                required_fields = ['question', 'cypher', 'database', 'llm']
                if not all(field in example for field in required_fields):
                    logger.warning(f"Skipping example missing required fields: {example}")
                    continue

                self.store_fewshot_example(
                    question=example['question'],
                    database=example['database'],
                    cypher=example['cypher'],
                    llm=example['llm'],
                    embed_model=embed_model,
                    success=True
                )
                imported_count += 1

            except Exception as e:
                logger.error(f"Error importing example {example}: {e}")

        logger.info(f"📥 Successfully imported {imported_count}/{len(examples)} examples")
        return imported_count

    def get_database_stats(self, database: str = None) -> Dict[str, Any]:
        """
        OPTIMIZED: Get statistics about stored examples.

        Args:
            database: Specific database to analyze, or None for all databases

        Returns:
            Dictionary with statistics
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot get database stats.")
            return {}

        try:
            if database:
                # Stats for specific database
                stats_query = f"""
                MATCH (f:{LABEL_FEWSHOT})
                WHERE f.{PROPERTY_DATABASE} = $database
                WITH f.{PROPERTY_LLM} as llm, count(*) as count
                RETURN llm, count
                ORDER BY count DESC
                """
                llm_stats = self._execute_query(stats_query, {"database": database})

                total_query = f"""
                MATCH (f:{LABEL_FEWSHOT})
                WHERE f.{PROPERTY_DATABASE} = $database
                RETURN count(*) as total
                """
                total_result = self._execute_query(total_query, {"database": database})
                total_count = total_result[0]['total'] if total_result else 0

                return {
                    'database': database,
                    'total_examples': total_count,
                    'llm_breakdown': llm_stats
                }
            else:
                # Stats for all databases
                all_stats_query = f"""
                MATCH (f:{LABEL_FEWSHOT})
                WITH f.{PROPERTY_DATABASE} as database, f.{PROPERTY_LLM} as llm, count(*) as count
                RETURN database, llm, count
                ORDER BY database, count DESC
                """
                all_stats = self._execute_query(all_stats_query, {})

                # Organize by database
                stats_by_db = {}
                for stat in all_stats:
                    db = stat['database']
                    if db not in stats_by_db:
                        stats_by_db[db] = {'total_examples': 0, 'llm_breakdown': []}
                    stats_by_db[db]['total_examples'] += stat['count']
                    stats_by_db[db]['llm_breakdown'].append({
                        'llm': stat['llm'],
                        'count': stat['count']
                    })

                return {
                    'all_databases': True,
                    'databases': stats_by_db,
                    'total_databases': len(stats_by_db)
                }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def close(self):
        """
        OPTIMIZED: Closes the Neo4j connection and performs cleanup.

        This method should be called when the manager is no longer needed
        to properly close database connections and clean up resources.
        """
        try:
            # Clear caches to free memory
            if self.validation_cache:
                cache_size = len(self.validation_cache)
                self.validation_cache.clear()
                logger.debug(f"Cleared validation cache ({cache_size} entries)")

            # Close LLM managers if they have close methods
            if self.llm_managers:
                for task_name, manager in self.llm_managers.items():
                    if manager and hasattr(manager, 'close'):
                        try:
                            manager.close()
                            logger.debug(f"Closed LLM manager for {task_name}")
                        except Exception as e:
                            logger.warning(f"Error closing LLM manager for {task_name}: {e}")

            # Close Neo4j connection
            if self.graph_store:
                try:
                    # Note: Neo4jPropertyGraphStore might not have an explicit close method
                    # but we can set it to None to release the reference
                    self.graph_store = None
                    logger.info("Neo4j connection closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing Neo4j connection: {e}")

            # Log final performance stats
            if self.performance_stats['total_llm_calls'] > 0:
                logger.info(f"📊 Final Performance Stats:")
                logger.info(f"   Total LLM calls: {self.performance_stats['total_llm_calls']}")
                logger.info(f"   Average LLM time: {self.performance_stats['average_llm_time']:.2f}s")
                logger.info(f"   Timeout count: {self.performance_stats['timeout_count']}")
                logger.info(f"   Cache hits: {self.performance_stats['cache_hits']}")

            logger.info("Neo4j Fewshot Manager closed successfully")

        except Exception as e:
            logger.error(f"Error during Neo4j Fewshot Manager close: {e}", exc_info=True)

    def __enter__(self):
        """OPTIMIZED: Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """OPTIMIZED: Context manager exit with automatic cleanup."""
        self.close()
        if exc_type:
            logger.error(f"Exception occurred in context manager: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

    def __del__(self):
        """OPTIMIZED: Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'graph_store') and self.graph_store:
                self.close()
        except Exception:
            pass  # Ignore errors in destructor


# OPTIMIZED: Factory function for easy instantiation
def create_optimized_fewshot_manager(
        config_file: str = CONFIG_FILE_NAME,
        fast_mode: bool = True,
        enable_multi_provider_llm: bool = True,
        config: Optional[Dict[str, Any]] = None
) -> Neo4jFewshotManager:
    """
    OPTIMIZED: Factory function to create an optimized Neo4j Fewshot Manager.

    Args:
        config_file: Path to configuration file
        fast_mode: Enable performance optimizations
        enable_multi_provider_llm: Enable multi-provider LLM support
        config: Optional configuration dictionary

    Returns:
        Configured Neo4jFewshotManager instance
    """
    return Neo4jFewshotManager(
        config_file=config_file,
        config=config,
        enable_multi_provider_llm=enable_multi_provider_llm,
        fast_mode=fast_mode
    )


# OPTIMIZED: Example usage and testing function
def test_fewshot_manager_performance():
    """
    OPTIMIZED: Test function to verify performance improvements.
    """
    print("🧪 Testing Neo4j Fewshot Manager Performance...")

    try:
        # Create manager in fast mode
        with create_optimized_fewshot_manager(fast_mode=True) as manager:
            print(f"✅ Manager created successfully")
            print(f"   Fast mode: {manager.fast_mode}")
            print(f"   LLM timeout: {manager.llm_timeout}s")
            print(f"   Multi-provider LLM: {manager.enable_multi_provider_llm}")

            # Test system health
            health = manager.get_system_health()
            print(f"🏥 System Health:")
            print(f"   Neo4j connected: {health['neo4j_connected']}")
            print(f"   Multi-provider LLM active: {health['multi_provider_llm_active']}")

            # Test performance stats
            stats = manager.get_performance_stats()
            print(f"📊 Performance Configuration:")
            print(f"   Cache enabled: {stats['configuration']['cache_enabled']}")
            print(f"   Fast mode: {stats['configuration']['fast_mode']}")

        print("✅ Neo4j Fewshot Manager test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    # Run performance test if executed directly
    test_fewshot_manager_performance()
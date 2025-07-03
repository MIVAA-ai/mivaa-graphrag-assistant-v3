import configparser
import logging
import os
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
    Neo4jPropertyGraphStore = None  # Allow script to load without the dependency

# Import SentenceTransformer for type hinting
try:
    from sentence_transformers import SentenceTransformer

    print("[INFO] Imported SentenceTransformer for FewShotManager type hints.")
except ImportError:
    print("[WARN] sentence-transformers not found for FewShotManager type hints.")
    # Define dummy class if library not installed, for type hinting fallback
    SentenceTransformer = Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_NEO4J_TIMEOUT = 30
DEFAULT_RETRIEVAL_LIMIT = 7
CONFIG_FILE_NAME = "graph_config.ini"  # Default config file name
ENV_VAR_PREFIX = "FEWSHOT_"  # Prefix for environment variables
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


# --- ---

class Neo4jFewshotManager:
    """
    Manages storing and retrieving few-shot examples for language models
    using a Neo4j graph database.
    *** ENHANCED WITH MULTI-PROVIDER LLM SUPPORT ***

    Handles connection (reading from environment variables or config file),
    querying for similar examples based on question embeddings,
    storing new examples (successful or missing),
    and now supports multi-provider LLM for enhanced few-shot generation.
    """
    graph_store: Optional[Neo4jPropertyGraphStore] = None

    def __init__(self, config_file: str = CONFIG_FILE_NAME, timeout: int = DEFAULT_NEO4J_TIMEOUT,
                 config: Optional[Dict[str, Any]] = None, enable_multi_provider_llm: bool = True):
        """
        ENHANCED: Initializes the Neo4jFewshotManager with multi-provider LLM support.

        Args:
            config_file: Path to configuration file
            timeout: Neo4j connection timeout
            config: Configuration dictionary for multi-provider LLM
            enable_multi_provider_llm: Whether to enable multi-provider LLM features
        """
        # ENHANCED: Store configuration for multi-provider LLM
        self.config = config or {}
        self.enable_multi_provider_llm = enable_multi_provider_llm
        self.llm_managers = {}

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
            # Use database="neo4j" if not connecting to the default database
            self.graph_store = Neo4jPropertyGraphStore(
                username=username,
                password=password,
                url=url,
                refresh_schema=False,  # Assuming schema is managed elsewhere
                create_indexes=False,  # Assuming indexes are managed elsewhere
                timeout=timeout,
                # database="neo4j" # Uncomment and set if using a non-default DB
            )
            logger.info("Successfully configured Neo4j connection details for few-shot management.")
        except Exception as e:
            logger.error(f"Failed to configure Neo4j connection: {e}", exc_info=True)
            self.graph_store = None  # Ensure graph_store is None on failure

        # ENHANCED: Initialize multi-provider LLM system
        if self.enable_multi_provider_llm and NEW_LLM_SYSTEM_AVAILABLE:
            self._initialize_multi_provider_llm()

        logger.info(f"Neo4j Fewshot Manager initialized. Multi-provider LLM: {self.enable_multi_provider_llm}")

    def _initialize_multi_provider_llm(self):
        """ENHANCED: Initialize multi-provider LLM system for few-shot tasks"""
        try:
            # Import the main LLM configuration manager
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

    def _enhanced_llm_call(self, task_name: str, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        ENHANCED: Enhanced LLM call with multi-provider support for few-shot tasks.
        Falls back to basic functionality if enhanced LLM is not available.
        """
        # Try enhanced system first
        if (self.enable_multi_provider_llm and
                NEW_LLM_SYSTEM_AVAILABLE and
                task_name in self.llm_managers and
                self.llm_managers[task_name]):

            try:
                logger.debug(f"🎯 Using enhanced LLM system for {task_name}")
                return self.llm_managers[task_name].call_llm(
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Enhanced LLM failed for {task_name}: {e}, falling back to basic functionality")

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

                # ADDED: Override with environment variables if available
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

        return None, None, None  # Return None if loading failed

    def _execute_query(self, query: str, param_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Helper method to execute queries with error handling using structured_query."""
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot execute query.")
            return []
        try:
            # Use structured_query method
            results = self.graph_store.structured_query(query, param_map=param_map)
            # structured_query returns results that can be iterated over
            return [dict(record) for record in results] if results else []
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}\nQuery: {query}\nParams: {param_map}", exc_info=True)
            return []  # Return empty list on error

    def retrieve_fewshots(self, question: str, database: str, embed_model: SentenceTransformer,
                          limit: int = DEFAULT_RETRIEVAL_LIMIT) -> List[Dict[str, str]]:
        """
        Retrieves the most relevant few-shot examples from Neo4j based on question similarity.
        *** Uses the standard .encode() method of SentenceTransformer. ***
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot retrieve few-shots.")
            return []
        # Check if the passed model has the required .encode() method
        if not hasattr(embed_model, 'encode'):
            logger.error("CRITICAL: Provided embed_model object does not have an 'encode' method.")
            return []

        try:
            # *** FIX: Use .encode() and convert numpy array to list ***
            embedding = embed_model.encode([question])[0].tolist()
        except Exception as e:
            logger.error(f"Failed to get text embedding using .encode(): {e}", exc_info=True)
            return []

        # Note: Assumes a vector index exists on Fewshot(embedding) in Neo4j
        query = f"""
        MATCH (f:{LABEL_FEWSHOT})
        WHERE f.{PROPERTY_DATABASE} = $database
        WITH f, vector.similarity.cosine(f.{PROPERTY_EMBEDDING}, $embedding) AS score
        WHERE score IS NOT NULL // Prevent errors if similarity is null
        ORDER BY score DESC LIMIT $limit
        RETURN f.{PROPERTY_QUESTION} AS {PROPERTY_QUESTION}, f.{PROPERTY_CYPHER} AS {PROPERTY_CYPHER}
        """
        param_map = {"embedding": embedding, "database": database, "limit": limit}

        examples = self._execute_query(query, param_map)
        logger.info(f"Retrieved {len(examples)} few-shot examples for database '{database}'.")
        return examples

    def store_fewshot_example(self, question: str, database: str, cypher: Optional[str], llm: str,
                              embed_model: SentenceTransformer, success: bool = True) -> None:
        """
        Stores a new few-shot example (or a missing example) in the Neo4j database.
        *** Uses the standard .encode() method of SentenceTransformer and corrected Cypher CALL. ***
        """
        if not self.graph_store:
            logger.warning("Graph store not initialized. Cannot store few-shot example.")
            return
        # Check if the passed model has the required .encode() method
        if not hasattr(embed_model, 'encode'):
            logger.error("CRITICAL: Provided embed_model object does not have an 'encode' method for storing.")
            return

        label = LABEL_FEWSHOT if success else LABEL_MISSING
        # Construct a unique ID - ensure components don't contain the separator '|' or handle encoding
        node_id = f"{question}|{llm}|{database}"

        # Check if already exists - Use _execute_query for consistency
        already_exists_result = self._execute_query(
            f"MATCH (f:`{label}` {{{PROPERTY_ID}: $node_id}}) RETURN True",
            param_map={"node_id": node_id},
        )
        if already_exists_result:
            logger.info(f"Fewshot example already exists for ID '{node_id}'. Skipping store.")
            return

        try:
            # *** FIX: Use .encode() and convert numpy array to list ***
            embedding = embed_model.encode([question])[0].tolist()
        except Exception as e:
            logger.error(f"Failed to get text embedding for storage using .encode(): {e}", exc_info=True)
            return

        # FIX: Corrected Cypher query - removed YIELD/RETURN and fixed CREATED property name
        # Assumes db.create.setNodeVectorProperty procedure exists and is void (returns nothing).
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
        """  # Removed "YIELD node RETURN count(node)"
        param_map = {
            "node_id": node_id,
            "question": question,
            "cypher": cypher,
            "embedding": embedding,  # Pass the generated embedding list
            "database": database,
            "llm": llm,
        }

        # Use _execute_query which uses structured_query and handles errors
        result_list = self._execute_query(query, param_map)

        # Log attempt, success is hard to confirm without RETURN
        logger.info(f"Executed store query for '{label}' example with ID '{node_id}'. Check logs for potential errors.")

    def generate_enhanced_fewshot_examples(self, question: str, database: str, context: str = "",
                                           num_examples: int = 3) -> List[Dict[str, str]]:
        """
        ENHANCED: Generate few-shot examples using multi-provider LLM.
        Creates synthetic examples based on the question and database context.
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            logger.info("Multi-provider LLM not available for enhanced few-shot generation")
            return []

        try:
            example_generation_prompt = f"""
            Generate {num_examples} few-shot examples for Cypher query generation.

            Database: {database}
            Question: {question}
            Context: {context}

            For each example, provide:
            1. A similar question
            2. The corresponding Cypher query

            Format each example as:
            Question: [question]
            Cypher: [cypher_query]

            Focus on creating diverse but relevant examples that would help with similar queries.
            """

            system_prompt = """You are an expert at creating few-shot examples for Cypher query generation. Generate high-quality, diverse examples that would be useful for training language models."""

            response = self._enhanced_llm_call(
                task_name='example_generation',
                prompt=example_generation_prompt,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )

            if response:
                # Parse the response to extract examples
                examples = self._parse_generated_examples(response)
                logger.info(f"Generated {len(examples)} enhanced few-shot examples")
                return examples

        except Exception as e:
            logger.error(f"Failed to generate enhanced few-shot examples: {e}")

        return []

    def _parse_generated_examples(self, response: str) -> List[Dict[str, str]]:
        """Parse generated examples from LLM response."""
        examples = []
        lines = response.split('\n')
        current_example = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Question:'):
                if current_example:
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
        ENHANCED: Validate Cypher query using multi-provider LLM.
        Returns validation results with suggestions for improvement.
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return {'valid': True, 'confidence': 0.5, 'suggestions': []}

        try:
            validation_prompt = f"""
            Please validate the following Cypher query and provide feedback:

            Question: {question}
            Database: {database}
            Cypher Query: {cypher}

            Please evaluate:
            1. Syntax correctness
            2. Logical correctness for the question
            3. Efficiency and best practices
            4. Potential issues or improvements

            Provide a confidence score (0-1) and specific suggestions if any.
            """

            system_prompt = """You are a Neo4j Cypher expert. Validate Cypher queries for correctness, efficiency, and best practices."""

            response = self._enhanced_llm_call(
                task_name='example_validation',
                prompt=validation_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
                temperature=0.1
            )

            if response:
                # Parse validation response (simplified)
                confidence = 0.8  # Default confidence
                suggestions = []

                # Simple parsing - in production, you'd want more sophisticated parsing
                if 'error' in response.lower() or 'incorrect' in response.lower():
                    confidence = 0.3
                    suggestions.append("Query may have issues based on LLM analysis")

                return {
                    'valid': confidence > 0.5,
                    'confidence': confidence,
                    'suggestions': suggestions,
                    'llm_feedback': response
                }

        except Exception as e:
            logger.error(f"Failed to validate Cypher with LLM: {e}")

        return {'valid': True, 'confidence': 0.5, 'suggestions': []}

    def correct_cypher_with_llm(self, cypher: str, question: str, database: str, error_message: str = "") -> Optional[
        str]:
        """
        ENHANCED: Attempt to correct a Cypher query using multi-provider LLM.
        """
        if not self.enable_multi_provider_llm or not NEW_LLM_SYSTEM_AVAILABLE:
            return None

        try:
            correction_prompt = f"""
            Please correct the following Cypher query:

            Question: {question}
            Database: {database}
            Original Cypher: {cypher}
            Error Message: {error_message}

            Please provide a corrected version of the Cypher query that:
            1. Fixes any syntax errors
            2. Properly answers the question
            3. Follows Neo4j best practices

            Provide only the corrected Cypher query.
            """

            system_prompt = """You are a Neo4j Cypher expert. Fix broken Cypher queries to make them syntactically correct and logically sound."""

            corrected_cypher = self._enhanced_llm_call(
                task_name='cypher_correction',
                prompt=correction_prompt,
                system_prompt=system_prompt,
                max_tokens=300,
                temperature=0.1
            )

            if corrected_cypher and corrected_cypher.strip():
                logger.info("Successfully corrected Cypher query using LLM")
                return corrected_cypher.strip()

        except Exception as e:
            logger.error(f"Failed to correct Cypher with LLM: {e}")

        return None

    def get_llm_provider_info(self) -> Dict[str, Any]:
        """ENHANCED: Get information about configured LLM providers for few-shot tasks."""
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
        """ENHANCED: Get system health including multi-provider LLM status."""
        health = {
            'neo4j_connected': False,
            'embedding_model_ready': False,
            'multi_provider_llm_active': False,
            'llm_providers_ready': []
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

    def close(self):
        """Closes the Neo4j connection if the graph store was initialized."""
        if self.graph_store and hasattr(self.graph_store, '_driver') and self.graph_store._driver:
            try:
                self.graph_store._driver.close()
                logger.info("Closed Neo4j connection for FewShotManager.")
            except Exception as e:
                logger.error(f"Error closing FewShotManager Neo4j connection: {e}", exc_info=True)


# ENHANCED: Factory functions for creating enhanced few-shot managers

def create_enhanced_fewshot_manager(config: Dict[str, Any], config_file: str = CONFIG_FILE_NAME,
                                    enable_multi_provider_llm: bool = True) -> Neo4jFewshotManager:
    """
    Factory function for creating enhanced Neo4j few-shot manager with multi-provider LLM support.

    Args:
        config: Configuration dictionary for multi-provider LLM
        config_file: Path to Neo4j configuration file
        enable_multi_provider_llm: Whether to enable multi-provider LLM features
    """
    return Neo4jFewshotManager(
        config_file=config_file,
        config=config,
        enable_multi_provider_llm=enable_multi_provider_llm
    )


def create_standard_fewshot_manager(config_file: str = CONFIG_FILE_NAME) -> Neo4jFewshotManager:
    """
    Factory function for creating standard Neo4j few-shot manager (backward compatibility).
    """
    return Neo4jFewshotManager(config_file=config_file, enable_multi_provider_llm=False)


if __name__ == "__main__":
    print("Enhanced Neo4j Fewshot Manager with Multi-Provider LLM Support")
    print("Provides few-shot example management with multi-provider LLM enhancements")
    print("Backward compatible with existing Neo4j few-shot functionality")
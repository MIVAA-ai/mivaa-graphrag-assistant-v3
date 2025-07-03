import configparser
import logging
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Import Neo4jPropertyGraphStore and handle potential ImportError
try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
except ImportError:
    print(
        "llama_index.graph_stores.neo4j not found."
        " Please install it: pip install llama-index-graph-stores-neo4j"
    )
    Neo4jPropertyGraphStore = None # Allow script to load without the dependency

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
CONFIG_FILE_NAME = "graph_config.ini" # Default config file name
ENV_VAR_PREFIX = "FEWSHOT_" # Prefix for environment variables
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
    *** CORRECTED VERSION USING .encode() ***

    Handles connection (reading from environment variables or config file),
    querying for similar examples based on question embeddings,
    and storing new examples (successful or missing).
    """
    graph_store: Optional[Neo4jPropertyGraphStore] = None

    def __init__(self, config_file: str = CONFIG_FILE_NAME, timeout: int = DEFAULT_NEO4J_TIMEOUT):
        """Initializes the Neo4jFewshotManager."""
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
                refresh_schema=False, # Assuming schema is managed elsewhere
                create_indexes=False, # Assuming indexes are managed elsewhere
                timeout=timeout,
                # database="neo4j" # Uncomment and set if using a non-default DB
            )
            logger.info("Successfully configured Neo4j connection details for few-shot management.")
        except Exception as e:
            logger.error(f"Failed to configure Neo4j connection: {e}", exc_info=True)
            self.graph_store = None # Ensure graph_store is None on failure

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
            return [] # Return empty list on error

    def retrieve_fewshots(self, question: str, database: str, embed_model: SentenceTransformer, limit: int = DEFAULT_RETRIEVAL_LIMIT) -> List[Dict[str, str]]:
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

    def store_fewshot_example(self, question: str, database: str, cypher: Optional[str], llm: str, embed_model: SentenceTransformer, success: bool = True) -> None:
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
        """ # Removed "YIELD node RETURN count(node)"
        param_map = {
            "node_id": node_id,
            "question": question,
            "cypher": cypher,
            "embedding": embedding, # Pass the generated embedding list
            "database": database,
            "llm": llm,
        }

        # Use _execute_query which uses structured_query and handles errors
        result_list = self._execute_query(query, param_map)

        # Log attempt, success is hard to confirm without RETURN
        logger.info(f"Executed store query for '{label}' example with ID '{node_id}'. Check logs for potential errors.")


    def close(self):
        """Closes the Neo4j connection if the graph store was initialized."""
        if self.graph_store and hasattr(self.graph_store, '_driver') and self.graph_store._driver:
            try:
                self.graph_store._driver.close()
                logger.info("Closed Neo4j connection for FewShotManager.")
            except Exception as e:
                logger.error(f"Error closing FewShotManager Neo4j connection: {e}", exc_info=True)
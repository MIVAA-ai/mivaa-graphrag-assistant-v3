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
from src.knowledge_graph.prompts import (
    TEXT_TO_CYPHER_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT,
    EVALUATE_EMPTY_RESULT_USER_PROMPT, # Ensure user prompts are imported too
    REVISE_EMPTY_RESULT_SYSTEM_PROMPT,
    REVISE_EMPTY_RESULT_USER_PROMPT     # Ensure user prompts are imported too
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
    # vvv Import BOTH if you might need sync elsewhere, or just AsyncGraphDatabase vvv
    from neo4j import GraphDatabase, AsyncGraphDatabase, exceptions as neo4j_exceptions
except ImportError:
    raise ImportError("Neo4j Python driver not found. Please install: pip install neo4j")

# LLM Imports
try:
    from src.knowledge_graph.llm import call_llm, extract_json_from_text, QuotaError
    print("[INFO] Using actual 'call_llm' function.")
except ImportError:
    print("[WARN] 'call_llm' not found. Using mock function.")
    # --- Mock LLM ---
    class QuotaError(Exception): pass
    def call_llm(*args, **kwargs):
        print("[WARN] Mock call_llm returning placeholder.")
        if "TEXT_TO_CYPHER_SYSTEM_PROMPT" in kwargs.get("system_prompt", ""):
             print("[WARN] Mock LLM: Simulating generic Cypher query")
             return """```cypher
             MATCH (e:Entity)-[r]-(t:Entity) WHERE toLower(e.name) CONTAINS 'some_entity' RETURN e.name as subject, r.original as predicate, t.name as object LIMIT 5
             ```"""
        elif "correct_cypher_step" in str(args) or "correct_cypher_step" in str(kwargs):
            print("[WARN] Mock LLM: Simulating corrected Cypher query")
            return """```cypher
            MATCH (e:Entity {name: 'corrected_entity'})-[r]-(t:Entity) RETURN e.name as subject, type(r) as type, t.name as object LIMIT 5
            ```"""
        else:
             print("[WARN] Mock LLM: Simulating Q&A synthesis")
             return "Placeholder answer based on provided context."
    def extract_json_from_text(text): return None
    # --- End Mock LLM ---

# Vector DB Imports
try:
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.api.types import EmbeddingFunction
except ImportError:
    raise ImportError("ChromaDB library not found. Please install: pip install chromadb")

# Embeddings Imports
try:
    # Import the specific class for instantiation
    from sentence_transformers import SentenceTransformer
    print("[INFO] Imported SentenceTransformer.")
    embeddings_available = True
except ImportError:
    print("[WARN] 'sentence-transformers' library not found (pip install sentence-transformers). Few-shot retrieval/storage might be affected.")
    embeddings_available = False
    # Define dummy class if needed elsewhere, though manager might handle absence
    class SentenceTransformer: pass


# Import Prompts
try:
    # Using the prompt with the fewshot placeholder
    from src.knowledge_graph.prompts import TEXT_TO_CYPHER_SYSTEM_PROMPT
    # Import the USER prompt template separately if needed for formatting examples
    from src.knowledge_graph.prompts import GENERATE_USER_TEMPLATE # Assuming this exists and is compatible
except ImportError:
    print("[WARN] Could not import TEXT_TO_CYPHER_SYSTEM_PROMPT or GENERATE_USER_TEMPLATE. Define manually or check prompts.py.")
    TEXT_TO_CYPHER_SYSTEM_PROMPT = """
    You are an expert Neo4j Cypher query translator. Your task is to convert natural language questions into precise Cypher queries based on the provided graph schema to retrieve relevant information.
    
    Graph Schema:
    {dynamic_schema}
    
    Core Task:
    1. Analyze the user's question (provided in the 'User Input' section below) to understand the specific information requested.
    2. Identify the key entities (wells, fields, persons, companies, skills, software, etc.) and the desired relationship(s) mentioned or implied.
    3. **Use Pre-linked Entities:** Check the 'Pre-linked Entities' section provided in the 'User Input'. This maps mentions from the question to canonical names found in the graph.
    4. Construct a Cypher query that retrieves the requested information using the provided **Graph Schema**.
        - Use `MATCH` clauses to specify the graph pattern. Use specific node labels (e.g., `:Well`, `:Formation`, `:Skill`) from the schema if implied by the question (e.g., "Which wells...?").
        - Filter Entities: Use `WHERE` clauses for filtering on entity names:
            - **Priority:** If a 'Canonical Name in Graph' is provided for a mention in the 'Pre-linked Entities' section, **use an exact, case-insensitive match** on that canonical name: `WHERE toLower(node.name) = toLower('canonical_name_provided')`.
            - **Fallback:** If no canonical name is provided for a mention (shows as 'None' in the pre-linked list) or the entity wasn't pre-linked, use fuzzy, case-insensitive matching on the original mention from the question: `WHERE toLower(node.name) CONTAINS toLower('original_mention')`.
        - Filter Relationships:
            - If the question implies a specific action/relationship type (e.g., "who drilled", "where located", "what skills"), match the specific relationship type from the schema if known (e.g., `MATCH (a)-[r:DRILLED_BY]->(b)`). Relationship types in the schema are typically UPPERCASE_SNAKE_CASE.
            - **CRITICAL SYNTAX:** When matching multiple relationship types OR using a variable with a type, use the pipe `|` separator *without* a colon before subsequent types. Correct: `[r:TYPE1|TYPE2]`, `[:TYPE1|TYPE2]`. Incorrect: `[r:TYPE1|:TYPE2]`, `[:TYPE1|:TYPE2]`.
            - If the relationship is less specific or the exact type is unknown (e.g., "connection between", "related to", "tell me about"), match any relationship (`MATCH (a)-[r]-(b)`) and filter on the original predicate text using fuzzy matching: `WHERE toLower(r.original) CONTAINS 'keyword'`. Use keywords extracted from the question. (Note: `r.original` property stores the original text predicate).
        - Use `RETURN` to specify the output, using aliases like `subject`, `predicate`, `object` where appropriate for clarity. Return distinct results if needed (`RETURN DISTINCT ...`).
        - **Prioritize returning specific properties (like `.name`) rather than entire nodes.**
        - **DO NOT use the generic UNION query pattern unless the question is extremely broad like "show all connections for X".** Focus on targeted queries based on the question's intent and the schema.
    5. If the question requires information directly from the source text, you can optionally include a match to the `:Chunk` node and return `c.text`.
    6. If the question cannot be answered using the provided schema or is too ambiguous, return the exact text "NO_QUERY_GENERATED".
    
    Query Formatting:
    - Enclose the final Cypher query in triple backticks ```cypher ... ```.
    - Only return the Cypher query or "NO_QUERY_GENERATED". Do not add explanations or comments outside the backticks.
    
    Examples:
    
    User Question: Who drilled well kg-d6-a#5?
    Cypher Query:
    ```cypher
    MATCH (operator:Entity)-[:DRILLED_BY]->(well:Entity)
    WHERE toLower(well.name) = 'kg-d6-a#5'
    RETURN operator.name AS operator
    ```
    
    User Question: What formations did well B-12#13 penetrate or encounter?
    Cypher Query:
    ```cypher
    MATCH (well:Entity)-[r:PENETRATES|ENCOUNTERED]->(formation:Entity)
    WHERE toLower(well.name) = 'b-12#13'
    RETURN DISTINCT formation.name AS formation
    ```
    
    User Question: Tell me about the Daman Formation. (General relationship)
    Cypher Query:
    ```cypher
    MATCH (e1:Entity)-[r]-(related:Entity)
    WHERE toLower(e1.name) = 'daman formation'
    OPTIONAL MATCH (e1)-[:FROM_CHUNK]->(c:Chunk)
    RETURN e1.name AS subject, type(r) AS type, r.original AS predicate, related.name AS related_entity, c.text AS chunk_text
    LIMIT 25
    ```
    
    User Question: List all wells drilled by Reliance. (Fuzzy entity, specific relationship type)
    Cypher Query:
    ```cypher
    MATCH (operator:Entity)-[:DRILLED_BY]->(well:Entity)
    WHERE toLower(operator.name) CONTAINS 'reliance' AND (toLower(well.name) CONTAINS 'well' OR toLower(well.name) CONTAINS '#')
    RETURN DISTINCT well.name AS well
    ```
    
    User Question: How is well A22 related to drilling? (Specific entity, fuzzy relationship)
    Cypher Query:
    ```cypher
    MATCH (e1:Entity)-[r]-(related:Entity)
    WHERE toLower(e1.name) = 'a22' AND toLower(r.original) CONTAINS 'drill'
    RETURN e1.name AS subject, type(r) AS type, r.original AS predicate, related.name AS related_entity
    LIMIT 25
    ```
    
    User Question: What is the capital of France?
    Output:
    NO_QUERY_GENERATED
    """

    # Define a basic user template if missing
    GENERATE_USER_TEMPLATE = """
    Schema: {schema}
    Few-shot Examples:
    {fewshot_examples}
    User input: {question}
    Cypher query:"""

# Prompts for evaluating and revising Cypher queries with empty results

EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT = """

You are a meticulous Cypher query analyst. Your objective is to diagnose why a given Cypher query returned zero results when executed against a graph database with the specified schema.

**Graph Schema:**
    {dynamic_schema}

**Your Task:**
Based *only* on the user's question, the executed Cypher query, and the schema, determine the *single most likely* reason for the empty result. Choose exclusively from these options:

1.  **QUERY_MISMATCH**: The query's structure (nodes, relationships, properties, filters) does not accurately reflect the semantic intent of the user's question given the schema. For example, the query looks for the wrong relationship type, filters on an inappropriate property, or misunderstands the core entities involved in the question.
2.  **DATA_ABSENT**: The query *does* accurately represent the user's question according to the schema, but the specific data requested simply does not exist in the graph. The query structure is appropriate for the question, but the graph lacks the necessary nodes or relationships satisfying the query's conditions.
3.  **AMBIGUOUS**: The user's question is too vague, unclear, or open to multiple interpretations, making it impossible to definitively determine if the query was mismatched or if the data is truly absent based *only* on the provided information.

**Output Format:**
Respond with ONLY ONE of the reason codes: `QUERY_MISMATCH`, `DATA_ABSENT`, or `AMBIGUOUS`. Do not provide any explanation or justification.

"""

EVALUATE_EMPTY_RESULT_USER_PROMPT = """

**Schema:**
{schema}

**User Question:**
{question}

**Executed Cypher Query (Returned 0 results):**
```cypher
{cypher}
```

**Diagnosis Code:**
"""


# --- Prompts for Revising Queries Evaluated as QUERY_MISMATCH ---

REVISE_EMPTY_RESULT_SYSTEM_PROMPT = """

You are an expert Neo4j Cypher query generator specializing in query correction. You are given a user question, a graph schema, and an initial Cypher query that returned zero results. The initial query has been evaluated as a `QUERY_MISMATCH`, meaning it likely failed to accurately capture the user's intent based on the question and schema.

**Graph Schema:**
{dynamic_schema}

**Your Task:**
Rewrite the original Cypher query to create a *revised* query that better aligns with the user's question and the provided schema.

**Revision Guidelines:**
1.  **Analyze Mismatch:** Identify the specific parts of the original query that likely caused the mismatch (e.g., wrong relationship type, incorrect node label, overly strict filtering on `name` property, wrong direction).
2.  **Reflect Intent:** Construct the revised query to target the entities, relationships, and properties implied by the user's question.
3.  **Consider Flexibility:**
    * If appropriate, make filters less strict (e.g., use `toLower(n.name) CONTAINS 'keyword'` instead of `toLower(n.name) = 'exact match'`).
    * If the relationship type was potentially wrong, try matching a more general pattern (`-[r]-`) and potentially filtering on `r.original` if keywords are available in the question.
    * Ensure the `RETURN` clause provides the specific information requested (e.g., return `entity.name` rather than the whole node).
4.  **Schema Adherence:** Ensure the revised query is valid according to the provided schema.
5.  **No Revision Case:** If, after careful analysis, you determine that the original query *was* the most plausible interpretation of the question despite returning no results, or if no meaningful revision seems possible to better capture the intent, then output the exact text `NO_REVISION`.

**Output Format:**
- If a revision is possible, output *only* the revised Cypher query enclosed in triple backticks: ```cypher\n[REVISED QUERY]\n```
- If no revision is appropriate, output *only* the exact text: `NO_REVISION`
- Do not include any explanations or comments outside the query block or the `NO_REVISION` text.

"""

REVISE_EMPTY_RESULT_USER_PROMPT = """

**Schema:**
{schema}

**User Question:**
{question}

**Original Cypher Query (Returned 0 results, evaluated as QUERY_MISMATCH):**
```cypher
{cypher}
```

**Revised Cypher Query or NO_REVISION:**
"""
# (Add this near the top of graph_rag_qa.py)
class EmbeddingModelWrapper:
    """Wraps a SentenceTransformer model to provide a get_text_embedding method."""
    def __init__(self, model):
        if not hasattr(model, 'encode'):
            raise ValueError("Wrapped model must have an 'encode' method.")
        self._model = model
        # Store the embedding dimension if easily accessible (optional)
        try:
             self.dimensions = self._model.get_sentence_embedding_dimension()
        except Exception:
             self.dimensions = None # Or determine from a test encode

    def get_text_embedding(self, text: str) -> List[float]:
        """Encodes text using the wrapped model's .encode() and returns a list."""
        # .encode() returns numpy array, convert to list
        embedding_array = self._model.encode([text])[0]
        return embedding_array.tolist()

    # Optional: Delegate other methods if needed
    def __getattr__(self, name):
        # Delegate other attribute access to the underlying model
        return getattr(self._model, name)

# Correction Step Imports
try:
    from text2cypher.correct_cypher import correct_cypher_step
    print("[INFO] Imported 'correct_cypher_step'.")
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    print("[INFO] Imported 'Neo4jGraphStore'.")
    llama_index_store_available = True
except ImportError as e:
    print(f"[WARN] Could not import 'correct_cypher_step' or 'Neo4jGraphStore' (Error: {e}). Cypher correction will be disabled.")
    llama_index_store_available = False
    async def correct_cypher_step(*args, **kwargs):
        print("[ERROR] correct_cypher_step not available due to import error.")
        await asyncio.sleep(0)
        return None

# <<< ADDED: Few-Shot Manager Import >>>
try:
    # Adjust path as needed
    from text2cypher.neo4j_fewshot_manager import Neo4jFewshotManager
    print("[INFO] Imported 'Neo4jFewshotManager'.")
    fewshot_manager_available = True
except ImportError as e:
    print(f"[WARN] Could not import 'Neo4jFewshotManager' (Error: {e}). Few-shot self-learning will be disabled.")
    print("[INFO] To enable few-shot learning, ensure 'text2cypher' is in your PYTHONPATH.")
    fewshot_manager_available = False
    # Define dummy class if import fails
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
     # ... (rest of logger setup) ...


class GraphRAGQA:
    """
    Handles GraphRAG Q&A with retry/correction and few-shot self-learning.
    """

    def __init__(self, *,
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 llm_instance_for_correction: Optional[Any],
                 llm_model: str, llm_api_key: str, llm_base_url: Optional[str] = None,
                 embedding_model_name: str,
                 chroma_path: str, collection_name: str,
                 # <<< ADDED: Database name for few-shot manager >>>
                 db_name: str = "neo4j", # Default or load from config
                 llm_config_extra: Optional[Dict[str, Any]] = None,
                 max_cypher_retries: int = 1):

        # --- Embedding Model & Function Initialization ---
        self.embed_model: Optional[SentenceTransformer] = None  # Keep track of the raw model
        self.embed_model_wrapper: Optional[EmbeddingModelWrapper] = None  # <<< ADDED Wrapper instance
        self.embedding_function: Optional[EmbeddingFunction] = None  # For Chroma

        # <<< --- ADD THIS LINE --- >>>
        self.embedding_model_name = embedding_model_name
        # <<< --- END ADDITION --- >>>

        if embeddings_available:
            try:
                logger.info(f"GraphRAGQA: Initializing embedding model object: {self.embedding_model_name}")
                # Store the raw SentenceTransformer model
                _raw_embed_model = SentenceTransformer(self.embedding_model_name)
                self.embed_model = _raw_embed_model  # Store raw model if needed elsewhere
                logger.info("GraphRAGQA: Raw embedding model object initialized.")

                # <<< Create the wrapper instance for the FewShot Manager >>>
                self.embed_model_wrapper = EmbeddingModelWrapper(_raw_embed_model)
                logger.info("GraphRAGQA: Embedding model wrapper created.")

                # Initialize ChromaDB embedding function wrapper (uses raw model name)
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
                logger.info("GraphRAGQA: ChromaDB Embedding function initialized.")

            except Exception as e:
                logger.error(
                    f"GraphRAGQA: Failed to initialize embedding model/wrapper/function '{self.embedding_model_name}'. Error: {e}",
                    exc_info=True)
                self.embed_model = None
                self.embed_model_wrapper = None  # <<< Set wrapper to None on error
                self.embedding_function = None
                self.is_vector_search_enabled = False
        else:
            logger.warning(
                "GraphRAGQA: Skipping embedding initialization because 'sentence-transformers' is not available.")
            self.embed_model = None
            self.embed_model_wrapper = None  # <<< Set wrapper to None
            self.embedding_function = None
            self.is_vector_search_enabled = False

        # --- (Rest of the __init__ method) ---

        """ Initializes components including retry and few-shot manager. """
        logger.info(f"Initializing GraphRAGQA Engine (Max Cypher Retries: {max_cypher_retries})...")
        self.llm_instance_for_correction = llm_instance_for_correction
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_config_extra = llm_config_extra or {}
        self.embedding_model_name = embedding_model_name
        self.max_cypher_retries = max_cypher_retries
        self.db_name = db_name # Store database name

        # --- Initialize status flags and components ---
        self.is_neo4j_connected = False
        self.is_vector_search_enabled = False
        self.driver: Optional[GraphDatabase.driver] = None
        self.embedding_function: Optional[EmbeddingFunction] = None
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.chroma_collection: Optional[chromadb.Collection] = None
        self.neo4j_graph_store_for_correction: Optional[Neo4jGraphStore] = None
        # <<< ADDED: Embedding model instance and few-shot manager >>>
        self.embed_model: Optional[SentenceTransformer] = None
        self.fewshot_manager: Optional[Neo4jFewshotManager] = None

        # --- Neo4j Connection ---
        try:
            logger.info(f"GraphRAGQA: Connecting to Neo4j SYNCHRONOUSLY at {neo4j_uri}")
            # vvv Use SYNC GraphDatabase vvv
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # vvv Optional: Verify connectivity synchronously vvv
            self.driver.verify_connectivity()
            self.is_neo4j_connected = True
            logger.info(f"GraphRAGQA: Successfully connected SYNCHRONOUS Neo4j Driver. Type: {type(self.driver)}")

            # vvv Cannot await verify_connectivity in sync __init__ vvv
            # You MUST remove or handle verify_connectivity asynchronously later.
            # Example: Remove the verify_connectivity() call from __init__ if present.
            # self.driver.verify_connectivity() # REMOVE THIS if it was here

            self.is_neo4j_connected = True  # Assume connected for now
            logger.info(f"GraphRAGQA: Initialized ASYNCHRONOUS Neo4j Driver. Type: {type(self.driver)}")  # Log type

            # --- Initialize Neo4jGraphStore for correction ---
            # This part likely uses its own connection based on config,
            # and might still use a sync connection internally unless configured otherwise.
            # Usually okay as it's often used in synchronous parts of LlamaIndex.
            if llama_index_store_available:
                try:
                    self.neo4j_graph_store_for_correction = Neo4jGraphStore(
                        username=neo4j_user, password=neo4j_password, url=neo4j_uri,
                        database=self.db_name, refresh_schema=False  # WORKAROUND
                    )
                    logger.info(
                        "GraphRAGQA: Initialized Neo4jGraphStore for correction step (schema refresh disabled).")
                except Exception as store_e:
                    logger.error(f"GraphRAGQA: Failed to initialize Neo4jGraphStore for correction: {store_e}",
                                 exc_info=True)
                    self.neo4j_graph_store_for_correction = None
            else:
                logger.warning("GraphRAGQA: Neo4jGraphStore not initialized for correction due to import failure.")
                self.neo4j_graph_store_for_correction = None

        except Exception as e:
            logger.error(f"GraphRAGQA: Fatal - Could not connect/initialize SYNCHRONOUS Neo4j Driver: {e}",
                         exc_info=True)
            self.is_neo4j_connected = False
            self.driver = None

        # --- Embedding Model & Function Initialization ---
        if embeddings_available:
            try:
                logger.info(f"GraphRAGQA: Initializing embedding model object: {self.embedding_model_name}")
                self.embed_model = SentenceTransformer(self.embedding_model_name)
                logger.info("GraphRAGQA: Embedding model object initialized.")
                # Initialize ChromaDB embedding function wrapper
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model_name
                )
                logger.info("GraphRAGQA: ChromaDB Embedding function initialized.")
            except Exception as e:
                 logger.error(f"GraphRAGQA: Failed to initialize embedding model/function '{self.embedding_model_name}'. Error: {e}", exc_info=True)
                 self.embed_model = None
                 self.embedding_function = None
                 self.is_vector_search_enabled = False # Vector search depends on this
        else:
            logger.warning("GraphRAGQA: Skipping embedding initialization because 'sentence-transformers' is not available.")
            self.embed_model = None
            self.embedding_function = None
            self.is_vector_search_enabled = False

        # --- ChromaDB Connection & Collection ---
        if self.embedding_function: # Only proceed if embedding function is ready
            # ... (Your existing ChromaDB init logic using self.embedding_function) ...
            try:
                logger.info(f"GraphRAGQA: Initializing ChromaDB client at path: {chroma_path}")
                Path(chroma_path).mkdir(parents=True, exist_ok=True)
                # With this consistent approach:
                settings = chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
                self.chroma_client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=settings
                )
                logger.info("GraphRAGQA: ChromaDB client initialized.")
                try:
                    logger.info(f"GraphRAGQA: Getting or creating ChromaDB collection: {collection_name}")
                    self.chroma_collection = self.chroma_client.get_or_create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    if self.chroma_collection:
                        self.is_vector_search_enabled = True
                        logger.info(f"GraphRAGQA: Successfully got/created ChromaDB collection '{collection_name}'. Count: {self.chroma_collection.count()}")
                    else:
                         logger.error(f"GraphRAGQA: Failed to get or create ChromaDB collection '{collection_name}' (returned None). Vector search disabled.")
                         self.is_vector_search_enabled = False
                except Exception as e:
                    logger.error(f"GraphRAGQA: Failed to get or create ChromaDB collection '{collection_name}'. Vector search disabled. Error: {e}", exc_info=True)
                    self.chroma_collection = None
                    self.is_vector_search_enabled = False
            except Exception as e:
                 logger.error(f"GraphRAGQA: Failed to initialize ChromaDB Persistent Client at path '{chroma_path}'. Vector search disabled. Error: {e}", exc_info=True)
                 self.chroma_client = None
                 self.chroma_collection = None
                 self.is_vector_search_enabled = False
        # No else needed here, is_vector_search_enabled remains False if embedding_function is None

        # <<< ADDED: Initialize Few-Shot Manager >>>
        if fewshot_manager_available:
            try:
                # Assumes Neo4jFewshotManager uses env vars for its connection details
                # or you modify it to accept connection params
                self.fewshot_manager = Neo4jFewshotManager()
                if self.fewshot_manager.graph_store: # Check if connection succeeded internally
                     logger.info("GraphRAGQA: Neo4jFewshotManager initialized successfully.")
                else:
                     logger.warning("GraphRAGQA: Neo4jFewshotManager initialized, but connection to its DB failed (check FEWSHOT_NEO4J env vars). Few-shot learning disabled.")
                     self.fewshot_manager = None # Disable if connection failed
            except Exception as fm_e:
                logger.error(f"GraphRAGQA: Failed to initialize Neo4jFewshotManager: {fm_e}", exc_info=True)
                self.fewshot_manager = None
        else:
            logger.warning("GraphRAGQA: Neo4jFewshotManager not initialized due to import failure.")
            self.fewshot_manager = None

        # Store base LLM config for Q&A synthesis
        self.llm_qna_config_base = {"model": llm_model, "api_key": llm_api_key, "base_url": llm_base_url}
        # Store any extra LLM parameters
        self.llm_qna_config_extra = llm_config_extra if llm_config_extra else {}

        logger.info(f"GraphRAGQA engine initialization complete. Neo4j Connected: {self.is_neo4j_connected}, Vector Search Enabled: {self.is_vector_search_enabled}, Few-Shot Learning Enabled: {self.fewshot_manager is not None}")


    def is_ready(self) -> bool:
        """Check if the engine has the minimum requirements to attempt Q&A."""
        return self.is_neo4j_connected and bool(self.llm_qna_config_base.get('api_key'))

        # (Inside the GraphRAGQA class in graph_rag_qa.py)

    async def close(self):
        """Closes the Neo4j database driver connection."""
        # Close main driver connection
        if self.driver:
            try:
                await self.driver.close()
                logger.info("GraphRAGQA: Closed main Neo4j Driver connection.")
                self.driver = None
            except Exception as e:
                logger.error("GraphRAGQA: Error closing main Neo4j Driver connection: %s", e, exc_info=True)

        # Close few-shot manager connection if it has one
        if self.fewshot_manager and hasattr(self.fewshot_manager,
                                            'graph_store') and self.fewshot_manager.graph_store:
            # <<< FIX: Use _driver instead of driver >>>
            if hasattr(self.fewshot_manager.graph_store, '_driver') and self.fewshot_manager.graph_store._driver:
                try:
                    self.fewshot_manager.graph_store._driver.close()
                    logger.info("GraphRAGQA: Closed Few-Shot Manager Neo4j connection.")
                except Exception as e:
                    logger.error("GraphRAGQA: Error closing Few-Shot Manager Neo4j connection: %s", e,
                                 exc_info=True)
            else:
                logger.warning(
                    "GraphRAGQA: Few-Shot Manager's graph_store does not have a closable '_driver' attribute.")

    # --- ADDED: Helper function to get schema string ---
    def _get_schema_string(self) -> str:
        """Retrieves the graph schema string using available methods."""
        schema_str = "Schema information could not be retrieved."  # Default message
        schema_retrieved_ok = False

        # Try LlamaIndex graph store method first
        if self.neo4j_graph_store_for_correction:
            try:
                retrieved_schema = self.neo4j_graph_store_for_correction.get_schema()
                if retrieved_schema and isinstance(retrieved_schema, str) and retrieved_schema.strip():
                    schema_str = retrieved_schema
                    logger.info("Successfully retrieved schema using Neo4jGraphStore.get_schema().")
                    schema_retrieved_ok = True
                else:
                    logger.warning(
                        f"Neo4jGraphStore.get_schema() did not return a valid string. Type: {type(retrieved_schema)}")
            except AttributeError:
                logger.warning("Neo4jGraphStore instance has no 'get_schema' method.")
            except Exception as schema_e:
                if "ProcedureNotFound" in str(schema_e):
                    logger.warning(
                        f"Could not get schema via Neo4jGraphStore: APOC procedure likely missing ({schema_e})")
                else:
                    logger.warning(f"Could not get schema via Neo4jGraphStore: {schema_e}")
                # Fallback will be attempted below if driver exists

        # Fallback schema retrieval using direct driver calls
        if not schema_retrieved_ok and self.driver:
            logger.info("Attempting fallback schema retrieval using direct driver calls.")
            try:
                with self.driver.session(database=self.db_name) as session:
                    labels_res = session.run(
                        "CALL db.labels() YIELD label RETURN collect(label) as labels").single()
                    rels_res = session.run(
                        "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as rel_types").single()
                    # Consider adding property keys for more detail if needed and efficient
                    # prop_res = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as props").single()
                labels_list = labels_res['labels'] if labels_res else []
                rels_list = rels_res['rel_types'] if rels_res else []
                # props_list = prop_res['props'] if prop_res else []
                # Format the schema string clearly for the LLM
                schema_str = f"- Node Labels: {labels_list}\n- Relationship Types: {rels_list}"
                # schema_str += f"\n- Property Keys: {props_list}" # Optional: Add properties
                logger.info(f"Using fallback schema: {schema_str}")
                schema_retrieved_ok = True
            except Exception as basic_schema_e:
                logger.error(f"Could not get basic schema from driver. Error: {basic_schema_e}", exc_info=True)
                # schema_str remains the default error message

        logger.debug(f"Schema string prepared for prompt:\n{schema_str}")
        return schema_str
    # --- END: Helper function ---

    def _format_fewshot_examples(self, examples: List[Dict]) -> str:
        """ Formats retrieved few-shot examples for inclusion in a prompt. """
        if not examples:
            return "No examples provided."
        # Format based on how the GENERATE_USER_TEMPLATE expects them
        # Assuming simple Question/Cypher pairs
        formatted = []
        for ex in examples:
            if ex.get("question") and ex.get("cypher"):
                 formatted.append(f"User input: {ex['question']}\nCypher query: {ex['cypher']}")
        return "\n\n".join(formatted) if formatted else "No valid examples found."

    # <<< NEW HELPER: Extract potential entities >>>
    def _extract_potential_entities(self, question: str) -> List[str]:
        """Extracts potential named entities from the question using simple patterns."""
        # Regex for capitalized words/phrases (potentially multi-word)
        # Handles simple cases like "Well A-1", "Daman Formation", "SLB"
        # May need refinement for more complex names or domain specifics
        pattern = r'\b([A-Z][a-zA-Z0-9#/\-.:]+(?:\s+[A-Z][a-zA-Z0-9#/\-.:]+)*)\b'
        mentions = re.findall(pattern, question)

        # Also look for terms in quotes
        quoted_mentions = re.findall(r'["\']([^"\']+)["\']', question)
        mentions.extend(quoted_mentions)

        # Simple filtering of common starting words
        common_starts = {"What", "Who", "Where", "When", "Why", "How", "Is", "Are", "The", "List", "Tell", "Find",
                         "Show"}
        filtered_mentions = [m.strip() for m in mentions if
                             m not in common_starts and len(m.strip()) > 1]  # Basic length filter

        # Deduplicate
        unique_mentions = sorted(list(set(filtered_mentions)), key=len, reverse=True)  # Longer first

        logger.debug(f"Extracted potential entity mentions: {unique_mentions}")
        return unique_mentions

    # <<< NEW HELPER: Link entities to graph (async) with Detailed Logging >>>
    def _link_entities(self, mentions: List[str]) -> Dict[str, Optional[str]]:
        """
        Looks up mentions in Neo4j to find canonical names, with detailed logging
        and error handling around database calls.
        """
        if not self.driver:
            logger.error("Entity Linker: Neo4j driver is not initialized. Cannot link entities.")
            return {m: None for m in mentions}
        if not mentions:
            logger.debug("Entity Linker: No mentions provided to link.")
            return {}

        linked_entities: Dict[str, Optional[str]] = {}
        logger.info(f"Entity Linker: Starting to link {len(mentions)} mentions. Driver type: {type(self.driver)}")

        # Use asyncio.gather for concurrent lookups
        def find_match(mention: str) -> Tuple[str, Optional[str]]:
            """Inner function to find match for a single mention."""
            canonical_name: Optional[str] = None
            mention_lower = mention.lower()
            query_success = False  # Flag to track if any query succeeded

            # --- 1. Try exact match (case-insensitive) ---
            exact_query = "MATCH (e:Entity) WHERE toLower(e.name) = $mention RETURN e.name LIMIT 1"
            logger.debug(f"Linker: Attempting exact match for '{mention}' (Query: {exact_query})")
            try:
                # Log before await
                logger.debug(f"Linker: Calling self.driver.execute_query (exact) for '{mention}'...")
                exact_result, summary, keys = self.driver.execute_query(
                    exact_query, mention=mention_lower, database_=self.db_name
                )
                # Log after successful await
                logger.debug(f"Linker: Await execute_query (exact) for '{mention}' completed.")
                query_success = True  # Mark query as successful

                if exact_result:
                    record_data = exact_result[0].data()
                    canonical_name = record_data.get('e.name')
                    if canonical_name:
                        logger.info(f"Linker: Exact match SUCCESS for '{mention}' -> '{canonical_name}'")
                    else:
                        logger.warning(
                            f"Linker: Exact match query ran for '{mention}' but 'e.name' key missing in result: {record_data}")
                else:
                    logger.debug(f"Linker: Exact match query ran for '{mention}', no record found.")

            except neo4j_exceptions.DriverError as de:
                logger.error(f"Linker: Neo4j DriverError during execute_query (exact) for '{mention}': {de}",
                             exc_info=True)
            except asyncio.TimeoutError as te:
                logger.error(f"Linker: Asyncio TimeoutError during execute_query (exact) for '{mention}': {te}",
                             exc_info=True)
            except RuntimeError as rte:
                # Catch the specific RuntimeError mentioned earlier
                logger.error(f"Linker: *** RuntimeError during execute_query (exact) for '{mention}': {rte} ***",
                             exc_info=True)
            except Exception as e:
                # Catch any other unexpected error during the query
                logger.error(
                    f"Linker: *** Unexpected Error during execute_query (exact) for '{mention}': {type(e).__name__} - {e} ***",
                    exc_info=True)

            # --- 2. Try CONTAINS match (shortest result) as fallback IF exact match failed ---
            if not canonical_name:  # Only proceed if exact match didn't yield a name
                contains_query = """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS $mention
                    RETURN e.name
                    ORDER BY size(e.name) ASC
                    LIMIT 1
                """
                logger.debug(f"Linker: Attempting CONTAINS match for '{mention}' (Query: CONTAINS ...)")
                try:
                    # Log before await
                    logger.debug(f"Linker: Calling self.driver.execute_query (contains) for '{mention}'...")
                    contains_result, summary, keys = self.driver.execute_query(
                        contains_query, mention=mention_lower, database_=self.db_name
                    )
                    # Log after successful await
                    logger.debug(f"Linker: Await execute_query (contains) for '{mention}' completed.")
                    query_success = True  # Mark query as successful (even if 0 results)

                    if contains_result:
                        record_data = contains_result[0].data()
                        canonical_name = record_data.get('e.name')
                        if canonical_name:
                            logger.info(f"Linker: CONTAINS match SUCCESS for '{mention}' -> '{canonical_name}'")
                        else:
                            logger.warning(
                                f"Linker: CONTAINS match query ran for '{mention}' but 'e.name' key missing in result: {record_data}")
                    else:
                        logger.debug(f"Linker: CONTAINS match query ran for '{mention}', no record found.")

                except neo4j_exceptions.DriverError as de:
                    logger.error(f"Linker: Neo4j DriverError during execute_query (contains) for '{mention}': {de}",
                                 exc_info=True)
                except asyncio.TimeoutError as te:
                    logger.error(
                        f"Linker: Asyncio TimeoutError during execute_query (contains) for '{mention}': {te}",
                        exc_info=True)
                except RuntimeError as rte:
                    logger.error(
                        f"Linker: *** RuntimeError during execute_query (contains) for '{mention}': {rte} ***",
                        exc_info=True)
                except Exception as e:
                    logger.error(
                        f"Linker: *** Unexpected Error during execute_query (contains) for '{mention}': {type(e).__name__} - {e} ***",
                        exc_info=True)

            # Log final outcome for the mention
            if not query_success:
                logger.warning(
                    f"Linker: FAILED to execute any query successfully for mention '{mention}' due to errors.")
            elif not canonical_name:
                logger.info(f"Linker: No match found for mention '{mention}' after all query attempts.")

            # Return the original mention and the found name (or None)
            return mention, canonical_name

        # --- End of inner find_match function ---

        # --- Replace asyncio.gather with a simple synchronous loop ---
        linked_entities: Dict[str, Optional[str]] = {}
        logger.debug(f"Entity Linker: Starting synchronous loop for {len(mentions)} mentions.")
        for mention in mentions:
            try:
                # Call the synchronous find_match helper
                m, cn = find_match(mention)
                linked_entities[m] = cn
            except Exception as loop_find_e:
                # Catch potential errors within the synchronous find_match call itself
                logger.error(f"Entity Linker: Error processing mention '{mention}' in loop: {loop_find_e}",
                             exc_info=True)
                linked_entities[mention] = None  # Ensure mention is in dict with None on error

        logger.debug("Entity Linker: Synchronous loop finished.")
        # --- End Replacement ---

        found_count = sum(1 for v in linked_entities.values() if v)
        logger.info(
            f"Entity Linker: Linking process finished. Found canonical names for {found_count}/{len(mentions)} mentions.")
        logger.debug(f"Entity Linker: Final linked entities mapping: {linked_entities}")
        return linked_entities

    # <<< MODIFIED: Function signature accepts linked_entities >>>
    def _generate_cypher_query(self, question: str, linked_entities: Dict[str, Optional[str]]) -> Optional[str]:
        """
        Uses an LLM to generate a Cypher query, incorporating few-shot examples,
        dynamic schema, and pre-linked entities. Includes robust extraction logic.
        """
        # <<< MODIFIED: Log message includes note about linked entities >>>
        # global user_prompt
        logger.debug(f"Generating Cypher query for question: '{question}' with pre-linked entities.")

        # --- Retrieve Few-Shot Examples (using Wrapper) ---
        # (Code for retrieving few-shot examples using self.embed_model_wrapper remains the same as the previous correct version)
        few_shot_examples_str = "No examples retrieved."
        retrieved_examples = []
        if self.fewshot_manager and self.embed_model_wrapper:
            logger.debug(
                f"Checking wrapper before retrieval: Type={type(self.embed_model_wrapper)}, Has Method?={hasattr(self.embed_model_wrapper, 'get_text_embedding')}")
            if not hasattr(self.embed_model_wrapper, 'get_text_embedding'):
                logger.error("CRITICAL: self.embed_model_wrapper lacks 'get_text_embedding' method!")
                few_shot_examples_str = "Error: Wrapper missing required method."
            else:
                try:
                    logger.info("Attempting to retrieve few-shot examples using wrapper...")
                    retrieved_examples = self.fewshot_manager.retrieve_fewshots(
                        question=question, database=self.db_name, embed_model=self.embed_model_wrapper
                    )
                    if retrieved_examples:
                        few_shot_examples_str = self._format_fewshot_examples(
                            retrieved_examples)  # Assumes _format_fewshot_examples exists
                        logger.info(f"Retrieved {len(retrieved_examples)} few-shot examples.")
                    else:
                        logger.info("No relevant few-shot examples found.")
                except Exception as fs_e:
                    logger.error(f"Error during few-shot retrieval call: {fs_e}", exc_info=True)
                    few_shot_examples_str = "Error retrieving examples."
        # ... (rest of if/elif block for manager/wrapper availability) ...

        # --- Prepare Schema (Dynamic Retrieval) ---
        # (Schema retrieval code remains the same as the previous correct version, using get_schema() and fallback)
        schema_str = "Schema information could not be retrieved."  # Default message
        schema_retrieved_ok = False
        if self.neo4j_graph_store_for_correction:
            try:
                # Attempt to get schema using LlamaIndex graph store method
                retrieved_schema = self.neo4j_graph_store_for_correction.get_schema()
                if retrieved_schema and isinstance(retrieved_schema, str) and retrieved_schema.strip():
                    schema_str = retrieved_schema  # Use the retrieved schema
                    logger.info("Successfully retrieved schema using Neo4jGraphStore.get_schema().")
                    logger.debug(f"Schema for prompt:\n{schema_str}")
                    schema_retrieved_ok = True
                else:
                    logger.warning(
                        f"Neo4jGraphStore.get_schema() did not return a valid string. Type: {type(retrieved_schema)}")
            except AttributeError:
                logger.warning("Neo4jGraphStore instance has no 'get_schema' method.")
            except Exception as schema_e:
                if "ProcedureNotFound" in str(schema_e):
                    logger.warning(
                        f"Could not get schema via Neo4jGraphStore: APOC procedure likely missing ({schema_e})")
                else:
                    logger.warning(f"Could not get schema via Neo4jGraphStore: {schema_e}")
                # Fallback will be attempted below if driver exists

        # Fallback schema retrieval using direct driver calls if store method failed/unavailable
        if not schema_retrieved_ok and self.driver:
            logger.info("Attempting fallback schema retrieval using direct driver calls.")
            try:
                with self.driver.session(database=self.db_name) as session:
                    # Get labels
                    labels_res = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels").single()
                    # Get relationship types
                    rels_res = session.run(
                        "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as rel_types").single()
                labels_list = labels_res['labels'] if labels_res else []
                rels_list = rels_res['rel_types'] if rels_res else []
                # Basic formatting (Consider adding properties for key labels/rels if needed)
                schema_str = f"- Node Labels: {labels_list}\n- Relationship Types: {rels_list}"
                logger.info(f"Using fallback schema: {schema_str}")
                schema_retrieved_ok = True  # Mark as OK since we got something
            except Exception as basic_schema_e:
                logger.error(f"Could not get basic schema from driver. Error: {basic_schema_e}", exc_info=True)
                # schema_str remains the default error message

        # --- Format Prompt for LLM (Injecting Schema) ---
        # <<< MODIFIED: Format the system prompt with the dynamic schema >>>
        try:
            # Assumes TEXT_TO_CYPHER_SYSTEM_PROMPT has {dynamic_schema} placeholder
            system_prompt = TEXT_TO_CYPHER_SYSTEM_PROMPT.format(dynamic_schema=schema_str)
        except KeyError:
            logger.error("Failed to format TEXT_TO_CYPHER_SYSTEM_PROMPT. Check for '{dynamic_schema}' placeholder.")
            system_prompt = TEXT_TO_CYPHER_SYSTEM_PROMPT  # Fallback to unformatted prompt

        # --- Format Linked Entities for Prompt ---
        # (This block defining linked_entities_prompt_str remains the same)
        linked_entities_str_parts = ["Mention -> Canonical Name in Graph (or None if not found):"]
        if linked_entities:
            for mention, canonical in sorted(linked_entities.items()):  # Sort for consistency
                linked_entities_str_parts.append(f"- '{mention}' -> '{canonical if canonical else 'None'}'")
        else:
            linked_entities_str_parts.append("(No specific entities pre-linked for this query)")
        linked_entities_prompt_str = "\n".join(linked_entities_str_parts)
        # --- End Format Linked Entities ---

        user_prompt: Optional[str] = None

        # <<< CORRECTED: Create structured input AFTER linked_entities_prompt_str is defined >>>
        # This combines the original question with the linked entity information
        structured_input = f"""User Question: {question}

        Pre-linked Entities:
        {linked_entities_prompt_str}
        """
        # Format user prompt using structured_input
        try:
            # Assumes GENERATE_USER_TEMPLATE uses {structured_input}, {schema}, {fewshot_examples}
            # You MUST update GENERATE_USER_TEMPLATE in prompts.py to expect {structured_input}
            user_prompt = GENERATE_USER_TEMPLATE.format(
                structured_input=structured_input,  # Pass the combined input
                schema=schema_str,
                fewshot_examples=few_shot_examples_str
            )
        except KeyError as e:
            logger.error(f"Failed to format GENERATE_USER_TEMPLATE. Check placeholders (e.g., missing {e}).")
            # Basic fallback if template is wrong
            user_prompt = f"{structured_input}\nCypher query:"

        # <<< FIX: Check if user_prompt was successfully created >>>
        if user_prompt is None:
            logger.error("Could not format user prompt for LLM call. Aborting Cypher generation.")
            return None  # Cannot proceed without a valid user prompt

        # # <<< MODIFIED: Format the user prompt with the dynamic schema >>>
        # try:
        #     # Assumes GENERATE_USER_TEMPLATE has {schema} placeholder
        #     user_prompt = GENERATE_USER_TEMPLATE.format(
        #         question=question, schema=schema_str, fewshot_examples=few_shot_examples_str
        #     )
        # except KeyError:
        #     logger.error("Failed to format GENERATE_USER_TEMPLATE. Check for '{schema}' placeholder.")
        #     user_prompt = f"User input: {question}\nCypher query:"  # Basic fallback

        # --- Call LLM for Cypher Generation ---
        try:
            temp = self.llm_qna_config_extra.get("cypher_temperature", 0.0)
            max_tokens = self.llm_qna_config_extra.get("cypher_max_tokens", 500)

            logger.debug(f"Calling LLM for Cypher generation with schema. Model: {self.llm_qna_config_base['model']}")
            # <<< MODIFIED: Pass formatted prompts to call_llm >>>
            response_text = call_llm(
                model=self.llm_qna_config_base['model'],
                user_prompt=user_prompt,  # Use formatted user prompt
                api_key=self.llm_qna_config_base['api_key'],
                system_prompt=system_prompt,  # Use formatted system prompt
                max_tokens=max_tokens,
                temperature=temp,
                base_url=self.llm_qna_config_base.get('base_url')  # Use .get for optional base_url
            )

            # --- Extract Cypher from Response (Existing Robust Logic) ---
            # (The existing logic for extracting the query from ```cypher ... ``` or raw response remains the same)
            generated_query: Optional[str] = None  # Initialize
            if not response_text:
                logger.warning("LLM call for Cypher generation returned empty response.")
                return None

            response_text_cleaned = response_text.strip()

            # 1. Try extracting from ```cypher ... ``` block
            cypher_match = re.search(r"```(?:cypher)?\s*([\s\S]+?)\s*```", response_text_cleaned, re.IGNORECASE)
            if cypher_match:
                generated_query = cypher_match.group(1).strip()
                logger.info("Extracted Cypher query from formatted code block.")
            # 2. If no block, check if the whole response looks like a query
            elif "NO_QUERY_GENERATED" not in response_text_cleaned.upper():
                # Basic check: does it start with common Cypher keywords?
                if re.match(r"^(MATCH|MERGE|CREATE|CALL|OPTIONAL MATCH)\b", response_text_cleaned, re.IGNORECASE):
                    logger.warning(
                        "LLM response did not contain backticks, but looks like a query. Using entire response.")
                    generated_query = response_text_cleaned
                else:
                    logger.warning(
                        f"LLM response did not contain a valid Cypher code block or 'NO_QUERY_GENERATED'. Response snippet: {response_text_cleaned[:500]}...")
                    return None  # Treat as failed generation if no block and doesn't look like query
            else:  # Contained "NO_QUERY_GENERATED"
                logger.info("LLM indicated no suitable Cypher query could be generated.")
                return None

            # 3. Validate the extracted/assumed query
            if generated_query:
                # Enhance validation: check for common keywords
                has_match = "MATCH" in generated_query.upper()
                has_return = "RETURN" in generated_query.upper()
                has_call = "CALL" in generated_query.upper()
                has_merge = "MERGE" in generated_query.upper()
                has_create = "CREATE" in generated_query.upper()
                has_delete = "DELETE" in generated_query.upper()

                # Allow basic valid patterns
                if (has_match and (has_return or has_call or has_delete)) or has_merge or has_create:
                    logger.info(
                        f"LLM generated Cypher query (Schema Provided: {schema_retrieved_ok}, Few-shots: {len(retrieved_examples)}):\n{generated_query}")
                    return generated_query  # Return valid query
                else:
                    logger.warning(f"Generated query seems invalid or incomplete based on keywords: {generated_query}")
                    return None  # Treat as invalid
            else:
                # This case should ideally be handled above, but as a safeguard
                logger.error("Query extraction logic failed unexpectedly.")
                return None

        except Exception as e:
            logger.error(f"Error during LLM call or Cypher extraction: {e}", exc_info=True)
            return None

    def _extract_main_entity(self, question: str) -> Optional[str]:
        # ... (implementation remains the same) ...
        patterns = [
            r"well\s+([\w\d#/\-.:]+)", r"project\s+([\w\d\s\-]+)", r"field\s+([\w\d\s\-]+)",
            r"formation\s+([\w\d\s\-]+)", r"platform\s+([\w\d\s\-]+)", r"company\s+([\w\d\s\.\-]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                entity = ' '.join(match.group(1).split()).lower()
                logger.info(f"Extracted entity using pattern '{pattern}': {entity}")
                return entity
        quoted_match = re.search(r'["\']([^"\']+)["\']', question)
        if quoted_match:
            entity = quoted_match.group(1).strip().lower()
            logger.info(f"Extracted entity using quotes: {entity}")
            return entity
        cap_words = re.findall(r'\b[A-Z][a-zA-Z0-9#/\-.:]+\b', question)
        cap_words = [w for w in cap_words if w not in ["What", "Who", "Where", "When", "Why", "How", "Is", "Are", "The", "List", "Tell"]]
        if cap_words:
             potential_entities = []
             current_entity = []
             for word in cap_words:
                 if word[0].isupper(): current_entity.append(word)
                 else:
                     if current_entity: potential_entities.append(" ".join(current_entity).lower())
                     current_entity = []
             if current_entity: potential_entities.append(" ".join(current_entity).lower())
             if potential_entities:
                  entity = max(potential_entities, key=len)
                  logger.info(f"Extracted entity using capitalization fallback: {entity}")
                  return entity
        logger.warning(f"Could not reliably extract main entity from question: '{question}'")
        return None


    def _query_neo4j(self, cypher_query: str, params: Optional[Dict]=None) -> List[Dict[str, Any]]:
        # ... (implementation remains the same - raises errors) ...
        if not self.is_neo4j_connected or not self.driver:
            logger.error("GraphRAGQA: Neo4j Driver not available for query.")
            raise ConnectionError("Neo4j Driver not connected or initialized.")
        if not cypher_query:
            logger.warning("GraphRAGQA: Skipping Neo4j query (no query provided).")
            return []
        logger.info("GraphRAGQA: Executing Cypher query...")
        logger.debug(f"GraphRAGQA: Query:\n{cypher_query}\nParams: {params}")
        records, summary, keys = self.driver.execute_query(
            cypher_query, parameters_=params, database_="neo4j"
        )
        count = len(records)
        logger.info("GraphRAGQA: Cypher query returned %d records.", count)
        return [record.data() for record in records]

    @monitor_performance("vector_search")

    # Add this as a class method in GraphRAGQA:

    def _query_vector_db(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Add caching logic at the start of your existing method
        if not hasattr(self, '_query_cache'):
            self._query_cache = {}

        cache_key = get_query_hash(question, top_k)  # Correct - no self.

        if cache_key in self._query_cache:
            logger.info("GraphRAGQA: Using cached vector search results")
            return self._query_cache[cache_key]

        if not self.is_vector_search_enabled or not self.embedding_function or not self.chroma_collection:
            logger.warning("GraphRAGQA: Vector search skipped (not enabled or components missing).")
            return []
        logger.info("GraphRAGQA: Performing vector search for question (top %d)...", top_k)
        try:
            results = self.chroma_collection.query(query_texts=[question], n_results=top_k, include=['documents', 'distances', 'metadatas'])
            logger.debug(f"GraphRAGQA: Raw ChromaDB query results: {results}")
            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                ids, documents, distances, metadatas = (results.get(k, [[]])[0] for k in ['ids', 'documents', 'distances', 'metadatas'])
                num_results = len(ids)
                documents = documents + ['[Error retrieving document]'] * (num_results - len(documents)) if len(documents) < num_results else documents[:num_results]
                distances = distances + [-1.0] * (num_results - len(distances)) if len(distances) < num_results else distances[:num_results]
                metadatas = metadatas + [{}] * (num_results - len(metadatas)) if len(metadatas) < num_results else metadatas[:num_results]
                for i in range(num_results):
                    formatted_results.append({"text": documents[i], "metadata": metadatas[i], "distance": distances[i]})
                logger.info("GraphRAGQA: Vector search returned %d formatted results.", len(formatted_results))
            else: logger.info("GraphRAGQA: Vector search returned no results.")
            return formatted_results
        except Exception as e:
            logger.error("GraphRAGQA: Error during vector database query: %s", e, exc_info=True)
            return []


    def _format_context(self, graph_results: Optional[List[Dict]], vector_results: List[Dict]) -> str:
        # ... (implementation remains the same) ...
        context_parts = []
        if vector_results:
            vector_context_str = "Relevant Text Snippets (Vector Search):\n---\n"
            vector_context_str += "\n---\n".join([
                 f"Source Document: {chunk.get('metadata', {}).get('source_document', 'Unknown')}\n"
                 f"Content: {chunk.get('text', '[No text found]')}"
                 for chunk in vector_results
            ])
            vector_context_str += "\n---"
            context_parts.append(vector_context_str)
        if graph_results is not None:
            graph_context_str = "Knowledge Graph Facts:\n---\n"
            if graph_results:
                graph_facts_list = []
                max_graph_records = 15
                seen_facts = set()
                for i, record in enumerate(graph_results):
                    if len(graph_facts_list) >= max_graph_records:
                        logger.warning("GraphRAGQA: Truncating graph facts in context (%d max).", max_graph_records)
                        break
                    subj = record.get('subject', record.get('e1.name', record.get('e.name', '?')))
                    pred_type = record.get('type', '?')
                    pred_orig = record.get('predicate', '?')
                    pred = pred_orig if pred_orig != '?' else pred_type
                    obj = record.get('object', record.get('related.name', record.get('related_entity', '?')))
                    if len(record) == 1: fact_str = f"- {list(record.keys())[0]}: {list(record.values())[0]}"
                    elif subj != '?' and pred != '?' and obj != '?': fact_str = f"- {subj} -[{pred}]-> {obj}"
                    else: fact_str = "- " + ", ".join([f"{k}: {v}" for k, v in record.items()])
                    fact_tuple = tuple(sorted(record.items()))
                    if fact_tuple in seen_facts: continue
                    seen_facts.add(fact_tuple)
                    graph_facts_list.append(fact_str)
                if graph_facts_list: graph_context_str += "\n".join(graph_facts_list)
                else: graph_context_str += "No specific facts found."
            else:
                 graph_context_str += "No relevant facts found or query failed/not generated."
            graph_context_str += "\n---"
            context_parts.append(graph_context_str)
        if not context_parts: return "No relevant context found in knowledge graph or vector search."
        if len(context_parts) == 1 and "Knowledge Graph Facts" in context_parts[0] and ("No relevant facts found" in context_parts[0] or "No specific facts found" in context_parts[0]):
            return "No relevant context found in knowledge graph or vector search."
        return "\n\n".join(context_parts).strip()


    def _synthesize_answer(self, query: str, context: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        # ... (implementation remains the same) ...
        logger.info("GraphRAGQA: Generating final answer using LLM with combined context...")
        final_system_prompt = """You are a helpful assistant answering questions based *only* on the provided context.
        The context may include 'Knowledge Graph Context' (structured facts) and 'Relevant Text Snippets' (unstructured text).
        1. First, check if the 'Knowledge Graph Context' directly answers the user's question about the entities mentioned. If they do, synthesize your answer primarily from these facts.
        2. Use the 'Relevant Text Snippets' ONLY to add supporting details or context IF they are clearly related to the entities and relationships already identified from the graph facts or the user's question.
        3. **CRITICAL:** Ignore any information in the 'Relevant Text Snippets' that seems unrelated to the core entities or the specific question asked, even if the snippets were retrieved. Do not merge unrelated topics.
        4. If the answer cannot be confidently determined from the relevant parts of the context, state that you don't know or cannot answer based on the provided information. Do not make up information."""
        final_user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        try:
            temp = self.llm_qna_config_extra.get("qna_temperature", 0.1)
            max_tokens = self.llm_qna_config_extra.get("qna_max_tokens", 500)
            answer_text = call_llm(
                model=self.llm_qna_config_base['model'], user_prompt=final_user_prompt,
                api_key=self.llm_qna_config_base['api_key'], system_prompt=final_system_prompt,
                temperature=temp, max_tokens=max_tokens, base_url=self.llm_qna_config_base['base_url']
            )
            logger.info("GraphRAGQA: Successfully generated final answer.")
            return {"answer": answer_text.strip() if answer_text else "Sorry, I could not generate an answer.", "sources": context_chunks}
        except Exception as e:
            logger.error(f"GraphRAGQA: Error during final LLM call for answer generation: {e}", exc_info=True)
            return {"answer": f"Sorry, I encountered an error generating the answer: {e}", "sources": context_chunks}


    def _get_corrected_cypher(self, question: str, failed_cypher: str, error_message: str) -> Optional[str]:
        # ... (implementation remains the same) ...
        logger.warning("Cypher correction step (_get_corrected_cypher) is temporarily disabled in sync mode.")
        # Return None immediately - effectively skipping correction
        return None

        # if not llama_index_store_available or not self.llm_instance_for_correction or not self.neo4j_graph_store_for_correction:
        #     logger.error("Cannot correct Cypher: Correction step function, LLM instance, or Graph Store missing/failed to import/initialize.")
        #     return None
        # try:
        #     logger.info(f"Attempting Cypher correction for query: {failed_cypher}")
        #     logger.debug(f"Correction based on error: {error_message}")
        #     corrected_query = correct_cypher_step(
        #         llm=self.llm_instance_for_correction, graph_store=self.neo4j_graph_store_for_correction,
        #         subquery=question, cypher=failed_cypher, errors=str(error_message)
        #     )
        #     if corrected_query and isinstance(corrected_query, str):
        #         corrected_query = corrected_query.strip()
        #         if corrected_query and corrected_query != failed_cypher.strip():
        #             logger.info(f"LLM suggested correction: {corrected_query}")
        #             return corrected_query
        #         else:
        #             logger.warning("Correction attempt yielded same or empty query.")
        #             return None
        #     else:
        #         logger.warning(f"Correction step did not return a valid string query. Response: {corrected_query}")
        #         return None
        # except Exception as e:
        #     logger.error(f"Error during Cypher correction LLM call: {e}", exc_info=True)
        #     return None

    def _evaluate_and_revise_empty_result_query(self, question: str, empty_query: str) -> Optional[str]:
        """
        Evaluates why a query returned 0 results and attempts revision if appropriate.
        Injects dynamically fetched schema into prompts.
        """
        logger.info(f"Evaluating query that returned empty results:\n{empty_query}")
        eval_llm_model = self.llm_qna_config_base.get('model')
        eval_api_key = self.llm_qna_config_base.get('api_key')
        eval_base_url = self.llm_qna_config_base.get('base_url')

        # Check if necessary components/config exist
        if not self.llm_instance_for_correction:  # Still useful to check if correction capability is intended
            logger.warning("Correction LLM instance not available, skipping evaluation/revision.")
            return None
        if not eval_llm_model or not eval_api_key:
            logger.error("Cannot evaluate/revise empty query: Main LLM model or API key missing in config.")
            return None

        # --- Get Schema (Use Helper Function) ---
        # <<< MODIFIED: Call the helper function >>>
        schema_str = self._get_schema_string()
        # Log if schema retrieval failed, but proceed as prompts have fallbacks
        if "could not be retrieved" in schema_str:
            logger.warning("Schema retrieval failed for evaluation/revision. Prompts will use default message.")
        # --- End Schema Retrieval ---

        # --- 1. Evaluate ---
        evaluation_result = "UNKNOWN"
        try:
            # <<< MODIFIED: Format system prompt with dynamic schema >>>
            try:
                # Assumes EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT has {dynamic_schema}
                eval_system_prompt = EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT.format(dynamic_schema=schema_str)
            except KeyError:
                logger.error("Failed to format EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT. Check for '{dynamic_schema}'.")
                eval_system_prompt = EVALUATE_EMPTY_RESULT_SYSTEM_PROMPT  # Fallback

            # <<< MODIFIED: Format user prompt with dynamic schema (uses {schema}) >>>
            try:
                # Assumes EVALUATE_EMPTY_RESULT_USER_PROMPT has {schema}
                eval_user_prompt = EVALUATE_EMPTY_RESULT_USER_PROMPT.format(question=question, cypher=empty_query,
                                                                            schema=schema_str)
            except KeyError:
                logger.error("Failed to format EVALUATE_EMPTY_RESULT_USER_PROMPT. Check for '{schema}'.")
                # Basic fallback
                eval_user_prompt = f"Question: {question}\nCypher:\n{empty_query}\nDiagnosis Code:"

            logger.debug("Calling LLM for empty query evaluation.")
            eval_response = call_llm(
                model=eval_llm_model, api_key=eval_api_key, base_url=eval_base_url,
                user_prompt=eval_user_prompt, system_prompt=eval_system_prompt,
                temperature=0.1, max_tokens=50
            )
            # (Processing eval_response remains the same)
            if eval_response and isinstance(eval_response, str):
                eval_response = eval_response.strip().upper()
                if eval_response in ["QUERY_MISMATCH", "DATA_ABSENT", "AMBIGUOUS"]:
                    evaluation_result = eval_response
                    logger.info(f"Evaluation result for empty query: {evaluation_result}")
                else:
                    logger.warning(f"Unexpected evaluation response: {eval_response}")
            else:
                logger.warning("Empty or invalid evaluation response from LLM.")

        except Exception as eval_e:
            logger.error(f"Error during empty result evaluation LLM call: {eval_e}", exc_info=True);
            return None

        # --- 2. Revise ---
        if evaluation_result == "QUERY_MISMATCH":
            logger.info("Evaluation suggests QUERY_MISMATCH, attempting revision...")
            try:
                # <<< MODIFIED: Format system prompt with dynamic schema >>>
                try:
                    # Assumes REVISE_EMPTY_RESULT_SYSTEM_PROMPT has {dynamic_schema}
                    revise_system_prompt = REVISE_EMPTY_RESULT_SYSTEM_PROMPT.format(dynamic_schema=schema_str)
                except KeyError:
                    logger.error("Failed to format REVISE_EMPTY_RESULT_SYSTEM_PROMPT. Check for '{dynamic_schema}'.")
                    revise_system_prompt = REVISE_EMPTY_RESULT_SYSTEM_PROMPT  # Fallback

                # <<< MODIFIED: Format user prompt with dynamic schema (uses {schema}) >>>
                try:
                    # Assumes REVISE_EMPTY_RESULT_USER_PROMPT has {schema}
                    revise_user_prompt = REVISE_EMPTY_RESULT_USER_PROMPT.format(question=question, cypher=empty_query,
                                                                                schema=schema_str)
                except KeyError:
                    logger.error("Failed to format REVISE_EMPTY_RESULT_USER_PROMPT. Check for '{schema}'.")
                    # Basic fallback
                    revise_user_prompt = f"Question: {question}\nOriginal Cypher:\n{empty_query}\nRevised Cypher or NO_REVISION:"

                logger.debug("Calling LLM for query revision.")
                revise_response = call_llm(
                    model=eval_llm_model, api_key=eval_api_key, base_url=eval_base_url,
                    user_prompt=revise_user_prompt, system_prompt=revise_system_prompt,
                    temperature=0.3, max_tokens=500
                )
                # (Logic for extracting and validating revised query remains the same)
                revised_query: Optional[str] = None
                if revise_response and isinstance(revise_response, str):
                    if "NO_REVISION" in revise_response.upper():
                        logger.info("LLM indicated NO_REVISION possible.")
                        return None
                    # --- Extract revised query (existing logic) ---
                    cypher_match = re.search(r"```(?:cypher)?\s*([\s\S]+?)\s*```", revise_response, re.IGNORECASE)
                    if cypher_match:
                        revised_query = cypher_match.group(1).strip()
                    elif re.match(r"^(MATCH|MERGE|CREATE|CALL|OPTIONAL MATCH)\b", revise_response.strip(),
                                  re.IGNORECASE):
                        revised_query = revise_response.strip()
                        logger.warning("Revision response did not contain backticks, but looks like a query.")
                    # --- Validate revised query (existing logic) ---
                    if revised_query and revised_query != empty_query.strip():
                        # Basic validation
                        if "MATCH" in revised_query.upper() and (
                                "RETURN" in revised_query.upper() or "DELETE" in revised_query.upper() or "CALL" in revised_query.upper()):
                            logger.info(f"LLM suggested revision:\n{revised_query}")
                            return revised_query
                        else:
                            logger.warning(f"Revised query seems invalid or incomplete: {revised_query}")
                            return None
                    else:
                        logger.warning("Revision attempt yielded same or empty query.")
                        return None
                else:
                    logger.warning("Empty or invalid revision response from LLM.")
                    return None

            except Exception as revise_e:
                logger.error(f"Error during empty result revision LLM call: {revise_e}", exc_info=True)
                return None
        else:
            logger.info(f"Skipping revision based on evaluation result: {evaluation_result}")
            return None
        return None  # Explicitly return None if no revision occurs


    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answers a user's question using Graph and Vector retrieval,
        with retry/correction for execution errors AND evaluation/revision for empty results.
        Also stores successful self-healing examples as few-shots.
        """
        logger.info(f"--- GraphRAGQA: Starting RAG pipeline for question: {question} ---")
        if not self.is_ready():
            logger.error("GraphRAGQA: Engine not ready.")
            return {"answer": "Error: Backend connection or configuration is not ready.", "sources": [], "cypher_query": "N/A"}

        retries = 0
        current_cypher_query: Optional[str] = None
        initial_cypher_query: Optional[str] = None # <<< Store initial query
        graph_results: Optional[List[Dict]] = None
        execution_error: Optional[Exception] = None
        correction_attempted = False
        revision_attempted = False
        self_healing_successful = False # <<< Flag for successful correction/revision
        linked_entities: Dict[str, Optional[str]] = {}  # Store linked entities

        # <<< ADDED: Step 0: Pre-Query Entity Linking >>>
        try:
            potential_mentions = self._extract_potential_entities(question)
            if potential_mentions:
                # Run the async linking function
                linked_entities = self._link_entities(potential_mentions)
        except Exception as link_e:
            logger.error(f"Error during pre-query entity linking: {link_e}", exc_info=True)
            # Continue without linked entities if linking fails

        # <<< Check and use the WRAPPER for storing few-shot examples >>>
        if self.fewshot_manager and self.embed_model_wrapper:
            try:
                logger.info(
                    f"Storing successful self-healing example: Q: '{question}', Cypher: '{current_cypher_query}'")
                self.fewshot_manager.store_fewshot_example(
                    question=question,
                    cypher=current_cypher_query,
                    llm=self.llm_model,
                    embed_model=self.embed_model_wrapper,  # <<< PASS WRAPPER HERE
                    database=self.db_name,
                    success=True
                )
            except Exception as store_e:
                logger.error(f"Failed to store few-shot example: {store_e}", exc_info=True)
        else:
            logger.warning("Cannot store few-shot example: Few-shot manager or embedding model wrapper not available.")

        # --- Step 1: Initial Cypher Query Generation (passing linked entities) ---
        # <<< MODIFIED: Pass linked_entities to generation function >>>
        initial_cypher_query = self._generate_cypher_query(question, linked_entities)
        current_cypher_query = initial_cypher_query

        # --- Step 2: Execute/Evaluate/Correct/Revise Loop ---
        if not current_cypher_query:
            logger.warning("GraphRAGQA: Initial Cypher generation failed.")
            graph_results = []
        else:
            while retries <= self.max_cypher_retries:
                logger.info(f"Attempting Cypher execution/evaluation (Attempt {retries + 1}/{self.max_cypher_retries + 1})...")
                execution_error = None
                try:
                    # --- Attempt Execution ---
                    graph_results = self._query_neo4j(current_cypher_query)
                    logger.info(f"Cypher query executed successfully (Attempt {retries + 1}). Found {len(graph_results)} results.")

                    # --- Evaluate if Results are Empty ---
                    if not graph_results:
                        logger.warning("Query executed successfully but returned 0 results.")
                        execution_error = None
                        retries += 1

                        if retries <= self.max_cypher_retries:
                            logger.info(f"Attempting evaluation/revision for empty result (Retry {retries}/{self.max_cypher_retries})...")
                            revision_attempted = True
                            revised_query = self._evaluate_and_revise_empty_result_query(question, current_cypher_query)

                            if revised_query:
                                current_cypher_query = revised_query
                                graph_results = None
                                continue # Retry with revised query
                            else:
                                logger.info("Evaluation/revision did not yield a new query. Stopping.")
                                graph_results = []
                                break
                        else:
                            logger.error(f"Max retries ({self.max_cypher_retries}) reached after empty result. Stopping.")
                            graph_results = []
                            break
                    else:
                        # Query succeeded AND returned results
                        # <<< Check if correction/revision happened before success >>>
                        if correction_attempted or revision_attempted:
                             self_healing_successful = True
                        break # Exit loop successfully

                except (neo4j_exceptions.CypherSyntaxError, neo4j_exceptions.ClientError, Exception) as e:
                    # --- Handle Execution Errors ---
                    execution_error = e
                    logger.warning(f"Cypher execution failed (Attempt {retries + 1}): {type(e).__name__} - {e}")
                    retries += 1

                    if retries <= self.max_cypher_retries:
                        logger.info(f"Attempting correction for execution error (Retry {retries}/{self.max_cypher_retries})...")
                        correction_attempted = True
                        corrected_query = self._get_corrected_cypher(question, current_cypher_query, str(e))

                        if corrected_query:
                            current_cypher_query = corrected_query
                            graph_results = None
                            continue # Retry with corrected query
                        else:
                            logger.error("Failed to get a valid correction. Stopping retries.")
                            graph_results = []
                            break
                    else:
                        logger.error(f"Max retries ({self.max_cypher_retries}) reached after execution error.")
                        graph_results = []
                        break

        # <<< ADDED: Store Few-Shot Example if Self-Healing Occurred >>>
        if self_healing_successful and current_cypher_query:
             if self.fewshot_manager and self.embed_model:
                 try:
                     logger.info(f"Storing successful self-healing example: Q: '{question}', Cypher: '{current_cypher_query}'")
                     # Assuming store_fewshot_example handles async if needed, or run in thread
                     # Make sure the manager has access to the necessary LLM/embedding info if needed internally
                     self.fewshot_manager.store_fewshot_example(
                         question=question,
                         cypher=current_cypher_query, # Store the FINAL successful query
                         llm=self.llm_model, # Store model name used for generation/correction
                         embed_model=self.embed_model_wrapper, # Pass the model object
                         database=self.db_name,
                         success=True # Mark as successful
                     )
                 except Exception as store_e:
                     logger.error(f"Failed to store few-shot example: {store_e}", exc_info=True)
             else:
                 logger.warning("Cannot store few-shot example: Few-shot manager or embedding model not available.")


        # --- Step 3: Perform Vector Search ---
        vector_search_top_k = self.llm_config_extra.get("vector_search_top_k", 5)
        similar_chunks = self._query_vector_db(question, top_k=vector_search_top_k)

        # --- Step 4: Format Combined Context ---
        context_str = self._format_context(graph_results, similar_chunks)
        logger.debug("GraphRAGQA: Combined Context for LLM:\n%s", context_str)

        # --- Step 5: Handle No Context Found ---
        if (graph_results is None or not graph_results) and not similar_chunks:
            logger.warning("GraphRAGQA: No context retrieved from graph or vector store.")
            final_answer = "Sorry, I could not find relevant information to answer your question."
            if execution_error: final_answer += f" (Graph query execution failed: {type(execution_error).__name__})"
            elif graph_results == []: final_answer += " (Graph query returned no results after attempts)."
            return {"answer": final_answer, "sources": [], "cypher_query": current_cypher_query or initial_cypher_query or "N/A"}

        # --- Step 6: Generate Final Answer with LLM ---
        answer_dict = self._synthesize_answer(question, context_str, similar_chunks)

        # --- Step 7: Augment Final Result ---
        answer_dict["cypher_query"] = current_cypher_query or initial_cypher_query or "N/A" # Show final query used
        # <<< MODIFIED: Include linked_entities in the return dict >>>
        answer_dict["linked_entities"] = linked_entities
        if execution_error:
             answer_dict["error_info"] = f"Query execution failed after retries ({retries-1} attempt(s)): {type(execution_error).__name__} - {execution_error}"
        elif self_healing_successful: # Use the flag here
             answer_dict["info"] = f"Query executed successfully after self-healing ({retries-1} attempt(s))."
        elif graph_results == [] :
             answer_dict["info"] = f"Query executed but returned no results after {retries-1} attempt(s)."

        return answer_dict


# --- Example Usage (main block) ---
if __name__ == "__main__":
    print("--- GraphRAGQA Script Start (Combined Vector/Graph with Retry & Self-Learning) ---")

    # --- Configuration Loading (Ensure this logic is correct as per previous steps) ---
    config_data = {}
    llm_for_correction = None
    try:
        # 1. Load from config.toml
        toml_config_path = Path("config.toml")
        if tomllib and toml_config_path.is_file():
            with open(toml_config_path, "rb") as f: config_toml = tomllib.load(f)
            logger.info("Loaded config from config.toml")
            llm_config = config_toml.get("llm", {})
            config_data['LLM_MODEL'] = llm_config.get("model")
            config_data['LLM_API_KEY'] = llm_config.get("api_key")
            config_data['LLM_BASE_URL'] = llm_config.get("base_url")
            config_data['LLM_EXTRA_PARAMS'] = llm_config.get("parameters", {})
            config_data['EMBEDDING_MODEL'] = config_toml.get("embeddings", {}).get("model_name", "all-MiniLM-L6-v2")
            config_data['CHROMA_PERSIST_PATH'] = config_toml.get("vector_db", {}).get('persist_directory', "./chroma_db_embeddings")
            config_data['COLLECTION_NAME'] = config_toml.get("vector_db", {}).get('collection_name', "doc_pipeline_embeddings")
            # <<< ADDED: Load DB Name >>>
            config_data['DB_NAME'] = config_toml.get("database", {}).get("name", "neo4j") # Example section/key
        else: logger.warning("config.toml not found or tomllib not available."); config_toml = {}

        # 2. Load from graph_config.ini (Fallback)
        config_path_ini = Path("graph_config.ini")
        if config_path_ini.is_file():
            neo4j_config = configparser.ConfigParser(); neo4j_config.read(config_path_ini)
            config_data.setdefault('NEO4J_URI', neo4j_config.get("neo4j", "uri", fallback=None))
            config_data.setdefault('NEO4J_USER', neo4j_config.get("neo4j", "user", fallback=None))
            config_data.setdefault('NEO4J_PASSWORD', neo4j_config.get("neo4j", "password", fallback=None))
            config_data.setdefault('CHROMA_PERSIST_PATH', neo4j_config.get("vector_db", "chroma_path", fallback=config_data.get('CHROMA_PERSIST_PATH')))
            config_data.setdefault('COLLECTION_NAME', neo4j_config.get("vector_db", "collection_name", fallback=config_data.get('COLLECTION_NAME')))
            # <<< ADDED: Load DB Name Fallback >>>
            config_data.setdefault('DB_NAME', neo4j_config.get("database", "name", fallback=config_data.get('DB_NAME', 'neo4j')))
            logger.info("Loaded/updated config from graph_config.ini")
        else: logger.warning("graph_config.ini not found.")

        # 3. Override with Environment Variables
        config_data['NEO4J_URI'] = os.getenv('NEO4J_URI', config_data.get('NEO4J_URI'))
        config_data['NEO4J_USER'] = os.getenv('NEO4J_USER', config_data.get('NEO4J_USER'))
        config_data['NEO4J_PASSWORD'] = os.getenv('NEO4J_PASSWORD', config_data.get('NEO4J_PASSWORD'))
        config_data['LLM_API_KEY'] = os.getenv('LLM_API_KEY', os.getenv('GOOGLE_API_KEY', config_data.get('LLM_API_KEY')))
        config_data['LLM_MODEL'] = os.getenv('LLM_MODEL', config_data.get('LLM_MODEL'))
        config_data['LLM_BASE_URL'] = os.getenv('LLM_BASE_URL', config_data.get('LLM_BASE_URL'))
        config_data['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL', config_data.get('EMBEDDING_MODEL'))
        config_data['CHROMA_PERSIST_PATH'] = os.getenv('CHROMA_PERSIST_PATH', config_data.get('CHROMA_PERSIST_PATH'))
        config_data['COLLECTION_NAME'] = os.getenv('COLLECTION_NAME', config_data.get('COLLECTION_NAME'))
        # <<< ADDED: Load DB Name Env Var >>>
        config_data['DB_NAME'] = os.getenv('DB_NAME', config_data.get('DB_NAME'))


        # 4. Final Validation
        required_keys_map = {
            "NEO4J_URI": config_data.get('NEO4J_URI'), "NEO4J_USER": config_data.get('NEO4J_USER'),
            "NEO4J_PASSWORD": config_data.get('NEO4J_PASSWORD'), "LLM_MODEL": config_data.get('LLM_MODEL'),
            "LLM_API_KEY": config_data.get('LLM_API_KEY'), "EMBEDDING_MODEL": config_data.get('EMBEDDING_MODEL'),
            "CHROMA_PERSIST_PATH": config_data.get('CHROMA_PERSIST_PATH'), "COLLECTION_NAME": config_data.get('COLLECTION_NAME'),
            "DB_NAME": config_data.get('DB_NAME') # <<< ADDED: Validate DB Name >>>
        }
        missing_keys = [k for k, v in required_keys_map.items() if v is None or (isinstance(v, str) and not v.strip())]
        if missing_keys: raise ValueError(f"Missing required configuration values: {', '.join(missing_keys)}")

        logger.info("Configuration loaded and validated successfully.")
        logger.debug(f"DB_NAME for few-shots: {config_data.get('DB_NAME')}")

        # 5. Instantiate LLM for Correction Step
        correction_llm_model = config_data['LLM_MODEL']
        correction_api_key = config_data['LLM_API_KEY']
        correction_base_url = config_data.get('LLM_BASE_URL')

        if llama_index_store_available:
            try:
                # >>> Using LlamaIndex Gemini LLM <<<
                from llama_index.llms.gemini import Gemini # Import Gemini LLM
                if correction_llm_model and correction_api_key:
                    llm_for_correction = Gemini(
                        model_name=correction_llm_model, # Gemini uses model_name
                        api_key=correction_api_key,
                        # Gemini doesn't typically use api_base in the same way
                        # temperature=0.1 # Set temperature if needed
                    )
                    logger.info(f"Initialized LlamaIndex LLM '{correction_llm_model}' for correction step.")
                else:
                    logger.warning("LLM Model or API Key missing for correction step. Correction disabled.")
                    llm_for_correction = None
            except ImportError:
                logger.warning("LlamaIndex LLM import failed (e.g., 'pip install llama-index-llms-gemini'). Correction disabled.")
                llm_for_correction = None
            except Exception as llm_init_e:
                 logger.error(f"Failed to initialize LLM for correction: {llm_init_e}", exc_info=True)
                 llm_for_correction = None
        else:
            logger.warning("Correction step dependencies not met. Correction disabled.")
            llm_for_correction = None

    except Exception as e:
        print(f"ERROR loading configuration or initializing correction LLM: {e}")
        logger.exception("Fatal error during setup.")
        sys.exit(1)

    # --- Initialize and Run QA Engine ---
    print("\n--- Initializing GraphRAG QA Engine (with Retry & Self-Learning) ---")
    qa_engine_instance = None
    try:
        qa_engine_instance = GraphRAGQA(
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
            db_name=config_data['DB_NAME'], # <<< Pass DB Name >>>
            llm_config_extra=config_data.get('LLM_EXTRA_PARAMS', {}),
            max_cypher_retries=1
        )

        if not qa_engine_instance.is_ready():
             print("\nFATAL: QA Engine failed to initialize correctly. Exiting.")
             sys.exit(1)

        print("QA engine ready.")
        print("\n🔍 Ask questions (type 'exit' or 'quit' to stop):")

        async def run_question(question_text):
            """Helper async function to call answer_question and print results."""
            print("Processing...")
            response_dict = await qa_engine_instance.answer_question(question_text)
            print(f"\n💡 Answer:\n{response_dict.get('answer', 'N/A')}\n")
            print(f"--- Cypher Query Used ---")
            print(f"{response_dict.get('cypher_query', 'N/A')}")
            if response_dict.get('error_info'): print(f"Error Info: {response_dict['error_info']}")
            if response_dict.get('info'): print(f"Info: {response_dict['info']}")
            print(f"-----------------------\n")
            if response_dict.get("sources"):
                print("--- Vector Sources ---")
                for i, src in enumerate(response_dict["sources"]):
                    src_doc = src.get("metadata", {}).get("source_document", "Unknown")
                    print(f"[{i+1}] From: {src_doc}")
                    print(f"   Text: {src.get('text', '')[:150]}...")
                print("--------------------\n")

        while True:
            try:
                question = input("❓ Your Question: ").strip()
                if not question: continue
                if question.lower() in {"exit", "quit"}: break
                asyncio.run(run_question(question))
            except EOFError: break
            except KeyboardInterrupt: break
        print("\nExiting interactive session.")

    except ConnectionError as e:
         print(f"\nFATAL: Could not initialize QA Engine (Neo4j Connection Failed): {e}")
         sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred during QA engine operation.")
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if qa_engine_instance:
            try:
                logger.info("Attempting to close QA engine resources...")
                # vvv Use asyncio.run() to execute the async close method vvv
                asyncio.run(qa_engine_instance.close())
                logger.info("QA engine resources closed successfully via asyncio.run.")
            except RuntimeError as re:
                # Handle cases where asyncio.run() might complain about loops
                if "cannot run event loop while another loop is running" in str(re):
                    logger.warning(
                        f"Could not close QA engine via asyncio.run due to existing loop: {re}. Manual resource cleanup might be needed.")
                elif "no running event loop" in str(re):
                    logger.warning(f"Could not close QA engine via asyncio.run as no loop was running: {re}.")
                else:
                    logger.error(f"RuntimeError closing QA engine: {re}", exc_info=True)
            except Exception as close_e:
                logger.error(f"Error closing QA engine: {close_e}", exc_info=True)

    print("\n--- GraphRAGQA Script End ---")

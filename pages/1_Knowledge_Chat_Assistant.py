# pages/1_Knowledge_Chat_Assistant.py

import streamlit as st
import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import time # Keep if used

st.set_page_config(page_title="Knowledge Chat Assistant")

# Logger setup
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers(): # Basic config if needed
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')

# --- Core Imports from GraphRAG_Document_AI_Platform.py ---
# Ensure these are accessible (e.g., GraphRAG_Document_AI_Platform.py in root)
try:
    from GraphRAG_Document_AI_Platform import (
        load_config,            # <<< IMPORT the function
        get_correction_llm,
        load_qa_engine,
        get_chroma_collection,  # Needed for DB check
        get_embedding_model     # Needed for get_chroma_collection
        # Add get_requests_session if QA engine needs it internally? Usually not directly needed here.
    )
    # Import the QA engine class directly for type hints if desired
    from graph_rag_qa import GraphRAGQA
    # Import Neo4j driver for DB count check (sync driver is fine for this)
    import neo4j
except ImportError as e:
    st.error(f"Error importing project modules in Chat Assistant page: {e}. Ensure GraphRAG_Document_AI_Platform.py and graph_rag_qa.py are accessible.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during imports: {e}")
    logger.error("Import error on Chat Page", exc_info=True)
    st.stop()

# --- Constants ---
# Define CHAT_HISTORY_FILE here as it's specific to this page's logic
CHAT_HISTORY_FILE = Path("./chat_history.json")

# --- Helper Functions for Chat History ---
def load_chat_history() -> List[Dict]:
    """Loads chat history from the JSON file."""
    if CHAT_HISTORY_FILE.is_file():
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            # Basic validation
            if isinstance(history, list):
                logger.info(f"Loaded {len(history)} messages from {CHAT_HISTORY_FILE}")
                return history
            else:
                logger.warning(f"Chat history file {CHAT_HISTORY_FILE} has invalid format (expected list). Starting fresh.")
                return []
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.error(f"Error loading chat history file {CHAT_HISTORY_FILE}: {e}", exc_info=True)
            return [] # Return empty list on error
    return [] # Return empty list if file doesn't exist

def save_chat_history(messages: List[Dict]):
    """Saves the current chat history to the JSON file."""
    try:
        # Ensure basic serializability (complex objects in 'sources' could fail)
        # No need to recreate list if session state is managed correctly
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)
        logger.debug(f"Saved {len(messages)} messages to {CHAT_HISTORY_FILE}")
    except TypeError as te:
        logger.error(f"Error serializing chat history (potential complex object): {te}", exc_info=True)
    except OSError as e:
        logger.error(f"Error saving chat history file {CHAT_HISTORY_FILE}: {e}", exc_info=True)
    except Exception as e:
         logger.error(f"Unexpected error saving chat history: {e}", exc_info=True)

# --- Streamlit Page Logic ---

st.title("💬 Chat Assistant")
st.markdown("Ask questions about the documents processed via the 'Data Ingestion' page.")

# --- Load Config and Initialize QA Engine ---
# Use functions imported from GraphRAG_Document_AI_Platform to access cached resources
qa_engine: Optional[GraphRAGQA] = None # Initialize
is_engine_ready_for_chat = False # Initialize

try:
    config = load_config() # <<< CALL the imported function
    if not config or not config.get('_CONFIG_VALID'):
        st.error("App configuration is invalid. Cannot initialize Q&A engine.")
        st.stop()

    correction_llm = get_correction_llm(config) # <<< CALL imported function
    qa_engine = load_qa_engine(config, correction_llm) # <<< CALL imported function

    is_engine_ready_for_chat = qa_engine and qa_engine.is_ready()
    if not is_engine_ready_for_chat:
        st.warning("Q&A Engine is not ready. Check backend connections (Neo4j, LLM API key).", icon="⚠️")

except Exception as e:
    logger.error(f"Error initializing resources for Chat page: {e}", exc_info=True)
    st.error(f"Failed to initialize necessary resources for chat: {e}")
    # is_engine_ready_for_chat remains False

# --- Initialize Chat History (now done safely after config/engine load attempt) ---
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# --- Display Context Message ---
if is_engine_ready_for_chat:
    neo4j_count = 0
    chroma_count = 0
    driver = None
    try:
        # Check Neo4j count
        if config.get('NEO4J_URI') and config.get('NEO4J_USER') and config.get('NEO4J_PASSWORD'):
            try:
                driver = neo4j.GraphDatabase.driver(config['NEO4J_URI'], auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD']))
                with driver.session(database=config.get('DB_NAME', 'neo4j')) as session:
                    result = session.run("MATCH (n:Entity) RETURN count(n) as count") # Count Entities
                    record = result.single()
                    neo4j_count = record["count"] if record else 0
                logger.debug(f"Checked Neo4j count: {neo4j_count}")
            except Exception as neo4j_e:
                logger.warning(f"Could not connect to Neo4j to check node count: {neo4j_e}")
        # Check Chroma count (using cached resource getter)
        chroma_collection = get_chroma_collection(
            config.get('CHROMA_PERSIST_PATH'),
            config.get('COLLECTION_NAME'),
            config.get('EMBEDDING_MODEL') # Pass embedding model name from config
        )
        if chroma_collection:
            try:
                 chroma_count = chroma_collection.count()
                 logger.debug(f"Checked ChromaDB count: {chroma_count}")
            except Exception as chroma_e:
                 logger.warning(f"Could not get ChromaDB collection count: {chroma_e}")

        if neo4j_count > 0 or chroma_count > 0:
            st.info(f"**Context:** Q&A based on data in Neo4j ({neo4j_count} entities) and ChromaDB ({chroma_count} vectors). Process more documents via the 'Data Ingestion' page.", icon="ℹ️")
        else:
            st.info("No data found in Neo4j or ChromaDB. Use the 'Data Ingestion' page to process documents first.", icon="ℹ️")

    except Exception as db_check_e:
        logger.warning(f"Could not check DB counts: {db_check_e}")
        st.info("Use the 'Data Ingestion' page to process documents.", icon="ℹ️")
    finally:
        if driver: driver.close()
else:
    st.info("Q&A Engine initializing or not ready...")

# --- Display Chat History ---
# Loop to display messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # --- REMOVED Expanders for sources/cypher from history display ---

chat_placeholder = "Ask a question about the processed documents..."
if not is_engine_ready_for_chat:
    chat_placeholder = "Q&A Engine not ready..."

# --- Section: Display Info for Last Response ---
st.divider() # Divider before the info section

with st.container(border=True): # Set border=True for a visual box
    st.caption("Details for Last Assistant Response:") # Add a small header

col_cypher, col_sources = st.columns(2) # Create two columns

# Get the very last message from history, if it exists and is from assistant
last_message = st.session_state.messages[-1] if st.session_state.messages else None

with col_cypher:
    # Check if the last message exists and is from the assistant
    if last_message and last_message["role"] == "assistant":
        # Get info directly from the last message dictionary
        last_cypher = last_message.get("cypher_query")
        last_error = last_message.get("error_info")
        last_info = last_message.get("info")

        # Display Cypher
        if last_cypher and last_cypher != "N/A":
            with st.expander("Show Cypher Query Used (Last Response)"):
                st.code(last_cypher, language="cypher")
        # Display Error or Info
        if last_error:
             st.warning(f"Query Info: {last_error}", icon="⚠️")
        elif last_info:
             st.caption(f"Query Info: {last_info}")

with col_sources:
     # Check if the last message exists and is from the assistant
    if last_message and last_message["role"] == "assistant":
        # Get sources directly from the last message dictionary
        last_sources = last_message.get("sources")
        if last_sources and isinstance(last_sources, list):
            with st.expander(f"Show Sources ({len(last_sources)}) (Last Response)"):
                # Keep the existing loop to display sources
                for i, source in enumerate(last_sources):
                    source_doc = "Unknown Document"
                    source_text = ""
                    metadata = {}
                    if isinstance(source, dict):
                        source_text = source.get("text", "[No text found]")
                        metadata = source.get("metadata", {})
                        source_doc = metadata.get("source_document", source_doc)
                    elif isinstance(source, str):
                        source_text = source
                    st.markdown(f"**Source {i+1} ({source_doc}):**\n> {source_text.replace('\n', '\n> ')}\n---")
# --- End Section ---

# --- Chat Input ---
if prompt := st.chat_input(chat_placeholder, disabled=(not is_engine_ready_for_chat), key="chat_input"):
    # --- REMOVED Clearing of last_response state ---

    # Append user message and save
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history(st.session_state.messages)
    # Streamlit displays user message automatically

    # Handle generic greetings or call QA engine
    normalized_prompt = prompt.strip().lower();
    generic_greetings = ["hi", "hello", "hey", "how are you", "how are you?", "what's up", "whats up", "sup", "good morning", "good afternoon", "good evening", "thanks", "thank you"] # Example list

    if normalized_prompt in generic_greetings:
        # --- Handle Greetings ---
        if "how are you" in normalized_prompt: response_content = "Ready to help! Ask about your documents."
        elif "thank" in normalized_prompt: response_content = "You're welcome!"
        else: response_content = "Hello! How can I help?"
        logger.info(f"Handling generic input: '{prompt}'")
        assistant_message = {"role": "assistant", "content": response_content}
        st.session_state.messages.append(assistant_message)
        save_chat_history(st.session_state.messages)
        st.rerun() # Rerun to show simple response

    else:
        # --- Call Q&A Engine ---
        response_dict = None # Initialize
        # Display thinking message in the chat history area
        with st.chat_message("assistant"):
             message_placeholder = st.empty() # Create placeholder
             message_placeholder.markdown("Thinking...") # Set thinking message

        # --- Correctly Indented Try/Except for QA Call ---
        try:
            start_time = time.time()
            logger.info(f"Running QA engine for prompt: {prompt}")
            # Call the QA engine (assumed to be synchronous now)
            response_dict = qa_engine.answer_question(prompt)
            duration = time.time() - start_time
            logger.info(f"QA engine finished in {duration:.2f}s")
            # Update the placeholder with the actual answer *after* the call returns
            message_placeholder.markdown(response_dict.get("answer", "Sorry, I could not generate an answer."))

        except Exception as e:
            logger.exception("Error running answer_question from Streamlit")
            error_message = f"An error occurred processing your question: {e}"
            message_placeholder.error(error_message) # Show error in placeholder
            # Prepare response dict for history even on error
            response_dict = {"answer": error_message, "sources": None, "cypher_query": None, "error_info": str(e)}
        # --- End Try/Except for QA Call ---

        # --- Prepare full assistant message for history AFTER try/except ---
        assistant_message = {"role": "assistant", "content": response_dict.get("answer", "No answer generated.")}
        # Add other keys if they exist in the response_dict
        for key in ["sources", "cypher_query", "error_info", "info"]:
            if key in response_dict and response_dict[key] is not None:
                assistant_message[key] = response_dict[key]

        # Append full message to history and save
        st.session_state.messages.append(assistant_message)
        save_chat_history(st.session_state.messages)

        # --- REMOVED Setting of last_response state ---

        # --- KEEP st.rerun() ---
        st.rerun() # Rerun to update the "Last Response Info" section above input box
        # --- End Keep ---
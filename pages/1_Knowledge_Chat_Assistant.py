# pages/1_Knowledge_Chat_Assistant.py - UPDATED WITH SYNC DATABASE INTEGRATION

import streamlit as st
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime

# Enhanced page configuration
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep the same CSS styling from the original file
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Modern color palette */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-chat: #ffffff;
        --border-light: #e2e8f0;
        --border-hover: #cbd5e1;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --radius-lg: 12px;
        --radius-xl: 16px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Base styling */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-secondary);
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Compact header - much smaller */
    .compact-header {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        text-align: center;
    }

    .compact-header h1 {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    .compact-header p {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
        font-weight: 400;
    }

    /* Enhanced chat messages - reduced spacing */
    .stChatMessage {
        background: var(--bg-chat) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-xl) !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }

    .stChatMessage:hover {
        border-color: var(--border-hover) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* User messages */
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border-color: #bfdbfe !important;
        margin: 0 0 0.5rem 0 !important;
        border-radius: 12px 12px 4px 12px !important;
    }

    /* Assistant messages */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: var(--bg-chat) !important;
        border-color: var(--border-light) !important;
        margin: 0 !important;
        border-radius: 4px 12px 12px 12px !important;
    }

    /* Better buttons */
    .stButton > button {
        background: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-lg) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stButton > button:hover {
        background: #2563eb !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .status-online {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-green);
    }

    .status-offline {
        background: rgba(239, 68, 68, 0.1);
        color: var(--accent-red);
    }

    /* Compact metrics */
    .compact-metric {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .compact-metric:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-sm);
    }

    .compact-metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .compact-metric-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    /* Performance indicators */
    .performance-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: var(--radius-lg);
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }

    .performance-good {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-green);
    }

    .performance-medium {
        background: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
    }

    .performance-slow {
        background: rgba(239, 68, 68, 0.1);
        color: var(--accent-red);
    }

    /* Conversation sidebar */
    .conversation-sidebar {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1rem;
        margin-bottom: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }

    .conversation-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 0.75rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .conversation-item:hover {
        border-color: var(--accent-blue);
        box-shadow: var(--shadow-sm);
    }

    .conversation-item.active {
        border-color: var(--accent-blue);
        background: rgba(59, 130, 246, 0.05);
    }

    .conversation-title {
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        font-size: 0.875rem;
    }

    .conversation-preview {
        color: var(--text-muted);
        font-size: 0.75rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .conversation-meta {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    /* Fade animations */
    .fade-in {
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stChatMessage {
            margin: 0.25rem 0 !important;
        }
        .compact-header {
            padding: 0.75rem 1rem;
        }
        .compact-header h1 {
            font-size: 1.25rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Logger setup
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s')

# Core Imports with error handling
try:
    from GraphRAG_Document_AI_Platform import (
        load_config,
        get_correction_llm,
        load_qa_engine
    )
    from graph_rag_qa import GraphRAGQA
    import neo4j

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"❌ Error importing project modules: {e}")
    IMPORTS_SUCCESSFUL = False

# Sync Database Integration Imports
try:
    # Import the NEW sync database system
    from src.chat.sync_database import create_sync_chat_service

    DATABASE_INTEGRATION_AVAILABLE = True
    logger.info("✅ Sync database chat integration available")
except ImportError as e:
    DATABASE_INTEGRATION_AVAILABLE = False
    logger.warning(f"⚠️ Sync database chat integration not available: {e}")

# Constants
CHAT_HISTORY_FILE = Path("./chat_history.json")


# Helper Functions
def load_chat_history() -> List[Dict]:
    """Load chat history with error handling - JSON fallback."""
    if CHAT_HISTORY_FILE.is_file():
        try:
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
            if isinstance(history, list):
                logger.info(f"📚 Loaded {len(history)} messages from JSON chat history")
                return history
            else:
                logger.warning("🔄 Invalid chat history format, starting fresh")
                return []
        except Exception as e:
            logger.error(f"💥 Error loading chat history: {e}")
            return []
    return []


def save_chat_history(messages: List[Dict]):
    """Save chat history with error handling - JSON fallback."""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)
        logger.debug(f"💾 Saved {len(messages)} messages to JSON chat history")
    except Exception as e:
        logger.error(f"💥 Error saving chat history: {e}")


def get_performance_indicator(duration: float) -> tuple:
    """Get performance indicator based on response time."""
    if duration < 2.0:
        return "⚡", "Excellent", "performance-good"
    elif duration < 5.0:
        return "⏱️", "Good", "performance-medium"
    else:
        return "⏳", "Slow", "performance-slow"


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y at %H:%M")
    except:
        return timestamp


def initialize_sync_chat_service(config):
    """Initialize the SYNC chat service with database or fallback to JSON."""
    try:
        if DATABASE_INTEGRATION_AVAILABLE and config:
            # Try to initialize SYNC database chat service
            chat_service = create_sync_chat_service(config)
            logger.info("✅ Sync database chat service initialized")
            return chat_service, True
        else:
            logger.info("📄 Using JSON file chat system")
            return None, False
    except Exception as e:
        logger.error(f"❌ Failed to initialize sync database chat service: {e}")
        return None, False


def load_messages_from_service(chat_service, use_database):
    """Load messages from database or JSON file."""
    try:
        if use_database and chat_service:
            # Load from SYNC database
            messages = chat_service.load_chat_history()
            logger.info(f"📚 Loaded {len(messages)} messages from sync database")
            return messages
        else:
            # Load from JSON file
            return load_chat_history()
    except Exception as e:
        logger.error(f"❌ Error loading messages: {e}")
        return load_chat_history()  # Fallback to JSON


def save_messages_to_service(messages, chat_service, use_database):
    """Save messages to database or JSON file."""
    try:
        if use_database and chat_service:
            # Save to SYNC database
            success = chat_service.save_chat_history(messages)
            if success:
                logger.debug(f"💾 Saved {len(messages)} messages to sync database")
            else:
                logger.warning("⚠️ Database save failed, falling back to JSON")
                save_chat_history(messages)
        else:
            # Save to JSON file
            save_chat_history(messages)
    except Exception as e:
        logger.error(f"❌ Error saving messages: {e}")
        # Fallback to JSON
        save_chat_history(messages)


def add_message_to_service(message, chat_service, use_database):
    """Add single message to database or JSON file."""
    try:
        if use_database and chat_service:
            # Add to SYNC database
            result = chat_service.add_message(message)
            if result:
                logger.debug("➕ Added message to sync database")
            else:
                logger.warning("⚠️ Database add failed")
        else:
            # Add to session state (will be saved with full history)
            pass
    except Exception as e:
        logger.error(f"❌ Error adding message: {e}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Compact header
st.markdown("""
<div class="compact-header fade-in">
    <h1>🔍 Knowledge Assistant</h1>
    <p>AI-powered document analysis with persistent chat history</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Enhanced with Database Status
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ System Control")

    # Initialize system components
    qa_engine: Optional[GraphRAGQA] = None
    is_engine_ready = False
    config = None
    neo4j_count = 0
    chat_service = None
    use_database = False

    if IMPORTS_SUCCESSFUL:
        # System initialization
        with st.spinner("🔄 Initializing systems..."):
            try:
                # Load configuration
                config = load_config()
                if config and config.get('_CONFIG_VALID'):
                    # Initialize QA engine
                    correction_llm = get_correction_llm(config)
                    qa_engine = load_qa_engine(config, correction_llm)
                    is_engine_ready = qa_engine and qa_engine.is_ready()

                    # Initialize SYNC chat service
                    chat_service, use_database = initialize_sync_chat_service(config)

                    if is_engine_ready:
                        st.success("✅ AI Engine Ready")
                    else:
                        st.error("❌ AI Engine Offline")
                else:
                    st.error("❌ Configuration Invalid")

            except Exception as e:
                logger.error(f"System initialization error: {e}")
                st.error(f"❌ Initialization failed: {e}")

        # Enhanced system status
        st.markdown("### 📊 System Status")

        col1, col2 = st.columns(2)

        with col1:
            # AI Engine Status
            if is_engine_ready:
                st.markdown("**AI Engine**")
                st.markdown('<div class="status-indicator status-online">● Online</div>', unsafe_allow_html=True)
            else:
                st.markdown("**AI Engine**")
                st.markdown('<div class="status-indicator status-offline">● Offline</div>', unsafe_allow_html=True)

        with col2:
            # Database Status
            if use_database:
                st.markdown("**Database**")
                st.markdown('<div class="status-indicator status-online">● Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown("**Storage**")
                st.markdown('<div class="status-indicator status-offline">● JSON File</div>', unsafe_allow_html=True)

        # Configuration status
        if config and config.get('_CONFIG_VALID'):
            st.markdown("**Configuration:** ✅ Valid")
        else:
            st.markdown("**Configuration:** ❌ Invalid")

        # Database connection indicator
        if DATABASE_INTEGRATION_AVAILABLE:
            st.markdown("**Chat System:** 🗄️ Database-Enhanced (Sync)")
        else:
            st.markdown("**Chat System:** 📄 File-Based")

        # Knowledge base metrics
        if is_engine_ready and config:
            st.markdown("### 📊 Knowledge Base")

            # Check Neo4j
            try:
                driver = neo4j.GraphDatabase.driver(
                    config['NEO4J_URI'],
                    auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD'])
                )
                with driver.session(database=config.get('DB_NAME', 'neo4j')) as session:
                    result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                    record = result.single()
                    neo4j_count = record["count"] if record else 0
                driver.close()
            except Exception as e:
                logger.warning(f"Neo4j check failed: {e}")

            # Compact metrics display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="compact-metric">
                    <div class="compact-metric-value">{neo4j_count:,}</div>
                    <div class="compact-metric-label">Entities</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="compact-metric">
                    <div class="compact-metric-value">📊</div>
                    <div class="compact-metric-label">GraphRAG</div>
                </div>
                """, unsafe_allow_html=True)

            # Health indicator
            if neo4j_count > 0:
                st.success("✅ Knowledge base operational")
            else:
                st.error("❌ No data found")

    else:
        st.error("❌ Core modules not available")

    # Initialize chat history with new SYNC system
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages_from_service(chat_service, use_database)

    if "chat_service" not in st.session_state:
        st.session_state.chat_service = chat_service

    if "use_database" not in st.session_state:
        st.session_state.use_database = use_database

    # Session analytics
    if st.session_state.messages:
        st.markdown("### 📈 Session Analytics")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])

        # Calculate average response time
        response_times = [m.get("response_time", 0) for m in st.session_state.messages if
                          m.get("role") == "assistant" and m.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="compact-metric">
                <div class="compact-metric-value">{total_messages}</div>
                <div class="compact-metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="compact-metric">
                <div class="compact-metric-value">{user_messages}</div>
                <div class="compact-metric-label">Queries</div>
            </div>
            """, unsafe_allow_html=True)

        if avg_response_time > 0:
            st.markdown(f"""
            <div class="compact-metric">
                <div class="compact-metric-value">{avg_response_time:.1f}s</div>
                <div class="compact-metric-label">Avg Response</div>
            </div>
            """, unsafe_allow_html=True)

    # Enhanced controls
    st.markdown("### 🔧 Controls")

    # Clear history with database awareness
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        save_messages_to_service([], st.session_state.chat_service, st.session_state.use_database)
        if st.session_state.use_database:
            st.success("✅ Sync database chat history cleared!")
        else:
            st.success("✅ JSON chat history cleared!")
        st.rerun()

    # Refresh system
    if st.button("🔄 Refresh System", use_container_width=True):
        # Clear caches
        st.cache_data.clear()

        # Reset session state
        for key in ['messages', 'chat_service', 'use_database']:
            if key in st.session_state:
                del st.session_state[key]

        st.success("🔄 System refreshed!")
        st.rerun()

    # Database migration (if available)
    if DATABASE_INTEGRATION_AVAILABLE and not st.session_state.use_database:
        if st.button("🔄 Migrate to Database", use_container_width=True):
            try:
                # Try to initialize sync database
                if config:
                    chat_service_new, use_database_new = initialize_sync_chat_service(config)
                    if use_database_new and chat_service_new:
                        # Migrate existing messages
                        if st.session_state.messages:
                            success = chat_service_new.save_chat_history(st.session_state.messages)
                            if success:
                                st.session_state.chat_service = chat_service_new
                                st.session_state.use_database = use_database_new
                                st.success("✅ Migrated to sync database successfully!")
                            else:
                                st.error("❌ Database migration failed")
                        else:
                            st.session_state.chat_service = chat_service_new
                            st.session_state.use_database = use_database_new
                            st.success("✅ Sync database chat system activated!")
                        st.rerun()
                    else:
                        st.error("❌ Sync database migration failed")
                else:
                    st.error("❌ Configuration not available")
            except Exception as e:
                st.error(f"❌ Migration error: {e}")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Performance indicator for last response
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg.get("role") == "assistant" and "response_time" in last_msg:
        duration = last_msg["response_time"]
        emoji, status, css_class = get_performance_indicator(duration)

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div class="performance-indicator {css_class}">
                {emoji} {status} ({duration:.1f}s)
            </div>
        </div>
        """, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Enhanced chat input with database awareness
if IMPORTS_SUCCESSFUL and is_engine_ready:
    if neo4j_count > 0:
        placeholder = "💼 Ask me about your business documents and data..."
    else:
        placeholder = "📋 Please process documents first, then start asking questions..."
else:
    placeholder = "🔧 System initializing - please wait..."

# Storage indicator in placeholder
if st.session_state.use_database:
    placeholder += " [💾 Database]"
else:
    placeholder += " [📄 File]"

# Main chat input
if prompt := st.chat_input(placeholder, disabled=not (IMPORTS_SUCCESSFUL and is_engine_ready)):
    # Add user message
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_message)

    # Save to appropriate storage
    add_message_to_service(user_message, st.session_state.chat_service, st.session_state.use_database)

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Handle greetings
    normalized_prompt = prompt.strip().lower()
    greetings = {
        "hi": "👋 Hello! I'm your Knowledge Assistant, ready to help you analyze your business documents.",
        "hello": "👋 Hello! How may I assist you with your document analysis needs today?",
        "how are you": "🤖 All systems operational and ready to serve your knowledge requirements.",
        "thank you": "🙏 You're welcome! I'm here whenever you need business intelligence insights.",
        "thanks": "😊 My pleasure! Feel free to ask about any documents in your knowledge base."
    }

    if normalized_prompt in greetings:
        response_content = greetings[normalized_prompt]

        with st.chat_message("assistant"):
            st.write(response_content)

        assistant_message = {
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.now().isoformat(),
            "response_time": 0.1
        }
        st.session_state.messages.append(assistant_message)

        # Save to appropriate storage
        add_message_to_service(assistant_message, st.session_state.chat_service, st.session_state.use_database)
        save_messages_to_service(st.session_state.messages, st.session_state.chat_service,
                                 st.session_state.use_database)

        st.rerun()

    else:
        # Process with QA engine
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing your query..."):
                start_time = time.time()

                try:
                    # Call QA engine
                    response_dict = qa_engine.answer_question(prompt)
                    duration = time.time() - start_time

                    # Display response
                    answer = response_dict.get("answer", "❌ I couldn't generate a comprehensive answer.")
                    st.write(answer)

                    # Prepare assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().isoformat(),
                        "response_time": duration
                    }

                    # Add metadata
                    for key in ["sources", "cypher_query", "error_info", "info", "linked_entities"]:
                        if key in response_dict and response_dict[key] is not None:
                            assistant_message[key] = response_dict[key]

                    # Save to session and storage
                    st.session_state.messages.append(assistant_message)
                    add_message_to_service(assistant_message, st.session_state.chat_service,
                                           st.session_state.use_database)
                    save_messages_to_service(st.session_state.messages, st.session_state.chat_service,
                                             st.session_state.use_database)

                    # Success feedback
                    emoji, status, _ = get_performance_indicator(duration)
                    storage_type = "sync database" if st.session_state.use_database else "file"
                    st.success(
                        f"{emoji} Analysis completed in {duration:.2f}s - {status} performance | 💾 Saved to {storage_type}")

                except Exception as e:
                    duration = time.time() - start_time
                    error_message = f"❌ **System Error:** Unable to process your request. Please try again."

                    st.error(error_message)
                    st.caption(f"Technical details: {str(e)}")

                    assistant_message = {
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().isoformat(),
                        "response_time": duration,
                        "error_info": str(e)
                    }

                    # Save error message too
                    st.session_state.messages.append(assistant_message)
                    add_message_to_service(assistant_message, st.session_state.chat_service,
                                           st.session_state.use_database)
                    save_messages_to_service(st.session_state.messages, st.session_state.chat_service,
                                             st.session_state.use_database)

                    logger.exception("Error in QA processing")

        st.rerun()

# ============================================================================
# FOOTER - System Information
# ============================================================================

# Show system information at the bottom
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.use_database:
        st.markdown("💾 **Storage:** Sync Database")
    else:
        st.markdown("📄 **Storage:** JSON File")

with col2:
    if DATABASE_INTEGRATION_AVAILABLE:
        st.markdown("🔧 **System:** Sync Database-Ready")
    else:
        st.markdown("🔧 **System:** File-Based")

with col3:
    message_count = len(st.session_state.messages)
    st.markdown(f"📊 **Messages:** {message_count}")

# Debug information (only in development)
try:
    debug_mode = st.secrets.get("debug_mode", False)
except:
    debug_mode = False

if debug_mode:
    with st.expander("🔍 Debug Information"):
        st.json({
            "database_integration_available": DATABASE_INTEGRATION_AVAILABLE,
            "use_database": st.session_state.use_database,
            "chat_service_active": st.session_state.chat_service is not None,
            "imports_successful": IMPORTS_SUCCESSFUL,
            "engine_ready": is_engine_ready,
            "neo4j_entities": neo4j_count,
            "message_count": len(st.session_state.messages)
        })
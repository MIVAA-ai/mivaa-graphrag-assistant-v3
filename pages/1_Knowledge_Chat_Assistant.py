# pages/1_Knowledge_Chat_Assistant.py - COMPLETE FIXED ENHANCED UI VERSION

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

# Enhanced CSS styling with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --bg-accent: #eff6ff;
        --border-light: #e2e8f0;
        --border-medium: #cbd5e1;
        --border-accent: #bfdbfe;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-blue-dark: #2563eb;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --accent-red: #ef4444;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
        --radius-full: 9999px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-secondary);
        color: var(--text-primary);
        line-height: 1.6;
    }

    /* Enhanced Header */
    .enhanced-header {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-xl);
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }

    .enhanced-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-green) 50%, var(--accent-orange) 100%);
    }

    .enhanced-header h1 {
        color: var(--text-primary);
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }

    .enhanced-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }

    /* Enhanced Sidebar Sections */
    .sidebar-header {
        background: linear-gradient(135deg, var(--bg-accent) 0%, var(--bg-primary) 100%);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-green) 100%);
    }

    .sidebar-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }

    .sidebar-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin: 0;
    }

    .enhanced-section {
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .enhanced-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: transparent;
        transition: all 0.2s ease;
    }

    .enhanced-section:hover {
        border-color: var(--border-medium);
        box-shadow: var(--shadow-sm);
        transform: translateX(2px);
    }

    .enhanced-section:hover::before {
        background: var(--accent-blue);
    }

    .section-title-enhanced {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    /* Enhanced User Container */
    .enhanced-user-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: linear-gradient(135deg, var(--bg-accent) 0%, #dbeafe 100%);
        border: 1px solid var(--border-accent);
        border-radius: var(--radius-full);
        transition: all 0.2s ease;
    }

    .enhanced-user-container:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }

    .user-avatar {
        width: 32px;
        height: 32px;
        background: var(--accent-blue);
        border-radius: var(--radius-full);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.875rem;
    }

    .user-info {
        flex: 1;
    }

    .user-name {
        font-weight: 600;
        color: var(--accent-blue);
        font-size: 0.875rem;
        margin: 0;
    }

    .user-status {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Enhanced Search Input */
    .enhanced-search-container {
        position: relative;
        margin-bottom: 0.75rem;
    }

    .enhanced-search-container .search-icon {
        position: absolute;
        left: 0.75rem;
        top: 50%;
        transform: translateY(-50%);
        color: var(--text-muted);
        font-size: 1rem;
        z-index: 1;
        pointer-events: none;
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: var(--accent-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-lg) !important;
        padding: 0.875rem 1.25rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: var(--accent-blue-dark) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Enhanced Conversation Items */
    .enhanced-conversation-item {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1rem;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .enhanced-conversation-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
        background: transparent;
        transition: all 0.2s ease;
    }

    .enhanced-conversation-item:hover {
        border-color: var(--accent-blue);
        transform: translateX(4px);
        box-shadow: var(--shadow-md);
    }

    .enhanced-conversation-item:hover::before {
        background: var(--accent-blue);
    }

    .enhanced-conversation-item.active {
        background: linear-gradient(135deg, var(--bg-accent) 0%, #dbeafe 100%);
        border-color: var(--accent-blue);
        transform: translateX(4px);
    }

    .enhanced-conversation-item.active::before {
        background: var(--accent-blue);
    }

    .conversation-title-enhanced {
        font-weight: 600;
        font-size: 0.875rem;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .conversation-meta-enhanced {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .conversation-preview {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
        line-height: 1.4;
        opacity: 0.8;
    }

    /* Enhanced Metrics */
    .enhanced-metrics-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .enhanced-metric-card {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .enhanced-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-green) 100%);
    }

    .enhanced-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .enhanced-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }

    .enhanced-metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.025em;
        font-weight: 500;
    }

    .enhanced-status-indicator {
        margin-top: 1rem;
        padding: 0.75rem;
        border-radius: var(--radius-lg);
        font-size: 0.875rem;
        font-weight: 500;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-green);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--accent-orange);
    }

    /* Enhanced Chat Messages */
    .stChatMessage {
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-xl) !important;
        padding: 1.25rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }

    .stChatMessage:hover {
        border-color: var(--border-medium) !important;
        box-shadow: var(--shadow-md) !important;
    }

    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, var(--bg-accent) 0%, #dbeafe 100%) !important;
        border-color: var(--border-accent) !important;
        border-radius: var(--radius-xl) var(--radius-xl) 4px var(--radius-xl) !important;
    }

    .stChatMessage[data-testid="chat-message-assistant"] {
        background: var(--bg-primary) !important;
        border-color: var(--border-light) !important;
        border-radius: 4px var(--radius-xl) var(--radius-xl) var(--radius-xl) !important;
    }

    /* Enhanced Performance Indicator */
    .enhanced-performance-indicator {
        margin: 1rem 0;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: var(--radius-lg);
        color: var(--accent-green);
        font-size: 0.875rem;
        font-weight: 600;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }

    .enhanced-performance-indicator.medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border-color: rgba(245, 158, 11, 0.2);
        color: var(--accent-orange);
    }

    .enhanced-performance-indicator.slow {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
    }

    /* Enhanced Footer */
    .enhanced-footer {
        padding: 1.5rem;
        border-top: 1px solid var(--border-light);
        background: var(--bg-primary);
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 1rem;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }

    .enhanced-footer-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fadeInUp 0.3s ease-out;
    }

    /* Text Input Enhancements */
    .stTextInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-lg) !important;
        padding: 0.875rem 1rem 0.875rem 2.5rem !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        background: var(--bg-primary) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* Chat Input Enhancement */
    .stChatInput > div {
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1), var(--shadow-md) !important;
    }
</style>
""", unsafe_allow_html=True)

# Logger setup with proper module name
logger = logging.getLogger("knowledge_chat_assistant")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
    )

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
    logger.info("✅ Core imports successful")
except ImportError as e:
    st.error(f"❌ Error importing project modules: {e}")
    IMPORTS_SUCCESSFUL = False
    logger.error(f"❌ Core imports failed: {e}")

# Database Integration Imports
try:
    from src.chat.sync_database import create_sync_chat_service

    DATABASE_INTEGRATION_AVAILABLE = True
    logger.info("✅ Database integration available")
except ImportError as e:
    DATABASE_INTEGRATION_AVAILABLE = False
    logger.warning(f"Database integration not available: {e}")

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
                logger.info(f"📚 Loaded {len(history)} messages from chat history")
                return history
            else:
                logger.warning("🔄 Invalid chat history format, starting fresh")
                return []
        except Exception as e:
            logger.error(f"💥 Error loading chat history: {e}")
            return []
    return []


def save_chat_history(messages: List[Dict]):
    """Save chat history with error handling."""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)
        logger.debug(f"💾 Saved {len(messages)} messages to chat history")
    except Exception as e:
        logger.error(f"💥 Error saving chat history: {e}")


def get_performance_indicator(duration: float) -> tuple:
    """Get performance indicator based on response time."""
    if duration < 2.0:
        return "⚡", "Excellent", "enhanced-performance-indicator"
    elif duration < 5.0:
        return "⏱️", "Good", "enhanced-performance-indicator medium"
    else:
        return "⏳", "Slow", "enhanced-performance-indicator slow"


def initialize_chat_service(config):
    """Initialize chat service - database with JSON fallback."""
    try:
        if DATABASE_INTEGRATION_AVAILABLE and config:
            chat_service = create_sync_chat_service(config)
            logger.info("✅ Database chat service initialized")
            return chat_service, True
        else:
            logger.info("📄 Using JSON file chat system")
            return None, False
    except Exception as e:
        logger.warning(f"Database initialization failed, using JSON fallback: {e}")
        return None, False


def load_messages_from_service(chat_service, use_database, user_id="default_user"):
    """Load messages from database or JSON file."""
    try:
        if use_database and chat_service:
            messages = chat_service.load_chat_history(user_id)
            logger.info(f"📚 Loaded {len(messages)} messages from database")
            return messages
        else:
            return load_chat_history()
    except Exception as e:
        logger.error(f"❌ Error loading messages: {e}")
        return load_chat_history()


def save_messages_to_service(messages, chat_service, use_database, user_id="default_user"):
    """Save messages to database or JSON file."""
    try:
        if use_database and chat_service:
            success = chat_service.save_chat_history(messages, user_id)
            if success:
                logger.debug(f"💾 Saved {len(messages)} messages to database")
            else:
                logger.warning("⚠️ Database save failed, falling back to JSON")
                save_chat_history(messages)
        else:
            save_chat_history(messages)
    except Exception as e:
        logger.error(f"❌ Error saving messages: {e}")
        save_chat_history(messages)


def add_message_to_service(message, chat_service, use_database, user_id="default_user"):
    """Add single message to database or JSON file."""
    try:
        if use_database and chat_service:
            result = chat_service.add_message(message, user_id)
            if result:
                logger.debug("➕ Added message to database")
    except Exception as e:
        logger.error(f"❌ Error adding message: {e}")


def get_conversations_list(chat_service, use_database, user_id="default_user"):
    """Get list of conversations for user."""
    try:
        if use_database and chat_service and hasattr(chat_service, 'get_conversations'):
            conversations = chat_service.get_conversations(user_id, limit=10)
            logger.info(f"📚 Retrieved {len(conversations)} conversations from database")
            return conversations
        else:
            return [{
                "id": "default_conversation",
                "title": f"Current Session - {datetime.now().strftime('%Y-%m-%d')}",
                "message_count": len(st.session_state.get('messages', [])),
                "updated_at": datetime.now().isoformat(),
                "last_message_preview": "Current session",
                "created_at": datetime.now().isoformat()
            }]
    except Exception as e:
        logger.error(f"Error getting conversation list: {e}")
        return []


def create_new_conversation(chat_service, use_database, user_id="default_user", title=None):
    """Create a new conversation."""
    try:
        if not title:
            title = f"New Chat - {datetime.now().strftime('%H:%M')}"

        if use_database and chat_service and hasattr(chat_service, 'create_conversation'):
            conversation_id = chat_service.create_conversation(title, user_id)
            logger.info(f"➕ Created new conversation: {conversation_id}")
            return conversation_id
        else:
            return f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        return None


def delete_conversation_by_id(chat_service, use_database, conversation_id):
    """Delete a conversation."""
    try:
        if use_database and chat_service and hasattr(chat_service, 'delete_conversation'):
            success = chat_service.delete_conversation(conversation_id)
            if success:
                logger.info(f"🗑️ Deleted conversation: {conversation_id}")
                return True
            else:
                logger.warning(f"⚠️ Failed to delete conversation: {conversation_id}")
                return False
        else:
            logger.warning("Delete not supported in JSON mode")
            return False
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return False


def update_conversation_title_by_id(chat_service, use_database, conversation_id, new_title):
    """Update conversation title."""
    try:
        if use_database and chat_service and hasattr(chat_service, 'update_conversation_title'):
            success = chat_service.update_conversation_title(conversation_id, new_title)
            if success:
                logger.info(f"✏️ Updated conversation title: {conversation_id}")
                return True
            else:
                logger.warning(f"⚠️ Failed to update conversation title: {conversation_id}")
                return False
        else:
            logger.warning("Title update not supported in JSON mode")
            return False
    except Exception as e:
        logger.error(f"Error updating conversation title: {e}")
        return False


def search_conversations_and_messages(chat_service, use_database, user_id, query):
    """Search conversations and messages."""
    try:
        if use_database and chat_service and hasattr(chat_service, 'search_conversations'):
            conv_results = chat_service.search_conversations(query, user_id)
            msg_results = chat_service.search_messages(query, user_id)

            logger.info(f"🔍 Search '{query}': {len(conv_results)} conversations, {len(msg_results)} messages")
            return {
                'conversations': conv_results,
                'messages': msg_results
            }
        else:
            messages = st.session_state.get('messages', [])
            filtered_messages = [
                msg for msg in messages
                if query.lower() in msg.get('content', '').lower()
            ]

            return {
                'conversations': [],
                'messages': [
                    {
                        'content': msg['content'],
                        'role': msg['role'],
                        'conversation_title': 'Current Session',
                        'highlight_snippet': msg['content'][:100] + "..." if len(msg['content']) > 100 else msg[
                            'content']
                    }
                    for msg in filtered_messages
                ]
            }
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return {'conversations': [], 'messages': []}


# Main Application
st.markdown("""
<div class="enhanced-header fade-in">
    <h1>🔍 Knowledge Assistant</h1>
    <p>AI-powered document analysis with intelligent conversation management</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    # Sidebar Header
    st.markdown("""
    <div class="sidebar-header fade-in">
        <div class="sidebar-title">
            🔍 Knowledge Assistant
        </div>
        <div class="sidebar-subtitle">
            AI-powered document intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize system components
    qa_engine: Optional[GraphRAGQA] = None
    is_engine_ready = False
    config = None
    neo4j_count = 0
    chat_service = None
    use_database = False

    if IMPORTS_SUCCESSFUL:
        with st.spinner("🔄 Initializing systems..."):
            try:
                config = load_config()
                if config and config.get('_CONFIG_VALID'):
                    correction_llm = get_correction_llm(config)
                    qa_engine = load_qa_engine(config, correction_llm)
                    is_engine_ready = qa_engine and qa_engine.is_ready()

                    chat_service, use_database = initialize_chat_service(config)

                    if is_engine_ready:
                        st.success("✅ System Ready")
                        logger.info("✅ System initialization complete")
                    else:
                        st.error("❌ System Offline")
                        logger.error("❌ System not ready")
                else:
                    st.error("❌ Configuration Invalid")
                    logger.error("❌ Invalid configuration")

            except Exception as e:
                logger.error(f"System initialization error: {e}")
                st.error(f"❌ Initialization failed: {e}")

    # Enhanced User Management Section
    st.markdown("""
    <div class="enhanced-section fade-in">
        <div class="section-title-enhanced">👤 User Session</div>
    """, unsafe_allow_html=True)

    if "current_user" not in st.session_state:
        st.session_state.current_user = "default_user"

    # Enhanced user display
    user_initials = "".join([word[0].upper() for word in st.session_state.current_user.split("_")])[:2]
    st.markdown(f"""
        <div class="enhanced-user-container">
            <div class="user-avatar">{user_initials}</div>
            <div class="user-info">
                <div class="user-name">{st.session_state.current_user}</div>
                <div class="user-status">● Active now</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User change option
    with st.expander("🔧 Change User"):
        new_user = st.text_input("Enter username:", value=st.session_state.current_user, key="user_change")
        if st.button("Switch User", key="switch_user") and new_user != st.session_state.current_user:
            st.session_state.current_user = new_user
            for key in ['messages', 'current_conversation_id', 'search_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success(f"Switched to user: {new_user}")
            logger.info(f"👤 User switched to: {new_user}")
            st.rerun()

    # Enhanced Search Section
    st.markdown("""
    <div class="enhanced-section fade-in">
        <div class="section-title-enhanced">🔍 Search</div>
    """, unsafe_allow_html=True)

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "show_search_results" not in st.session_state:
        st.session_state.show_search_results = False

    # Enhanced search input with proper label
    st.markdown('<div class="enhanced-search-container"><span class="search-icon">🔍</span>', unsafe_allow_html=True)
    search_query = st.text_input(
        "Search conversations and messages",
        value=st.session_state.search_query,
        placeholder="Search conversations and messages...",
        label_visibility="collapsed",
        key="search_input"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced search buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔍", key="search_btn", disabled=not search_query.strip(), help="Search"):
            if search_query.strip():
                st.session_state.search_query = search_query
                st.session_state.search_results = search_conversations_and_messages(
                    chat_service, use_database, st.session_state.current_user, search_query.strip()
                )
                st.session_state.show_search_results = True
                logger.info(f"🔍 Search performed: {search_query.strip()}")
                st.rerun()

    with col2:
        if st.button("✖️", key="clear_search", help="Clear"):
            st.session_state.search_query = ""
            st.session_state.search_results = None
            st.session_state.show_search_results = False
            logger.info("🔍 Search cleared")
            st.rerun()

    # Display search results
    if st.session_state.show_search_results and st.session_state.search_results:
        search_results = st.session_state.search_results

        if search_results['conversations']:
            st.markdown("**📂 Conversations:**")
            for conv in search_results['conversations'][:3]:
                if st.button(f"💬 {conv['title'][:30]}...", key=f"search_conv_{conv['id']}", use_container_width=True):
                    st.session_state.current_conversation_id = conv['id']
                    st.session_state.show_search_results = False
                    logger.info(f"📂 Switched to conversation: {conv['id']}")
                    st.rerun()
                st.caption(f"📊 {conv['message_count']} messages")

        if search_results['messages']:
            st.markdown("**💬 Messages:**")
            for i, msg in enumerate(search_results['messages'][:3]):
                st.markdown(f"**{msg['conversation_title']}**")
                st.caption(f"{msg['highlight_snippet']}")
                st.caption(f"Role: {msg['role']}")
                if i < 2:  # Don't add separator after last item
                    st.markdown("---")

        total_convs = len(search_results['conversations'])
        total_msgs = len(search_results['messages'])
        st.caption(f"Found: {total_convs} conversations, {total_msgs} messages")

    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Conversation Management
    st.markdown("""
    <div class="enhanced-section fade-in">
        <div class="section-title-enhanced">💬 Conversations</div>
    """, unsafe_allow_html=True)

    # Initialize chat history and conversation management
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages_from_service(
            chat_service, use_database, st.session_state.current_user
        )

    if "chat_service" not in st.session_state:
        st.session_state.chat_service = chat_service

    if "use_database" not in st.session_state:
        st.session_state.use_database = use_database

    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = "default_conversation"

    # Enhanced New conversation button
    if st.button("➕ New Conversation", key="new_conv", use_container_width=True):
        if st.session_state.messages:
            save_messages_to_service(
                st.session_state.messages,
                st.session_state.chat_service,
                st.session_state.use_database,
                st.session_state.current_user
            )

        new_conv_id = create_new_conversation(
            st.session_state.chat_service,
            st.session_state.use_database,
            st.session_state.current_user
        )

        if new_conv_id:
            st.session_state.current_conversation_id = new_conv_id
            st.session_state.messages = []
            st.success("✅ New conversation started!")
            logger.info(f"➕ New conversation created: {new_conv_id}")
            st.rerun()

    # Get and display conversations
    conversations = get_conversations_list(
        st.session_state.chat_service,
        st.session_state.use_database,
        st.session_state.current_user
    )

    if conversations:
        st.markdown("**Recent Conversations:**")

        for conv in conversations[:5]:
            is_current = conv["id"] == st.session_state.current_conversation_id

            # Create enhanced conversation item
            conv_class = "enhanced-conversation-item active" if is_current else "enhanced-conversation-item"
            conv_icon = "🟢" if is_current else "💬"

            st.markdown(f"""
            <div class="{conv_class}">
                <div class="conversation-title-enhanced">{conv_icon} {conv['title'][:30]}{'...' if len(conv['title']) > 30 else ''}</div>
                <div class="conversation-meta-enhanced">📊 {conv['message_count']} messages</div>
            """, unsafe_allow_html=True)

            if conv.get('last_message_preview'):
                st.markdown(f'<div class="conversation-preview">{conv["last_message_preview"][:40]}...</div>',
                            unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Main conversation button
            if not is_current:
                if st.button(
                        f"Open Conversation",
                        key=f"conv_main_{conv['id']}",
                        use_container_width=True
                ):
                    if st.session_state.messages:
                        save_messages_to_service(
                            st.session_state.messages,
                            st.session_state.chat_service,
                            st.session_state.use_database,
                            st.session_state.current_user
                        )

                    st.session_state.current_conversation_id = conv["id"]
                    st.session_state.messages = load_messages_from_service(
                        st.session_state.chat_service,
                        st.session_state.use_database,
                        st.session_state.current_user
                    )
                    logger.info(f"💬 Switched to conversation: {conv['id']}")
                    st.rerun()
            else:
                st.info("📍 Current conversation")

            # Action buttons for non-current conversations
            if not is_current and use_database:
                col_edit, col_delete = st.columns(2)
                with col_edit:
                    if st.button("✏️", key=f"edit_{conv['id']}", help="Edit title", use_container_width=True):
                        st.session_state[f"editing_title_{conv['id']}"] = True
                        st.rerun()

                with col_delete:
                    if st.button("🗑️", key=f"delete_{conv['id']}", help="Delete conversation",
                                 use_container_width=True):
                        if delete_conversation_by_id(
                                st.session_state.chat_service,
                                st.session_state.use_database,
                                conv['id']
                        ):
                            st.success(f"✅ Deleted: {conv['title'][:20]}...")
                            logger.info(f"🗑️ Deleted conversation: {conv['id']}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete conversation")

            # Edit title interface
            if st.session_state.get(f"editing_title_{conv['id']}", False):
                st.markdown("**Edit Title:**")
                new_title = st.text_input(
                    "New title:",
                    value=conv['title'],
                    key=f"title_input_{conv['id']}",
                    placeholder="Enter new conversation title..."
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾", key=f"save_{conv['id']}", use_container_width=True, help="Save"):
                        if new_title.strip():
                            if update_conversation_title_by_id(
                                    st.session_state.chat_service,
                                    st.session_state.use_database,
                                    conv['id'],
                                    new_title.strip()
                            ):
                                st.success("✅ Title updated!")
                                st.session_state[f"editing_title_{conv['id']}"] = False
                                logger.info(f"✏️ Title updated for conversation: {conv['id']}")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update title")
                        else:
                            st.warning("⚠️ Title cannot be empty")

                with col_cancel:
                    if st.button("❌", key=f"cancel_{conv['id']}", use_container_width=True, help="Cancel"):
                        st.session_state[f"editing_title_{conv['id']}"] = False
                        st.rerun()

            st.markdown("---")

    else:
        st.info("No conversations yet. Start by asking a question!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Knowledge Base Section
    if is_engine_ready and config:
        st.markdown("""
        <div class="enhanced-section fade-in">
            <div class="section-title-enhanced">📊 Knowledge Base</div>
        """, unsafe_allow_html=True)

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
            logger.info(f"📊 Neo4j entities count: {neo4j_count}")
        except Exception as e:
            logger.warning(f"Neo4j check failed: {e}")
            neo4j_count = 0

        # Enhanced metrics display
        st.markdown(f"""
        <div class="enhanced-metrics-grid">
            <div class="enhanced-metric-card">
                <div class="enhanced-metric-value">{neo4j_count:,}</div>
                <div class="enhanced-metric-label">Entities</div>
            </div>
            <div class="enhanced-metric-card">
                <div class="enhanced-metric-value">📊</div>
                <div class="enhanced-metric-label">GraphRAG</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced status indicator
        status_class = "status-success" if neo4j_count > 0 else "status-warning"
        status_text = "✅ Knowledge base operational" if neo4j_count > 0 else "⚠️ Process documents first"

        st.markdown(f"""
        <div class="enhanced-status-indicator {status_class}">
            {status_text}
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Session Analytics
    if st.session_state.messages:
        st.markdown("""
        <div class="enhanced-section fade-in">
            <div class="section-title-enhanced">📈 Session Analytics</div>
        """, unsafe_allow_html=True)

        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])

        response_times = [m.get("response_time", 0) for m in st.session_state.messages if
                          m.get("role") == "assistant" and m.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        st.markdown(f"""
        <div class="enhanced-metrics-grid">
            <div class="enhanced-metric-card">
                <div class="enhanced-metric-value">{total_messages}</div>
                <div class="enhanced-metric-label">Messages</div>
            </div>
            <div class="enhanced-metric-card">
                <div class="enhanced-metric-value">{user_messages}</div>
                <div class="enhanced-metric-label">Queries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if avg_response_time > 0:
            st.markdown(f"""
            <div class="enhanced-metric-card" style="grid-column: 1 / -1; margin-top: 1rem;">
                <div class="enhanced-metric-value">{avg_response_time:.1f}s</div>
                <div class="enhanced-metric-label">Avg Response</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Controls Section
    st.markdown("""
    <div class="enhanced-section fade-in">
        <div class="section-title-enhanced">🔧 Controls</div>
    """, unsafe_allow_html=True)

    # Enhanced controls with animation effects
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️", key="clear_chat", help="Clear Current Chat", use_container_width=True):
            st.session_state.messages = []
            save_messages_to_service(
                [], st.session_state.chat_service, st.session_state.use_database,
                st.session_state.current_user
            )
            st.success("✅ Current conversation cleared!")
            logger.info("🗑️ Current conversation cleared")
            st.rerun()

    with col2:
        if st.button("🔄", key="refresh_system", help="Refresh System", use_container_width=True):
            st.cache_data.clear()
            for key in ['messages', 'chat_service', 'use_database', 'search_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("🔄 System refreshed!")
            logger.info("🔄 System refreshed")
            st.rerun()

    # Additional enhanced controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊", key="show_analytics", help="Show Analytics", use_container_width=True):
            st.info("📊 Analytics feature coming soon!")

    with col2:
        if st.button("⚙️", key="settings", help="Settings", use_container_width=True):
            st.info("⚙️ Settings panel coming soon!")

    with col3:
        if st.button("❓", key="help", help="Help & Support", use_container_width=True):
            st.info("❓ Help documentation coming soon!")

    st.markdown('</div>', unsafe_allow_html=True)

# Main Chat Interface
# Enhanced performance indicator for last response
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg.get("role") == "assistant" and "response_time" in last_msg:
        duration = last_msg["response_time"]
        emoji, status, css_class = get_performance_indicator(duration)

        st.markdown(f"""
        <div class="{css_class} fade-in">
            {emoji} {status} Response Time ({duration:.1f}s)
        </div>
        """, unsafe_allow_html=True)

# Current conversation info
if st.session_state.get(
        'current_conversation_id') and st.session_state.current_conversation_id != 'default_conversation':
    st.info(f"💬 **Current Conversation:** {st.session_state.current_conversation_id}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Enhanced chat input
if IMPORTS_SUCCESSFUL and is_engine_ready:
    if neo4j_count > 0:
        placeholder = "💼 Ask me about your business documents and data..."
    else:
        placeholder = "📋 Please process documents first, then start asking questions..."
else:
    placeholder = "🔧 System initializing - please wait..."

# Main chat input
if prompt := st.chat_input(placeholder, disabled=not (IMPORTS_SUCCESSFUL and is_engine_ready)):
    # Clear search results when starting new conversation
    if st.session_state.get('show_search_results'):
        st.session_state.show_search_results = False

    # Add user message
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_message)
    logger.info(f"👤 User query: {prompt[:50]}...")

    # Save to storage
    add_message_to_service(
        user_message, st.session_state.chat_service, st.session_state.use_database,
        st.session_state.current_user
    )

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
        logger.info(f"🤖 Greeting response sent")

        # Save to storage
        add_message_to_service(
            assistant_message, st.session_state.chat_service, st.session_state.use_database,
            st.session_state.current_user
        )
        save_messages_to_service(
            st.session_state.messages, st.session_state.chat_service,
            st.session_state.use_database, st.session_state.current_user
        )

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
                    add_message_to_service(
                        assistant_message, st.session_state.chat_service,
                        st.session_state.use_database, st.session_state.current_user
                    )
                    save_messages_to_service(
                        st.session_state.messages, st.session_state.chat_service,
                        st.session_state.use_database, st.session_state.current_user
                    )

                    # Success feedback
                    emoji, status, _ = get_performance_indicator(duration)
                    st.success(f"{emoji} Analysis completed in {duration:.2f}s - {status} performance")
                    logger.info(f"🤖 Query processed successfully in {duration:.2f}s")

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
                    add_message_to_service(
                        assistant_message, st.session_state.chat_service,
                        st.session_state.use_database, st.session_state.current_user
                    )
                    save_messages_to_service(
                        st.session_state.messages, st.session_state.chat_service,
                        st.session_state.use_database, st.session_state.current_user
                    )

                    logger.error(f"❌ Query processing failed: {str(e)}")

        st.rerun()

# Enhanced Footer
st.markdown("""
<div class="enhanced-footer fade-in">
    <div class="enhanced-footer-item">
        <span>👤</span>
        <strong>User:</strong> {user}
    </div>
    <div class="enhanced-footer-item">
        <span>💬</span>
        <strong>Mode:</strong> {mode}
    </div>
    <div class="enhanced-footer-item">
        <span>📊</span>
        <strong>Messages:</strong> {count}
    </div>
</div>
""".format(
    user=st.session_state.get('current_user', 'default_user'),
    mode="Active Session" if st.session_state.get('current_conversation_id') == 'default_conversation' else "Database",
    count=len(st.session_state.messages)
), unsafe_allow_html=True)

# Debug information (only in development)
try:
    debug_mode = st.secrets.get("debug_mode", False)
except:
    debug_mode = False

if debug_mode:
    with st.expander("🔍 Debug Information"):
        st.json({
            "database_available": DATABASE_INTEGRATION_AVAILABLE,
            "use_database": st.session_state.get('use_database', False),
            "current_user": st.session_state.get('current_user'),
            "current_conversation_id": st.session_state.get('current_conversation_id'),
            "imports_successful": IMPORTS_SUCCESSFUL,
            "engine_ready": is_engine_ready,
            "neo4j_entities": neo4j_count,
            "message_count": len(st.session_state.messages),
            "search_active": st.session_state.get('show_search_results', False)
        })
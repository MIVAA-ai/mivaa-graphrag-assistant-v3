# pages/1_Knowledge_Chat_Assistant.py

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

# Modern, clean styling focused on chat experience
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

    /* Single container for question-answer pairs */
    .chat-pair {
        background: var(--bg-chat);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-xl);
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: var(--shadow-sm);
    }

    /* User messages - part of pair */
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border-color: #bfdbfe !important;
        margin: 0 0 0.5rem 0 !important;
        border-radius: 12px 12px 4px 12px !important;
    }

    /* Assistant messages - part of pair */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: var(--bg-chat) !important;
        border-color: var(--border-light) !important;
        margin: 0 !important;
        border-radius: 4px 12px 12px 12px !important;
    }

    /* Normal chat input - not fixed */
    .stChatInput {
        position: relative !important;
        bottom: auto !important;
        left: auto !important;
        right: auto !important;
        z-index: auto !important;
        max-width: 100% !important;
        margin: 1rem 0 !important;
    }

    .stChatInput > div {
        background: var(--bg-primary) !important;
        border: 2px solid var(--border-light) !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: all 0.2s ease !important;
    }

    .stChatInput > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    .stChatInput textarea {
        border: none !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        padding: 1rem 1.25rem !important;
        min-height: 2.5rem !important;
        resize: none !important;
        background: transparent !important;
    }

    .stChatInput textarea::placeholder {
        color: var(--text-muted) !important;
        font-style: italic !important;
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

    /* Sidebar improvements */
    .css-1d391kg {
        background: var(--bg-primary) !important;
        border-right: 1px solid var(--border-light) !important;
    }

    /* Chat container spacing - normal spacing */
    .main .block-container {
        padding-bottom: 2rem !important;
    }

    /* Success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: var(--radius-lg) !important;
        border: none !important;
        box-shadow: var(--shadow-sm) !important;
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

    /* Smooth animations */
    .fade-in {
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
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

# Constants
CHAT_HISTORY_FILE = Path("./chat_history.json")


# Helper Functions
def load_chat_history() -> List[Dict]:
    """Load chat history with error handling."""
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


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Compact header instead of big banner
st.markdown("""
<div class="compact-header fade-in">
    <h1>🔍 Knowledge Assistant</h1>
    <p>AI-powered document analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Compact Control Panel
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ System Control")

    # Initialize system status
    qa_engine: Optional[GraphRAGQA] = None
    is_engine_ready = False
    config = None
    neo4j_count = 0

    if IMPORTS_SUCCESSFUL:
        # System initialization
        with st.spinner("🔄 Initializing..."):
            try:
                config = load_config()
                if config and config.get('_CONFIG_VALID'):
                    correction_llm = get_correction_llm(config)
                    qa_engine = load_qa_engine(config, correction_llm)
                    is_engine_ready = qa_engine and qa_engine.is_ready()

                    if is_engine_ready:
                        st.success("✅ AI Engine Ready")
                    else:
                        st.error("❌ AI Engine Offline")
                else:
                    st.error("❌ Configuration Invalid")

            except Exception as e:
                logger.error(f"System initialization error: {e}")
                st.error(f"❌ Initialization failed: {e}")

        # Compact system status
        st.markdown("### 📊 Status")

        col1, col2 = st.columns(2)

        with col1:
            if is_engine_ready:
                st.markdown("**AI Engine**")
                st.markdown('<div class="status-indicator status-online">● Online</div>', unsafe_allow_html=True)
            else:
                st.markdown("**AI Engine**")
                st.markdown('<div class="status-indicator status-offline">● Offline</div>', unsafe_allow_html=True)

        with col2:
            if config and config.get('_CONFIG_VALID'):
                st.markdown("**Config**")
                st.markdown('<div class="status-indicator status-online">● Valid</div>', unsafe_allow_html=True)
            else:
                st.markdown("**Config**")
                st.markdown('<div class="status-indicator status-offline">● Invalid</div>', unsafe_allow_html=True)

        # Compact knowledge base metrics
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
                    <div class="compact-metric-value">9</div>
                    <div class="compact-metric-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)

            # Health indicator
            if neo4j_count > 0:
                st.success("✅ Knowledge base operational")
            else:
                st.error("❌ No data found")

    else:
        st.error("❌ Core modules not available")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Compact session analytics
    if st.session_state.messages:
        st.markdown("### 📈 Session")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])

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

    # Compact controls
    st.markdown("### 🔧 Controls")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        save_chat_history([])
        st.success("Chat history cleared!")
        st.rerun()

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Performance indicator for last response (more subtle)
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
    # Add user message
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_message)
    save_chat_history(st.session_state.messages)

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
        save_chat_history(st.session_state.messages)
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

                    # Save to history
                    st.session_state.messages.append(assistant_message)
                    save_chat_history(st.session_state.messages)

                    # Success feedback
                    emoji, status, _ = get_performance_indicator(duration)
                    st.success(f"{emoji} Analysis completed in {duration:.2f}s - {status} performance")

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

                    st.session_state.messages.append(assistant_message)
                    save_chat_history(st.session_state.messages)

                    logger.exception("Error in QA processing")

        st.rerun()
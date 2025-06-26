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

# Elegant and subtle styling
st.markdown("""
<style>
    /* Import sophisticated fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Elegant color palette */
    :root {
        /* Sophisticated neutrals */
        --elegant-slate-50: #f8fafc;
        --elegant-slate-100: #f1f5f9;
        --elegant-slate-200: #e2e8f0;
        --elegant-slate-300: #cbd5e1;
        --elegant-slate-400: #94a3b8;
        --elegant-slate-500: #64748b;
        --elegant-slate-600: #475569;
        --elegant-slate-700: #334155;
        --elegant-slate-800: #1e293b;
        --elegant-slate-900: #0f172a;

        /* Refined accent colors */
        --elegant-blue: #3b82f6;
        --elegant-blue-light: #60a5fa;
        --elegant-emerald: #10b981;
        --elegant-amber: #f59e0b;
        --elegant-rose: #f43f5e;

        /* Elegant shadows */
        --elegant-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --elegant-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --elegant-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --elegant-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);

        /* Subtle gradients */
        --elegant-gradient-blue: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        --elegant-gradient-emerald: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --elegant-gradient-surface: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }

    /* Base styling with elegance */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--elegant-slate-50);
        color: var(--elegant-slate-700);
        line-height: 1.7;
    }

    /* Hide Streamlit branding elegantly */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Elegant main header */
    .elegant-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%); /* Darker, more sophisticated slate gradient */
        padding: 2rem 2.5rem; /* Reduced from 3rem to 2rem for smaller size */
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: var(--elegant-shadow-lg);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    .elegant-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        animation: elegant-drift 30s infinite linear; /* Slower, more subtle animation */
    }

    @keyframes elegant-drift {
        0% { transform: translateX(0) translateY(0); }
        100% { transform: translateX(-60px) translateY(-60px); }
    }

    .elegant-header h1 {
        color: white;
        font-weight: 300;
        font-size: 2rem; /* Reduced from 2.5rem for more elegant proportions */
        letter-spacing: -0.02em;
        margin: 0;
        text-align: center;
        position: relative;
        z-index: 1;
    }

    .elegant-header p {
        color: rgba(255, 255, 255, 0.75); /* Slightly more muted opacity */
        font-weight: 300; /* Lighter weight for more elegance */
        font-size: 0.95rem; /* Reduced from 1.125rem for subtle sophistication */
        margin: 0.75rem 0 0 0; /* Reduced top margin */
        text-align: center;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em; /* Subtle letter spacing for refinement */
    }

    /* Sophisticated buttons */
    .stButton > button {
        background: var(--elegant-gradient-blue);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.25);
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.35);
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Elegant status indicators */
    .elegant-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 24px;
        font-size: 0.875rem;
        font-weight: 500;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }

    .elegant-status:hover {
        transform: translateY(-1px);
        box-shadow: var(--elegant-shadow-sm);
    }

    .status-online {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: var(--elegant-emerald);
    }

    .status-offline {
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.2);
        color: var(--elegant-rose);
    }

    /* Refined metric cards */
    .elegant-metric-card {
        background: var(--elegant-gradient-surface);
        border: 1px solid var(--elegant-slate-200);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    .elegant-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--elegant-gradient-blue);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .elegant-metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--elegant-shadow-lg);
        border-color: var(--elegant-slate-300);
    }

    .elegant-metric-card:hover::before {
        opacity: 1;
    }

    .elegant-metric-value {
        font-size: 2.5rem;
        font-weight: 300;
        color: var(--elegant-slate-800);
        margin-bottom: 0.5rem;
        line-height: 1;
    }

    .elegant-metric-label {
        color: var(--elegant-slate-500);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Performance indicators with elegance */
    .performance-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 16px;
        font-weight: 500;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }

    .performance-good {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: var(--elegant-emerald);
    }

    .performance-medium {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: var(--elegant-amber);
    }

    .performance-slow {
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.2);
        color: var(--elegant-rose);
    }

    /* Elegant sidebar styling */
    .css-1d391kg {
        background: var(--elegant-gradient-surface);
        border-right: 1px solid var(--elegant-slate-200);
    }

    /* Refined chat interface */
    .stChatMessage {
        background: var(--elegant-gradient-surface);
        border: 1px solid var(--elegant-slate-200);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--elegant-shadow-sm);
        transition: all 0.2s ease;
    }

    .stChatMessage:hover {
        box-shadow: var(--elegant-shadow);
        border-color: var(--elegant-slate-300);
    }

    /* Elegant animations */
    @keyframes elegant-fade-in {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .elegant-enter {
        animation: elegant-fade-in 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Responsive elegance */
    @media (max-width: 768px) {
        .elegant-header {
            padding: 2rem 1.25rem;
            border-radius: 16px;
        }

        .elegant-header h1 {
            font-size: 1.75rem;
        }

        .elegant-header p {
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }

        .elegant-metric-card {
            padding: 1.5rem;
        }

        .elegant-metric-value {
            font-size: 2rem;
        }
    }

    /* Subtle loading states */
    .elegant-spinner {
        border: 2px solid var(--elegant-slate-200);
        border-top: 2px solid var(--elegant-blue);
        border-radius: 50%;
        animation: elegant-spin 1s linear infinite;
    }

    @keyframes elegant-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
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

# Elegant main header
st.markdown("""
<div class="elegant-header elegant-enter">
    <h1>🔍 Knowledge Assistant</h1>
    <p>Intelligent document analysis powered by advanced AI</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Elegant Control Panel
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ System Control")

    # Initialize system status
    qa_engine: Optional[GraphRAGQA] = None
    is_engine_ready = False
    config = None
    neo4j_count = 0

    if IMPORTS_SUCCESSFUL:
        # System initialization with elegant loading
        with st.spinner("🔄 Initializing system..."):
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

        # Elegant system status
        st.markdown("### 📊 System Status")

        col1, col2 = st.columns(2)

        with col1:
            if is_engine_ready:
                st.markdown("**AI Engine**")
                st.markdown('<div class="elegant-status status-online">● Online</div>', unsafe_allow_html=True)
            else:
                st.markdown("**AI Engine**")
                st.markdown('<div class="elegant-status status-offline">● Offline</div>', unsafe_allow_html=True)

        with col2:
            if config and config.get('_CONFIG_VALID'):
                st.markdown("**Configuration**")
                st.markdown('<div class="elegant-status status-online">● Valid</div>', unsafe_allow_html=True)
            else:
                st.markdown("**Configuration**")
                st.markdown('<div class="elegant-status status-offline">● Invalid</div>', unsafe_allow_html=True)

        # Elegant knowledge base metrics
        if is_engine_ready and config:
            st.markdown("### 📊 Knowledge Base")

            # Check Neo4j with elegant error handling
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

            # Elegant metrics display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">{neo4j_count:,}</div>
                    <div class="elegant-metric-label">Entities</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="elegant-metric-card">
                    <div class="elegant-metric-value">9</div>
                    <div class="elegant-metric-label">Documents</div>
                </div>
                """, unsafe_allow_html=True)

            # Elegant health indicator
            if neo4j_count > 0:
                st.success("✅ **Knowledge base operational**")
            else:
                st.error("❌ **No data found**")

    else:
        st.error("❌ Core modules not available")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Elegant session analytics
    if st.session_state.messages:
        st.markdown("### 📈 Session Analytics")
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="elegant-metric-card">
                <div class="elegant-metric-value">{total_messages}</div>
                <div class="elegant-metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="elegant-metric-card">
                <div class="elegant-metric-value">{user_messages}</div>
                <div class="elegant-metric-label">Queries</div>
            </div>
            """, unsafe_allow_html=True)

    # Elegant system controls
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

# Elegant performance indicator for last response
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if last_msg.get("role") == "assistant" and "response_time" in last_msg:
        duration = last_msg["response_time"]
        emoji, status, css_class = get_performance_indicator(duration)

        st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem auto; max-width: 300px;">
            <div class="performance-indicator {css_class}">
                {emoji} Performance: {status} ({duration:.1f}s)
            </div>
        </div>
        """, unsafe_allow_html=True)

# Display chat messages with elegant styling
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Elegant chat input
if IMPORTS_SUCCESSFUL and is_engine_ready:
    if neo4j_count > 0:
        placeholder = "💼 Ask me about your business documents and data..."
    else:
        placeholder = "📋 Please process documents first, then start asking questions..."
else:
    placeholder = "🔧 System initializing - please wait..."

# Main chat input with enhanced functionality
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

    # Handle greetings with elegance
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

                    # Elegant success feedback
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
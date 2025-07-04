# src/chat/sync_database.py
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json
import asyncpg
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)


class SyncDatabaseManager:
    """
    Synchronous database manager that properly handles async operations
    in Streamlit's threading environment.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._pool = None
        self._loop = None
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the database connection in a separate thread with its own event loop"""
        try:
            logger.info("🔄 Initializing sync database manager...")

            # Create a separate thread for async operations
            self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self._thread.start()

            # Wait for initialization to complete
            future = self._submit_async_task(self._async_initialize())
            result = future.result(timeout=30)  # 30 second timeout

            self._initialized = result
            if result:
                logger.info("✅ Sync database manager initialized successfully")
            else:
                logger.error("❌ Failed to initialize sync database manager")

            return result

        except Exception as e:
            logger.error(f"❌ Sync database initialization failed: {e}")
            return False

    def _run_async_loop(self):
        """Run event loop in separate thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def _submit_async_task(self, coro):
        """Submit async task to the dedicated event loop"""
        if not self._loop:
            raise RuntimeError("Event loop not initialized")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future

    async def _async_initialize(self) -> bool:
        """Async initialization logic"""
        try:
            # Get database configuration
            db_config = self.config.get('database', {})

            # Build connection parameters
            connection_params = {
                'host': db_config.get('host', 'localhost'),
                'port': db_config.get('port', 5432),
                'database': db_config.get('database', 'graphrag_chat'),
                'user': db_config.get('user', 'postgres'),
                'password': db_config.get('password', 'password'),
                'min_size': db_config.get('min_pool_size', 2),
                'max_size': db_config.get('max_pool_size', 10),
                'command_timeout': 60
            }

            # Override with environment variables
            import os
            connection_params.update({
                'host': os.getenv('DB_HOST', connection_params['host']),
                'port': int(os.getenv('DB_PORT', connection_params['port'])),
                'database': os.getenv('DB_NAME', connection_params['database']),
                'user': os.getenv('DB_USER', connection_params['user']),
                'password': os.getenv('DB_PASSWORD', connection_params['password'])
            })

            logger.info(
                f"🔗 Connecting to PostgreSQL: {connection_params['user']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=connection_params['host'],
                port=connection_params['port'],
                database=connection_params['database'],
                user=connection_params['user'],
                password=connection_params['password'],
                min_size=connection_params['min_size'],
                max_size=connection_params['max_size'],
                command_timeout=connection_params['command_timeout']
            )

            # Create schema
            await self._ensure_schema()

            return True

        except Exception as e:
            logger.error(f"💥 Async initialization failed: {e}")
            return False

    async def _ensure_schema(self):
        """Create database schema if it doesn't exist"""
        schema_sql = """
        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";

        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            preferences JSONB DEFAULT '{}'
        );

        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
            title VARCHAR(500) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_archived BOOLEAN DEFAULT FALSE,
            metadata JSONB DEFAULT '{}',
            message_count INTEGER DEFAULT 0
        );

        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp VARCHAR(50) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

            -- AI Response Metadata
            response_time FLOAT,
            model_used VARCHAR(100),
            tokens_used INTEGER,

            -- RAG Attribution
            sources JSONB,
            cypher_query TEXT,
            linked_entities JSONB,
            info JSONB,

            -- User Feedback
            user_rating INTEGER CHECK (user_rating BETWEEN -1 AND 1),
            user_feedback TEXT,

            -- Technical
            error_info TEXT,
            processing_status VARCHAR(20) DEFAULT 'completed'
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated 
            ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created 
            ON messages(conversation_id, created_at);

        -- Create default user
        INSERT INTO users (username, email) 
        VALUES ('default_user', 'default@example.com')
        ON CONFLICT (username) DO NOTHING;
        """

        async with self._pool.acquire() as conn:
            await conn.execute(schema_sql)
            logger.info("✅ Database schema ensured")

    # ==========================================================================
    # SYNC INTERFACE METHODS
    # ==========================================================================

    def load_messages(self, user_id: str = "default_user", conversation_id: str = None) -> List[Dict[str, Any]]:
        """Load messages synchronously"""
        if not self._initialized:
            logger.error("Database not initialized")
            return []

        try:
            future = self._submit_async_task(self._async_load_messages(user_id, conversation_id))
            return future.result(timeout=10)
        except Exception as e:
            logger.error(f"💥 Error loading messages: {e}")
            return []

    def save_message(self, message_dict: Dict[str, Any], user_id: str = "default_user",
                     conversation_id: str = None) -> str:
        """Save single message synchronously"""
        if not self._initialized:
            logger.error("Database not initialized")
            return ""

        try:
            future = self._submit_async_task(self._async_save_message(message_dict, user_id, conversation_id))
            return future.result(timeout=10)
        except Exception as e:
            logger.error(f"💥 Error saving message: {e}")
            return ""

    def save_messages(self, messages: List[Dict[str, Any]], user_id: str = "default_user") -> bool:
        """Save multiple messages synchronously"""
        if not self._initialized:
            logger.error("Database not initialized")
            return False

        try:
            future = self._submit_async_task(self._async_save_messages(messages, user_id))
            return future.result(timeout=15)
        except Exception as e:
            logger.error(f"💥 Error saving messages: {e}")
            return False

    def get_conversations(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Get conversations synchronously"""
        if not self._initialized:
            logger.error("Database not initialized")
            return []

        try:
            future = self._submit_async_task(self._async_get_conversations(user_id))
            return future.result(timeout=10)
        except Exception as e:
            logger.error(f"💥 Error getting conversations: {e}")
            return []

    # ==========================================================================
    # ASYNC IMPLEMENTATION METHODS
    # ==========================================================================

    async def _async_load_messages(self, user_id: str, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Load messages async implementation"""
        try:
            # Ensure user exists
            await self._ensure_user(user_id)

            # Get conversation ID
            if conversation_id is None:
                conversation_id = await self._get_or_create_default_conversation(user_id)

            # Load messages
            query = """
            SELECT role, content, timestamp, response_time, model_used, tokens_used,
                   sources, cypher_query, linked_entities, info, error_info, user_rating, user_feedback
            FROM messages 
            WHERE conversation_id = $1 
            ORDER BY created_at ASC
            """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, conversation_id)

                messages = []
                for row in rows:
                    message = {
                        "role": row['role'],
                        "content": row['content'],
                        "timestamp": row['timestamp']
                    }

                    # Add optional fields
                    if row['response_time']:
                        message["response_time"] = row['response_time']
                    if row['model_used']:
                        message["model_used"] = row['model_used']
                    if row['tokens_used']:
                        message["tokens_used"] = row['tokens_used']
                    if row['sources']:
                        message["sources"] = row['sources']
                    if row['cypher_query']:
                        message["cypher_query"] = row['cypher_query']
                    if row['linked_entities']:
                        message["linked_entities"] = row['linked_entities']
                    if row['info']:
                        message["info"] = row['info']
                    if row['error_info']:
                        message["error_info"] = row['error_info']
                    if row['user_rating'] is not None:
                        message["user_rating"] = row['user_rating']
                    if row['user_feedback']:
                        message["user_feedback"] = row['user_feedback']

                    messages.append(message)

                logger.info(f"📚 Loaded {len(messages)} messages from database")
                return messages

        except Exception as e:
            logger.error(f"💥 Error loading messages: {e}")
            return []

    async def _async_save_message(self, message_dict: Dict[str, Any], user_id: str, conversation_id: str = None) -> str:
        """Save single message async implementation"""
        try:
            # Ensure user exists
            await self._ensure_user(user_id)

            # Get conversation ID
            if conversation_id is None:
                conversation_id = await self._get_or_create_default_conversation(user_id)

            # Insert message
            message_id = str(uuid.uuid4())
            query = """
            INSERT INTO messages (
                id, conversation_id, role, content, timestamp, created_at,
                response_time, model_used, tokens_used, sources, cypher_query,
                linked_entities, info, error_info, user_rating, user_feedback
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """

            async with self._pool.acquire() as conn:
                await conn.execute(
                    query,
                    message_id,
                    conversation_id,
                    message_dict["role"],
                    message_dict["content"],
                    message_dict["timestamp"],
                    datetime.utcnow(),
                    message_dict.get("response_time"),
                    message_dict.get("model_used"),
                    message_dict.get("tokens_used"),
                    json.dumps(message_dict.get("sources")) if message_dict.get("sources") else None,
                    message_dict.get("cypher_query"),
                    json.dumps(message_dict.get("linked_entities")) if message_dict.get("linked_entities") else None,
                    json.dumps(message_dict.get("info")) if message_dict.get("info") else None,
                    message_dict.get("error_info"),
                    message_dict.get("user_rating"),
                    message_dict.get("user_feedback")
                )

                # Update conversation message count
                await conn.execute(
                    "UPDATE conversations SET message_count = message_count + 1, updated_at = $1 WHERE id = $2",
                    datetime.utcnow(), conversation_id
                )

            logger.debug(f"➕ Added message to conversation {conversation_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"💥 Error saving message: {e}")
            raise

    async def _async_save_messages(self, messages: List[Dict[str, Any]], user_id: str) -> bool:
        """Save multiple messages async implementation"""
        try:
            # Ensure user exists
            await self._ensure_user(user_id)

            # Get or create conversation
            conversation_id = await self._get_or_create_default_conversation(user_id)

            # Clear existing messages
            async with self._pool.acquire() as conn:
                await conn.execute("DELETE FROM messages WHERE conversation_id = $1", conversation_id)

                # Insert all messages
                for message_dict in messages:
                    message_id = str(uuid.uuid4())
                    await conn.execute("""
                        INSERT INTO messages (
                            id, conversation_id, role, content, timestamp, created_at,
                            response_time, model_used, tokens_used, sources, cypher_query,
                            linked_entities, info, error_info, user_rating, user_feedback
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                                       message_id,
                                       conversation_id,
                                       message_dict["role"],
                                       message_dict["content"],
                                       message_dict["timestamp"],
                                       datetime.utcnow(),
                                       message_dict.get("response_time"),
                                       message_dict.get("model_used"),
                                       message_dict.get("tokens_used"),
                                       json.dumps(message_dict.get("sources")) if message_dict.get("sources") else None,
                                       message_dict.get("cypher_query"),
                                       json.dumps(message_dict.get("linked_entities")) if message_dict.get(
                                           "linked_entities") else None,
                                       json.dumps(message_dict.get("info")) if message_dict.get("info") else None,
                                       message_dict.get("error_info"),
                                       message_dict.get("user_rating"),
                                       message_dict.get("user_feedback")
                                       )

                # Update conversation
                await conn.execute(
                    "UPDATE conversations SET message_count = $1, updated_at = $2 WHERE id = $3",
                    len(messages), datetime.utcnow(), conversation_id
                )

            logger.info(f"💾 Saved {len(messages)} messages to database")
            return True

        except Exception as e:
            logger.error(f"💥 Error saving messages: {e}")
            return False

    async def _async_get_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversations async implementation"""
        try:
            query = """
            SELECT id, title, created_at, updated_at, message_count
            FROM conversations 
            WHERE user_id = $1 AND is_archived = FALSE
            ORDER BY updated_at DESC
            LIMIT 50
            """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, user_id)

                conversations = []
                for row in rows:
                    conversations.append({
                        "id": str(row['id']),
                        "title": row['title'],
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat(),
                        "message_count": row['message_count']
                    })

                return conversations

        except Exception as e:
            logger.error(f"💥 Error getting conversations: {e}")
            return []

    async def _ensure_user(self, user_id: str):
        """Ensure user exists"""
        try:
            async with self._pool.acquire() as conn:
                # Check if user exists
                result = await conn.fetchval("SELECT id FROM users WHERE username = $1", user_id)

                if not result:
                    # Create user
                    await conn.execute(
                        "INSERT INTO users (username, email) VALUES ($1, $2) ON CONFLICT (username) DO NOTHING",
                        user_id, f"{user_id}@example.com"
                    )
                    logger.info(f"➕ Created user: {user_id}")
                else:
                    # Update last active
                    await conn.execute(
                        "UPDATE users SET last_active = $1 WHERE username = $2",
                        datetime.utcnow(), user_id
                    )

        except Exception as e:
            logger.warning(f"Could not ensure user {user_id}: {e}")

    async def _get_or_create_default_conversation(self, user_id: str) -> str:
        """Get or create default conversation"""
        try:
            async with self._pool.acquire() as conn:
                # Try to get existing conversation
                result = await conn.fetchval(
                    "SELECT id FROM conversations WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 1",
                    user_id
                )

                if result:
                    return str(result)
                else:
                    # Create new conversation
                    conversation_id = str(uuid.uuid4())
                    title = f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                    await conn.execute("""
                        INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5)
                    """, conversation_id, user_id, title, datetime.utcnow(), datetime.utcnow())

                    logger.info(f"➕ Created conversation: {conversation_id}")
                    return conversation_id

        except Exception as e:
            logger.error(f"Error getting/creating conversation: {e}")
            raise

    def close(self):
        """Close database connections"""
        if self._loop and self._pool:
            try:
                future = self._submit_async_task(self._pool.close())
                future.result(timeout=5)
            except:
                pass

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._executor:
            self._executor.shutdown(wait=True)


# ==========================================================================
# SIMPLE CHAT SERVICE USING SYNC DATABASE
# ==========================================================================

class SyncChatService:
    """Simple chat service using sync database manager"""

    def __init__(self, db_manager: SyncDatabaseManager):
        self.db = db_manager

    def load_chat_history(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Load chat history"""
        return self.db.load_messages(user_id)

    def save_chat_history(self, messages: List[Dict[str, Any]], user_id: str = "default_user") -> bool:
        """Save chat history"""
        return self.db.save_messages(messages, user_id)

    def add_message(self, message_dict: Dict[str, Any], user_id: str = "default_user") -> str:
        """Add single message"""
        return self.db.save_message(message_dict, user_id)

    def get_conversations(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Get conversations"""
        return self.db.get_conversations(user_id)


# ==========================================================================
# FACTORY FUNCTION
# ==========================================================================

def create_sync_chat_service(config: Dict[str, Any]) -> SyncChatService:
    """Create sync chat service"""
    db_manager = SyncDatabaseManager(config)

    if not db_manager.initialize():
        raise RuntimeError("Failed to initialize sync database manager")

    return SyncChatService(db_manager)
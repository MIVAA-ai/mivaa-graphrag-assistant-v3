# src/chat/sync_database.py - COMPLETE VERSION with Conversation Management
import logging
import uuid
import json
import psycopg2
import psycopg2.extras
import psycopg2.pool
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class SyncDatabaseManager:
    """Pure synchronous database manager using psycopg2"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._pool = None
        self._initialized = False
        self._connection_params = None

    def initialize(self) -> bool:
        """Initialize the database connection - PURE SYNC VERSION"""
        try:
            logger.info("🔄 Initializing sync database manager...")

            # Build connection parameters
            self._connection_params = self._build_connection_params()

            # Create connection pool
            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self._connection_params['host'],
                port=self._connection_params['port'],
                database=self._connection_params['database'],
                user=self._connection_params['user'],
                password=self._connection_params['password']
            )

            # Test connection and create schema
            self._ensure_schema()

            self._initialized = True
            logger.info("✅ Sync database manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Sync database initialization failed: {e}")
            return False

    def _build_connection_params(self) -> Dict[str, Any]:
        """Build connection parameters from config"""
        db_config = self.config.get('database', {})

        connection_params = {
            'host': db_config.get('host', 'localhost'),
            'port': db_config.get('port', 5432),
            'database': db_config.get('database', 'graphrag_chat'),
            'user': db_config.get('user', 'postgres'),
            'password': db_config.get('password', 'password')
        }

        # Override with environment variables
        connection_params.update({
            'host': os.getenv('DB_HOST', connection_params['host']),
            'port': int(os.getenv('DB_PORT', connection_params['port'])),
            'database': os.getenv('DB_NAME', connection_params['database']),
            'user': os.getenv('DB_USER', connection_params['user']),
            'password': os.getenv('DB_PASSWORD', connection_params['password'])
        })

        return connection_params

    def _get_connection(self):
        """Get a connection from the pool"""
        if not self._pool:
            raise RuntimeError("Database not initialized")
        return self._pool.getconn()

    def _put_connection(self, conn):
        """Return a connection to the pool"""
        if self._pool:
            self._pool.putconn(conn)

    def _ensure_schema(self):
        """Create database schema"""
        schema_sql = """
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";

        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            username VARCHAR(100) UNIQUE NOT NULL,
            email VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            preferences JSONB DEFAULT '{}'
        );

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

        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            timestamp VARCHAR(50) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            response_time FLOAT,
            model_used VARCHAR(100),
            tokens_used INTEGER,
            sources JSONB,
            cypher_query TEXT,
            linked_entities JSONB,
            info JSONB,
            user_rating INTEGER,
            user_feedback TEXT,
            error_info TEXT,
            processing_status VARCHAR(20) DEFAULT 'completed'
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_conversations_title_search ON conversations USING gin (title gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_messages_content_search ON messages USING gin (content gin_trgm_ops);

        INSERT INTO users (username, email) VALUES ('default_user', 'default@example.com') ON CONFLICT (username) DO NOTHING;
        """

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
                conn.commit()
                logger.info("✅ Database schema ensured")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create schema: {e}")
            raise
        finally:
            self._put_connection(conn)

    # ============================================================================
    # CONVERSATION MANAGEMENT METHODS
    # ============================================================================

    def get_conversations(self, user_id: str = "default_user", limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations for user from database"""
        if not self._initialized:
            logger.warning("Database not initialized, returning empty list")
            return []

        try:
            query = """
            SELECT c.id, c.title, c.created_at, c.updated_at, c.message_count, c.is_archived,
                   m.content as last_message_preview
            FROM conversations c
            LEFT JOIN LATERAL (
                SELECT content 
                FROM messages 
                WHERE conversation_id = c.id 
                ORDER BY created_at DESC 
                LIMIT 1
            ) m ON true
            WHERE c.user_id = %s AND c.is_archived = FALSE
            ORDER BY c.updated_at DESC
            LIMIT %s
            """

            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, (user_id, limit))
                    rows = cur.fetchall()

                    conversations = []
                    for row in rows:
                        conv = {
                            "id": str(row['id']),
                            "title": row['title'],
                            "created_at": row['created_at'].isoformat(),
                            "updated_at": row['updated_at'].isoformat(),
                            "message_count": row['message_count'],
                            "is_archived": row['is_archived'],
                            "last_message_preview": row['last_message_preview'][:100] + "..." if row[
                                                                                                     'last_message_preview'] and len(
                                row['last_message_preview']) > 100 else row['last_message_preview']
                        }
                        conversations.append(conv)

                    logger.info(f"📚 Retrieved {len(conversations)} conversations for user {user_id}")
                    return conversations

            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error getting conversations: {e}")
            return []

    def create_conversation(self, title: str, user_id: str = "default_user") -> str:
        """Create a new conversation"""
        if not self._initialized:
            logger.warning("Database not initialized, cannot create conversation")
            return ""

        try:
            self._ensure_user(user_id)
            conversation_id = str(uuid.uuid4())

            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (conversation_id, user_id, title, datetime.utcnow(), datetime.utcnow()))

                    conn.commit()
                    logger.info(f"➕ Created conversation: {conversation_id}")
                    return conversation_id

            except Exception as e:
                conn.rollback()
                raise
            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error creating conversation: {e}")
            raise

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        if not self._initialized:
            logger.warning("Database not initialized, cannot delete conversation")
            return False

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    # Delete conversation (messages will be cascade deleted)
                    cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected > 0:
                        logger.info(f"🗑️ Deleted conversation: {conversation_id}")
                        return True
                    else:
                        logger.warning(f"⚠️ Conversation not found: {conversation_id}")
                        return False

            except Exception as e:
                conn.rollback()
                raise
            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error deleting conversation: {e}")
            return False

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title"""
        if not self._initialized:
            logger.warning("Database not initialized, cannot update conversation")
            return False

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE conversations 
                        SET title = %s, updated_at = %s 
                        WHERE id = %s
                    """, (title, datetime.utcnow(), conversation_id))

                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected > 0:
                        logger.info(f"✏️ Updated conversation title: {conversation_id}")
                        return True
                    else:
                        logger.warning(f"⚠️ Conversation not found: {conversation_id}")
                        return False

            except Exception as e:
                conn.rollback()
                raise
            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error updating conversation title: {e}")
            return False

    # ============================================================================
    # SEARCH METHODS
    # ============================================================================

    def search_conversations(self, user_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search conversations by title and content"""
        if not self._initialized:
            logger.warning("Database not initialized, returning empty search results")
            return []

        try:
            search_query = """
            SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, c.message_count,
                   ts_rank(to_tsvector('english', c.title), plainto_tsquery('english', %s)) as title_rank,
                   CASE 
                       WHEN c.title ILIKE %s THEN 1.0
                       ELSE 0.0
                   END as exact_match
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.user_id = %s 
            AND c.is_archived = FALSE
            AND (
                c.title ILIKE %s 
                OR m.content ILIKE %s
                OR to_tsvector('english', c.title) @@ plainto_tsquery('english', %s)
                OR to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
            )
            ORDER BY exact_match DESC, title_rank DESC, c.updated_at DESC
            LIMIT %s
            """

            like_query = f"%{query}%"

            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(search_query, (
                        query, like_query, user_id, like_query, like_query, query, query, limit
                    ))
                    rows = cur.fetchall()

                    conversations = []
                    for row in rows:
                        conv = {
                            "id": str(row['id']),
                            "title": row['title'],
                            "created_at": row['created_at'].isoformat(),
                            "updated_at": row['updated_at'].isoformat(),
                            "message_count": row['message_count'],
                            "relevance_score": float(row.get('title_rank', 0))
                        }
                        conversations.append(conv)

                    logger.info(f"🔍 Found {len(conversations)} conversations matching '{query}'")
                    return conversations

            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error searching conversations: {e}")
            return []

    def search_messages(self, user_id: str, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search messages by content"""
        if not self._initialized:
            logger.warning("Database not initialized, returning empty search results")
            return []

        try:
            search_query = """
            SELECT m.id, m.conversation_id, m.role, m.content, m.created_at,
                   c.title as conversation_title,
                   ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', %s)) as content_rank
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = %s
            AND c.is_archived = FALSE
            AND (
                m.content ILIKE %s
                OR to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
            )
            ORDER BY content_rank DESC, m.created_at DESC
            LIMIT %s
            """

            like_query = f"%{query}%"

            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(search_query, (query, user_id, like_query, query, limit))
                    rows = cur.fetchall()

                    messages = []
                    for row in rows:
                        message = {
                            "id": str(row['id']),
                            "conversation_id": str(row['conversation_id']),
                            "role": row['role'],
                            "content": row['content'],
                            "created_at": row['created_at'].isoformat(),
                            "conversation_title": row['conversation_title'],
                            "relevance_score": float(row.get('content_rank', 0)),
                            "highlight_snippet": self._create_highlight_snippet(row['content'], query)
                        }
                        messages.append(message)

                    logger.info(f"🔍 Found {len(messages)} messages matching '{query}'")
                    return messages

            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error searching messages: {e}")
            return []

    def _create_highlight_snippet(self, content: str, query: str, context_length: int = 100) -> str:
        """Create a highlighted snippet around the search term"""
        try:
            content_lower = content.lower()
            query_lower = query.lower()

            index = content_lower.find(query_lower)
            if index == -1:
                return content[:context_length] + "..." if len(content) > context_length else content

            start = max(0, index - context_length // 2)
            end = min(len(content), index + len(query) + context_length // 2)

            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."

            return snippet
        except Exception:
            return content[:context_length] + "..." if len(content) > context_length else content

    # ============================================================================
    # EXISTING MESSAGE METHODS (UNCHANGED)
    # ============================================================================

    def load_messages(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Load messages synchronously"""
        if not self._initialized:
            logger.warning("Database not initialized, returning empty list")
            return []

        try:
            self._ensure_user(user_id)
            conversation_id = self._get_or_create_default_conversation(user_id)

            query = """
            SELECT role, content, timestamp, response_time, model_used, tokens_used,
                   sources, cypher_query, linked_entities, info, error_info
            FROM messages WHERE conversation_id = %s ORDER BY created_at ASC
            """

            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, (conversation_id,))
                    rows = cur.fetchall()

                    messages = []
                    for row in rows:
                        message = {
                            "role": row['role'],
                            "content": row['content'],
                            "timestamp": row['timestamp']
                        }

                        # Add optional fields only if they exist
                        if row['response_time']:
                            message["response_time"] = row['response_time']
                        if row['model_used']:
                            message["model_used"] = row['model_used']
                        if row['sources']:
                            message["sources"] = row['sources']
                        if row['cypher_query']:
                            message["cypher_query"] = row['cypher_query']
                        if row['info']:
                            message["info"] = row['info']
                        if row['error_info']:
                            message["error_info"] = row['error_info']

                        messages.append(message)

                    logger.info(f"📚 Loaded {len(messages)} messages from sync database")
                    return messages

            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error loading messages: {e}")
            return []

    def save_messages(self, messages: List[Dict[str, Any]], user_id: str = "default_user") -> bool:
        """Save messages synchronously"""
        if not self._initialized:
            logger.warning("Database not initialized, cannot save messages")
            return False

        try:
            self._ensure_user(user_id)
            conversation_id = self._get_or_create_default_conversation(user_id)

            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    # Clear existing messages for this conversation
                    cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))

                    # Insert all messages
                    for message_dict in messages:
                        message_id = str(uuid.uuid4())
                        cur.execute("""
                            INSERT INTO messages (
                                id, conversation_id, role, content, timestamp, created_at,
                                response_time, model_used, tokens_used, sources, cypher_query,
                                linked_entities, info, error_info
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            message_id, conversation_id, message_dict["role"], message_dict["content"],
                            message_dict["timestamp"], datetime.utcnow(),
                            message_dict.get("response_time"), message_dict.get("model_used"),
                            message_dict.get("tokens_used"),
                            json.dumps(message_dict.get("sources")) if message_dict.get("sources") else None,
                            message_dict.get("cypher_query"),
                            json.dumps(message_dict.get("linked_entities")) if message_dict.get(
                                "linked_entities") else None,
                            json.dumps(message_dict.get("info")) if message_dict.get("info") else None,
                            message_dict.get("error_info")
                        ))

                    # Update conversation
                    cur.execute(
                        "UPDATE conversations SET message_count = %s, updated_at = %s WHERE id = %s",
                        (len(messages), datetime.utcnow(), conversation_id)
                    )

                    conn.commit()

                logger.info(f"💾 Saved {len(messages)} messages to sync database")
                return True

            except Exception as e:
                conn.rollback()
                raise
            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error saving messages: {e}")
            return False

    def save_message(self, message_dict: Dict[str, Any], user_id: str = "default_user") -> str:
        """Save single message synchronously"""
        if not self._initialized:
            logger.warning("Database not initialized, cannot save message")
            return ""

        try:
            self._ensure_user(user_id)
            conversation_id = self._get_or_create_default_conversation(user_id)

            message_id = str(uuid.uuid4())

            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO messages (
                            id, conversation_id, role, content, timestamp, created_at,
                            response_time, model_used, error_info
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        message_id, conversation_id, message_dict["role"], message_dict["content"],
                        message_dict["timestamp"], datetime.utcnow(),
                        message_dict.get("response_time"), message_dict.get("model_used"),
                        message_dict.get("error_info")
                    ))

                    cur.execute(
                        "UPDATE conversations SET message_count = message_count + 1, updated_at = %s WHERE id = %s",
                        (datetime.utcnow(), conversation_id)
                    )

                    conn.commit()

                logger.debug(f"➕ Added message to sync database")
                return conversation_id

            except Exception as e:
                conn.rollback()
                raise
            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"💥 Error saving message: {e}")
            raise

    def _ensure_user(self, user_id: str):
        """Ensure user exists"""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM users WHERE username = %s", (user_id,))
                    result = cur.fetchone()

                    if not result:
                        cur.execute(
                            "INSERT INTO users (username, email) VALUES (%s, %s) ON CONFLICT (username) DO NOTHING",
                            (user_id, f"{user_id}@example.com")
                        )
                        conn.commit()
                        logger.info(f"➕ Created user: {user_id}")

            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.warning(f"Could not ensure user {user_id}: {e}")

    def _get_or_create_default_conversation(self, user_id: str) -> str:
        """Get or create default conversation"""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM conversations WHERE user_id = %s ORDER BY updated_at DESC LIMIT 1",
                        (user_id,)
                    )
                    result = cur.fetchone()

                    if result:
                        return str(result[0])
                    else:
                        conversation_id = str(uuid.uuid4())
                        title = f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                        cur.execute("""
                            INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (conversation_id, user_id, title, datetime.utcnow(), datetime.utcnow()))

                        conn.commit()
                        logger.info(f"➕ Created conversation: {conversation_id}")
                        return conversation_id

            finally:
                self._put_connection(conn)

        except Exception as e:
            logger.error(f"Error getting/creating conversation: {e}")
            raise

    def close(self):
        """Close the database connection pool"""
        try:
            if self._pool:
                self._pool.closeall()
                logger.info("🔒 Database connection pool closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._initialized


class SyncChatService:
    """Enhanced chat service with conversation management and search"""

    def __init__(self, db_manager: SyncDatabaseManager):
        self.db = db_manager

    # ============================================================================
    # EXISTING METHODS (UNCHANGED)
    # ============================================================================

    def load_chat_history(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        return self.db.load_messages(user_id)

    def save_chat_history(self, messages: List[Dict[str, Any]], user_id: str = "default_user") -> bool:
        return self.db.save_messages(messages, user_id)

    def add_message(self, message_dict: Dict[str, Any], user_id: str = "default_user") -> str:
        return self.db.save_message(message_dict, user_id)

    # ============================================================================
    # NEW CONVERSATION MANAGEMENT METHODS
    # ============================================================================

    def get_conversations(self, user_id: str = "default_user", limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations for user"""
        return self.db.get_conversations(user_id, limit)

    def create_conversation(self, title: str, user_id: str = "default_user") -> str:
        """Create a new conversation"""
        return self.db.create_conversation(title, user_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        return self.db.delete_conversation(conversation_id)

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title"""
        return self.db.update_conversation_title(conversation_id, title)

    # ============================================================================
    # NEW SEARCH METHODS
    # ============================================================================

    def search_conversations(self, query: str, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Search conversations"""
        return self.db.search_conversations(user_id, query)

    def search_messages(self, query: str, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Search messages"""
        return self.db.search_messages(user_id, query)


def create_sync_chat_service(config: Dict[str, Any]) -> SyncChatService:
    """Create sync chat service"""
    db_manager = SyncDatabaseManager(config)

    if not db_manager.initialize():
        raise RuntimeError("Failed to initialize sync database manager")

    return SyncChatService(db_manager)
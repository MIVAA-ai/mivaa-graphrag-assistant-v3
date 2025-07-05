# src/chat/sync_database.py - PURE SYNC VERSION using psycopg2
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
    """Simple chat service using sync database"""

    def __init__(self, db_manager: SyncDatabaseManager):
        self.db = db_manager

    def load_chat_history(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        return self.db.load_messages(user_id)

    def save_chat_history(self, messages: List[Dict[str, Any]], user_id: str = "default_user") -> bool:
        return self.db.save_messages(messages, user_id)

    def add_message(self, message_dict: Dict[str, Any], user_id: str = "default_user") -> str:
        return self.db.save_message(message_dict, user_id)


def create_sync_chat_service(config: Dict[str, Any]) -> SyncChatService:
    """Create sync chat service"""
    db_manager = SyncDatabaseManager(config)

    if not db_manager.initialize():
        raise RuntimeError("Failed to initialize sync database manager")

    return SyncChatService(db_manager)
# src/chat/database.py
import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async PostgreSQL database manager for chat system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize database connection pool and create schema if needed"""
        try:
            # Get database configuration
            db_config = self.config.get('database', {})

            # Build connection string
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

            # Override with environment variables if available
            if os.getenv('DATABASE_URL'):
                # Parse DATABASE_URL if provided
                import urllib.parse as urlparse
                url = urlparse.urlparse(os.getenv('DATABASE_URL'))
                connection_params.update({
                    'host': url.hostname,
                    'port': url.port or 5432,
                    'database': url.path[1:],  # Remove leading slash
                    'user': url.username,
                    'password': url.password
                })
            else:
                # Individual environment variables
                connection_params.update({
                    'host': os.getenv('DB_HOST', connection_params['host']),
                    'port': int(os.getenv('DB_PORT', connection_params['port'])),
                    'database': os.getenv('DB_NAME', connection_params['database']),
                    'user': os.getenv('DB_USER', connection_params['user']),
                    'password': os.getenv('DB_PASSWORD', connection_params['password'])
                })

            logger.info(
                f"Connecting to PostgreSQL: {connection_params['user']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}")

            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=connection_params['host'],
                port=connection_params['port'],
                database=connection_params['database'],
                user=connection_params['user'],
                password=connection_params['password'],
                min_size=connection_params['min_size'],
                max_size=connection_params['max_size'],
                command_timeout=connection_params['command_timeout']
            )

            # Test connection and create schema
            await self._ensure_schema()

            self._initialized = True
            logger.info("✅ Database connection pool initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
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
            timestamp VARCHAR(50) NOT NULL,  -- Store as string to match current format
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

        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated 
            ON conversations(user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_conversations_user_created 
            ON conversations(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created 
            ON messages(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_messages_role 
            ON messages(role);

        -- Full-text search indexes (for Phase 2)
        CREATE INDEX IF NOT EXISTS idx_conversations_title_search 
            ON conversations USING gin (title gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_messages_content_search 
            ON messages USING gin (content gin_trgm_ops);

        -- Create default user if not exists
        INSERT INTO users (username, email) 
        VALUES ('default_user', 'default@example.com')
        ON CONFLICT (username) DO NOTHING;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)
            logger.info("✅ Database schema ensured")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        if not self._initialized or not self.pool:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self.pool.acquire() as conn:
            yield conn

    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._initialized and self.pool is not None


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Dict[str, Any]) -> DatabaseManager:
    """Get or create global database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
    return _db_manager


async def initialize_database(config: Dict[str, Any]) -> bool:
    """Initialize the global database manager"""
    db_manager = get_database_manager(config)
    return await db_manager.initialize()


async def close_database():
    """Close the global database manager"""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


# Utility functions for sync compatibility with Streamlit
def run_async(coro):
    """Run async function in sync context (for Streamlit compatibility)"""
    try:
        # Try to get current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use a different approach
            import concurrent.futures
            import threading

            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            # If no loop is running, we can run directly
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)

# src/chat/repositories.py
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import uuid

from .models import ChatMessage, Conversation, User, MessageRole
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for conversation data access"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create(self, title: str, user_id: str = "default_user", metadata: Dict[str, Any] = None) -> Conversation:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        query = """
        INSERT INTO conversations (id, user_id, title, created_at, updated_at, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING *
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(
                query,
                conversation_id,
                user_id,
                title,
                now,
                now,
                json.dumps(metadata or {})
            )

            return Conversation(
                id=str(row['id']),
                user_id=row['user_id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                is_archived=row['is_archived'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                message_count=row['message_count']
            )

    async def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        query = """
        SELECT * FROM conversations WHERE id = $1
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, conversation_id)

            if not row:
                return None

            return Conversation(
                id=str(row['id']),
                user_id=row['user_id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                is_archived=row['is_archived'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                message_count=row['message_count']
            )

    async def get_by_user(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Conversation]:
        """Get conversations for a user"""
        query = """
        SELECT c.*, 
               m.content as last_message_preview
        FROM conversations c
        LEFT JOIN LATERAL (
            SELECT content 
            FROM messages 
            WHERE conversation_id = c.id 
            ORDER BY created_at DESC 
            LIMIT 1
        ) m ON true
        WHERE c.user_id = $1 AND c.is_archived = FALSE
        ORDER BY c.updated_at DESC
        LIMIT $2 OFFSET $3
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, user_id, limit, offset)

            conversations = []
            for row in rows:
                conv = Conversation(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    title=row['title'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    is_archived=row['is_archived'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    message_count=row['message_count']
                )

                # Add preview
                if row['last_message_preview']:
                    conv.last_message_preview = row['last_message_preview'][:100] + "..." if len(
                        row['last_message_preview']) > 100 else row['last_message_preview']

                conversations.append(conv)

            return conversations

    async def update(self, conversation_id: str, **updates) -> Optional[Conversation]:
        """Update conversation"""
        if not updates:
            return await self.get_by_id(conversation_id)

        # Build dynamic update query
        set_clauses = []
        params = []
        param_idx = 1

        for key, value in updates.items():
            if key in ['title', 'is_archived', 'metadata']:
                set_clauses.append(f"{key} = ${param_idx}")
                if key == 'metadata':
                    params.append(json.dumps(value))
                else:
                    params.append(value)
                param_idx += 1

        if not set_clauses:
            return await self.get_by_id(conversation_id)

        # Always update updated_at
        set_clauses.append(f"updated_at = ${param_idx}")
        params.append(datetime.utcnow())
        param_idx += 1

        params.append(conversation_id)

        query = f"""
        UPDATE conversations 
        SET {', '.join(set_clauses)}
        WHERE id = ${param_idx}
        RETURNING *
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, *params)

            if not row:
                return None

            return Conversation(
                id=str(row['id']),
                user_id=row['user_id'],
                title=row['title'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                is_archived=row['is_archived'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                message_count=row['message_count']
            )

    async def delete(self, conversation_id: str) -> bool:
        """Delete conversation (cascade deletes messages)"""
        query = "DELETE FROM conversations WHERE id = $1"

        async with self.db.get_connection() as conn:
            result = await conn.execute(query, conversation_id)
            return result == "DELETE 1"

    async def increment_message_count(self, conversation_id: str) -> bool:
        """Increment message count and update timestamp"""
        query = """
        UPDATE conversations 
        SET message_count = message_count + 1, updated_at = $1
        WHERE id = $2
        """

        async with self.db.get_connection() as conn:
            result = await conn.execute(query, datetime.utcnow(), conversation_id)
            return result == "UPDATE 1"

    async def search(self, user_id: str, query: str, limit: int = 10) -> List[Conversation]:
        """Search conversations by title and content (Phase 2)"""
        search_query = """
        SELECT DISTINCT c.*, 
               ts_rank(to_tsvector('english', c.title), plainto_tsquery('english', $2)) as rank
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.user_id = $1 
        AND c.is_archived = FALSE
        AND (
            c.title ILIKE '%' || $2 || '%' 
            OR m.content ILIKE '%' || $2 || '%'
        )
        ORDER BY rank DESC, c.updated_at DESC
        LIMIT $3
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(search_query, user_id, query, limit)

            conversations = []
            for row in rows:
                conv = Conversation(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    title=row['title'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    is_archived=row['is_archived'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    message_count=row['message_count']
                )
                conversations.append(conv)

            return conversations


class MessageRepository:
    """Repository for message data access"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create(self, message: ChatMessage) -> ChatMessage:
        """Create a new message"""
        query = """
        INSERT INTO messages (
            id, conversation_id, role, content, timestamp, created_at,
            response_time, model_used, tokens_used, sources, cypher_query,
            linked_entities, info, user_rating, user_feedback, error_info,
            processing_status
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
        ) RETURNING *
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(
                query,
                message.id,
                message.conversation_id,
                message.role.value,
                message.content,
                message.timestamp,
                message.created_at,
                message.response_time,
                message.model_used,
                message.tokens_used,
                json.dumps(message.sources) if message.sources else None,
                message.cypher_query,
                json.dumps(message.linked_entities) if message.linked_entities else None,
                json.dumps(message.info) if message.info else None,
                message.user_rating,
                message.user_feedback,
                message.error_info,
                message.processing_status.value
            )

            return self._row_to_message(row)

    async def get_by_conversation(self, conversation_id: str, limit: int = 100, offset: int = 0) -> List[ChatMessage]:
        """Get messages for a conversation"""
        query = """
        SELECT * FROM messages 
        WHERE conversation_id = $1 
        ORDER BY created_at ASC
        LIMIT $2 OFFSET $3
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, conversation_id, limit, offset)
            return [self._row_to_message(row) for row in rows]

    async def get_by_id(self, message_id: str) -> Optional[ChatMessage]:
        """Get message by ID"""
        query = "SELECT * FROM messages WHERE id = $1"

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, message_id)
            return self._row_to_message(row) if row else None

    async def update(self, message_id: str, **updates) -> Optional[ChatMessage]:
        """Update message"""
        if not updates:
            return await self.get_by_id(message_id)

        # Build dynamic update query
        set_clauses = []
        params = []
        param_idx = 1

        for key, value in updates.items():
            if key in ['content', 'user_rating', 'user_feedback', 'error_info', 'processing_status']:
                set_clauses.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not set_clauses:
            return await self.get_by_id(message_id)

        params.append(message_id)

        query = f"""
        UPDATE messages 
        SET {', '.join(set_clauses)}
        WHERE id = ${param_idx}
        RETURNING *
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, *params)
            return self._row_to_message(row) if row else None

    async def delete(self, message_id: str) -> bool:
        """Delete message"""
        query = "DELETE FROM messages WHERE id = $1"

        async with self.db.get_connection() as conn:
            result = await conn.execute(query, message_id)
            return result == "DELETE 1"

    async def search_content(self, query: str, user_id: str = None, limit: int = 20) -> List[ChatMessage]:
        """Search messages by content (Phase 2)"""
        if user_id:
            search_query = """
            SELECT m.*, 
                   ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) as rank
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = $2
            AND c.is_archived = FALSE
            AND m.content ILIKE '%' || $1 || '%'
            ORDER BY rank DESC, m.created_at DESC
            LIMIT $3
            """
            params = [query, user_id, limit]
        else:
            search_query = """
            SELECT m.*, 
                   ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) as rank
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.is_archived = FALSE
            AND m.content ILIKE '%' || $1 || '%'
            ORDER BY rank DESC, m.created_at DESC
            LIMIT $2
            """
            params = [query, limit]

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(search_query, *params)
            return [self._row_to_message(row) for row in rows]

    def _row_to_message(self, row) -> ChatMessage:
        """Convert database row to message model"""
        return ChatMessage(
            id=str(row['id']),
            conversation_id=str(row['conversation_id']),
            role=MessageRole(row['role']),
            content=row['content'],
            timestamp=row['timestamp'],
            created_at=row['created_at'],
            response_time=row['response_time'],
            model_used=row['model_used'],
            tokens_used=row['tokens_used'],
            sources=json.loads(row['sources']) if row['sources'] else None,
            cypher_query=row['cypher_query'],
            linked_entities=json.loads(row['linked_entities']) if row['linked_entities'] else None,
            info=json.loads(row['info']) if row['info'] else None,
            user_rating=row['user_rating'],
            user_feedback=row['user_feedback'],
            error_info=row['error_info'],
            processing_status=row['processing_status']
        )


class UserRepository:
    """Repository for user data access"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create(self, username: str, email: str = None) -> User:
        """Create a new user"""
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()

        query = """
        INSERT INTO users (id, username, email, created_at, last_active)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, user_id, username, email, now, now)

            return User(
                id=str(row['id']),
                username=row['username'],
                email=row['email'],
                created_at=row['created_at'],
                last_active=row['last_active'],
                preferences=json.loads(row['preferences']) if row['preferences'] else {}
            )

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = $1"

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, username)

            if not row:
                return None

            return User(
                id=str(row['id']),
                username=row['username'],
                email=row['email'],
                created_at=row['created_at'],
                last_active=row['last_active'],
                preferences=json.loads(row['preferences']) if row['preferences'] else {}
            )

    async def update_last_active(self, username: str) -> bool:
        """Update user's last active timestamp"""
        query = "UPDATE users SET last_active = $1 WHERE username = $2"

        async with self.db.get_connection() as conn:
            result = await conn.execute(query, datetime.utcnow(), username)
            return result == "UPDATE 1"
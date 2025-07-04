# src/chat/services.py
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from .models import ChatMessage, Conversation, User, MessageRole, message_to_dict, dict_to_message
from .repositories import ConversationRepository, MessageRepository, UserRepository
from .database import DatabaseManager, run_async

logger = logging.getLogger(__name__)


class ChatService:
    """High-level chat service that provides a clean interface for the Streamlit app"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.conversation_repo = ConversationRepository(db_manager)
        self.message_repo = MessageRepository(db_manager)
        self.user_repo = UserRepository(db_manager)

    # ==========================================================================
    # SYNC INTERFACE FOR STREAMLIT COMPATIBILITY
    # ==========================================================================

    def load_chat_history(self, user_id: str = "default_user", conversation_id: str = None) -> List[Dict[str, Any]]:
        """
        Load chat history in the same format as the current JSON system.

        Args:
            user_id: User ID (defaults to "default_user")
            conversation_id: Specific conversation ID, or None for latest

        Returns:
            List of message dictionaries in the same format as current JSON system
        """
        return run_async(self._load_chat_history_async(user_id, conversation_id))

    def save_chat_history(self, messages: List[Dict[str, Any]], user_id: str = "default_user",
                          conversation_id: str = None) -> str:
        """
        Save chat history from the current JSON format.

        Args:
            messages: List of message dictionaries in current JSON format
            user_id: User ID
            conversation_id: Conversation ID, or None to create/use default

        Returns:
            Conversation ID
        """
        return run_async(self._save_chat_history_async(messages, user_id, conversation_id))

    def add_message(self, message_dict: Dict[str, Any], user_id: str = "default_user",
                    conversation_id: str = None) -> str:
        """
        Add a single message to the chat history.

        Args:
            message_dict: Message in current JSON format
            user_id: User ID
            conversation_id: Conversation ID, or None to use/create default

        Returns:
            Conversation ID
        """
        return run_async(self._add_message_async(message_dict, user_id, conversation_id))

    def get_conversations(self, user_id: str = "default_user", limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of conversations for a user"""
        return run_async(self._get_conversations_async(user_id, limit))

    def create_conversation(self, title: str, user_id: str = "default_user") -> str:
        """Create a new conversation and return its ID"""
        return run_async(self._create_conversation_async(title, user_id))

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title"""
        return run_async(self._update_conversation_title_async(conversation_id, title))

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        return run_async(self._delete_conversation_async(conversation_id))

    def search_conversations(self, query: str, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Search conversations (Phase 2)"""
        return run_async(self._search_conversations_async(query, user_id))

    def add_message_feedback(self, message_id: str, rating: int, feedback: str = None) -> bool:
        """Add feedback to a message (Phase 2)"""
        return run_async(self._add_message_feedback_async(message_id, rating, feedback))

    # ==========================================================================
    # ASYNC IMPLEMENTATION
    # ==========================================================================

    async def _load_chat_history_async(self, user_id: str, conversation_id: str = None) -> List[Dict[str, Any]]:
        """Load chat history async implementation"""
        try:
            # If no conversation_id provided, get the latest conversation
            if conversation_id is None:
                conversations = await self.conversation_repo.get_by_user(user_id, limit=1)
                if not conversations:
                    # No conversations exist, return empty
                    return []
                conversation_id = conversations[0].id

            # Load messages for the conversation
            messages = await self.message_repo.get_by_conversation(conversation_id)

            # Convert to JSON format
            result = [message_to_dict(message) for message in messages]

            logger.info(f"📚 Loaded {len(result)} messages from conversation {conversation_id}")
            return result

        except Exception as e:
            logger.error(f"💥 Error loading chat history: {e}")
            return []

    async def _save_chat_history_async(self, messages: List[Dict[str, Any]], user_id: str,
                                       conversation_id: str = None) -> str:
        """Save chat history async implementation"""
        try:
            # Ensure user exists
            await self._ensure_user(user_id)

            # If no conversation_id, create or get default conversation
            if conversation_id is None:
                conversation_id = await self._get_or_create_default_conversation(user_id)

            # Clear existing messages for this conversation
            existing_messages = await self.message_repo.get_by_conversation(conversation_id)
            for msg in existing_messages:
                await self.message_repo.delete(msg.id)

            # Add all messages
            for msg_dict in messages:
                message = dict_to_message(msg_dict, conversation_id)
                await self.message_repo.create(message)

            # Update conversation message count
            await self.conversation_repo.update(conversation_id, message_count=len(messages))

            logger.info(f"💾 Saved {len(messages)} messages to conversation {conversation_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"💥 Error saving chat history: {e}")
            raise

    async def _add_message_async(self, message_dict: Dict[str, Any], user_id: str, conversation_id: str = None) -> str:
        """Add single message async implementation"""
        try:
            # Ensure user exists
            await self._ensure_user(user_id)

            # If no conversation_id, create or get default conversation
            if conversation_id is None:
                conversation_id = await self._get_or_create_default_conversation(user_id)

            # Convert and save message
            message = dict_to_message(message_dict, conversation_id)
            await self.message_repo.create(message)

            # Update conversation
            await self.conversation_repo.increment_message_count(conversation_id)

            logger.debug(f"➕ Added message to conversation {conversation_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"💥 Error adding message: {e}")
            raise

    async def _get_conversations_async(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversations async implementation"""
        try:
            conversations = await self.conversation_repo.get_by_user(user_id, limit=limit)

            result = []
            for conv in conversations:
                result.append({
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "message_count": conv.message_count,
                    "last_message_preview": conv.last_message_preview,
                    "is_archived": conv.is_archived
                })

            return result

        except Exception as e:
            logger.error(f"💥 Error getting conversations: {e}")
            return []

    async def _create_conversation_async(self, title: str, user_id: str) -> str:
        """Create conversation async implementation"""
        try:
            await self._ensure_user(user_id)
            conversation = await self.conversation_repo.create(title, user_id)
            logger.info(f"➕ Created conversation: {conversation.id}")
            return conversation.id

        except Exception as e:
            logger.error(f"💥 Error creating conversation: {e}")
            raise

    async def _update_conversation_title_async(self, conversation_id: str, title: str) -> bool:
        """Update conversation title async implementation"""
        try:
            result = await self.conversation_repo.update(conversation_id, title=title)
            return result is not None

        except Exception as e:
            logger.error(f"💥 Error updating conversation title: {e}")
            return False

    async def _delete_conversation_async(self, conversation_id: str) -> bool:
        """Delete conversation async implementation"""
        try:
            return await self.conversation_repo.delete(conversation_id)

        except Exception as e:
            logger.error(f"💥 Error deleting conversation: {e}")
            return False

    async def _search_conversations_async(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Search conversations async implementation (Phase 2)"""
        try:
            conversations = await self.conversation_repo.search(user_id, query)

            result = []
            for conv in conversations:
                result.append({
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "message_count": conv.message_count,
                    "is_archived": conv.is_archived
                })

            return result

        except Exception as e:
            logger.error(f"💥 Error searching conversations: {e}")
            return []

    async def _add_message_feedback_async(self, message_id: str, rating: int, feedback: str = None) -> bool:
        """Add message feedback async implementation (Phase 2)"""
        try:
            # Find message by ID (we'll need to add this method)
            # For now, let's add a simple implementation
            result = await self.message_repo.update(
                message_id,
                user_rating=rating,
                user_feedback=feedback
            )
            return result is not None

        except Exception as e:
            logger.error(f"💥 Error adding message feedback: {e}")
            return False

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    async def _ensure_user(self, user_id: str):
        """Ensure user exists in database"""
        try:
            user = await self.user_repo.get_by_username(user_id)
            if not user:
                await self.user_repo.create(user_id)
                logger.info(f"➕ Created user: {user_id}")
            else:
                await self.user_repo.update_last_active(user_id)
        except Exception as e:
            logger.warning(f"Could not ensure user {user_id}: {e}")

    async def _get_or_create_default_conversation(self, user_id: str) -> str:
        """Get or create a default conversation for the user"""
        try:
            # Try to get existing conversations
            conversations = await self.conversation_repo.get_by_user(user_id, limit=1)

            if conversations:
                return conversations[0].id
            else:
                # Create new default conversation
                title = f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                conversation = await self.conversation_repo.create(title, user_id)
                return conversation.id

        except Exception as e:
            logger.error(f"Error getting/creating default conversation: {e}")
            raise

    # ==========================================================================
    # AUTO-TITLE GENERATION (Phase 2)
    # ==========================================================================

    def generate_conversation_title(self, messages: List[Dict[str, Any]], llm_manager=None) -> str:
        """Generate conversation title from messages (Phase 2)"""
        return run_async(self._generate_conversation_title_async(messages, llm_manager))

    async def _generate_conversation_title_async(self, messages: List[Dict[str, Any]], llm_manager=None) -> str:
        """Generate conversation title async implementation (Phase 2)"""
        try:
            if not messages:
                return f"New Chat - {datetime.now().strftime('%Y-%m-%d')}"

            # Get first user message for title generation
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if not user_messages:
                return f"New Chat - {datetime.now().strftime('%Y-%m-%d')}"

            first_message = user_messages[0]["content"]

            # If we have an LLM manager, use it to generate a title
            if llm_manager:
                try:
                    prompt = f"Generate a short, descriptive title (max 50 characters) for a conversation that starts with: '{first_message[:200]}'"
                    title = await llm_manager.call_llm(prompt,
                                                       "You are a helpful assistant that creates concise conversation titles.")
                    title = title.strip().replace('"', '').replace("'", "")

                    if len(title) > 50:
                        title = title[:47] + "..."

                    return title if title else self._fallback_title(first_message)

                except Exception as e:
                    logger.warning(f"LLM title generation failed: {e}")
                    return self._fallback_title(first_message)
            else:
                return self._fallback_title(first_message)

        except Exception as e:
            logger.error(f"Error generating conversation title: {e}")
            return f"Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def _fallback_title(self, first_message: str) -> str:
        """Generate fallback title from first message"""
        # Extract key words and create a title
        words = first_message.split()[:6]  # First 6 words
        title = " ".join(words)

        if len(title) > 50:
            title = title[:47] + "..."

        return title if title else f"Chat - {datetime.now().strftime('%Y-%m-%d')}"


# ==========================================================================
# FACTORY FUNCTION FOR STREAMLIT
# ==========================================================================

def create_chat_service(config: Dict[str, Any]) -> ChatService:
    """Create and initialize chat service for Streamlit app"""
    from .database import get_database_manager, initialize_database

    # Get or create database manager
    db_manager = get_database_manager(config)

    # Initialize if needed
    if not db_manager.is_initialized:
        success = run_async(initialize_database(config))
        if not success:
            raise RuntimeError("Failed to initialize database")

    return ChatService(db_manager)
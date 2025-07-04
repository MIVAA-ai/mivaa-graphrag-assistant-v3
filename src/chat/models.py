# src/chat/models.py
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    STREAMING = "streaming"


class ChatMessage(BaseModel):
    """Chat message model matching current JSON structure"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    role: MessageRole
    content: str
    timestamp: str  # Keep as string to match current format
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # AI Response Metadata (from current system)
    response_time: Optional[float] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None

    # RAG Attribution (from current system)
    sources: Optional[List[Dict[str, Any]]] = None
    cypher_query: Optional[str] = None
    linked_entities: Optional[List[Dict[str, Any]]] = None
    info: Optional[Dict[str, Any]] = None

    # User Feedback (new for Phase 2)
    user_rating: Optional[int] = None  # -1, 0, 1
    user_feedback: Optional[str] = None

    # Technical Details (from current system)
    error_info: Optional[str] = None
    processing_status: ProcessingStatus = ProcessingStatus.COMPLETED

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Conversation(BaseModel):
    """Conversation model for organizing chat sessions"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"  # Start with default user
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_archived: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0

    # Computed fields for UI
    last_message_preview: Optional[str] = None
    last_activity: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class User(BaseModel):
    """Simple user model for multi-user support"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    preferences: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatSearchResult(BaseModel):
    """Search result model for Phase 2"""
    message: ChatMessage
    conversation: Conversation
    relevance_score: float
    highlight_snippet: str


# Helper functions for backward compatibility with current JSON format
def message_to_dict(message: ChatMessage) -> Dict[str, Any]:
    """Convert message model to dict format matching current JSON structure"""
    result = {
        "role": message.role.value,
        "content": message.content,
        "timestamp": message.timestamp
    }

    # Add optional fields only if they exist
    if message.response_time is not None:
        result["response_time"] = message.response_time
    if message.model_used:
        result["model_used"] = message.model_used
    if message.tokens_used:
        result["tokens_used"] = message.tokens_used
    if message.sources:
        result["sources"] = message.sources
    if message.cypher_query:
        result["cypher_query"] = message.cypher_query
    if message.linked_entities:
        result["linked_entities"] = message.linked_entities
    if message.info:
        result["info"] = message.info
    if message.error_info:
        result["error_info"] = message.error_info
    if message.user_rating is not None:
        result["user_rating"] = message.user_rating
    if message.user_feedback:
        result["user_feedback"] = message.user_feedback

    return result


def dict_to_message(data: Dict[str, Any], conversation_id: str) -> ChatMessage:
    """Convert dict from JSON format to message model"""
    # Handle timestamp
    timestamp = data.get("timestamp", datetime.utcnow().isoformat())

    return ChatMessage(
        conversation_id=conversation_id,
        role=MessageRole(data["role"]),
        content=data["content"],
        timestamp=timestamp,
        response_time=data.get("response_time"),
        model_used=data.get("model_used"),
        tokens_used=data.get("tokens_used"),
        sources=data.get("sources"),
        cypher_query=data.get("cypher_query"),
        linked_entities=data.get("linked_entities"),
        info=data.get("info"),
        error_info=data.get("error_info"),
        user_rating=data.get("user_rating"),
        user_feedback=data.get("user_feedback")
    )
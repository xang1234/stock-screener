"""Assistant transcript models for conversation history and messages."""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..database import Base


class Conversation(Base):
    """An assistant conversation session."""

    __tablename__ = "chatbot_conversations"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(36), nullable=False, unique=True, index=True)  # UUID

    # Metadata
    title = Column(String(200))  # Auto-generated from first message
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Status
    is_active = Column(Boolean, default=True)
    message_count = Column(Integer, default=0)

    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Individual message in a conversation."""

    __tablename__ = "chatbot_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(36), ForeignKey("chatbot_conversations.conversation_id"), nullable=False, index=True)

    # Message content
    role = Column(String(20), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)

    # Agent metadata (for assistant messages)
    agent_type = Column(String(50))  # planning, action, validation, answer

    # Tool execution details (for tool messages)
    tool_name = Column(String(100))
    tool_input = Column(JSON)
    tool_output = Column(JSON)

    # Reasoning chain (stored for transparency)
    reasoning = Column(Text)  # Agent's thinking process

    # Aggregated tool calls and thinking traces (JSON arrays)
    tool_calls = Column(JSON)  # List of tool calls made during response
    thinking_traces = Column(JSON)  # List of thinking/reasoning traces

    # Source references (for citations in responses)
    # Note: Using "source_references" instead of "references" to avoid SQLite reserved keyword
    source_references = Column(JSON)  # List of reference objects with type, title, url, section, snippet

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("idx_conversation_messages", "conversation_id", "created_at"),
    )

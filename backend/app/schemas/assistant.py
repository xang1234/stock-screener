"""Schemas for the Hermes-backed Assistant API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssistantConversationCreate(BaseModel):
    """Request body for creating a new assistant conversation."""

    title: str | None = Field(None, min_length=1, max_length=200)


class AssistantMessageCreate(BaseModel):
    """Request body for sending a user message to the assistant."""

    content: str = Field(..., min_length=1, max_length=10_000)


class AssistantReferenceItem(BaseModel):
    """One rendered source reference for an assistant response."""

    type: str
    title: str
    url: str
    section: str | None = None
    snippet: str | None = None
    reference_number: int | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized.startswith("/") and not normalized.startswith("//"):
            return normalized
        parsed = urlparse(normalized)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return normalized
        raise ValueError("url must be an internal app path or an http(s) URL")


class AssistantMessageResponse(BaseModel):
    """Persisted assistant or user message."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: str
    role: str
    content: str
    agent_type: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    source_references: list[AssistantReferenceItem] | None = None
    created_at: datetime


class AssistantConversationResponse(BaseModel):
    """Conversation summary returned by list/create endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int


class AssistantConversationDetailResponse(AssistantConversationResponse):
    """Conversation payload with full message history."""

    messages: list[AssistantMessageResponse]


class AssistantConversationListResponse(BaseModel):
    """Paginated conversation list."""

    conversations: list[AssistantConversationResponse]
    total: int


class AssistantHealthResponse(BaseModel):
    """Operator-facing assistant runtime health."""

    status: str
    available: bool
    streaming: bool = True
    popup_enabled: bool = True
    model: str | None = None
    detail: str | None = None


class WatchlistSummary(BaseModel):
    """Resolved watchlist descriptor."""

    id: int
    name: str


class AssistantWatchlistAddPreviewRequest(BaseModel):
    """Request body for previewing a watchlist add action."""

    watchlist: str = Field(..., min_length=1, max_length=100)
    symbols: list[str] = Field(..., min_length=1, max_length=50)
    reason: str | None = Field(None, max_length=500)


class AssistantWatchlistAddPreviewResponse(BaseModel):
    """Deterministic diff preview before watchlist mutation."""

    watchlist: WatchlistSummary
    requested_symbols: list[str]
    addable_symbols: list[str]
    existing_symbols: list[str]
    invalid_symbols: list[str]
    reason: str | None = None
    summary: str

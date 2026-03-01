"""Data model contracts for cross-module use."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SourceKind(str, Enum):
    LIST = "list"
    USER = "user"


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    kind: SourceKind
    value: str
    enabled: bool = True
    label: str | None = None
    tab: str | None = None


@dataclass(frozen=True)
class TweetItem:
    tweet_id: str
    created_at: datetime | None
    author_handle: str | None
    text: str | None
    source_id: str
    is_reply: bool | None = None
    is_repost: bool | None = None
    is_pinned: bool | None = None
    has_quote: bool | None = None
    quote_tweet_id: str | None = None


@dataclass(frozen=True)
class Checkpoint:
    source_id: str
    last_seen_id: str | None = None
    last_seen_time: datetime | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

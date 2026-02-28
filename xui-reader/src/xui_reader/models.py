"""Data model contracts for cross-module use."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class TweetItem:
    tweet_id: str
    created_at: datetime
    author_handle: str
    text: str
    source_id: str


@dataclass(frozen=True)
class Checkpoint:
    source_id: str
    last_seen_id: str | None = None
    last_seen_time: datetime | None = None
    updated_at: datetime | None = None

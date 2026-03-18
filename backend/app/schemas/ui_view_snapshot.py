"""Schemas for UI bootstrap snapshot endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UISnapshotEnvelope(BaseModel):
    """Published snapshot response wrapper."""

    snapshot_revision: str
    source_revision: str
    published_at: datetime
    is_stale: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)

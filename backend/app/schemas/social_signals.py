"""Typed public and administrator Social Signal API contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


MarketCode = Literal["US", "HK", "CN", "JP", "TW"]
WindowCode = Literal["1d", "7d", "14d"]
QueueView = Literal["actionable", "all", "watch", "risk_off"]
RankMode = Literal["blended", "pure_social"]


class SocialSourceAttempt(BaseModel):
    name: str
    read_status: Literal["pending", "success", "failed"]
    received_count: int | None = None
    history_status: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class SocialLatestAttempt(BaseModel):
    run_id: str
    status: Literal[
        "collecting", "collection_failed", "processing", "failed", "published"
    ]
    started_at: datetime
    completed_at: datetime | None = None
    sources: list[SocialSourceAttempt] = Field(default_factory=list)


class SocialAvailabilityResponse(BaseModel):
    supported: bool
    available: bool
    reason_code: str | None = None
    market: MarketCode
    latest_attempt: SocialLatestAttempt | None = None


class SocialSummaryResponse(SocialAvailabilityResponse):
    generated_at: datetime | None = None
    published_at: datetime | None = None
    stale: bool | None = None
    top_signals: list[dict[str, Any]] = Field(default_factory=list)
    dominant_themes: list[dict[str, Any]] = Field(default_factory=list)
    enabled_source_count: int | None = None
    participating_source_count: int | None = None


class SocialQueueItem(BaseModel):
    candidate_key: str
    canonical_symbol: str
    market: MarketCode | None
    state: str
    social_score: float | None
    confirmation_score: float | None
    queue_score: float | None
    latest_mention: datetime | None
    mention_count: int
    observed_list_count: int
    enabled_list_count: int
    normalization_scope: str
    formula_version: str
    coverage: list[str] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)


class SocialQueueResponse(SocialAvailabilityResponse):
    window: WindowCode
    view: QueueView
    rank_mode: RankMode
    page: int
    page_size: int
    total: int
    items: list[SocialQueueItem] = Field(default_factory=list)
    run_id: str | None = None
    generated_at: datetime | None = None
    published_at: datetime | None = None
    stale: bool | None = None


class SocialSectionResponse(SocialAvailabilityResponse):
    window: WindowCode
    section: Literal["context", "unresolved"]
    page: int
    page_size: int
    total: int
    items: list[SocialQueueItem] = Field(default_factory=list)
    run_id: str | None = None


class SocialEvidencePost(BaseModel):
    post_id: str
    author_handle: str
    created_at: datetime
    excerpt: str = Field(max_length=280)
    url: str
    source_names: list[str] = Field(default_factory=list)
    engagement: dict[str, int | None] = Field(default_factory=dict)


class SocialEvidenceResponse(BaseModel):
    supported: bool
    available: bool
    reason_code: str | None = None
    candidate_key: str
    window: WindowCode
    item: SocialQueueItem | None = None
    posts: list[SocialEvidencePost] = Field(default_factory=list, max_length=3)
    related_listings: list[dict[str, Any]] = Field(default_factory=list)


class SocialRuntimeUpdate(BaseModel):
    mode: Literal["off", "validation", "live"]
    provider: Literal["disabled", "official", "xui"]
    expected_version: int = Field(ge=0)


class SocialSourceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    list_ref: str = Field(min_length=1, max_length=200)


class SocialSourceRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    expected_version: int = Field(ge=1)


class SocialSourceVersionRequest(BaseModel):
    expected_version: int = Field(ge=1)


class SocialSourceTransitionRequest(SocialSourceVersionRequest):
    target: Literal["enabled", "disabled", "archived"]


class SocialAssociationDecisionRequest(BaseModel):
    target: Literal["accepted", "rejected"]
    reason: str = Field(min_length=1, max_length=500)
    expected_version: int = Field(ge=1)


class SocialCompanyIdentityUpdate(BaseModel):
    expected_version: int = Field(ge=0)
    entries: list[dict[str, str]] = Field(max_length=50000)


__all__ = [
    "MarketCode", "QueueView", "RankMode", "SocialAvailabilityResponse",
    "SocialAssociationDecisionRequest", "SocialCompanyIdentityUpdate",
    "SocialEvidencePost", "SocialEvidenceResponse", "SocialQueueItem",
    "SocialQueueResponse", "SocialSectionResponse", "SocialSummaryResponse", "WindowCode",
    "SocialRuntimeUpdate", "SocialSourceCreateRequest", "SocialSourceRenameRequest",
    "SocialSourceTransitionRequest", "SocialSourceVersionRequest",
]

"""Request/response schemas for company-exposure research operations."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class ResearchRequestBody(BaseModel):
    """An authenticated research request. Identity comes only from the
    admin credential; any ``actor``-like field is rejected."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["verify", "refresh", "discover"] = "verify"
    security_id: int | None = Field(default=None, gt=0)
    symbol: str | None = Field(
        default=None, min_length=1, max_length=20, pattern=r"^[A-Za-z0-9.\-]+$"
    )
    economic_theme_id: UUID
    idempotency_key: str = Field(
        min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$"
    )
    supplied_links: list[AnyHttpUrl] = Field(default_factory=list, max_length=5)
    supplied_cik: str | None = Field(default=None, pattern=r"^\d{1,10}$")

    @model_validator(mode="after")
    def _one_listing(self) -> ResearchRequestBody:
        if (self.security_id is None) == (self.symbol is None):
            raise ValueError("exactly one of security_id or symbol is required")
        return self


class IssuerLinkProposalView(BaseModel):
    state: str
    link_revision_id: str | None
    reason: str


class ResearchRequestResponse(BaseModel):
    job_id: str
    created: bool
    state: str | None
    dispatch: Literal["queued", "not_dispatched", "not_needed"]
    issuer_link_proposal: IssuerLinkProposalView | None = None


class ResearchJobResponse(BaseModel):
    view_kind: Literal["research_progress"]
    accepted: Literal[False]
    job_id: str
    root_job_id: str
    kind: str
    security_id: int | None
    issuer_id: str | None
    economic_theme_id: str
    market: str | None
    requested_by: str
    created_at: str | None
    state: str | None
    condition: str | None
    assessment_revision_id: str | None
    stages: list[dict[str, Any]]
    events: list[dict[str, Any]]


class ShadowPreviewResponse(BaseModel):
    view_kind: Literal["shadow_preview"]
    authoritative_membership: Literal[False]
    job_id: str
    state: str | None
    assessment_id: str
    assessment_revision_id: str
    revision_number: int
    input_manifest_hash: str
    assessed_at: str | None
    issuer_id: str
    issuer_link_revision_ids: list[str]
    economic_theme_id: str
    theme_fingerprint: str | None
    policies: dict[str, Any]
    coverage: list[dict[str, Any]]
    unresolved_questions: list[str]
    conflicts: list[dict[str, Any]]
    claims: list[dict[str, Any]]

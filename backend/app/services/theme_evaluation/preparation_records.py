"""Preparation artifacts are separate from frozen version-1 evidence bundles."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from .records import SHA, URL, Record


class DocumentHandoff(Record):
    source_text_sha256: SHA
    language: str | None = None
    image_urls: list[URL] = Field(default_factory=list)
    local_images: list[str] = Field(default_factory=list)


class Handoff(Record):
    bundle_id: SHA
    documents: dict[str, DocumentHandoff] = Field(default_factory=dict)
    references: dict[str, URL] = Field(default_factory=dict)


class PreparationRequest(Record):
    stage: Literal["article", "text", "image"]
    input_sha256: SHA
    provider: str
    model: str | None
    policy_version: str
    options: dict = Field(default_factory=dict)


class PreparationResult(Record):
    request: PreparationRequest
    status: Literal["success", "partial", "unavailable", "needs_review"]
    payload: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    assets: list[SHA] = Field(default_factory=list)
    source_url: URL | None = None
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def outcome_consistent(self):
        if self.status == "success" and not self.payload:
            raise ValueError("success_requires_payload")
        if self.status == "unavailable" and not self.warnings:
            raise ValueError("unavailable_requires_reason")
        return self


class PreparationBinding(Record):
    source_kind: Literal["document", "reference"]
    source_id: str
    source_text_sha256: SHA
    result_id: SHA
    parent_result_id: SHA | None = None
    input_locator: str | None = None


class PreparationManifest(Record):
    schema_version: Literal[1] = 1
    bundle_id: SHA
    handoff: Handoff
    bindings: list[PreparationBinding]
    evidence_review: Literal["pending"] = "pending"
    extraction: Literal["awaiting_evidence_approval"] = "awaiting_evidence_approval"

"""Typed preparation outcomes own their content and provenance invariants."""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from .article_recovery import ArticleRecovery
from .bundle import sha256
from .multilingual_preparation import TextPreparation
from .preparation_models import ImageObservation
from .records import SHA, URL, Record


class Request(Record):
    input_sha256: SHA
    provider: str
    model: str | None
    policy_version: str


class ArticleRequest(Request):
    stage: Literal["article"] = "article"
    destination_url: URL | None = None


class TextRequest(Request):
    stage: Literal["text"] = "text"
    language: str | None = None
    target_language: str = "en"
    max_chars: int = Field(default=4000, ge=1, le=10000)
    method: Literal["prepare", "translation_import"] = "prepare"


class ImageRequest(Request):
    stage: Literal["image"] = "image"


PreparationRequest = Annotated[
    ArticleRequest | TextRequest | ImageRequest, Field(discriminator="stage")
]


class Result(Record):
    assets: list[SHA] = Field(default_factory=list)
    source_url: URL | None = None
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ArticleResult(Result):
    stage: Literal["article"] = "article"
    request: ArticleRequest
    payload: ArticleRecovery | None = None
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_article(self):
        if self.payload is None:
            if not self.failure_reasons:
                raise ValueError("unavailable_requires_reason")
        elif (
            self.payload.response_sha256 != self.request.input_sha256
            or self.payload.response_sha256 not in self.assets
        ):
            raise ValueError("article_asset_mismatch")
        elif self.failure_reasons:
            raise ValueError("article_payload_has_failure")
        return self

    @property
    def status(self):
        if not self.payload or not self.payload.text.strip():
            return "unavailable"
        if self.payload.capture_status != "full":
            return "partial"
        return "needs_review" if self.payload.warnings else "success"

    @property
    def warnings(self):
        warnings = self.payload.warnings if self.payload else self.failure_reasons
        if self.status != "success":
            return list(dict.fromkeys([*warnings, "browser_followup_required"]))
        return warnings

    @property
    def source_text(self):
        return self.payload.text if self.payload else ""

    @property
    def has_input(self):
        return self.payload is not None


class TextResult(Result):
    stage: Literal["text"] = "text"
    request: TextRequest
    payload: TextPreparation

    @model_validator(mode="after")
    def valid_text(self):
        if sha256(self.source_text.encode()) != self.request.input_sha256:
            raise ValueError("translation_source_hash_mismatch")
        if (
            self.payload.supplied_language != self.request.language
            or self.payload.target_language != self.request.target_language
        ):
            raise ValueError("translation_request_language_mismatch")
        return self

    @property
    def status(self):
        return self.payload.status

    @property
    def warnings(self):
        return self.payload.warnings

    @property
    def source_text(self):
        return "".join(s.original for s in self.payload.segments)

    @property
    def has_input(self):
        return True


class ImageResult(Result):
    stage: Literal["image"] = "image"
    request: ImageRequest
    payload: ImageObservation | None = None
    failure_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_image(self):
        if self.payload is None:
            if not self.failure_reasons:
                raise ValueError("unavailable_requires_reason")
        elif self.request.input_sha256 not in self.assets:
            raise ValueError("image_asset_required")
        elif self.failure_reasons:
            raise ValueError("image_payload_has_failure")
        return self

    @property
    def status(self):
        if self.payload is None:
            return "unavailable"
        return "needs_review" if self.payload.uncertainties else "success"

    @property
    def warnings(self):
        return self.payload.uncertainties if self.payload else self.failure_reasons

    @property
    def source_text(self):
        return self.payload.transcription if self.payload else ""

    @property
    def has_input(self):
        return self.request.input_sha256 in self.assets


PreparationResult = Annotated[
    ArticleResult | TextResult | ImageResult, Field(discriminator="stage")
]
RESULT_ADAPTER = TypeAdapter(PreparationResult)

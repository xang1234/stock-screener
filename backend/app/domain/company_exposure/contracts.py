"""Company-exposure research contracts.

These value objects freeze the vocabulary shared by persistence, research
work, assessment, publication and readers of the company exposure map
(``docs/superpowers/specs/2026-09-25-company-exposure-map-design.md``).

The rules they encode:

* research is issuer-centric; membership is listing-specific;
* source evidence, claims, assessments, membership decisions and grounding
  context are distinct objects with immutable revisions;
* "verified" means primary-backed under the recorded policy, per claim;
* support basis, conclusion, freshness and commercial status are separate
  axes, never collapsed into one confidence score;
* materiality is disclosed, reproducibly calculated, primary-supported
  qualitative, or unknown -- never ``exposure_strength``/confidence;
* subscription LLM usage is counted in requests/tokens; no dollar cost is
  invented, and uncertain dispatches are never refunded;
* the feature is disabled by default and paid search needs explicit
  enablement plus a spending cap, never just a credential.

Exposure-support values here are deliberately not the economic taxonomy's
``ExposureSupport`` enum: primary research support is contextual inference
when applied to a different source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID


class ResearchMode(StrEnum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    LIVE = "live"


class LLMBillingMode(StrEnum):
    SUBSCRIPTION = "subscription"


class SupportBasis(StrEnum):
    PRIMARY_EXPLICIT = "primary_explicit"
    PRIMARY_SYNTHESIS = "primary_synthesis"
    SECONDARY_REPORTED = "secondary_reported"
    INFERRED_UNVERIFIED = "inferred_unverified"
    UNRESOLVED = "unresolved"


PRIMARY_SUPPORT_BASES = frozenset(
    {SupportBasis.PRIMARY_EXPLICIT, SupportBasis.PRIMARY_SYNTHESIS}
)


class Conclusion(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    DISPUTED = "disputed"
    UNKNOWN = "unknown"


class FreshnessState(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    UNDATED = "undated"


class CommercialStatus(StrEnum):
    RESEARCH = "research"
    ANNOUNCED = "announced"
    QUALIFICATION = "qualification"
    COMMERCIALLY_AVAILABLE = "commercially_available"
    SHIPPING_OR_OPERATING = "shipping_or_operating"
    DISCONTINUED = "discontinued"
    UNKNOWN = "unknown"


class ClaimKind(StrEnum):
    PARTICIPATION = "participation"
    ROLE = "role"
    PRODUCT_APPLICATION = "product_application"
    CUSTOMER_RELATIONSHIP = "customer_relationship"
    COMMERCIAL_STATUS = "commercial_status"
    MATERIALITY = "materiality"
    EXPOSURE_END = "exposure_end"


class ReportingScope(StrEnum):
    ISSUER_CONSOLIDATED = "issuer_consolidated"
    ISSUER_STANDALONE = "issuer_standalone"
    SEGMENT_OR_SUBSIDIARY = "segment_or_subsidiary"


class MaterialityBasis(StrEnum):
    DISCLOSED = "disclosed"
    CALCULATED = "calculated"
    QUALITATIVE = "qualitative"
    UNKNOWN = "unknown"


class QualitativeMateriality(StrEnum):
    CORE_BUSINESS = "core_business"
    EXPLICITLY_MATERIAL = "explicitly_material"
    EXPLICITLY_LIMITED = "explicitly_limited"
    UNKNOWN = "unknown"


class AutomaticUse(StrEnum):
    ALLOWED = "allowed"
    HELD = "held"


class EvidenceRole(StrEnum):
    """How a piece of evidence may participate in a claim's support."""

    ORIGINAL_PRIMARY = "original_primary"
    ORIGINAL_SECONDARY = "original_secondary"
    DERIVATIVE_NOT_INDEPENDENT_SOURCE = "derivative_not_independent_source"
    RETRIEVAL_AID_ONLY = "retrieval_aid_only"


class ResearchRequestKind(StrEnum):
    VERIFY = "verify"
    REFRESH = "refresh"
    DISCOVER = "discover"


class ResearchJobState(StrEnum):
    QUEUED = "queued"
    RESEARCHING = "researching"
    EVIDENCE_READY = "evidence_ready"
    ASSESSING = "assessing"
    READY_FOR_PUBLICATION = "ready_for_publication"
    PUBLISHED = "published"
    PARTIAL = "partial"
    HELD = "held"
    PAUSED_ALLOWANCE = "paused_allowance"
    PAUSED_SEARCH_BUDGET = "paused_search_budget"
    PAUSED_STORAGE = "paused_storage"
    UNAVAILABLE_CAPABILITY = "unavailable_capability"
    RETRYABLE_FAILURE = "retryable_failure"
    REVIEW_REQUIRED = "review_required"
    TERMINAL_FAILURE = "terminal_failure"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATES = frozenset(
    {
        ResearchJobState.READY_FOR_PUBLICATION,
        ResearchJobState.PUBLISHED,
        ResearchJobState.TERMINAL_FAILURE,
        ResearchJobState.CANCELLED,
    }
)


class DispatchPhase(StrEnum):
    """Whether a provider request may have reached the provider (spec §10.2)."""

    PRE_DISPATCH = "pre_dispatch"
    DISPATCHED = "dispatched"
    UNCERTAIN = "uncertain"


class ReservationState(StrEnum):
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    RELEASED = "released"
    RECONCILED = "reconciled"
    UNCERTAIN = "uncertain"
    EXPIRED_UNCERTAIN = "expired_uncertain"
    PAUSED_ALLOWANCE = "paused_allowance"


class ResourceUnit(StrEnum):
    REQUESTS = "requests"
    REPORTED_TOKENS = "reported_tokens"
    CURRENCY_AMOUNT = "currency_amount"
    BLOB_BYTES = "blob_bytes"


class LinkState(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"


class LinkAcceptancePolicy(StrEnum):
    ADMINISTRATOR_REVIEWED = "administrator_reviewed"
    OFFICIAL_REGISTRY_SINGLE_LISTING = "official_registry_single_listing"
    LEGACY_ATTESTATION_IMPORT = "legacy_attestation_import"


class CoverageOutcome(StrEnum):
    COMPLETE_FOR_REQUESTED_SCOPE = "complete_for_requested_scope"
    PARTIAL = "partial"
    NO_MATCHING_DOCUMENT = "no_matching_document"
    NOT_CONFIGURED = "not_configured"
    PERMISSION_UNAVAILABLE = "permission_unavailable"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE_CAPABILITY = "unavailable_capability"
    BLOCKED_DESTINATION = "blocked_destination"
    FETCH_FAILED = "fetch_failed"
    DISABLED = "disabled"
    OMITTED = "omitted"


LAUNCH_MARKETS = ("US", "HK", "JP", "TW")
SERVICE_PRINCIPAL = "system:company-exposure-research"
UNKNOWN_MATERIALITY_WORDING = "Not separately disclosed in reviewed evidence"


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_non_negative(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ResearchLimits:
    """Approved V1 defaults (spec §8.3, §9.7, §10.3).

    ``daily_request_allocation``/``daily_token_allocation`` have no guessed
    default: absence blocks new provider work.
    """

    research_mode: ResearchMode = ResearchMode.DISABLED
    paid_search_enabled: bool = False
    llm_billing_mode: LLMBillingMode = LLMBillingMode.SUBSCRIPTION
    concurrent_investigations: int = 1
    concurrent_provider_requests: int = 1
    verification_pairs_per_request: int = 1
    new_issuers_per_discovery_job: int = 10
    search_queries_per_root_job: int = 6
    documents_per_issuer: int = 12
    download_bytes_per_document: int = 25 * 1024 * 1024
    text_pages_per_document: int = 300
    passages_per_issuer: int = 24
    image_pages_per_issuer: int = 8
    browser_navigations_per_issuer: int = 4
    provider_attempts_per_issuer: int = 24
    provider_attempts_per_root_job: int = 24
    storage_max_bytes: int = 5 * 1024 * 1024 * 1024
    storage_min_free_bytes: int = 1024 * 1024 * 1024
    scratch_bytes: int = 512 * 1024 * 1024
    daily_request_allocation: int | None = None
    daily_token_allocation: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "research_mode", ResearchMode(self.research_mode))
        object.__setattr__(
            self, "llm_billing_mode", LLMBillingMode(self.llm_billing_mode)
        )
        if not isinstance(self.paid_search_enabled, bool):
            raise ValueError("paid_search_enabled must be boolean")
        for name in (
            "concurrent_investigations",
            "concurrent_provider_requests",
            "verification_pairs_per_request",
            "new_issuers_per_discovery_job",
            "search_queries_per_root_job",
            "documents_per_issuer",
            "download_bytes_per_document",
            "text_pages_per_document",
            "passages_per_issuer",
            "image_pages_per_issuer",
            "browser_navigations_per_issuer",
            "provider_attempts_per_issuer",
            "provider_attempts_per_root_job",
            "storage_max_bytes",
            "storage_min_free_bytes",
            "scratch_bytes",
        ):
            _require_non_negative(getattr(self, name), name)
        for name in ("daily_request_allocation", "daily_token_allocation"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(value, name)

    @property
    def provider_dispatch_configured(self) -> bool:
        return self.daily_request_allocation is not None


@dataclass(frozen=True, slots=True)
class CoverageItem:
    """A searched route/document/page and what happened; never a finding."""

    route: str
    outcome: CoverageOutcome
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.route, "route")
        object.__setattr__(self, "outcome", CoverageOutcome(self.outcome))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Discovery leads only; never verification evidence."""

    items: tuple[dict[str, Any], ...] = ()
    coverage: tuple[CoverageItem, ...] = ()

    @classmethod
    def disabled(cls, reason: str) -> "SearchResult":
        return cls(
            coverage=(
                CoverageItem(
                    route="search", outcome=CoverageOutcome.DISABLED, reason=reason
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    status: str
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def skipped(cls, reason: str) -> "TaskOutcome":
        return cls(status="skipped", reason=reason)

    @classmethod
    def completed(cls, **detail: Any) -> "TaskOutcome":
        return cls(status="completed", detail=detail)


def _json_default(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non_finite_decimal")
        return format(value, "f")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive_datetime_not_serializable")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda item: json.dumps(item, default=str))
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "value") and isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"not_json_serializable:{type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Stable JSON: sorted keys, UUID/Decimal as strings, UTC datetimes.

    Floats are rejected so hashed payloads never depend on binary rounding.
    """

    def _reject_float(item: Any) -> Any:
        if isinstance(item, float):
            raise ValueError("float_not_permitted_use_decimal")
        if isinstance(item, dict):
            return {str(k): _reject_float(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [_reject_float(v) for v in item]
        return item

    return json.dumps(
        _reject_float(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


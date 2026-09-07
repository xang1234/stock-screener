"""Immutable values shared by social-signal use cases and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Mapping


SUPPORTED_PROVIDERS = frozenset({"official", "xui"})
SUPPORTED_MARKETS = frozenset({"US", "HK", "CN", "JP", "TW"})
_MAX_CLOCK_SKEW = timedelta(minutes=5)


def _required(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"blank_{field}")


def validate_utc_timestamp(
    value: datetime,
    field: str,
    *,
    reference_at: datetime | None = None,
) -> datetime:
    """Validate ingress time against an explicit trusted reference, never wall time."""
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"naive_timestamp:{field}")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"non_utc_timestamp:{field}")
    if reference_at is not None:
        validate_utc_timestamp(reference_at, "reference_at")
        if value > reference_at + _MAX_CLOCK_SKEW:
            raise ValueError(f"future_timestamp:{field}")
    return value


def _utc(value: datetime | None, field: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    validate_utc_timestamp(value, field)  # type: ignore[arg-type]


def _deeply_immutable(value: object, field: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"immutable_tuple:{field}")
    for item in value:
        if isinstance(item, (list, dict, set)):
            raise TypeError(f"immutable_tuple:{field}")
        if isinstance(item, tuple):
            _deeply_immutable(item, field)


def _choice(value: str, allowed: set[str] | frozenset[str], field: str) -> None:
    if value not in allowed:
        raise ValueError(f"invalid_{field}")


@dataclass(frozen=True, slots=True)
class SocialReadRequest:
    request_id: str
    source_id: str
    list_id: str
    intent: str
    observed_at: datetime
    limit: int
    target_published_after: datetime
    application_progress: str | None = None

    def __post_init__(self) -> None:
        _required(self.request_id, "request_id")
        _required(self.source_id, "source_id")
        _required(self.list_id, "list_id")
        _choice(self.intent, {"initial", "incremental", "test"}, "read_intent")
        _utc(self.observed_at, "observed_at")
        _utc(self.target_published_after, "target_published_after")
        validate_utc_timestamp(
            self.target_published_after,
            "target_published_after",
            reference_at=self.observed_at,
        )
        if self.limit <= 0:
            raise ValueError("invalid_limit")


@dataclass(frozen=True, slots=True)
class SocialPostRecord:
    provider: str
    provider_post_id: str
    source_id: str
    text: str
    url: str
    author_handle: str
    created_at: datetime
    observed_at: datetime
    likes: int | None = None
    reposts: int | None = None
    replies: int | None = None
    quotes: int | None = None
    bookmarks: int | None = None
    views: int | None = None
    canonical_url: str | None = None
    is_repost: bool = False
    quoted_text: str | None = None
    # Saved extraction judgments. No scorer infers a thesis or a copied claim.
    has_new_thesis: bool | None = None
    canonical_claim_key: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError("unsupported_provider")
        for field in ("provider_post_id", "source_id", "text", "url", "author_handle"):
            _required(getattr(self, field), field)
        _utc(self.created_at, "created_at")
        _utc(self.observed_at, "observed_at")
        try:
            validate_utc_timestamp(
                self.created_at, "created_at", reference_at=self.observed_at
            )
        except ValueError as exc:
            if str(exc) == "future_timestamp:created_at":
                raise ValueError("future_timestamp") from exc
            raise
        for field in ("likes", "reposts", "replies", "quotes", "bookmarks", "views"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"negative_metric:{field}")
        if self.canonical_url is not None:
            _required(self.canonical_url, "canonical_url")
        if self.canonical_claim_key is not None:
            _required(self.canonical_claim_key, "canonical_claim_key")

    @classmethod
    def from_untrusted(
        cls,
        payload: Mapping[str, Any],
        *,
        provider: str,
        source_id: str,
        observed_at: datetime,
    ) -> SocialPostRecord:
        allowed = {
            "tweet_id", "id", "provider_post_id", "created_at", "text", "url",
            "author_handle", "username", "likes", "reposts", "replies", "quotes",
            "bookmarks", "views", "canonical_url", "is_repost", "quoted_text",
        }
        unexpected = set(payload) - allowed
        if unexpected:
            raise ValueError("unexpected_payload_fields")
        created = payload.get("created_at")
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("invalid_timestamp:created_at") from exc
        _utc(created, "created_at")
        _utc(observed_at, "observed_at")
        if created > observed_at + _MAX_CLOCK_SKEW:
            raise ValueError("future_timestamp")
        return cls(
            provider=provider,
            provider_post_id=str(payload.get("provider_post_id") or payload.get("tweet_id") or payload.get("id") or ""),
            source_id=source_id,
            text=str(payload.get("text") or ""),
            url=str(payload.get("url") or ""),
            author_handle=str(payload.get("author_handle") or payload.get("username") or ""),
            created_at=created,
            observed_at=observed_at,
            likes=payload.get("likes"),
            reposts=payload.get("reposts"),
            replies=payload.get("replies"),
            quotes=payload.get("quotes"),
            bookmarks=payload.get("bookmarks"),
            views=payload.get("views"),
            canonical_url=payload.get("canonical_url"),
            is_repost=bool(payload.get("is_repost", False)),
            quoted_text=payload.get("quoted_text"),
        )


@dataclass(frozen=True, slots=True)
class SocialSourceOutcome:
    read_status: str
    processing_status: str
    history_status: str
    coverage_reason_codes: tuple[str, ...]
    known_gap_intervals: tuple[tuple[datetime, datetime], ...]
    observed_oldest_at: datetime | None
    observed_newest_at: datetime | None
    received_count: int
    committed_progress: str | None
    error_code: str | None

    def __post_init__(self) -> None:
        _deeply_immutable(self.coverage_reason_codes, "coverage_reason_codes")
        _deeply_immutable(self.known_gap_intervals, "known_gap_intervals")
        _choice(self.read_status, {"success", "failed"}, "read_status")
        _choice(self.processing_status, {"pending", "complete", "failed"}, "processing_status")
        _choice(self.history_status, {"warming_up", "limited", "observed_window"}, "history_status")
        if self.received_count < 0:
            raise ValueError("negative_received_count")
        _utc(self.observed_oldest_at, "observed_oldest_at", nullable=True)
        _utc(self.observed_newest_at, "observed_newest_at", nullable=True)
        if (self.observed_oldest_at is None) != (self.observed_newest_at is None):
            raise ValueError("inconsistent_source_outcome:unpaired_bounds")
        if (
            self.observed_oldest_at is not None
            and self.observed_newest_at is not None
            and self.observed_oldest_at > self.observed_newest_at
        ):
            raise ValueError("inconsistent_source_outcome:reversed_bounds")
        if self.read_status == "success" and self.error_code is not None:
            raise ValueError("inconsistent_source_outcome:success_with_error")
        if self.read_status == "failed":
            if self.error_code is None:
                raise ValueError("inconsistent_source_outcome:failure_without_error")
            if self.processing_status == "complete":
                raise ValueError("inconsistent_source_outcome:failed_read_complete")
            if self.committed_progress is not None:
                raise ValueError("inconsistent_source_outcome:failed_read_progress")
            if self.history_status == "observed_window":
                raise ValueError("inconsistent_source_outcome:failed_read_coverage")
        for start, end in self.known_gap_intervals:
            _utc(start, "gap_start")
            _utc(end, "gap_end")
            if start > end:
                raise ValueError("invalid_gap_interval")


@dataclass(frozen=True, slots=True)
class SocialSourceBatch:
    request: SocialReadRequest
    posts: tuple[SocialPostRecord, ...]
    outcome: SocialSourceOutcome

    def __post_init__(self) -> None:
        _deeply_immutable(self.posts, "posts")


@dataclass(frozen=True, slots=True)
class SourceTestOutcome:
    provider: str
    status: str
    sample_count: int
    tested_at: datetime
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError("unsupported_provider")
        _choice(
            self.status,
            {"passed", "failed", "rate_limited", "reauthentication_required", "provider_error"},
            "test_status",
        )
        if not 0 <= self.sample_count <= 5:
            raise ValueError("invalid_sample_count")
        _utc(self.tested_at, "tested_at")
        if self.reason_code is not None:
            _required(self.reason_code, "reason_code")


@dataclass(frozen=True, slots=True)
class SocialSourceView:
    source_id: str
    name: str
    canonical_url: str
    list_id: str
    lifecycle: str
    provenance: str
    test_outcome: SourceTestOutcome | None
    collected_at: datetime | None
    version: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        for field in ("source_id", "name", "canonical_url", "list_id", "lifecycle", "provenance"):
            _required(getattr(self, field), field)
        if self.version < 0:
            raise ValueError("negative_version")
        _utc(self.collected_at, "collected_at", nullable=True)
        _utc(self.created_at, "created_at")
        _utc(self.updated_at, "updated_at")


@dataclass(frozen=True, slots=True)
class SocialSourceAuditView:
    action: str
    actor: str
    occurred_at: datetime
    before_metadata: tuple[tuple[str, str | None], ...]
    after_metadata: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        _deeply_immutable(self.before_metadata, "before_metadata")
        _deeply_immutable(self.after_metadata, "after_metadata")
        _required(self.action, "action")
        _required(self.actor, "actor")
        _utc(self.occurred_at, "occurred_at")


@dataclass(frozen=True, slots=True)
class TickerResolution:
    raw_token: str
    symbol: str | None
    market: str | None
    security_id: str | None
    status: str
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _deeply_immutable(self.reason_codes, "reason_codes")
        _required(self.raw_token, "raw_token")
        _choice(self.status, {"resolved", "unresolved"}, "resolution_status")
        if self.market is not None and self.market not in SUPPORTED_MARKETS:
            raise ValueError("unsupported_market")


@dataclass(frozen=True, slots=True)
class SocialEvidenceInput:
    candidate_key: str
    canonical_symbol: str
    market: str
    posts: tuple[SocialPostRecord, ...]
    enabled_source_ids: tuple[str, ...] = ()
    history_complete: bool = False
    coverage_reasons: tuple[str, ...] = ()
    resolved: bool = True
    security_kind: str = "stock"

    def __post_init__(self) -> None:
        _deeply_immutable(self.posts, "posts")
        _deeply_immutable(self.enabled_source_ids, "enabled_source_ids")
        _deeply_immutable(self.coverage_reasons, "coverage_reasons")
        _choice(self.security_kind, {"stock", "thematic_etf", "broad_etf", "macro"}, "security_kind")
        _required(self.candidate_key, "candidate_key")
        _required(self.canonical_symbol, "canonical_symbol")
        if self.market not in SUPPORTED_MARKETS:
            raise ValueError("unsupported_market")


@dataclass(frozen=True, slots=True)
class ThemeMarketEvidence:
    """Frozen Market measurement supplied by the shared Theme projection."""

    theme_key: str
    market: str
    session_date: date
    benchmark_symbol: str
    basket_version: str
    accepted_company_count: int
    components: tuple[tuple[str, Decimal | None], ...]
    measured_company_counts: tuple[tuple[str, int], ...]
    reasons: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field in ("theme_key", "benchmark_symbol", "basket_version"):
            _required(getattr(self, field), field)
        _choice(self.market, SUPPORTED_MARKETS, "market")
        if not isinstance(self.session_date, date) or isinstance(self.session_date, datetime):
            raise ValueError("invalid_session_date")
        if self.accepted_company_count < 0:
            raise ValueError("negative_accepted_company_count")
        for field in ("components", "measured_company_counts", "reasons"):
            values = getattr(self, field)
            _deeply_immutable(values, field)
            if len({k for k, _ in values}) != len(values):
                raise ValueError(f"duplicate_keys:{field}")
        if any(n < 0 or n > self.accepted_company_count for _, n in self.measured_company_counts):
            raise ValueError("invalid_measured_company_count")


@dataclass(frozen=True, slots=True)
class ConfirmationInput:
    candidate_key: str
    market: str
    observed_at: datetime
    setup_score: Decimal | None = None
    rs_rating_1m: Decimal | None = None
    rs_rating_3m: Decimal | None = None
    group_rank: int | None = None
    market_group_count: int | None = None
    theme_confirmations: tuple[ThemeMarketEvidence, ...] = ()
    market_benchmark: str | None = None

    def __post_init__(self) -> None:
        _deeply_immutable(self.theme_confirmations, "theme_confirmations")
        _required(self.candidate_key, "candidate_key")
        if self.market not in SUPPORTED_MARKETS:
            raise ValueError("unsupported_market")
        _utc(self.observed_at, "observed_at")


@dataclass(frozen=True, slots=True)
class ComponentScore:
    value: Decimal | None
    available_weight: Decimal
    total_weight: Decimal
    reasons: tuple[str, ...] = ()
    observed_count: int = 0
    input_count: int = 0
    components: tuple[tuple[str, ComponentScore], ...] = ()
    selected_key: str | None = None

    def __post_init__(self) -> None:
        _deeply_immutable(self.reasons, "reasons")
        _deeply_immutable(self.components, "components")


@dataclass(frozen=True, slots=True)
class SignalStateInput:
    """Saved required-check judgments; freshness is resolved by the calendar reader."""

    resolved: bool
    active: bool
    market: str | None
    security_kind: str = "stock"
    feature_fresh: bool | None = None
    market_fresh: bool | None = None
    liquidity_eligible: bool | None = None
    setup_ready: bool | None = None
    setup_score: Decimal | None = None
    market_exposure: Decimal | None = None


@dataclass(frozen=True, slots=True)
class SignalStateDecision:
    state: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _deeply_immutable(self.reasons, "reasons")


@dataclass(frozen=True, slots=True)
class SocialScoreResult:
    social_score: Decimal | None
    components: tuple[tuple[str, ComponentScore], ...]
    state: SignalStateDecision
    candidate_key: str = ""
    canonical_symbol: str = ""
    market: str | None = None
    normalization_scope: str = "global_fallback"
    latest_mention: datetime | None = None
    mention_count: int = 0
    observed_list_count: int = 0
    enabled_list_count: int = 0
    post_memberships: tuple[tuple[str, tuple[str, ...]], ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()
    acceleration: Decimal | None = None
    engagement_sum: float | None = None
    formula_version: str = "social-signal-v1"

    def __post_init__(self) -> None:
        _deeply_immutable(self.components, "components")
        _deeply_immutable(self.post_memberships, "post_memberships")
        _deeply_immutable(self.exclusions, "exclusions")
        _utc(self.latest_mention, "latest_mention", nullable=True)


@dataclass(frozen=True, slots=True)
class SocialRunResult:
    run_id: str
    mode: str
    processing_status: str
    published: bool
    coverage_summary: tuple[tuple[str, str], ...]
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _deeply_immutable(self.coverage_summary, "coverage_summary")
        _deeply_immutable(self.reason_codes, "reason_codes")


@dataclass(frozen=True, slots=True)
class SocialSnapshotRecord:
    run_id: str
    candidate_id: str
    symbol: str | None
    market: str | None
    candidate_state: str
    social_score: Decimal | None
    confirmation_score: Decimal | None
    queue_score: Decimal | None
    pinned_inputs: tuple[tuple[str, str], ...]
    coverage: tuple[str, ...]
    latest_mention: datetime | None
    canonical_symbol: str
    candidate_key: str

    def __post_init__(self) -> None:
        _deeply_immutable(self.pinned_inputs, "pinned_inputs")
        _deeply_immutable(self.coverage, "coverage")
        portions_available = self.social_score is not None and self.confirmation_score is not None
        if portions_available != (self.queue_score is not None):
            raise ValueError("inconsistent_snapshot_scores")


@dataclass(frozen=True, slots=True)
class QueuePage:
    items: tuple[SocialSnapshotRecord, ...]
    page: int
    page_size: int
    total: int

    def __post_init__(self) -> None:
        _deeply_immutable(self.items, "items")


@dataclass(frozen=True, slots=True)
class DispatchResult:
    dispatch_id: str
    accepted: bool
    reason_code: str | None = None

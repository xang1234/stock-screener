"""Immutable values shared by social-signal use cases and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, Mapping


SUPPORTED_PROVIDERS = frozenset({"official", "xui"})
SUPPORTED_MARKETS = frozenset({"US", "HK", "CN", "JP", "TW"})
_MAX_CLOCK_SKEW = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class BacklogResult:
    succeeded: int
    deferred: int
    failed: int
    outside_window: int
    next_reset_at: datetime


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
    proposed_progress: str | None = None
    rate_limit_reset_at: datetime | None = None

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
        _utc(self.rate_limit_reset_at, "rate_limit_reset_at", nullable=True)
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
            if self.proposed_progress is not None:
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
    company_id: str | None = None
    related_symbols: tuple[str, ...] = ()
    security_kind: str = "stock"
    ranking_eligible: bool = False
    company_count_eligible: bool = False
    explicit_listing: bool = False

    def __post_init__(self) -> None:
        _deeply_immutable(self.reason_codes, "reason_codes")
        _deeply_immutable(self.related_symbols, "related_symbols")
        _choice(self.security_kind, {"stock", "thematic_etf", "broad_etf", "macro"}, "security_kind")
        _required(self.raw_token, "raw_token")
        _choice(self.status, {"resolved", "unresolved"}, "resolution_status")
        if self.market is not None and self.market not in SUPPORTED_MARKETS:
            raise ValueError("unsupported_market")
        if self.company_count_eligible and (not self.company_id or self.status != "resolved" or self.security_kind != "stock"):
            raise ValueError("invalid_company_count_eligibility")


@dataclass(frozen=True, slots=True)
class ExtractionClaim:
    post_id: str
    theme_key: str
    raw_theme: str
    company_token: str
    relationship: str
    excerpt: str
    support: Literal["supported", "uncertain", "unsupported"]
    duplicate_of_post_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("post_id", "theme_key", "raw_theme", "company_token", "relationship", "excerpt"):
            _required(getattr(self, field), field)
        _choice(self.support, {"supported", "uncertain", "unsupported"}, "support")
        _deeply_immutable(self.duplicate_of_post_ids, "duplicate_of_post_ids")
        for post_id in self.duplicate_of_post_ids:
            _required(post_id, "duplicate_post_id")


@dataclass(frozen=True, slots=True)
class ExtractionPostJudgment:
    post_id: str
    has_new_thesis: bool
    canonical_claim_key: str | None

    def __post_init__(self) -> None:
        _required(self.post_id, "post_id")
        if type(self.has_new_thesis) is not bool:
            raise ValueError("invalid_has_new_thesis")
        if self.canonical_claim_key is not None:
            _required(self.canonical_claim_key, "canonical_claim_key")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    input_hash: str
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    claims: tuple[ExtractionClaim, ...]
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    judgments: tuple[ExtractionPostJudgment, ...] = ()

    def __post_init__(self) -> None:
        for field in ("input_hash", "provider", "model", "prompt_version", "schema_version"):
            _required(getattr(self, field), field)
        for field, record_type in (("claims", ExtractionClaim), ("judgments", ExtractionPostJudgment)):
            _deeply_immutable(getattr(self, field), field)
            if any(not isinstance(item, record_type) for item in getattr(self, field)):
                raise TypeError(f"invalid_{field}")
        for field in ("usage_input_tokens", "usage_output_tokens"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"invalid_{field}")


@dataclass(frozen=True, slots=True)
class ThemeProjection:
    run_id: str
    policy_version: str
    proposals: tuple[ExtractionClaim, ...]
    registry_version: int
    identity_version: int
    identity_policy_version: str
    prepared_at: datetime
    work_ids: tuple[int, ...]
    resolutions: tuple[TickerResolution, ...]
    pipeline: str

    def __post_init__(self):
        _utc(self.prepared_at, "prepared_at")
        _deeply_immutable(self.proposals, "proposals")
        _deeply_immutable(self.work_ids, "work_ids")
        _deeply_immutable(self.resolutions, "resolutions")
        if len(self.resolutions) != len(self.proposals):
            raise ValueError("projection_resolution_mismatch")
        _choice(self.pipeline, {"technical", "fundamental"}, "pipeline")


@dataclass(frozen=True, slots=True)
class EffectiveThemeMembership:
    canonical_symbol: str
    market: str
    company_key: str | None
    company_count_eligible: bool
    origins: tuple[str, ...]


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
    membership: tuple[EffectiveThemeMembership, ...] = ()
    identity_version: int = 0
    identity_policy_version: str = ""
    registry_version: int = 0
    input_sessions: tuple[tuple[str, date | None], ...] = ()
    feature_run_ids: tuple[tuple[str, int | None], ...] = ()
    selected_listings: tuple[tuple[str, str, str], ...] = ()
    price_sessions: tuple[tuple[str, date | None], ...] = ()
    company_observations: tuple[tuple[str, str, str, tuple[Decimal, ...]], ...] = ()
    benchmark_return_1m: Decimal | None = None
    benchmark_candidates: tuple[str, ...] = ()
    benchmark_registry_version: str = ""
    benchmark_selection: str = ""

    def __post_init__(self) -> None:
        for field in ("theme_key", "benchmark_symbol", "basket_version"):
            _required(getattr(self, field), field)
        _choice(self.market, SUPPORTED_MARKETS, "market")
        if not isinstance(self.session_date, date) or isinstance(self.session_date, datetime):
            raise ValueError("invalid_session_date")
        if self.accepted_company_count < 0:
            raise ValueError("negative_accepted_company_count")
        for field in ("membership", "input_sessions", "feature_run_ids", "selected_listings", "price_sessions", "company_observations", "benchmark_candidates"):
            _deeply_immutable(getattr(self, field), field)
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
class DailyFreshness:
    required_session: date | None
    actual_session: date | None
    fresh: bool
    reason: str | None = None

    @property
    def signal_state_value(self) -> bool | None:
        """Unknown calendar/session is unavailable; known invalid/stale is false."""
        if (self.required_session is None or self.actual_session is None
                or self.reason in {"calendar_unavailable", "listing_mic_unknown"}):
            return None
        return self.fresh


@dataclass(frozen=True, slots=True)
class PinnedFeatureRun:
    market: str
    run_id: int | None


@dataclass(frozen=True, slots=True)
class ConfirmationFacts:
    feature_run_id: int | None
    market_exposure_id: int | None
    feature_freshness: DailyFreshness
    market_freshness: DailyFreshness
    benchmark_symbol: str | None
    rs_rating: Decimal | None
    market_exposure: Decimal | None
    reasons: tuple[str, ...]
    setup_score: Decimal | None = None
    setup_ready: bool | None = None
    rs_rating_1m: Decimal | None = None
    rs_rating_3m: Decimal | None = None
    liquidity_eligible: bool | None = None
    avg_dollar_volume: Decimal | None = None

    def __post_init__(self):
        _deeply_immutable(self.reasons, "reasons")

    def to_signal_state_input(self, *, resolved: bool, active: bool, market: str | None,
                              security_kind: str = "stock") -> SignalStateInput:
        """Combine caller-owned security identity with these exact frozen checks."""
        return SignalStateInput(resolved, active, market, security_kind,
            self.feature_freshness.signal_state_value, self.market_freshness.signal_state_value,
            self.liquidity_eligible, self.setup_ready, self.setup_score, self.market_exposure)


@dataclass(frozen=True, slots=True)
class MarketConfirmationContext:
    market: str
    observed_at: datetime
    exposure_id: int | None
    freshness: DailyFreshness
    exposure_score: Decimal | None
    benchmark_symbol: str | None
    benchmark_candidates: tuple[str, ...]
    benchmark_registry_version: str

    def __post_init__(self):
        _utc(self.observed_at, "observed_at")
        _deeply_immutable(self.benchmark_candidates, "benchmark_candidates")


@dataclass(frozen=True, slots=True)
class GroupConfirmationContext:
    session_date: date | None
    formula_version: str | None
    market_rs_run_id: int | None
    cohort: tuple[tuple[str, int], ...]
    row_ids: tuple[int, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        _deeply_immutable(self.cohort, "cohort")
        _deeply_immutable(self.row_ids, "row_ids")


@dataclass(frozen=True, slots=True)
class MarketConfirmationBatch:
    pinned_run: PinnedFeatureRun
    market_context: MarketConfirmationContext
    group_context: GroupConfirmationContext
    inputs: tuple[ConfirmationInput, ...]
    facts: tuple[tuple[str, ConfirmationFacts], ...]
    theme_evidence: tuple[ThemeMarketEvidence, ...] = ()
    theme_reasons: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        for field in ("inputs", "facts", "theme_evidence", "theme_reasons"):
            _deeply_immutable(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class SocialPublicationContext:
    market_batches: tuple[MarketConfirmationBatch, ...]
    formula_version: str = "social-signal-v1"
    extraction_versions: tuple[tuple[str, str, str], ...] = ()
    source_progress: tuple[tuple[str, str | None], ...] = ()
    candidates: tuple[CandidatePublicationContext, ...] = ()

    def __post_init__(self):
        for field in ("market_batches", "extraction_versions", "source_progress", "candidates"):
            _deeply_immutable(getattr(self, field), field)
        if any(not isinstance(batch, MarketConfirmationBatch) for batch in self.market_batches):
            raise TypeError("invalid_market_batch")
        markets = tuple(batch.pinned_run.market for batch in self.market_batches)
        if len(set(markets)) != len(markets):
            raise ValueError("duplicate_market_context")
        _required(self.formula_version, "formula_version")
        if any(not isinstance(v, CandidatePublicationContext) for v in self.candidates):
            raise TypeError("invalid_candidate_context")
        if len({(v.candidate_key, v.window_days) for v in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate_candidate_context")


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
class CandidatePublicationContext:
    candidate_key: str
    window_days: int
    state_input: SignalStateInput
    state_decision: SignalStateDecision
    social_result: SocialScoreResult | None
    confirmation: ComponentScore | None

    def __post_init__(self):
        _required(self.candidate_key, "candidate_key")
        if self.window_days not in {1, 7, 14}:
            raise ValueError("invalid_window")
        if self.social_result is not None and (self.social_result.candidate_key != self.candidate_key
                or self.social_result.market != self.state_input.market):
            raise ValueError("candidate_context_identity_mismatch")


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
    window_days: int = 14
    mention_count: int = 0
    observed_list_count: int = 0
    enabled_list_count: int = 0
    normalization_scope: str = "global"
    formula_version: str = "social-signal-v1"

    def __post_init__(self) -> None:
        _deeply_immutable(self.pinned_inputs, "pinned_inputs")
        _deeply_immutable(self.coverage, "coverage")
        if self.window_days not in {1, 7, 14}:
            raise ValueError("invalid_window")
        portions_available = self.social_score is not None and self.confirmation_score is not None
        if portions_available != (self.queue_score is not None):
            raise ValueError("inconsistent_snapshot_scores")


@dataclass(frozen=True, slots=True)
class PreparedSocialPublication:
    run_id: str
    registry_version: int
    as_of: datetime
    work_ids: tuple[int, ...]
    rows: tuple[SocialSnapshotRecord, ...]
    theme_evidence: tuple[ThemeMarketEvidence, ...] = ()
    context: SocialPublicationContext | None = None

    def __post_init__(self):
        _utc(self.as_of, "as_of")
        for field in ("work_ids", "rows", "theme_evidence"):
            _deeply_immutable(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class ReplayInput:
    content_item_id: int
    input_hash: str
    disposition: str


@dataclass(frozen=True, slots=True)
class SocialReplayManifest:
    saved_run_id: str
    historical_work_ids: tuple[int, ...]
    participating_source_ids: tuple[str, ...]
    missing_sources: tuple[tuple[str, str], ...]
    required_inputs: tuple[ReplayInput, ...]
    carry_in_work_ids: tuple[int, ...]
    coverage_reasons: tuple[str, ...]

    def __post_init__(self):
        for field in ("historical_work_ids", "participating_source_ids", "missing_sources",
                      "required_inputs", "carry_in_work_ids", "coverage_reasons"):
            _deeply_immutable(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class SocialCurrentInputManifest:
    required_inputs: tuple[ReplayInput, ...]
    carry_in_work_ids: tuple[int, ...]
    audit_work_ids: tuple[int, ...]
    coverage_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        for field in ("required_inputs", "carry_in_work_ids", "audit_work_ids", "coverage_reasons"):
            _deeply_immutable(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class SavedSocialRunInputs:
    run_id: str
    as_of: datetime
    registry_version: int
    batches: tuple[SocialSourceBatch, ...]
    content_ids: tuple[tuple[str, int], ...]
    work_ids: tuple[int, ...]
    replay_manifest: SocialReplayManifest | None = None
    current_manifest: SocialCurrentInputManifest | None = None

    def __post_init__(self):
        _utc(self.as_of, "as_of")
        for field in ("batches", "content_ids", "work_ids"):
            _deeply_immutable(getattr(self, field), field)


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

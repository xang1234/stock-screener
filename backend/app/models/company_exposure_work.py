"""Research work, provider attempts and typed resource accounting.

Immutable history (append-only):
  runtime policy revisions, request envelopes, request events, candidates,
  stage input manifests, reservations and reservation events, provider
  attempts/results, reusable success artifacts and coverage items.

Operational state (mutable, never evidence):
  leased work items, resource-pool counters and root-budget counters. These
  are caches guarded by row locks; the append-only ledger explains them.

Amounts are integers in each unit's base quantity: requests, reported
tokens, blob bytes, or currency micro-units (``amount_micros``) with an
explicit currency code. No dollar amount is ever derived from tokens.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)

from app.database import Base
from app.models.company_exposure_common import append_only, created_at, uuid_pk

_UNITS = "'requests','reported_tokens','currency_amount','blob_bytes'"
_RESERVATION_STATES = (
    "'reserved','dispatched','released','reconciled','uncertain',"
    "'expired_uncertain','paused_allowance'"
)
_DISPATCH_PHASES = "'pre_dispatch','dispatched','uncertain'"


@append_only
class ExposureRuntimePolicyRevision(Base):
    """Versioned operator policy: limits, routes, costing, permissions, stages."""

    __tablename__ = "company_exposure_runtime_policy_revisions"

    id = uuid_pk()
    namespace = Column(String(40), nullable=False)
    subject_key = Column(String(200), nullable=False)
    revision_number = Column(Integer, nullable=False)
    payload = Column(JSON, nullable=False)
    payload_hash = Column(String(64), nullable=False)
    parent_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_runtime_policy_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    approval_state = Column(String(16), nullable=False)
    principal = Column(String(200), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "namespace IN ('research_limits','subscription_route','search_cost',"
            "'acquisition_permission','theme_research','feature_stage')",
            name="ck_cx_policy_namespace",
        ),
        CheckConstraint(
            "approval_state IN ('proposed','approved','rejected')",
            name="ck_cx_policy_approval",
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_policy_revision"),
        UniqueConstraint(
            "namespace", "subject_key", "revision_number", name="uq_cx_policy_revision"
        ),
    )


@append_only
class ExposureResearchRequest(Base):
    """Immutable request envelope; one intentional investigation."""

    __tablename__ = "company_exposure_research_requests"

    id = uuid_pk()
    root_request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=True,
    )
    parent_request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=True,
    )
    kind = Column(String(16), nullable=False)
    requester_principal = Column(String(200), nullable=False)
    idempotency_namespace = Column(String(200), nullable=False)
    idempotency_key = Column(String(200), nullable=False)
    security_id = Column(
        Integer, ForeignKey("stock_universe.id", ondelete="RESTRICT"), nullable=True
    )
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=True,
    )
    economic_theme_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("economic_themes.id", ondelete="RESTRICT"),
        nullable=False,
    )
    market = Column(String(8), nullable=True)
    supplied_links = Column(JSON, nullable=False)
    requested_limits = Column(JSON, nullable=False)
    policy_revision_ids = Column(JSON, nullable=False)
    trigger_origin = Column(String(80), nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "kind IN ('verify','refresh','discover')", name="ck_cx_request_kind"
        ),
        CheckConstraint(
            "security_id IS NOT NULL OR issuer_id IS NOT NULL OR kind = 'discover'",
            name="ck_cx_request_subject",
        ),
        CheckConstraint(
            "(parent_request_id IS NULL) = (root_request_id IS NULL)",
            name="ck_cx_request_root_parent",
        ),
        UniqueConstraint(
            "idempotency_namespace", "idempotency_key", name="uq_cx_request_idempotency"
        ),
        Index("ix_cx_request_root", "root_request_id"),
    )

    @property
    def effective_root_id(self):
        return self.root_request_id or self.id


@append_only
class ResearchEvent(Base):
    __tablename__ = "company_exposure_research_events"

    id = uuid_pk()
    request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    sequence = Column(Integer, nullable=False)
    state = Column(String(40), nullable=False)
    detail = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint("sequence > 0", name="ck_cx_event_sequence"),
        UniqueConstraint("request_id", "sequence", name="uq_cx_event_sequence"),
    )


@append_only
class ResearchCandidate(Base):
    __tablename__ = "company_exposure_research_candidates"

    id = uuid_pk()
    root_request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    economic_theme_id = Column(Uuid(as_uuid=True), nullable=False)
    issuer_id = Column(Uuid(as_uuid=True), nullable=True)
    security_id = Column(Integer, nullable=True)
    seed_provenance = Column(JSON, nullable=False)
    rationale = Column(Text, nullable=False)
    state = Column(String(40), nullable=False)
    created_at = created_at()


class ResearchWorkItem(Base):
    """Leased stage work. Operational state only; results live in history."""

    __tablename__ = "company_exposure_work_items"

    id = uuid_pk()
    request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    root_request_id = Column(Uuid(as_uuid=True), nullable=False)
    stage = Column(String(40), nullable=False)
    input_hash = Column(String(64), nullable=False)
    policy_bundle_version = Column(String(80), nullable=False)
    priority = Column(Integer, nullable=False, default=0)
    status = Column(String(24), nullable=False)
    pause_reason = Column(String(80), nullable=True)
    available_at = Column(DateTime(timezone=True), nullable=False)
    lease_token = Column(Uuid(as_uuid=True), nullable=True)
    lease_owner = Column(String(200), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    claim_count = Column(Integer, nullable=False, default=0)
    created_at = created_at()
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','leased','retryable','paused','completed',"
            "'cancelled','failed')",
            name="ck_cx_work_status",
        ),
        UniqueConstraint(
            "request_id",
            "stage",
            "input_hash",
            "policy_bundle_version",
            name="uq_cx_work_key",
        ),
        Index("ix_cx_work_claim", "status", "available_at"),
    )


# Canonical name used by the plan's interface list.
ResearchWorkLease = ResearchWorkItem


@append_only
class ResearchInputManifest(Base):
    """Frozen inputs of one stage attempt: evidence, policies, identity, prior."""

    __tablename__ = "company_exposure_input_manifests"

    id = uuid_pk()
    request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    stage = Column(String(40), nullable=False)
    input_hash = Column(String(64), nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        UniqueConstraint(
            "request_id", "stage", "input_hash", name="uq_cx_input_manifest"
        ),
    )


class ResearchResourcePool(Base):
    """Shared capacity counter for one pool/unit/period (operational)."""

    __tablename__ = "company_exposure_resource_pools"

    id = uuid_pk()
    pool_key = Column(String(160), nullable=False)
    unit = Column(String(24), nullable=False)
    period = Column(String(32), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=True)
    capacity = Column(BigInteger, nullable=True)
    reserved_amount = Column(BigInteger, nullable=False, default=0)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = created_at()
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(f"unit IN ({_UNITS})", name="ck_cx_pool_unit"),
        CheckConstraint("reserved_amount >= 0", name="ck_cx_pool_reserved"),
        UniqueConstraint("pool_key", "unit", "period", name="uq_cx_pool_period"),
    )


class ResearchRootBudget(Base):
    """Cumulative per-root-job limit; children and retries share it."""

    __tablename__ = "company_exposure_root_budgets"

    id = uuid_pk()
    root_request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    budget_key = Column(String(80), nullable=False)
    limit_amount = Column(BigInteger, nullable=False)
    used_amount = Column(BigInteger, nullable=False, default=0)
    created_at = created_at()
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint("used_amount >= 0", name="ck_cx_root_used"),
        CheckConstraint("used_amount <= limit_amount", name="ck_cx_root_limit"),
        UniqueConstraint("root_request_id", "budget_key", name="uq_cx_root_budget"),
    )


@append_only
class ResearchReservation(Base):
    """A bounded claim on a pool (and optionally a root budget) before I/O."""

    __tablename__ = "company_exposure_reservations"

    id = uuid_pk()
    pool_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_resource_pools.id", ondelete="RESTRICT"),
        nullable=False,
    )
    root_request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=True,
    )
    root_budget_key = Column(String(80), nullable=True)
    unit = Column(String(24), nullable=False)
    amount = Column(BigInteger, nullable=False)
    currency = Column(String(8), nullable=True)
    period = Column(String(32), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=True)
    purpose = Column(String(80), nullable=False)
    logical_operation_key = Column(String(200), nullable=False)
    policy_revision_id = Column(Uuid(as_uuid=True), nullable=True)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(f"unit IN ({_UNITS})", name="ck_cx_reservation_unit"),
        CheckConstraint("amount >= 0", name="ck_cx_reservation_amount"),
        CheckConstraint(
            "(unit = 'currency_amount') = (currency IS NOT NULL)",
            name="ck_cx_reservation_currency",
        ),
        Index("ix_cx_reservation_pool", "pool_id"),
    )


@append_only
class ResearchReservationEvent(Base):
    __tablename__ = "company_exposure_reservation_events"

    id = uuid_pk()
    reservation_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_reservations.id", ondelete="RESTRICT"),
        nullable=False,
    )
    sequence = Column(Integer, nullable=False)
    state = Column(String(24), nullable=False)
    dispatch_phase = Column(String(16), nullable=True)
    actual_amount = Column(BigInteger, nullable=True)
    actual_known = Column(Boolean, nullable=False, default=False)
    detail = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            f"state IN ({_RESERVATION_STATES})", name="ck_cx_reservation_state"
        ),
        CheckConstraint(
            f"dispatch_phase IS NULL OR dispatch_phase IN ({_DISPATCH_PHASES})",
            name="ck_cx_reservation_phase",
        ),
        CheckConstraint("sequence > 0", name="ck_cx_reservation_event_sequence"),
        UniqueConstraint(
            "reservation_id", "sequence", name="uq_cx_reservation_event_sequence"
        ),
    )


@append_only
class ResearchProviderAttempt(Base):
    __tablename__ = "company_exposure_provider_attempts"

    id = uuid_pk()
    logical_operation_key = Column(String(200), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=True,
    )
    root_request_id = Column(Uuid(as_uuid=True), nullable=True)
    operation = Column(String(80), nullable=False)
    route = Column(String(80), nullable=False)
    model = Column(String(120), nullable=False)
    parameters = Column(JSON, nullable=False)
    input_hash = Column(String(64), nullable=False)
    policy_hash = Column(String(64), nullable=False)
    reservation_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_reservations.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at = created_at()

    __table_args__ = (
        CheckConstraint("attempt_number > 0", name="ck_cx_attempt_number"),
        UniqueConstraint(
            "logical_operation_key", "attempt_number", name="uq_cx_attempt_number"
        ),
    )


@append_only
class ResearchProviderResult(Base):
    __tablename__ = "company_exposure_provider_results"

    id = uuid_pk()
    attempt_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_provider_attempts.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    outcome = Column(String(24), nullable=False)
    dispatch_phase = Column(String(16), nullable=False)
    failure_code = Column(String(80), nullable=True)
    provider_request_id = Column(String(200), nullable=True)
    reported_usage = Column(JSON, nullable=False)
    usage_known = Column(Boolean, nullable=False, default=False)
    response_hash = Column(String(64), nullable=True)
    retry_after_seconds = Column(Integer, nullable=True)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "outcome IN ('success','retryable_failure','uncertain','terminal_failure')",
            name="ck_cx_result_outcome",
        ),
        CheckConstraint(
            f"dispatch_phase IN ({_DISPATCH_PHASES})", name="ck_cx_result_phase"
        ),
    )


@append_only
class ResearchArtifact(Base):
    """A reusable successful result; failures never occupy this key."""

    __tablename__ = "company_exposure_artifacts"

    id = uuid_pk()
    operation = Column(String(80), nullable=False)
    input_hash = Column(String(64), nullable=False)
    policy_hash = Column(String(64), nullable=False)
    model_identity = Column(String(160), nullable=False)
    result_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_provider_results.id", ondelete="RESTRICT"),
        nullable=True,
    )
    payload = Column(JSON, nullable=False)
    payload_hash = Column(String(64), nullable=False)
    created_at = created_at()

    __table_args__ = (
        UniqueConstraint(
            "operation",
            "input_hash",
            "policy_hash",
            "model_identity",
            name="uq_cx_artifact_key",
        ),
    )


@append_only
class ResearchCoverageItem(Base):
    """What was searched/fetched/processed and with what outcome."""

    __tablename__ = "company_exposure_coverage_items"

    id = uuid_pk()
    request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=False,
    )
    stage = Column(String(40), nullable=False)
    route = Column(String(120), nullable=False)
    subject = Column(String(512), nullable=True)
    outcome = Column(String(40), nullable=False)
    reason = Column(String(120), nullable=True)
    detail = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (Index("ix_cx_coverage_request", "request_id"),)

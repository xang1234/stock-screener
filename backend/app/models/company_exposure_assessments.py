"""Claim-level exposure assessments and safety holds.

* ``ExposureClaim`` is a stable proposition identity: issuer, theme, kind,
  product/activity key and reporting scope. A materially different
  proposition is a new claim, never a reinterpretation of an old ID.
* ``ExposureClaimRevision`` is sealable: evidence links and a materiality
  measure are attached while unsealed; after sealing, neither the revision
  nor its children can change.
* ``IssuerThemeAssessment`` is the stable dossier for one issuer–theme pair;
  ``AssessmentRevision`` selects exact claim revisions (old still-valid
  claims can be carried forward next to newer ones).
* Composite foreign keys stop a selection from crossing issuer/theme scope.
* Use holds are append-only apply/lift revisions scoped to one subject, so a
  disputed customer relationship cannot hold unrelated role claims.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)

from app.database import Base
from app.models.company_exposure_common import (
    append_only,
    created_at,
    sealable,
    sealed_child,
    uuid_pk,
)

_CLAIM_KINDS = (
    "'participation','role','product_application','customer_relationship',"
    "'commercial_status','materiality','exposure_end'"
)
_SCOPES = "'issuer_consolidated','issuer_standalone','segment_or_subsidiary'"
_SUPPORT = (
    "'primary_explicit','primary_synthesis','secondary_reported',"
    "'inferred_unverified','unresolved'"
)
_CONCLUSIONS = "'supported','contradicted','disputed','unknown'"
_FRESHNESS = "'current','stale','undated'"
_COMMERCIAL = (
    "'research','announced','qualification','commercially_available',"
    "'shipping_or_operating','discontinued','unknown'"
)
_EVIDENCE_ROLES = (
    "'original_primary','original_secondary','derivative_not_independent_source',"
    "'retrieval_aid_only'"
)


@append_only
class ExposureClaim(Base):
    __tablename__ = "company_exposure_claims"

    id = uuid_pk()
    proposition_key = Column(String(64), nullable=False, unique=True)
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    economic_theme_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("economic_themes.id", ondelete="RESTRICT"),
        nullable=False,
    )
    claim_kind = Column(String(32), nullable=False)
    product_or_activity_key = Column(String(200), nullable=False)
    reporting_scope = Column(String(32), nullable=False)
    scope_label = Column(String(200), nullable=True)
    normalized_proposition = Column(Text, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(f"claim_kind IN ({_CLAIM_KINDS})", name="ck_cx_claim_kind"),
        CheckConstraint(f"reporting_scope IN ({_SCOPES})", name="ck_cx_claim_scope"),
        CheckConstraint(
            "(reporting_scope = 'segment_or_subsidiary') = (scope_label IS NOT NULL)",
            name="ck_cx_claim_scope_label",
        ),
        UniqueConstraint(
            "id", "issuer_id", "economic_theme_id", name="uq_cx_claim_scope"
        ),
        Index("ix_cx_claim_pair", "issuer_id", "economic_theme_id"),
    )


@sealable
class ExposureClaimRevision(Base):
    __tablename__ = "company_exposure_claim_revisions"

    id = uuid_pk()
    claim_id = Column(Uuid(as_uuid=True), nullable=False)
    issuer_id = Column(Uuid(as_uuid=True), nullable=False)
    economic_theme_id = Column(Uuid(as_uuid=True), nullable=False)
    revision_number = Column(Integer, nullable=False)
    statement = Column(Text, nullable=False)
    evaluated_theme_fingerprint = Column(String(64), nullable=False)
    role = Column(String(80), nullable=True)
    commercial_status = Column(String(32), nullable=False)
    support_basis = Column(String(32), nullable=False)
    conclusion = Column(String(16), nullable=False)
    freshness_state = Column(String(16), nullable=False)
    hold_reasons = Column(JSON, nullable=False)
    effective_start = Column(DateTime(timezone=True), nullable=True)
    effective_end = Column(DateTime(timezone=True), nullable=True)
    reporting_period = Column(String(64), nullable=True)
    source_publication_time = Column(DateTime(timezone=True), nullable=True)
    first_available_to_app_at = Column(DateTime(timezone=True), nullable=True)
    assessed_at = Column(DateTime(timezone=True), nullable=False)
    supported_as_of = Column(DateTime(timezone=True), nullable=True)
    fresh_until = Column(DateTime(timezone=True), nullable=True)
    verification_policy_version = Column(String(80), nullable=False)
    model_attempt_refs = Column(JSON, nullable=False)
    supersedes_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_claim_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    supersession_kind = Column(String(24), nullable=True)
    status = Column(String(16), nullable=False, default="unsealed")
    semantic_hash = Column(String(64), nullable=True)
    sealed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = created_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["claim_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_claims.id",
                "company_exposure_claims.issuer_id",
                "company_exposure_claims.economic_theme_id",
            ],
            name="fk_cx_claim_revision_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            f"commercial_status IN ({_COMMERCIAL})", name="ck_cx_claim_commercial"
        ),
        CheckConstraint(f"support_basis IN ({_SUPPORT})", name="ck_cx_claim_support"),
        CheckConstraint(f"conclusion IN ({_CONCLUSIONS})", name="ck_cx_claim_conclusion"),
        CheckConstraint(
            f"freshness_state IN ({_FRESHNESS})", name="ck_cx_claim_freshness"
        ),
        CheckConstraint(
            "supersession_kind IS NULL OR supersession_kind IN ('correction','supersession')",
            name="ck_cx_claim_supersession",
        ),
        CheckConstraint(
            "(supersedes_revision_id IS NULL) = (supersession_kind IS NULL)",
            name="ck_cx_claim_supersession_pair",
        ),
        CheckConstraint(
            "status IN ('unsealed','sealed')", name="ck_cx_claim_revision_status"
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_claim_revision_number"),
        UniqueConstraint("claim_id", "revision_number", name="uq_cx_claim_revision"),
        UniqueConstraint(
            "id",
            "claim_id",
            "issuer_id",
            "economic_theme_id",
            name="uq_cx_claim_revision_scope",
        ),
    )


@sealed_child("claim_revision_id", ExposureClaimRevision)
@append_only
class ClaimEvidenceLink(Base):
    """One supporting or conflicting evidence edge of a claim revision."""

    __tablename__ = "company_exposure_claim_evidence_links"

    id = uuid_pk()
    claim_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_claim_revisions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    direction = Column(String(16), nullable=False)
    evidence_role = Column(String(48), nullable=False)
    passage_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_passages.id", ondelete="RESTRICT"),
        nullable=True,
    )
    derivative_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_passage_derivatives.id", ondelete="RESTRICT"),
        nullable=True,
    )
    premise_claim_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_claim_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    quote = Column(Text, nullable=True)
    locator = Column(JSON, nullable=False)
    attribution = Column(JSON, nullable=False)
    join_scope = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "direction IN ('supporting','conflicting')", name="ck_cx_link_direction"
        ),
        CheckConstraint(
            f"evidence_role IN ({_EVIDENCE_ROLES})", name="ck_cx_link_role"
        ),
        CheckConstraint(
            "(CASE WHEN passage_id IS NULL THEN 0 ELSE 1 END)"
            " + (CASE WHEN derivative_id IS NULL THEN 0 ELSE 1 END)"
            " + (CASE WHEN premise_claim_revision_id IS NULL THEN 0 ELSE 1 END) = 1",
            name="ck_cx_link_one_target",
        ),
        CheckConstraint(
            "derivative_id IS NULL OR evidence_role = 'derivative_not_independent_source'",
            name="ck_cx_link_derivative_role",
        ),
        CheckConstraint(
            "evidence_role <> 'original_primary' OR passage_id IS NOT NULL",
            name="ck_cx_link_primary_is_passage",
        ),
        CheckConstraint(
            "premise_claim_revision_id IS NULL OR premise_claim_revision_id <> claim_revision_id",
            name="ck_cx_link_no_self_premise",
        ),
        Index("ix_cx_link_revision", "claim_revision_id"),
    )


@sealed_child("claim_revision_id", ExposureClaimRevision)
@append_only
class MaterialityMeasure(Base):
    """Typed materiality; Decimal values stored as strings, never floats."""

    __tablename__ = "company_exposure_materiality_measures"

    id = uuid_pk()
    claim_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_claim_revisions.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    basis = Column(String(16), nullable=False)
    metric = Column(String(40), nullable=True)
    value_low = Column(String(64), nullable=True)
    value_high = Column(String(64), nullable=True)
    qualitative_label = Column(String(32), nullable=True)
    unit = Column(String(40), nullable=True)
    currency = Column(String(8), nullable=True)
    denominator_definition = Column(Text, nullable=True)
    reporting_scope = Column(String(32), nullable=False)
    scope_label = Column(String(200), nullable=True)
    period = Column(String(64), nullable=True)
    as_of = Column(DateTime(timezone=True), nullable=True)
    original_precision = Column(String(40), nullable=True)
    formula = Column(JSON, nullable=False)
    operand_refs = Column(JSON, nullable=False)
    supporting_passage_ids = Column(JSON, nullable=False)
    raw_reported = Column(JSON, nullable=False)
    hold_reasons = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "basis IN ('disclosed','calculated','qualitative','unknown')",
            name="ck_cx_materiality_basis",
        ),
        CheckConstraint(
            "qualitative_label IS NULL OR qualitative_label IN "
            "('core_business','explicitly_material','explicitly_limited','unknown')",
            name="ck_cx_materiality_label",
        ),
        CheckConstraint(
            "basis <> 'qualitative' OR qualitative_label IS NOT NULL",
            name="ck_cx_materiality_qualitative",
        ),
        CheckConstraint(
            "basis NOT IN ('disclosed','calculated') OR "
            "(metric IS NOT NULL AND value_low IS NOT NULL AND period IS NOT NULL)",
            name="ck_cx_materiality_numeric",
        ),
        CheckConstraint(
            f"reporting_scope IN ({_SCOPES})", name="ck_cx_materiality_scope"
        ),
    )


@append_only
class IssuerThemeAssessment(Base):
    """Stable dossier for one issuer–theme relationship."""

    __tablename__ = "company_exposure_assessments"

    id = uuid_pk()
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    economic_theme_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("economic_themes.id", ondelete="RESTRICT"),
        nullable=False,
    )
    created_at = created_at()

    __table_args__ = (
        UniqueConstraint(
            "issuer_id", "economic_theme_id", name="uq_cx_assessment_pair"
        ),
        UniqueConstraint(
            "id", "issuer_id", "economic_theme_id", name="uq_cx_assessment_scope"
        ),
    )


@sealable
class AssessmentRevision(Base):
    __tablename__ = "company_exposure_assessment_revisions"

    id = uuid_pk()
    assessment_id = Column(Uuid(as_uuid=True), nullable=False)
    issuer_id = Column(Uuid(as_uuid=True), nullable=False)
    economic_theme_id = Column(Uuid(as_uuid=True), nullable=False)
    revision_number = Column(Integer, nullable=False)
    input_manifest_hash = Column(String(64), nullable=False)
    input_manifest = Column(JSON, nullable=False)
    prior_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_assessment_revisions.id", ondelete="RESTRICT"),
        nullable=True,
    )
    request_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_research_requests.id", ondelete="RESTRICT"),
        nullable=True,
    )
    coverage = Column(JSON, nullable=False)
    unresolved_questions = Column(JSON, nullable=False)
    conflicts = Column(JSON, nullable=False)
    assessed_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(16), nullable=False, default="unsealed")
    semantic_hash = Column(String(64), nullable=True)
    sealed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = created_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["assessment_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_assessments.id",
                "company_exposure_assessments.issuer_id",
                "company_exposure_assessments.economic_theme_id",
            ],
            name="fk_cx_assessment_revision_scope",
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "status IN ('unsealed','sealed')", name="ck_cx_assessment_status"
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_assessment_revision"),
        UniqueConstraint(
            "assessment_id", "revision_number", name="uq_cx_assessment_revision"
        ),
        UniqueConstraint(
            "assessment_id", "input_manifest_hash", name="uq_cx_assessment_replay"
        ),
        UniqueConstraint(
            "id",
            "issuer_id",
            "economic_theme_id",
            name="uq_cx_assessment_revision_scope",
        ),
    )


@sealed_child("assessment_revision_id", AssessmentRevision)
@append_only
class AssessmentClaimSelection(Base):
    """One selected claim revision inside a dossier revision."""

    __tablename__ = "company_exposure_assessment_claim_selections"

    id = uuid_pk()
    assessment_revision_id = Column(Uuid(as_uuid=True), nullable=False)
    claim_id = Column(Uuid(as_uuid=True), nullable=False)
    claim_revision_id = Column(Uuid(as_uuid=True), nullable=False)
    issuer_id = Column(Uuid(as_uuid=True), nullable=False)
    economic_theme_id = Column(Uuid(as_uuid=True), nullable=False)
    carried_forward = Column(Boolean, nullable=False, default=False)
    selection_reason = Column(String(80), nullable=False)
    created_at = created_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["assessment_revision_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_assessment_revisions.id",
                "company_exposure_assessment_revisions.issuer_id",
                "company_exposure_assessment_revisions.economic_theme_id",
            ],
            name="fk_cx_selection_assessment_scope",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["claim_revision_id", "claim_id", "issuer_id", "economic_theme_id"],
            [
                "company_exposure_claim_revisions.id",
                "company_exposure_claim_revisions.claim_id",
                "company_exposure_claim_revisions.issuer_id",
                "company_exposure_claim_revisions.economic_theme_id",
            ],
            name="fk_cx_selection_claim_scope",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "assessment_revision_id", "claim_id", name="uq_cx_selection_claim"
        ),
    )


@append_only
class ExposureUseHoldRevision(Base):
    """Append-only apply/lift history for one subject-scoped hold stream."""

    __tablename__ = "company_exposure_use_holds"

    id = uuid_pk()
    subject_kind = Column(String(32), nullable=False)
    subject_id = Column(String(64), nullable=False)
    hold_kind = Column(String(40), nullable=False)
    stream_key = Column(String(200), nullable=False)
    revision_number = Column(Integer, nullable=False)
    action = Column(String(8), nullable=False)
    reason = Column(Text, nullable=False)
    actor = Column(String(200), nullable=False)
    lifted_hold_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_use_holds.id", ondelete="RESTRICT"),
        nullable=True,
    )
    lift_support = Column(JSON, nullable=True)
    detail = Column(JSON, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(
            "subject_kind IN ('claim','claim_revision','issuer','issuer_link',"
            "'assessment','security')",
            name="ck_cx_hold_subject",
        ),
        CheckConstraint(
            "hold_kind IN ('stale','undated','disputed','conflict','identity','policy',"
            "'language_ambiguity','source_integrity','exposure_end','manual')",
            name="ck_cx_hold_kind",
        ),
        CheckConstraint("action IN ('apply','lift')", name="ck_cx_hold_action"),
        CheckConstraint(
            "action = 'apply' OR (lifted_hold_id IS NOT NULL AND lift_support IS NOT NULL)",
            name="ck_cx_hold_lift_requires_support",
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_hold_revision"),
        UniqueConstraint("stream_key", "revision_number", name="uq_cx_hold_revision"),
        Index("ix_cx_hold_subject", "subject_kind", "subject_id"),
    )

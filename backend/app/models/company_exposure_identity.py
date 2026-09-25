"""Issuer registry and listing links for company-exposure research.

An issuer is the reporting/business entity being assessed, not a ticker.
Identifiers are scheme-scoped (a US CIK and a Taiwan company code never
collide). Issuer-to-security links are append-only numbered revisions per
security, so accepted/proposed/rejected history is retained.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)

from app.database import Base
from app.models.company_exposure_common import append_only, created_at, uuid_pk

_LINK_STATES = "'proposed','accepted','rejected','review_required'"
_ACCEPTANCE_POLICIES = (
    "'administrator_reviewed','official_registry_single_listing',"
    "'legacy_attestation_import'"
)


@append_only
class ExposureIssuer(Base):
    """Stable issuer identity. Descriptive names live in evidence/revisions."""

    __tablename__ = "company_exposure_issuers"

    id = uuid_pk()
    provenance = Column(JSON, nullable=False)
    created_by = Column(String(200), nullable=False)
    created_at = created_at()


@append_only
class IssuerIdentifierRevision(Base):
    """One official identifier claim for an issuer, scheme- and market-scoped."""

    __tablename__ = "company_exposure_issuer_identifier_revisions"

    id = uuid_pk()
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    market = Column(String(8), nullable=False)
    scheme = Column(String(40), nullable=False)
    value = Column(String(120), nullable=False)
    revision_number = Column(Integer, nullable=False)
    state = Column(String(24), nullable=False)
    acceptance_policy = Column(String(48), nullable=True)
    evidence = Column(JSON, nullable=False)
    actor = Column(String(200), nullable=False)
    reason = Column(Text, nullable=False)
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(f"state IN ({_LINK_STATES})", name="ck_cx_identifier_state"),
        CheckConstraint(
            f"acceptance_policy IS NULL OR acceptance_policy IN ({_ACCEPTANCE_POLICIES})",
            name="ck_cx_identifier_policy",
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_identifier_revision"),
        UniqueConstraint(
            "market",
            "scheme",
            "value",
            "revision_number",
            name="uq_cx_identifier_revision",
        ),
        Index("ix_cx_identifier_issuer", "issuer_id"),
    )


@append_only
class IssuerSecurityLinkRevision(Base):
    """Numbered revision of which issuer a listed security belongs to."""

    __tablename__ = "company_exposure_issuer_security_links"

    id = uuid_pk()
    security_id = Column(
        Integer, ForeignKey("stock_universe.id", ondelete="RESTRICT"), nullable=False
    )
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    revision_number = Column(Integer, nullable=False)
    state = Column(String(24), nullable=False)
    acceptance_policy = Column(String(48), nullable=False)
    link_scope = Column(String(40), nullable=False, default="listing")
    actor = Column(String(200), nullable=False)
    reason = Column(Text, nullable=False)
    evidence = Column(JSON, nullable=False)
    # Legal name / identifiers / ticker as observed, so historical counts do
    # not depend on mutable StockUniverse rows.
    snapshot = Column(JSON, nullable=False)
    prior_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuer_security_links.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at = created_at()

    __table_args__ = (
        CheckConstraint(f"state IN ({_LINK_STATES})", name="ck_cx_link_state"),
        CheckConstraint(
            f"acceptance_policy IN ({_ACCEPTANCE_POLICIES})", name="ck_cx_link_policy"
        ),
        CheckConstraint("revision_number > 0", name="ck_cx_link_revision"),
        UniqueConstraint(
            "security_id", "revision_number", name="uq_cx_link_security_revision"
        ),
        Index("ix_cx_link_issuer", "issuer_id"),
    )


@append_only
class LegacyIssuerAttestationBridge(Base):
    """Exact provenance of an imported Social administrator attestation."""

    __tablename__ = "company_exposure_legacy_attestation_bridges"

    id = uuid_pk()
    configuration_version = Column(Integer, nullable=False)
    configuration_hash = Column(String(64), nullable=False)
    policy_version = Column(String(80), nullable=False)
    legacy_company_id = Column(String(500), nullable=False)
    legacy_symbol = Column(String(20), nullable=False)
    verified_at = Column(String(64), nullable=False)
    security_id = Column(
        Integer, ForeignKey("stock_universe.id", ondelete="RESTRICT"), nullable=False
    )
    verification_reference = Column(Text, nullable=True)
    issuer_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    link_revision_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("company_exposure_issuer_security_links.id", ondelete="RESTRICT"),
        nullable=False,
    )
    audit_provenance = Column(JSON, nullable=False)
    actor = Column(String(200), nullable=False)
    created_at = created_at()

    __table_args__ = (
        UniqueConstraint(
            "configuration_version",
            "configuration_hash",
            "legacy_company_id",
            "security_id",
            name="uq_cx_legacy_bridge_import",
        ),
    )

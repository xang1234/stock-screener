"""Deterministic inputs for company-exposure tests.

Factories build typed inputs and valid database rows only. They never
return an expected business conclusion and never dispatch on a case ID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid5

FIXED_NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
_NAMESPACE = UUID("6f5a3c2e-9d7b-4e2a-8c1f-0b3d5e7a9c11")


def fixed_uuid(label: str) -> UUID:
    """Stable UUID for a readable label, so failures name their inputs."""

    return uuid5(_NAMESPACE, label)


@dataclass
class FixedClock:
    """Injectable UTC clock; tests advance it explicitly."""

    now_value: datetime = field(default=FIXED_NOW)

    def now(self) -> datetime:
        return self.now_value

    def advance(self, **delta) -> datetime:
        self.now_value = self.now_value + timedelta(**delta)
        return self.now_value

    def advance_to(self, when: datetime) -> datetime:
        if when.tzinfo is None:
            raise ValueError("clock requires timezone-aware datetimes")
        self.now_value = when
        return self.now_value


def make_security(db, symbol: str, *, market: str = "US", exchange: str = "NASDAQ"):
    """Persist a StockUniverse listing (the existing security authority)."""

    from app.models.stock_universe import StockUniverse

    row = StockUniverse(
        symbol=symbol,
        name=f"{symbol} Listing",
        market=market,
        exchange=exchange,
        is_active=True,
        status="active",
    )
    db.add(row)
    db.flush()
    return row


def make_issuer(db, label: str = "issuer", *, created_by: str = "test:factory"):
    from app.models.company_exposure import ExposureIssuer

    row = ExposureIssuer(
        id=fixed_uuid(label), provenance={"label": label}, created_by=created_by
    )
    db.add(row)
    db.flush()
    return row


def make_document(db, identity_key: str, *, provider: str = "sec", issuer=None):
    from app.models.company_exposure import ExposureDocument

    row = ExposureDocument(
        id=fixed_uuid(f"document:{identity_key}"),
        identity_key=identity_key,
        provider=provider,
        source_kind="periodic_report",
        market="US",
        issuer_id=issuer.id if issuer is not None else None,
    )
    db.add(row)
    db.flush()
    return row


def make_revision(db, document, content: bytes, *, published_at=None, period=None):
    import hashlib

    from app.models.company_exposure import ExposureDocumentRevision

    digest = hashlib.sha256(content).hexdigest()
    row = ExposureDocumentRevision(
        document_id=document.id,
        content_hash=digest,
        media_type="text/html",
        byte_length=len(content),
        blob_key=f"sha256/{digest[:2]}/{digest}",
        published_at=published_at,
        reporting_period=period,
        first_available_at=FIXED_NOW,
        correction_identity={},
        document_metadata={},
    )
    db.add(row)
    db.flush()
    return row


def make_theme(db, label: str = "theme", *, created_by: str = "test:factory"):
    """Persist a stable Economic Theme identity (UUID only; no revision)."""

    from app.models.economic_taxonomy import EconomicTheme

    row = EconomicTheme(id=fixed_uuid(f"theme:{label}"), created_by=created_by)
    db.add(row)
    db.flush()
    return row


def make_claim(
    db,
    issuer,
    theme,
    *,
    kind: str = "role",
    product_key: str = "hbm-test-equipment",
    scope: str = "issuer_consolidated",
    scope_label: str | None = None,
):
    from app.domain.company_exposure.contracts import content_hash
    from app.models.company_exposure import ExposureClaim

    key = content_hash(
        {
            "issuer": issuer.id,
            "theme": theme.id,
            "kind": kind,
            "product": product_key,
            "scope": scope,
            "scope_label": scope_label,
        }
    )
    row = ExposureClaim(
        proposition_key=key,
        issuer_id=issuer.id,
        economic_theme_id=theme.id,
        claim_kind=kind,
        product_or_activity_key=product_key,
        reporting_scope=scope,
        scope_label=scope_label,
        normalized_proposition=f"{kind}:{product_key}",
    )
    db.add(row)
    db.flush()
    return row


def make_claim_revision(
    db,
    claim,
    *,
    number: int = 1,
    supported_as_of=None,
    period: str | None = None,
    support_basis: str = "primary_explicit",
    commercial_status: str = "shipping_or_operating",
    seal: bool = True,
):
    from app.models.company_exposure import ExposureClaimRevision

    row = ExposureClaimRevision(
        claim_id=claim.id,
        issuer_id=claim.issuer_id,
        economic_theme_id=claim.economic_theme_id,
        revision_number=number,
        statement=claim.normalized_proposition,
        evaluated_theme_fingerprint="f" * 64,
        commercial_status=commercial_status,
        support_basis=support_basis,
        conclusion="supported",
        freshness_state="current",
        hold_reasons=[],
        reporting_period=period,
        assessed_at=FIXED_NOW,
        supported_as_of=supported_as_of,
        verification_policy_version="verification-v1",
        model_attempt_refs=[],
        status="unsealed",
    )
    db.add(row)
    db.flush()
    if seal:
        seal_row(db, row)
    return row


def seal_row(db, row):
    """Seal a sealable row with a hash of its identity (test helper)."""

    from app.domain.company_exposure.contracts import content_hash

    row.status = "sealed"
    row.semantic_hash = content_hash({"id": row.id})
    row.sealed_at = FIXED_NOW
    db.flush()
    return row

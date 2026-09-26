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

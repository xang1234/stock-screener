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


class FakeGoTransport:
    """httpx transport replaying queued Go responses; records every request."""

    def __init__(self):
        import httpx

        self._queue: list = []
        self.requests: list = []
        self.transport = httpx.MockTransport(self._handle)

    def queue_status(self, status: int, headers: dict | None = None):
        self._queue.append(("status", status, headers or {}))

    def queue_json(self, content: dict, *, usage: dict | None = None):
        self._queue.append(("json", content, usage))

    def queue_exception(self, error: Exception):
        self._queue.append(("raise", error, None))

    def _handle(self, request):
        import json

        import httpx

        self.requests.append(
            type(
                "Recorded",
                (),
                {"url": str(request.url), "json": json.loads(request.content)},
            )()
        )
        if not self._queue:
            raise AssertionError("unexpected provider call")
        kind, value, extra = self._queue.pop(0)
        if kind == "raise":
            raise value
        if kind == "status":
            return httpx.Response(value, headers=extra)
        body = {
            "id": f"resp-{len(self.requests)}",
            "model": "kimi-k2.6",
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
            ],
        }
        if extra is not None:
            body["usage"] = extra
        return httpx.Response(200, json=body)


class FakeRateGate:
    """Records provider pacing acquisitions; optionally reports an outage."""

    def __init__(self, *, unavailable: bool = False):
        self.provider_names: list[str] = []
        self.keys: list[str] = []
        self.unavailable = unavailable

    def acquire(self, provider, market=None, timeout_s=60.0):
        from app.services.company_exposure.pacing import PacingUnavailable, RateTicket

        if self.unavailable:
            raise PacingUnavailable("distributed_pacing_unavailable")
        key = f"{provider}:{(market or 'shared').lower()}"
        self.provider_names.append(provider)
        self.keys.append(key)
        return RateTicket(provider=provider, key=key, waited_seconds=0.0)


def make_text_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid text PDF (Helvetica) with one string per page.

    Synthetic test input only; never counted as a company disclosure.
    """

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    page_ids = [3 + 2 * index for index in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    font_id = 3 + 2 * len(pages)
    for index, text in enumerate(pages):
        content_id = page_ids[index] + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode()
        )
        lines = text.split("\n")
        stream_lines = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
        for line in lines:
            stream_lines.append(f"({escape(line)}) Tj T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1")
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(output)

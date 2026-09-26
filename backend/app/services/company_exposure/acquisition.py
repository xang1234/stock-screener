"""Bounded, permitted acquisition of original documents.

Order for one target: charge the root document budget → reserve worst-case
storage bytes → commit (no locks held) → for each HTTP hop, acquire shared
provider pacing → fetch → sniff content → retain.

* Identical bytes add a capture/check event to the existing revision; they
  never create a revision or advance any business-evidence date.
* New bytes become a new immutable revision whose publication/reporting
  metadata comes from the target, not from the download time.
* A 404, block, size limit, pacing outage or full store is a typed coverage
  gap. It never becomes negative exposure evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    CaptureResult,
    CoverageItem,
    CoverageOutcome,
    DocumentTarget,
    ResearchLimits,
    bytes_hash,
    utc_now,
)
from app.infra.db.repositories.company_exposure_work_repo import (
    ROOT_DOCUMENTS,
    ReservationLedger,
)
from app.models.company_exposure import (
    DocumentCaptureEvent,
    ExposureDocument,
    ExposureDocumentRevision,
)
from app.services.company_exposure.network import (
    FetchRequest,
    PublicDocumentTransport,
    sniff_media_type,
)
from app.services.company_exposure.pacing import PacingUnavailable, ResearchRateGate
from app.services.company_exposure.storage import OriginalStore, storage_lock

_GAP_OUTCOMES = {
    "http_status_404": CoverageOutcome.NO_MATCHING_DOCUMENT,
    "http_status_410": CoverageOutcome.NO_MATCHING_DOCUMENT,
    "http_status_401": CoverageOutcome.PERMISSION_UNAVAILABLE,
    "http_status_403": CoverageOutcome.PERMISSION_UNAVAILABLE,
    "rate_limited": CoverageOutcome.RATE_LIMITED,
    "blocked_destination": CoverageOutcome.BLOCKED_DESTINATION,
    "origin_not_permitted": CoverageOutcome.BLOCKED_DESTINATION,
    "url_credentials_forbidden": CoverageOutcome.BLOCKED_DESTINATION,
}


@dataclass(frozen=True, slots=True)
class JobBudgetRef:
    root_request_id: UUID | None


class DocumentAcquisitionRegistry:
    def __init__(
        self,
        session: Session,
        *,
        transport: PublicDocumentTransport,
        store: OriginalStore,
        rate_gate: ResearchRateGate,
        user_agent: str,
        limits: ResearchLimits | None = None,
        commit: Callable[[], None] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.session = session
        self.transport = transport
        self.store = store
        self.rate_gate = rate_gate
        self.user_agent = user_agent
        self.limits = limits or ResearchLimits()
        self.commit = commit or session.commit
        self.clock = clock
        self.ledger = ReservationLedger(session, clock=clock)

    # -- helpers --------------------------------------------------------------

    def _document(self, target: DocumentTarget) -> ExposureDocument:
        existing = self.session.execute(
            select(ExposureDocument).where(
                ExposureDocument.identity_key == target.identity_key
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        try:
            with self.session.begin_nested():
                document = ExposureDocument(
                    identity_key=target.identity_key,
                    provider=target.provider,
                    provider_document_id=target.provider_document_id,
                    canonical_url=target.url,
                    verified_origin=target.verified_origin,
                    publisher=target.publisher,
                    market=target.market,
                    source_kind=target.source_kind,
                    issuer_id=target.issuer_id,
                )
                self.session.add(document)
                self.session.flush()
                return document
        except IntegrityError:
            return self.session.execute(
                select(ExposureDocument).where(
                    ExposureDocument.identity_key == target.identity_key
                )
            ).scalar_one()

    def _capture(
        self,
        document: ExposureDocument,
        *,
        outcome: str,
        url: str,
        revision: ExposureDocumentRevision | None = None,
        changed: bool = False,
        status: int | None = None,
        content_hash: str | None = None,
        byte_length: int | None = None,
        metadata: dict | None = None,
    ) -> DocumentCaptureEvent:
        event = DocumentCaptureEvent(
            document_id=document.id,
            revision_id=None if revision is None else revision.id,
            outcome=outcome,
            changed=changed,
            sanitized_url=url,
            http_status=status,
            content_hash=content_hash,
            byte_length=byte_length,
            observed_metadata=metadata or {},
            retrieved_at=self.clock(),
        )
        self.session.add(event)
        self.session.flush()
        return event

    def _gap(
        self, target: DocumentTarget, code: str, *, document=None, capture=None
    ) -> CaptureResult:
        outcome = _GAP_OUTCOMES.get(code, CoverageOutcome.FETCH_FAILED)
        return CaptureResult(
            document_id=None if document is None else document.id,
            revision_id=None,
            capture_id=None if capture is None else capture.id,
            content_hash=None,
            changed=False,
            coverage=CoverageItem(
                route=target.adapter,
                outcome=outcome,
                reason=code,
                detail={"identity_key": target.identity_key},
            ),
        )

    # -- fetch ---------------------------------------------------------------------

    def fetch(self, target: DocumentTarget, budget: JobBudgetRef) -> CaptureResult:
        if not target.retention_permitted:
            return self._gap(target, "retention_not_permitted")
        document = self._document(target)
        if budget.root_request_id is not None:
            charged = self.ledger.consume_root(budget.root_request_id, ROOT_DOCUMENTS)
            if not charged.allowed:
                self.commit()
                return self._gap(target, charged.reason or "root_budget_exhausted")
        bound = min(
            target.max_bytes or self.limits.download_bytes_per_document,
            self.limits.download_bytes_per_document,
        )
        ticket = self.store.reserve(
            bound, purpose="document_download", operation_key=target.identity_key
        )
        if not ticket.allowed:
            self.commit()
            return CaptureResult(
                document.id,
                None,
                None,
                None,
                False,
                CoverageItem(
                    route=target.adapter,
                    outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                    reason="paused_storage",
                    detail={"required": bound, "available": ticket.available},
                ),
            )
        # No database locks are held across pacing waits or network I/O.
        self.commit()

        def pace(_url: str) -> None:
            self.rate_gate.acquire(target.rate_provider, target.rate_market)

        try:
            response = self.transport.fetch_once(
                FetchRequest(
                    url=target.url,
                    allowed_hosts=target.allowed_hosts,
                    max_bytes=bound,
                    user_agent=self.user_agent,
                    accept=target.accept,
                ),
                before_request=pace,
            )
        except PacingUnavailable as exc:
            self.store.release(ticket, reason=exc.code)
            capture = self._capture(document, outcome=exc.code, url=target.url)
            self.commit()
            return self._gap(target, exc.code, document=document, capture=capture)

        if not response.ok or response.not_modified:
            self.store.release(ticket, reason=response.failure_code or "not_modified")
            capture = self._capture(
                document,
                outcome=response.failure_code or "not_modified",
                url=response.final_url,
                status=response.status,
                metadata={"hops": list(response.hops)},
            )
            self.commit()
            if response.not_modified:
                return self._unchanged(target, document, capture)
            return self._gap(target, response.failure_code, document=document, capture=capture)

        media_type, refusal = sniff_media_type(response.body, response.content_type)
        if refusal is not None:
            self.store.release(ticket, reason=refusal)
            capture = self._capture(
                document, outcome=refusal, url=response.final_url, status=response.status
            )
            self.commit()
            return self._gap(target, refusal, document=document, capture=capture)

        digest = bytes_hash(response.body)
        storage_lock(self.session, exclusive=False)
        existing = self.session.execute(
            select(ExposureDocumentRevision).where(
                ExposureDocumentRevision.document_id == document.id,
                ExposureDocumentRevision.content_hash == digest,
            )
        ).scalar_one_or_none()
        if existing is not None and self.store.exists(digest):
            self.store.release(ticket, reason="identical_content")
            capture = self._capture(
                document,
                outcome="unchanged",
                url=response.final_url,
                revision=existing,
                changed=False,
                status=response.status,
                content_hash=digest,
                byte_length=len(response.body),
            )
            self.commit()
            return CaptureResult(
                document.id,
                existing.id,
                capture.id,
                digest,
                False,
                CoverageItem(
                    route=target.adapter,
                    outcome=CoverageOutcome.COMPLETE_FOR_REQUESTED_SCOPE,
                    reason="unchanged",
                ),
            )
        blob = self.store.put(response.body, media_type, ticket)
        revision = existing or ExposureDocumentRevision(
            document_id=document.id,
            content_hash=digest,
            media_type=media_type,
            byte_length=len(response.body),
            blob_key=blob.key,
            language=target.metadata.get("language"),
            published_at=target.published_at,
            reporting_period=target.reporting_period,
            effective_at=target.effective_at,
            first_available_at=self.clock(),
            correction_identity=dict(target.correction_identity),
            document_metadata={
                **target.metadata,
                "declared_content_type": response.content_type,
                "etag": response.etag,
                "last_modified": response.last_modified,
            },
        )
        if existing is None:
            self.session.add(revision)
            self.session.flush()
        capture = self._capture(
            document,
            outcome="retrieved",
            url=response.final_url,
            revision=revision,
            changed=existing is None,
            status=response.status,
            content_hash=digest,
            byte_length=len(response.body),
            metadata={"hops": list(response.hops)},
        )
        self.commit()
        return CaptureResult(
            document.id,
            revision.id,
            capture.id,
            digest,
            existing is None,
            CoverageItem(
                route=target.adapter,
                outcome=CoverageOutcome.COMPLETE_FOR_REQUESTED_SCOPE,
                reason="retrieved",
            ),
        )

    def _unchanged(self, target, document, capture) -> CaptureResult:
        latest = self.session.execute(
            select(ExposureDocumentRevision)
            .where(ExposureDocumentRevision.document_id == document.id)
            .order_by(ExposureDocumentRevision.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        return CaptureResult(
            document.id,
            None if latest is None else latest.id,
            capture.id,
            None if latest is None else latest.content_hash,
            False,
            CoverageItem(
                route=target.adapter,
                outcome=CoverageOutcome.COMPLETE_FOR_REQUESTED_SCOPE,
                reason="not_modified",
            ),
        )

    def ingest_supplied(
        self,
        target: DocumentTarget,
        data: bytes,
        *,
        declared_content_type: str = "",
    ) -> CaptureResult:
        """Retain an administrator-supplied original (no network)."""

        document = self._document(target)
        media_type, refusal = sniff_media_type(data, declared_content_type)
        if refusal is not None:
            capture = self._capture(document, outcome=refusal, url=target.url)
            self.commit()
            return self._gap(target, refusal, document=document, capture=capture)
        if len(data) > self.limits.download_bytes_per_document:
            return self._gap(target, "response_size_limit", document=document)
        ticket = self.store.reserve(
            len(data), purpose="supplied_document", operation_key=target.identity_key
        )
        if not ticket.allowed:
            return CaptureResult(
                document.id, None, None, None, False,
                CoverageItem(
                    route=target.adapter,
                    outcome=CoverageOutcome.UNAVAILABLE_CAPABILITY,
                    reason="paused_storage",
                ),
            )
        digest = bytes_hash(data)
        storage_lock(self.session, exclusive=False)
        existing = self.session.execute(
            select(ExposureDocumentRevision).where(
                ExposureDocumentRevision.document_id == document.id,
                ExposureDocumentRevision.content_hash == digest,
            )
        ).scalar_one_or_none()
        if existing is not None and self.store.exists(digest):
            self.store.release(ticket, reason="identical_content")
            capture = self._capture(
                document, outcome="unchanged", url=target.url, revision=existing,
                content_hash=digest, byte_length=len(data),
            )
            self.commit()
            return CaptureResult(
                document.id, existing.id, capture.id, digest, False,
                CoverageItem(
                    route=target.adapter,
                    outcome=CoverageOutcome.COMPLETE_FOR_REQUESTED_SCOPE,
                    reason="unchanged",
                ),
            )
        blob = self.store.put(data, media_type, ticket)
        revision = existing or ExposureDocumentRevision(
            document_id=document.id,
            content_hash=digest,
            media_type=media_type,
            byte_length=len(data),
            blob_key=blob.key,
            language=target.metadata.get("language"),
            published_at=target.published_at,
            reporting_period=target.reporting_period,
            effective_at=target.effective_at,
            first_available_at=self.clock(),
            correction_identity=dict(target.correction_identity),
            document_metadata={**target.metadata, "supplied": True},
        )
        if existing is None:
            self.session.add(revision)
            self.session.flush()
        capture = self._capture(
            document, outcome="supplied", url=target.url, revision=revision,
            changed=existing is None, content_hash=digest, byte_length=len(data),
        )
        self.commit()
        return CaptureResult(
            document.id, revision.id, capture.id, digest, existing is None,
            CoverageItem(
                route=target.adapter,
                outcome=CoverageOutcome.COMPLETE_FOR_REQUESTED_SCOPE,
                reason="supplied",
            ),
        )

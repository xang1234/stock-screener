"""Durable, leased work for Economic Theme processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.economic_taxonomy.contracts import ProviderAttemptOutcome
from app.models.economic_taxonomy_runtime import (
    DimensionProposal,
    EconomicExposureCandidate,
    ProcessingRequest,
    ProcessingRequestEvent,
    ProviderAttempt,
    ProviderAttemptEvent,
    TaxonomyAuthority,
)
from app.services.economic_taxonomy_fence import producer_write

LEASE_DURATION = timedelta(minutes=5)
OPERATOR_REQUEUE_REASON = "operator_requeue"


class WorkLeaseError(RuntimeError):
    """The caller does not hold the active request lease."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EconomicTaxonomyWorkRepository:
    def __init__(self, session: Session):
        self.session = session

    def enqueue_request(
        self,
        *,
        source_lineage_id: UUID,
        evidence_packet_id: UUID,
        policy_bundle_version: str,
        available_at: datetime | None = None,
    ) -> ProcessingRequest:
        key = (
            ProcessingRequest.source_lineage_id == source_lineage_id,
            ProcessingRequest.evidence_packet_id == evidence_packet_id,
            ProcessingRequest.policy_bundle_version == policy_bundle_version,
        )
        existing = self.session.execute(
            select(ProcessingRequest).where(*key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        try:
            with self.session.begin_nested():
                request = ProcessingRequest(
                    source_lineage_id=source_lineage_id,
                    evidence_packet_id=evidence_packet_id,
                    policy_bundle_version=policy_bundle_version,
                    status="pending",
                    available_at=available_at or _utcnow(),
                )
                self.session.add(request)
                self.session.flush()
                self._append_request_event(request.id, "enqueued", {})
                self.session.flush()
                return request
        except IntegrityError:
            return self.session.execute(
                select(ProcessingRequest).where(*key)
            ).scalar_one()

    def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime | None = None,
    ) -> ProcessingRequest | None:
        now = now or _utcnow()
        # Claiming is operational state. Read the epoch without locking the
        # singleton authority row so SKIP LOCKED can distribute independent
        # requests across workers. Completion rechecks it under the fence.
        authority = self.session.get(TaxonomyAuthority, 1)
        if authority is None:
            return None
        if (
            authority.mode not in {"shadow", "dual", "economic"}
            or authority.writes_fenced
        ):
            return None
        request = self.session.execute(
            select(ProcessingRequest)
            .where(
                ProcessingRequest.available_at <= now,
                or_(
                    ProcessingRequest.status.in_(("pending", "retryable")),
                    and_(
                        ProcessingRequest.status == "leased",
                        ProcessingRequest.lease_expires_at <= now,
                    ),
                ),
            )
            .order_by(ProcessingRequest.available_at, ProcessingRequest.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        ).scalar_one_or_none()
        if request is None:
            return None
        request.status = "leased"
        request.lease_token = uuid4()
        request.lease_owner = worker_id
        request.lease_expires_at = now + LEASE_DURATION
        request.observed_processing_head_revision = authority.processing_head_revision
        request.observed_authority_epoch = authority.authority_epoch
        request.completion_code = None
        self._append_request_event(
            request.id,
            "claimed",
            {
                "worker_id": worker_id,
                "lease_token": str(request.lease_token),
                "processing_head_revision": authority.processing_head_revision,
                "authority_epoch": authority.authority_epoch,
            },
        )
        self.session.flush()
        return request

    def retry(
        self,
        request_id: UUID,
        *,
        lease_token: UUID | None = None,
        available_at: datetime | None = None,
        reason: str = "retry_requested",
    ) -> ProcessingRequest:
        request = self._lock_request(request_id)
        if lease_token is not None:
            self._require_lease(request, lease_token)
        request.status = "retryable"
        request.available_at = available_at or _utcnow()
        request.completion_code = reason
        self._clear_lease(request)
        self._append_request_event(request.id, "retry_scheduled", {"reason": reason})
        self.session.flush()
        return request

    def complete(
        self,
        request_id: UUID,
        *,
        lease_token: UUID,
        result_payload: dict,
        classification_attempt_id: UUID | None = None,
    ) -> ProcessingRequest:
        observed = self.session.get(ProcessingRequest, request_id)
        if observed is None:
            raise KeyError(f"processing request {request_id} not found")
        if observed.observed_authority_epoch is None:
            raise WorkLeaseError("request has no observed authority epoch")

        with producer_write(
            self.session,
            expected_epoch=observed.observed_authority_epoch,
            allowed_modes={"shadow", "dual", "economic"},
        ) as authority:
            request = self._lock_request(request_id)
            self._require_lease(request, lease_token)
            if (
                request.observed_processing_head_revision
                != authority.processing_head_revision
            ):
                previous_head = request.observed_processing_head_revision
                request.status = "pending"
                request.available_at = _utcnow()
                request.observed_processing_head_revision = (
                    authority.processing_head_revision
                )
                request.completion_code = "stale_processing_head"
                self._clear_lease(request)
                self._append_request_event(
                    request.id,
                    "superseded_before_acceptance",
                    {
                        "old_processing_head_revision": previous_head,
                        "new_processing_head_revision": authority.processing_head_revision,
                        "superseded_attempt_id": (
                            str(classification_attempt_id)
                            if classification_attempt_id is not None
                            else None
                        ),
                    },
                )
                self.session.flush()
                return request

            request.status = "completed"
            request.completion_code = "completed"
            request.result_payload = dict(result_payload)
            self._clear_lease(request)
            self._append_request_event(
                request.id,
                "completed",
                {
                    "processing_head_revision": authority.processing_head_revision,
                    "classification_attempt_id": (
                        str(classification_attempt_id)
                        if classification_attempt_id is not None
                        else None
                    ),
                },
            )
            self.session.flush()
            return request

    def begin_provider_attempt(
        self,
        logical_request_id: UUID,
        *,
        operation: str,
        dispatch_id: str | None = None,
    ) -> ProviderAttempt:
        dispatch_id = dispatch_id or str(uuid4())
        existing = self.session.execute(
            select(ProviderAttempt).where(
                ProviderAttempt.logical_request_id == logical_request_id,
                ProviderAttempt.operation_kind == operation,
                ProviderAttempt.dispatch_id == dispatch_id,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        self._lock_request(logical_request_id)
        existing = self.session.execute(
            select(ProviderAttempt).where(
                ProviderAttempt.logical_request_id == logical_request_id,
                ProviderAttempt.operation_kind == operation,
                ProviderAttempt.dispatch_id == dispatch_id,
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        current = self.session.scalar(
            select(func.max(ProviderAttempt.attempt_number)).where(
                ProviderAttempt.logical_request_id == logical_request_id,
                ProviderAttempt.operation_kind == operation,
            )
        )
        attempt = ProviderAttempt(
            logical_request_id=logical_request_id,
            operation_kind=operation,
            attempt_number=int(current or 0) + 1,
            dispatch_id=dispatch_id,
        )
        self.session.add(attempt)
        self.session.flush()
        return attempt

    def fail_attempt(
        self,
        attempt_id: UUID,
        *,
        outcome: str,
        payload: dict | None = None,
    ) -> ProviderAttemptEvent:
        if outcome not in {
            ProviderAttemptOutcome.RETRYABLE_FAILURE.value,
            ProviderAttemptOutcome.UNCERTAIN.value,
            ProviderAttemptOutcome.TERMINAL_FAILURE.value,
        }:
            raise ValueError(
                "failure outcome must be retryable, uncertain, or terminal"
            )
        return self._finish_attempt(attempt_id, outcome, None, payload)

    def succeed_attempt(
        self,
        attempt_id: UUID,
        *,
        result_artifact_id: UUID,
        payload: dict | None = None,
    ) -> ProviderAttemptEvent:
        return self._finish_attempt(
            attempt_id,
            ProviderAttemptOutcome.SUCCESS.value,
            result_artifact_id,
            payload,
        )

    def record_candidate(
        self, request_id: UUID, *, candidate_key: str, payload: dict
    ) -> EconomicExposureCandidate:
        return self._record_child(
            EconomicExposureCandidate,
            request_id,
            EconomicExposureCandidate.candidate_key,
            candidate_key,
            payload,
        )

    def record_dimension_proposal(
        self, request_id: UUID, *, dimension_key: str, payload: dict
    ) -> DimensionProposal:
        return self._record_child(
            DimensionProposal,
            request_id,
            DimensionProposal.dimension_key,
            dimension_key,
            payload,
        )

    def _finish_attempt(
        self,
        attempt_id: UUID,
        outcome: str,
        result_artifact_id: UUID | None,
        payload: dict | None,
    ) -> ProviderAttemptEvent:
        attempt = self.session.execute(
            select(ProviderAttempt)
            .where(ProviderAttempt.id == attempt_id)
            .with_for_update()
        ).scalar_one_or_none()
        if attempt is None:
            raise KeyError(f"provider attempt {attempt_id} not found")
        existing = self.session.execute(
            select(ProviderAttemptEvent)
            .where(ProviderAttemptEvent.provider_attempt_id == attempt.id)
            .order_by(ProviderAttemptEvent.sequence_number.desc())
            .limit(1)
        ).scalar_one_or_none()
        if existing is not None:
            if (
                existing.outcome == outcome
                and existing.result_artifact_id == result_artifact_id
            ):
                return existing
            raise ValueError("provider attempt already has a terminal outcome")
        event = ProviderAttemptEvent(
            provider_attempt_id=attempt.id,
            sequence_number=1,
            outcome=outcome,
            result_artifact_id=result_artifact_id,
            event_payload=dict(payload or {}),
        )
        self.session.add(event)
        self.session.flush()
        self.session.expire(attempt, ["events"])
        return event

    def _record_child(self, model, request_id, key_column, key, payload):
        self._lock_request(request_id)
        existing = self.session.execute(
            select(model).where(
                model.processing_request_id == request_id,
                key_column == key,
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.payload != dict(payload):
                raise ValueError("idempotency key payload conflict")
            return existing
        field_name = key_column.key
        row = model(
            processing_request_id=request_id,
            **{field_name: key},
            payload=dict(payload),
        )
        self.session.add(row)
        self.session.flush()
        return row

    def _lock_request(self, request_id: UUID) -> ProcessingRequest:
        request = self.session.execute(
            select(ProcessingRequest)
            .where(ProcessingRequest.id == request_id)
            .with_for_update()
        ).scalar_one_or_none()
        if request is None:
            raise KeyError(f"processing request {request_id} not found")
        return request

    @staticmethod
    def _require_lease(request: ProcessingRequest, lease_token: UUID) -> None:
        if request.status != "leased" or request.lease_token != lease_token:
            raise WorkLeaseError("lease token does not own request")

    @staticmethod
    def _clear_lease(request: ProcessingRequest) -> None:
        request.lease_token = None
        request.lease_owner = None
        request.lease_expires_at = None

    def _append_request_event(
        self, request_id: UUID, event_type: str, payload: dict
    ) -> ProcessingRequestEvent:
        sequence = self.session.scalar(
            select(func.max(ProcessingRequestEvent.sequence_number)).where(
                ProcessingRequestEvent.processing_request_id == request_id
            )
        )
        event = ProcessingRequestEvent(
            processing_request_id=request_id,
            sequence_number=int(sequence or 0) + 1,
            event_type=event_type,
            event_payload=dict(payload),
        )
        self.session.add(event)
        return event

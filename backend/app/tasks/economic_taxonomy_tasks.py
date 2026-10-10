"""Bounded Celery entry points for the Economic Taxonomy runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import aliased

from app.celery_app import celery_app
from app.database import SessionLocal
from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    EconomicTaxonomyWorkRepository,
)
from app.models.economic_taxonomy import FacetDimension
from app.models.economic_taxonomy_runtime import (
    EvidencePacket,
    EvidencePrecedenceRevision,
    GenerationInputManifest,
    LensEligibilityRevision,
    ProcessingRequest,
    ProviderAttempt,
    ServingGeneration,
    ServingGenerationEvent,
    TaxonomyAuthority,
    TaxonomyProjectionEvent,
    TaxonomySourceRevisionLog,
)
from app.services.economic_exposure_claim_review import (
    ClaimReviewSchemaError,
    EconomicExposureClaimReviewer,
)
from app.services.economic_exposure_extraction import (
    BudgetExhausted,
    EconomicExposureExtractor,
    EvidenceSchemaError,
    ProviderOutcomeUncertain,
    ProviderTerminalFailure,
    RetryableProviderFailure,
)
from app.services.economic_taxonomy_fence import producer_write
from app.services.economic_taxonomy_processor import (
    EconomicTaxonomyProcessor,
    ProviderResultUnavailable,
    ResolutionReviewRequired,
)
from app.services.economic_taxonomy_publication import (
    EconomicTaxonomyPublicationCoordinator,
    ManifestChanged,
)
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.services.economic_theme_lifecycle_service import EconomicThemeLifecycleService
from app.services.economic_theme_metrics_service import EconomicThemeMetricsService
from app.utils.datetime_utils import as_aware_utc as _utc

logger = logging.getLogger(__name__)
SYSTEM_PRINCIPAL = AdminPrincipal(
    subject="system:economic-taxonomy-refresh",
    auth_method="service_principal",
    roles=frozenset({"taxonomy:review"}),
)
POLICY_BUNDLE_VERSION = "economic-taxonomy-v1"
MAX_BATCH_SIZE = 500
PUBLICATION_COALESCE_WINDOW = timedelta(minutes=5)

# Per-request provider and claim-review outcomes terminalize one processing
# request. That request contributes no interpretation (the lineage keeps its
# previous completed attempt), so routine publication proceeds; the revisions
# are reported as attention items instead of pausing every refresh.
_ATTENTION_REVISION_KINDS = frozenset(
    {
        "claim_review_required",
        "provider_invalid_schema",
        "provider_outcome_uncertain",
        "provider_terminal_failure",
    }
)
_ROUTINE_REVISION_KINDS = _ATTENTION_REVISION_KINDS | frozenset(
    {
        "administrator_decision",
        "association_revision",
        "classification_attempt",
        "development_observation",
        "evidence",
        "legacy_event_mapping",
        "lens_eligibility",
        "lifecycle_change",
        "metrics_refresh",
        "proposal_decision",
        "social_theme_projection",
    }
)


class EconomicProcessingPipeline(Protocol):
    def process(self, request_id: UUID, lease_token: UUID): ...


class EconomicExtractionReviewPipeline:
    """Compose provider work with the fenced taxonomy processor.

    Extraction and review own their short persistence transactions and perform
    provider I/O outside database locks.  The processor then performs the only
    authoritative classification commit under the shared writer fence.
    """

    def __init__(
        self,
        session_factory,
        *,
        extractor: EconomicExposureExtractor,
        reviewer: EconomicExposureClaimReviewer,
        processor: EconomicTaxonomyProcessor,
    ):
        self.session_factory = session_factory
        self.extractor = extractor
        self.reviewer = reviewer
        self.processor = processor

    def process(self, request_id: UUID, lease_token: UUID):
        extraction = self.extractor.extract(request_id)
        for _stale_review_retry in range(5):
            facet_hash = self._facet_hash_for_request(request_id)
            self.reviewer.review(
                extraction,
                request_id=request_id,
                facet_hash=facet_hash,
            )
            try:
                return self.processor.process(request_id, lease_token)
            except ProviderResultUnavailable as exc:
                # A concurrent head change can select a different facet-catalog
                # hash after the provider review. Re-review the reusable
                # extraction; other non-accepted review states are durable.
                if str(exc) != "claim_review_artifact_missing":
                    raise
        raise RuntimeError("claim_review_head_changed_repeatedly")

    def _facet_hash_for_request(self, request_id: UUID) -> str:
        with self.session_factory() as session:
            request = session.get(ProcessingRequest, request_id)
            if request is None:
                raise KeyError(f"processing request {request_id} not found")
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.processing_taxonomy_version_id is None:
                raise RuntimeError("processing_taxonomy_head_missing")
            version_id = EconomicTaxonomyProcessor._input_version_for_request(
                session, request=request, authority=authority
            )
            return EconomicTaxonomyRepository(session).facet_catalog_semantic_hash(
                version_id
            )


@dataclass(frozen=True, slots=True)
class DirtyRevisionClassification:
    routine_revision_ids: tuple[str, ...]
    held_revision_ids: tuple[str, ...]
    # Routine revisions that still need operator attention.
    attention_revision_ids: tuple[str, ...] = ()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bounded_limit(limit: int) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_BATCH_SIZE
    ):
        raise ValueError(f"limit must be between 1 and {MAX_BATCH_SIZE}")
    return limit


def refresh_is_coalesced(last_published_at: datetime | None, *, now: datetime) -> bool:
    if last_published_at is None:
        return False
    return _utc(now) - _utc(last_published_at) < PUBLICATION_COALESCE_WINDOW


def classify_dirty_revisions(
    revisions: Iterable[TaxonomySourceRevisionLog | Any],
) -> DirtyRevisionClassification:
    return _classify_revision_pairs(
        (str(revision.id), str(revision.revision_kind)) for revision in revisions
    )


def _classify_revision_pairs(
    revisions: Iterable[tuple[str, str]],
) -> DirtyRevisionClassification:
    routine: list[str] = []
    held: list[str] = []
    attention: list[str] = []
    for revision_id, revision_kind in revisions:
        target = routine if revision_kind in _ROUTINE_REVISION_KINDS else held
        target.append(revision_id)
        if revision_kind in _ATTENTION_REVISION_KINDS:
            attention.append(revision_id)
    return DirtyRevisionClassification(tuple(routine), tuple(held), tuple(attention))


class EconomicTaxonomyTaskService:
    """Small orchestration layer; every entry point is bounded and mode-aware."""

    def __init__(
        self,
        session_factory,
        *,
        pipeline: EconomicProcessingPipeline | None = None,
        coordinator: EconomicTaxonomyPublicationCoordinator | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.session_factory = session_factory
        self.pipeline = pipeline
        self.coordinator = coordinator or EconomicTaxonomyPublicationCoordinator(
            session_factory, clock=clock
        )
        self.clock = clock

    def discover(self, *, limit: int = 50) -> dict[str, Any]:
        limit = _bounded_limit(limit)
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.mode == "legacy":
                return {"status": "skipped", "reason": "legacy_mode"}
            if authority.writes_fenced:
                return {"status": "skipped", "reason": "writes_fenced"}
            epoch = authority.authority_epoch
            discovered = 0
            enqueued = 0
            with producer_write(
                session,
                expected_epoch=epoch,
                allowed_modes={"shadow", "dual", "economic"},
            ) as locked:
                precedence = aliased(EvidencePrecedenceRevision)
                eligibility = aliased(LensEligibilityRevision)
                latest_effective_revision = (
                    select(func.max(EvidencePrecedenceRevision.revision_number))
                    .where(
                        EvidencePrecedenceRevision.source_lineage_id
                        == precedence.source_lineage_id,
                        EvidencePrecedenceRevision.disposition == "effective",
                    )
                    .correlate(precedence)
                    .scalar_subquery()
                )
                latest_eligibility_revision = (
                    select(func.max(LensEligibilityRevision.revision_number))
                    .where(
                        LensEligibilityRevision.evidence_packet_id
                        == eligibility.evidence_packet_id
                    )
                    .correlate(eligibility)
                    .scalar_subquery()
                )
                existing_request = (
                    select(ProcessingRequest.id)
                    .where(
                        ProcessingRequest.source_lineage_id
                        == EvidencePacket.source_lineage_id,
                        ProcessingRequest.evidence_packet_id == EvidencePacket.id,
                        ProcessingRequest.policy_bundle_version
                        == POLICY_BUNDLE_VERSION,
                    )
                    .correlate(EvidencePacket)
                    .exists()
                )
                packets = session.scalars(
                    select(EvidencePacket)
                    .join(
                        precedence,
                        precedence.evidence_packet_id == EvidencePacket.id,
                    )
                    .join(
                        eligibility,
                        eligibility.evidence_packet_id == EvidencePacket.id,
                    )
                    .where(
                        precedence.disposition == "effective",
                        precedence.revision_number == latest_effective_revision,
                        eligibility.revision_number
                        == latest_eligibility_revision,
                        func.json_array_length(eligibility.evidence_channels) > 0,
                        ~existing_request,
                    )
                    .order_by(EvidencePacket.source_lineage_id)
                    .limit(limit)
                ).all()
                work = EconomicTaxonomyWorkRepository(session)
                for packet in packets:
                    discovered += 1
                    request = work.enqueue_request(
                        source_lineage_id=packet.source_lineage_id,
                        evidence_packet_id=packet.id,
                        policy_bundle_version=POLICY_BUNDLE_VERSION,
                        available_at=packet.available_at,
                    )
                    enqueued += 1
                    self._append_packet_revision(
                        session,
                        packet=packet,
                        authority_epoch=locked.authority_epoch,
                    )
                    del request
                session.commit()
            return {
                "status": "ok",
                "discovered": discovered,
                "enqueued": enqueued,
                "limit": limit,
            }

    def process(
        self, *, limit: int = 50, worker_id: str = "economic-taxonomy-worker"
    ) -> dict[str, Any]:
        limit = _bounded_limit(limit)
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        mode = self._mode()
        if mode == "legacy":
            return {"status": "skipped", "reason": "legacy_mode"}
        if mode not in {"shadow", "dual", "economic"}:
            return {"status": "skipped", "reason": "authority_unavailable"}
        if self.pipeline is None:
            return {"status": "skipped", "reason": "processor_unconfigured"}

        result: dict[str, Any] = {
            "status": "ok",
            "claimed": 0,
            "processed": 0,
            "retryable": 0,
            "held": 0,
            "terminal_failures": 0,
        }
        for _index in range(limit):
            with self.session_factory() as session:
                request = EconomicTaxonomyWorkRepository(session).claim_next(
                    worker_id=worker_id, now=self.clock()
                )
                if request is None:
                    session.rollback()
                    break
                request_id = request.id
                lease_token = request.lease_token
                session.commit()
            result["claimed"] += 1
            try:
                self.pipeline.process(request_id, lease_token)
                result["processed"] += 1
            except BudgetExhausted:
                self._retry_request(
                    request_id,
                    lease_token,
                    reason="budget_exhausted",
                    delay=timedelta(minutes=5),
                )
                result["retryable"] += 1
                result["reason"] = "budget_exhausted"
                break
            except RetryableProviderFailure:
                delay = self._retry_delay(request_id)
                self._retry_request(
                    request_id,
                    lease_token,
                    reason="provider_retryable",
                    delay=delay,
                )
                result["retryable"] += 1
            except ProviderOutcomeUncertain:
                self._terminal_request(
                    request_id,
                    lease_token,
                    reason="provider_outcome_uncertain",
                    revision_kind="provider_outcome_uncertain",
                )
                result["held"] += 1
            except ResolutionReviewRequired as exc:
                reason = str(exc) or "resolution_review_required"
                revision_kind = (
                    "dimension_proposal"
                    if "dimension" in reason
                    else "ambiguous_identity"
                )
                self._terminal_request(
                    request_id,
                    lease_token,
                    reason=reason,
                    revision_kind=revision_kind,
                )
                result["held"] += 1
            except ProviderResultUnavailable as exc:
                self._terminal_request(
                    request_id,
                    lease_token,
                    reason=str(exc) or "claim_review_not_accepted",
                    revision_kind="claim_review_required",
                )
                result["held"] += 1
            except (EvidenceSchemaError, ClaimReviewSchemaError) as exc:
                self._terminal_request(
                    request_id,
                    lease_token,
                    reason=str(exc) or "invalid_provider_schema",
                    revision_kind="provider_invalid_schema",
                )
                result["terminal_failures"] += 1
            except ProviderTerminalFailure as exc:
                self._terminal_request(
                    request_id,
                    lease_token,
                    reason=str(exc) or "provider_terminal_failure",
                    revision_kind="provider_terminal_failure",
                )
                result["terminal_failures"] += 1
        return result

    def deliver(
        self, *, limit: int = 100, worker_id: str = "taxonomy-outbox-worker"
    ) -> dict[str, Any]:
        limit = _bounded_limit(limit)
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None:
                return {"status": "skipped", "reason": "authority_missing"}
            if authority.writes_fenced:
                return {"status": "skipped", "reason": "writes_fenced"}
            claims = EconomicTaxonomyRuntimeService(
                session
            ).claim_deliveries_from_published_generations(
                worker_id=worker_id,
                expected_epoch=authority.authority_epoch,
                now=self.clock(),
                limit=limit,
            )
            generation_ids = sorted(
                {
                    str(value)
                    for value in session.scalars(
                        select(TaxonomyProjectionEvent.serving_generation_id).where(
                            TaxonomyProjectionEvent.id.in_(
                                [claim.projection_event_id for claim in claims]
                            )
                        )
                    )
                    if value is not None
                }
            )
            session.commit()

        checkpointed = 0
        failures = 0
        for claim in claims:
            try:
                with self.session_factory() as session:
                    EconomicTaxonomyRuntimeService(session).apply_delivery(
                        claim,
                        expected_epoch=claim.claimed_epoch,
                        now=self.clock(),
                    )
                    session.commit()
                checkpointed += 1
            except Exception as exc:  # noqa: BLE001 - isolate each durable delivery
                failures += 1
                with self.session_factory() as session:
                    EconomicTaxonomyRuntimeService(session).record_delivery_failure(
                        claim,
                        outcome="retryable_failure",
                        error=str(exc),
                    )
                    session.commit()
        rollback_state = self.coordinator.refresh_rollback_availability()
        return {
            "status": "ok",
            "claimed": len(claims),
            "claimed_generation_ids": generation_ids,
            "checkpointed": checkpointed,
            "failures": failures,
            "rollback_state": rollback_state,
        }

    def refresh(self) -> dict[str, Any]:
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.mode != "economic":
                return {"status": "skipped", "reason": "not_economic_mode"}
            if authority.writes_fenced:
                return {"status": "skipped", "reason": "writes_fenced"}
            if authority.serving_generation_id is None:
                return {"status": "skipped", "reason": "serving_generation_missing"}
            current = session.get(ServingGeneration, authority.serving_generation_id)
            manifest = session.get(
                GenerationInputManifest, current.generation_input_manifest_id
            )
            included_ids = {
                str(row[5])
                for row in (manifest.committed_revision_tuples or [])
                if len(row) >= 6
            }
            dirty = [
                row
                for row in session.scalars(
                    select(TaxonomySourceRevisionLog).order_by(
                        TaxonomySourceRevisionLog.committed_at,
                        TaxonomySourceRevisionLog.id,
                    )
                )
                if str(row.id) not in included_ids
            ]
            classified = classify_dirty_revisions(dirty)
            if classified.held_revision_ids:
                return {
                    "status": "held",
                    "reason": "review_required",
                    "held_revision_ids": list(classified.held_revision_ids),
                    "routine_revision_ids": list(classified.routine_revision_ids),
                }
            if not dirty:
                return {"status": "skipped", "reason": "clean"}
            last_published = session.scalar(
                select(ServingGenerationEvent.created_at)
                .where(
                    ServingGenerationEvent.serving_generation_id == current.id,
                    ServingGenerationEvent.event_type == "published",
                )
                .order_by(ServingGenerationEvent.created_at.desc())
                .limit(1)
            )
            if refresh_is_coalesced(last_published, now=self.clock()):
                return {
                    "status": "skipped",
                    "reason": "coalesced",
                    "dirty_revision_count": len(dirty),
                }
            capability_id = current.reader_capability_manifest_id
        cutoff = self.coordinator.capture_cutoff(principal=SYSTEM_PRINCIPAL)
        with self.session_factory() as session:
            parent_revision_ids: set[str] = set()
            if cutoff.expected_parent_generation_id is not None:
                parent = session.get(
                    ServingGeneration, cutoff.expected_parent_generation_id
                )
                if parent is not None:
                    parent_manifest = session.get(
                        GenerationInputManifest,
                        parent.generation_input_manifest_id,
                    )
                    if parent_manifest is not None:
                        parent_revision_ids = {
                            str(row[5])
                            for row in (
                                parent_manifest.committed_revision_tuples or []
                            )
                            if len(row) >= 6
                        }
            captured = _classify_revision_pairs(
                (str(row[5]), str(row[2]))
                for row in cutoff.committed_revision_tuples
                if len(row) >= 6 and str(row[5]) not in parent_revision_ids
            )
        if captured.held_revision_ids:
            return {
                "status": "held",
                "reason": "review_required",
                "held_revision_ids": list(captured.held_revision_ids),
                "routine_revision_ids": list(captured.routine_revision_ids),
            }
        prepared = self.coordinator.prepare_generation(
            cutoff,
            principal=SYSTEM_PRINCIPAL,
            reader_capability_manifest_id=capability_id,
            target_mode="economic",
        )
        try:
            published = self.coordinator.publish_generation(
                prepared.id, principal=SYSTEM_PRINCIPAL
            )
        except ManifestChanged as exc:
            return {
                "status": "retryable",
                "reason": str(exc),
                "prepared_generation_id": str(prepared.id),
                "backlog_preserved": True,
            }
        if captured.attention_revision_ids:
            logger.warning(
                "Economic Taxonomy published around %d terminal processing "
                "request(s) that need operator attention: %s",
                len(captured.attention_revision_ids),
                ", ".join(captured.attention_revision_ids),
            )
        return {
            "status": "published",
            "generation_id": str(published.id),
            "authority_epoch": published.authority_epoch,
            "routine_revision_ids": list(captured.routine_revision_ids),
            "attention_revision_ids": list(captured.attention_revision_ids),
        }

    def apply_lifecycle(
        self,
        *,
        interpretation_set_id: UUID | None = None,
    ) -> dict[str, Any]:
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.mode == "legacy":
                return {"status": "skipped", "reason": "legacy_mode"}
            selected_id = interpretation_set_id
            if selected_id is None and authority.serving_generation_id is not None:
                generation = session.get(
                    ServingGeneration, authority.serving_generation_id
                )
                selected_id = generation.interpretation_set_id
            if selected_id is None:
                return {"status": "skipped", "reason": "interpretation_missing"}
            result = EconomicThemeLifecycleService(session).propose_lifecycle_snapshot(
                as_of=self.clock(),
                principal=SYSTEM_PRINCIPAL,
                expected_epoch=authority.authority_epoch,
                interpretation_set_id=selected_id,
            )
            session.commit()
            return {
                "status": "ok",
                "taxonomy_version_id": str(result.taxonomy_version_id),
                "transition_count": result.transition_count,
            }

    def calculate_metrics(
        self,
        *,
        taxonomy_version_id: UUID | None = None,
        interpretation_set_id: UUID | None = None,
        generation_input_manifest_id: UUID | None = None,
    ) -> dict[str, Any]:
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.mode == "legacy":
                return {"status": "skipped", "reason": "legacy_mode"}
            generation = (
                session.get(ServingGeneration, authority.serving_generation_id)
                if authority.serving_generation_id is not None
                else None
            )
            taxonomy_id = taxonomy_version_id or (
                authority.processing_taxonomy_version_id
            )
            interpretation_id = interpretation_set_id or (
                generation.interpretation_set_id if generation else None
            )
            manifest_id = generation_input_manifest_id or (
                generation.generation_input_manifest_id if generation else None
            )
            if None in {taxonomy_id, interpretation_id, manifest_id}:
                return {"status": "skipped", "reason": "metrics_inputs_missing"}
            expected_epoch = authority.authority_epoch
            with producer_write(
                session,
                expected_epoch=expected_epoch,
                allowed_modes={"shadow", "dual", "economic"},
            ) as locked:
                revision = EconomicThemeMetricsService(session).calculate_metrics(
                    taxonomy_version_id=taxonomy_id,
                    interpretation_set_id=interpretation_id,
                    generation_input_manifest_id=manifest_id,
                    as_of=self.clock(),
                    actor=SYSTEM_PRINCIPAL.subject,
                )
                logical_key = "metrics:economic-theme-metrics-v1"
                already_logged = session.scalar(
                    select(TaxonomySourceRevisionLog.id).where(
                        TaxonomySourceRevisionLog.producer_kind == "economic_taxonomy",
                        TaxonomySourceRevisionLog.logical_source_key == logical_key,
                        TaxonomySourceRevisionLog.revision_kind == "metrics_refresh",
                        TaxonomySourceRevisionLog.content_hash
                        == str(revision.semantic_hash),
                    )
                )
                if already_logged is None:
                    latest = session.scalar(
                        select(
                            func.max(TaxonomySourceRevisionLog.revision_number)
                        ).where(
                            TaxonomySourceRevisionLog.producer_kind
                            == "economic_taxonomy",
                            TaxonomySourceRevisionLog.logical_source_key == logical_key,
                            TaxonomySourceRevisionLog.revision_kind
                            == "metrics_refresh",
                        )
                    )
                    EconomicTaxonomyPublicationRepository(
                        session
                    ).append_source_revision(
                        producer_kind="economic_taxonomy",
                        logical_source_key=logical_key,
                        revision_kind="metrics_refresh",
                        revision_number=int(latest or 0) + 1,
                        content_hash=str(revision.semantic_hash),
                        authority_epoch=locked.authority_epoch,
                    )
                session.commit()
            return {"status": "ok", "metrics_revision_id": str(revision.id)}

    def _mode(self) -> str | None:
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.writes_fenced:
                return None
            return authority.mode

    def _retry_request(
        self,
        request_id: UUID,
        lease_token: UUID,
        *,
        reason: str,
        delay: timedelta,
    ) -> None:
        with self.session_factory() as session:
            observed = session.get(ProcessingRequest, request_id)
            if observed is None or observed.observed_authority_epoch is None:
                return
            with producer_write(
                session,
                expected_epoch=observed.observed_authority_epoch,
                allowed_modes={"shadow", "dual", "economic"},
            ):
                EconomicTaxonomyWorkRepository(session).retry(
                    request_id,
                    lease_token=lease_token,
                    available_at=self.clock() + delay,
                    reason=reason,
                )
                session.commit()

    def _retry_delay(self, request_id: UUID) -> timedelta:
        with self.session_factory() as session:
            attempts = session.scalar(
                select(func.count(ProviderAttempt.id)).where(
                    ProviderAttempt.logical_request_id == request_id
                )
            )
        return timedelta(seconds=min(300, 2 ** min(int(attempts or 1), 8)))

    def _terminal_request(
        self,
        request_id: UUID,
        lease_token: UUID,
        *,
        reason: str,
        revision_kind: str,
    ) -> None:
        with self.session_factory() as session:
            observed = session.get(ProcessingRequest, request_id)
            if observed is None or observed.observed_authority_epoch is None:
                return
            with producer_write(
                session,
                expected_epoch=observed.observed_authority_epoch,
                allowed_modes={"shadow", "dual", "economic"},
            ) as authority:
                request = session.execute(
                    select(ProcessingRequest)
                    .where(ProcessingRequest.id == request_id)
                    .with_for_update()
                ).scalar_one()
                if request.lease_token != lease_token:
                    return
                request.status = "terminal_failure"
                request.completion_code = reason[:80]
                request.lease_token = None
                request.lease_owner = None
                request.lease_expires_at = None
                EconomicTaxonomyWorkRepository(session)._append_request_event(
                    request.id, "terminal_failure", {"reason": reason}
                )
                EconomicTaxonomyPublicationRepository(session).append_source_revision(
                    producer_kind="economic_taxonomy",
                    logical_source_key=f"processing_request:{request.id}",
                    revision_kind=revision_kind,
                    revision_number=1,
                    content_hash=str(request.id),
                    authority_epoch=authority.authority_epoch,
                )
                session.commit()

    @staticmethod
    def _append_packet_revision(session, *, packet, authority_epoch: int) -> None:
        repository = EconomicTaxonomyPublicationRepository(session)
        logical_key = f"evidence_packet:{packet.id}"
        existing = session.scalar(
            select(TaxonomySourceRevisionLog.id).where(
                TaxonomySourceRevisionLog.producer_kind == "evidence",
                TaxonomySourceRevisionLog.logical_source_key == logical_key,
                TaxonomySourceRevisionLog.revision_kind == "evidence",
                TaxonomySourceRevisionLog.revision_number
                == packet.evidence_revision_ordinal,
            )
        )
        if existing is None:
            repository.append_source_revision(
                producer_kind="evidence",
                logical_source_key=logical_key,
                revision_kind="evidence",
                revision_number=packet.evidence_revision_ordinal,
                content_hash=packet.packet_hash,
                authority_epoch=authority_epoch,
            )

_PIPELINE_FACTORY: Callable[[], EconomicProcessingPipeline] | None = None


def configure_economic_taxonomy_pipeline(
    factory: Callable[[], EconomicProcessingPipeline] | None,
) -> None:
    """Bind the provider-aware extraction/review/resolve pipeline for workers."""

    global _PIPELINE_FACTORY
    _PIPELINE_FACTORY = factory


def build_default_economic_taxonomy_pipeline(session_factory):
    """Construct the deployed pipeline when sanctioned provider credentials exist."""

    from app.services.economic_taxonomy_llm_provider import (
        build_economic_taxonomy_provider,
    )

    provider, reservations = build_economic_taxonomy_provider(session_factory)
    if provider is None:
        return None
    with session_factory() as session:
        authority = session.get(TaxonomyAuthority, 1)
        if authority is None or authority.processing_taxonomy_version_id is None:
            return None
        approved_dimensions = set(
            session.scalars(
                select(FacetDimension.key).where(
                    FacetDimension.taxonomy_version_id
                    == authority.processing_taxonomy_version_id
                )
            )
        )
    extractor = EconomicExposureExtractor(
        session_factory,
        provider=provider,
        extraction_policy_version="economic-extraction-v1",
        approved_dimensions=approved_dimensions,
        reservations=reservations,
    )
    reviewer = EconomicExposureClaimReviewer(
        session_factory,
        provider=provider,
        claim_review_policy_version="economic-claim-review-v1",
        approved_dimensions=approved_dimensions,
        reservations=reservations,
    )
    processor = EconomicTaxonomyProcessor(
        session_factory,
        resolver_policy_version="economic-resolver-v1",
        naming_policy_version="economic-naming-v1",
        derivation_policy_version="economic-derivation-v1",
        lifecycle_policy_version="economic-lifecycle-v1",
    )
    return EconomicExtractionReviewPipeline(
        session_factory,
        extractor=extractor,
        reviewer=reviewer,
        processor=processor,
    )


def _service() -> EconomicTaxonomyTaskService:
    pipeline = (
        _PIPELINE_FACTORY()
        if _PIPELINE_FACTORY is not None
        else build_default_economic_taxonomy_pipeline(SessionLocal)
    )
    return EconomicTaxonomyTaskService(SessionLocal, pipeline=pipeline)


@celery_app.task(
    name="app.tasks.economic_taxonomy_tasks.discover_economic_taxonomy_work"
)
def discover_economic_taxonomy_work(limit: int = 50):
    return _service().discover(limit=limit)


@celery_app.task(
    name="app.tasks.economic_taxonomy_tasks.process_economic_taxonomy_work"
)
def process_economic_taxonomy_work(limit: int = 50):
    return _service().process(
        limit=limit,
        worker_id=f"celery:{process_economic_taxonomy_work.request.id or 'manual'}",
    )


@celery_app.task(name="app.tasks.economic_taxonomy_tasks.deliver_taxonomy_outbox")
def deliver_taxonomy_outbox(limit: int = 100):
    return _service().deliver(
        limit=limit,
        worker_id=f"celery:{deliver_taxonomy_outbox.request.id or 'manual'}",
    )


@celery_app.task(
    name="app.tasks.economic_taxonomy_tasks.refresh_economic_taxonomy_generation",
    autoretry_for=(OperationalError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=300,
    max_retries=None,
)
def refresh_economic_taxonomy_generation():
    return _service().refresh()


@celery_app.task(
    name="app.tasks.economic_taxonomy_tasks.backfill_legacy_developments"
)
def backfill_legacy_developments(taxonomy_version_id: str | None = None):
    """Keep the processing version's legacy development links mapped (#513).

    The snapshot builder fails closed on legacy observations this has not
    covered; unchanged legacy links make the run a cheap no-op.
    """
    from app.services.economic_development_backfill import (
        backfill_legacy_developments as backfill,
    )

    with SessionLocal.begin() as db:
        return backfill(
            db, UUID(taxonomy_version_id) if taxonomy_version_id else None
        )


@celery_app.task(
    name="app.tasks.economic_taxonomy_tasks.apply_economic_theme_lifecycle"
)
def apply_economic_theme_lifecycle(interpretation_set_id: str | None = None):
    return _service().apply_lifecycle(
        interpretation_set_id=(
            UUID(interpretation_set_id) if interpretation_set_id else None
        )
    )


@celery_app.task(
    name="app.tasks.economic_taxonomy_tasks.calculate_economic_theme_metrics"
)
def calculate_economic_theme_metrics(
    taxonomy_version_id: str | None = None,
    interpretation_set_id: str | None = None,
    generation_input_manifest_id: str | None = None,
):
    return _service().calculate_metrics(
        taxonomy_version_id=(
            UUID(taxonomy_version_id) if taxonomy_version_id else None
        ),
        interpretation_set_id=(
            UUID(interpretation_set_id) if interpretation_set_id else None
        ),
        generation_input_manifest_id=(
            UUID(generation_input_manifest_id) if generation_input_manifest_id else None
        ),
    )

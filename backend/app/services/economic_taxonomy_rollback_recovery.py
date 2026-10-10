"""Compatibility recovery used before an Economic Taxonomy rollback."""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, select

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.models.economic_taxonomy_runtime import (
    TaxonomyAuthority,
    TaxonomyProjectionDeliveryAttempt,
    TaxonomyProjectionDeliveryEvent,
    TaxonomyProjectionEvent,
    TaxonomySourceRevisionLog,
)
from app.services.economic_taxonomy_fence import exclusive_publication
from app.services.economic_taxonomy_publication_contracts import (
    CompatibilityNotAcknowledged,
)
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.utils.file_hashing import canonical_json_sha256 as _hash


class RollbackRecovery:
    def __init__(self, session_factory, *, clock):
        self.session_factory = session_factory
        self.clock = clock

    def begin(
        self,
        *,
        generation_id: UUID,
        principal: AdminPrincipal,
        reason: str,
    ) -> None:
        with self.session_factory() as session:  # noqa: SIM117
            with exclusive_publication(session) as authority:
                authority.writes_fenced = True
                authority.rollback_state = "rollback_recovery"
                authority.rollback_reason = reason
                latest = session.scalar(
                    select(func.max(TaxonomySourceRevisionLog.revision_number)).where(
                        TaxonomySourceRevisionLog.producer_kind == "publication",
                        TaxonomySourceRevisionLog.logical_source_key
                        == f"rollback:{generation_id}",
                        TaxonomySourceRevisionLog.revision_kind == "rollback_recovery",
                    )
                )
                EconomicTaxonomyPublicationRepository(session).append_source_revision(
                    producer_kind="publication",
                    logical_source_key=f"rollback:{generation_id}",
                    revision_kind="rollback_recovery",
                    revision_number=int(latest or 0) + 1,
                    content_hash=_hash(
                        {
                            "generation_id": str(generation_id),
                            "reason": reason,
                            "actor": principal.subject,
                        }
                    ),
                    authority_epoch=authority.authority_epoch,
                )
                session.commit()

    def rebuild_legacy_projections(
        self,
        generation_id: UUID,
        *,
        principal: AdminPrincipal,
    ) -> None:
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            rows = session.scalars(
                select(TaxonomyProjectionEvent)
                .where(
                    TaxonomyProjectionEvent.serving_generation_id == generation_id,
                    TaxonomyProjectionEvent.delivery_scope == "candidate",
                )
                .order_by(
                    TaxonomyProjectionEvent.source_lineage,
                    TaxonomyProjectionEvent.projection_kind,
                    TaxonomyProjectionEvent.projection_revision,
                )
            ).all()
            runtime = EconomicTaxonomyRuntimeService(session)
            now = self.clock()
            for event in rows:
                completed = session.scalar(
                    select(TaxonomyProjectionDeliveryEvent.id)
                    .join(
                        TaxonomyProjectionDeliveryAttempt,
                        TaxonomyProjectionDeliveryAttempt.id
                        == TaxonomyProjectionDeliveryEvent.delivery_attempt_id,
                    )
                    .where(
                        TaxonomyProjectionDeliveryAttempt.projection_event_id
                        == event.id,
                        TaxonomyProjectionDeliveryEvent.outcome.in_(
                            ("success", "stale_noop")
                        ),
                    )
                )
                if completed is not None:
                    continue
                attempt_number = (
                    int(
                        session.scalar(
                            select(
                                func.max(
                                    TaxonomyProjectionDeliveryAttempt.attempt_number
                                )
                            ).where(
                                TaxonomyProjectionDeliveryAttempt.projection_event_id
                                == event.id
                            )
                        )
                        or 0
                    )
                    + 1
                )
                attempt = TaxonomyProjectionDeliveryAttempt(
                    projection_event_id=event.id,
                    attempt_number=attempt_number,
                    claimed_epoch=authority.authority_epoch,
                    lease_token=uuid4(),
                    lease_owner=f"rollback-recovery:{principal.subject}",
                    lease_expires_at=now + timedelta(minutes=5),
                )
                session.add(attempt)
                session.flush()
                applied = runtime.apply_projection_event(
                    event,
                    authority_epoch=authority.authority_epoch,
                    now=now,
                )
                session.add(
                    TaxonomyProjectionDeliveryEvent(
                        delivery_attempt_id=attempt.id,
                        outcome="success" if applied else "stale_noop",
                        details={"rollback_recovery": True},
                    )
                )
            session.flush()
            if not runtime.generation_acknowledged(generation_id):
                raise CompatibilityNotAcknowledged(
                    "rollback_recovery_projection_incomplete"
                )
            authority.rollback_state = "recovered"
            authority.rollback_reason = None
            session.commit()

    def mark_failed(self, reason: str) -> None:
        with self.session_factory() as session:  # noqa: SIM117
            with exclusive_publication(session) as authority:
                authority.writes_fenced = True
                authority.rollback_state = "recovery_failed"
                authority.rollback_reason = reason
                session.commit()

    def refresh_availability(self) -> str:
        with self.session_factory() as session:  # noqa: SIM117
            with exclusive_publication(session) as authority:
                generation_id = authority.serving_generation_id
                if generation_id is None:
                    state = authority.rollback_state
                elif EconomicTaxonomyRuntimeService(session).generation_acknowledged(
                    generation_id
                ):
                    authority.rollback_state = "ready"
                    authority.rollback_reason = None
                    state = "ready"
                else:
                    authority.rollback_state = "temporarily_unavailable"
                    authority.rollback_reason = "compatibility_delivery_pending"
                    state = authority.rollback_state
                session.commit()
                return state

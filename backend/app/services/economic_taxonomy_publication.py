"""Bounded, crash-safe publication for Economic Taxonomy generations.

The coordinator deliberately owns three separate transactions: capture under a
short exclusive fence, preparation without that fence, and a final short
compare-and-set publication.  Provider calls and delivery waits never occur in
the fenced transactions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.models.social_analysis import EconomicSocialAssociationRevision
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
    PublicationInvariantError,
)
from app.models.economic_taxonomy import TaxonomyVersion
from app.models.economic_taxonomy_runtime import (
    ClassificationAttempt,
    DevelopmentSelectionRevision,
    EvidencePacket,
    EvidencePrecedenceRevision,
    GenerationInputManifest,
    LensEligibilityRevision,
    ServingGeneration,
    ServingGenerationEvent,
    SourceLineage,
    TaxonomyAuthority,
    TaxonomyProjectionEvent,
)
from app.models.theme_intelligence import (
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
)
from app.services.economic_taxonomy_fence import exclusive_publication
from app.services.economic_social_taxonomy_adapter import (
    EconomicSocialTaxonomyAdapter,
)
from app.services.economic_taxonomy_interpretations import (
    EconomicTaxonomyInterpretationService,
)
from app.services.economic_taxonomy_publication_compatibility import (
    CompatibilityProjection,
    build_default_compatibility_projections,
)
from app.services.economic_taxonomy_publication_contracts import (
    BenchmarkRejected,
    CompatibilityNotAcknowledged,
    InjectedPublicationCrash,
    ManifestChanged,
    PreparationContext,
    PreparedGeneration,
    PublicationCutoff,
    PublicationError,
    PublicationForbidden,
    PublishedGeneration,
    ReaderCapabilityRejected,
)
from app.services.economic_taxonomy_publication_preparation import (
    BenchmarkVerifier,
    CompatibilityBuilder,
    PublicationPreparer,
)
from app.services.economic_taxonomy_publication_validation import (
    publication_change_reason,
    switch_reader_pointers,
    verified_benchmark,
    verify_generation,
)
from app.services.economic_taxonomy_rollback_recovery import RollbackRecovery
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService

__all__ = (
    "BenchmarkRejected",
    "CompatibilityNotAcknowledged",
    "CompatibilityProjection",
    "EconomicTaxonomyPublicationCoordinator",
    "InjectedPublicationCrash",
    "ManifestChanged",
    "PreparationContext",
    "PreparedGeneration",
    "PublicationCutoff",
    "PublicationError",
    "PublicationForbidden",
    "PublishedGeneration",
    "ReaderCapabilityRejected",
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EconomicTaxonomyPublicationCoordinator:
    REQUIRED_BACKEND_CONTRACT = 1
    REQUIRED_FRONTEND_CONTRACT = 1
    REQUIRED_MIGRATION_VERSION = "0055"
    REQUIRED_POLICY_BUNDLE = "economic-taxonomy-v1"
    READER_KEYS = ("economic_taxonomy", "economic_themes")

    def __init__(
        self,
        session_factory,
        *,
        clock: Callable[[], datetime] = _utcnow,
        compatibility_builder: CompatibilityBuilder | None = None,
        benchmark_verifier: BenchmarkVerifier | None = None,
    ):
        self.session_factory = session_factory
        self.clock = clock
        self.compatibility_builder = (
            compatibility_builder or build_default_compatibility_projections
        )
        self.benchmark_verifier = benchmark_verifier or partial(
            verified_benchmark,
            policy_bundle=self.REQUIRED_POLICY_BUNDLE,
        )
        self.rollback_recovery = RollbackRecovery(
            session_factory,
            clock=clock,
        )

    def capture_cutoff(
        self,
        *,
        principal: AdminPrincipal,
        selections: Sequence[Mapping[str, Any]] | None = None,
    ) -> PublicationCutoff:
        self._require_admin(principal)
        with self.session_factory() as session:
            with exclusive_publication(session) as authority:
                if authority.processing_taxonomy_version_id is None:
                    raise PublicationInvariantError(
                        "processing_taxonomy_version_required"
                    )
                taxonomy = session.get(
                    TaxonomyVersion, authority.processing_taxonomy_version_id
                )
                if taxonomy is None or taxonomy.status != "sealed":
                    raise PublicationInvariantError(
                        "sealed_processing_taxonomy_required"
                    )
                selected = (
                    [dict(value) for value in selections]
                    if selections is not None
                    else self._current_selections(session, authority)
                )
                manifest = EconomicTaxonomyPublicationRepository(
                    session
                ).capture_manifest(
                    actor=principal.subject,
                    selections=selected,
                    authority=authority,
                )
                session.flush()
                cutoff = PublicationCutoff(
                    manifest_id=manifest.id,
                    manifest_semantic_hash=str(manifest.semantic_hash),
                    processing_taxonomy_version_id=taxonomy.id,
                    processing_head_revision=authority.processing_head_revision,
                    expected_parent_generation_id=authority.serving_generation_id,
                    semantic_invalidation_revision=(
                        authority.semantic_invalidation_revision
                    ),
                    authority_epoch=authority.authority_epoch,
                    committed_revision_tuples=tuple(
                        tuple(value) for value in manifest.committed_revision_tuples
                    ),
                )
                session.commit()
            return cutoff

    def prepare_generation(
        self,
        cutoff: PublicationCutoff,
        *,
        principal: AdminPrincipal,
        reader_capability_manifest_id: UUID,
        target_mode: str | None = None,
    ) -> PreparedGeneration:
        self._require_admin(principal)
        target_mode = target_mode or self._current_mode()
        if target_mode not in {"legacy", "shadow", "dual", "economic"}:
            raise PublicationInvariantError("invalid_target_mode")
        return PublicationPreparer(
            self.session_factory,
            clock=self.clock,
            compatibility_builder=self.compatibility_builder,
            benchmark_verifier=self.benchmark_verifier,
            required_backend_contract=self.REQUIRED_BACKEND_CONTRACT,
            required_frontend_contract=self.REQUIRED_FRONTEND_CONTRACT,
            required_migration_version=self.REQUIRED_MIGRATION_VERSION,
            required_policy_bundle=self.REQUIRED_POLICY_BUNDLE,
        ).prepare(
            cutoff,
            principal=principal,
            reader_capability_manifest_id=reader_capability_manifest_id,
            target_mode=target_mode,
        )

    def publish_generation(
        self,
        generation_id: UUID,
        *,
        principal: AdminPrincipal,
        notify_delivery_workers: Callable[[], Any] | None = None,
        before_commit: Callable[[], Any] | None = None,
    ) -> PublishedGeneration:
        self._require_admin(principal)
        changed: str | None = None
        result: PublishedGeneration | None = None
        with self.session_factory() as session:  # noqa: SIM117
            with exclusive_publication(session) as authority:
                generation = session.get(ServingGeneration, generation_id)
                if generation is None:
                    raise KeyError(f"serving generation {generation_id} not found")
                events = self._generation_events(session, generation.id)
                event_types = {event.event_type for event in events}
                if "published" in event_types:
                    result = PublishedGeneration(
                        id=generation.id,
                        authority_epoch=authority.authority_epoch,
                        mode=authority.mode,
                        rollback_state=authority.rollback_state,
                    )
                elif "abandoned" in event_types:
                    raise ManifestChanged("generation_abandoned")
                else:
                    prepared = next(
                        (event for event in events if event.event_type == "prepared"),
                        None,
                    )
                    if prepared is None:
                        raise PublicationInvariantError("generation_not_prepared")
                    changed = publication_change_reason(
                        session, authority, generation, prepared.details
                    )
                    if changed is not None:
                        self._append_generation_event(
                            session,
                            generation.id,
                            "abandoned",
                            principal.subject,
                            {"reason": changed},
                        )
                    else:
                        verify_generation(
                            session,
                            authority,
                            generation,
                            prepared.details,
                            reader_keys=self.READER_KEYS,
                            backend_contract=self.REQUIRED_BACKEND_CONTRACT,
                            frontend_contract=self.REQUIRED_FRONTEND_CONTRACT,
                            migration_version=self.REQUIRED_MIGRATION_VERSION,
                            policy_bundle=self.REQUIRED_POLICY_BUNDLE,
                        )
                        previous_id = authority.serving_generation_id
                        target_mode = str(prepared.details["target_mode"])
                        next_epoch = authority.authority_epoch + 1
                        switch_reader_pointers(
                            session,
                            generation=generation,
                            authority_epoch=next_epoch,
                            reader_keys=self.READER_KEYS,
                        )
                        authority.serving_generation_id = generation.id
                        authority.processing_taxonomy_version_id = (
                            generation.taxonomy_version_id
                        )
                        authority.mode = target_mode
                        authority.authority_epoch = next_epoch
                        authority.cutover_catch_up_cursor = list(
                            session.get(
                                GenerationInputManifest,
                                generation.generation_input_manifest_id,
                            ).committed_revision_tuples
                        )
                        projection_count = session.scalar(
                            select(func.count(TaxonomyProjectionEvent.id)).where(
                                TaxonomyProjectionEvent.serving_generation_id
                                == generation.id,
                                TaxonomyProjectionEvent.delivery_scope == "candidate",
                            )
                        )
                        authority.rollback_state = (
                            "temporarily_unavailable"
                            if target_mode != "legacy" and int(projection_count or 0)
                            else "ready"
                        )
                        authority.rollback_reason = (
                            "compatibility_delivery_pending"
                            if authority.rollback_state == "temporarily_unavailable"
                            else None
                        )
                        authority.writes_fenced = False
                        self._append_generation_event(
                            session,
                            generation.id,
                            "published",
                            principal.subject,
                            {
                                "authority_epoch": next_epoch,
                                "mode": target_mode,
                                "manifest_id": str(
                                    generation.generation_input_manifest_id
                                ),
                            },
                        )
                        if previous_id is not None:
                            self._append_generation_event(
                                session,
                                previous_id,
                                "superseded",
                                principal.subject,
                                {"by_generation_id": str(generation.id)},
                            )
                        if before_commit is not None:
                            before_commit()
                        result = PublishedGeneration(
                            id=generation.id,
                            authority_epoch=next_epoch,
                            mode=target_mode,
                            rollback_state=authority.rollback_state,
                        )
                session.commit()

        if changed is not None:
            raise ManifestChanged(changed)
        if result is None:
            raise PublicationError("publication_result_missing")
        EconomicTaxonomyRuntimeService.notify_delivery_workers(notify_delivery_workers)
        return result

    def abandon(
        self,
        generation_id: UUID,
        *,
        principal: AdminPrincipal,
        reason: str,
    ) -> None:
        self._require_admin(principal)
        if not reason.strip():
            raise ValueError("abandon_reason_required")
        with self.session_factory() as session:
            generation = session.get(ServingGeneration, generation_id)
            if generation is None:
                raise KeyError(f"serving generation {generation_id} not found")
            event_types = {
                event.event_type
                for event in self._generation_events(session, generation.id)
            }
            if "published" in event_types:
                raise PublicationError("published_generation_cannot_be_abandoned")
            if "abandoned" not in event_types:
                self._append_generation_event(
                    session,
                    generation.id,
                    "abandoned",
                    principal.subject,
                    {"reason": reason},
                )
            session.commit()

    def rollback(
        self,
        *,
        principal: AdminPrincipal,
        reader_capability_manifest_id: UUID | None = None,
        reason: str = "operator_requested_rollback",
    ) -> PublishedGeneration:
        self._require_admin(principal)
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            if authority is None or authority.serving_generation_id is None:
                raise PublicationError("serving_generation_required_for_rollback")
            current_id = authority.serving_generation_id
            current = session.get(ServingGeneration, current_id)
            if current is None:
                raise PublicationError("serving_generation_missing")
            capability_id = reader_capability_manifest_id or (
                current.reader_capability_manifest_id
            )
            selections = list(
                session.get(
                    GenerationInputManifest,
                    current.generation_input_manifest_id,
                ).selections
            )
            healthy = EconomicTaxonomyRuntimeService(session).generation_acknowledged(
                current_id
            )

        if not healthy:
            self.rollback_recovery.begin(
                generation_id=current_id,
                principal=principal,
                reason=reason,
            )
            try:
                self.rollback_recovery.rebuild_legacy_projections(
                    current_id, principal=principal
                )
            except Exception as exc:
                self.rollback_recovery.mark_failed(str(exc))
                raise

        # Before the legacy generation is built, and again once legacy mode
        # has stopped the economic producer, for what it recorded in between.
        self.rollback_recovery.project_developments()
        cutoff = self.capture_cutoff(principal=principal, selections=selections)
        prepared = self.prepare_generation(
            cutoff,
            principal=principal,
            reader_capability_manifest_id=capability_id,
            target_mode="legacy",
        )
        published = self.publish_generation(prepared.id, principal=principal)
        self.rollback_recovery.project_developments()
        return published

    def refresh_rollback_availability(self) -> str:
        """Promote rollback readiness after asynchronous mirrors catch up."""

        return self.rollback_recovery.refresh_availability()

    @staticmethod
    def _generation_events(
        session: Session, generation_id: UUID
    ) -> list[ServingGenerationEvent]:
        return list(
            session.scalars(
                select(ServingGenerationEvent)
                .where(ServingGenerationEvent.serving_generation_id == generation_id)
                .order_by(ServingGenerationEvent.sequence_number)
            )
        )

    def _append_generation_event(
        self,
        session: Session,
        generation_id: UUID,
        event_type: str,
        actor: str,
        details: Mapping[str, Any],
    ) -> ServingGenerationEvent:
        latest = session.scalar(
            select(func.max(ServingGenerationEvent.sequence_number)).where(
                ServingGenerationEvent.serving_generation_id == generation_id
            )
        )
        event = ServingGenerationEvent(
            serving_generation_id=generation_id,
            sequence_number=int(latest or 0) + 1,
            event_type=event_type,
            actor=actor,
            details=dict(details),
        )
        session.add(event)
        session.flush()
        return event

    def _current_mode(self) -> str:
        with self.session_factory() as session:
            authority = session.get(TaxonomyAuthority, 1)
            return authority.mode if authority is not None else "legacy"

    @staticmethod
    def _current_selections(
        session: Session, authority: TaxonomyAuthority
    ) -> list[dict[str, Any]]:
        previous: dict[str, dict[str, Any]] = {}
        if authority.serving_generation_id is not None:
            generation = session.get(ServingGeneration, authority.serving_generation_id)
            if generation is not None:
                manifest = session.get(
                    GenerationInputManifest, generation.generation_input_manifest_id
                )
                previous = {
                    str(value.get("lineage")): dict(value)
                    for value in (manifest.selections or [])
                    if value.get("lineage")
                }

        interpretation_service = EconomicTaxonomyInterpretationService(
            lambda: session
        )
        selections: list[dict[str, Any]] = []
        for lineage in session.scalars(
            select(SourceLineage).order_by(SourceLineage.id)
        ):
            lineage_id = lineage.id
            precedence_revision = session.scalar(
                select(func.max(EvidencePrecedenceRevision.revision_number)).where(
                    EvidencePrecedenceRevision.source_lineage_id == lineage_id
                )
            )
            if precedence_revision is None:
                continue
            packet = session.scalar(
                select(EvidencePacket)
                .join(
                    EvidencePrecedenceRevision,
                    EvidencePrecedenceRevision.evidence_packet_id
                    == EvidencePacket.id,
                )
                .where(
                    EvidencePrecedenceRevision.source_lineage_id == lineage_id,
                    EvidencePrecedenceRevision.disposition == "effective",
                )
                .order_by(EvidencePrecedenceRevision.revision_number.desc())
                .limit(1)
            )
            if packet is None:
                continue
            attempt = interpretation_service._default_attempt(
                session,
                lineage_id,
                precedence_cutoff=precedence_revision,
            )
            old = previous.get(str(lineage_id), {})
            override_id = old.get("interpretation_override_revision_id")
            if override_id is not None:
                try:
                    selected_attempt_id = UUID(str(old["selected_attempt_id"]))
                    parsed_override_id = UUID(str(override_id))
                    selected = session.get(ClassificationAttempt, selected_attempt_id)
                    if selected is None or not interpretation_service._attempt_completed(
                        session, selected
                    ):
                        raise ValueError("override attempt is not completed")
                    interpretation_service._validated_override(
                        session,
                        parsed_override_id,
                        lineage_id=lineage_id,
                        selected_attempt_id=selected.id,
                    )
                    selected_packet = interpretation_service._packet_for_attempt(
                        session, selected
                    )
                    if (
                        selected_packet.source_lineage_id != lineage_id
                        or interpretation_service._packet_precedence_state(
                            session,
                            selected_packet.id,
                            cutoff=precedence_revision,
                        )
                        not in {"effective", "equivalent"}
                    ):
                        raise ValueError("override evidence is no longer selectable")
                except (KeyError, TypeError, ValueError) as exc:
                    raise PublicationInvariantError(
                        "interpretation_override_replacement_required"
                    ) from exc
                attempt = selected
                packet = selected_packet
            elif attempt is not None:
                # A newer effective packet may still be awaiting processing.
                # Keep serving the default completed attempt with its own
                # packet until the newer packet has a completed attempt.
                packet = interpretation_service._packet_for_attempt(session, attempt)
            eligibility = session.scalar(
                select(LensEligibilityRevision)
                .where(LensEligibilityRevision.evidence_packet_id == packet.id)
                .order_by(LensEligibilityRevision.revision_number.desc())
                .limit(1)
            )
            if eligibility is None:
                if override_id is not None:
                    raise PublicationInvariantError(
                        "interpretation_override_replacement_required"
                    )
                continue
            development_selections = (
                EconomicTaxonomyPublicationCoordinator._development_selections(
                    session, lineage.source_family_id
                )
            )
            projection_revision = session.scalar(
                select(func.max(TaxonomyProjectionEvent.projection_revision)).where(
                    TaxonomyProjectionEvent.source_lineage == str(lineage_id)
                )
            )
            selection = {
                    "lineage": str(lineage_id),
                    "evidence_packet_id": str(packet.id),
                    "selected_attempt_id": str(attempt.id) if attempt else None,
                    "eligibility_revision": eligibility.revision_number,
                    "evidence_precedence_revision": int(precedence_revision),
                    "interpretation_override_revision_id": (
                        str(override_id) if override_id is not None else None
                    ),
                    "constituent_decision_revision": old.get(
                        "constituent_decision_revision"
                    ),
                    "social_association_revision": None,
                    "social_decision_revision": None,
                    "development_identity": (
                        development_selections[0]["development_identity"]
                        if len(development_selections) == 1
                        else None
                    ),
                    "development_revision": (
                        development_selections[0]["revision_number"]
                        if len(development_selections) == 1
                        else None
                    ),
                    "mapping_revision": max(1, authority.processing_head_revision),
                    "metrics_policy_revision": 1,
                    "compatibility_projection_revision": max(
                        1, int(projection_revision or 0) + 1
                    ),
                }
            if len(development_selections) > 1:
                selection["development_selections"] = development_selections
            selections.append(selection)
        social_revisions = (
            EconomicTaxonomyPublicationCoordinator._current_social_revisions(session)
        )
        if social_revisions:
            if not selections:
                raise PublicationInvariantError(
                    "social_revision_without_source_selection"
                )
            selections[0]["social_association_revision"] = social_revisions[0]
            selections[0]["social_association_revisions"] = social_revisions
        return selections

    @staticmethod
    def _current_social_revisions(session: Session) -> list[dict[str, Any]]:
        latest_by_association = {}
        for revision in session.scalars(
            select(EconomicSocialAssociationRevision).order_by(
                EconomicSocialAssociationRevision.association_id,
                EconomicSocialAssociationRevision.revision_number.desc(),
            )
        ):
            latest_by_association.setdefault(revision.association_id, revision)
        adapter = EconomicSocialTaxonomyAdapter(session)
        result = []
        for association_id in sorted(latest_by_association, key=str):
            revision = latest_by_association[association_id]
            ref = adapter.pin_revision(revision.id)
            result.append(
                {
                    "association_id": str(ref.association_id),
                    "revision_number": ref.revision_number,
                    "decision_revision_id": (
                        str(ref.decision_revision_id)
                        if ref.decision_revision_id is not None
                        else None
                    ),
                }
            )
        return result

    @staticmethod
    def _development_selections(
        session: Session, source_family_id: UUID
    ) -> list[dict[str, Any]]:
        identities = {
            value
            for value in session.scalars(
                select(ThemeDevelopmentEvent.development_identity)
                .join(
                    ThemeDevelopmentObservation,
                    ThemeDevelopmentObservation.event_id == ThemeDevelopmentEvent.id,
                )
                .where(
                    ThemeDevelopmentObservation.source_family_id == source_family_id,
                    ThemeDevelopmentEvent.development_identity.is_not(None),
                )
            )
            if value is not None
        }
        selections = []
        for identity in sorted(identities, key=str):
            revision = session.scalar(
                select(DevelopmentSelectionRevision)
                .where(
                    DevelopmentSelectionRevision.development_identity == identity
                )
                .order_by(DevelopmentSelectionRevision.revision_number.desc())
                .limit(1)
            )
            if revision is not None:
                selections.append(
                    {
                        "development_identity": str(identity),
                        "revision_number": revision.revision_number,
                    }
                )
        return selections

    @staticmethod
    def _require_admin(principal: AdminPrincipal) -> None:
        if not principal.can_review_taxonomy:
            raise PublicationForbidden("taxonomy_publication_forbidden")

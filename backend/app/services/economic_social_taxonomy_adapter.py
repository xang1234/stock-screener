"""Economic Social association revisions, admission, and legacy mirroring."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.domain.economic_taxonomy.policy import reconcile_social_decisions
from app.infra.db.models.social_analysis import (
    EconomicSocialAssociation,
    EconomicSocialAssociationRevision,
    EconomicSocialAssociationSource,
    EconomicSocialDecisionRevision,
    SocialExtractionWork,
    SocialRunWork,
    SocialThemeAssociation,
    SocialThemeDecision,
)
from app.infra.db.models.social_signals import SocialSignalRun
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.models.economic_taxonomy_runtime import (
    ClaimAssignment,
    ClassificationAttempt,
    EvidencePacket,
    ProcessingRequest,
    SocialAssociationRevisionRef,
    TaxonomyAuthority,
)
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeCluster
from app.services.economic_source_admission import (
    AdmissionResult,
    EconomicSourceAdmissionService,
    EvidenceAdmission,
)
from app.services.economic_taxonomy_fence import producer_write
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.services.theme_identity_normalization import (
    canonical_theme_key,
    social_membership_key,
)
from app.utils.file_hashing import canonical_json_sha256 as _semantic_hash


@dataclass(frozen=True, slots=True)
class EconomicSocialProjectionResult:
    association_id: UUID
    association_revision_id: UUID
    global_association_count: int
    bridge_count: int
    state: str
    live: bool


@dataclass(frozen=True, slots=True)
class EconomicSocialMembership:
    association_id: UUID
    association_revision_id: UUID
    revision_number: int
    economic_theme_id: UUID
    security_id: int
    state: str
    live: bool
    admission_state: str
    mirror_state: str


@dataclass(frozen=True, slots=True)
class SocialEvidenceAdmissionResult:
    source_family_id: UUID
    source_lineage_id: UUID
    packet_id: UUID
    effective_packet_id: UUID | None
    evidence_revision_ordinal: int
    precedence_state: str
    admission_state: str
    live: bool


class EconomicSocialTaxonomyAdapter:
    """Preserve legacy Social history while projecting global memberships."""

    def __init__(self, db):
        self.db = db

    def get_or_create_association(
        self, economic_theme_id: UUID, security_id: int
    ) -> EconomicSocialAssociation:
        association = self.db.scalar(
            select(EconomicSocialAssociation).where(
                EconomicSocialAssociation.economic_theme_id == economic_theme_id,
                EconomicSocialAssociation.security_id == security_id,
            )
        )
        if association is not None:
            return association
        try:
            with self.db.begin_nested():
                association = EconomicSocialAssociation(
                    economic_theme_id=economic_theme_id,
                    security_id=security_id,
                )
                self.db.add(association)
                self.db.flush()
                return association
        except IntegrityError:
            return self.db.scalar(
                select(EconomicSocialAssociation).where(
                    EconomicSocialAssociation.economic_theme_id
                    == economic_theme_id,
                    EconomicSocialAssociation.security_id == security_id,
                )
            )

    def project_legacy_associations(
        self,
        *,
        economic_theme_id: UUID,
        security_id: int,
        legacy_association_ids: tuple[int, ...],
    ) -> EconomicSocialProjectionResult:
        expected_epoch = self._current_epoch()
        with producer_write(
            self.db,
            expected_epoch=expected_epoch,
            allowed_modes={"legacy", "shadow", "dual", "economic"},
        ) as authority:
            return self._project_legacy_associations(
                economic_theme_id=economic_theme_id,
                security_id=security_id,
                legacy_association_ids=legacy_association_ids,
                authority_epoch=authority.authority_epoch,
            )

    def _project_legacy_associations(
        self,
        *,
        economic_theme_id: UUID,
        security_id: int,
        legacy_association_ids: tuple[int, ...],
        authority_epoch: int,
    ) -> EconomicSocialProjectionResult:
        if not legacy_association_ids:
            raise ValueError("legacy_association_ids_required")
        association = self.get_or_create_association(economic_theme_id, security_id)
        legacy_rows = self.db.scalars(
            select(SocialThemeAssociation).where(
                SocialThemeAssociation.id.in_(set(legacy_association_ids))
            )
        ).all()
        if {row.id for row in legacy_rows} != set(legacy_association_ids):
            raise ValueError("legacy_association_missing")
        security = self.db.get(StockUniverse, security_id)
        if security is None or any(
            row.market != security.market or row.canonical_symbol != security.symbol
            for row in legacy_rows
        ):
            raise ValueError("legacy_association_security_mismatch")

        for legacy in legacy_rows:
            source_key = f"legacy_association:{legacy.id}"
            existing = self.db.scalar(
                select(EconomicSocialAssociationSource).where(
                    EconomicSocialAssociationSource.association_id == association.id,
                    EconomicSocialAssociationSource.source_kind
                    == "legacy_association",
                    EconomicSocialAssociationSource.source_key == source_key,
                )
            )
            if existing is None:
                self.db.add(
                    EconomicSocialAssociationSource(
                        association_id=association.id,
                        source_kind="legacy_association",
                        source_key=source_key,
                        legacy_association_id=legacy.id,
                    )
                )
            for work_id in sorted(set(legacy.evidence_work_ids or ())):
                work_key = f"social_work:{work_id}"
                work_source = self.db.scalar(
                    select(EconomicSocialAssociationSource).where(
                        EconomicSocialAssociationSource.association_id
                        == association.id,
                        EconomicSocialAssociationSource.source_kind == "social_work",
                        EconomicSocialAssociationSource.source_key == work_key,
                    )
                )
                if work_source is None:
                    self.db.add(
                        EconomicSocialAssociationSource(
                            association_id=association.id,
                            source_kind="social_work",
                            source_key=work_key,
                            social_work_id=work_id,
                        )
                    )
        self.db.flush()

        all_legacy_ids = tuple(
            self.db.scalars(
                select(EconomicSocialAssociationSource.legacy_association_id).where(
                    EconomicSocialAssociationSource.association_id == association.id,
                    EconomicSocialAssociationSource.source_kind
                    == "legacy_association",
                    EconomicSocialAssociationSource.legacy_association_id.is_not(None),
                )
            )
        )
        legacy_rows = self.db.scalars(
            select(SocialThemeAssociation).where(
                SocialThemeAssociation.id.in_(all_legacy_ids)
            )
        ).all()

        admin_states = {
            row.state for row in legacy_rows if row.decision_owner == "admin"
        }
        if {"accepted", "rejected"} <= admin_states:
            reconciled = reconcile_social_decisions(("accepted", "rejected"))
        elif "rejected" in admin_states:
            reconciled = reconcile_social_decisions(("rejected",))
        elif "accepted" in admin_states:
            reconciled = reconcile_social_decisions(("accepted",))
        else:
            reconciled = reconcile_social_decisions(
                tuple(
                    row.state for row in sorted(legacy_rows, key=lambda row: row.id)
                )
            )
        legacy_decisions = self.db.scalars(
            select(SocialThemeDecision).where(
                SocialThemeDecision.association_id.in_(set(all_legacy_ids))
            )
        ).all()
        source_payload = {
            "legacy_associations": [
                {
                    "association_id": row.id,
                    "decision_owner": row.decision_owner,
                    "state": row.state,
                    "version": row.version,
                }
                for row in sorted(legacy_rows, key=lambda row: row.id)
            ],
            "legacy_decision_ids": sorted(row.id for row in legacy_decisions),
        }
        digest = _semantic_hash(source_payload)
        revision = self.db.scalar(
            select(EconomicSocialAssociationRevision).where(
                EconomicSocialAssociationRevision.association_id == association.id,
                EconomicSocialAssociationRevision.reconciliation_hash == digest,
            )
        )
        if revision is None:
            decision = self._create_decision(
                association.id,
                state=reconciled.state,
                idempotency_key=f"legacy-reconciliation:{digest}",
                actor="system:economic-social-reconciliation",
                reason="legacy_association_reconciliation",
                source_payload=source_payload,
            )
            revision = self._create_revision(
                association.id,
                state=reconciled.state,
                live=reconciled.live,
                admission_state="live",
                mirror_state="acknowledged",
                reconciliation_hash=digest,
                decision_revision_id=decision.id,
                details=source_payload,
                authority_epoch=authority_epoch,
            )
        bridge_count = self.db.scalar(
            select(func.count())
            .select_from(EconomicSocialAssociationSource)
            .where(
                EconomicSocialAssociationSource.association_id == association.id,
                EconomicSocialAssociationSource.source_kind == "legacy_association",
            )
        )
        return EconomicSocialProjectionResult(
            association_id=association.id,
            association_revision_id=revision.id,
            global_association_count=1,
            bridge_count=int(bridge_count or 0),
            state=revision.state,
            live=revision.live,
        )

    def revise(
        self,
        association_id: UUID,
        *,
        state: str,
        idempotency_key: str,
        actor: str,
        reason: str,
        mirror_acknowledged: bool,
        admission_state: str = "live",
        evidence_packet_id: UUID | None = None,
        expected_revision: int | None = None,
    ) -> EconomicSocialAssociationRevision:
        expected_epoch = self._current_epoch()
        with producer_write(
            self.db,
            expected_epoch=expected_epoch,
            allowed_modes={"legacy", "shadow", "dual", "economic"},
        ) as authority:
            return self._revise(
                association_id,
                state=state,
                idempotency_key=idempotency_key,
                actor=actor,
                reason=reason,
                mirror_acknowledged=mirror_acknowledged,
                admission_state=admission_state,
                evidence_packet_id=evidence_packet_id,
                authority_epoch=authority.authority_epoch,
                expected_revision=expected_revision,
            )

    def _revise(
        self,
        association_id: UUID,
        *,
        state: str,
        idempotency_key: str,
        actor: str,
        reason: str,
        mirror_acknowledged: bool,
        admission_state: str,
        evidence_packet_id: UUID | None,
        authority_epoch: int,
        expected_revision: int | None = None,
    ) -> EconomicSocialAssociationRevision:
        if state not in {
            "proposed",
            "accepted",
            "rejected",
            "conflict_review_required",
        }:
            raise ValueError("invalid_economic_social_decision")
        existing_decision = self.db.scalar(
            select(EconomicSocialDecisionRevision).where(
                EconomicSocialDecisionRevision.association_id == association_id,
                EconomicSocialDecisionRevision.idempotency_key == idempotency_key,
            )
        )
        if existing_decision is not None:
            if (
                existing_decision.state != state
                or existing_decision.actor != actor
                or existing_decision.reason != reason
            ):
                raise ValueError("economic_social_decision_idempotency_conflict")
            return self.db.scalar(
                select(EconomicSocialAssociationRevision).where(
                    EconomicSocialAssociationRevision.decision_revision_id
                    == existing_decision.id
                )
            )
        # Checked under the producer fence, so concurrent decisions on the
        # same revision serialize and the later one sees the newer revision.
        if expected_revision is not None and expected_revision != self.db.scalar(
            select(func.max(EconomicSocialAssociationRevision.revision_number)).where(
                EconomicSocialAssociationRevision.association_id == association_id
            )
        ):
            raise ValueError("association_version_conflict")
        decision = self._create_decision(
            association_id,
            state=state,
            idempotency_key=idempotency_key,
            actor=actor,
            reason=reason,
            source_payload={},
        )
        authority = self.db.get(TaxonomyAuthority, 1)
        mirror_enabled = (
            authority is not None
            and authority.mode in {"dual", "economic"}
            and authority.serving_generation_id is not None
        )
        pending_mirror = state == "accepted" and not mirror_acknowledged
        # A previously mirrored acceptance must be retracted from legacy
        # readers, otherwise rollback would expose the stale acceptance.
        pending_retraction = (
            state != "accepted"
            and mirror_enabled
            and bool(self._accepted_legacy_mirrors(association_id))
        )
        effective_state = "pending_legacy_mirror" if pending_mirror else state
        live = (
            state == "accepted"
            and admission_state == "live"
            and mirror_acknowledged
        )
        projection_event_id = None
        if mirror_enabled and (pending_mirror or pending_retraction):
            association = self.db.get(EconomicSocialAssociation, association_id)
            lineage = f"economic-social-association:{association_id}"
            event = EconomicTaxonomyRuntimeService(self.db).stage_projection_fanout(
                generation_id=authority.serving_generation_id,
                affected_lineages=(lineage,),
                projection_kind="social_membership",
                projection_version=1,
                target="legacy",
                payload_by_lineage={
                    lineage: {
                        "association_id": str(association_id),
                        "economic_theme_id": str(association.economic_theme_id),
                        "security_id": association.security_id,
                        "state": state,
                    }
                },
                staged_epoch=authority.authority_epoch,
                origin_representation="economic",
                selected_interpretation_version=f"social-decision:{decision.id}",
                mapping_version="economic-social-v1",
            )[0]
            projection_event_id = event.id
        return self._create_revision(
            association_id,
            state=effective_state,
            live=live,
            admission_state=admission_state,
            mirror_state=(
                "pending"
                if pending_mirror or pending_retraction
                else "acknowledged" if state == "accepted" else "not_required"
            ),
            reconciliation_hash=_semantic_hash(
                {
                    "decision_revision_id": decision.id,
                    "admission_state": admission_state,
                    "mirror_acknowledged": mirror_acknowledged,
                    "evidence_packet_id": evidence_packet_id,
                }
            ),
            decision_revision_id=decision.id,
            evidence_packet_id=evidence_packet_id,
            projection_event_id=projection_event_id,
            details={"requested_state": state},
            authority_epoch=authority_epoch,
        )

    def project_native_assignments(
        self,
        assignments,
        *,
        evidence_packet_id: UUID,
        authority_epoch: int,
        social_evidence_packet_id: UUID | None = None,
    ) -> tuple[EconomicSocialAssociationRevision, ...]:
        """Project classified Social claims while the processor fence is held."""

        packet = self.db.get(EvidencePacket, evidence_packet_id)
        if packet is None:
            raise KeyError(f"evidence packet {evidence_packet_id} not found")
        social_packets = []
        if social_evidence_packet_id is not None:
            social_packet = self.db.get(EvidencePacket, social_evidence_packet_id)
            if (
                social_packet is None
                or social_packet.source_lineage_id != packet.source_lineage_id
            ):
                raise ValueError("social_evidence_packet_lineage_mismatch")
            social_packets = [social_packet]
        else:
            social_packets = [
                row
                for row in self.db.scalars(
                    select(EvidencePacket)
                    .where(EvidencePacket.source_lineage_id == packet.source_lineage_id)
                    .order_by(
                        EvidencePacket.evidence_revision_ordinal.desc(),
                        EvidencePacket.id,
                    )
                ).all()
                if isinstance(row.source_metadata, dict)
                and row.source_metadata.get("social_work_id") is not None
                and row.precedence_state in {"effective", "equivalent"}
            ]
        if not social_packets:
            return ()
        social_packet = social_packets[0]
        if (
            not isinstance(social_packet.source_metadata, dict)
            or social_packet.source_metadata.get("social_work_id") is None
            or social_packet.precedence_state not in {"effective", "equivalent"}
        ):
            return ()
        metadata = dict(social_packet.source_metadata or {})
        if metadata.get("social_admission_state", "live") != "live":
            return ()
        work_id = metadata.get("social_work_id")
        if not isinstance(work_id, int) or isinstance(work_id, bool):
            return ()
        state_by_pair = {
            (str(row.get("theme_key") or ""), row.get("security_id")): str(
                row.get("state") or "proposed"
            )
            for row in metadata.get("social_memberships") or ()
            if isinstance(row, dict)
        }
        membership_by_key = {
            str(
                row.get("membership_key")
                or social_membership_key(
                    str(row.get("theme_key") or ""), row.get("security_id")
                )
            ): row
            for row in metadata.get("social_memberships") or ()
            if isinstance(row, dict)
            and isinstance(row.get("security_id"), int)
            and not isinstance(row.get("security_id"), bool)
        }
        revisions = []
        for assignment in assignments:
            theme_key = canonical_theme_key(
                str(assignment.claim_payload.get("display_name") or "")
            )
            for security_payload in assignment.claim_payload.get("securities") or ():
                if not isinstance(security_payload, dict):
                    continue
                security_id = security_payload.get("security_id")
                if (
                    not isinstance(security_id, int)
                    or isinstance(security_id, bool)
                    or self.db.get(StockUniverse, security_id) is None
                ):
                    continue
                association = self.get_or_create_association(
                    assignment.economic_theme_id, security_id
                )
                source_key = f"social_work:{work_id}:packet:{social_packet.id}"
                source = self.db.scalar(
                    select(EconomicSocialAssociationSource).where(
                        EconomicSocialAssociationSource.association_id
                        == association.id,
                        EconomicSocialAssociationSource.source_kind == "social_work",
                        EconomicSocialAssociationSource.source_key == source_key,
                    )
                )
                if source is None:
                    self.db.add(
                        EconomicSocialAssociationSource(
                            association_id=association.id,
                            source_kind="social_work",
                            source_key=source_key,
                            social_work_id=work_id,
                            evidence_packet_id=social_packet.id,
                        )
                    )
                    self.db.flush()
                linked_states = [
                    str(membership_by_key[key].get("state") or "proposed")
                    for key in assignment.claim_payload.get(
                        "source_membership_keys"
                    ) or ()
                    if key in membership_by_key
                    and membership_by_key[key].get("security_id") == security_id
                ]
                requested_state = (
                    reconcile_social_decisions(linked_states).state
                    if linked_states
                    else state_by_pair.get((theme_key, security_id), "proposed")
                )
                if requested_state not in {
                    "proposed",
                    "accepted",
                    "rejected",
                    "conflict_review_required",
                }:
                    requested_state = "proposed"
                current = self.db.scalar(
                    select(EconomicSocialAssociationRevision)
                    .where(
                        EconomicSocialAssociationRevision.association_id
                        == association.id
                    )
                    .order_by(
                        EconomicSocialAssociationRevision.revision_number.desc()
                    )
                    .limit(1)
                )
                current_state = (
                    str((current.details or {}).get("requested_state") or current.state)
                    if current is not None
                    else None
                )
                if current_state == "conflict_review_required":
                    continue
                if (
                    requested_state == "proposed"
                    and current_state in {None, "proposed"}
                    and self._automatically_accepted(
                        association, now=social_packet.available_at
                    )
                ):
                    requested_state = "accepted"
                if current_state in {"accepted", "rejected"} and requested_state == "proposed":
                    continue
                if current_state == "rejected" and requested_state == "accepted":
                    continue
                if current_state == requested_state:
                    continue
                revisions.append(
                    self._revise(
                        association.id,
                        state=requested_state,
                        idempotency_key=(
                            f"classification:{assignment.classification_attempt_id}:"
                            f"assignment:{assignment.id}:security:{security_id}:"
                            f"social-packet:{social_packet.id}"
                        ),
                        actor="system:economic-taxonomy-refresh",
                        reason="classified_social_membership",
                        mirror_acknowledged=requested_state != "accepted",
                        admission_state="live",
                        evidence_packet_id=social_packet.id,
                        authority_epoch=authority_epoch,
                    )
                )
        return tuple(revisions)

    def _automatically_accepted(self, association, *, now) -> bool:
        """The automatic Social rule on economic evidence (#515).

        Two independent authors within 14 days, for a verified company,
        counted over the association's Social works. The window ends at the
        newest Social packet the association has seen (``now`` at least), so
        processing an older packet last still counts the newer posts. A claim
        the catalog places under another theme does not count.
        """
        from app.services.social_company_identity_service import SocialCompanyIdentityService
        from app.services.social_theme_projection_service import (
            SocialThemeProjectionService,
            _decode,
            qualifying_social_evidence,
        )
        from app.services.social_ticker_resolver import SocialTickerResolver

        def utc(value):  # SQLite drops the zone of a stored UTC time
            return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

        security = self.db.get(StockUniverse, association.security_id)
        identity = SocialCompanyIdentityService(self.db).read()
        resolver = SocialTickerResolver(
            self.db, verified_company_ids=identity.verified_company_ids
        )
        company = resolver.resolve(security.symbol, security.market)
        if not company.company_count_eligible:
            return False
        sources = self.db.execute(
            select(EconomicSocialAssociationSource.social_work_id, EvidencePacket.available_at)
            .outerjoin(EvidencePacket, EvidencePacket.id == EconomicSocialAssociationSource.evidence_packet_id)
            .where(
                EconomicSocialAssociationSource.association_id == association.id,
                EconomicSocialAssociationSource.social_work_id.is_not(None),
            )
        ).all()
        now = max([utc(now), *(utc(at) for _, at in sources if at is not None)])
        catalog = SocialThemeProjectionService(self.db)._economic_catalog()
        decoded = []
        for work_id in sorted({work_id for work_id, _ in sources}):
            work = self.db.get(SocialExtractionWork, work_id)
            created = (work.input_snapshot_json or {}).get("created_at")
            try:
                # Cheap pre-filter: only posts inside the window are decoded.
                if created and not now - timedelta(days=14) <= utc(datetime.fromisoformat(created)) <= now:
                    continue
                decoded.append((work, *_decode(work)))
            except ValueError:
                continue  # unreadable saved work never counts

        def about_this_theme(claim):
            mapped = catalog.get(canonical_theme_key(claim.raw_theme))
            return mapped is None or mapped[0] == association.economic_theme_id

        rows = qualifying_social_evidence(decoded, now, resolver, about_this_theme)
        return len({row[4] for row in rows if row[3] == company.company_id}) >= 2

    def project_equivalent_social_packet(
        self,
        *,
        evidence_packet_id: UUID,
        effective_packet_id: UUID,
        authority_epoch: int,
    ) -> tuple[EconomicSocialAssociationRevision, ...]:
        """Reuse completed classification while applying new Social decisions."""

        packet = self.db.get(EvidencePacket, evidence_packet_id)
        effective = self.db.get(EvidencePacket, effective_packet_id)
        if (
            packet is None
            or effective is None
            or packet.source_lineage_id != effective.source_lineage_id
            or packet.precedence_state != "equivalent"
        ):
            raise ValueError("equivalent_social_packet_mismatch")
        attempt_id = self.db.scalar(
            select(ClassificationAttempt.id)
            .join(
                ProcessingRequest,
                ProcessingRequest.id == ClassificationAttempt.processing_request_id,
            )
            .where(
                ProcessingRequest.evidence_packet_id == effective.id,
                ProcessingRequest.status == "completed",
                ClassificationAttempt.result_status == "completed",
            )
            .order_by(
                ClassificationAttempt.created_at.desc(),
                ClassificationAttempt.id.desc(),
            )
            .limit(1)
        )
        if attempt_id is None:
            return ()
        assignments = self.db.scalars(
            select(ClaimAssignment)
            .where(ClaimAssignment.classification_attempt_id == attempt_id)
            .order_by(ClaimAssignment.created_at, ClaimAssignment.id)
        ).all()
        return self.project_native_assignments(
            assignments,
            evidence_packet_id=effective.id,
            social_evidence_packet_id=packet.id,
            authority_epoch=authority_epoch,
        )

    def apply_legacy_mirror(
        self,
        association_revision_id: UUID,
        *,
        now: datetime | None = None,
    ) -> EconomicSocialAssociationRevision:
        """Create compatibility rows and append an acknowledged global revision."""

        authority = self.db.get(TaxonomyAuthority, 1)
        expected_epoch = authority.authority_epoch if authority is not None else 1
        with producer_write(
            self.db,
            expected_epoch=expected_epoch,
            allowed_modes={"legacy", "shadow", "dual", "economic"},
        ) as authority:
            return self._apply_legacy_mirror(
                association_revision_id,
                now=now,
                authority_epoch=authority.authority_epoch,
            )

    def _apply_legacy_mirror(
        self,
        association_revision_id: UUID,
        *,
        now: datetime | None = None,
        authority_epoch: int,
    ) -> EconomicSocialAssociationRevision:
        """Apply the mirror while the shared authority fence is held."""

        now = now or datetime.now(timezone.utc)
        revision = self.db.get(
            EconomicSocialAssociationRevision, association_revision_id
        )
        if revision is None:
            raise KeyError(
                f"economic social revision {association_revision_id} not found"
            )
        if revision.state != "pending_legacy_mirror":
            if revision.mirror_state == "pending":
                return self._apply_legacy_retraction(
                    revision, now=now, authority_epoch=authority_epoch
                )
            return revision
        association = self.db.get(EconomicSocialAssociation, revision.association_id)
        security = self.db.get(StockUniverse, association.security_id)
        if security is None:
            raise ValueError("economic_social_security_missing")
        canonical_key = f"economic_{association.economic_theme_id.hex}"
        cluster = self.db.scalar(
            select(ThemeCluster).where(
                ThemeCluster.pipeline == "technical",
                ThemeCluster.canonical_key == canonical_key,
            )
        )
        if cluster is None:
            display_name = f"Economic Theme {association.economic_theme_id}"
            cluster = ThemeCluster(
                name=display_name,
                display_name=display_name,
                canonical_key=canonical_key,
                pipeline="technical",
                aliases=[],
                discovery_source="economic_mirror",
                first_seen_at=now,
                last_seen_at=now,
                lifecycle_state="candidate",
                is_active=True,
            )
            self.db.add(cluster)
            self.db.flush()
        legacy = self.db.scalar(
            select(SocialThemeAssociation).where(
                SocialThemeAssociation.theme_cluster_id == cluster.id,
                SocialThemeAssociation.market == security.market,
                SocialThemeAssociation.canonical_symbol == security.symbol,
            )
        )
        if legacy is None:
            legacy = SocialThemeAssociation(
                theme_cluster_id=cluster.id,
                company_key=None,
                market=security.market,
                canonical_symbol=security.symbol,
                state="accepted",
                origin="social",
                decision_owner="system",
                evidence_work_ids=[],
                policy_version="economic-social-v1",
                version=1,
                first_seen_at=now,
                accepted_at=now,
                updated_at=now,
            )
            self.db.add(legacy)
            self.db.flush()
        source_key = f"legacy_association:{legacy.id}"
        bridge = self.db.scalar(
            select(EconomicSocialAssociationSource).where(
                EconomicSocialAssociationSource.association_id == association.id,
                EconomicSocialAssociationSource.source_kind == "legacy_association",
                EconomicSocialAssociationSource.source_key == source_key,
            )
        )
        if bridge is None:
            self.db.add(
                EconomicSocialAssociationSource(
                    association_id=association.id,
                    source_kind="legacy_association",
                    source_key=source_key,
                    legacy_association_id=legacy.id,
                )
            )
            self.db.flush()
        # Every legacy row bridged to this membership mirrors its acceptance,
        # including rows an earlier economic decision retracted.
        for bridged in self._bridged_legacy_rows(association.id):
            if bridged.state != "accepted":
                self._set_legacy_mirror_state(bridged, "accepted", now=now)
                bridged.accepted_at = bridged.accepted_at or now
        digest = _semantic_hash(
            {
                "acknowledges_revision_id": revision.id,
                "legacy_association_id": legacy.id,
            }
        )
        existing = self.db.scalar(
            select(EconomicSocialAssociationRevision).where(
                EconomicSocialAssociationRevision.association_id == association.id,
                EconomicSocialAssociationRevision.reconciliation_hash == digest,
            )
        )
        if existing is not None:
            return existing
        return self._create_revision(
            association.id,
            state="accepted",
            live=revision.admission_state == "live",
            admission_state=revision.admission_state,
            mirror_state="acknowledged",
            reconciliation_hash=digest,
            decision_revision_id=revision.decision_revision_id,
            evidence_packet_id=revision.evidence_packet_id,
            projection_event_id=revision.projection_event_id,
            details={
                "acknowledges_revision_id": str(revision.id),
                "legacy_association_id": legacy.id,
            },
            authority_epoch=authority_epoch,
        )

    def _bridged_legacy_rows(
        self, association_id: UUID
    ) -> list[SocialThemeAssociation]:
        return list(
            self.db.scalars(
                select(SocialThemeAssociation)
                .join(
                    EconomicSocialAssociationSource,
                    EconomicSocialAssociationSource.legacy_association_id
                    == SocialThemeAssociation.id,
                )
                .where(
                    EconomicSocialAssociationSource.association_id == association_id,
                    EconomicSocialAssociationSource.source_kind
                    == "legacy_association",
                )
                .order_by(SocialThemeAssociation.id)
            )
        )

    def _accepted_legacy_mirrors(
        self, association_id: UUID
    ) -> list[SocialThemeAssociation]:
        return [
            row
            for row in self._bridged_legacy_rows(association_id)
            if row.state == "accepted"
        ]

    def _set_legacy_mirror_state(
        self, legacy: SocialThemeAssociation, target: str, *, now: datetime
    ) -> None:
        self.db.add(
            SocialThemeDecision(
                association_id=legacy.id,
                run_id=None,
                actor="system:economic-taxonomy-mirror",
                reason="economic_social_decision_mirror",
                before_state=legacy.state,
                after_state=target,
                policy_version="economic-social-v1",
                evidence_work_ids=list(legacy.evidence_work_ids or []),
                created_at=now,
            )
        )
        legacy.state = target
        legacy.version += 1
        legacy.updated_at = now

    def _apply_legacy_retraction(
        self,
        revision: EconomicSocialAssociationRevision,
        *,
        now: datetime,
        authority_epoch: int,
    ) -> EconomicSocialAssociationRevision:
        """Withdraw a mirrored legacy acceptance after a non-accepted decision."""

        # Legacy rows only model proposed/accepted/rejected; a conflict under
        # review is not an acceptance, so it reads as proposed.
        target = "rejected" if revision.state == "rejected" else "proposed"
        legacy_ids = []
        for legacy in self._accepted_legacy_mirrors(revision.association_id):
            self._set_legacy_mirror_state(legacy, target, now=now)
            legacy_ids.append(legacy.id)
        digest = _semantic_hash(
            {
                "acknowledges_revision_id": revision.id,
                "retracted_legacy_association_ids": legacy_ids,
            }
        )
        existing = self.db.scalar(
            select(EconomicSocialAssociationRevision).where(
                EconomicSocialAssociationRevision.association_id
                == revision.association_id,
                EconomicSocialAssociationRevision.reconciliation_hash == digest,
            )
        )
        if existing is not None:
            return existing
        return self._create_revision(
            revision.association_id,
            state=revision.state,
            live=False,
            admission_state=revision.admission_state,
            mirror_state="acknowledged",
            reconciliation_hash=digest,
            decision_revision_id=revision.decision_revision_id,
            evidence_packet_id=revision.evidence_packet_id,
            projection_event_id=revision.projection_event_id,
            details={
                "acknowledges_revision_id": str(revision.id),
                "requested_state": revision.state,
                "retracted_legacy_association_ids": legacy_ids,
            },
            authority_epoch=authority_epoch,
        )

    def pin_revision(self, revision_id: UUID) -> SocialAssociationRevisionRef:
        revision = self.db.get(EconomicSocialAssociationRevision, revision_id)
        if revision is None:
            raise KeyError(f"economic social revision {revision_id} not found")
        existing = self.db.scalar(
            select(SocialAssociationRevisionRef).where(
                SocialAssociationRevisionRef.association_id == revision.association_id,
                SocialAssociationRevisionRef.revision_number
                == revision.revision_number,
            )
        )
        if existing is not None:
            return existing
        ref = SocialAssociationRevisionRef(
            association_id=revision.association_id,
            revision_number=revision.revision_number,
            decision_revision_id=revision.decision_revision_id,
        )
        self.db.add(ref)
        self.db.flush()
        return ref

    def membership(self, revision_ref_id: UUID) -> EconomicSocialMembership:
        ref = self.db.get(SocialAssociationRevisionRef, revision_ref_id)
        if ref is None:
            raise KeyError(f"social association revision ref {revision_ref_id} not found")
        revision = self.db.scalar(
            select(EconomicSocialAssociationRevision).where(
                EconomicSocialAssociationRevision.association_id
                == ref.association_id,
                EconomicSocialAssociationRevision.revision_number
                == ref.revision_number,
            )
        )
        association = self.db.get(EconomicSocialAssociation, ref.association_id)
        if revision is None or association is None:
            raise ValueError("economic_social_revision_ref_invalid")
        return EconomicSocialMembership(
            association_id=association.id,
            association_revision_id=revision.id,
            revision_number=revision.revision_number,
            economic_theme_id=association.economic_theme_id,
            security_id=association.security_id,
            state=revision.state,
            live=revision.live,
            admission_state=revision.admission_state,
            mirror_state=revision.mirror_state,
        )

    def admit_saved_work(
        self,
        work_id: int,
        evidence: EvidenceAdmission,
        *,
        policy_admitted: bool = True,
        exploratory: bool = False,
        association_id: UUID | None = None,
    ) -> SocialEvidenceAdmissionResult:
        work = self.db.get(SocialExtractionWork, work_id)
        if work is None:
            raise KeyError(f"social work {work_id} not found")
        evidence = replace(
            evidence,
            source_metadata={
                **dict(evidence.source_metadata),
                "social_work_id": work_id,
            },
        )
        published = self.db.scalar(
            select(SocialRunWork.work_id)
            .join(SocialSignalRun, SocialSignalRun.id == SocialRunWork.run_id)
            .where(
                SocialRunWork.work_id == work_id,
                SocialSignalRun.mode == "live",
                SocialSignalRun.status == "published",
            )
            .limit(1)
        )
        # Decided before admission: only work eligible to go live may supersede
        # a legacy X content packet (#500).
        eligible = bool(
            work.state == "succeeded"
            and published is not None
            and policy_admitted
            and not exploratory
        )
        admitted: AdmissionResult = EconomicSourceAdmissionService(
            self.db
        ).admit_social_work(evidence, supersede_content=eligible)
        live = eligible and admitted.precedence_state == "effective"
        if association_id is not None:
            source_key = f"social_work:{work_id}:packet:{admitted.packet_id}"
            existing = self.db.scalar(
                select(EconomicSocialAssociationSource).where(
                    EconomicSocialAssociationSource.association_id == association_id,
                    EconomicSocialAssociationSource.source_kind == "social_work",
                    EconomicSocialAssociationSource.source_key == source_key,
                )
            )
            if existing is None:
                self.db.add(
                    EconomicSocialAssociationSource(
                        association_id=association_id,
                        source_kind="social_work",
                        source_key=source_key,
                        social_work_id=work_id,
                        evidence_packet_id=admitted.packet_id,
                    )
                )
                self.db.flush()
        return SocialEvidenceAdmissionResult(
            source_family_id=admitted.source_family_id,
            source_lineage_id=admitted.source_lineage_id,
            packet_id=admitted.packet_id,
            effective_packet_id=admitted.effective_packet_id,
            evidence_revision_ordinal=admitted.evidence_revision_ordinal,
            precedence_state=admitted.precedence_state,
            admission_state="live" if live else "review_only",
            live=live,
        )

    def latest_associations(self):
        """Every association as (association, latest revision, security), oldest first."""
        latest = (
            select(
                EconomicSocialAssociationRevision.association_id,
                func.max(EconomicSocialAssociationRevision.revision_number).label("number"),
            )
            .group_by(EconomicSocialAssociationRevision.association_id)
            .subquery()
        )
        return self.db.execute(
            select(
                EconomicSocialAssociation,
                EconomicSocialAssociationRevision,
                StockUniverse,
            )
            .join(latest, latest.c.association_id == EconomicSocialAssociation.id)
            .join(
                EconomicSocialAssociationRevision,
                (EconomicSocialAssociationRevision.association_id == latest.c.association_id)
                & (EconomicSocialAssociationRevision.revision_number == latest.c.number),
            )
            .join(StockUniverse, StockUniverse.id == EconomicSocialAssociation.security_id)
            .order_by(EconomicSocialAssociation.created_at, EconomicSocialAssociation.id)
        ).all()

    def current_live_memberships(
        self, economic_theme_id: UUID
    ) -> tuple[EconomicSocialMembership, ...]:
        memberships = []
        associations = self.db.scalars(
            select(EconomicSocialAssociation).where(
                EconomicSocialAssociation.economic_theme_id == economic_theme_id
            )
        ).all()
        for association in associations:
            revision = self.db.scalar(
                select(EconomicSocialAssociationRevision)
                .where(
                    EconomicSocialAssociationRevision.association_id
                    == association.id
                )
                .order_by(EconomicSocialAssociationRevision.revision_number.desc())
                .limit(1)
            )
            if revision is not None and revision.live:
                memberships.append(
                    EconomicSocialMembership(
                        association_id=association.id,
                        association_revision_id=revision.id,
                        revision_number=revision.revision_number,
                        economic_theme_id=association.economic_theme_id,
                        security_id=association.security_id,
                        state=revision.state,
                        live=revision.live,
                        admission_state=revision.admission_state,
                        mirror_state=revision.mirror_state,
                    )
                )
        return tuple(sorted(memberships, key=lambda item: item.security_id))

    def _create_decision(
        self,
        association_id: UUID,
        *,
        state: str,
        idempotency_key: str,
        actor: str,
        reason: str,
        source_payload: dict,
    ) -> EconomicSocialDecisionRevision:
        current = self.db.scalar(
            select(func.max(EconomicSocialDecisionRevision.revision_number)).where(
                EconomicSocialDecisionRevision.association_id == association_id
            )
        )
        decision = EconomicSocialDecisionRevision(
            association_id=association_id,
            revision_number=int(current or 0) + 1,
            state=state,
            idempotency_key=idempotency_key,
            actor=actor,
            reason=reason,
            source_payload=source_payload,
        )
        self.db.add(decision)
        self.db.flush()
        return decision

    def _create_revision(
        self,
        association_id: UUID,
        *,
        state: str,
        live: bool,
        admission_state: str,
        mirror_state: str,
        reconciliation_hash: str,
        decision_revision_id: UUID | None,
        details: dict,
        authority_epoch: int,
        evidence_packet_id: UUID | None = None,
        projection_event_id: UUID | None = None,
    ) -> EconomicSocialAssociationRevision:
        current = self.db.scalar(
            select(func.max(EconomicSocialAssociationRevision.revision_number)).where(
                EconomicSocialAssociationRevision.association_id == association_id
            )
        )
        revision = EconomicSocialAssociationRevision(
            association_id=association_id,
            revision_number=int(current or 0) + 1,
            decision_revision_id=decision_revision_id,
            state=state,
            live=live,
            admission_state=admission_state,
            mirror_state=mirror_state,
            evidence_packet_id=evidence_packet_id,
            projection_event_id=projection_event_id,
            reconciliation_hash=reconciliation_hash,
            details=details,
        )
        self.db.add(revision)
        self.db.flush()
        EconomicTaxonomyPublicationRepository(self.db).append_source_revision(
            producer_kind="economic_social",
            logical_source_key=f"economic_social_association:{association_id}",
            revision_kind=(
                "social_conflict"
                if state == "conflict_review_required"
                else "association_revision"
            ),
            revision_number=revision.revision_number,
            content_hash=reconciliation_hash,
            authority_epoch=authority_epoch,
        )
        return revision

    def _current_epoch(self) -> int:
        authority = self.db.get(TaxonomyAuthority, 1)
        return authority.authority_epoch if authority is not None else 1

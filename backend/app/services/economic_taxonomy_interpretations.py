"""Build immutable, manifest-pinned accepted interpretation sets."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.economic_taxonomy.contracts import (
    AdminPrincipal,
    GenerationInputSelection,
    InterpretationCandidate,
)
from app.domain.economic_taxonomy.contracts import (
    SocialAssociationRevisionRef as SocialRevisionContract,
)
from app.domain.economic_taxonomy.policy import choose_interpretation
from app.models.economic_taxonomy_runtime import (
    ClaimAssignment,
    ClassificationAttempt,
    ClassificationAttemptEvent,
    ConstituentDecisionRevision,
    DevelopmentSelectionRevision,
    EvidencePacket,
    EvidencePrecedenceRevision,
    GenerationInputManifest,
    InterpretationOverrideRevision,
    InterpretationSelection,
    InterpretationSet,
    LensEligibilityRevision,
    ProcessingRequest,
    SocialAssociationRevisionRef,
    SourceLineage,
)
from app.services.economic_theme_observation_service import materialize_assignment_facts
from app.utils.file_hashing import canonical_json_sha256 as _hash


class InvalidInterpretation(ValueError):
    pass


def manifest_social_revision_contracts(raw_entries) -> tuple[tuple[UUID, int], ...]:
    """Return every immutable Social association revision pinned by a manifest."""

    contracts = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest_selection_must_be_object")
        values = entry.get("social_association_revisions")
        if values is None:
            single = entry.get("social_association_revision")
            values = [] if single is None else [single]
        if not isinstance(values, list):
            raise ValueError("invalid_social_association_revisions")
        for value in values:
            try:
                association_id = UUID(str(value["association_id"]))
                revision_number = int(value["revision_number"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid_social_association_revision") from exc
            if revision_number <= 0:
                raise ValueError("invalid_social_association_revision")
            contracts.add((association_id, revision_number))
    return tuple(sorted(contracts, key=lambda value: (str(value[0]), value[1])))


def create_interpretation_override(
    session: Session,
    *,
    source_lineage_id: UUID,
    selected_attempt_id: UUID,
    reason: str,
    principal: AdminPrincipal,
) -> InterpretationOverrideRevision:
    if not principal.can_review_taxonomy:
        raise PermissionError("override requires an authenticated taxonomy reviewer")
    if not reason or not reason.strip():
        raise ValueError("override reason must be non-empty")
    session.execute(
        select(SourceLineage.id)
        .where(SourceLineage.id == source_lineage_id)
        .with_for_update()
    ).scalar_one()
    current = session.scalar(
        select(func.max(InterpretationOverrideRevision.revision_number)).where(
            InterpretationOverrideRevision.source_lineage_id == source_lineage_id
        )
    )
    row = InterpretationOverrideRevision(
        source_lineage_id=source_lineage_id,
        revision_number=int(current or 0) + 1,
        override_kind="select_attempt",
        payload={
            "selected_attempt_id": str(selected_attempt_id),
            "authenticated": True,
            "auth_method": principal.auth_method,
            "roles": sorted(principal.roles),
        },
        reason=reason,
        created_by=principal.subject,
    )
    session.add(row)
    session.flush()
    return row


class EconomicTaxonomyInterpretationService:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def build_interpretation_set(
        self, manifest_id: UUID, *, actor: str = "system:economic-taxonomy-refresh"
    ) -> InterpretationSet:
        with self.session_factory() as session:
            manifest = session.execute(
                select(GenerationInputManifest)
                .where(GenerationInputManifest.id == manifest_id)
                .with_for_update()
            ).scalar_one_or_none()
            if manifest is None:
                raise KeyError(f"generation input manifest {manifest_id} not found")
            if manifest.status != "sealed":
                raise InvalidInterpretation("manifest_not_sealed")
            existing = session.execute(
                select(InterpretationSet).where(
                    InterpretationSet.generation_input_manifest_id == manifest.id
                )
            ).scalar_one_or_none()
            if existing is not None:
                session.expunge(existing)
                return existing

            raw_entries = list(manifest.selections or [])
            parsed_entries = [self._parse_entry(entry) for entry in raw_entries]
            social_refs = self._manifest_social_refs(session, raw_entries)
            try:
                lineage_ids = [UUID(entry.lineage) for entry in parsed_entries]
            except (TypeError, ValueError) as exc:
                raise InvalidInterpretation("invalid_manifest_identity") from exc
            if len(lineage_ids) != len(set(lineage_ids)):
                raise InvalidInterpretation("duplicate_lineage_selection")

            chosen = []
            for raw_entry, entry in zip(raw_entries, parsed_entries, strict=True):
                selection = self._resolve_manifest_entry(
                    session,
                    manifest=manifest,
                    raw_entry=raw_entry,
                    entry=entry,
                )
                if selection is not None:
                    chosen.append(selection)

            interpretation = InterpretationSet(
                status="unsealed",
                generation_input_manifest_id=manifest.id,
                created_by=actor,
            )
            session.add(interpretation)
            session.flush()
            semantic_rows = []
            for resolved in chosen:
                row = InterpretationSelection(
                    interpretation_set_id=interpretation.id,
                    source_lineage_id=resolved["lineage_id"],
                    evidence_packet_id=resolved["packet"].id,
                    selected_classification_attempt_id=resolved["attempt"].id,
                    pinned_evidence_revision_ordinal=resolved[
                        "packet"
                    ].evidence_revision_ordinal,
                    interpretation_override_revision_id=(
                        resolved["override"].id if resolved["override"] else None
                    ),
                    social_association_revision_ref_id=(
                        resolved["social_ref"].id if resolved["social_ref"] else None
                    ),
                )
                session.add(row)
                semantic_rows.append(
                    {
                        "lineage_id": str(resolved["lineage_id"]),
                        "packet_id": str(resolved["packet"].id),
                        "ordinal": resolved["packet"].evidence_revision_ordinal,
                        "attempt_id": str(resolved["attempt"].id),
                        "override_id": (
                            str(resolved["override"].id)
                            if resolved["override"]
                            else None
                        ),
                        "social_ref_id": (
                            str(resolved["social_ref"].id)
                            if resolved["social_ref"]
                            else None
                        ),
                    }
                )
            semantic_rows.sort(key=lambda row: row["lineage_id"])
            semantic_payload = {
                "selections": semantic_rows,
                "social_ref_ids": [str(ref.id) for ref in social_refs],
            }
            session.flush()
            interpretation.seal(
                semantic_hash=_hash(semantic_payload),
                artifact_integrity_hash=_hash(
                    {
                        "interpretation_set_id": str(interpretation.id),
                        "manifest_id": str(manifest.id),
                        "created_by": actor,
                        **semantic_payload,
                    }
                ),
            )
            for resolved in chosen:
                materialize_assignment_facts(
                    session,
                    classification_attempt_id=resolved["attempt"].id,
                    evidence_channels=resolved["eligibility"].evidence_channels,
                    detector_policy_version="signals-v1",
                )
            session.commit()
            session.refresh(interpretation)
            session.expunge(interpretation)
            return interpretation

    def packet_for_attempt(self, session, attempt) -> EvidencePacket:
        """The evidence packet an attempt classified."""
        return self._packet_for_attempt(session, attempt)

    def default_attempt(self, session, source_lineage_id: UUID) -> ClassificationAttempt | None:
        """``choose_default`` in the caller's session (no session factory needed)."""
        return self._default_attempt(session, source_lineage_id)

    def serving_attempt(self, session, source_lineage_id: UUID) -> ClassificationAttempt | None:
        """The attempt users see for the lineage (#513).

        A reviewer override carried by the serving generation's manifest wins,
        validated as publication validates it; otherwise the policy default.
        """
        from app.models.economic_taxonomy_runtime import ServingGeneration, TaxonomyAuthority

        authority = session.get(TaxonomyAuthority, 1)
        generation = (
            session.get(ServingGeneration, authority.serving_generation_id)
            if authority is not None and authority.serving_generation_id is not None
            else None
        )
        manifest = (
            session.get(GenerationInputManifest, generation.generation_input_manifest_id)
            if generation is not None
            else None
        )
        for entry in (manifest.selections or []) if manifest is not None else []:
            if str(entry.get("lineage")) != str(source_lineage_id):
                continue
            override_id = entry.get("interpretation_override_revision_id")
            if not override_id:
                break
            try:
                selected = session.get(ClassificationAttempt, UUID(str(entry["selected_attempt_id"])))
                if selected is not None and self._attempt_completed(session, selected):
                    self._validated_override(
                        session,
                        UUID(str(override_id)),
                        lineage_id=source_lineage_id,
                        selected_attempt_id=selected.id,
                    )
                    return selected
            except (KeyError, TypeError, ValueError):
                pass  # InvalidInterpretation is a ValueError: fall back to the default
            break
        return self._default_attempt(session, source_lineage_id)

    def choose_default(self, source_lineage_id: UUID) -> ClassificationAttempt | None:
        with self.session_factory() as session:
            chosen = self._default_attempt(session, source_lineage_id)
            if chosen is not None:
                session.expunge(chosen)
            return chosen

    def _resolve_manifest_entry(self, session, *, manifest, raw_entry, entry):
        try:
            lineage_id = UUID(entry.lineage)
            packet_id = UUID(entry.evidence_packet_id)
        except (TypeError, ValueError) as exc:
            raise InvalidInterpretation("invalid_manifest_identity") from exc
        packet = session.get(EvidencePacket, packet_id)
        if packet is None or packet.source_lineage_id != lineage_id:
            raise InvalidInterpretation("manifest_packet_lineage_mismatch")
        precedence = session.execute(
            select(EvidencePrecedenceRevision).where(
                EvidencePrecedenceRevision.source_lineage_id == lineage_id,
                EvidencePrecedenceRevision.revision_number
                == entry.evidence_precedence_revision,
            )
        ).scalar_one_or_none()
        if precedence is None:
            raise InvalidInterpretation("precedence_revision_not_pinned")
        eligibility = session.execute(
            select(LensEligibilityRevision).where(
                LensEligibilityRevision.source_lineage_id == lineage_id,
                LensEligibilityRevision.evidence_packet_id == packet_id,
                LensEligibilityRevision.revision_number == entry.eligibility_revision,
            )
        ).scalar_one_or_none()
        if eligibility is None:
            raise InvalidInterpretation("eligibility_revision_not_pinned")

        default = self._default_attempt(
            session,
            lineage_id,
            precedence_cutoff=entry.evidence_precedence_revision,
            attempt_created_cutoff=manifest.created_at,
        )
        try:
            selected_id = (
                UUID(entry.selected_attempt_id) if entry.selected_attempt_id else None
            )
        except (TypeError, ValueError) as exc:
            raise InvalidInterpretation("invalid_attempt_identity") from exc
        if selected_id is None:
            if default is not None:
                raise InvalidInterpretation("manifest_attempt_missing")
            return None
        selected = session.get(ClassificationAttempt, selected_id)
        if selected is None:
            raise InvalidInterpretation("attempt_not_found")
        selected_packet = self._packet_for_attempt(session, selected)
        if not self._attempt_completed(session, selected):
            raise InvalidInterpretation("attempt_not_completed")
        selected_precedence = self._packet_precedence_state(
            session,
            selected_packet.id,
            cutoff=entry.evidence_precedence_revision,
        )
        if selected_precedence not in {"effective", "equivalent"}:
            raise InvalidInterpretation("attempt_evidence_not_effective")
        if selected_packet.id != packet.id:
            raise InvalidInterpretation("manifest_attempt_packet_mismatch")

        override = None
        if entry.interpretation_override_revision_id:
            override = self._validated_override(
                session,
                UUID(entry.interpretation_override_revision_id),
                lineage_id=lineage_id,
                selected_attempt_id=selected.id,
            )
        elif default is None or default.id != selected.id:
            raise InvalidInterpretation("manifest_attempt_not_default")

        social_ref = self._social_ref(session, raw_entry)
        self._validate_auxiliary_revisions(
            session,
            raw_entry=raw_entry,
            entry=entry,
            social_ref=social_ref,
        )
        return {
            "lineage_id": lineage_id,
            "packet": packet,
            "attempt": selected,
            "override": override,
            "social_ref": social_ref,
            "eligibility": eligibility,
        }

    def _default_attempt(
        self,
        session,
        lineage_id,
        *,
        precedence_cutoff: int | None = None,
        attempt_created_cutoff=None,
    ):
        query = (
            select(ClassificationAttempt, EvidencePacket)
            .join(
                ProcessingRequest,
                ProcessingRequest.id == ClassificationAttempt.processing_request_id,
            )
            .join(
                EvidencePacket,
                EvidencePacket.id == ProcessingRequest.evidence_packet_id,
            )
            .where(ProcessingRequest.source_lineage_id == lineage_id)
        )
        if attempt_created_cutoff is not None:
            query = query.where(
                ClassificationAttempt.created_at <= attempt_created_cutoff
            )
        rows = session.execute(query).all()
        by_id = {}
        candidates = []
        for attempt, packet in rows:
            status = (
                "completed" if self._attempt_completed(session, attempt) else "failed"
            )
            try:
                provider_order = (
                    int(packet.provider_revision_order)
                    if packet.provider_revision_order is not None
                    else None
                )
            except (TypeError, ValueError):
                provider_order = None
            candidate = InterpretationCandidate(
                attempt_id=str(attempt.id),
                status=status,
                evidence_revision_ordinal=packet.evidence_revision_ordinal,
                provider_revision_order=provider_order,
                assignments=tuple(
                    str(value)
                    for value in session.scalars(
                        select(ClaimAssignment.economic_theme_id).where(
                            ClaimAssignment.classification_attempt_id == attempt.id
                        )
                    )
                ),
                precedence_state=(
                    self._packet_precedence_state(
                        session,
                        packet.id,
                        cutoff=precedence_cutoff,
                    )
                    or "superseded"
                ),
            )
            by_id[candidate.attempt_id] = attempt
            candidates.append(candidate)
        chosen = choose_interpretation(previous=None, candidates=candidates)
        return by_id[chosen.attempt_id] if chosen is not None else None

    @staticmethod
    def _packet_precedence_state(session, packet_id, *, cutoff=None):
        query = (
            select(EvidencePrecedenceRevision.disposition)
            .where(EvidencePrecedenceRevision.evidence_packet_id == packet_id)
            .order_by(EvidencePrecedenceRevision.revision_number.desc())
            .limit(1)
        )
        if cutoff is not None:
            query = query.where(EvidencePrecedenceRevision.revision_number <= cutoff)
        return session.scalar(query)

    @staticmethod
    def _attempt_completed(session, attempt):
        if attempt.result_status != "completed":
            return False
        return (
            session.scalar(
                select(ClassificationAttemptEvent.id).where(
                    ClassificationAttemptEvent.classification_attempt_id == attempt.id,
                    ClassificationAttemptEvent.event_type == "completed",
                )
            )
            is not None
        )

    @staticmethod
    def _packet_for_attempt(session, attempt):
        return session.execute(
            select(EvidencePacket)
            .join(
                ProcessingRequest,
                ProcessingRequest.evidence_packet_id == EvidencePacket.id,
            )
            .where(ProcessingRequest.id == attempt.processing_request_id)
        ).scalar_one()

    @staticmethod
    def _validated_override(session, override_id, *, lineage_id, selected_attempt_id):
        override = session.get(InterpretationOverrideRevision, override_id)
        if override is None or override.source_lineage_id != lineage_id:
            raise InvalidInterpretation("override_not_pinned")
        payload = override.payload or {}
        if (
            override.override_kind != "select_attempt"
            or payload.get("authenticated") is not True
            or "taxonomy:review" not in set(payload.get("roles") or [])
            or payload.get("selected_attempt_id") != str(selected_attempt_id)
        ):
            raise InvalidInterpretation("override_not_authenticated")
        return override

    @staticmethod
    def _social_ref(session, raw_entry):
        raw = raw_entry.get("social_association_revision")
        if raw is None:
            return None
        association_id = UUID(str(raw["association_id"]))
        revision_number = int(raw["revision_number"])
        row = session.execute(
            select(SocialAssociationRevisionRef).where(
                SocialAssociationRevisionRef.association_id == association_id,
                SocialAssociationRevisionRef.revision_number == revision_number,
            )
        ).scalar_one_or_none()
        if row is None:
            raise InvalidInterpretation("social_revision_not_pinned")
        return row

    @staticmethod
    def _manifest_social_refs(session, raw_entries):
        try:
            contracts = manifest_social_revision_contracts(raw_entries)
        except ValueError as exc:
            raise InvalidInterpretation(str(exc)) from exc
        refs = []
        for association_id, revision_number in contracts:
            row = session.scalar(
                select(SocialAssociationRevisionRef).where(
                    SocialAssociationRevisionRef.association_id == association_id,
                    SocialAssociationRevisionRef.revision_number == revision_number,
                )
            )
            if row is None:
                raise InvalidInterpretation("social_revision_not_pinned")
            refs.append(row)
        return tuple(refs)

    @staticmethod
    def _validate_auxiliary_revisions(
        session,
        *,
        raw_entry,
        entry,
        social_ref,
    ):
        for revision in {
            entry.constituent_decision_revision,
            entry.social_decision_revision,
        } - {None}:
            if social_ref is None:
                raise InvalidInterpretation("social_decision_without_association")
            exists = session.scalar(
                select(ConstituentDecisionRevision.id).where(
                    ConstituentDecisionRevision.association_revision_ref_id
                    == social_ref.id,
                    ConstituentDecisionRevision.revision_number == revision,
                )
            )
            if exists is None:
                raise InvalidInterpretation("constituent_decision_not_pinned")

        development_selections = raw_entry.get("development_selections")
        if development_selections is None:
            development_selections = (
                [
                    {
                        "development_identity": raw_entry.get(
                            "development_identity"
                        ),
                        "revision_number": entry.development_revision,
                    }
                ]
                if entry.development_revision is not None
                else []
            )
        if not isinstance(development_selections, list):
            raise InvalidInterpretation("invalid_development_selections")
        seen = set()
        for development in development_selections:
            if not isinstance(development, dict):
                raise InvalidInterpretation("invalid_development_selection")
            try:
                identity = UUID(str(development["development_identity"]))
                revision_number = int(development["revision_number"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidInterpretation("invalid_development_selection") from exc
            if identity in seen or revision_number <= 0:
                raise InvalidInterpretation("invalid_development_selection")
            seen.add(identity)
            row = session.scalar(
                select(DevelopmentSelectionRevision.id).where(
                    DevelopmentSelectionRevision.development_identity == identity,
                    DevelopmentSelectionRevision.revision_number == revision_number,
                )
            )
            if row is None:
                raise InvalidInterpretation("development_revision_not_pinned")

    @staticmethod
    def _parse_entry(raw_entry):
        if not isinstance(raw_entry, dict):
            raise InvalidInterpretation("manifest_selection_must_be_object")
        try:
            social = raw_entry.get("social_association_revision")
            social_contract = (
                SocialRevisionContract(
                    association_id=str(social["association_id"]),
                    revision_number=int(social["revision_number"]),
                )
                if social is not None
                else None
            )

            def optional_revision(key):
                return int(raw_entry[key]) if raw_entry.get(key) is not None else None

            return GenerationInputSelection(
                lineage=str(raw_entry["lineage"]),
                evidence_packet_id=str(raw_entry["evidence_packet_id"]),
                selected_attempt_id=(
                    str(raw_entry["selected_attempt_id"])
                    if raw_entry.get("selected_attempt_id") is not None
                    else None
                ),
                eligibility_revision=int(raw_entry["eligibility_revision"]),
                evidence_precedence_revision=int(
                    raw_entry["evidence_precedence_revision"]
                ),
                interpretation_override_revision_id=(
                    str(raw_entry["interpretation_override_revision_id"])
                    if raw_entry.get("interpretation_override_revision_id")
                    else None
                ),
                constituent_decision_revision=optional_revision(
                    "constituent_decision_revision"
                ),
                social_association_revision=social_contract,
                social_decision_revision=optional_revision("social_decision_revision"),
                development_revision=optional_revision("development_revision"),
                mapping_revision=int(raw_entry["mapping_revision"]),
                metrics_policy_revision=int(raw_entry["metrics_policy_revision"]),
                compatibility_projection_revision=int(
                    raw_entry["compatibility_projection_revision"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidInterpretation("invalid_manifest_selection") from exc


def build_interpretation_set(
    session_factory,
    manifest_id: UUID,
    *,
    actor: str = "system:economic-taxonomy-refresh",
) -> InterpretationSet:
    return EconomicTaxonomyInterpretationService(
        session_factory
    ).build_interpretation_set(manifest_id, actor=actor)

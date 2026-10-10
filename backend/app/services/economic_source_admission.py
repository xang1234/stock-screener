"""Admit immutable, route-aware evidence for Economic Theme processing."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.economic_taxonomy.contracts import (
    EvidenceChannel,
    EvidencePacketDescriptor,
    EvidencePrecedenceDecision,
    SourceLineageKey,
)
from app.domain.economic_taxonomy.policy import decide_evidence_precedence
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
)
from app.models.economic_taxonomy_runtime import (
    EvidencePacket,
    EvidencePrecedenceRevision,
    LensEligibilityRevision,
    SourceFamily,
    SourceLineage,
    TaxonomyAuthority,
)
from app.services.economic_taxonomy_fence import producer_write
from app.services.twitter_content_identity import x_post_id_from_url
from app.utils.file_hashing import canonical_json_sha256 as _hash


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ordered(values) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values}))


# Capture route for news/RSS/Substack/Reddit/legacy-X content ingestion (#471).
CONTENT_INGESTION_ROUTE = "content_ingestion"


def post_family_key(provider: str, canonical_item_id: str) -> str:
    return f"{provider.strip().lower()}:post:{canonical_item_id}"


def content_family_key(source_type: str, external_id: str | None, url: str | None) -> str | None:
    """Source family for an ingested content item, or None if it has no identity.

    X posts join Social's ``x:post:<tweet_id>`` family (#500) and need a status
    URL; other items need an external id, or unrelated rows would share one.
    """
    if source_type == "twitter":
        post_id = x_post_id_from_url(url)
        return post_family_key("x", post_id) if post_id else None
    if not (external_id or "").strip():
        return None
    return post_family_key(source_type, external_id)


def content_route_record_id(content_item_id: int, source_id: int | None) -> str:
    """One route record per (item, source) observation; feeds can share items."""
    return f"{content_item_id}:{source_id}"


@dataclass(frozen=True, slots=True)
class EvidenceAdmission:
    provider: str
    capture_route: str
    original_text: str
    preparation_version: str
    canonical_item_id: str | None = None
    canonical_source_family: str | None = None
    route_record_id: str | None = None
    translated_text: str | None = None
    translation_version: str | None = None
    attachment_hashes: tuple[str, ...] = ()
    extracted_text_hashes: tuple[str, ...] = ()
    grounding_snapshot: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_revision_id: str | None = None
    provider_revision_order: int | None = None
    captured_at: datetime = field(default_factory=_utcnow)
    observed_at: datetime | None = None
    available_at: datetime = field(default_factory=_utcnow)
    evidence_channels: tuple[str, ...] = ()
    supersedes_packet_id: UUID | None = None
    equivalent_packet_id: UUID | None = None
    scope_suffix: str | None = None
    admission_policy_key: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.provider, "provider"),
            (self.capture_route, "capture_route"),
            (self.preparation_version, "preparation_version"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.canonical_source_family is None and not self.canonical_item_id:
            raise ValueError("canonical_item_id or canonical_source_family is required")
        SourceLineageKey(
            canonical_source_family=self.family_key,
            scope_suffix=self.scope_suffix,
            admission_policy_key=self.admission_policy_key,
        )
        allowed = {channel.value for channel in EvidenceChannel}
        unknown = set(self.evidence_channels) - allowed
        if unknown:
            raise ValueError(f"unknown evidence channels: {sorted(unknown)}")

    @property
    def family_key(self) -> str:
        if self.canonical_source_family:
            return self.canonical_source_family.strip()
        return post_family_key(self.provider, self.canonical_item_id)


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    source_family_id: UUID
    source_lineage_id: UUID
    packet_id: UUID
    effective_packet_id: UUID | None
    evidence_revision_ordinal: int
    precedence_state: str
    packet_hash: str
    evidence_content_fingerprint: str


@dataclass(frozen=True, slots=True)
class LensEligibilityResult:
    packet_id: UUID
    revision_number: int
    evidence_channels: tuple[str, ...]
    enqueued_request_id: None = None


class EconomicSourceAdmissionService:
    """Resolve source identity and preserve exact evidence packet history."""

    def __init__(self, session: Session):
        self.session = session

    def admit_content(self, evidence: EvidenceAdmission) -> AdmissionResult:
        # Ingestion re-polls items with a fresh capture time, so a recapture of
        # text already admitted for this lineage reuses that packet. A source's
        # grant applies to the post even when this capture is held (behind
        # Social, or behind another feed's differing text), so it lends its
        # channels to whichever packet is in force (#500).
        return self.admit(
            evidence,
            reuse_admitted_content=True,
            lend_channels=True,
        )

    def admit_social_work(
        self, evidence: EvidenceAdmission, *, supersede_content: bool = True
    ) -> AdmissionResult:
        # Social's prepared capture of an X post outranks the raw legacy X
        # content capture of the same post, whose fingerprint it can never
        # match (#500), but only for work eligible to go live: review-only work
        # must not displace admitted content. No fingerprint reuse here: Social
        # recaptures with new metadata must record equivalent packets.
        return self.admit(
            evidence,
            supersede_routes=(
                frozenset({CONTENT_INGESTION_ROUTE}) if supersede_content else frozenset()
            ),
        )

    def admit(
        self,
        evidence: EvidenceAdmission,
        *,
        reuse_admitted_content: bool = False,
        supersede_routes: frozenset[str] = frozenset(),
        lend_channels: bool = False,
    ) -> AdmissionResult:
        expected_epoch = self._current_epoch()
        with producer_write(
            self.session,
            expected_epoch=expected_epoch,
            allowed_modes={"legacy", "shadow", "dual", "economic"},
        ) as authority:
            result = self._admit(
                evidence,
                authority_epoch=authority.authority_epoch,
                reuse_admitted_content=reuse_admitted_content,
                supersede_routes=supersede_routes,
            )
            # The lens discovery reads is the effective packet's; merging is a
            # no-op when the capture is effective or already merged.
            if lend_channels and evidence.evidence_channels:
                effective = self.effective_packet(result.source_lineage_id)
                if effective is not None:
                    self._merge_equivalent_eligibility(
                        effective,
                        evidence.evidence_channels,
                        authority_epoch=authority.authority_epoch,
                    )
            return result

    def _admit(
        self,
        evidence: EvidenceAdmission,
        *,
        authority_epoch: int,
        reuse_admitted_content: bool = False,
        supersede_routes: frozenset[str] = frozenset(),
    ) -> AdmissionResult:
        family = self._get_or_create_family(evidence)
        lineage = self._get_or_create_lineage(family, evidence)
        self.session.execute(
            select(SourceLineage.id)
            .where(SourceLineage.id == lineage.id)
            .with_for_update()
        ).scalar_one()
        # An explicit supersession of an effective packet from an outranked
        # route, recorded on the packet so precedence advances on stated
        # provenance rather than admission order. Applies while that packet is
        # in force, and to an identical re-run of the superseding admission
        # (whose hash includes the link), so a retry stays a no-op. Archive and
        # partial captures never supersede; they stay review-only.
        if (
            supersede_routes
            and evidence.supersedes_packet_id is None
            and evidence.equivalent_packet_id is None
            and not self._is_archive_or_partial(evidence)
        ):
            target = self._latest_effective_from_routes(lineage.id, supersede_routes)
            if target is not None:
                # Carry the superseded capture's lens (e.g. a legacy X source's
                # technical/fundamental grants); lens is outside the packet hash.
                superseding = replace(
                    evidence,
                    supersedes_packet_id=target.id,
                    evidence_channels=tuple(
                        sorted(set(evidence.evidence_channels) | self._latest_channels(target.id))
                    ),
                )
                current = self.effective_packet(lineage.id)
                rerun_hash = self._packet_hash(
                    superseding, self._content_fingerprint(superseding)
                )
                if (current is not None and current.id == target.id) or self.session.scalar(
                    select(EvidencePacket.id).where(
                        EvidencePacket.source_lineage_id == lineage.id,
                        EvidencePacket.packet_hash == rerun_hash,
                    )
                ) is not None:
                    evidence = superseding

        fingerprint = self._content_fingerprint(evidence)
        packet_hash = self._packet_hash(evidence, fingerprint)
        existing = self.session.execute(
            select(EvidencePacket).where(
                EvidencePacket.source_lineage_id == lineage.id,
                EvidencePacket.packet_hash == packet_hash,
            )
        ).scalar_one_or_none()
        # Only revisionless recaptures reuse by fingerprint: an ordered or
        # explicitly linked capture (e.g. a reversion A -> B -> A) must reach
        # the precedence policy, and so must any capture while the lineage has
        # no effective packet (e.g. only a held late archive). Reuse is limited
        # to the same capture route and record, so another route or source
        # records its own (equivalent) packet and keeps its provenance.
        effective_now = (
            self.effective_packet(lineage.id)
            if existing is None
            and reuse_admitted_content
            and evidence.provider_revision_id is None
            and evidence.provider_revision_order is None
            and evidence.supersedes_packet_id is None
            and evidence.equivalent_packet_id is None
            else None
        )
        if effective_now is not None:
            same_capture = [
                packet
                for packet in self.session.execute(
                    select(EvidencePacket)
                    .where(
                        EvidencePacket.source_lineage_id == lineage.id,
                        EvidencePacket.evidence_content_fingerprint == fingerprint,
                        EvidencePacket.capture_route == evidence.capture_route,
                    )
                    .order_by(EvidencePacket.evidence_revision_ordinal)
                ).scalars()
                if (packet.source_metadata or {}).get("route_record_id")
                == evidence.route_record_id
            ]
            # A displaced packet (A -> B) is a possible reversion: precedence.
            ranked = [
                (rank, packet)
                for packet in same_capture
                if (rank := self._standing(packet, effective_now)) is not None
            ]
            existing = min(ranked, key=lambda pair: pair[0], default=(None, None))[1]
        if existing is not None:
            effective = self.effective_packet(lineage.id)
            # The stored state is the one at admission; a later correction may
            # have displaced this packet since (#556).
            state = self._current_state(existing, effective)
            if state in {"effective", "equivalent"}:
                self._merge_equivalent_eligibility(
                    effective,
                    evidence.evidence_channels,
                    authority_epoch=authority_epoch,
                )
            return self._result(
                family, lineage, existing, effective, precedence_state=state
            )

        accepted = self.effective_packet(lineage.id)
        packet_id = uuid4()
        candidate = EvidencePacketDescriptor(
            packet_id=str(packet_id),
            evidence_content_fingerprint=fingerprint,
            provider_revision_id=evidence.provider_revision_id,
            provider_revision_order=evidence.provider_revision_order,
            supersedes_packet_id=(
                str(evidence.supersedes_packet_id)
                if evidence.supersedes_packet_id is not None
                else None
            ),
            equivalent_to_packet_id=(
                str(evidence.equivalent_packet_id)
                if evidence.equivalent_packet_id is not None
                else None
            ),
            captured_from=evidence.capture_route,
            attachment_hashes=frozenset(evidence.attachment_hashes),
        )
        if accepted is None and self._is_unordered_archive(evidence):
            decision = EvidencePrecedenceDecision.HOLD_REVIEW
        else:
            decision = decide_evidence_precedence(
                self._descriptor(accepted) if accepted is not None else None,
                candidate,
            )
        state = {
            EvidencePrecedenceDecision.ADVANCE: "effective",
            EvidencePrecedenceDecision.REUSE_EQUIVALENT: "equivalent",
            EvidencePrecedenceDecision.IGNORE_SUPERSEDED: "superseded",
            EvidencePrecedenceDecision.HOLD_REVIEW: "hold_review",
        }[decision]
        equivalent_id = evidence.equivalent_packet_id
        if decision is EvidencePrecedenceDecision.REUSE_EQUIVALENT and accepted:
            equivalent_id = accepted.id

        packet = EvidencePacket(
            id=packet_id,
            source_lineage_id=lineage.id,
            packet_hash=packet_hash,
            evidence_content_fingerprint=fingerprint,
            provider_revision_id=evidence.provider_revision_id,
            provider_revision_order=(
                str(evidence.provider_revision_order)
                if evidence.provider_revision_order is not None
                else None
            ),
            capture_route=evidence.capture_route,
            captured_at=evidence.captured_at,
            supersedes_evidence_packet_id=evidence.supersedes_packet_id,
            equivalent_evidence_packet_id=equivalent_id,
            precedence_state=state,
            original_text_ref=evidence.original_text,
            translated_text_ref=evidence.translated_text,
            translation_version=evidence.translation_version,
            attachment_hashes=list(_ordered(evidence.attachment_hashes)),
            extracted_text_hashes=list(_ordered(evidence.extracted_text_hashes)),
            grounding_snapshot=dict(evidence.grounding_snapshot),
            preparation_version=evidence.preparation_version,
            source_metadata={
                **dict(evidence.source_metadata),
                "capture_route": evidence.capture_route,
                "route_record_id": evidence.route_record_id,
            },
            observed_at=evidence.observed_at,
            available_at=evidence.available_at,
        )
        self.session.add(packet)
        self.session.flush()
        precedence_revision = self._next_precedence_revision(lineage.id)
        self.session.add(
            EvidencePrecedenceRevision(
                source_lineage_id=lineage.id,
                evidence_packet_id=packet.id,
                revision_number=precedence_revision,
                disposition=state,
                related_evidence_packet_id=(accepted.id if accepted else None),
                reason=decision.value,
            )
        )
        self.session.add(
            LensEligibilityRevision(
                source_lineage_id=lineage.id,
                evidence_packet_id=packet.id,
                revision_number=1,
                evidence_channels=list(_ordered(evidence.evidence_channels)),
                reason="admission",
            )
        )
        self.session.flush()
        effective = packet if state == "effective" else accepted
        if state == "equivalent" and effective is not None:
            self._merge_equivalent_eligibility(
                effective,
                evidence.evidence_channels,
                authority_epoch=authority_epoch,
            )
        return self._result(family, lineage, packet, effective)

    def _merge_equivalent_eligibility(
        self,
        effective: EvidencePacket,
        evidence_channels: tuple[str, ...],
        *,
        authority_epoch: int,
    ) -> None:
        latest = self.session.execute(
            select(LensEligibilityRevision)
            .where(LensEligibilityRevision.evidence_packet_id == effective.id)
            .order_by(LensEligibilityRevision.revision_number.desc())
            .limit(1)
        ).scalar_one_or_none()
        merged = tuple(
            sorted(set(latest.evidence_channels if latest else ()) | set(evidence_channels))
        )
        if latest is not None and tuple(sorted(latest.evidence_channels)) == merged:
            return
        self._revise_lens_eligibility(
            effective.id,
            add=None,
            remove=None,
            evidence_channels=merged,
            reason="equivalent_evidence_admission",
            authority_epoch=authority_epoch,
        )

    def add_observation_channels(
        self,
        *,
        family_key: str,
        capture_route: str,
        route_record_id: str,
        channels: Collection[str],
        reason: str,
    ) -> bool:
        """Add lens channels for one capture observation already admitted.

        A source's grant applies to the post, whichever packet is now in force
        (e.g. Social's, after it superseded a legacy X capture), so channels go
        to the lineage's effective packet once the observation has any packet.
        Returns whether a revision was written; an observation not yet admitted
        is left to the backfill, which replays the grants.
        """
        lineage = self.session.execute(
            select(SourceLineage)
            .join(SourceFamily, SourceFamily.id == SourceLineage.source_family_id)
            .where(
                SourceFamily.canonical_source_key == family_key,
                SourceLineage.scope_suffix == "",
            )
        ).scalar_one_or_none()
        effective = self.effective_packet(lineage.id) if lineage is not None else None
        if effective is None:
            return False
        observed = any(
            (packet.source_metadata or {}).get("route_record_id") == route_record_id
            for packet in self.session.execute(
                select(EvidencePacket).where(
                    EvidencePacket.source_lineage_id == lineage.id,
                    EvidencePacket.capture_route == capture_route,
                )
            ).scalars()
        )
        current = self._latest_channels(effective.id)
        if not observed or set(channels) <= current:
            return False
        self.revise_lens_eligibility(
            effective.id, evidence_channels=tuple(current | set(channels)), reason=reason
        )
        return True

    def _current_state(
        self, packet: EvidencePacket, effective: EvidencePacket | None
    ) -> str:
        """The packet's precedence now, read from the revision log."""
        if effective is None:
            return packet.precedence_state
        rank = self._standing(packet, effective)
        if rank == 0:
            return "effective" if packet.id == effective.id else "equivalent"
        return "hold_review" if rank == 1 else "superseded"

    def _standing(self, packet: EvidencePacket, effective: EvidencePacket) -> int | None:
        """Rank a packet that still stands against the current effective one.

        0: the effective packet or an equivalent of it; 1: still held for
        review; None: displaced. Stored precedence_state is fixed at admission,
        so standing is read from the precedence revision log instead.
        """
        disposition = self._latest_disposition(packet.id)
        if packet.id == effective.id or (
            disposition == "equivalent"
            and packet.equivalent_evidence_packet_id == effective.id
        ):
            return 0
        return 1 if disposition == "hold_review" else None

    def latest_channels(self, packet_id: UUID) -> set[str]:
        """The lens channels the packet is currently eligible for."""
        return self._latest_channels(packet_id)

    def _latest_channels(self, packet_id: UUID) -> set[str]:
        channels = self.session.scalar(
            select(LensEligibilityRevision.evidence_channels)
            .where(LensEligibilityRevision.evidence_packet_id == packet_id)
            .order_by(LensEligibilityRevision.revision_number.desc())
            .limit(1)
        )
        return set(channels or ())

    def _latest_effective_from_routes(
        self, lineage_id: UUID, routes: frozenset[str]
    ) -> EvidencePacket | None:
        """The packet from ``routes`` most recently made effective in the lineage."""
        return self.session.execute(
            select(EvidencePacket)
            .join(
                EvidencePrecedenceRevision,
                EvidencePrecedenceRevision.evidence_packet_id == EvidencePacket.id,
            )
            .where(
                EvidencePrecedenceRevision.source_lineage_id == lineage_id,
                EvidencePrecedenceRevision.disposition == "effective",
                EvidencePacket.capture_route.in_(routes),
            )
            .order_by(EvidencePrecedenceRevision.revision_number.desc())
            .limit(1)
        ).scalar_one_or_none()

    def _latest_disposition(self, packet_id: UUID) -> str | None:
        return self.session.scalar(
            select(EvidencePrecedenceRevision.disposition)
            .where(EvidencePrecedenceRevision.evidence_packet_id == packet_id)
            .order_by(EvidencePrecedenceRevision.revision_number.desc())
            .limit(1)
        )

    def effective_packet(self, lineage_id: UUID) -> EvidencePacket | None:
        return self.session.execute(
            select(EvidencePacket)
            .join(
                EvidencePrecedenceRevision,
                EvidencePrecedenceRevision.evidence_packet_id == EvidencePacket.id,
            )
            .where(
                EvidencePrecedenceRevision.source_lineage_id == lineage_id,
                EvidencePrecedenceRevision.disposition == "effective",
            )
            .order_by(
                EvidencePrecedenceRevision.revision_number.desc(),
                EvidencePrecedenceRevision.created_at.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()

    def revise_lens_eligibility(
        self,
        packet_id: UUID,
        *,
        add: str | None = None,
        remove: str | None = None,
        evidence_channels: tuple[str, ...] | None = None,
        reason: str,
    ) -> LensEligibilityResult:
        expected_epoch = self._current_epoch()
        with producer_write(
            self.session,
            expected_epoch=expected_epoch,
            allowed_modes={"legacy", "shadow", "dual", "economic"},
        ) as authority:
            return self._revise_lens_eligibility(
                packet_id,
                add=add,
                remove=remove,
                evidence_channels=evidence_channels,
                reason=reason,
                authority_epoch=authority.authority_epoch,
            )

    def _revise_lens_eligibility(
        self,
        packet_id: UUID,
        *,
        add: str | None,
        remove: str | None,
        evidence_channels: tuple[str, ...] | None,
        reason: str,
        authority_epoch: int,
    ) -> LensEligibilityResult:
        packet = self.session.get(EvidencePacket, packet_id)
        if packet is None:
            raise KeyError(f"evidence packet {packet_id} not found")
        self.session.execute(
            select(SourceLineage.id)
            .where(SourceLineage.id == packet.source_lineage_id)
            .with_for_update()
        ).scalar_one()
        latest = self.session.execute(
            select(LensEligibilityRevision)
            .where(LensEligibilityRevision.evidence_packet_id == packet.id)
            .order_by(LensEligibilityRevision.revision_number.desc())
            .limit(1)
        ).scalar_one_or_none()
        channels = set(latest.evidence_channels if latest else ())
        if evidence_channels is not None:
            channels = set(evidence_channels)
        if add is not None:
            channels.add(add)
        if remove is not None:
            channels.discard(remove)
        allowed = {channel.value for channel in EvidenceChannel}
        if channels - allowed:
            raise ValueError(f"unknown evidence channels: {sorted(channels - allowed)}")
        revision = LensEligibilityRevision(
            source_lineage_id=packet.source_lineage_id,
            evidence_packet_id=packet.id,
            revision_number=(latest.revision_number + 1 if latest else 1),
            evidence_channels=sorted(channels),
            reason=reason,
        )
        self.session.add(revision)
        self.session.flush()
        EconomicTaxonomyPublicationRepository(self.session).append_source_revision(
            producer_kind="evidence",
            logical_source_key=f"evidence_packet:{packet.id}",
            revision_kind="lens_eligibility",
            revision_number=revision.revision_number,
            content_hash=_hash(
                {
                    "evidence_packet_id": str(packet.id),
                    "revision_number": revision.revision_number,
                    "evidence_channels": list(revision.evidence_channels),
                    "reason": reason,
                }
            ),
            authority_epoch=authority_epoch,
        )
        return LensEligibilityResult(
            packet_id=packet.id,
            revision_number=revision.revision_number,
            evidence_channels=tuple(revision.evidence_channels),
        )

    def _current_epoch(self) -> int:
        authority = self.session.get(TaxonomyAuthority, 1)
        return authority.authority_epoch if authority is not None else 1

    def _get_or_create_family(self, evidence: EvidenceAdmission) -> SourceFamily:
        family = self.session.execute(
            select(SourceFamily).where(
                SourceFamily.canonical_source_key == evidence.family_key
            )
        ).scalar_one_or_none()
        if family is not None:
            return family
        try:
            with self.session.begin_nested():
                family = SourceFamily(
                    provider=evidence.provider.strip().lower(),
                    canonical_source_key=evidence.family_key,
                    canonical_item_id=evidence.canonical_item_id,
                )
                self.session.add(family)
                self.session.flush()
                return family
        except IntegrityError:
            return self.session.execute(
                select(SourceFamily).where(
                    SourceFamily.canonical_source_key == evidence.family_key
                )
            ).scalar_one()

    def _get_or_create_lineage(
        self, family: SourceFamily, evidence: EvidenceAdmission
    ) -> SourceLineage:
        suffix = evidence.scope_suffix or ""
        lineage = self.session.execute(
            select(SourceLineage).where(
                SourceLineage.source_family_id == family.id,
                SourceLineage.scope_suffix == suffix,
            )
        ).scalar_one_or_none()
        if lineage is not None:
            if lineage.admission_policy_key != evidence.admission_policy_key:
                raise ValueError("lineage admission policy does not match")
            return lineage
        try:
            with self.session.begin_nested():
                lineage = SourceLineage(
                    source_family_id=family.id,
                    scope_suffix=suffix,
                    admission_policy_key=evidence.admission_policy_key,
                )
                self.session.add(lineage)
                self.session.flush()
                return lineage
        except IntegrityError:
            return self.session.execute(
                select(SourceLineage).where(
                    SourceLineage.source_family_id == family.id,
                    SourceLineage.scope_suffix == suffix,
                )
            ).scalar_one()

    @staticmethod
    def _content_fingerprint(evidence: EvidenceAdmission) -> str:
        return _hash(
            {
                "original_text": evidence.original_text,
                "translated_text": evidence.translated_text,
                "translation_version": evidence.translation_version,
                "attachment_hashes": _ordered(evidence.attachment_hashes),
                "extracted_text_hashes": _ordered(evidence.extracted_text_hashes),
                "grounding_snapshot": dict(evidence.grounding_snapshot),
                "preparation_version": evidence.preparation_version,
            }
        )

    @staticmethod
    def _packet_hash(evidence: EvidenceAdmission, fingerprint: str) -> str:
        return _hash(
            {
                "evidence_content_fingerprint": fingerprint,
                "provider_revision_id": evidence.provider_revision_id,
                "provider_revision_order": evidence.provider_revision_order,
                "capture_route": evidence.capture_route,
                "route_record_id": evidence.route_record_id,
                "captured_at": evidence.captured_at,
                "observed_at": evidence.observed_at,
                "available_at": evidence.available_at,
                "source_metadata": dict(evidence.source_metadata),
                "supersedes_packet_id": evidence.supersedes_packet_id,
                "equivalent_packet_id": evidence.equivalent_packet_id,
            }
        )

    @staticmethod
    def _descriptor(packet: EvidencePacket) -> EvidencePacketDescriptor:
        provider_order = None
        if packet.provider_revision_order is not None:
            try:
                provider_order = int(packet.provider_revision_order)
            except (TypeError, ValueError):
                provider_order = None
        return EvidencePacketDescriptor(
            packet_id=str(packet.id),
            evidence_content_fingerprint=packet.evidence_content_fingerprint,
            provider_revision_id=packet.provider_revision_id,
            provider_revision_order=provider_order,
            supersedes_packet_id=(
                str(packet.supersedes_evidence_packet_id)
                if packet.supersedes_evidence_packet_id
                else None
            ),
            equivalent_to_packet_id=(
                str(packet.equivalent_evidence_packet_id)
                if packet.equivalent_evidence_packet_id
                else None
            ),
            captured_from=packet.capture_route,
            attachment_hashes=frozenset(packet.attachment_hashes or ()),
        )

    def _next_precedence_revision(self, lineage_id: UUID) -> int:
        current = self.session.scalar(
            select(func.max(EvidencePrecedenceRevision.revision_number)).where(
                EvidencePrecedenceRevision.source_lineage_id == lineage_id
            )
        )
        return int(current or 0) + 1

    @staticmethod
    def _is_unordered_archive(evidence: EvidenceAdmission) -> bool:
        return (
            evidence.provider_revision_id is None
            and evidence.provider_revision_order is None
            and EconomicSourceAdmissionService._is_archive_or_partial(evidence)
        )

    @staticmethod
    def _is_archive_or_partial(evidence: EvidenceAdmission) -> bool:
        return (
            "archive" in evidence.capture_route.lower()
            or bool(evidence.source_metadata.get("archived"))
            or bool(evidence.source_metadata.get("partial_recapture"))
        )

    @staticmethod
    def _result(
        family: SourceFamily,
        lineage: SourceLineage,
        packet: EvidencePacket,
        effective: EvidencePacket | None,
        *,
        precedence_state: str | None = None,
    ) -> AdmissionResult:
        return AdmissionResult(
            source_family_id=family.id,
            source_lineage_id=lineage.id,
            packet_id=packet.id,
            effective_packet_id=effective.id if effective else None,
            evidence_revision_ordinal=packet.evidence_revision_ordinal,
            precedence_state=precedence_state or packet.precedence_state,
            packet_hash=packet.packet_hash,
            evidence_content_fingerprint=packet.evidence_content_fingerprint,
        )

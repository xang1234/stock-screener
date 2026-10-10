"""Persistence boundary for bounded taxonomy publication inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.orm import Session

from app.models.economic_taxonomy import TaxonomyVersion
from app.models.economic_taxonomy_runtime import (
    GenerationInputManifest,
    InterpretationSet,
    MetricsRevision,
    ReaderCapabilityManifest,
    ReaderSnapshotBundle,
    SemanticInvalidationRevision,
    ServingGeneration,
    ServingGenerationEvent,
    TaxonomyAuthority,
    TaxonomySourceRevisionLog,
)
from app.utils.file_hashing import canonical_json_sha256 as _hash_payload


class PublicationInvariantError(ValueError):
    """A candidate generation does not form one coherent sealed payload."""


@dataclass(frozen=True, slots=True)
class ManifestValidation:
    valid: bool
    error: str | None = None


class EconomicTaxonomyPublicationRepository:
    def __init__(self, session: Session):
        self.session = session

    def lock_authority(self) -> TaxonomyAuthority:
        defaults = {
            "id": 1,
            "mode": "legacy",
            "processing_head_revision": 0,
            "authority_epoch": 1,
            "writes_fenced": False,
            "semantic_invalidation_revision": 0,
            "cutover_catch_up_cursor": [],
            "rollback_state": "ready",
        }
        if self.session.get_bind().dialect.name == "postgresql":
            self.session.execute(
                postgresql_insert(TaxonomyAuthority)
                .values(**defaults)
                .on_conflict_do_nothing(index_elements=[TaxonomyAuthority.id])
            )
        # populate_existing: a caller that loaded the row earlier in this
        # session must see what the lock read, or fence checks would use
        # pre-lock values (#556).
        authority = self.session.execute(
            select(TaxonomyAuthority)
            .where(TaxonomyAuthority.id == 1)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        if authority is None:
            authority = TaxonomyAuthority(**defaults)
            self.session.add(authority)
            self.session.flush()
        return authority

    def append_source_revision(
        self,
        *,
        producer_kind: str,
        logical_source_key: str,
        revision_kind: str,
        revision_number: int,
        content_hash: str,
        authority_epoch: int,
        revision_id: UUID | None = None,
    ) -> TaxonomySourceRevisionLog:
        row = TaxonomySourceRevisionLog(
            id=revision_id or uuid4(),
            producer_kind=producer_kind,
            logical_source_key=logical_source_key,
            revision_kind=revision_kind,
            revision_number=revision_number,
            content_hash=content_hash,
            authority_epoch=authority_epoch,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def append_semantic_invalidation(
        self, *, reason: str, actor: str
    ) -> SemanticInvalidationRevision:
        authority = self.lock_authority()
        next_revision = authority.semantic_invalidation_revision + 1
        row = SemanticInvalidationRevision(
            revision_number=next_revision,
            reason=reason,
            created_by=actor,
        )
        authority.semantic_invalidation_revision = next_revision
        self.session.add(row)
        self.session.flush()
        return row

    def capture_manifest(
        self,
        *,
        actor: str,
        selections: list[dict],
        authority: TaxonomyAuthority | None = None,
    ) -> GenerationInputManifest:
        authority = authority or self.lock_authority()
        revisions = self.session.execute(
            select(TaxonomySourceRevisionLog).order_by(
                TaxonomySourceRevisionLog.producer_kind,
                TaxonomySourceRevisionLog.logical_source_key,
                TaxonomySourceRevisionLog.revision_kind,
                TaxonomySourceRevisionLog.revision_number,
                TaxonomySourceRevisionLog.id,
            )
        ).scalars()
        tuples = [
            [
                row.producer_kind,
                row.logical_source_key,
                row.revision_kind,
                row.revision_number,
                row.content_hash,
                str(row.id),
            ]
            for row in revisions
        ]
        normalized_selections = sorted(
            selections,
            key=lambda item: (
                str(item.get("lineage", "")),
                str(item.get("evidence_packet_id", "")),
                json.dumps(item, sort_keys=True, separators=(",", ":")),
            ),
        )
        semantic_payload = {
            "expected_parent_generation_id": (
                str(authority.serving_generation_id)
                if authority.serving_generation_id
                else None
            ),
            "semantic_invalidation_revision": authority.semantic_invalidation_revision,
            "committed_revision_tuples": tuples,
            "selections": normalized_selections,
        }
        manifest = GenerationInputManifest(
            status="unsealed",
            expected_parent_generation_id=authority.serving_generation_id,
            semantic_invalidation_revision=authority.semantic_invalidation_revision,
            committed_revision_tuples=tuples,
            selections=normalized_selections,
            created_by=actor,
        )
        self.session.add(manifest)
        self.session.flush()
        manifest.seal(
            semantic_hash=_hash_payload(semantic_payload),
            artifact_integrity_hash=_hash_payload(
                {**semantic_payload, "created_by": actor, "manifest_id": str(manifest.id)}
            ),
        )
        self.session.flush()
        return manifest

    def validate_manifest(
        self, manifest: GenerationInputManifest
    ) -> ManifestValidation:
        authority = self.session.get(TaxonomyAuthority, 1)
        if authority is None:
            return ManifestValidation(False, "authority_missing")
        if (
            authority.semantic_invalidation_revision
            != manifest.semantic_invalidation_revision
        ):
            return ManifestValidation(False, "semantic_invalidation_changed")
        if authority.serving_generation_id != manifest.expected_parent_generation_id:
            return ManifestValidation(False, "expected_parent_changed")
        if manifest.status != "sealed":
            return ManifestValidation(False, "manifest_unsealed")
        return ManifestValidation(True)

    def prepare_generation(
        self,
        *,
        taxonomy_version_id: UUID,
        interpretation_set_id: UUID,
        metrics_revision_id: UUID,
        generation_input_manifest_id: UUID,
        reader_snapshot_bundle_id: UUID,
        reader_capability_manifest_id: UUID,
        semantic_hash: str,
        artifact_integrity_hash: str,
        actor: str,
        prepared_details: dict | None = None,
    ) -> ServingGeneration:
        taxonomy = self.session.get(TaxonomyVersion, taxonomy_version_id)
        interpretation = self.session.get(InterpretationSet, interpretation_set_id)
        metrics = self.session.get(MetricsRevision, metrics_revision_id)
        manifest = self.session.get(GenerationInputManifest, generation_input_manifest_id)
        snapshots = self.session.get(ReaderSnapshotBundle, reader_snapshot_bundle_id)
        capability = self.session.get(
            ReaderCapabilityManifest, reader_capability_manifest_id
        )
        if any(
            item is None
            for item in (
                taxonomy,
                interpretation,
                metrics,
                manifest,
                snapshots,
                capability,
            )
        ):
            raise PublicationInvariantError("generation_reference_missing")
        if taxonomy.status != "sealed":
            raise PublicationInvariantError("taxonomy_unsealed")
        if any(
            payload.status != "sealed"
            for payload in (interpretation, metrics, manifest, snapshots)
        ):
            raise PublicationInvariantError("generation_payload_unsealed")
        if {
            interpretation.generation_input_manifest_id,
            metrics.generation_input_manifest_id,
            snapshots.generation_input_manifest_id,
        } != {manifest.id}:
            raise PublicationInvariantError("manifest_mismatch")

        generation = ServingGeneration(
            expected_parent_generation_id=manifest.expected_parent_generation_id,
            taxonomy_version_id=taxonomy.id,
            interpretation_set_id=interpretation.id,
            metrics_revision_id=metrics.id,
            generation_input_manifest_id=manifest.id,
            reader_snapshot_bundle_id=snapshots.id,
            reader_capability_manifest_id=capability.id,
            semantic_hash=semantic_hash,
            artifact_integrity_hash=artifact_integrity_hash,
            created_by=actor,
        )
        generation.events.append(
            ServingGenerationEvent(
                sequence_number=1,
                event_type="prepared",
                actor=actor,
                details=dict(prepared_details or {}),
            )
        )
        self.session.add(generation)
        return generation

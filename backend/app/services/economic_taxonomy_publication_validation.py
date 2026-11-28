"""Validation and pointer-switch helpers for taxonomy publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
    PublicationInvariantError,
)
from app.models.economic_taxonomy import TaxonomyVersion
from app.models.economic_taxonomy_runtime import (
    GenerationInputManifest,
    InterpretationSet,
    MetricsRevision,
    ReaderCapabilityManifest,
    ReaderSnapshotBundle,
    ReaderSnapshotEntry,
    ReaderSnapshotPointer,
    ServingGeneration,
    TaxonomyAuthority,
    TaxonomyProjectionEvent,
)
from app.services.economic_taxonomy_benchmark_store import (
    benchmark_anchor_version,
    find_verified_benchmark,
    verify_benchmark_reference,
)
from app.services.economic_taxonomy_publication_compatibility import event_descriptor
from app.services.economic_taxonomy_publication_contracts import (
    BenchmarkRejected,
    CompatibilityNotAcknowledged,
    PreparationContext,
    ReaderCapabilityRejected,
)
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.utils.file_hashing import canonical_json_sha256 as _hash


def publication_change_reason(
    session: Session,
    authority: TaxonomyAuthority,
    generation: ServingGeneration,
    details: Mapping[str, Any],
) -> str | None:
    manifest = session.get(
        GenerationInputManifest, generation.generation_input_manifest_id
    )
    validation = EconomicTaxonomyPublicationRepository(session).validate_manifest(
        manifest
    )
    if not validation.valid:
        return validation.error
    if str(authority.processing_taxonomy_version_id) != str(
        details.get("processing_taxonomy_version_id")
    ):
        return "processing_taxonomy_changed"
    if authority.processing_head_revision != int(
        details.get("processing_head_revision", -1)
    ):
        return "processing_head_changed"
    return None


def verify_capability(
    capability: ReaderCapabilityManifest | None,
    *,
    backend_contract: int,
    frontend_contract: int,
    migration_version: str,
) -> None:
    if capability is None:
        raise ReaderCapabilityRejected("reader_capability_missing")
    if (
        capability.backend_contract != backend_contract
        or capability.frontend_contract != frontend_contract
        or capability.migration_version != migration_version
        or not capability.consumer_test_hash.strip()
    ):
        raise ReaderCapabilityRejected("reader_capability_incompatible")


def verified_benchmark(
    session: Session,
    context: PreparationContext,
    *,
    policy_bundle: str,
) -> Mapping[str, Any]:
    try:
        anchor = benchmark_anchor_version(session, context.taxonomy_version_id)
        return find_verified_benchmark(
            session,
            taxonomy_semantic_hash=str(anchor.semantic_hash),
            policy_bundle=policy_bundle,
        )
    except (LookupError, ValueError) as exc:
        raise BenchmarkRejected(str(exc)) from exc


def require_prior_acknowledgement(
    session: Session,
    generation_id,
) -> None:
    if generation_id is None:
        return
    if not EconomicTaxonomyRuntimeService(session).generation_acknowledged(
        generation_id
    ):
        raise CompatibilityNotAcknowledged(
            "prior_generation_compatibility_not_acknowledged"
        )


def verify_generation(
    session: Session,
    authority: TaxonomyAuthority,
    generation: ServingGeneration,
    details: Mapping[str, Any],
    *,
    reader_keys: Sequence[str],
    backend_contract: int,
    frontend_contract: int,
    migration_version: str,
    policy_bundle: str,
) -> None:
    taxonomy = session.get(TaxonomyVersion, generation.taxonomy_version_id)
    manifest = session.get(
        GenerationInputManifest, generation.generation_input_manifest_id
    )
    interpretation = session.get(InterpretationSet, generation.interpretation_set_id)
    metrics = session.get(MetricsRevision, generation.metrics_revision_id)
    snapshots = session.get(ReaderSnapshotBundle, generation.reader_snapshot_bundle_id)
    capability = session.get(
        ReaderCapabilityManifest, generation.reader_capability_manifest_id
    )
    if any(
        value is None
        for value in (taxonomy, manifest, interpretation, metrics, snapshots)
    ):
        raise PublicationInvariantError("generation_reference_missing")
    if taxonomy.status != "sealed" or any(
        value.status != "sealed"
        for value in (manifest, interpretation, metrics, snapshots)
    ):
        raise PublicationInvariantError("generation_payload_unsealed")
    component_hashes = details.get("component_hashes", {})
    actual_hashes = {
        "taxonomy": taxonomy.semantic_hash,
        "manifest": manifest.semantic_hash,
        "interpretation": interpretation.semantic_hash,
        "metrics": metrics.semantic_hash,
        "snapshot": snapshots.semantic_hash,
    }
    if component_hashes != actual_hashes:
        raise PublicationInvariantError("generation_component_hash_changed")
    artifact_hashes = details.get("component_artifact_hashes", {})
    actual_artifact_hashes = {
        "taxonomy": taxonomy.artifact_integrity_hash,
        "manifest": manifest.artifact_integrity_hash,
        "interpretation": interpretation.artifact_integrity_hash,
        "metrics": metrics.artifact_integrity_hash,
        "snapshot": snapshots.artifact_integrity_hash,
    }
    if artifact_hashes != actual_artifact_hashes:
        raise PublicationInvariantError("generation_artifact_hash_changed")
    semantic_payload = {
        "taxonomy_hash": taxonomy.semantic_hash,
        "manifest_hash": manifest.semantic_hash,
        "interpretation_hash": interpretation.semantic_hash,
        "metrics_hash": metrics.semantic_hash,
        "snapshot_hash": snapshots.semantic_hash,
        "benchmark": details.get("benchmark"),
        "target_mode": details.get("target_mode"),
        "expected_projections": list(details.get("expected_projections", [])),
    }
    if generation.semantic_hash != _hash(semantic_payload):
        raise PublicationInvariantError("generation_semantic_hash_changed")
    artifact_payload = {
        **semantic_payload,
        "taxonomy_artifact_hash": taxonomy.artifact_integrity_hash,
        "manifest_artifact_hash": manifest.artifact_integrity_hash,
        "interpretation_artifact_hash": interpretation.artifact_integrity_hash,
        "metrics_artifact_hash": metrics.artifact_integrity_hash,
        "snapshot_artifact_hash": snapshots.artifact_integrity_hash,
        "created_by": generation.created_by,
    }
    if generation.artifact_integrity_hash != _hash(artifact_payload):
        raise PublicationInvariantError("generation_artifact_integrity_changed")
    if manifest.semantic_hash != details.get("manifest_semantic_hash"):
        raise PublicationInvariantError("manifest_hash_changed")
    verify_snapshot_entries(session, snapshots, reader_keys=reader_keys)
    if details.get("target_mode") != "shadow":
        try:
            verify_benchmark_reference(
                session,
                details.get("benchmark") or {},
                taxonomy_semantic_hash=str(
                    benchmark_anchor_version(session, taxonomy.id).semantic_hash
                ),
                policy_bundle=policy_bundle,
            )
        except (LookupError, ValueError) as exc:
            raise BenchmarkRejected(str(exc)) from exc
    verify_capability(
        capability,
        backend_contract=backend_contract,
        frontend_contract=frontend_contract,
        migration_version=migration_version,
    )
    require_prior_acknowledgement(session, authority.serving_generation_id)
    verify_staged_projections(session, generation, details)


def verify_snapshot_entries(
    session: Session,
    bundle: ReaderSnapshotBundle,
    *,
    reader_keys: Sequence[str],
) -> None:
    entries = session.scalars(
        select(ReaderSnapshotEntry)
        .where(ReaderSnapshotEntry.reader_snapshot_bundle_id == bundle.id)
        .order_by(ReaderSnapshotEntry.snapshot_kind, ReaderSnapshotEntry.resource_key)
    ).all()
    if {entry.snapshot_kind for entry in entries} != set(reader_keys):
        raise PublicationInvariantError("reader_snapshot_incomplete")
    actual = [
        {
            "snapshot_kind": entry.snapshot_kind,
            "resource_key": entry.resource_key,
            "payload_hash": _hash(entry.payload),
        }
        for entry in entries
    ]
    if any(
        entry.payload_hash != item["payload_hash"]
        for entry, item in zip(entries, actual, strict=True)
    ):
        raise PublicationInvariantError("reader_snapshot_entry_hash_changed")
    indexed = sorted(
        bundle.payload.get("entries", []),
        key=lambda value: (value["snapshot_kind"], value["resource_key"]),
    )
    if indexed != actual:
        raise PublicationInvariantError("reader_snapshot_index_changed")


def verify_staged_projections(
    session: Session,
    generation: ServingGeneration,
    details: Mapping[str, Any],
) -> None:
    expected = list(details.get("expected_projections", []))
    rows = session.scalars(
        select(TaxonomyProjectionEvent)
        .where(
            TaxonomyProjectionEvent.serving_generation_id == generation.id,
            TaxonomyProjectionEvent.delivery_scope == "candidate",
        )
        .order_by(
            TaxonomyProjectionEvent.source_lineage,
            TaxonomyProjectionEvent.projection_kind,
            TaxonomyProjectionEvent.target_representation,
            TaxonomyProjectionEvent.projection_revision,
        )
    ).all()
    if [event_descriptor(row) for row in rows] != expected:
        raise PublicationInvariantError("candidate_projection_set_changed")


def switch_reader_pointers(
    session: Session,
    *,
    generation: ServingGeneration,
    authority_epoch: int,
    reader_keys: Sequence[str],
) -> None:
    bundle_keys = set(
        session.scalars(
            select(ReaderSnapshotEntry.snapshot_kind).where(
                ReaderSnapshotEntry.reader_snapshot_bundle_id
                == generation.reader_snapshot_bundle_id
            )
        )
    )
    if bundle_keys != set(reader_keys):
        raise PublicationInvariantError("reader_snapshot_incomplete")
    for reader_key in reader_keys:
        pointer = session.get(ReaderSnapshotPointer, reader_key)
        if pointer is None:
            pointer = ReaderSnapshotPointer(reader_key=reader_key)
            session.add(pointer)
        pointer.serving_generation_id = generation.id
        pointer.reader_snapshot_bundle_id = generation.reader_snapshot_bundle_id
        pointer.authority_epoch = authority_epoch

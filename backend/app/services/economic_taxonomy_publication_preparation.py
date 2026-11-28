"""Build immutable artifacts for a candidate serving generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.domain.economic_taxonomy.contracts import AdminPrincipal
from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    EconomicTaxonomyPublicationRepository,
    PublicationInvariantError,
)
from app.models.economic_taxonomy import TaxonomyVersion
from app.models.economic_taxonomy_runtime import (
    GenerationInputManifest,
    ReaderCapabilityManifest,
)
from app.services.economic_taxonomy_benchmark_store import (
    benchmark_anchor_version,
    verify_benchmark_reference,
)
from app.services.economic_taxonomy_interpretations import (
    EconomicTaxonomyInterpretationService,
)
from app.services.economic_taxonomy_publication_compatibility import (
    CompatibilityProjection,
    assign_projection_revisions,
    projection_descriptor,
)
from app.services.economic_taxonomy_publication_contracts import (
    BenchmarkRejected,
    PreparationContext,
    PreparedGeneration,
    PublicationCutoff,
)
from app.services.economic_taxonomy_publication_validation import (
    require_prior_acknowledgement,
    verify_capability,
)
from app.services.economic_taxonomy_runtime import EconomicTaxonomyRuntimeService
from app.services.economic_taxonomy_snapshot_builder import (
    GenerationSnapshotInputs,
    build_snapshot_bundle,
)
from app.services.economic_theme_metrics_service import EconomicThemeMetricsService
from app.utils.file_hashing import canonical_json_sha256 as _hash

CompatibilityBuilder = Callable[
    [Session, PreparationContext], Sequence[CompatibilityProjection]
]
BenchmarkVerifier = Callable[[Session, PreparationContext], Mapping[str, Any] | bool]


class PublicationPreparer:
    def __init__(
        self,
        session_factory,
        *,
        clock,
        compatibility_builder: CompatibilityBuilder,
        benchmark_verifier: BenchmarkVerifier,
        required_backend_contract: int,
        required_frontend_contract: int,
        required_migration_version: str,
        required_policy_bundle: str,
    ):
        self.session_factory = session_factory
        self.clock = clock
        self.compatibility_builder = compatibility_builder
        self.benchmark_verifier = benchmark_verifier
        self.required_backend_contract = required_backend_contract
        self.required_frontend_contract = required_frontend_contract
        self.required_migration_version = required_migration_version
        self.required_policy_bundle = required_policy_bundle

    def prepare(
        self,
        cutoff: PublicationCutoff,
        *,
        principal: AdminPrincipal,
        reader_capability_manifest_id,
        target_mode: str,
    ) -> PreparedGeneration:
        interpretation = EconomicTaxonomyInterpretationService(
            self.session_factory
        ).build_interpretation_set(cutoff.manifest_id, actor=principal.subject)

        with self.session_factory() as session:
            manifest = session.get(GenerationInputManifest, cutoff.manifest_id)
            taxonomy = session.get(
                TaxonomyVersion, cutoff.processing_taxonomy_version_id
            )
            capability = session.get(
                ReaderCapabilityManifest, reader_capability_manifest_id
            )
            if manifest is None or manifest.status != "sealed":
                raise PublicationInvariantError("sealed_manifest_required")
            if manifest.semantic_hash != cutoff.manifest_semantic_hash:
                raise PublicationInvariantError("captured_manifest_hash_changed")
            if taxonomy is None or taxonomy.status != "sealed":
                raise PublicationInvariantError("sealed_processing_taxonomy_required")
            verify_capability(
                capability,
                backend_contract=self.required_backend_contract,
                frontend_contract=self.required_frontend_contract,
                migration_version=self.required_migration_version,
            )
            require_prior_acknowledgement(session, cutoff.expected_parent_generation_id)

            metrics = EconomicThemeMetricsService(session).calculate_metrics(
                taxonomy_version_id=taxonomy.id,
                interpretation_set_id=interpretation.id,
                generation_input_manifest_id=manifest.id,
                as_of=self.clock(),
                actor=principal.subject,
            )
            snapshots = build_snapshot_bundle(
                session,
                GenerationSnapshotInputs(
                    taxonomy_version_id=taxonomy.id,
                    interpretation_set_id=interpretation.id,
                    generation_input_manifest_id=manifest.id,
                    metrics_revision_id=metrics.id,
                    created_by=principal.subject,
                ),
            )
            context = PreparationContext(
                manifest_id=manifest.id,
                taxonomy_version_id=taxonomy.id,
                interpretation_set_id=interpretation.id,
                metrics_revision_id=metrics.id,
                reader_snapshot_bundle_id=snapshots.id,
                expected_parent_generation_id=cutoff.expected_parent_generation_id,
                staged_epoch=cutoff.authority_epoch,
                target_mode=target_mode,
            )
            benchmark = None
            if target_mode != "shadow":
                benchmark = self.benchmark_verifier(session, context)
                if benchmark is False or not isinstance(benchmark, Mapping):
                    raise BenchmarkRejected("economic_taxonomy_benchmark_failed")
                if not bool(benchmark.get("passed", False)):
                    raise BenchmarkRejected("economic_taxonomy_benchmark_failed")
                try:
                    anchor = benchmark_anchor_version(session, taxonomy.id)
                    benchmark = verify_benchmark_reference(
                        session,
                        benchmark,
                        taxonomy_semantic_hash=str(anchor.semantic_hash),
                        policy_bundle=self.required_policy_bundle,
                    )
                except (LookupError, ValueError) as exc:
                    raise BenchmarkRejected(str(exc)) from exc

            projections = assign_projection_revisions(
                session, self.compatibility_builder(session, context)
            )
            expected_projections = [
                projection_descriptor(value) for value in projections
            ]
            semantic_payload = {
                "taxonomy_hash": taxonomy.semantic_hash,
                "manifest_hash": manifest.semantic_hash,
                "interpretation_hash": interpretation.semantic_hash,
                "metrics_hash": metrics.semantic_hash,
                "snapshot_hash": snapshots.semantic_hash,
                "benchmark": benchmark,
                "target_mode": target_mode,
                "expected_projections": expected_projections,
            }
            artifact_payload = {
                **semantic_payload,
                "taxonomy_artifact_hash": taxonomy.artifact_integrity_hash,
                "manifest_artifact_hash": manifest.artifact_integrity_hash,
                "interpretation_artifact_hash": (
                    interpretation.artifact_integrity_hash
                ),
                "metrics_artifact_hash": metrics.artifact_integrity_hash,
                "snapshot_artifact_hash": snapshots.artifact_integrity_hash,
                "created_by": principal.subject,
            }
            details = {
                "target_mode": target_mode,
                "processing_taxonomy_version_id": str(taxonomy.id),
                "processing_head_revision": cutoff.processing_head_revision,
                "expected_parent_generation_id": (
                    str(cutoff.expected_parent_generation_id)
                    if cutoff.expected_parent_generation_id
                    else None
                ),
                "semantic_invalidation_revision": (
                    cutoff.semantic_invalidation_revision
                ),
                "manifest_semantic_hash": manifest.semantic_hash,
                "component_hashes": {
                    "taxonomy": taxonomy.semantic_hash,
                    "manifest": manifest.semantic_hash,
                    "interpretation": interpretation.semantic_hash,
                    "metrics": metrics.semantic_hash,
                    "snapshot": snapshots.semantic_hash,
                },
                "component_artifact_hashes": {
                    "taxonomy": taxonomy.artifact_integrity_hash,
                    "manifest": manifest.artifact_integrity_hash,
                    "interpretation": interpretation.artifact_integrity_hash,
                    "metrics": metrics.artifact_integrity_hash,
                    "snapshot": snapshots.artifact_integrity_hash,
                },
                "expected_projections": expected_projections,
                "benchmark": benchmark,
            }
            generation = EconomicTaxonomyPublicationRepository(
                session
            ).prepare_generation(
                taxonomy_version_id=taxonomy.id,
                interpretation_set_id=interpretation.id,
                metrics_revision_id=metrics.id,
                generation_input_manifest_id=manifest.id,
                reader_snapshot_bundle_id=snapshots.id,
                reader_capability_manifest_id=capability.id,
                semantic_hash=_hash(semantic_payload),
                artifact_integrity_hash=_hash(artifact_payload),
                actor=principal.subject,
                prepared_details=details,
            )
            session.flush()
            runtime = EconomicTaxonomyRuntimeService(session)
            for projection in projections:
                runtime.stage_projection(
                    generation_id=generation.id,
                    source_lineage=projection.source_lineage,
                    projection_revision=int(projection.projection_revision),
                    projection_kind=projection.projection_kind,
                    projection_version=projection.projection_version,
                    target=projection.target,
                    payload=projection.payload,
                    staged_epoch=cutoff.authority_epoch,
                    origin_representation=projection.origin_representation,
                    selected_interpretation_version=(
                        projection.selected_interpretation_version
                    ),
                    mapping_version=projection.mapping_version,
                    delivery_scope="candidate",
                )
            session.commit()
            return PreparedGeneration(
                id=generation.id,
                manifest_id=manifest.id,
                taxonomy_version_id=taxonomy.id,
                interpretation_set_id=interpretation.id,
                metrics_revision_id=metrics.id,
                reader_snapshot_bundle_id=snapshots.id,
                target_mode=target_mode,
            )

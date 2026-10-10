"""Build immutable Economic Taxonomy reader snapshots for publication."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infra.db.models.social_analysis import (
    EconomicSocialAssociation,
    EconomicSocialAssociationRevision,
)
from app.models.economic_taxonomy import (
    EconomicThemeAlias,
    EconomicThemeFacet,
    EconomicThemeRedirect,
    EconomicThemeRelationship,
    EconomicThemeRevision,
    FacetDimension,
    FacetValue,
    LegacyClaimAllocation,
    LegacyDestinationMapping,
    LegacyIdentityDisposition,
    TaxonomyVersion,
)
from app.models.economic_taxonomy_runtime import (
    ClaimAssignment,
    ClassificationAttempt,
    DevelopmentSelectionRevision,
    GenerationInputManifest,
    InterpretationSelection,
    InterpretationSet,
    MetricsRevision,
    ProcessingRequest,
    ReaderSnapshotBundle,
    ReaderSnapshotEntry,
    SocialAssociationRevisionRef,
    SourceLineage,
    TaxonomyOperationEvent,
    TaxonomyOperationPreview,
    TaxonomyOperationRequest,
    ThemeConstituentExposure,
    ThemeMetric,
)
from app.models.stock_universe import StockUniverse
from app.models.theme_intelligence import (
    EconomicDevelopmentBackfill,
    EconomicThemeDevelopment,
    ThemeDevelopmentObservation,
)
from app.services.economic_taxonomy_interpretations import (
    manifest_social_revision_contracts,
)
from app.services.economic_theme_observation_service import (
    observation_rows_for_interpretation,
    signal_rows_for_interpretation,
)
from app.utils.file_hashing import canonical_json_sha256 as _snapshot_hash


class SnapshotBundleError(ValueError):
    """A reader bundle cannot be built from incoherent generation inputs."""


@dataclass(frozen=True, slots=True)
class GenerationSnapshotInputs:
    taxonomy_version_id: UUID
    interpretation_set_id: UUID
    generation_input_manifest_id: UUID
    metrics_revision_id: UUID
    created_by: str


_ECONOMIC_RANKING_VIEWS = (
    "technical_attention",
    "fundamental_attention",
    "narrative_attention",
    "emerging",
    "broad_confirmation",
)


def build_snapshot_bundle(
    db: Session,
    generation_inputs: GenerationSnapshotInputs | Mapping[str, Any],
) -> ReaderSnapshotBundle:
    """Build and seal immutable API/UI payload rows without switching pointers."""

    inputs = (
        generation_inputs
        if isinstance(generation_inputs, GenerationSnapshotInputs)
        else GenerationSnapshotInputs(**dict(generation_inputs))
    )
    taxonomy = db.get(TaxonomyVersion, inputs.taxonomy_version_id)
    interpretation = db.get(InterpretationSet, inputs.interpretation_set_id)
    manifest = db.get(
        GenerationInputManifest, inputs.generation_input_manifest_id
    )
    metrics = db.get(MetricsRevision, inputs.metrics_revision_id)
    if taxonomy is None or taxonomy.status != "sealed":
        raise SnapshotBundleError("sealed_taxonomy_required")
    if interpretation is None or interpretation.status != "sealed":
        raise SnapshotBundleError("sealed_interpretation_required")
    if manifest is None or manifest.status != "sealed":
        raise SnapshotBundleError("sealed_manifest_required")
    if metrics is None or metrics.status != "sealed":
        raise SnapshotBundleError("sealed_metrics_required")
    if {
        interpretation.generation_input_manifest_id,
        metrics.generation_input_manifest_id,
    } != {manifest.id}:
        raise SnapshotBundleError("snapshot_manifest_mismatch")

    catalog, review = _build_economic_snapshot_payloads(
        db,
        taxonomy=taxonomy,
        interpretation=interpretation,
        manifest=manifest,
        metrics=metrics,
    )
    entries = (
        ("economic_themes", "catalog", catalog),
        ("economic_taxonomy", "review", review),
    )
    entry_index = [
        {
            "snapshot_kind": kind,
            "resource_key": key,
            "payload_hash": _snapshot_hash(payload),
        }
        for kind, key, payload in entries
    ]
    bundle_summary = {
        "taxonomy_version_id": str(taxonomy.id),
        "interpretation_set_id": str(interpretation.id),
        "generation_input_manifest_id": str(manifest.id),
        "generation_input_manifest_hash": manifest.semantic_hash,
        "metrics_revision_id": str(metrics.id),
        "entries": entry_index,
    }
    bundle = ReaderSnapshotBundle(
        status="unsealed",
        generation_input_manifest_id=manifest.id,
        payload=bundle_summary,
        created_by=inputs.created_by,
    )
    db.add(bundle)
    db.flush()
    for kind, key, payload in entries:
        db.add(
            ReaderSnapshotEntry(
                reader_snapshot_bundle_id=bundle.id,
                snapshot_kind=kind,
                resource_key=key,
                payload=payload,
                payload_hash=_snapshot_hash(payload),
            )
        )
    db.flush()
    bundle.seal(
        semantic_hash=_snapshot_hash(
            {
                "taxonomy_semantic_hash": taxonomy.semantic_hash,
                "manifest_semantic_hash": manifest.semantic_hash,
                "interpretation_semantic_hash": interpretation.semantic_hash,
                "metrics_semantic_hash": metrics.semantic_hash,
                "entries": entry_index,
            }
        ),
        artifact_integrity_hash=_snapshot_hash(
            {"bundle": bundle_summary, "entries": [value[2] for value in entries]}
        ),
    )
    db.flush()
    return bundle


def _build_economic_snapshot_payloads(
    db,
    *,
    taxonomy,
    interpretation,
    manifest,
    metrics,
):
    revisions = db.scalars(
        select(EconomicThemeRevision)
        .where(EconomicThemeRevision.taxonomy_version_id == taxonomy.id)
        .order_by(EconomicThemeRevision.display_name, EconomicThemeRevision.theme_id)
    ).all()
    theme_ids = {row.theme_id for row in revisions}
    aliases = db.scalars(
        select(EconomicThemeAlias).where(
            EconomicThemeAlias.taxonomy_version_id == taxonomy.id
        )
    ).all()
    dimensions = db.scalars(
        select(FacetDimension).where(FacetDimension.taxonomy_version_id == taxonomy.id)
    ).all()
    values = db.scalars(
        select(FacetValue).where(FacetValue.taxonomy_version_id == taxonomy.id)
    ).all()
    facets = db.scalars(
        select(EconomicThemeFacet).where(
            EconomicThemeFacet.taxonomy_version_id == taxonomy.id
        )
    ).all()
    relationships = db.scalars(
        select(EconomicThemeRelationship).where(
            EconomicThemeRelationship.taxonomy_version_id == taxonomy.id
        )
    ).all()
    dispositions = db.scalars(
        select(LegacyIdentityDisposition).where(
            LegacyIdentityDisposition.taxonomy_version_id == taxonomy.id
        )
    ).all()
    destinations = db.scalars(
        select(LegacyDestinationMapping).where(
            LegacyDestinationMapping.taxonomy_version_id == taxonomy.id
        )
    ).all()
    allocations = db.scalars(
        select(LegacyClaimAllocation).where(
            LegacyClaimAllocation.taxonomy_version_id == taxonomy.id
        )
    ).all()
    redirects = db.scalars(
        select(EconomicThemeRedirect).where(
            EconomicThemeRedirect.taxonomy_version_id == taxonomy.id
        )
    ).all()
    metric_rows = db.scalars(
        select(ThemeMetric).where(ThemeMetric.metrics_revision_id == metrics.id)
    ).all()

    dimension_keys = {row.key for row in dimensions}
    value_keys = {(row.dimension_key, row.normalized_value) for row in values}
    referenced_theme_ids = {
        row.theme_id for row in aliases
    } | {row.theme_id for row in facets}
    referenced_theme_ids |= {
        endpoint
        for row in relationships
        for endpoint in (row.source_theme_id, row.target_theme_id)
    }
    referenced_theme_ids |= {
        row.destination_theme_id for row in destinations
    }
    referenced_theme_ids |= {
        row.destination_theme_id
        for row in allocations
        if row.destination_theme_id is not None
    }
    referenced_theme_ids |= {
        endpoint
        for row in redirects
        for endpoint in (row.source_theme_id, row.target_theme_id)
    }
    referenced_theme_ids |= {row.economic_theme_id for row in metric_rows}
    if not referenced_theme_ids.issubset(theme_ids):
        raise SnapshotBundleError("snapshot_reference_not_in_taxonomy")
    if any(row.dimension_key not in dimension_keys for row in facets):
        raise SnapshotBundleError("snapshot_facet_dimension_missing")
    if any(
        (row.dimension_key, row.normalized_value) not in value_keys for row in facets
    ):
        raise SnapshotBundleError("snapshot_facet_value_missing")

    assignments = db.scalars(
        select(ClaimAssignment)
        .join(
            InterpretationSelection,
            InterpretationSelection.selected_classification_attempt_id
            == ClaimAssignment.classification_attempt_id,
        )
        .where(
            InterpretationSelection.interpretation_set_id == interpretation.id
        )
    ).all()
    assignment_by_id = {row.id: row for row in assignments}
    assignment_ids = tuple(assignment_by_id)
    observation_rows = [
        observation
        for observation, _assignment in observation_rows_for_interpretation(
            db,
            interpretation_set_id=interpretation.id,
            manifest_id=manifest.id,
        )
    ]
    constituent_rows = (
        db.scalars(
            select(ThemeConstituentExposure).where(
                ThemeConstituentExposure.claim_assignment_id.in_(assignment_ids)
            )
        ).all()
        if assignment_ids
        else []
    )
    signal_rows = [
        signal
        for signal, _assignment in signal_rows_for_interpretation(
            db,
            interpretation_set_id=interpretation.id,
            manifest_id=manifest.id,
        )
    ]
    if any(row.economic_theme_id not in theme_ids for row in assignments):
        raise SnapshotBundleError("snapshot_reference_not_in_taxonomy")

    assignment_family = (
        dict(
            db.execute(
                select(ClaimAssignment.id, SourceLineage.source_family_id)
                .join(
                    ClassificationAttempt,
                    ClassificationAttempt.id
                    == ClaimAssignment.classification_attempt_id,
                )
                .join(
                    ProcessingRequest,
                    ProcessingRequest.id
                    == ClassificationAttempt.processing_request_id,
                )
                .join(
                    SourceLineage,
                    SourceLineage.id == ProcessingRequest.source_lineage_id,
                )
                .where(ClaimAssignment.id.in_(assignment_ids))
            ).tuples().all()
        )
        if assignment_ids
        else {}
    )
    security_ids = {
        row.security_id for row in constituent_rows if row.security_id is not None
    } | {row.security_id for row in signal_rows if row.security_id is not None}
    securities = (
        {
            row.id: row
            for row in db.scalars(
                select(StockUniverse).where(StockUniverse.id.in_(security_ids))
            )
        }
        if security_ids
        else {}
    )

    development_ids = _pinned_development_observation_ids(db, manifest)
    development_rows = _development_rows(
        db,
        observation_ids=development_ids,
        taxonomy_version_id=taxonomy.id,
    )
    if any(theme_id not in theme_ids for _, theme_id in development_rows):
        raise SnapshotBundleError("snapshot_reference_not_in_taxonomy")

    social_by_theme = _pinned_social_memberships(db, interpretation.id)
    if any(theme_id not in theme_ids for theme_id in social_by_theme):
        raise SnapshotBundleError("snapshot_reference_not_in_taxonomy")

    themes = []
    for revision in revisions:
        theme_aliases = sorted(
            row.alias for row in aliases if row.theme_id == revision.theme_id
        )
        theme_facets = sorted(
            (
                {
                    "dimension": row.dimension_key,
                    "value": row.normalized_value,
                    "display_value": next(
                        value.display_value
                        for value in values
                        if value.dimension_key == row.dimension_key
                        and value.normalized_value == row.normalized_value
                    ),
                }
                for row in facets
                if row.theme_id == revision.theme_id
            ),
            key=lambda value: (value["dimension"], value["value"]),
        )
        theme_metrics = {}
        for view in _ECONOMIC_RANKING_VIEWS:
            row = next(
                (
                    value
                    for value in metric_rows
                    if value.economic_theme_id == revision.theme_id
                    and value.ranking_view == view
                ),
                None,
            )
            theme_metrics[view] = (
                {
                    "availability": (row.components or {}).get(
                        "availability", "available" if row.available else "unavailable"
                    ),
                    "raw_value": row.raw_value,
                    "percentile": row.percentile,
                    "components": dict(row.components or {}),
                }
                if row is not None
                else {
                    "availability": "unavailable",
                    "raw_value": None,
                    "percentile": None,
                    "components": {"reason": "not_computed"},
                }
            )
        theme_observations = [
            row
            for row in observation_rows
            if revision.theme_id
            in {
                UUID(value)
                for value in row.payload.get("economic_theme_ids", [])
            }
            or assignment_by_id[row.claim_assignment_id].economic_theme_id
            == revision.theme_id
        ]
        source_family_ids = {
            assignment_family.get(row.claim_assignment_id)
            for row in theme_observations
            if assignment_family.get(row.claim_assignment_id) is not None
        }
        themes.append(
            {
                "economic_theme_id": str(revision.theme_id),
                "display_name": revision.display_name,
                "definition": revision.definition,
                "mechanism": revision.mechanism,
                "lifecycle": revision.lifecycle,
                "aliases": theme_aliases,
                "facets": theme_facets,
                "metrics": theme_metrics,
                "direct_observation_count": len(
                    {row.id for row in theme_observations if row.observation_kind == "primary"}
                ),
                "derived_observation_count": len(
                    {row.id for row in theme_observations if row.observation_kind != "primary"}
                ),
                "deduplicated_source_family_count": len(source_family_ids),
                "signals": [
                    {
                        "signal_id": str(row.id),
                        "security_id": row.security_id,
                        "canonical_symbol": (
                            securities[row.security_id].symbol
                            if row.security_id in securities
                            else None
                        ),
                        "market": (
                            securities[row.security_id].market
                            if row.security_id in securities
                            else None
                        ),
                        "signal_kind": row.signal_kind,
                        "available_at": row.available_at.isoformat(),
                        "payload": dict(row.payload),
                    }
                    for row in signal_rows
                    if assignment_by_id[row.claim_assignment_id].economic_theme_id
                    == revision.theme_id
                ],
                "constituents": [
                    {
                        "exposure_id": str(row.id),
                        "security_id": row.security_id,
                        "canonical_symbol": (
                            securities[row.security_id].symbol
                            if row.security_id in securities
                            else None
                        ),
                        "market": (
                            securities[row.security_id].market
                            if row.security_id in securities
                            else None
                        ),
                        "exposure_kind": row.exposure_kind,
                        "exposure_strength": row.exposure_strength,
                        "payload": dict(row.payload),
                    }
                    for row in constituent_rows
                    if assignment_by_id[row.claim_assignment_id].economic_theme_id
                    == revision.theme_id
                ],
                "developments": [
                    {
                        "observation_id": row.id,
                        "analysis_channel": row.analysis_channel,
                        "development_support": row.development_support,
                        "classification": row.classification,
                        "facts": dict(row.facts),
                        "citations": list(row.citations),
                    }
                    for row, theme_id in development_rows
                    if theme_id == revision.theme_id
                ],
                "relationships": [
                    _relationship_payload(row)
                    for row in relationships
                    if revision.theme_id
                    in {row.source_theme_id, row.target_theme_id}
                ],
                "mappings": [
                    _mapping_payload(row)
                    for row in destinations
                    if row.destination_theme_id == revision.theme_id
                ],
                "social_memberships": social_by_theme.get(revision.theme_id, []),
                "reconciliation_state": (
                    "conflict_review_required"
                    if any(
                        item["state"] == "conflict_review_required"
                        for item in social_by_theme.get(revision.theme_id, [])
                    )
                    else "reconciled"
                    if social_by_theme.get(revision.theme_id)
                    else "not_applicable"
                ),
            }
        )

    relationship_payloads = sorted(
        (_relationship_payload(row) for row in relationships),
        key=lambda row: (
            row["source_theme_id"],
            row["target_theme_id"],
            row["kind"],
        ),
    )
    mapping_payloads = sorted(
        (_mapping_payload(row) for row in destinations),
        key=lambda row: (
            row["legacy_theme_cluster_id"],
            row["destination_theme_id"],
        ),
    )
    base = {
        "taxonomy_version_id": str(taxonomy.id),
        "taxonomy_semantic_hash": taxonomy.semantic_hash,
        "interpretation_set_id": str(interpretation.id),
        "generation_input_manifest_id": str(manifest.id),
        "generation_input_manifest_hash": manifest.semantic_hash,
        "metrics_revision_id": str(metrics.id),
        "metrics_formula_version": metrics.formula_version,
        "metrics_as_of": metrics.as_of.isoformat(),
        "pinned_revisions": list(manifest.selections or []),
        "committed_revision_tuples": list(
            manifest.committed_revision_tuples or []
        ),
        "relationships": relationship_payloads,
        "mappings": mapping_payloads,
        "redirects": [
            {
                "source_theme_id": str(row.source_theme_id),
                "target_theme_id": str(row.target_theme_id),
                "reason": row.reason,
            }
            for row in redirects
        ],
    }
    catalog = {**base, "themes": themes}
    review = {
        **base,
        "dispositions": [
            {
                "legacy_theme_cluster_id": row.legacy_theme_cluster_id,
                "disposition": row.disposition,
                "review_comment": row.review_comment,
            }
            for row in dispositions
        ],
        "allocations": [
            {
                "legacy_theme_cluster_id": row.legacy_theme_cluster_id,
                "allocation_kind": row.allocation_kind,
                "allocation_key": row.allocation_key,
                "destination_theme_id": (
                    str(row.destination_theme_id)
                    if row.destination_theme_id is not None
                    else None
                ),
                "reviewed_exclusion": row.reviewed_exclusion,
            }
            for row in allocations
        ],
        "reconciliation": [
            {
                "economic_theme_id": str(theme_id),
                "memberships": rows,
            }
            for theme_id, rows in sorted(
                social_by_theme.items(), key=lambda value: str(value[0])
            )
        ],
        "operation_previews": _operation_preview_payloads(db, taxonomy.id),
    }
    return catalog, review


def _relationship_payload(row):
    return {
        "id": str(row.id),
        "source_theme_id": str(row.source_theme_id),
        "target_theme_id": str(row.target_theme_id),
        "kind": row.kind,
        "direction": row.direction,
        "discriminator": row.discriminator,
    }


def _mapping_payload(row):
    return {
        "legacy_theme_cluster_id": row.legacy_theme_cluster_id,
        "destination_theme_id": str(row.destination_theme_id),
        "review_comment": row.review_comment,
    }


def _pinned_development_observation_ids(db, manifest):
    observation_ids = set()
    for raw in manifest.selections or []:
        selections = raw.get("development_selections")
        if selections is None:
            selections = (
                [
                    {
                        "development_identity": raw.get("development_identity"),
                        "revision_number": raw.get("development_revision"),
                    }
                ]
                if raw.get("development_revision") is not None
                else []
            )
        if not isinstance(selections, list):
            raise SnapshotBundleError("invalid_development_selections")
        for selection in selections:
            try:
                identity = UUID(str(selection["development_identity"]))
                revision_number = int(selection["revision_number"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SnapshotBundleError("invalid_development_selection") from exc
            revision = db.scalar(
                select(DevelopmentSelectionRevision).where(
                    DevelopmentSelectionRevision.development_identity == identity,
                    DevelopmentSelectionRevision.revision_number == revision_number,
                )
            )
            if revision is None:
                raise SnapshotBundleError("development_revision_not_pinned")
            observation_ids.update(revision.payload.get("observation_ids", []))
    return observation_ids


def _development_rows(db, *, observation_ids, taxonomy_version_id):
    """Pinned observations with their economic theme links (#513).

    Legacy links arrive as ``legacy_mapping`` rows from the version's backfill
    (``economic_development_backfill``). An observation recorded by the legacy
    producer (no source family) that the backfill has not covered fails the
    build rather than losing its themes.
    """
    if not observation_ids:
        return []
    ids = {int(value) for value in observation_ids}
    observations = {
        row.id: row
        for row in db.scalars(
            select(ThemeDevelopmentObservation).where(
                ThemeDevelopmentObservation.id.in_(ids)
            )
        )
    }
    backfill = db.get(EconomicDevelopmentBackfill, taxonomy_version_id)
    covered = backfill.through_observation_id if backfill is not None else 0
    if any(
        row.source_family_id is None and row.id > covered
        for row in observations.values()
    ):
        raise SnapshotBundleError("legacy_development_backfill_missing")
    links = select(EconomicThemeDevelopment).where(
        EconomicThemeDevelopment.observation_id.in_(ids)
    )
    if backfill is None:
        # The legacy_mapping rows are another version's.
        links = links.where(EconomicThemeDevelopment.link_origin != "legacy_mapping")
    theme_links = {
        (row.observation_id, row.economic_theme_id) for row in db.scalars(links)
    }
    return [
        (observations[observation_id], theme_id)
        for observation_id, theme_id in sorted(
            theme_links,
            key=lambda value: (value[0], str(value[1])),
        )
        if observation_id in observations
    ]


def _pinned_social_memberships(db, interpretation_set_id):
    interpretation = db.get(InterpretationSet, interpretation_set_id)
    manifest = (
        db.get(GenerationInputManifest, interpretation.generation_input_manifest_id)
        if interpretation is not None
        else None
    )
    if manifest is None:
        raise SnapshotBundleError("generation_manifest_missing")
    try:
        contracts = manifest_social_revision_contracts(manifest.selections or [])
    except ValueError as exc:
        raise SnapshotBundleError(str(exc)) from exc
    refs_by_identity = {
        (ref.association_id, ref.revision_number): ref
        for ref in db.scalars(
            select(SocialAssociationRevisionRef)
            .join(
                InterpretationSelection,
                InterpretationSelection.social_association_revision_ref_id
                == SocialAssociationRevisionRef.id,
            )
            .where(
                InterpretationSelection.interpretation_set_id
                == interpretation_set_id
            )
        )
    }
    for association_id, revision_number in contracts:
        ref = db.scalar(
            select(SocialAssociationRevisionRef).where(
                SocialAssociationRevisionRef.association_id == association_id,
                SocialAssociationRevisionRef.revision_number == revision_number,
            )
        )
        if ref is None:
            raise SnapshotBundleError("social_revision_not_pinned")
        refs_by_identity[(association_id, revision_number)] = ref
    refs = refs_by_identity.values()
    grouped = {}
    for ref in refs:
        association = db.get(EconomicSocialAssociation, ref.association_id)
        revision = db.scalar(
            select(EconomicSocialAssociationRevision).where(
                EconomicSocialAssociationRevision.association_id
                == ref.association_id,
                EconomicSocialAssociationRevision.revision_number
                == ref.revision_number,
            )
        )
        if association is None or revision is None:
            raise SnapshotBundleError("social_revision_not_pinned")
        security = db.get(StockUniverse, association.security_id)
        grouped.setdefault(association.economic_theme_id, []).append(
            {
                "association_revision_ref_id": str(ref.id),
                "association_id": str(association.id),
                "association_revision_id": str(revision.id),
                "revision_number": revision.revision_number,
                "decision_revision_id": (
                    str(revision.decision_revision_id)
                    if revision.decision_revision_id
                    else None
                ),
                "security_id": association.security_id,
                "canonical_symbol": security.symbol if security is not None else None,
                "market": security.market if security is not None else None,
                "state": revision.state,
                "live": revision.live,
                "admission_state": revision.admission_state,
                "mirror_state": revision.mirror_state,
            }
        )
    return grouped


def _operation_preview_payloads(db, current_taxonomy_version_id):
    rows = db.execute(
        select(TaxonomyOperationRequest, TaxonomyOperationPreview).join(
            TaxonomyOperationPreview,
            TaxonomyOperationPreview.operation_request_id
            == TaxonomyOperationRequest.id,
        )
    ).all()
    payloads = []
    for request, preview in rows:
        latest_event = db.scalar(
            select(TaxonomyOperationEvent)
            .where(TaxonomyOperationEvent.operation_request_id == request.id)
            .order_by(TaxonomyOperationEvent.sequence_number.desc())
            .limit(1)
        )
        payloads.append(
            {
                "operation_request_id": str(request.id),
                "operation_kind": request.operation_kind,
                "base_taxonomy_version_id": str(request.base_taxonomy_version_id),
                "candidate_taxonomy_version_id": str(
                    preview.candidate_taxonomy_version_id
                ),
                "preview_hash": preview.preview_hash,
                "affected_identities": deepcopy(preview.affected_identities or []),
                "assignments": deepcopy(preview.assignments or []),
                "mappings": deepcopy(preview.mappings or []),
                "validation_errors": deepcopy(preview.validation_errors or []),
                "stale": request.base_taxonomy_version_id
                != current_taxonomy_version_id,
                "status": latest_event.event_type if latest_event else "previewed",
                "reviewer_reason": latest_event.reason if latest_event else None,
            }
        )
    return sorted(payloads, key=lambda row: row["operation_request_id"])

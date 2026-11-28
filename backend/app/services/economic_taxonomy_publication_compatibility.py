"""Compatibility projection construction for Economic Taxonomy publication."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, aliased

from app.infra.db.repositories.economic_taxonomy_publication_repo import (
    PublicationInvariantError,
)
from app.models.economic_taxonomy import EconomicThemeRevision
from app.models.economic_taxonomy_runtime import (
    ClaimAssignment,
    GenerationInputManifest,
    InterpretationSelection,
    ServingGeneration,
    ServingGenerationEvent,
    TaxonomyProjectionEvent,
    ThemeConstituentExposure,
)
from app.models.stock_universe import StockUniverse
from app.services.economic_taxonomy_publication_contracts import PreparationContext
from app.utils.file_hashing import canonical_json_sha256 as _hash


@dataclass(frozen=True, slots=True)
class CompatibilityProjection:
    source_lineage: str
    projection_kind: str
    projection_version: int
    target: str
    payload: Mapping[str, Any]
    origin_representation: str
    selected_interpretation_version: str
    mapping_version: str
    projection_revision: int | None = None


def assign_projection_revisions(
    session: Session,
    projections: Sequence[CompatibilityProjection],
) -> list[CompatibilityProjection]:
    assigned: list[CompatibilityProjection] = []
    seen: set[tuple[str, str, str]] = set()
    for projection in sorted(
        projections,
        key=lambda value: (
            value.source_lineage,
            value.projection_kind,
            value.target,
        ),
    ):
        key = (
            projection.source_lineage,
            projection.projection_kind,
            projection.target,
        )
        if key in seen:
            raise PublicationInvariantError("duplicate_candidate_projection")
        seen.add(key)
        revision = projection.projection_revision
        if revision is None:
            latest = session.scalar(
                select(func.max(TaxonomyProjectionEvent.projection_revision)).where(
                    TaxonomyProjectionEvent.source_lineage == projection.source_lineage,
                    TaxonomyProjectionEvent.projection_kind
                    == projection.projection_kind,
                    TaxonomyProjectionEvent.target_representation == projection.target,
                )
            )
            revision = int(latest or 0) + 1
        assigned.append(replace(projection, projection_revision=revision))
    return assigned


def projection_descriptor(projection: CompatibilityProjection) -> dict[str, Any]:
    payload = {
        **dict(projection.payload),
        "origin_representation": projection.origin_representation,
        "selected_interpretation_version": (projection.selected_interpretation_version),
        "mapping_version": projection.mapping_version,
    }
    return {
        "source_lineage": projection.source_lineage,
        "projection_revision": projection.projection_revision,
        "projection_kind": projection.projection_kind,
        "projection_version": projection.projection_version,
        "target": projection.target,
        "payload_hash": _hash(payload),
    }


def event_descriptor(event: TaxonomyProjectionEvent) -> dict[str, Any]:
    return {
        "source_lineage": event.source_lineage,
        "projection_revision": event.projection_revision,
        "projection_kind": event.projection_kind,
        "projection_version": event.projection_version,
        "target": event.target_representation,
        "payload_hash": event.payload_hash,
    }


def _generation_modes():
    generation = aliased(ServingGeneration)
    prepared = aliased(ServingGenerationEvent)
    rows = select(
        generation.id,
        generation.expected_parent_generation_id.label("parent_id"),
        prepared.details["target_mode"].as_string().label("mode"),
    ).join(
        prepared,
        and_(
            prepared.serving_generation_id == generation.id,
            prepared.event_type == "prepared",
        ),
    )
    return generation, rows


def live_ancestor_generation_id(session: Session, generation_id):
    """Return the nearest ancestor-or-self generation that was not shadow.

    Shadow generations stage only shadow-scope projections, so the lineages
    legacy readers last received live on the generation before them.
    """
    start, rows = _generation_modes()
    chain = rows.where(start.id == generation_id).cte("live_ancestor", recursive=True)
    parent, rows = _generation_modes()
    chain = chain.union(
        rows.join(chain, parent.id == chain.c.parent_id).where(
            chain.c.mode == "shadow"
        )
    )
    return session.scalar(
        select(chain.c.id).where(chain.c.mode.is_distinct_from("shadow"))
    )


def build_default_compatibility_projections(
    session: Session,
    context: PreparationContext,
) -> Sequence[CompatibilityProjection]:
    """Build a complete candidate projection, carrying parent-owned lineages."""

    manifest = session.get(GenerationInputManifest, context.manifest_id)
    current: dict[str, dict[str, set[Any]]] = {}
    theme_identity_by_string: dict[str, Any] = {}
    for lineage_id, theme_id, assignment_id in session.execute(
        select(
            InterpretationSelection.source_lineage_id,
            ClaimAssignment.economic_theme_id,
            ClaimAssignment.id,
        )
        .join(
            ClaimAssignment,
            ClaimAssignment.classification_attempt_id
            == InterpretationSelection.selected_classification_attempt_id,
        )
        .where(
            InterpretationSelection.interpretation_set_id
            == context.interpretation_set_id
        )
    ):
        theme_key = str(theme_id)
        theme_identity_by_string[theme_key] = theme_id
        current.setdefault(str(lineage_id), {}).setdefault(theme_key, set()).add(
            assignment_id
        )
    for value in manifest.selections or []:
        if value.get("lineage"):
            current.setdefault(str(value["lineage"]), {})

    assignment_ids = {
        assignment_id
        for themes in current.values()
        for ids in themes.values()
        for assignment_id in ids
    }
    symbols_by_assignment: dict[Any, set[str]] = defaultdict(set)
    if assignment_ids:
        for assignment_id, symbol in session.execute(
            select(
                ThemeConstituentExposure.claim_assignment_id,
                StockUniverse.symbol,
            )
            .join(
                StockUniverse,
                StockUniverse.id == ThemeConstituentExposure.security_id,
            )
            .where(ThemeConstituentExposure.claim_assignment_id.in_(assignment_ids))
        ):
            symbols_by_assignment[assignment_id].add(str(symbol))

    theme_ids = set(theme_identity_by_string.values())
    revisions = (
        {
            str(row.theme_id): row
            for row in session.scalars(
                select(EconomicThemeRevision).where(
                    EconomicThemeRevision.taxonomy_version_id
                    == context.taxonomy_version_id,
                    EconomicThemeRevision.theme_id.in_(theme_ids),
                )
            )
        }
        if theme_ids
        else {}
    )

    projections: dict[tuple[str, str, str], CompatibilityProjection] = {}
    previous_theme_lineages: set[str] = set()
    parent_id = live_ancestor_generation_id(
        session, context.expected_parent_generation_id
    )
    if parent_id is not None:
        parent_rows = session.scalars(
            select(TaxonomyProjectionEvent)
            .where(
                TaxonomyProjectionEvent.serving_generation_id == parent_id,
                TaxonomyProjectionEvent.delivery_scope == "candidate",
            )
            .order_by(TaxonomyProjectionEvent.projection_revision)
        ).all()
        for event in parent_rows:
            if event.projection_kind == "social_membership":
                # Social mirrors are one-time side effects. Preparation requires the
                # parent generation to be fully acknowledged, so cloning them would
                # create a new event with no pending association revision to mirror.
                continue
            key = (
                event.source_lineage,
                event.projection_kind,
                event.target_representation,
            )
            if (
                event.projection_kind == "legacy_theme"
                and event.target_representation == "legacy"
            ):
                previous_theme_lineages.add(event.source_lineage)
                continue
            payload = dict(event.payload)
            selected_version = str(
                payload.pop(
                    "selected_interpretation_version",
                    context.interpretation_set_id,
                )
            )
            mapping_version = str(
                payload.pop("mapping_version", context.taxonomy_version_id)
            )
            origin = str(
                payload.pop("origin_representation", event.origin_representation)
            )
            projections[key] = CompatibilityProjection(
                source_lineage=event.source_lineage,
                projection_kind=event.projection_kind,
                projection_version=event.projection_version,
                target=event.target_representation,
                payload=payload,
                origin_representation=origin,
                selected_interpretation_version=selected_version,
                mapping_version=mapping_version,
            )

    for lineage in sorted(set(current) | previous_theme_lineages):
        key = (lineage, "legacy_theme", "legacy")
        themes = current.get(lineage, {})
        details = []
        for theme_id, theme_assignment_ids in sorted(themes.items()):
            revision = revisions.get(theme_id)
            details.append(
                {
                    "economic_theme_id": theme_id,
                    "display_name": (
                        revision.display_name
                        if revision is not None
                        else f"Economic Theme {theme_id}"
                    ),
                    "definition": revision.definition if revision is not None else None,
                    "lifecycle": (
                        revision.lifecycle if revision is not None else "provisional"
                    ),
                    "constituents": sorted(
                        {
                            symbol
                            for assignment_id in theme_assignment_ids
                            for symbol in symbols_by_assignment.get(assignment_id, ())
                        }
                    ),
                }
            )
        projections[key] = CompatibilityProjection(
            source_lineage=lineage,
            projection_kind="legacy_theme",
            projection_version=1,
            target="legacy",
            payload={"themes": sorted(themes), "theme_details": details},
            origin_representation="economic",
            selected_interpretation_version=str(context.interpretation_set_id),
            mapping_version=str(context.taxonomy_version_id),
        )
    return list(projections.values())

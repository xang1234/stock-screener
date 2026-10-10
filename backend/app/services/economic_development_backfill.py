"""Legacy development links as economic rows, one taxonomy version at a time (#513).

The snapshot builder reads only ``EconomicThemeDevelopment``. Legacy
``ThemeDevelopmentTheme`` links reach it as ``legacy_mapping`` rows written
here with the version's sealed allocations and destinations; the
``EconomicDevelopmentBackfill`` marker says which version they are for. At
rollback, economic-window links are projected back onto legacy themes.
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, func, select, tuple_

from app.models.economic_taxonomy import (
    LegacyClaimAllocation,
    LegacyDestinationMapping,
    TaxonomyVersion,
)
from app.models.economic_taxonomy_runtime import TaxonomyAuthority
from app.models.theme import ThemeCluster
from app.services.economic_taxonomy_fence import exclusive_publication
from app.models.theme_intelligence import (
    EconomicDevelopmentBackfill,
    EconomicThemeDevelopment,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)

logger = logging.getLogger(__name__)


def map_legacy_links(links, *, allocations, destinations):
    """``{(observation_id, economic_theme_id)}`` for legacy ``(observation_id, theme_id)`` links.

    A reviewed allocation wins (a reviewed exclusion maps nothing); otherwise a
    legacy theme with one destination maps to it, and one split across several
    destinations needs an allocation.
    """
    allocation_by_claim = {
        (row.legacy_theme_cluster_id, row.allocation_kind, row.allocation_key): row
        for row in allocations
    }
    destinations_by_legacy = {}
    for row in destinations:
        destinations_by_legacy.setdefault(row.legacy_theme_cluster_id, []).append(
            row.destination_theme_id
        )
    mapped = set()
    for observation_id, theme_id in links:
        allocation = allocation_by_claim.get(
            (theme_id, "development", f"theme_development_observation:{observation_id}")
        )
        if allocation is not None:
            if allocation.destination_theme_id is not None:
                mapped.add((observation_id, allocation.destination_theme_id))
            continue
        targets = destinations_by_legacy.get(theme_id, [])
        if len(targets) == 1:
            mapped.add((observation_id, targets[0]))
        elif len(targets) > 1:
            raise ValueError("legacy_development_allocation_missing")
    return mapped


def _version_rows(db, model, taxonomy_version_id):
    return db.scalars(select(model).where(model.taxonomy_version_id == taxonomy_version_id)).all()


def backfill_legacy_developments(db, taxonomy_version_id=None):
    """Write the version's ``legacy_mapping`` rows, replacing any other version's.

    Defaults to the processing version. Skips when the marker already covers
    the current legacy links (their count and highest observation).

    Only links on legacy-producer observations (no source family) are mapped:
    economic observations carry native links, and their legacy links are the
    rollback projection of those (which may land on a split legacy theme).
    """
    follow_processing = taxonomy_version_id is None
    if follow_processing:
        authority = db.get(TaxonomyAuthority, 1)
        taxonomy_version_id = authority.processing_taxonomy_version_id if authority else None
    version = db.get(TaxonomyVersion, taxonomy_version_id) if taxonomy_version_id else None
    if version is None or version.status != "sealed":
        # A draft's allocations can still change under an unchanged fingerprint.
        return {"status": "no_sealed_taxonomy"}
    if _is_current(db, version.id):
        return {"status": "current", "taxonomy_version_id": str(version.id)}
    # Drains legacy writers (they hold the shared producer fence), so every
    # observation at or below the watermark has committed its links; it also
    # serializes backfills.
    with exclusive_publication(db) as locked:
        db.refresh(locked)
        if follow_processing and locked.processing_taxonomy_version_id != version.id:
            # The processor moved on while this waited; the next run maps the
            # new version instead of replacing its rows with obsolete ones.
            return {"status": "processing_version_changed"}
        if _is_current(db, version.id):
            return {"status": "current", "taxonomy_version_id": str(version.id)}
        through = db.scalar(select(func.max(ThemeDevelopmentObservation.id))) or 0
        count, max_observation = _fingerprint(db)
        mapped = map_legacy_links(
            db.execute(
                _on_legacy_observations(
                    select(ThemeDevelopmentTheme.observation_id, ThemeDevelopmentTheme.theme_id),
                    ThemeDevelopmentTheme.observation_id,
                )
            ),
            allocations=_version_rows(db, LegacyClaimAllocation, version.id),
            destinations=_version_rows(db, LegacyDestinationMapping, version.id),
        )
        current, other = set(), set()
        for observation_id, theme_id, origin in db.execute(
            _on_legacy_observations(
                select(
                    EconomicThemeDevelopment.observation_id,
                    EconomicThemeDevelopment.economic_theme_id,
                    EconomicThemeDevelopment.link_origin,
                ),
                EconomicThemeDevelopment.observation_id,
            )
        ):
            (current if origin == "legacy_mapping" else other).add((observation_id, theme_id))
        # Apply the difference: a full rewrite each minute would bloat the table.
        stale = current - mapped
        if stale:
            db.execute(
                delete(EconomicThemeDevelopment).where(
                    EconomicThemeDevelopment.link_origin == "legacy_mapping",
                    tuple_(
                        EconomicThemeDevelopment.observation_id,
                        EconomicThemeDevelopment.economic_theme_id,
                    ).in_(stale),
                )
            )
        added = mapped - current - other
        db.add_all(
            EconomicThemeDevelopment(
                observation_id=observation_id,
                economic_theme_id=theme_id,
                link_origin="legacy_mapping",
            )
            for observation_id, theme_id in added
        )
        _write_marker(db, version.id, through, count, max_observation)
    return {
        "status": "backfilled",
        "taxonomy_version_id": str(version.id),
        "added": len(added),
        "removed": len(stale),
    }


def _on_legacy_observations(query, observation_column):
    return query.join(
        ThemeDevelopmentObservation, ThemeDevelopmentObservation.id == observation_column
    ).where(ThemeDevelopmentObservation.source_family_id.is_(None))


def _fingerprint(db):
    """``(legacy link count, highest legacy observation)``.

    The observation is any legacy one, linked or not: the builder checks every
    pinned legacy observation against the watermark.
    """
    count = db.scalar(
        _on_legacy_observations(
            select(func.count()).select_from(ThemeDevelopmentTheme),
            ThemeDevelopmentTheme.observation_id,
        )
    )
    highest = db.scalar(
        select(func.max(ThemeDevelopmentObservation.id)).where(
            ThemeDevelopmentObservation.source_family_id.is_(None)
        )
    )
    return count, highest


def _is_current(db, taxonomy_version_id):
    marker = db.scalar(
        select(EconomicDevelopmentBackfill).where(
            EconomicDevelopmentBackfill.taxonomy_version_id == taxonomy_version_id
        )
    )
    count, max_observation = _fingerprint(db)
    return (
        marker is not None
        and marker.legacy_link_count == count
        and marker.legacy_max_observation_id == max_observation
    )


def _write_marker(db, taxonomy_version_id, through, count, max_observation):
    db.execute(delete(EconomicDevelopmentBackfill))
    db.add(
        EconomicDevelopmentBackfill(
            taxonomy_version_id=taxonomy_version_id,
            through_observation_id=through,
            legacy_link_count=count,
            legacy_max_observation_id=max_observation,
        )
    )
    db.flush()


def project_economic_developments(db, taxonomy_version_id):
    """Legacy links for developments recorded under economic authority (rollback).

    An economic theme gets a legacy link only when exactly one of the version's
    legacy destinations for it is in the observation's pipeline; ambiguous and
    unmapped ones are skipped and counted.
    """
    legacy_by_theme = {}
    for row in _version_rows(db, LegacyDestinationMapping, taxonomy_version_id):
        legacy_by_theme.setdefault(row.destination_theme_id, set()).add(row.legacy_theme_cluster_id)
    pipelines = dict(
        db.execute(
            select(ThemeCluster.id, ThemeCluster.pipeline).where(
                ThemeCluster.id.in_(set().union(*legacy_by_theme.values()))
            )
        ).all()
    )
    existing = set(db.execute(select(ThemeDevelopmentTheme.observation_id, ThemeDevelopmentTheme.theme_id)).all())
    projected, skipped = set(), 0
    for observation_id, theme_id, pipeline in db.execute(
        select(
            EconomicThemeDevelopment.observation_id,
            EconomicThemeDevelopment.economic_theme_id,
            ThemeDevelopmentObservation.pipeline,
        )
        .join(
            ThemeDevelopmentObservation,
            ThemeDevelopmentObservation.id == EconomicThemeDevelopment.observation_id,
        )
        .where(EconomicThemeDevelopment.link_origin == "economic_native")
    ):
        candidates = [
            legacy_id
            for legacy_id in legacy_by_theme.get(theme_id, ())
            if pipelines.get(legacy_id) == pipeline
        ]
        if len(candidates) != 1:
            skipped += 1
            continue
        projected.add((observation_id, candidates[0]))
    new = projected - existing
    db.add_all(
        ThemeDevelopmentTheme(observation_id=observation_id, theme_id=theme_id)
        for observation_id, theme_id in new
    )
    db.flush()
    if skipped:
        logger.warning("rollback projection skipped %d unmapped or ambiguous development links", skipped)
    return {"projected": len(new), "skipped": skipped}

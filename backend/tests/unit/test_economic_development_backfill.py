"""Legacy development links reach the snapshot as economic rows (#513)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy_runtime import SourceFamily
from app.models.theme import ContentItem, ThemeCluster
from app.models.theme_intelligence import (
    EconomicDevelopmentBackfill,
    EconomicThemeDevelopment,
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)
from app.services.economic_development_backfill import (
    backfill_legacy_developments,
    project_economic_developments,
)
from app.services.economic_taxonomy_snapshot_builder import (
    SnapshotBundleError,
    _development_rows,
)

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _legacy(db, pipeline="technical"):
    cluster = ThemeCluster(
        name=f"legacy-{uuid4().hex[:6]}",
        display_name="Copper",
        canonical_key=f"copper-{uuid4().hex[:8]}",
        pipeline=pipeline,
    )
    db.add(cluster)
    db.flush()
    return cluster


def _version(db, mapping):
    """A sealed version: ``{legacy cluster: [theme names]}`` destinations."""
    repo = EconomicTaxonomyRepository(db)
    draft = repo.create_draft(actor="test:reviewer", reason="backfill")
    themes = {}
    for cluster, names in mapping.items():
        repo.set_legacy_disposition(
            draft.id,
            cluster.id,
            disposition="mapped" if len(names) == 1 else "split_required",
            actor="test:reviewer",
        )
        for name in names:
            if name not in themes:
                themes[name] = repo.create_theme(
                    draft.id,
                    display_name=name,
                    definition=f"{name} economics",
                    mechanism=f"{name} margins",
                    lifecycle="established",
                    lifecycle_policy_version="v1",
                )
            repo.add_legacy_destination(draft.id, cluster.id, themes[name].id, actor="test:reviewer")
        if len(names) > 1:
            # A split seals only with an allocation; this one covers no real claim.
            repo.allocate_legacy_claim(
                draft.id,
                cluster.id,
                allocation_kind="development",
                allocation_key="theme_development_observation:0",
                destination_theme_id=themes[names[0]].id,
                actor="test:reviewer",
            )
    return repo.seal_draft(draft.id), themes


def _family(db):
    family = SourceFamily(provider="news", canonical_source_key=f"news:{uuid4().hex}")
    db.add(family)
    db.flush()
    return family


def _observation(db, pipeline="technical", *, family=None):
    item = ContentItem(source_type="news", external_id=uuid4().hex, content="x", published_at=NOW)
    event = ThemeDevelopmentEvent(
        pipeline=pipeline,
        event_key=uuid4().hex,
        canonical_event_key=uuid4().hex,
        development_identity=uuid4(),
        identity={},
    )
    db.add_all([item, event])
    db.flush()
    observation = ThemeDevelopmentObservation(
        event_id=event.id,
        content_item_id=item.id,
        pipeline=pipeline,
        analysis_channel=pipeline,
        development_support="present",
        revision="r1",
        observation_key=uuid4().hex,
        facts={},
        citations=[],
        classification="new",
        available_at=NOW,
        superseded=False,
        source_family_id=family,
    )
    db.add(observation)
    db.flush()
    return observation


def _economic_links(db, origin="legacy_mapping"):
    return {
        (row.observation_id, row.economic_theme_id)
        for row in db.scalars(
            select(EconomicThemeDevelopment).where(EconomicThemeDevelopment.link_origin == origin)
        )
    }


def test_backfill_maps_legacy_links_and_skips_when_unchanged(db_session):
    cluster = _legacy(db_session)
    version, themes = _version(db_session, {cluster: ["Copper Miners"]})
    observation = _observation(db_session)
    db_session.add(ThemeDevelopmentTheme(observation_id=observation.id, theme_id=cluster.id))
    db_session.flush()

    assert backfill_legacy_developments(db_session, version.id)["status"] == "backfilled"
    assert _economic_links(db_session) == {(observation.id, themes["Copper Miners"].id)}
    assert backfill_legacy_developments(db_session, version.id)["status"] == "current"

    later = _observation(db_session)
    db_session.add(ThemeDevelopmentTheme(observation_id=later.id, theme_id=cluster.id))
    db_session.flush()
    assert backfill_legacy_developments(db_session, version.id)["status"] == "backfilled"
    assert (later.id, themes["Copper Miners"].id) in _economic_links(db_session)
    assert db_session.get(EconomicDevelopmentBackfill, version.id).through_observation_id >= later.id


def test_backfill_for_a_new_version_replaces_the_old_rows(db_session):
    cluster = _legacy(db_session)
    old, old_themes = _version(db_session, {cluster: ["Copper Miners"]})
    new, new_themes = _version(db_session, {cluster: ["Copper Producers"]})
    observation = _observation(db_session)
    db_session.add(ThemeDevelopmentTheme(observation_id=observation.id, theme_id=cluster.id))
    db_session.flush()

    backfill_legacy_developments(db_session, old.id)
    backfill_legacy_developments(db_session, new.id)

    assert _economic_links(db_session) == {(observation.id, new_themes["Copper Producers"].id)}
    assert db_session.get(EconomicDevelopmentBackfill, old.id) is None
    # A build of the old version must not read the new version's rows.
    with pytest.raises(SnapshotBundleError, match="legacy_development_backfill_missing"):
        _development_rows(db_session, observation_ids=[observation.id], taxonomy_version_id=old.id)


def test_split_legacy_theme_without_allocation_fails_the_backfill(db_session):
    cluster = _legacy(db_session)
    version, _themes = _version(db_session, {cluster: ["Copper Miners", "Copper Smelters"]})
    observation = _observation(db_session)
    db_session.add(ThemeDevelopmentTheme(observation_id=observation.id, theme_id=cluster.id))
    db_session.flush()

    with pytest.raises(ValueError, match="legacy_development_allocation_missing"):
        backfill_legacy_developments(db_session, version.id)
    assert db_session.get(EconomicDevelopmentBackfill, version.id) is None


def test_builder_reads_native_links_without_a_backfill(db_session):
    cluster = _legacy(db_session)
    version, themes = _version(db_session, {cluster: ["Copper Miners"]})
    family = _family(db_session)
    native = _observation(db_session, family=family.id)
    db_session.add(
        EconomicThemeDevelopment(
            observation_id=native.id,
            economic_theme_id=themes["Copper Miners"].id,
            link_origin="economic_native",
        )
    )
    db_session.flush()

    rows = _development_rows(db_session, observation_ids=[native.id], taxonomy_version_id=version.id)

    assert [(row.id, theme_id) for row, theme_id in rows] == [(native.id, themes["Copper Miners"].id)]


def test_rollback_projects_only_unambiguous_same_pipeline_links(db_session):
    technical = _legacy(db_session, "technical")
    fundamental = _legacy(db_session, "fundamental")
    other_technical = _legacy(db_session, "technical")
    version, themes = _version(
        db_session,
        {
            technical: ["Copper Miners"],
            fundamental: ["Copper Miners"],
            other_technical: ["Copper Miners", "Gold Miners"],
        },
    )
    family = _family(db_session)
    fundamental_obs = _observation(db_session, "fundamental", family=family.id)
    technical_obs = _observation(db_session, "technical", family=family.id)
    narrative_obs = _observation(db_session, "narrative", family=family.id)
    gold_obs = _observation(db_session, "technical", family=family.id)
    for observation, theme in (
        (fundamental_obs, "Copper Miners"),
        (technical_obs, "Copper Miners"),  # two technical legacy themes: ambiguous
        (narrative_obs, "Copper Miners"),  # no narrative legacy pipeline
        (gold_obs, "Gold Miners"),
    ):
        db_session.add(
            EconomicThemeDevelopment(
                observation_id=observation.id,
                economic_theme_id=themes[theme].id,
                link_origin="economic_native",
            )
        )
    db_session.flush()

    result = project_economic_developments(db_session, version.id)

    links = set(db_session.execute(select(ThemeDevelopmentTheme.observation_id, ThemeDevelopmentTheme.theme_id)))
    assert links == {(fundamental_obs.id, fundamental.id), (gold_obs.id, other_technical.id)}
    assert result == {"projected": 2, "skipped": 2}

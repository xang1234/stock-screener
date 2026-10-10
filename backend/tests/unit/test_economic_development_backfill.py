"""Legacy development links reach the snapshot as economic rows (#513)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy_runtime import SourceFamily, TaxonomyAuthority
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


def test_a_linkless_legacy_observation_advances_the_watermark(db_session):
    # It changes no link, but the builder checks every pinned legacy
    # observation against the watermark.
    cluster = _legacy(db_session)
    version, _themes = _version(db_session, {cluster: ["Copper Miners"]})
    backfill_legacy_developments(db_session, version.id)
    linkless = _observation(db_session)

    assert backfill_legacy_developments(db_session, version.id)["status"] == "backfilled"
    assert _development_rows(
        db_session, observation_ids=[linkless.id], taxonomy_version_id=version.id
    ) == []


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
    assert result == {"projected": 2, "retracted": 0, "skipped": 2}


def test_a_rollback_projection_onto_a_split_theme_does_not_block_the_backfill(db_session):
    # Projected links sit on economic (family) observations, which carry their
    # native links already: the backfill must not map them (or need an
    # allocation for them).
    split = _legacy(db_session)
    version, themes = _version(db_session, {split: ["Copper Miners", "Gold Miners"]})
    native = _observation(db_session, family=_family(db_session).id)
    db_session.add(
        EconomicThemeDevelopment(
            observation_id=native.id,
            economic_theme_id=themes["Gold Miners"].id,
            link_origin="economic_native",
        )
    )
    db_session.flush()
    assert project_economic_developments(db_session, version.id)["projected"] == 1

    assert backfill_legacy_developments(db_session, version.id)["status"] == "backfilled"
    assert _economic_links(db_session) == set()


def test_backfill_drains_legacy_writers_before_reading_the_watermark(db_session, monkeypatch):
    import app.services.economic_development_backfill as backfill_module

    cluster = _legacy(db_session)
    version, _themes = _version(db_session, {cluster: ["Copper Miners"]})
    calls = []
    real = backfill_module.exclusive_publication

    def recording(session):
        calls.append(session)
        return real(session)

    monkeypatch.setattr(backfill_module, "exclusive_publication", recording)

    backfill_legacy_developments(db_session, version.id)

    assert calls == [db_session]


def test_builder_fails_when_the_backfill_moves_to_another_version_mid_read(db_session, monkeypatch):
    import app.services.economic_taxonomy_snapshot_builder as builder

    cluster = _legacy(db_session)
    old, _old = _version(db_session, {cluster: ["Copper Miners"]})
    new, _new = _version(db_session, {cluster: ["Copper Producers"]})
    observation = _observation(db_session)
    db_session.add(ThemeDevelopmentTheme(observation_id=observation.id, theme_id=cluster.id))
    db_session.flush()
    backfill_legacy_developments(db_session, old.id)
    real = builder._backfill_marker
    reads = []

    def switching(db, version_id):
        reads.append(version_id)
        if len(reads) == 2:  # a backfill for the new version committed in between
            backfill_legacy_developments(db, new.id)
        return real(db, version_id)

    monkeypatch.setattr(builder, "_backfill_marker", switching)

    with pytest.raises(SnapshotBundleError, match="legacy_development_backfill_changed"):
        _development_rows(db_session, observation_ids=[observation.id], taxonomy_version_id=old.id)


def test_backfill_skips_a_processing_version_replaced_while_it_waited(db_session, monkeypatch):
    import app.services.economic_development_backfill as backfill_module

    cluster = _legacy(db_session)
    old, _old = _version(db_session, {cluster: ["Copper Miners"]})
    new, _new = _version(db_session, {cluster: ["Copper Producers"]})
    db_session.add(
        TaxonomyAuthority(
            id=1,
            mode="economic",
            processing_taxonomy_version_id=old.id,
            processing_head_revision=1,
            authority_epoch=1,
            writes_fenced=False,
            rollback_state="ready",
        )
    )
    observation = _observation(db_session)
    db_session.add(ThemeDevelopmentTheme(observation_id=observation.id, theme_id=cluster.id))
    db_session.flush()
    real = backfill_module.exclusive_publication

    def processor_commits_first(session):
        # The processor moved to the new version while this run waited.
        session.get(TaxonomyAuthority, 1).processing_taxonomy_version_id = new.id
        session.flush()
        return real(session)

    monkeypatch.setattr(backfill_module, "exclusive_publication", processor_commits_first)

    assert backfill_legacy_developments(db_session)["status"] == "processing_version_changed"
    assert db_session.get(EconomicDevelopmentBackfill, old.id) is None


def test_the_backfill_task_only_follows_the_processing_version():
    # An explicit version could replace the processing version's rows with an
    # obsolete one's; only the defaulted (revalidated) path is schedulable.
    import inspect

    from app.tasks.economic_taxonomy_tasks import backfill_legacy_developments as task

    assert list(inspect.signature(task.run).parameters) == []


def test_stale_mapping_rows_are_deleted_in_chunks(db_session, monkeypatch):
    import app.services.economic_development_backfill as backfill_module

    first, second = _legacy(db_session), _legacy(db_session)
    old, _old = _version(db_session, {first: ["Copper Miners"], second: ["Gold Miners"]})
    new, new_themes = _version(db_session, {first: ["Copper Producers"], second: ["Gold Producers"]})
    observations = [_observation(db_session) for _ in range(3)]
    for observation in observations:
        for cluster in (first, second):
            db_session.add(ThemeDevelopmentTheme(observation_id=observation.id, theme_id=cluster.id))
    db_session.flush()
    backfill_legacy_developments(db_session, old.id)
    monkeypatch.setattr(backfill_module, "_DELETE_CHUNK", 2)

    backfill_legacy_developments(db_session, new.id)

    expected = {
        (observation.id, new_themes[name].id)
        for observation in observations
        for name in ("Copper Producers", "Gold Producers")
    }
    assert _economic_links(db_session) == expected


def test_a_later_rollback_retracts_links_its_version_no_longer_projects(db_session):
    # An earlier rollback projected under the old version's destinations; the
    # current version no longer maps that theme, so its link must go.
    cluster = _legacy(db_session)
    old, old_themes = _version(db_session, {cluster: ["Copper Miners"]})
    new, _new = _version(db_session, {cluster: ["Copper Producers"]})
    native = _observation(db_session, family=_family(db_session).id)
    db_session.add(
        EconomicThemeDevelopment(
            observation_id=native.id,
            economic_theme_id=old_themes["Copper Miners"].id,
            link_origin="economic_native",
        )
    )
    db_session.flush()
    project_economic_developments(db_session, old.id)

    project_economic_developments(db_session, new.id)

    assert db_session.scalars(
        select(ThemeDevelopmentTheme).where(ThemeDevelopmentTheme.observation_id == native.id)
    ).all() == []

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.infra.db.repositories.economic_taxonomy_repo import EconomicTaxonomyRepository
from app.models.economic_taxonomy import EconomicTheme
from app.models.economic_taxonomy_runtime import (
    DevelopmentSelectionRevision,
    GenerationInputManifest,
    ImmutableRuntimePayload,
    InterpretationSet,
    MetricsRevision,
    ReaderSnapshotEntry,
    ThemeMetric,
)
from app.models.theme import ContentItem, ThemeCluster
from app.models.theme_intelligence import (
    ThemeDevelopmentEvent,
    ThemeDevelopmentObservation,
    ThemeDevelopmentTheme,
)
from app.services.economic_development_backfill import backfill_legacy_developments
from app.services.economic_taxonomy_snapshot_builder import (
    GenerationSnapshotInputs,
    SnapshotBundleError,
    build_snapshot_bundle,
)

from .economic_taxonomy_reader_helpers import seed_generation


def test_build_snapshot_bundle_seals_immutable_payload_rows(db_session):
    seeded = seed_generation(db_session)
    entries = db_session.query(ReaderSnapshotEntry).filter_by(
        reader_snapshot_bundle_id=seeded["bundle"].id
    ).all()

    assert seeded["bundle"].status == "sealed"
    assert {(row.snapshot_kind, row.resource_key) for row in entries} == {
        ("economic_themes", "catalog"),
        ("economic_taxonomy", "review"),
    }
    original = entries[0].payload.copy()
    db_session.add(
        DevelopmentSelectionRevision(
            development_identity=uuid4(),
            revision_number=1,
            selected=False,
            payload={},
        )
    )
    db_session.flush()
    assert entries[0].payload == original

    entries[0].payload = {"changed": True}
    with pytest.raises(ImmutableRuntimePayload, match="sealed_payload_immutable"):
        db_session.flush()


def test_snapshot_rejects_metric_theme_outside_taxonomy(db_session):
    seeded = seed_generation(db_session)
    rogue = EconomicTheme(created_by="test:reader")
    db_session.add(rogue)
    db_session.flush()
    metrics = MetricsRevision(
        status="unsealed",
        interpretation_set_id=seeded["interpretation"].id,
        generation_input_manifest_id=seeded["manifest"].id,
        formula_version="metrics-rogue",
        as_of=datetime(2026, 9, 21, 13, 0, tzinfo=timezone.utc),
        created_by="test:reader",
    )
    db_session.add(metrics)
    db_session.flush()
    db_session.add(
        ThemeMetric(
            metrics_revision_id=metrics.id,
            economic_theme_id=rogue.id,
            ranking_view="technical_attention",
            available=False,
            components={"availability": "unavailable"},
        )
    )
    db_session.flush()
    metrics.seal(
        semantic_hash="rogue-metrics",
        artifact_integrity_hash="rogue-metrics-artifact",
    )
    db_session.flush()
    with pytest.raises(SnapshotBundleError, match="snapshot_reference_not_in_taxonomy"):
        build_snapshot_bundle(
            db_session,
            GenerationSnapshotInputs(
                taxonomy_version_id=seeded["taxonomy"].id,
                interpretation_set_id=seeded["interpretation"].id,
                generation_input_manifest_id=seeded["manifest"].id,
                metrics_revision_id=metrics.id,
                created_by="test:reader",
            ),
        )


def test_snapshot_resolves_reviewed_split_legacy_development(db_session):
    repo = EconomicTaxonomyRepository(db_session)
    taxonomy = repo.create_draft(actor="test:reviewer", reason="legacy development")
    petroleum = repo.create_theme(
        taxonomy.id,
        display_name="Petroleum Refining",
        definition="Petroleum refining economics",
        mechanism="Petroleum refining margins",
        lifecycle="established",
        lifecycle_policy_version="v1",
    )
    metals = repo.create_theme(
        taxonomy.id,
        display_name="Metals Refining",
        definition="Metals refining economics",
        mechanism="Metals refining margins",
        lifecycle="established",
        lifecycle_policy_version="v1",
    )
    legacy = ThemeCluster(
        name="Refining",
        display_name="Refining",
        canonical_key="refining",
        pipeline="fundamental",
    )
    item = ContentItem(
        source_type="news",
        external_id="legacy-refining-development",
        content="A refinery expansion was approved.",
        published_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )
    event_identity = uuid4()
    development = ThemeDevelopmentEvent(
        pipeline="fundamental",
        event_key="legacy-refining-development",
        canonical_event_key="legacy-refining-development",
        development_identity=event_identity,
        identity={"actor": "Refiner", "action": "expansion"},
    )
    db_session.add_all([legacy, item, development])
    db_session.flush()
    observation = ThemeDevelopmentObservation(
        event_id=development.id,
        content_item_id=item.id,
        pipeline="fundamental",
        analysis_channel="fundamental",
        development_support="present",
        revision="legacy-development-v1",
        observation_key="legacy-refining-observation",
        facts={"summary": "A refinery expansion was approved."},
        citations=[],
        classification="new",
        available_at=datetime(2026, 9, 20, tzinfo=timezone.utc),
        superseded=False,
    )
    db_session.add(observation)
    db_session.flush()
    db_session.add(
        ThemeDevelopmentTheme(
            observation_id=observation.id,
            theme_id=legacy.id,
        )
    )
    repo.set_legacy_disposition(
        taxonomy.id,
        legacy.id,
        disposition="split_required",
        actor="test:reviewer",
    )
    for theme in (petroleum, metals):
        repo.add_legacy_destination(
            taxonomy.id,
            legacy.id,
            theme.id,
            actor="test:reviewer",
        )
    repo.allocate_legacy_claim(
        taxonomy.id,
        legacy.id,
        allocation_kind="development",
        allocation_key=f"theme_development_observation:{observation.id}",
        destination_theme_id=petroleum.id,
        actor="test:reviewer",
    )
    taxonomy = repo.seal_draft(taxonomy.id)
    selection = DevelopmentSelectionRevision(
        development_identity=event_identity,
        revision_number=1,
        selected=True,
        payload={"observation_ids": [observation.id]},
    )
    manifest = GenerationInputManifest(
        status="unsealed",
        semantic_invalidation_revision=0,
        committed_revision_tuples=[],
        selections=[
            {
                "lineage": "legacy:refining",
                "development_selections": [
                    {
                        "development_identity": str(event_identity),
                        "revision_number": 1,
                    }
                ],
            }
        ],
        created_by="test:reviewer",
    )
    db_session.add_all([selection, manifest])
    db_session.flush()
    manifest.seal(
        semantic_hash="legacy-development-manifest",
        artifact_integrity_hash="legacy-development-manifest-artifact",
    )
    interpretation = InterpretationSet(
        status="unsealed",
        generation_input_manifest_id=manifest.id,
        created_by="test:reviewer",
    )
    db_session.add(interpretation)
    db_session.flush()
    interpretation.seal(
        semantic_hash="legacy-development-interpretation",
        artifact_integrity_hash="legacy-development-interpretation-artifact",
    )
    metrics = MetricsRevision(
        status="unsealed",
        interpretation_set_id=interpretation.id,
        generation_input_manifest_id=manifest.id,
        formula_version="metrics-v1",
        as_of=datetime(2026, 9, 21, tzinfo=timezone.utc),
        created_by="test:reviewer",
    )
    db_session.add(metrics)
    db_session.flush()
    metrics.seal(
        semantic_hash="legacy-development-metrics",
        artifact_integrity_hash="legacy-development-metrics-artifact",
    )

    inputs = GenerationSnapshotInputs(
        taxonomy_version_id=taxonomy.id,
        interpretation_set_id=interpretation.id,
        generation_input_manifest_id=manifest.id,
        metrics_revision_id=metrics.id,
        created_by="test:reviewer",
    )
    # The builder reads no legacy links: an unmapped legacy observation fails
    # it until the version's backfill covers it (#513).
    with db_session.begin_nested(), pytest.raises(
        SnapshotBundleError, match="legacy_development_backfill_missing"
    ):
        build_snapshot_bundle(db_session, inputs)
    assert backfill_legacy_developments(db_session, taxonomy.id)["status"] == "backfilled"

    bundle = build_snapshot_bundle(db_session, inputs)
    catalog = db_session.scalar(
        select(ReaderSnapshotEntry).where(
            ReaderSnapshotEntry.reader_snapshot_bundle_id == bundle.id,
            ReaderSnapshotEntry.snapshot_kind == "economic_themes",
        )
    ).payload
    by_id = {row["economic_theme_id"]: row for row in catalog["themes"]}

    assert [row["observation_id"] for row in by_id[str(petroleum.id)]["developments"]] == [
        observation.id
    ]
    assert by_id[str(metals.id)]["developments"] == []

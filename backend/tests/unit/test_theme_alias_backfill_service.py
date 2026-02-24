"""Tests for theme alias backfill service."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ThemeAlias, ThemeCluster, ThemeMention
from app.services.theme_alias_backfill_service import ThemeAliasBackfillService


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    return session


def test_backfill_inserts_aliases_from_clusters_and_mentions():
    db = _make_session()
    try:
        cluster = ThemeCluster(
            name="AI Infrastructure",
            canonical_key="ai_infrastructure",
            display_name="AI Infrastructure",
            aliases=["AI Infra"],
            pipeline="technical",
        )
        db.add(cluster)
        db.flush()
        db.add(
            ThemeMention(
                theme_cluster_id=cluster.id,
                pipeline="technical",
                source_type="news",
                raw_theme="A.I. Infrastructure",
                canonical_theme="AI Infrastructure",
            )
        )
        db.commit()

        report = ThemeAliasBackfillService(db).run(dry_run=False)

        aliases = db.query(ThemeAlias).order_by(ThemeAlias.alias_key.asc()).all()
        keys = {(row.pipeline, row.alias_key, row.theme_cluster_id) for row in aliases}
        assert ("technical", "ai_infrastructure", cluster.id) in keys
        assert ("technical", "ai_infra", cluster.id) in keys
        assert report["totals"]["inserted"] == 2
        assert report["totals"]["collisions_total"] == 0
        assert report["collisions"]["inventory"] == []
    finally:
        db.close()


def test_backfill_reports_collisions_and_keeps_existing_mapping():
    db = _make_session()
    try:
        existing_cluster = ThemeCluster(
            name="Legacy Robotics",
            canonical_key="legacy_robotics",
            display_name="Legacy Robotics",
            aliases=[],
            pipeline="technical",
        )
        candidate_cluster = ThemeCluster(
            name="Robotics",
            canonical_key="robotics",
            display_name="Robotics",
            aliases=[],
            pipeline="technical",
        )
        collision_cluster = ThemeCluster(
            name="AI Infrastructure",
            canonical_key="ai_infrastructure",
            display_name="Shared Collision Alias",
            aliases=[],
            pipeline="technical",
        )
        collision_cluster_2 = ThemeCluster(
            name="AI Infra Platform",
            canonical_key="ai_infra_platform",
            display_name="Shared Collision Alias",
            aliases=[],
            pipeline="technical",
        )
        db.add_all([existing_cluster, candidate_cluster, collision_cluster, collision_cluster_2])
        db.flush()

        db.add(
            ThemeAlias(
                theme_cluster_id=existing_cluster.id,
                pipeline="technical",
                alias_text="Robotics",
                alias_key="robotic",
                source="llm_extraction",
                confidence=0.7,
                evidence_count=3,
                is_active=True,
            )
        )
        db.commit()

        report = ThemeAliasBackfillService(db).run(dry_run=False)

        robotics = db.query(ThemeAlias).filter(
            ThemeAlias.pipeline == "technical",
            ThemeAlias.alias_key == "robotic",
        ).one()
        assert robotics.theme_cluster_id == existing_cluster.id
        assert report["totals"]["collisions_total"] >= 2
        assert report["collisions"]["by_bucket"]["existing_alias_conflict"] >= 1
        assert report["collisions"]["by_bucket"]["candidate_cluster_collision"] >= 1
        assert report["remediation_actions"]
    finally:
        db.close()

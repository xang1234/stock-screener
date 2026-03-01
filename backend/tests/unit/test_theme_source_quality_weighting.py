"""Contract checks for theme source-type quality weighting."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ThemeCluster, ThemeMention
from app.services.theme_discovery_service import ThemeDiscoveryService


def test_twitter_source_quality_weight_remains_0_70() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    cluster = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai_infrastructure",
        display_name="AI Infrastructure",
        pipeline="technical",
        is_active=True,
    )
    db.add(cluster)
    db.commit()

    mention = ThemeMention(
        content_item_id=1,
        source_type="twitter",
        source_name="@alice",
        raw_theme="AI Infrastructure",
        canonical_theme="ai infrastructure",
        theme_cluster_id=cluster.id,
        pipeline="technical",
        confidence=1.0,
        mentioned_at=datetime.utcnow() - timedelta(days=1),
    )
    db.add(mention)
    db.commit()

    service = ThemeDiscoveryService.__new__(ThemeDiscoveryService)
    service.db = db
    service.pipeline = "technical"
    service.theme_policy_overrides = {}

    snapshot = service._lifecycle_snapshot(cluster.id, now=datetime.utcnow())
    assert snapshot["mentions_30d"] == 1
    assert snapshot["avg_quality_confidence_30d"] == 0.7

    db.close()

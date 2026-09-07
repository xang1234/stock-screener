"""Contract checks for theme source-type quality weighting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import ThemeCluster, ThemeMention, ContentSource, ContentItem
from app.services.theme_evidence_eligibility_service import grant_eligibility
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

    source = ContentSource(name="@alice", source_type="twitter", is_active=True)
    db.add(source)
    db.flush()
    item = ContentItem(source_id=source.id, source_type="twitter")
    db.add(item)
    db.flush()
    grant_eligibility(db, item.id, "technical", "legacy", source.id, datetime.now(timezone.utc))
    mention = ThemeMention(
        content_item_id=item.id,
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

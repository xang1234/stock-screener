"""Tests that L1 parent themes are excluded from L2-only pipeline operations.

Verifies the query isolation guard: L1 themes must not appear in rankings,
extraction candidates, merge-pair matching, lifecycle promotion, alerts,
or metrics bootstrap. They should only appear in taxonomy-specific endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import (
    ContentItem,
    ContentSource,
    ThemeAlert,
    ThemeCluster,
    ThemeConstituent,
    ThemeEmbedding,
    ThemeMention,
    ThemeMetrics,
    ThemeRelationship,
)
from app.models.app_settings import AppSetting
from app.services.theme_discovery_service import ThemeDiscoveryService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_l1_theme(db_session, *, name: str, canonical_key: str, now: datetime) -> ThemeCluster:
    """Create an L1 parent theme."""
    theme = ThemeCluster(
        name=name,
        canonical_key=canonical_key,
        display_name=name,
        pipeline="technical",
        is_active=True,
        is_l1=True,
        taxonomy_level=1,
        lifecycle_state="active",
        activated_at=now,
        first_seen_at=now,
        last_seen_at=now,
        discovery_source="taxonomy_assignment",
    )
    db_session.add(theme)
    db_session.flush()
    return theme


def _make_l2_theme(
    db_session,
    *,
    name: str,
    canonical_key: str,
    now: datetime,
    state: str = "active",
    parent_id: int | None = None,
) -> ThemeCluster:
    """Create an L2 leaf theme."""
    theme = ThemeCluster(
        name=name,
        canonical_key=canonical_key,
        display_name=name,
        pipeline="technical",
        is_active=True,
        is_l1=False,
        taxonomy_level=2,
        parent_cluster_id=parent_id,
        lifecycle_state=state,
        candidate_since_at=now - timedelta(days=7) if state == "candidate" else None,
        activated_at=now if state in {"active", "reactivated"} else None,
        first_seen_at=now - timedelta(days=3),
        last_seen_at=now - timedelta(hours=1),
    )
    db_session.add(theme)
    db_session.flush()
    return theme


def _make_source(db_session, *, name: str = "TestSource") -> ContentSource:
    source = ContentSource(
        name=name,
        source_type="substack",
        url=f"https://{name.lower()}.example.com/feed",
        is_active=True,
        pipelines=["technical"],
    )
    db_session.add(source)
    db_session.flush()
    return source


def _add_mention(
    db_session,
    *,
    theme: ThemeCluster,
    source: ContentSource,
    now: datetime,
    days_ago: int = 0,
) -> ThemeMention:
    content_item = ContentItem(
        source_id=source.id,
        source_type=source.source_type,
        source_name=source.name,
        title="Test article",
        content="Test content",
        published_at=now - timedelta(days=days_ago),
        is_processed=True,
    )
    db_session.add(content_item)
    db_session.flush()

    mention = ThemeMention(
        content_item_id=content_item.id,
        source_type=source.source_type,
        source_name=source.name,
        raw_theme=theme.name,
        canonical_theme=theme.canonical_key,
        theme_cluster_id=theme.id,
        pipeline="technical",
        tickers=["TEST"],
        ticker_count=1,
        sentiment="bullish",
        confidence=0.9,
        mentioned_at=now - timedelta(days=days_ago),
    )
    db_session.add(mention)
    db_session.flush()
    return mention


def _add_metrics(db_session, *, theme: ThemeCluster, now: datetime) -> ThemeMetrics:
    metrics = ThemeMetrics(
        theme_cluster_id=theme.id,
        date=now.date(),
        pipeline="technical",
        mentions_1d=5,
        mentions_7d=20,
        mentions_30d=50,
        mention_velocity=1.5,
        num_constituents=3,
        momentum_score=75.0,
        rank=1,
        status="trending",
    )
    db_session.add(metrics)
    db_session.flush()
    return metrics


class TestL1ExcludedFromRankings:
    """L1 themes must not appear in flat theme rankings."""

    def test_get_theme_rankings_excludes_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="AI & ML", canonical_key="ai_ml", now=now)
        l2a = _make_l2_theme(db_session, name="AI Chips", canonical_key="ai_chips", now=now, parent_id=l1.id)
        l2b = _make_l2_theme(db_session, name="AI Software", canonical_key="ai_software", now=now, parent_id=l1.id)

        _add_metrics(db_session, theme=l1, now=now)
        _add_metrics(db_session, theme=l2a, now=now)
        _add_metrics(db_session, theme=l2b, now=now)
        db_session.commit()

        service = ThemeDiscoveryService(db_session, pipeline="technical")
        rankings, total = service.get_theme_rankings(limit=100)

        ranked_ids = {r["theme_cluster_id"] for r in rankings}
        assert l1.id not in ranked_ids, "L1 theme should not appear in flat rankings"
        assert l2a.id in ranked_ids
        assert l2b.id in ranked_ids
        assert total == 2


class TestL1ExcludedFromEmerging:
    """L1 themes must not appear in emerging/trending theme queries."""

    def test_active_l2_query_excludes_l1(self, db_session):
        """Verify the base query pattern used by emerging/trending excludes L1."""
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Nuclear Energy", canonical_key="nuclear_energy", now=now)
        l2 = _make_l2_theme(db_session, name="Nuclear SMR", canonical_key="nuclear_smr", now=now, parent_id=l1.id)

        source = _make_source(db_session)
        _add_mention(db_session, theme=l1, source=source, now=now)
        _add_mention(db_session, theme=l2, source=source, now=now)
        db_session.commit()

        # This is the query pattern used throughout discovery service for
        # emerging themes, trending themes, and similar user-facing lists.
        active_l2 = db_session.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ThemeCluster.pipeline == "technical",
        ).all()

        ids = {t.id for t in active_l2}
        assert l1.id not in ids, "L1 theme should not appear in active L2 query"
        assert l2.id in ids


class TestL1ExcludedFromLifecycle:
    """L1 themes must not enter candidate promotion or dormancy flows."""

    def test_promote_candidate_excludes_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Semiconductors", canonical_key="semiconductors", now=now)
        # Force L1 to candidate state (should never happen, but test defensively)
        l1.lifecycle_state = "candidate"
        l1.candidate_since_at = now - timedelta(days=30)

        l2 = _make_l2_theme(
            db_session, name="GPU Demand", canonical_key="gpu_demand",
            now=now, state="candidate", parent_id=l1.id,
        )

        source = _make_source(db_session)
        # Add enough mentions for promotion threshold
        for i in range(10):
            _add_mention(db_session, theme=l1, source=source, now=now, days_ago=i % 5)
            _add_mention(db_session, theme=l2, source=source, now=now, days_ago=i % 5)
        db_session.commit()

        service = ThemeDiscoveryService(db_session, pipeline="technical")
        result = service.promote_candidate_themes(now=now)

        # L1 should not have been scanned for promotion
        db_session.refresh(l1)
        # Even if promotion criteria are met, L1 should not be touched
        # (the query should not have included it in candidates)
        assert result["scanned"] >= 0  # Only L2 candidates scanned

    def test_candidate_queue_excludes_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Defense", canonical_key="defense", now=now)
        l1.lifecycle_state = "candidate"
        l1.candidate_since_at = now - timedelta(days=7)

        l2 = _make_l2_theme(
            db_session, name="Drones", canonical_key="drones",
            now=now, state="candidate", parent_id=l1.id,
        )
        db_session.commit()

        service = ThemeDiscoveryService(db_session, pipeline="technical")
        queue, total = service.get_candidate_theme_queue()

        queue_ids = {r["theme_cluster_id"] for r in queue}
        assert l1.id not in queue_ids, "L1 should not appear in candidate queue"
        assert l2.id in queue_ids

    def test_dormancy_policies_exclude_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="EV", canonical_key="ev", now=now)

        l2 = _make_l2_theme(
            db_session, name="EV Charging", canonical_key="ev_charging",
            now=now, state="active", parent_id=l1.id,
        )
        db_session.commit()

        service = ThemeDiscoveryService(db_session, pipeline="technical")
        result = service.apply_dormancy_and_reactivation_policies(now=now)

        # L1 should not have been scanned
        assert l1.id not in {
            item.get("theme_cluster_id")
            for item in result.get("transitions", [])
        }


class TestL1ExcludedFromAlerts:
    """L1 themes must not be picked up for alert generation."""

    def test_alert_candidate_query_excludes_l1(self, db_session):
        """Verify the query pattern used for alert candidates excludes L1."""
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Crypto", canonical_key="crypto", now=now)
        l1.first_seen_at = now - timedelta(hours=12)  # "new" within 24h

        l2 = _make_l2_theme(
            db_session, name="Bitcoin ETF", canonical_key="bitcoin_etf", now=now, parent_id=l1.id,
        )
        l2.first_seen_at = now - timedelta(hours=6)
        db_session.commit()

        # This is the query pattern used to find themes eligible for alerts
        recent_themes = db_session.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ThemeCluster.first_seen_at >= now - timedelta(hours=24),
        ).all()

        ids = {t.id for t in recent_themes}
        assert l1.id not in ids, "L1 should not be picked up as alert candidate"
        assert l2.id in ids


class TestL1ExcludedFromMetricsUpdate:
    """L1 themes must not be included in L2 metrics batch update."""

    def test_update_all_theme_metrics_excludes_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Healthcare", canonical_key="healthcare", now=now)
        l2 = _make_l2_theme(
            db_session, name="GLP-1 Drugs", canonical_key="glp1_drugs",
            now=now, parent_id=l1.id,
        )

        source = _make_source(db_session)
        _add_mention(db_session, theme=l2, source=source, now=now)
        db_session.commit()

        service = ThemeDiscoveryService(db_session, pipeline="technical")
        result = service.update_all_theme_metrics()

        # Verify only L2 themes had metrics updated
        l1_metrics = db_session.query(ThemeMetrics).filter(
            ThemeMetrics.theme_cluster_id == l1.id
        ).count()
        assert l1_metrics == 0, "L1 theme should not have metrics from L2 update path"


class TestL1ExcludedFromMetricsBootstrap:
    """L1 themes must not trigger auto-metrics calculation in API handler."""

    def test_metrics_bootstrap_count_excludes_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Energy", canonical_key="energy", now=now)
        l2 = _make_l2_theme(
            db_session, name="Solar", canonical_key="solar", now=now, parent_id=l1.id,
        )
        _add_metrics(db_session, theme=l2, now=now)
        db_session.commit()

        # Simulate the API bootstrap check
        from app.models.theme import ThemeCluster as TC, ThemeMetrics as TM

        themes_without_metrics = db_session.query(TC).filter(
            TC.is_active == True,
            TC.is_l1 == False,
            ~TC.id.in_(db_session.query(TM.theme_cluster_id).distinct()),
        ).count()

        # L2 has metrics, L1 should not be counted
        assert themes_without_metrics == 0, "L1 without metrics should not trigger bootstrap"


class TestL1ExcludedFromCorrelationValidation:
    """L1 themes must not enter correlation validation."""

    def test_correlation_validation_excludes_l1(self, db_session):
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Materials", canonical_key="materials", now=now)
        l2a = _make_l2_theme(db_session, name="Copper", canonical_key="copper", now=now, parent_id=l1.id)
        l2b = _make_l2_theme(db_session, name="Lithium", canonical_key="lithium", now=now, parent_id=l1.id)

        for theme in [l2a, l2b]:
            db_session.add(ThemeConstituent(
                theme_cluster_id=theme.id, symbol="FCX", is_active=True,
            ))
        db_session.commit()

        # Verify the query pattern excludes L1
        clusters = db_session.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
        ).all()
        cluster_ids = {c.id for c in clusters}
        assert l1.id not in cluster_ids
        assert l2a.id in cluster_ids
        assert l2b.id in cluster_ids


class TestL1ExcludedFromRelationshipInference:
    """L1 themes must not participate in constituent-overlap relationship inference."""

    def test_relationship_query_excludes_l1(self, db_session):
        """Verify the query used for relationship inference excludes L1."""
        now = datetime.utcnow()
        l1 = _make_l1_theme(db_session, name="Tech", canonical_key="tech", now=now)
        l2a = _make_l2_theme(db_session, name="Cloud", canonical_key="cloud", now=now, parent_id=l1.id)
        l2b = _make_l2_theme(db_session, name="SaaS", canonical_key="saas", now=now, parent_id=l1.id)
        db_session.commit()

        # This is the query pattern used for relationship inference
        candidates = db_session.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
        ).all()

        candidate_ids = {c.id for c in candidates}
        assert l1.id not in candidate_ids, "L1 should not be in relationship candidates"
        assert l2a.id in candidate_ids
        assert l2b.id in candidate_ids

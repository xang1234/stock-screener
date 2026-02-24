"""Tests for lifecycle automation policies and relationship inference."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import (
    ContentItem,
    ContentSource,
    ThemeCluster,
    ThemeConstituent,
    ThemeLifecycleTransition,
    ThemeMention,
    ThemeMergeSuggestion,
    ThemeRelationship,
)
from app.services.theme_discovery_service import ThemeDiscoveryService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_source(db_session, *, name: str, source_type: str) -> ContentSource:
    source = ContentSource(
        name=name,
        source_type=source_type,
        url=f"https://{name.lower().replace(' ', '-')}.example.com/feed",
        is_active=True,
        pipelines=["technical", "fundamental"],
    )
    db_session.add(source)
    db_session.flush()
    return source


def _make_theme(
    db_session,
    *,
    name: str,
    canonical_key: str,
    state: str,
    now: datetime,
) -> ThemeCluster:
    theme = ThemeCluster(
        name=name,
        canonical_key=canonical_key,
        display_name=name,
        pipeline="technical",
        is_active=True,
        lifecycle_state=state,
        candidate_since_at=now - timedelta(days=7),
        activated_at=(now - timedelta(days=7)) if state in {"active", "reactivated", "dormant"} else None,
        first_seen_at=now - timedelta(days=7),
        last_seen_at=now - timedelta(days=1),
    )
    db_session.add(theme)
    db_session.flush()
    return theme


def _add_mention(
    db_session,
    *,
    theme: ThemeCluster,
    source: ContentSource,
    now: datetime,
    days_ago: int,
    confidence: float,
    external_suffix: str,
) -> None:
    published_at = now - timedelta(days=days_ago)
    content = ContentItem(
        source_id=source.id,
        source_type=source.source_type,
        source_name=source.name,
        external_id=f"{theme.canonical_key}-{external_suffix}",
        title=f"{theme.name} mention {external_suffix}",
        content=f"Evidence for {theme.name}",
        published_at=published_at,
        is_processed=True,
        processed_at=published_at,
    )
    db_session.add(content)
    db_session.flush()
    mention = ThemeMention(
        content_item_id=content.id,
        source_type=source.source_type,
        source_name=source.name,
        raw_theme=theme.display_name,
        canonical_theme=theme.display_name,
        theme_cluster_id=theme.id,
        pipeline="technical",
        tickers=[],
        ticker_count=0,
        sentiment="bullish",
        confidence=confidence,
        excerpt="Theme evidence",
        mentioned_at=published_at,
    )
    db_session.add(mention)


def test_candidate_promotion_policy_promotes_theme_with_persistent_diverse_evidence(db_session):
    now = datetime(2026, 2, 24, 15, 0, 0)
    source_a = _make_source(db_session, name="Alpha Desk", source_type="news")
    source_b = _make_source(db_session, name="Bravo Research", source_type="substack")
    theme = _make_theme(
        db_session,
        name="AI Infrastructure",
        canonical_key="ai_infrastructure",
        state="candidate",
        now=now,
    )
    _add_mention(db_session, theme=theme, source=source_a, now=now, days_ago=1, confidence=0.92, external_suffix="1")
    _add_mention(db_session, theme=theme, source=source_b, now=now, days_ago=2, confidence=0.88, external_suffix="2")
    _add_mention(db_session, theme=theme, source=source_a, now=now, days_ago=3, confidence=0.89, external_suffix="3")
    _add_mention(db_session, theme=theme, source=source_b, now=now, days_ago=5, confidence=0.91, external_suffix="4")
    db_session.commit()

    service = ThemeDiscoveryService(db_session, pipeline="technical")
    result = service.promote_candidate_themes(now=now)

    db_session.refresh(theme)
    assert result["promoted"] == 1
    assert theme.lifecycle_state == "active"
    assert theme.activated_at is not None
    assert theme.lifecycle_state_metadata["promotion_count"] == 1
    assert theme.lifecycle_state_metadata["continuity_id"] == "technical:ai_infrastructure"

    transitions = db_session.query(ThemeLifecycleTransition).filter(
        ThemeLifecycleTransition.theme_cluster_id == theme.id
    ).all()
    assert len(transitions) == 1
    assert transitions[0].from_state == "candidate"
    assert transitions[0].to_state == "active"


def test_dormancy_and_reactivation_policies_increment_counters(db_session):
    now = datetime(2026, 2, 24, 16, 0, 0)
    source_a = _make_source(db_session, name="Gamma Wire", source_type="news")
    source_b = _make_source(db_session, name="Delta Feed", source_type="substack")
    theme = _make_theme(
        db_session,
        name="Grid Modernization",
        canonical_key="grid_modernization",
        state="active",
        now=now,
    )
    _add_mention(db_session, theme=theme, source=source_a, now=now, days_ago=40, confidence=0.87, external_suffix="old")
    db_session.commit()

    service = ThemeDiscoveryService(db_session, pipeline="technical")
    dormancy_result = service.apply_dormancy_and_reactivation_policies(now=now)

    db_session.refresh(theme)
    assert dormancy_result["to_dormant"] == 1
    assert theme.lifecycle_state == "dormant"
    assert theme.lifecycle_state_metadata["dormancy_count"] == 1

    _add_mention(db_session, theme=theme, source=source_a, now=now, days_ago=1, confidence=0.90, external_suffix="new1")
    _add_mention(db_session, theme=theme, source=source_b, now=now, days_ago=2, confidence=0.86, external_suffix="new2")
    db_session.commit()

    reactivation_result = service.apply_dormancy_and_reactivation_policies(now=now + timedelta(minutes=1))

    db_session.refresh(theme)
    assert reactivation_result["to_reactivated"] == 1
    assert theme.lifecycle_state == "reactivated"
    assert theme.lifecycle_state_metadata["reactivation_count"] == 1
    assert theme.lifecycle_state_metadata["continuity_id"] == "technical:grid_modernization"


def test_relationship_inference_writes_merge_and_overlap_edges(db_session):
    now = datetime.utcnow()
    theme_a = _make_theme(
        db_session,
        name="AI Chips",
        canonical_key="ai_chips",
        state="active",
        now=now,
    )
    theme_b = _make_theme(
        db_session,
        name="AI Infrastructure",
        canonical_key="ai_infrastructure",
        state="active",
        now=now,
    )
    theme_c = _make_theme(
        db_session,
        name="Defense Primes",
        canonical_key="defense_primes",
        state="active",
        now=now,
    )
    db_session.add_all(
        [
            ThemeConstituent(theme_cluster_id=theme_a.id, symbol="NVDA", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_a.id, symbol="AVGO", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_a.id, symbol="AMD", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_b.id, symbol="NVDA", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_b.id, symbol="AVGO", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_b.id, symbol="AMD", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_b.id, symbol="SMCI", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_c.id, symbol="LMT", is_active=True),
            ThemeConstituent(theme_cluster_id=theme_c.id, symbol="NOC", is_active=True),
        ]
    )
    db_session.add(
        ThemeMergeSuggestion(
            source_cluster_id=theme_b.id,
            target_cluster_id=theme_c.id,
            embedding_similarity=0.73,
            llm_confidence=0.91,
            llm_relationship="distinct",
            llm_reasoning="Different sectors with little overlap.",
            status="rejected",
        )
    )
    db_session.commit()

    service = ThemeDiscoveryService(db_session, pipeline="technical")
    result = service.infer_theme_relationships(max_merge_suggestions=50)

    assert result["merge_edges_written"] >= 1
    assert result["rule_edges_written"] >= 1

    edges = db_session.query(ThemeRelationship).all()
    edge_keys = {(edge.source_cluster_id, edge.target_cluster_id, edge.relationship_type) for edge in edges}
    assert (theme_a.id, theme_b.id, "subset") in edge_keys

    distinct_pair = tuple(sorted([theme_b.id, theme_c.id]))
    assert any(
        tuple(sorted([edge.source_cluster_id, edge.target_cluster_id])) == distinct_pair
        and edge.relationship_type == "distinct"
        for edge in edges
    )

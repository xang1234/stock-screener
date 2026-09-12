"""Tests for theme matching telemetry aggregation endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from app.api.v1.themes import (
    get_candidate_theme_queue,
    get_lifecycle_transitions,
    get_matching_telemetry,
    get_relationship_graph,
    get_theme_rankings,
    review_candidate_themes,
)
from app.database import Base
from app.models.theme import (
    ThemeCluster,
    ThemeLifecycleTransition,
    ThemeMention,
    ThemeRelationship,
)
from app.schemas.theme import CandidateThemeReviewRequest
from app.services.theme_discovery_service import ThemeDiscoveryService
from app.services.theme_equivalence_service import ThemeEquivalenceService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _mention(
    *,
    source_type: str | None,
    pipeline: str,
    method: str | None,
    threshold_version: str | None,
    confidence: float | None,
    score: float | None,
    fallback_reason: str | None = None,
    days_ago: int = 1,
) -> ThemeMention:
    return ThemeMention(
        source_type=source_type,
        source_name=source_type,
        raw_theme=f"{source_type}-{method}",
        canonical_theme=f"{source_type}-{method}",
        pipeline=pipeline,
        tickers=[],
        ticker_count=0,
        sentiment="neutral",
        confidence=confidence,
        excerpt="",
        mentioned_at=datetime.utcnow() - timedelta(days=days_ago),
        match_method=method,
        match_score=score,
        match_threshold=0.85 if threshold_version == "embedding-v1" else 1.0,
        threshold_version=threshold_version,
        match_fallback_reason=fallback_reason,
    )


def test_matching_telemetry_aggregates_method_source_and_threshold_breakdowns(db_session):
    db_session.add_all(
        [
            _mention(
                source_type="news",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="match-v1",
                confidence=0.32,
                score=0.0,
                fallback_reason="no_existing_cluster_match",
            ),
            _mention(
                source_type="news",
                pipeline="technical",
                method="fuzzy_lexical",
                threshold_version="match-v1",
                confidence=0.81,
                score=0.92,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="embedding_similarity",
                threshold_version="embedding-v1",
                confidence=0.88,
                score=0.91,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="embedding-v1",
                confidence=0.75,
                score=0.0,
                fallback_reason="embedding_ambiguous_review",
            ),
            # Outside window; should be excluded.
            _mention(
                source_type="news",
                pipeline="technical",
                method="exact_alias_key",
                threshold_version="match-v1",
                confidence=0.9,
                score=1.0,
                days_ago=90,
            ),
        ]
    )
    db_session.commit()

    payload = get_matching_telemetry(
        days=30,
        pipeline="technical",
        db=db_session,
    )

    assert payload.total_mentions == 4
    assert payload.new_cluster_count == 2
    assert payload.attach_count == 2
    assert payload.new_cluster_rate == 0.5
    assert payload.attach_rate == 0.5
    assert {item.key for item in payload.by_threshold_version} == {"match-v1", "embedding-v1"}
    assert {item.key for item in payload.by_source_type} == {"news", "substack"}
    assert any(item.method == "create_new_cluster" and item.count == 2 for item in payload.method_distribution)
    assert any(item.reason == "none" and item.count == 2 for item in payload.decision_reason_distribution)


def test_matching_telemetry_supports_source_and_threshold_filters(db_session):
    db_session.add_all(
        [
            _mention(
                source_type="news",
                pipeline="technical",
                method="fuzzy_lexical",
                threshold_version="match-v1",
                confidence=0.78,
                score=0.90,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="embedding_similarity",
                threshold_version="embedding-v1",
                confidence=0.86,
                score=0.93,
            ),
            _mention(
                source_type="substack",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="embedding-v1",
                confidence=0.65,
                score=0.0,
                fallback_reason="embedding_low_confidence_review",
            ),
        ]
    )
    db_session.commit()

    payload = get_matching_telemetry(
        days=30,
        pipeline="technical",
        source_type="substack",
        threshold_version="embedding-v1",
        db=db_session,
    )

    assert payload.total_mentions == 2
    assert payload.new_cluster_count == 1
    assert payload.attach_count == 1
    assert payload.by_threshold_version[0].key == "embedding-v1"
    assert payload.by_source_type[0].key == "substack"
    assert all(item.method in {"embedding_similarity", "create_new_cluster"} for item in payload.method_distribution)


def test_matching_telemetry_unknown_threshold_filter_maps_to_null_column(db_session):
    db_session.add_all(
        [
            _mention(
                source_type="news",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version=None,
                confidence=0.5,
                score=0.0,
            ),
            _mention(
                source_type="news",
                pipeline="technical",
                method="fuzzy_lexical",
                threshold_version="match-v1",
                confidence=0.8,
                score=0.9,
            ),
        ]
    )
    db_session.commit()

    payload = get_matching_telemetry(
        days=30,
        pipeline="technical",
        source_type="news",
        threshold_version="unknown",
        db=db_session,
    )

    assert payload.total_mentions == 1
    assert payload.by_source_type[0].key == "news"
    assert payload.by_threshold_version[0].key == "unknown"


def test_matching_telemetry_does_not_count_unknown_method_as_attach(db_session):
    db_session.add_all(
        [
            _mention(
                source_type="news",
                pipeline="technical",
                method="create_new_cluster",
                threshold_version="match-v1",
                confidence=0.3,
                score=0.0,
            ),
            _mention(
                source_type="news",
                pipeline="technical",
                method=None,
                threshold_version="match-v1",
                confidence=0.7,
                score=0.5,
            ),
        ]
    )
    db_session.commit()

    payload = get_matching_telemetry(days=30, pipeline="technical", db=db_session)

    assert payload.total_mentions == 2
    assert payload.new_cluster_count == 1
    assert payload.attach_count == 0
    assert payload.new_cluster_rate == 0.5
    assert payload.attach_rate == 0.0


def test_rankings_empty_response_preserves_requested_pipeline(db_session):
    payload = get_theme_rankings(
        limit=20,
        offset=0,
        status=None,
        source_types=None,
        lifecycle_states=None,
        pipeline="fundamental",
        recalculate=False,
        db=db_session,
    )

    assert payload.pipeline == "fundamental"
    assert payload.total_themes == 0
    assert payload.rankings == []


def test_get_lifecycle_transitions_returns_rows_with_context(db_session):
    now = datetime.utcnow()
    cluster = ThemeCluster(
        name="AI Infrastructure",
        canonical_key="ai_infrastructure",
        display_name="AI Infrastructure",
        pipeline="technical",
        is_active=True,
        lifecycle_state="active",
    )
    db_session.add(cluster)
    db_session.flush()
    db_session.add(
        ThemeLifecycleTransition(
            theme_cluster_id=cluster.id,
            from_state="candidate",
            to_state="active",
            actor="system",
            job_name="promote_candidate_themes",
            rule_version="lifecycle-v2",
            reason="candidate_promotion_thresholds_met",
            transition_metadata={"foo": "bar"},
            transitioned_at=now,
        )
    )
    db_session.commit()

    payload = get_lifecycle_transitions(
        limit=20,
        offset=0,
        pipeline="technical",
        theme_cluster_id=cluster.id,
        to_state="active",
        db=db_session,
    )

    assert payload.total == 1
    assert len(payload.transitions) == 1
    row = payload.transitions[0]
    assert row.theme_cluster_id == cluster.id
    assert row.from_state == "candidate"
    assert row.to_state == "active"
    assert row.transition_metadata == {"foo": "bar"}
    assert row.transition_history_path.endswith(f"theme_cluster_id={cluster.id}")


def test_get_candidate_theme_queue_returns_evidence_and_bands(db_session):
    from datetime import timezone

    from app.models.theme import ContentItem, ContentSource
    from app.services.theme_evidence_eligibility_service import grant_eligibility
    def legacy_item(source_type):
        source = ContentSource(name=source_type, source_type=source_type, is_active=True)
        db_session.add(source)
        db_session.flush()
        item = ContentItem(source_id=source.id, source_type=source_type)
        db_session.add(item)
        db_session.flush()
        grant_eligibility(db_session, item.id, "technical", "legacy", source.id, datetime.now(timezone.utc))
        return item.id

    now = datetime.utcnow()
    candidate = ThemeCluster(
        name="AI Grid",
        canonical_key="ai_grid",
        display_name="AI Grid",
        pipeline="technical",
        is_active=True,
        lifecycle_state="candidate",
        candidate_since_at=now - timedelta(days=3),
        first_seen_at=now - timedelta(days=5),
        lifecycle_state_metadata={"reason": "candidate_review_pending"},
    )
    db_session.add(candidate)
    db_session.flush()
    db_session.add_all(
        [
            ThemeMention(
                content_item_id=legacy_item("news"),
                source_type="news",
                source_name="news",
                raw_theme="AI Grid",
                canonical_theme="ai_grid",
                theme_cluster_id=candidate.id,
                pipeline="technical",
                tickers=[],
                ticker_count=0,
                sentiment="neutral",
                confidence=0.82,
                mentioned_at=now - timedelta(days=1),
            ),
            ThemeMention(
                content_item_id=legacy_item("substack"),
                source_type="substack",
                source_name="substack",
                raw_theme="AI Grid",
                canonical_theme="ai_grid",
                theme_cluster_id=candidate.id,
                pipeline="technical",
                tickers=[],
                ticker_count=0,
                sentiment="neutral",
                confidence=0.78,
                mentioned_at=now - timedelta(days=2),
            ),
        ]
    )
    db_session.commit()

    payload = get_candidate_theme_queue(limit=20, offset=0, pipeline="technical", db=db_session)

    assert payload.total == 1
    assert len(payload.items) == 1
    row = payload.items[0]
    assert row.theme_cluster_id == candidate.id
    assert row.evidence["mentions_7d"] >= 2
    assert row.confidence_band in {"0.70-0.84", "0.85-1.00"}
    assert payload.confidence_bands[0].count >= 1


def test_get_candidate_theme_queue_band_summary_is_global_not_page_scoped(db_session):
    now = datetime.utcnow()
    candidate_a = ThemeCluster(
        name="Grid Compute",
        canonical_key="grid_compute",
        display_name="Grid Compute",
        pipeline="technical",
        is_active=True,
        lifecycle_state="candidate",
        candidate_since_at=now - timedelta(days=3),
        first_seen_at=now - timedelta(days=10),
    )
    candidate_b = ThemeCluster(
        name="Power Cooling",
        canonical_key="power_cooling",
        display_name="Power Cooling",
        pipeline="technical",
        is_active=True,
        lifecycle_state="candidate",
        candidate_since_at=now - timedelta(days=2),
        first_seen_at=now - timedelta(days=9),
    )
    db_session.add_all([candidate_a, candidate_b])
    db_session.flush()
    db_session.add_all(
        [
            ThemeMention(
                source_type="news",
                source_name="news",
                raw_theme="Grid Compute",
                canonical_theme="grid_compute",
                theme_cluster_id=candidate_a.id,
                pipeline="technical",
                tickers=[],
                ticker_count=0,
                sentiment="neutral",
                confidence=0.92,
                mentioned_at=now - timedelta(days=1),
            ),
            ThemeMention(
                source_type="substack",
                source_name="substack",
                raw_theme="Power Cooling",
                canonical_theme="power_cooling",
                theme_cluster_id=candidate_b.id,
                pipeline="technical",
                tickers=[],
                ticker_count=0,
                sentiment="neutral",
                confidence=0.45,
                mentioned_at=now - timedelta(days=1),
            ),
        ]
    )
    db_session.commit()

    payload = get_candidate_theme_queue(limit=1, offset=0, pipeline="technical", db=db_session)

    assert payload.total == 2
    assert len(payload.items) == 1
    assert sum(bucket.count for bucket in payload.confidence_bands) == 2


def test_candidate_review_uses_group_representatives(db_session):
    now = datetime.utcnow()
    member = ThemeCluster(
        name="CPO Candidate",
        canonical_key="cpo_candidate",
        display_name="CPO Candidate",
        pipeline="technical",
        is_active=True,
        lifecycle_state="candidate",
        candidate_since_at=now - timedelta(days=2),
        first_seen_at=now - timedelta(days=3),
    )
    representative = ThemeCluster(
        name="Co-Packaged Optics Candidate",
        canonical_key="co_packaged_optics_candidate",
        display_name="Co-Packaged Optics Candidate",
        pipeline="technical",
        is_active=True,
        lifecycle_state="candidate",
        candidate_since_at=now - timedelta(days=1),
        first_seen_at=now - timedelta(days=3),
    )
    db_session.add_all([member, representative])
    db_session.flush()
    ThemeEquivalenceService(db_session).apply(
        member.id,
        representative.id,
        actor="reviewer",
        reason="Equivalent exposure",
        key="candidate-review-group",
    )
    db_session.commit()

    service = ThemeDiscoveryService(db_session, pipeline="technical")
    rows, total = service.get_candidate_theme_queue(limit=20)
    bands = service.get_candidate_theme_confidence_bands()
    result = service.review_candidate_themes(
        theme_cluster_ids=[member.id, representative.id],
        action="promote",
        actor="analyst:test",
    )

    db_session.refresh(member)
    db_session.refresh(representative)
    assert total == 1
    assert [row["theme_cluster_id"] for row in rows] == [representative.id]
    assert sum(row["count"] for row in bands) == 1
    assert result["updated"] == 1
    assert result["results"][0]["theme_cluster_id"] == representative.id
    assert representative.lifecycle_state == "active"
    assert member.lifecycle_state == "candidate"


def test_review_candidate_themes_promote_transitions_to_active(db_session):
    candidate = ThemeCluster(
        name="Grid Load",
        canonical_key="grid_load",
        display_name="Grid Load",
        pipeline="technical",
        is_active=True,
        lifecycle_state="candidate",
        first_seen_at=datetime.utcnow() - timedelta(days=10),
    )
    db_session.add(candidate)
    db_session.commit()

    payload = CandidateThemeReviewRequest(theme_cluster_ids=[candidate.id], action="promote", actor="analyst:test")
    result = review_candidate_themes(payload=payload, pipeline="technical", db=db_session)

    assert result.success is True
    assert result.updated == 1
    refreshed = db_session.query(ThemeCluster).filter(ThemeCluster.id == candidate.id).one()
    assert refreshed.lifecycle_state == "active"
    assert refreshed.is_active is True
    transition = db_session.query(ThemeLifecycleTransition).filter(
        ThemeLifecycleTransition.theme_cluster_id == candidate.id
    ).one()
    assert transition.to_state == "active"
    assert transition.actor == "analyst:test"


def test_get_relationship_graph_returns_nodes_and_edges(db_session):
    source = ThemeCluster(
        name="Power Infrastructure",
        canonical_key="power_infra",
        display_name="Power Infrastructure",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    peer = ThemeCluster(
        name="Utilities Capex",
        canonical_key="utilities_capex",
        display_name="Utilities Capex",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    db_session.add_all([source, peer])
    db_session.flush()
    db_session.add(
        ThemeRelationship(
            source_cluster_id=source.id,
            target_cluster_id=peer.id,
            pipeline="technical",
            relationship_type="related",
            confidence=0.83,
            provenance="test_fixture",
            evidence={"overlap_count": 4},
            is_active=True,
        )
    )
    db_session.commit()

    payload = get_relationship_graph(
        theme_cluster_id=source.id,
        pipeline="technical",
        limit=50,
        db=db_session,
    )

    assert payload.theme_cluster_id == source.id
    assert payload.total_nodes >= 2
    assert payload.total_edges >= 1
    assert any(node.theme_cluster_id == source.id and node.is_root for node in payload.nodes)
    assert any(edge.relationship_type == "related" for edge in payload.edges)


def test_get_relationship_graph_canonicalizes_group_members(db_session):
    alias = ThemeCluster(
        name="CPO",
        canonical_key="cpo",
        display_name="CPO",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    representative = ThemeCluster(
        name="Co-Packaged Optics",
        canonical_key="co_packaged_optics",
        display_name="Co-Packaged Optics",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    peer_alias = ThemeCluster(
        name="Optical Engines",
        canonical_key="optical_engines",
        display_name="Optical Engines",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    peer_representative = ThemeCluster(
        name="Silicon Photonics",
        canonical_key="silicon_photonics",
        display_name="Silicon Photonics",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    other_peer = ThemeCluster(
        name="Optical Networking",
        canonical_key="optical_networking",
        display_name="Optical Networking",
        pipeline="technical",
        lifecycle_state="active",
        is_active=True,
    )
    db_session.add_all(
        [alias, representative, peer_alias, peer_representative, other_peer]
    )
    db_session.flush()
    grouping = ThemeEquivalenceService(db_session)
    grouping.apply(
        alias.id,
        representative.id,
        actor="reviewer",
        reason="Equivalent exposure",
        key="graph-group",
    )
    grouping.apply(
        peer_alias.id,
        peer_representative.id,
        actor="reviewer",
        reason="Equivalent exposure",
        key="graph-peer-group",
    )
    db_session.add_all(
        [
            ThemeRelationship(
                source_cluster_id=alias.id,
                target_cluster_id=representative.id,
                pipeline="technical",
                relationship_type="related",
                confidence=0.99,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
            ThemeRelationship(
                source_cluster_id=alias.id,
                target_cluster_id=peer_alias.id,
                pipeline="technical",
                relationship_type="related",
                confidence=0.81,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
            ThemeRelationship(
                source_cluster_id=representative.id,
                target_cluster_id=peer_representative.id,
                pipeline="technical",
                relationship_type="related",
                confidence=0.91,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
            ThemeRelationship(
                source_cluster_id=representative.id,
                target_cluster_id=other_peer.id,
                pipeline="technical",
                relationship_type="subset",
                confidence=0.85,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
            ThemeRelationship(
                source_cluster_id=peer_representative.id,
                target_cluster_id=other_peer.id,
                pipeline="technical",
                relationship_type="related",
                confidence=0.98,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
            ThemeRelationship(
                source_cluster_id=peer_representative.id,
                target_cluster_id=other_peer.id,
                pipeline="technical",
                relationship_type="distinct",
                confidence=0.97,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
        ]
    )
    db_session.commit()

    payload = get_relationship_graph(
        theme_cluster_id=alias.id,
        pipeline="technical",
        limit=2,
        db=db_session,
    )

    assert payload.theme_cluster_id == representative.id
    assert {node.theme_cluster_id for node in payload.nodes} == {
        representative.id,
        peer_representative.id,
        other_peer.id,
    }
    assert all(
        node.theme_cluster_id not in {alias.id, peer_alias.id}
        for node in payload.nodes
    )
    assert {
        (edge.source_theme_id, edge.target_theme_id, edge.relationship_type)
        for edge in payload.edges
    } == {
        (representative.id, peer_representative.id, "related"),
        (representative.id, other_peer.id, "subset"),
    }
    related = next(
        edge for edge in payload.edges if edge.relationship_type == "related"
    )
    assert related.confidence == 0.91


def test_relationship_graph_canonicalizes_unordered_edges_after_grouping(
    db_session,
):
    alias = ThemeCluster(
        name="CPO Alias",
        canonical_key="cpo_alias",
        display_name="CPO Alias",
        pipeline="technical",
        is_active=True,
    )
    peer = ThemeCluster(
        name="Optical Interconnects",
        canonical_key="optical_interconnects",
        display_name="Optical Interconnects",
        pipeline="technical",
        is_active=True,
    )
    representative = ThemeCluster(
        name="Co-Packaged Optics",
        canonical_key="co_packaged_optics",
        display_name="Co-Packaged Optics",
        pipeline="technical",
        is_active=True,
    )
    db_session.add_all([alias, peer, representative])
    db_session.flush()
    ThemeEquivalenceService(db_session).apply(
        alias.id,
        representative.id,
        actor="reviewer",
        reason="Equivalent exposure",
        key="unordered-edge-group",
    )
    db_session.add_all(
        [
            ThemeRelationship(
                source_cluster_id=alias.id,
                target_cluster_id=peer.id,
                pipeline="technical",
                relationship_type="related",
                confidence=0.81,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
            ThemeRelationship(
                source_cluster_id=peer.id,
                target_cluster_id=representative.id,
                pipeline="technical",
                relationship_type="related",
                confidence=0.91,
                provenance="test_fixture",
                evidence={},
                is_active=True,
            ),
        ]
    )
    db_session.commit()

    payload = get_relationship_graph(
        theme_cluster_id=representative.id,
        pipeline="technical",
        limit=10,
        db=db_session,
    )

    assert len(payload.edges) == 1
    assert (
        payload.edges[0].source_theme_id,
        payload.edges[0].target_theme_id,
        payload.edges[0].relationship_type,
    ) == (peer.id, representative.id, "related")
    assert payload.edges[0].confidence == 0.91

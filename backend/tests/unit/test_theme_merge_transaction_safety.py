"""Tests for merge transaction safety and idempotent approval retries."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.theme import (
    ThemeCluster,
    ThemeConstituent,
    ThemeMention,
    ThemeMergeHistory,
    ThemeMergeSuggestion,
)
from app.services.theme_merging_service import ThemeMergingService


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_service(db_session) -> ThemeMergingService:
    service = ThemeMergingService.__new__(ThemeMergingService)
    service.db = db_session
    service.update_theme_embedding = lambda _theme: (None, False)
    return service


def _make_cluster(db_session, *, key: str, name: str) -> ThemeCluster:
    cluster = ThemeCluster(
        canonical_key=key,
        display_name=name,
        name=name,
        aliases=[name],
        description=f"{name} description",
        category="technology",
        pipeline="technical",
        is_active=True,
        lifecycle_state="active",
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
    )
    db_session.add(cluster)
    db_session.flush()
    return cluster


def test_approve_suggestion_retry_returns_idempotent_success(db_session):
    source = _make_cluster(db_session, key="ai_source", name="AI Source")
    target = _make_cluster(db_session, key="ai_target", name="AI Target")

    db_session.add(
        ThemeConstituent(
            theme_cluster_id=source.id,
            symbol="NVDA",
            mention_count=3,
            confidence=0.9,
        )
    )
    db_session.add(
        ThemeMention(
            content_item_id=1,
            source_type="news",
            source_name="test",
            raw_theme="AI Source",
            canonical_theme="ai source",
            theme_cluster_id=source.id,
            pipeline="technical",
            tickers=["NVDA"],
            ticker_count=1,
            confidence=0.9,
            mentioned_at=datetime.utcnow(),
        )
    )
    suggestion = ThemeMergeSuggestion(
        source_cluster_id=source.id,
        target_cluster_id=target.id,
        embedding_similarity=0.96,
        llm_confidence=0.92,
        status="pending",
    )
    db_session.add(suggestion)
    db_session.commit()

    service = _make_service(db_session)

    first = service.approve_suggestion(suggestion.id, idempotency_key="manual-merge-123")
    second = service.approve_suggestion(suggestion.id, idempotency_key="manual-merge-123")

    assert first["success"] is True
    assert first.get("idempotent_replay") is not True
    assert second["success"] is True
    assert second["idempotent_replay"] is True
    assert second["idempotency_key"] == "manual-merge-123"
    assert second["constituents_merged"] == first["constituents_merged"]
    assert second["mentions_merged"] == first["mentions_merged"]

    assert db_session.query(ThemeMergeHistory).count() == 1
    refreshed_suggestion = db_session.query(ThemeMergeSuggestion).filter(
        ThemeMergeSuggestion.id == suggestion.id
    ).one()
    assert refreshed_suggestion.status == "approved"
    assert refreshed_suggestion.approval_idempotency_key == "manual-merge-123"
    assert refreshed_suggestion.approval_result_json is not None


def test_approve_suggestion_rejects_different_idempotency_key_after_success(db_session):
    source = _make_cluster(db_session, key="strict_source", name="Strict Source")
    target = _make_cluster(db_session, key="strict_target", name="Strict Target")
    suggestion = ThemeMergeSuggestion(
        source_cluster_id=source.id,
        target_cluster_id=target.id,
        pair_min_cluster_id=min(source.id, target.id),
        pair_max_cluster_id=max(source.id, target.id),
        embedding_similarity=0.97,
        status="pending",
    )
    db_session.add(suggestion)
    db_session.commit()

    service = _make_service(db_session)
    ok = service.approve_suggestion(suggestion.id, idempotency_key="strict-key-a")
    mismatch = service.approve_suggestion(suggestion.id, idempotency_key="strict-key-b")

    assert ok["success"] is True
    assert mismatch["success"] is False
    assert "Idempotency key mismatch" in mismatch["error"]


def test_execute_merge_rejects_stale_suggestion_state_without_side_effects(db_session):
    source = _make_cluster(db_session, key="edge_source", name="Edge Source")
    target = _make_cluster(db_session, key="edge_target", name="Edge Target")
    suggestion = ThemeMergeSuggestion(
        source_cluster_id=source.id,
        target_cluster_id=target.id,
        embedding_similarity=0.91,
        status="rejected",
    )
    db_session.add(suggestion)
    db_session.commit()

    service = _make_service(db_session)
    result = service.execute_merge(
        source.id,
        target.id,
        merge_type="manual",
        suggestion=suggestion,
        expected_suggestion_status="pending",
        final_suggestion_status="approved",
    )

    assert result["success"] is False
    assert "status changed" in result["error"]
    assert db_session.query(ThemeMergeHistory).count() == 0
    assert db_session.query(ThemeCluster).filter(ThemeCluster.id == source.id).one().is_active is True


def test_create_merge_suggestion_deduplicates_reversed_pairs(db_session):
    left = _make_cluster(db_session, key="left_theme", name="Left Theme")
    right = _make_cluster(db_session, key="right_theme", name="Right Theme")
    db_session.commit()

    service = _make_service(db_session)
    first = service.create_merge_suggestion(left.id, right.id, 0.91, {"confidence": 0.7})
    second = service.create_merge_suggestion(right.id, left.id, 0.95, {"confidence": 0.9})

    assert first is not None
    assert second is not None
    assert first.id == second.id
    all_rows = db_session.query(ThemeMergeSuggestion).all()
    assert len(all_rows) == 1
    only = all_rows[0]
    assert only.pair_min_cluster_id == min(left.id, right.id)
    assert only.pair_max_cluster_id == max(left.id, right.id)
    assert only.embedding_similarity == 0.95


def test_get_merge_suggestions_exposes_canonical_and_legacy_contract_fields(db_session):
    source = _make_cluster(db_session, key="contract_source", name="Contract Source")
    target = _make_cluster(db_session, key="contract_target", name="Contract Target")
    suggestion = ThemeMergeSuggestion(
        source_cluster_id=source.id,
        target_cluster_id=target.id,
        pair_min_cluster_id=min(source.id, target.id),
        pair_max_cluster_id=max(source.id, target.id),
        embedding_similarity=0.88,
        llm_confidence=0.77,
        llm_relationship="identical",
        llm_reasoning="Same underlying concept",
        suggested_canonical_name="Contract Target",
        status="pending",
    )
    db_session.add(suggestion)
    db_session.commit()

    service = _make_service(db_session)
    payload = service.get_merge_suggestions(status="pending", limit=10)
    assert len(payload) == 1
    row = payload[0]

    assert row["source_theme_id"] == source.id
    assert row["source_theme_name"] == "Contract Source"
    assert row["target_theme_id"] == target.id
    assert row["target_theme_name"] == "Contract Target"
    assert row["similarity_score"] == 0.88
    assert row["relationship_type"] == "identical"
    assert row["reasoning"] == "Same underlying concept"
    assert row["suggested_name"] == "Contract Target"

    # Legacy fields still present during migration window.
    assert row["source_cluster_id"] == source.id
    assert row["source_name"] == "Contract Source"
    assert row["target_cluster_id"] == target.id
    assert row["target_name"] == "Contract Target"
    assert row["embedding_similarity"] == 0.88
    assert row["llm_relationship"] == "identical"


def test_create_merge_suggestion_updates_legacy_row_without_canonical_pair_ids(db_session):
    source = _make_cluster(db_session, key="legacy_source", name="Legacy Source")
    target = _make_cluster(db_session, key="legacy_target", name="Legacy Target")
    db_session.commit()

    # Simulate a pre-migration row where canonical pair fields were not backfilled yet.
    legacy = ThemeMergeSuggestion(
        source_cluster_id=target.id,
        target_cluster_id=source.id,
        pair_min_cluster_id=None,
        pair_max_cluster_id=None,
        embedding_similarity=0.81,
        status="pending",
    )
    db_session.add(legacy)
    db_session.commit()

    service = _make_service(db_session)
    updated = service.create_merge_suggestion(source.id, target.id, 0.93, {"confidence": 0.88})

    assert updated is not None
    assert updated.id == legacy.id
    assert updated.pair_min_cluster_id == min(source.id, target.id)
    assert updated.pair_max_cluster_id == max(source.id, target.id)
    assert updated.embedding_similarity == 0.93
    assert db_session.query(ThemeMergeSuggestion).count() == 1

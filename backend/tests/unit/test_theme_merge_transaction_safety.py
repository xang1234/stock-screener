"""Tests for merge transaction safety and idempotent approval retries."""

from __future__ import annotations

from datetime import datetime, timedelta

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


def test_generate_dry_run_merge_plan_outputs_tiers_waves_without_mutation(db_session):
    first = _make_cluster(db_session, key="ai_infra", name="AI Infrastructure")
    second = _make_cluster(db_session, key="ai_datacenter", name="AI Datacenter")
    third = _make_cluster(db_session, key="defense_theme", name="Defense Theme")
    db_session.commit()

    service = _make_service(db_session)
    service.find_all_similar_pairs = lambda **kwargs: [
        {"theme1_id": first.id, "theme2_id": second.id, "similarity": 0.97, "pipeline": "technical"},
        {"theme1_id": second.id, "theme2_id": third.id, "similarity": 0.89, "pipeline": "technical"},
    ]

    def _fake_llm(theme_a, theme_b, similarity):
        if {theme_a.id, theme_b.id} == {first.id, second.id}:
            return {"should_merge": True, "confidence": 0.94, "relationship": "identical"}
        return {"should_merge": False, "confidence": 0.88, "relationship": "distinct"}

    service.verify_merge_with_llm = _fake_llm

    before_suggestions = db_session.query(ThemeMergeSuggestion).count()
    before_history = db_session.query(ThemeMergeHistory).count()
    report = service.generate_dry_run_merge_plan(limit_pairs=20, pipeline="technical")

    assert report["total_pairs_analyzed"] == 2
    assert any(bucket["tier"] == "high" and bucket["count"] >= 1 for bucket in report["confidence_distribution"])
    assert len(report["do_not_merge"]) == 1
    assert report["do_not_merge"][0]["relationship"] == "distinct"
    assert len(report["merge_groups"]) == 1
    assert report["merge_groups"][0]["theme_ids"] == [first.id, second.id]
    assert len(report["waves"]) >= 1

    # Dry-run planner must not mutate merge suggestions/history.
    assert db_session.query(ThemeMergeSuggestion).count() == before_suggestions
    assert db_session.query(ThemeMergeHistory).count() == before_history


def test_generate_dry_run_merge_plan_handles_non_numeric_confidence_and_contiguous_waves(db_session):
    first = _make_cluster(db_session, key="alpha_theme", name="Alpha Theme")
    second = _make_cluster(db_session, key="beta_theme", name="Beta Theme")
    third = _make_cluster(db_session, key="gamma_theme", name="Gamma Theme")
    db_session.commit()

    service = _make_service(db_session)
    service.find_all_similar_pairs = lambda **kwargs: [
        {"theme1_id": first.id, "theme2_id": second.id, "similarity": 0.92, "pipeline": "technical"},
        {"theme1_id": second.id, "theme2_id": third.id, "similarity": 0.87, "pipeline": "technical"},
    ]

    def _fake_llm(theme_a, theme_b, _similarity):
        if {theme_a.id, theme_b.id} == {first.id, second.id}:
            return {"should_merge": True, "confidence": "high", "relationship": "identical"}
        return {"should_merge": True, "confidence": 0.72, "relationship": "subset"}

    service.verify_merge_with_llm = _fake_llm
    report = service.generate_dry_run_merge_plan(limit_pairs=20, pipeline="technical")

    # Non-numeric confidence should not crash and should fallback to 0.0
    # Assert via analyzed outputs: a low-confidence row exists from malformed confidence.
    assert any(bucket["tier"] == "low" and bucket["count"] >= 1 for bucket in report["confidence_distribution"])

    # Waves should be contiguous starting from 1 regardless of missing high tier.
    if report["waves"]:
        assert [wave["wave"] for wave in report["waves"]] == list(range(1, len(report["waves"]) + 1))

    # Wave 3 pair_count includes low-group pair_count + ambiguity queue size.
    low_wave = next((wave for wave in report["waves"] if wave["confidence_tier"] == "low"), None)
    if low_wave is not None:
        low_group_pair_count = sum(
            group["pair_count"] for group in report["merge_groups"] if group["confidence_tier"] == "low"
        )
        assert low_wave["pair_count"] == low_group_pair_count + len(report["ambiguity_clusters"])


def test_run_strict_auto_merge_wave_dry_run_excludes_non_identical_relationships(db_session):
    first = _make_cluster(db_session, key="wave1_alpha", name="Wave1 Alpha")
    second = _make_cluster(db_session, key="wave1_beta", name="Wave1 Beta")
    third = _make_cluster(db_session, key="wave1_gamma", name="Wave1 Gamma")
    db_session.commit()

    service = _make_service(db_session)
    service.find_all_similar_pairs = lambda **kwargs: [
        {"theme1_id": first.id, "theme2_id": second.id, "similarity": 0.97, "pipeline": "technical"},
        {"theme1_id": second.id, "theme2_id": third.id, "similarity": 0.98, "pipeline": "technical"},
    ]

    def _fake_llm(theme_a, theme_b, _similarity):
        if {theme_a.id, theme_b.id} == {first.id, second.id}:
            return {"should_merge": True, "confidence": 0.95, "relationship": "identical"}
        return {"should_merge": True, "confidence": 0.97, "relationship": "subset"}

    service.verify_merge_with_llm = _fake_llm
    result = service.run_strict_auto_merge_wave(pipeline="technical", dry_run=True, limit_pairs=10)

    assert result["candidate_pairs"] == 2
    assert result["processed_pairs"] == 2
    assert result["eligible_pairs"] == 1
    assert result["auto_merged"] == 1
    assert result["reconciliation"]["active_themes_delta"] == 0
    assert any(
        action["reason"] == "excluded_relationship:subset"
        for action in result["merge_actions"]
    )
    assert db_session.query(ThemeMergeHistory).count() == 0


def test_run_strict_auto_merge_wave_executes_only_identical_high_confidence_pairs(db_session):
    first = _make_cluster(db_session, key="wave1_exec_alpha", name="Wave1 Exec Alpha")
    second = _make_cluster(db_session, key="wave1_exec_beta", name="Wave1 Exec Beta")
    third = _make_cluster(db_session, key="wave1_exec_gamma", name="Wave1 Exec Gamma")
    fourth = _make_cluster(db_session, key="wave1_exec_delta", name="Wave1 Exec Delta")
    db_session.add(
        ThemeConstituent(
            theme_cluster_id=first.id,
            symbol="NVDA",
            mention_count=3,
            confidence=0.9,
        )
    )
    db_session.add(
        ThemeConstituent(
            theme_cluster_id=second.id,
            symbol="AMD",
            mention_count=2,
            confidence=0.8,
        )
    )
    db_session.commit()

    service = _make_service(db_session)
    service.find_all_similar_pairs = lambda **kwargs: [
        {"theme1_id": first.id, "theme2_id": second.id, "similarity": 0.97, "pipeline": "technical"},
        {"theme1_id": third.id, "theme2_id": fourth.id, "similarity": 0.98, "pipeline": "technical"},
    ]

    def _fake_llm(theme_a, theme_b, _similarity):
        if {theme_a.id, theme_b.id} == {first.id, second.id}:
            return {"should_merge": True, "confidence": 0.95, "relationship": "identical"}
        return {"should_merge": True, "confidence": 0.99, "relationship": "related"}

    service.verify_merge_with_llm = _fake_llm
    result = service.run_strict_auto_merge_wave(pipeline="technical", dry_run=False, limit_pairs=10)

    assert result["candidate_pairs"] == 2
    assert result["eligible_pairs"] == 1
    assert result["auto_merged"] == 1
    assert result["reconciliation"]["active_themes_delta"] == -1
    assert result["reconciliation"]["merge_history_delta"] == 1
    assert result["reconciliation_package"]["wave_name"] == "wave1-strict-auto"
    assert len(result["reconciliation_package"]["artifact_hash"]) == 64
    assert result["reconciliation_package"]["reassignment_stats"]["merges_applied"] == 1
    assert len(result["reconciliation_package"]["rollback_references"]) == 1
    assert any(
        action["reason"] == "excluded_relationship:related"
        for action in result["merge_actions"]
    )

    active_clusters = db_session.query(ThemeCluster).filter(ThemeCluster.is_active == True).all()
    active_ids = {cluster.id for cluster in active_clusters}
    assert third.id in active_ids
    assert len(active_ids) == 3


def test_run_strict_auto_merge_wave_skips_pairs_with_rejected_suggestions(db_session):
    first = _make_cluster(db_session, key="wave1_reject_alpha", name="Wave1 Reject Alpha")
    second = _make_cluster(db_session, key="wave1_reject_beta", name="Wave1 Reject Beta")
    db_session.commit()
    db_session.add(
        ThemeMergeSuggestion(
            source_cluster_id=first.id,
            target_cluster_id=second.id,
            pair_min_cluster_id=min(first.id, second.id),
            pair_max_cluster_id=max(first.id, second.id),
            embedding_similarity=0.99,
            llm_confidence=0.99,
            llm_relationship="identical",
            status="rejected",
        )
    )
    db_session.commit()

    service = _make_service(db_session)
    service.find_all_similar_pairs = lambda **kwargs: [
        {"theme1_id": first.id, "theme2_id": second.id, "similarity": 0.99, "pipeline": "technical"},
    ]
    service.verify_merge_with_llm = lambda *_args, **_kwargs: {
        "should_merge": True,
        "confidence": 0.99,
        "relationship": "identical",
    }

    result = service.run_strict_auto_merge_wave(pipeline="technical", dry_run=False, limit_pairs=5)

    assert result["candidate_pairs"] == 1
    assert result["processed_pairs"] == 0
    assert result["eligible_pairs"] == 0
    assert result["auto_merged"] == 0
    assert result["errors"] == []
    assert any(
        reason["reason"] == "existing_suggestion" and reason["count"] == 1
        for reason in result["skip_reasons"]
    )
    assert db_session.query(ThemeMergeHistory).count() == 0


def test_run_manual_review_wave_tracks_sla_and_disagreement_with_audit_trail(db_session):
    source_a = _make_cluster(db_session, key="manual_source_a", name="Manual Source A")
    target_a = _make_cluster(db_session, key="manual_target_a", name="Manual Target A")
    source_b = _make_cluster(db_session, key="manual_source_b", name="Manual Source B")
    target_b = _make_cluster(db_session, key="manual_target_b", name="Manual Target B")
    db_session.commit()

    db_session.add(
        ThemeConstituent(
            theme_cluster_id=source_a.id,
            symbol="MSFT",
            mention_count=4,
            confidence=0.91,
        )
    )
    suggestion_a = ThemeMergeSuggestion(
        source_cluster_id=source_a.id,
        target_cluster_id=target_a.id,
        pair_min_cluster_id=min(source_a.id, target_a.id),
        pair_max_cluster_id=max(source_a.id, target_a.id),
        embedding_similarity=0.91,
        llm_confidence=0.78,
        llm_relationship="identical",
        status="pending",
        created_at=datetime.utcnow() - timedelta(hours=30),
    )
    suggestion_b = ThemeMergeSuggestion(
        source_cluster_id=source_b.id,
        target_cluster_id=target_b.id,
        pair_min_cluster_id=min(source_b.id, target_b.id),
        pair_max_cluster_id=max(source_b.id, target_b.id),
        embedding_similarity=0.89,
        llm_confidence=0.76,
        llm_relationship="subset",
        status="pending",
        created_at=datetime.utcnow() - timedelta(hours=2),
    )
    db_session.add_all([suggestion_a, suggestion_b])
    db_session.commit()

    service = _make_service(db_session)
    result = service.run_manual_review_wave(
        pipeline="technical",
        decisions=[
            {"suggestion_id": suggestion_a.id, "action": "approve", "reviewer": "analyst-a"},
            {"suggestion_id": suggestion_b.id, "action": "reject", "reviewer": "analyst-b", "note": "Needs deeper review"},
        ],
        sla_target_hours=24.0,
        queue_limit=20,
        dry_run=False,
    )

    assert result["queue_size"] == 2
    assert result["reviewed"] == 2
    assert result["approved"] == 1
    assert result["rejected"] == 1
    assert result["errors"] == 0
    assert result["queue_closed"] is True
    assert result["pending_after"] == 0
    assert result["metrics"]["sla_breaches"] == 1
    assert result["metrics"]["disagreement_rate"] == 0.5
    assert result["reconciliation"]["merge_history_delta"] == 1
    assert result["reconciliation_package"]["wave_name"] == "wave2-manual-review"
    assert len(result["reconciliation_package"]["artifact_hash"]) == 64
    assert result["reconciliation_package"]["reassignment_stats"]["merges_applied"] == 1
    assert len(result["reconciliation_package"]["rollback_references"]) == 1
    assert db_session.query(ThemeMergeHistory).count() == 1
    rejected = db_session.query(ThemeMergeSuggestion).filter(ThemeMergeSuggestion.id == suggestion_b.id).one()
    assert rejected.status == "rejected"
    assert rejected.approval_result_json is not None
    assert any(row["status"] == "approved" for row in result["audit_trail"])
    assert any(row["status"] == "rejected" for row in result["audit_trail"])


def test_run_manual_review_wave_skips_invalid_actions_without_mutation(db_session):
    source = _make_cluster(db_session, key="manual_skip_source", name="Manual Skip Source")
    target = _make_cluster(db_session, key="manual_skip_target", name="Manual Skip Target")
    db_session.commit()
    suggestion = ThemeMergeSuggestion(
        source_cluster_id=source.id,
        target_cluster_id=target.id,
        pair_min_cluster_id=min(source.id, target.id),
        pair_max_cluster_id=max(source.id, target.id),
        embedding_similarity=0.90,
        llm_confidence=0.72,
        llm_relationship="identical",
        status="pending",
    )
    db_session.add(suggestion)
    db_session.commit()

    service = _make_service(db_session)
    result = service.run_manual_review_wave(
        pipeline="technical",
        decisions=[{"suggestion_id": suggestion.id, "action": "defer", "reviewer": "analyst-c"}],
        dry_run=False,
    )

    assert result["queue_size"] == 1
    assert result["reviewed"] == 0
    assert result["approved"] == 0
    assert result["rejected"] == 0
    assert result["skipped"] == 1
    assert result["pending_after"] == 1
    refreshed = db_session.query(ThemeMergeSuggestion).filter(ThemeMergeSuggestion.id == suggestion.id).one()
    assert refreshed.status == "pending"


def test_run_manual_review_wave_scans_past_ineligible_rows_before_limit(db_session):
    ineligible_source = _make_cluster(db_session, key="manual_ineligible_source", name="Manual Ineligible Source")
    ineligible_target = _make_cluster(db_session, key="manual_ineligible_target", name="Manual Ineligible Target")
    eligible_source = _make_cluster(db_session, key="manual_eligible_source", name="Manual Eligible Source")
    eligible_target = _make_cluster(db_session, key="manual_eligible_target", name="Manual Eligible Target")
    db_session.commit()
    db_session.add(
        ThemeConstituent(
            theme_cluster_id=eligible_source.id,
            symbol="AAPL",
            mention_count=2,
            confidence=0.8,
        )
    )
    # Higher-confidence but ineligible row (related relationship) comes first in ordering.
    db_session.add(
        ThemeMergeSuggestion(
            source_cluster_id=ineligible_source.id,
            target_cluster_id=ineligible_target.id,
            pair_min_cluster_id=min(ineligible_source.id, ineligible_target.id),
            pair_max_cluster_id=max(ineligible_source.id, ineligible_target.id),
            embedding_similarity=0.95,
            llm_confidence=0.99,
            llm_relationship="related",
            status="pending",
        )
    )
    eligible = ThemeMergeSuggestion(
        source_cluster_id=eligible_source.id,
        target_cluster_id=eligible_target.id,
        pair_min_cluster_id=min(eligible_source.id, eligible_target.id),
        pair_max_cluster_id=max(eligible_source.id, eligible_target.id),
        embedding_similarity=0.90,
        llm_confidence=0.76,
        llm_relationship="identical",
        status="pending",
    )
    db_session.add(eligible)
    db_session.commit()

    service = _make_service(db_session)
    result = service.run_manual_review_wave(
        pipeline="technical",
        decisions=[{"suggestion_id": eligible.id, "action": "approve", "reviewer": "analyst-z"}],
        queue_limit=1,
        dry_run=False,
    )

    assert result["queue_size"] == 1
    assert result["reviewed"] == 1
    assert result["approved"] == 1
    assert result["errors"] == 0
    assert db_session.query(ThemeMergeHistory).count() == 1

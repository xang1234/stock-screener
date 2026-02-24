"""Tests for SqlThemeAliasRepository using in-memory SQLite."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.infra.db.repositories.theme_alias_repo import SqlThemeAliasRepository
from app.models.theme import ThemeAlias, ThemeCluster


@pytest.fixture
def repo(session: Session) -> SqlThemeAliasRepository:
    return SqlThemeAliasRepository(session)


def _make_cluster(session: Session, *, key: str, name: str, pipeline: str = "technical") -> ThemeCluster:
    cluster = ThemeCluster(
        canonical_key=key,
        display_name=name,
        name=name,
        pipeline=pipeline,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
        is_active=True,
    )
    session.add(cluster)
    session.flush()
    return cluster


class TestRecordObservation:
    def test_creates_alias_row(self, repo: SqlThemeAliasRepository, session: Session):
        cluster = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")

        row = repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
            source="llm_extraction",
            confidence=0.8,
        )

        assert row.id is not None
        assert row.alias_key == "ai_infrastructure"
        assert row.evidence_count == 1
        assert row.confidence == 0.8

    def test_updates_existing_alias_quality_metrics(self, repo: SqlThemeAliasRepository, session: Session):
        cluster = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")
        repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
            confidence=0.8,
        )

        updated = repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="A.I. Infrastructure",
            confidence=0.4,
        )

        assert updated.evidence_count == 2
        assert 0.59 < updated.confidence < 0.61
        assert updated.alias_text == "A.I. Infrastructure"

    def test_does_not_reassign_existing_alias_to_other_cluster(self, repo: SqlThemeAliasRepository, session: Session):
        cluster_a = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")
        cluster_b = _make_cluster(session, key="ai_semiconductors", name="AI Semiconductors")
        created = repo.record_observation(
            theme_cluster_id=cluster_a.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
        )

        updated = repo.record_observation(
            theme_cluster_id=cluster_b.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
            confidence=0.9,
        )

        assert updated.id == created.id
        assert updated.theme_cluster_id == cluster_a.id

    def test_rejects_unknown_alias_key(self, repo: SqlThemeAliasRepository, session: Session):
        cluster = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")

        with pytest.raises(ValueError, match="unknown_theme"):
            repo.record_observation(
                theme_cluster_id=cluster.id,
                pipeline="technical",
                alias_text="!!!",
            )

    def test_reactivates_inactive_alias_and_rebinds_cluster(self, repo: SqlThemeAliasRepository, session: Session):
        cluster_a = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")
        cluster_b = _make_cluster(session, key="ai_semiconductors", name="AI Semiconductors")
        created = repo.record_observation(
            theme_cluster_id=cluster_a.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
            source="manual",
        )
        created.is_active = False
        session.flush()

        updated = repo.record_observation(
            theme_cluster_id=cluster_b.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
            source="llm_extraction",
            confidence=0.9,
        )

        assert updated.id == created.id
        assert updated.is_active is True
        assert updated.theme_cluster_id == cluster_b.id
        assert updated.source == "manual"


class TestLookupAndDeactivate:
    def test_find_exact_returns_active_row(self, repo: SqlThemeAliasRepository, session: Session):
        cluster = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")
        repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
        )

        found = repo.find_exact(pipeline="technical", alias_key="ai_infrastructure")

        assert found is not None
        assert found.theme_cluster_id == cluster.id

    def test_deactivate_hides_from_exact_lookup(self, repo: SqlThemeAliasRepository, session: Session):
        cluster = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")
        repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
        )

        assert repo.deactivate(pipeline="technical", alias_key="ai_infrastructure") is True
        assert repo.find_exact(pipeline="technical", alias_key="ai_infrastructure") is None

    def test_list_for_cluster_orders_by_quality_and_recency(self, repo: SqlThemeAliasRepository, session: Session):
        cluster = _make_cluster(session, key="ai_infrastructure", name="AI Infrastructure")
        older = datetime.utcnow() - timedelta(days=1)
        newer = datetime.utcnow()

        a = repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Infrastructure",
            confidence=0.9,
            seen_at=older,
        )
        b = repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Buildout",
            confidence=0.7,
            seen_at=newer,
        )
        # Boost evidence_count for b so it sorts first.
        repo.record_observation(
            theme_cluster_id=cluster.id,
            pipeline="technical",
            alias_text="AI Buildout",
            confidence=0.8,
            seen_at=newer,
        )

        rows = repo.list_for_cluster(theme_cluster_id=cluster.id)
        assert rows[0].id == b.id
        assert {rows[0].id, rows[1].id} == {a.id, b.id}

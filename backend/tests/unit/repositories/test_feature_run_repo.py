"""Tests for SqlFeatureRunRepository using in-memory SQLite.

Uses a real SQLAlchemy engine against :memory: SQLite — not mocks.
This catches column-name errors, constraint violations, and JSON
serialization issues that mocks would miss.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session

from app.domain.common.errors import EntityNotFoundError, InvalidTransitionError
from app.domain.feature_store.models import (
    FeatureRunDomain,
    RunStats,
    RunStatus,
    RunType,
)
from app.domain.feature_store.quality import DQResult, DQSeverity
from app.infra.db.models.feature_store import (  # noqa: F401 — register models
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.infra.db.repositories.feature_run_repo import SqlFeatureRunRepository


@pytest.fixture
def repo(session: Session) -> SqlFeatureRunRepository:
    return SqlFeatureRunRepository(session)


def _make_stats() -> RunStats:
    return RunStats(
        total_symbols=100,
        processed_symbols=95,
        failed_symbols=5,
        duration_seconds=12.5,
    )


class TestStartRun:
    def test_creates_running_record(self, repo: SqlFeatureRunRepository, session: Session):
        result = repo.start_run(
            as_of_date=date(2026, 2, 17),
            run_type=RunType.DAILY_SNAPSHOT,
            code_version="v1.0.0",
        )
        assert result.status == RunStatus.RUNNING
        assert result.as_of_date == date(2026, 2, 17)
        assert result.code_version == "v1.0.0"

        # Verify it's actually in the DB
        row = session.get(FeatureRun, result.id)
        assert row is not None
        assert row.status == "running"

    def test_returns_domain_object(self, repo: SqlFeatureRunRepository):
        result = repo.start_run(
            as_of_date=date(2026, 1, 1),
            run_type=RunType.MANUAL,
            correlation_id="corr-123",
            universe_hash="abc",
            input_hash="def",
        )
        assert isinstance(result, FeatureRunDomain)
        assert result.id is not None
        assert result.run_type == RunType.MANUAL
        assert result.correlation_id == "corr-123"
        assert result.universe_hash == "abc"
        assert result.input_hash == "def"
        assert result.completed_at is None
        assert result.stats is None
        assert result.warnings == ()


class TestMarkCompleted:
    def test_transitions_and_stores_stats(self, repo: SqlFeatureRunRepository):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        stats = _make_stats()

        result = repo.mark_completed(run.id, stats, warnings=("low coverage",))

        assert result.status == RunStatus.COMPLETED
        assert result.completed_at is not None
        assert result.stats == stats
        assert result.warnings == ("low coverage",)

    def test_nonexistent_raises_not_found(self, repo: SqlFeatureRunRepository):
        with pytest.raises(EntityNotFoundError):
            repo.mark_completed(9999, _make_stats())

    def test_wrong_status_raises_invalid_transition(self, repo: SqlFeatureRunRepository):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())

        # Already COMPLETED — can't complete again
        with pytest.raises(InvalidTransitionError):
            repo.mark_completed(run.id, _make_stats())


class TestMarkQuarantined:
    def test_stores_dq_results(self, repo: SqlFeatureRunRepository):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())

        dq_results = [
            DQResult(
                check_name="row_count",
                passed=False,
                severity=DQSeverity.CRITICAL,
                actual_value=0.5,
                threshold=0.9,
                message="Row count too low",
            ),
        ]
        result = repo.mark_quarantined(run.id, dq_results)

        assert result.status == RunStatus.QUARANTINED
        assert "Row count too low" in result.warnings


class TestPublishAtomically:
    def test_updates_status_and_pointer(self, repo: SqlFeatureRunRepository, session: Session):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())

        result = repo.publish_atomically(run.id)

        assert result.status == RunStatus.PUBLISHED
        # Verify pointer was created
        pointer = (
            session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        assert pointer is not None
        assert pointer.run_id == run.id

    def test_creates_pointer_if_missing(self, repo: SqlFeatureRunRepository, session: Session):
        """First publish ever — pointer row should be created."""
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())

        repo.publish_atomically(run.id)

        pointer = (
            session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        assert pointer is not None

    def test_updates_existing_pointer(self, repo: SqlFeatureRunRepository, session: Session):
        """Second publish — pointer should be updated, not duplicated."""
        run1 = repo.start_run(date(2026, 2, 16), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run1.id, _make_stats())
        repo.publish_atomically(run1.id)

        run2 = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run2.id, _make_stats())
        repo.publish_atomically(run2.id)

        pointers = (
            session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .all()
        )
        assert len(pointers) == 1
        assert pointers[0].run_id == run2.id

    def test_non_completed_raises_invalid_transition(self, repo: SqlFeatureRunRepository):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        # Still RUNNING — can't publish
        with pytest.raises(InvalidTransitionError):
            repo.publish_atomically(run.id)

    def test_publish_from_quarantined_state(self, repo: SqlFeatureRunRepository, session: Session):
        """QUARANTINED → PUBLISHED is a valid override transition."""
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())
        repo.mark_quarantined(
            run.id,
            [
                DQResult(
                    check_name="row_count",
                    passed=False,
                    severity=DQSeverity.CRITICAL,
                    actual_value=0.5,
                    threshold=0.9,
                    message="Row count too low",
                ),
            ],
        )

        result = repo.publish_atomically(run.id)

        assert result.status == RunStatus.PUBLISHED
        pointer = (
            session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        assert pointer is not None
        assert pointer.run_id == run.id

    def test_rollback_prevents_partial_publish(self, repo: SqlFeatureRunRepository, session: Session):
        """Atomic crash simulation: commit setup, then flush publish + rollback."""
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())
        # Commit the "completed" state so it survives rollback
        session.commit()

        # Publish flushes but we DON'T commit
        repo.publish_atomically(run.id)
        # Simulate crash — rollback
        session.rollback()

        # Status should be reverted to "completed"
        row = session.get(FeatureRun, run.id)
        assert row.status == RunStatus.COMPLETED.value

        # Pointer should not exist
        pointer = (
            session.query(FeatureRunPointer)
            .filter(FeatureRunPointer.key == "latest_published")
            .first()
        )
        assert pointer is None


class TestGetLatestPublished:
    def test_returns_published_run(self, repo: SqlFeatureRunRepository):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        repo.mark_completed(run.id, _make_stats())
        repo.publish_atomically(run.id)

        result = repo.get_latest_published()
        assert result is not None
        assert result.id == run.id
        assert result.status == RunStatus.PUBLISHED

    def test_returns_none_when_no_pointer(self, repo: SqlFeatureRunRepository):
        assert repo.get_latest_published() is None


class TestGetRun:
    def test_returns_domain_object(self, repo: SqlFeatureRunRepository):
        run = repo.start_run(date(2026, 2, 17), RunType.DAILY_SNAPSHOT)
        result = repo.get_run(run.id)
        assert isinstance(result, FeatureRunDomain)
        assert result.id == run.id

    def test_nonexistent_raises_not_found(self, repo: SqlFeatureRunRepository):
        with pytest.raises(EntityNotFoundError):
            repo.get_run(9999)

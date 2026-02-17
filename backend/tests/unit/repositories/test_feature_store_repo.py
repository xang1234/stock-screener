"""Tests for SqlFeatureStoreRepository using in-memory SQLite.

Uses a real SQLAlchemy engine against :memory: SQLite to verify
UPSERT behavior, batching, pointer-based queries, and pagination.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.domain.common.errors import EntityNotFoundError
from app.domain.common.query import FilterSpec, PageSpec, SortSpec
from app.domain.feature_store.models import FeaturePage, FeatureRow, FeatureRowWrite
from app.infra.db.models.feature_store import (  # noqa: F401 — register models
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository


@pytest.fixture
def session():
    """Create an in-memory SQLite session with all feature store tables."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_fk_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    sess = factory()
    yield sess
    sess.close()


def _create_run(session: Session, as_of_date: date = date(2026, 2, 17), status: str = "completed") -> int:
    """Insert a FeatureRun directly and return its id."""
    run = FeatureRun(
        as_of_date=as_of_date,
        run_type="daily_snapshot",
        status=status,
    )
    session.add(run)
    session.flush()
    return run.id


def _set_pointer(session: Session, run_id: int) -> None:
    """Set the 'latest_published' pointer."""
    pointer = FeatureRunPointer(key="latest_published", run_id=run_id)
    session.add(pointer)
    session.flush()


def _make_row(symbol: str, as_of_date: date = date(2026, 2, 17), score: float = 80.0) -> FeatureRowWrite:
    return FeatureRowWrite(
        symbol=symbol,
        as_of_date=as_of_date,
        composite_score=score,
        overall_rating=3,
        passes_count=2,
        details={"minervini_score": score, "rs_rating": 85.0},
    )


@pytest.fixture
def repo(session: Session) -> SqlFeatureStoreRepository:
    return SqlFeatureStoreRepository(session)


class TestUpsertSnapshotRows:
    def test_persists_rows(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        rows = [_make_row("AAPL"), _make_row("MSFT", score=90.0)]

        count = repo.upsert_snapshot_rows(run_id, rows)

        assert count == 2
        db_count = session.query(StockFeatureDaily).filter_by(run_id=run_id).count()
        assert db_count == 2

    def test_updates_on_conflict(self, repo: SqlFeatureStoreRepository, session: Session):
        """Re-inserting the same (run_id, symbol) should update, not error."""
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(run_id, [_make_row("AAPL", score=80.0)])

        # Upsert same symbol with different score
        repo.upsert_snapshot_rows(run_id, [_make_row("AAPL", score=95.0)])

        rows = session.query(StockFeatureDaily).filter_by(run_id=run_id, symbol="AAPL").all()
        assert len(rows) == 1
        assert rows[0].composite_score == 95.0

    def test_large_batch_succeeds(self, repo: SqlFeatureStoreRepository, session: Session):
        """More than 500 rows should work (verifies internal batching)."""
        run_id = _create_run(session)
        rows = [_make_row(f"SYM{i:04d}", score=float(i)) for i in range(600)]

        count = repo.upsert_snapshot_rows(run_id, rows)

        assert count == 600
        db_count = session.query(StockFeatureDaily).filter_by(run_id=run_id).count()
        assert db_count == 600


class TestSaveRunUniverseSymbols:
    def test_persists_symbols(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        repo.save_run_universe_symbols(run_id, ["AAPL", "MSFT", "GOOG"])

        count = session.query(FeatureRunUniverseSymbol).filter_by(run_id=run_id).count()
        assert count == 3


class TestCountByRunId:
    def test_returns_count(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(run_id, [_make_row("AAPL"), _make_row("MSFT")])

        assert repo.count_by_run_id(run_id) == 2

    def test_returns_zero_for_empty_run(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        assert repo.count_by_run_id(run_id) == 0


class TestQueryRun:
    def test_returns_paginated_results(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        rows = [_make_row(f"SYM{i}", score=float(i)) for i in range(10)]
        repo.upsert_snapshot_rows(run_id, rows)

        result = repo.query_run(run_id, page=PageSpec(page=1, per_page=5))

        assert isinstance(result, FeaturePage)
        assert len(result.items) == 5
        assert result.total == 10
        assert result.total_pages == 2

    def test_nonexistent_raises_not_found(self, repo: SqlFeatureStoreRepository):
        with pytest.raises(EntityNotFoundError):
            repo.query_run(9999)

    def test_items_are_feature_rows(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(run_id, [_make_row("AAPL", score=85.0)])

        result = repo.query_run(run_id)
        assert len(result.items) == 1
        item = result.items[0]
        assert isinstance(item, FeatureRow)
        assert item.symbol == "AAPL"
        assert item.composite_score == 85.0
        assert item.run_id == run_id
        assert item.details == {"minervini_score": 85.0, "rs_rating": 85.0}


class TestQueryLatest:
    def test_uses_pointer(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session, status="published")
        _set_pointer(session, run_id)
        repo.upsert_snapshot_rows(run_id, [_make_row("AAPL"), _make_row("MSFT")])

        result = repo.query_latest()

        assert result.total == 2
        symbols = {item.symbol for item in result.items}
        assert symbols == {"AAPL", "MSFT"}

    def test_returns_empty_when_no_pointer(self, repo: SqlFeatureStoreRepository):
        result = repo.query_latest()

        assert result.total == 0
        assert result.items == ()
        assert result.page == 1

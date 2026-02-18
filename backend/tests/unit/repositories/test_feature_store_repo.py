"""Tests for SqlFeatureStoreRepository using in-memory SQLite.

Uses a real SQLAlchemy engine against :memory: SQLite to verify
UPSERT behavior, batching, pointer-based queries, pagination,
DQ inputs, FilterSpec, sort ordering, and N+1 detection.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session

from app.domain.common.errors import EntityNotFoundError
from app.domain.common.query import (
    FilterSpec,
    PageSpec,
    SortOrder,
    SortSpec,
)
from app.domain.feature_store.models import FeaturePage, FeatureRow, FeatureRowWrite
from app.domain.feature_store.quality import DQInputs
from app.infra.db.models.feature_store import (  # noqa: F401 — register models
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository

from .conftest import count_queries


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


# ---------------------------------------------------------------------------
# TestGetRunDqInputs
# ---------------------------------------------------------------------------


class TestGetRunDqInputs:
    def test_builds_dq_inputs_from_persisted_data(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(
            run_id,
            [_make_row("AAPL", score=80.0), _make_row("MSFT", score=90.0), _make_row("GOOG", score=70.0)],
        )
        repo.save_run_universe_symbols(run_id, ["AAPL", "MSFT", "GOOG"])

        dq = repo.get_run_dq_inputs(run_id)

        assert isinstance(dq, DQInputs)
        assert dq.expected_row_count == 3
        assert dq.actual_row_count == 3
        assert dq.null_score_count == 0
        assert dq.total_row_count == 3
        assert set(dq.scores) == {80.0, 90.0, 70.0}
        assert len(dq.ratings) == 3
        assert set(dq.universe_symbols) == {"AAPL", "MSFT", "GOOG"}
        assert set(dq.result_symbols) == {"AAPL", "MSFT", "GOOG"}

    def test_counts_null_scores(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        rows = [
            _make_row("AAPL", score=80.0),
            FeatureRowWrite(
                symbol="NULL_SYM",
                as_of_date=date(2026, 2, 17),
                composite_score=None,
                overall_rating=None,
                passes_count=0,
                details=None,
            ),
        ]
        repo.upsert_snapshot_rows(run_id, rows)

        dq = repo.get_run_dq_inputs(run_id)

        assert dq.null_score_count == 1
        assert dq.actual_row_count == 2
        # Null score excluded from scores tuple
        assert dq.scores == (80.0,)

    def test_empty_run_returns_zeros(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)

        dq = repo.get_run_dq_inputs(run_id)

        assert dq.expected_row_count == 0
        assert dq.actual_row_count == 0
        assert dq.null_score_count == 0
        assert dq.scores == ()
        assert dq.ratings == ()
        assert dq.universe_symbols == ()
        assert dq.result_symbols == ()

    def test_universe_without_results(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        repo.save_run_universe_symbols(run_id, ["AAPL", "MSFT"])

        dq = repo.get_run_dq_inputs(run_id)

        assert dq.expected_row_count == 2
        assert dq.actual_row_count == 0

    def test_results_without_universe(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(run_id, [_make_row("AAPL"), _make_row("MSFT")])

        dq = repo.get_run_dq_inputs(run_id)

        assert dq.expected_row_count == 0
        assert dq.actual_row_count == 2


# ---------------------------------------------------------------------------
# TestFilterSpec
# ---------------------------------------------------------------------------


class TestFilterSpec:
    def test_range_filter_on_composite_score(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        scores = [10.0, 30.0, 50.0, 70.0, 90.0]
        rows = [_make_row(f"SYM{i}", score=s) for i, s in enumerate(scores)]
        repo.upsert_snapshot_rows(run_id, rows)

        filters = FilterSpec()
        filters.add_range("composite_score", min_value=40.0, max_value=80.0)
        result = repo.query_run(run_id, filters=filters)

        assert result.total == 2
        result_scores = {item.composite_score for item in result.items}
        assert result_scores == {50.0, 70.0}

    def test_range_filter_on_json_field(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        """Verifies json_extract + CAST path via _JSON_FIELD_MAP."""
        run_id = _create_run(session)
        rows = [
            FeatureRowWrite(
                symbol="LOW", as_of_date=date(2026, 2, 17),
                composite_score=50.0, overall_rating=2, passes_count=1,
                details={"minervini_score": 20.0, "rs_rating": 40.0},
            ),
            FeatureRowWrite(
                symbol="HIGH", as_of_date=date(2026, 2, 17),
                composite_score=80.0, overall_rating=4, passes_count=3,
                details={"minervini_score": 85.0, "rs_rating": 95.0},
            ),
        ]
        repo.upsert_snapshot_rows(run_id, rows)

        filters = FilterSpec()
        filters.add_range("minervini_score", min_value=50.0)
        result = repo.query_run(run_id, filters=filters)

        assert result.total == 1
        assert result.items[0].symbol == "HIGH"

    def test_categorical_filter_include(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        for sym in ["AAPL", "MSFT", "GOOG", "TSLA"]:
            repo.upsert_snapshot_rows(run_id, [_make_row(sym)])

        filters = FilterSpec()
        filters.add_categorical("symbol", ("AAPL", "MSFT"))
        result = repo.query_run(run_id, filters=filters)

        assert result.total == 2
        symbols = {item.symbol for item in result.items}
        assert symbols == {"AAPL", "MSFT"}

    def test_text_search_on_symbol(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        for sym in ["AAPL", "MSFT", "GOOG"]:
            repo.upsert_snapshot_rows(run_id, [_make_row(sym)])

        filters = FilterSpec()
        filters.add_text_search("symbol", "AA")
        result = repo.query_run(run_id, filters=filters)

        assert result.total == 1
        assert result.items[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# TestSortOrdering
# ---------------------------------------------------------------------------


class TestSortOrdering:
    def test_sort_composite_score_ascending(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(
            run_id,
            [_make_row("C", score=30.0), _make_row("A", score=10.0), _make_row("B", score=20.0)],
        )

        result = repo.query_run(
            run_id, sort=SortSpec(field="composite_score", order=SortOrder.ASC)
        )

        scores = [item.composite_score for item in result.items]
        assert scores == [10.0, 20.0, 30.0]

    def test_sort_composite_score_descending(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(
            run_id,
            [_make_row("C", score=30.0), _make_row("A", score=10.0), _make_row("B", score=20.0)],
        )

        result = repo.query_run(
            run_id, sort=SortSpec(field="composite_score", order=SortOrder.DESC)
        )

        scores = [item.composite_score for item in result.items]
        assert scores == [30.0, 20.0, 10.0]


# ---------------------------------------------------------------------------
# TestPagination
# ---------------------------------------------------------------------------


class TestPagination:
    def test_page_beyond_total_returns_empty_items(
        self, repo: SqlFeatureStoreRepository, session: Session
    ):
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(
            run_id,
            [_make_row("A"), _make_row("B"), _make_row("C")],
        )

        result = repo.query_run(run_id, page=PageSpec(page=5, per_page=10))

        assert result.items == ()
        assert result.total == 3


# ---------------------------------------------------------------------------
# TestQueryRunNoNPlusOne
# ---------------------------------------------------------------------------


class TestQueryRunNoNPlusOne:
    def test_query_run_uses_constant_queries(
        self, repo: SqlFeatureStoreRepository, session: Session, engine
    ):
        """Verify query_run executes at most 3 SQL statements (GET + COUNT + SELECT)."""
        run_id = _create_run(session)
        repo.upsert_snapshot_rows(
            run_id,
            [_make_row(f"SYM{i}") for i in range(20)],
        )

        with count_queries(engine) as counter:
            repo.query_run(run_id, page=PageSpec(page=1, per_page=10))

        # 1 = session.get(FeatureRun, run_id), 2 = COUNT(*), 3 = SELECT...LIMIT
        assert counter["count"] <= 3, f"Expected <=3 queries, got {counter['count']}"


# ---------------------------------------------------------------------------
# TestUpsertSnapshotRows (extension)
# ---------------------------------------------------------------------------


class TestUpsertSnapshotRowsExtended:
    def test_empty_rows_returns_zero(self, repo: SqlFeatureStoreRepository, session: Session):
        run_id = _create_run(session)
        assert repo.upsert_snapshot_rows(run_id, []) == 0

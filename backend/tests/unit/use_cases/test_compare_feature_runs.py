"""Tests for CompareFeatureRunsUseCase."""

from datetime import date

import pytest

from app.domain.common.errors import EntityNotFoundError, ValidationError
from app.domain.feature_store.models import (
    FeatureRow,
    FeatureRowWrite,
    RunStats,
    RunType,
)
from app.use_cases.feature_store.compare_runs import (
    CompareFeatureRunsUseCase,
    CompareRunsQuery,
)
from tests.unit.use_cases.conftest import (
    FakeFeatureRunRepository,
    FakeFeatureStoreRepository,
    FakeUnitOfWork,
)


def _make_uow() -> tuple[FakeUnitOfWork, FakeFeatureRunRepository, FakeFeatureStoreRepository]:
    store = FakeFeatureStoreRepository()
    runs = FakeFeatureRunRepository(row_counter=store.count_by_run_id)
    uow = FakeUnitOfWork(feature_runs=runs, feature_store=store)
    return uow, runs, store


def _seed_run(runs_repo, store, as_of, rows_data):
    """Create a published run with given symbol/score/rating rows."""
    r = runs_repo.start_run(as_of, RunType.DAILY_SNAPSHOT)
    runs_repo.mark_completed(r.id, RunStats(len(rows_data), len(rows_data), 0, 10.0))
    runs_repo.publish_atomically(r.id)
    store.upsert_snapshot_rows(r.id, [
        FeatureRowWrite(sym, as_of, score, rating, 1, {})
        for sym, score, rating in rows_data
    ])
    return r


class TestCompareFeatureRunsUseCase:
    """Test suite for CompareFeatureRunsUseCase."""

    def test_identical_runs_no_movers(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        rows = [("AAPL", 80.0, 4), ("MSFT", 70.0, 3)]
        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), rows)
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), rows)

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id))
        assert result.summary.total_common == 2
        assert len(result.movers) == 0
        assert len(result.added) == 0
        assert len(result.removed) == 0
        assert result.summary.avg_score_change == 0.0

    def test_added_removed_detected(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), [
            ("AAPL", 80.0, 4), ("OLD", 50.0, 2),
        ])
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), [
            ("AAPL", 80.0, 4), ("NEW", 65.0, 3),
        ])

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id))
        added_syms = [e.symbol for e in result.added]
        removed_syms = [e.symbol for e in result.removed]
        assert "NEW" in added_syms
        assert "OLD" in removed_syms
        assert result.added[0].score == 65.0

    def test_score_deltas_sorted_by_abs(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), [
            ("AAPL", 50.0, 3), ("MSFT", 60.0, 3), ("GOOG", 70.0, 4),
        ])
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), [
            ("AAPL", 80.0, 4), ("MSFT", 65.0, 3), ("GOOG", 50.0, 2),
        ])

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id))
        # AAPL delta=30, GOOG delta=-20, MSFT delta=5
        assert len(result.movers) == 3
        assert result.movers[0].symbol == "AAPL"  # |30|
        assert result.movers[0].score_delta == 30.0
        assert result.movers[1].symbol == "GOOG"  # |-20|
        assert result.movers[1].score_delta == -20.0
        assert result.movers[2].symbol == "MSFT"  # |5|
        assert result.movers[2].score_delta == 5.0

    def test_equal_abs_score_deltas_are_sorted_by_symbol(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), [
            ("SNOW", 60.0, 3),
            ("PANW", 79.0, 4),
            ("NVDA", 88.0, 4),
        ])
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), [
            ("SNOW", 55.0, 3),
            ("PANW", 84.0, 5),
            ("NVDA", 92.0, 5),
        ])

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id))

        assert [delta.symbol for delta in result.movers] == ["PANW", "SNOW", "NVDA"]

    def test_rating_upgrades_downgrades(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), [
            ("UP", 50.0, 3),   # Watch → Buy (upgrade)
            ("DOWN", 80.0, 4), # Buy → Watch (downgrade)
            ("SAME", 60.0, 3), # Watch → Watch (no change)
        ])
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), [
            ("UP", 75.0, 4),
            ("DOWN", 55.0, 3),
            ("SAME", 60.0, 3),
        ])

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id))
        assert result.summary.upgraded_count == 1
        assert result.summary.downgraded_count == 1

    def test_limit_respected(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        # Create 10 symbols with different score deltas
        rows_a = [(f"S{i:02d}", 50.0, 3) for i in range(10)]
        rows_b = [(f"S{i:02d}", 50.0 + (i + 1) * 5, 3) for i in range(10)]
        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), rows_a)
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), rows_b)

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id, limit=3))
        assert len(result.movers) == 3

    def test_same_run_raises(self):
        with pytest.raises(ValidationError, match="different"):
            CompareRunsQuery(run_a=1, run_b=1)

    def test_limit_too_low_raises(self):
        with pytest.raises(ValidationError, match="limit"):
            CompareRunsQuery(run_a=1, run_b=2, limit=0)

    def test_limit_too_high_raises(self):
        with pytest.raises(ValidationError, match="limit"):
            CompareRunsQuery(run_a=1, run_b=2, limit=501)

    def test_run_not_found_raises(self):
        uow, _, _ = _make_uow()
        uc = CompareFeatureRunsUseCase()
        with pytest.raises(EntityNotFoundError):
            uc.execute(uow, CompareRunsQuery(run_a=999, run_b=998))

    def test_added_sorted_by_score_desc(self):
        uow, runs_repo, store = _make_uow()
        uc = CompareFeatureRunsUseCase()

        r_a = _seed_run(runs_repo, store, date(2026, 2, 15), [])
        r_b = _seed_run(runs_repo, store, date(2026, 2, 16), [
            ("LOW", 30.0, 2), ("HIGH", 90.0, 5), ("MID", 60.0, 3),
        ])

        result = uc.execute(uow, CompareRunsQuery(run_a=r_a.id, run_b=r_b.id))
        assert [e.symbol for e in result.added] == ["HIGH", "MID", "LOW"]

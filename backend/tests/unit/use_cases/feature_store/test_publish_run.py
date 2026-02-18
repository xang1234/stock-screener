"""Tests for PublishFeatureRunUseCase.

Uses in-memory fakes from conftest.py — no DB, no mocks, pure behaviour tests.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.common.errors import ValidationError
from app.domain.feature_store.models import RunStatus, RunType
from app.domain.feature_store.quality import DQInputs, DQThresholds
from app.use_cases.feature_store.publish_run import (
    PublishFeatureRunUseCase,
    PublishRunCommand,
)
from tests.unit.use_cases.conftest import (
    FakeFeatureRunRepository,
    FakeFeatureStoreRepository,
    FakeUnitOfWork,
    FakeUniverseRepository,
)


AS_OF = date(2026, 2, 17)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uow_with_completed_run(
    symbols: list[str] | None = None,
    scores: list[float] | None = None,
    ratings: list[int] | None = None,
) -> tuple[FakeUnitOfWork, int]:
    """Create a UoW with a COMPLETED run and feature rows."""
    sym_list = symbols or ["AAPL", "MSFT", "GOOGL"]
    score_list = scores or [75.0, 80.0, 70.0]
    rating_list = ratings or [4, 5, 3]

    universe = FakeUniverseRepository(sym_list)
    feature_runs = FakeFeatureRunRepository()
    feature_store = FakeFeatureStoreRepository()
    uow = FakeUnitOfWork(
        universe=universe,
        feature_runs=feature_runs,
        feature_store=feature_store,
    )

    # Start and complete a run
    from app.domain.feature_store.models import FeatureRowWrite, RunStats

    run = feature_runs.start_run(as_of_date=AS_OF, run_type=RunType.DAILY_SNAPSHOT)
    run_id = run.id
    stats = RunStats(
        total_symbols=len(sym_list),
        processed_symbols=len(sym_list),
        failed_symbols=0,
        duration_seconds=1.0,
    )
    feature_runs.mark_completed(run_id, stats)

    # Save universe and feature rows
    feature_store.save_run_universe_symbols(run_id, sym_list)
    rows = [
        FeatureRowWrite(
            symbol=sym, as_of_date=AS_OF,
            composite_score=sc, overall_rating=rt,
            passes_count=1, details=None,
        )
        for sym, sc, rt in zip(sym_list, score_list, rating_list)
    ]
    feature_store.upsert_snapshot_rows(run_id, rows)

    return uow, run_id


def _make_good_dq_inputs() -> DQInputs:
    """DQInputs that pass all checks with default thresholds."""
    return DQInputs(
        expected_row_count=3,
        actual_row_count=3,
        null_score_count=0,
        total_row_count=3,
        scores=(75.0, 80.0, 70.0),
        ratings=(4, 5, 3),
        universe_symbols=("AAPL", "MSFT", "GOOGL"),
        result_symbols=("AAPL", "MSFT", "GOOGL"),
    )


def _make_bad_row_count_inputs() -> DQInputs:
    """DQInputs where row count fails (CRITICAL)."""
    return DQInputs(
        expected_row_count=10,
        actual_row_count=2,
        null_score_count=0,
        total_row_count=2,
        scores=(75.0, 80.0),
        ratings=(4, 5),
        universe_symbols=tuple(f"SYM{i}" for i in range(10)),
        result_symbols=("SYM0", "SYM1"),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_completed_all_dq_pass_publishes(self):
        uow, run_id = _make_uow_with_completed_run()
        cmd = PublishRunCommand(run_id=run_id, dq_inputs=_make_good_dq_inputs())
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.PUBLISHED.value
        assert result.dq_passed is True
        assert result.run_id == run_id

    def test_all_five_dq_checks_in_report(self):
        uow, run_id = _make_uow_with_completed_run()
        cmd = PublishRunCommand(run_id=run_id, dq_inputs=_make_good_dq_inputs())
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert len(result.dq_report) == 5


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


class TestQuarantine:
    def test_critical_failure_quarantines(self):
        uow, run_id = _make_uow_with_completed_run()
        cmd = PublishRunCommand(
            run_id=run_id, dq_inputs=_make_bad_row_count_inputs()
        )
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.QUARANTINED.value
        assert result.dq_passed is False


# ---------------------------------------------------------------------------
# Force publish
# ---------------------------------------------------------------------------


class TestForcePublish:
    def _quarantine_run(self, uow, run_id):
        """Push a COMPLETED run through quarantine."""
        cmd = PublishRunCommand(
            run_id=run_id, dq_inputs=_make_bad_row_count_inputs()
        )
        result = PublishFeatureRunUseCase().execute(uow, cmd)
        assert result.status == RunStatus.QUARANTINED.value
        return result

    def test_force_publish_quarantined(self):
        uow, run_id = _make_uow_with_completed_run()
        self._quarantine_run(uow, run_id)

        cmd = PublishRunCommand(run_id=run_id, force_publish=True)
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.PUBLISHED.value
        assert result.dq_passed is False

    def test_force_publish_includes_original_warnings(self):
        uow, run_id = _make_uow_with_completed_run()
        self._quarantine_run(uow, run_id)

        cmd = PublishRunCommand(run_id=run_id, force_publish=True)
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert "Force-published" in result.warnings[0]
        # Original quarantine warnings should be appended
        assert len(result.warnings) > 1

    def test_force_publish_non_quarantined_raises(self):
        uow, run_id = _make_uow_with_completed_run()
        cmd = PublishRunCommand(run_id=run_id, force_publish=True)

        with pytest.raises(ValidationError, match="QUARANTINED"):
            PublishFeatureRunUseCase().execute(uow, cmd)


# ---------------------------------------------------------------------------
# Status validation
# ---------------------------------------------------------------------------


class TestStatusValidation:
    def test_running_status_raises(self):
        """Cannot publish a RUNNING run."""
        universe = FakeUniverseRepository(["AAPL"])
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        uow = FakeUnitOfWork(
            universe=universe,
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        run = feature_runs.start_run(as_of_date=AS_OF, run_type=RunType.DAILY_SNAPSHOT)

        cmd = PublishRunCommand(
            run_id=run.id, dq_inputs=_make_good_dq_inputs()
        )
        with pytest.raises(ValidationError, match="COMPLETED"):
            PublishFeatureRunUseCase().execute(uow, cmd)


# ---------------------------------------------------------------------------
# DQ input loading
# ---------------------------------------------------------------------------


class TestDQInputLoading:
    def test_precomputed_inputs_used(self):
        """When dq_inputs is provided, it's used directly (not loaded from DB)."""
        uow, run_id = _make_uow_with_completed_run()
        custom_inputs = _make_good_dq_inputs()
        cmd = PublishRunCommand(run_id=run_id, dq_inputs=custom_inputs)
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.PUBLISHED.value

    def test_inputs_loaded_from_db_when_none(self):
        """When dq_inputs is None, inputs are loaded from the repository."""
        uow, run_id = _make_uow_with_completed_run()
        cmd = PublishRunCommand(run_id=run_id)  # dq_inputs=None
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.PUBLISHED.value
        assert len(result.dq_report) == 5


# ---------------------------------------------------------------------------
# Warning-only failures
# ---------------------------------------------------------------------------


class TestWarningOnlyFailures:
    def test_warning_failures_still_publish(self):
        """Non-CRITICAL DQ failures don't block publishing."""
        # All same rating → rating_distribution WARNING fails, but publishes
        uow, run_id = _make_uow_with_completed_run(
            scores=[50.0, 50.0, 50.0],
            ratings=[3, 3, 3],
        )
        inputs = DQInputs(
            expected_row_count=3,
            actual_row_count=3,
            null_score_count=0,
            total_row_count=3,
            scores=(50.0, 50.0, 50.0),
            ratings=(3, 3, 3),
            universe_symbols=("AAPL", "MSFT", "GOOGL"),
            result_symbols=("AAPL", "MSFT", "GOOGL"),
        )
        cmd = PublishRunCommand(run_id=run_id, dq_inputs=inputs)
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.PUBLISHED.value
        assert result.dq_passed is True
        # But should have warnings about rating distribution
        assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    def test_custom_thresholds_applied(self):
        uow, run_id = _make_uow_with_completed_run()
        # Use strict threshold that will cause row_count to fail
        strict = DQThresholds(row_count_threshold=1.0)
        inputs = DQInputs(
            expected_row_count=10,  # more than actual 3
            actual_row_count=3,
            null_score_count=0,
            total_row_count=3,
            scores=(75.0, 80.0, 70.0),
            ratings=(4, 5, 3),
            universe_symbols=tuple(f"SYM{i}" for i in range(10)),
            result_symbols=("AAPL", "MSFT", "GOOGL"),
        )
        cmd = PublishRunCommand(
            run_id=run_id, dq_inputs=inputs, dq_thresholds=strict,
        )
        result = PublishFeatureRunUseCase().execute(uow, cmd)

        assert result.status == RunStatus.QUARANTINED.value


# ---------------------------------------------------------------------------
# Empty universe guard
# ---------------------------------------------------------------------------


class TestEmptyUniverseGuard:
    def test_zero_expected_row_count_raises(self):
        uow, run_id = _make_uow_with_completed_run()
        inputs = DQInputs(
            expected_row_count=0,
            actual_row_count=0,
            null_score_count=0,
            total_row_count=0,
            scores=(),
            ratings=(),
            universe_symbols=(),
            result_symbols=(),
        )
        cmd = PublishRunCommand(run_id=run_id, dq_inputs=inputs)

        with pytest.raises(ValidationError, match="no universe"):
            PublishFeatureRunUseCase().execute(uow, cmd)

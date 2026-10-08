"""Canonical Market RS Celery task tests."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.market_rs_inputs import MarketRsInputUnavailable
from app.services.market_rs_rollout_contracts import (
    ActivationValidationReport,
    BackfillReport,
    MarketRsActivationArtifactPolicy,
)
from app.services.market_rs_rollout_executor import (
    MarketRsActivationExecutionError,
    MarketRsActivationOutcome,
)


def _patch_task_dependencies(monkeypatch):
    from app.tasks import market_rs_tasks as module

    fake_db = MagicMock()
    fake_calendar = MagicMock()
    fake_calendar.is_trading_day.return_value = True
    fake_service = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_market_calendar_service", lambda: fake_calendar)
    monkeypatch.setattr(module, "get_market_rs_snapshot_service", lambda: fake_service)
    return module, fake_db, fake_calendar, fake_service


def test_calculate_market_rs_snapshot_returns_stable_completed_shape(monkeypatch):
    module, fake_db, fake_calendar, fake_service = _patch_task_dependencies(monkeypatch)
    fake_service.calculate.return_value = SimpleNamespace(
        id=42,
        status="completed",
        market="US",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        eligible_symbol_count=5000,
        diagnostics_json={"history_gaps": {"share": 0.0}},
    )

    result = module.calculate_market_rs_snapshot.run(
        market="us",
        calculation_date="2026-04-10",
    )

    assert result == {
        "status": "completed",
        "market": "US",
        "as_of_date": "2026-04-10",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": 42,
        "eligible_symbol_count": 5000,
        "history_gaps": {"share": 0.0},
    }
    fake_calendar.is_trading_day.assert_called_once_with("US", date(2026, 4, 10))
    fake_service.calculate.assert_called_once_with(
        fake_db,
        market="US",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        rebuild_incompatible=True,
    )
    fake_db.close.assert_called_once_with()


def test_calculate_market_rs_snapshot_resolves_bootstrap_date_when_omitted(monkeypatch):
    module, fake_db, fake_calendar, fake_service = _patch_task_dependencies(monkeypatch)
    fake_calendar.last_completed_trading_day.return_value = date(2026, 4, 10)
    fake_service.calculate.return_value = SimpleNamespace(
        id=43,
        status="completed",
        market="HK",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        eligible_symbol_count=800,
        diagnostics_json={"history_gaps": {"share": 0.0}},
    )

    result = module.calculate_market_rs_snapshot.run(
        market="HK",
        activity_lifecycle="bootstrap",
    )

    assert result["status"] == "completed"
    assert result["as_of_date"] == "2026-04-10"
    fake_calendar.last_completed_trading_day.assert_called_once_with("HK")
    fake_service.calculate.assert_called_once_with(
        fake_db,
        market="HK",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
        rebuild_incompatible=True,
    )


def test_calculate_market_rs_snapshot_returns_input_diagnostics(monkeypatch):
    module, fake_db, _fake_calendar, fake_service = _patch_task_dependencies(
        monkeypatch
    )
    fake_service.calculate.side_effect = MarketRsInputUnavailable(
        "benchmark missing",
        reason_code="benchmark_anchor_missing",
        diagnostics={"missing_anchor_dates": {"SPY": ["2025-04-10"]}},
        benchmark_symbol="SPY",
        universe_hash="abc123",
        expected_symbol_count=5000,
    )

    result = module.calculate_market_rs_snapshot.run(
        market="US",
        calculation_date="2026-04-10",
    )

    assert result == {
        "status": "failed",
        "market": "US",
        "as_of_date": "2026-04-10",
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "reason_code": "benchmark_anchor_missing",
        "diagnostics": {
            "missing_anchor_dates": {"SPY": ["2025-04-10"]},
            "benchmark_symbol": "SPY",
            "universe_hash": "abc123",
            "expected_symbol_count": 5000,
        },
    }
    fake_db.close.assert_called_once_with()


def test_calculate_market_rs_snapshot_rejects_non_trading_date(monkeypatch):
    module, fake_db, fake_calendar, fake_service = _patch_task_dependencies(monkeypatch)
    fake_calendar.is_trading_day.return_value = False

    result = module.calculate_market_rs_snapshot.run(
        market="US",
        calculation_date="2026-04-11",
    )

    assert result["status"] == "failed"
    assert result["reason_code"] == "not_trading_day"
    fake_service.calculate.assert_not_called()
    fake_db.close.assert_not_called()


def test_calculate_market_rs_snapshot_rejects_shared_market():
    from app.tasks import market_rs_tasks as module

    result = module.calculate_market_rs_snapshot.run(
        market="SHARED",
        calculation_date="2026-04-10",
    )

    assert result["status"] == "failed"
    assert result["reason_code"] == "invalid_market"


def _patch_bootstrap_rollout_dependencies(monkeypatch):
    from app.tasks import market_rs_tasks as module

    db = MagicMock()
    calendar = MagicMock()
    calendar.last_completed_trading_day.return_value = date(2026, 7, 29)
    executor = MagicMock()
    started = MagicMock()
    completed = MagicMock()
    failed = MagicMock()
    rollout = MagicMock()
    through_date = calendar.last_completed_trading_day.return_value
    rollout.resolve_bootstrap_through_date.return_value = SimpleNamespace(
        market="US",
        requested_through_date=through_date,
        selected_through_date=through_date,
        benchmark_through_date=through_date,
        benchmark_lag_days=0,
        reason_code="requested_date_ready",
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: db)
    monkeypatch.setattr(module, "get_market_calendar_service", lambda: calendar)
    monkeypatch.setattr(module, "get_market_rs_rollout_service", lambda: rollout)
    monkeypatch.setattr(module, "get_market_rs_activation_executor", lambda: executor)
    monkeypatch.setattr(module, "mark_market_activity_started", started)
    monkeypatch.setattr(module, "mark_market_activity_completed", completed)
    monkeypatch.setattr(module, "mark_market_activity_failed", failed)
    return module, db, calendar, executor, started, completed, failed


def test_bootstrap_balanced_market_rs_requires_successful_activation(monkeypatch):
    (
        module,
        db,
        calendar,
        executor,
        started,
        completed,
        failed,
    ) = _patch_bootstrap_rollout_dependencies(monkeypatch)
    through_date = date(2026, 7, 29)
    executor.execute.return_value = MarketRsActivationOutcome(
        backfill=BackfillReport(
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            requested_start_date=through_date,
            through_date=through_date,
            first_valid_date=through_date,
            candidate_count=1,
            completed_count=1,
            failed_count=0,
            latest_run_id=99,
            group_row_count=1,
            results=(),
        ),
        market="US",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        feature_run_id=99,
        validation=ActivationValidationReport(
            market="US",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            through_date=through_date,
            first_valid_date=through_date,
            candidate_count=1,
            latest_market_rs_run_id=99,
            latest_universe_hash="universe",
            feature_run_id=99,
            feature_universe_hash="universe",
            static_bundle_sha256="bundle",
            errors=(),
            artifact_policy=MarketRsActivationArtifactPolicy.LIVE_RUNTIME,
        ),
        static_staging_dir=None,
    )

    result = module.bootstrap_balanced_market_rs.run(
        market="us",
        activity_lifecycle="bootstrap",
    )

    assert result["status"] == "activated"
    assert result["formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert executor.execute.call_args.kwargs["request"].market == "US"
    assert executor.execute.call_args.kwargs["request"].artifact_policy is (
        MarketRsActivationArtifactPolicy.LIVE_RUNTIME
    )
    assert executor.execute.call_args.kwargs["request"].static_staging_dir is None
    calendar.last_completed_trading_day.assert_called_once_with("US")
    started.assert_called_once()
    completed.assert_called_once()
    failed.assert_not_called()
    db.close.assert_called_once_with()


def test_bootstrap_balanced_market_rs_uses_rollout_resolved_through_date(
    monkeypatch,
):
    (
        module,
        db,
        calendar,
        executor,
        _started,
        _completed,
        _failed,
    ) = _patch_bootstrap_rollout_dependencies(monkeypatch)
    requested_through_date = date(2026, 7, 31)
    selected_through_date = date(2026, 7, 30)
    calendar.last_completed_trading_day.return_value = requested_through_date
    rollout = MagicMock()
    rollout.resolve_bootstrap_through_date.return_value = SimpleNamespace(
        selected_through_date=selected_through_date,
        reason_code="benchmark_ready_lag",
    )
    monkeypatch.setattr(
        module,
        "get_market_rs_rollout_service",
        lambda: rollout,
    )
    executor.execute.return_value = MarketRsActivationOutcome(
        backfill=BackfillReport(
            market="HK",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            requested_start_date=selected_through_date,
            through_date=selected_through_date,
            first_valid_date=selected_through_date,
            candidate_count=1,
            completed_count=1,
            failed_count=0,
            latest_run_id=99,
            group_row_count=1,
            results=(),
        ),
        market="HK",
        formula_version=BALANCED_RS_FORMULA_VERSION,
        feature_run_id=99,
        validation=ActivationValidationReport(
            market="HK",
            formula_version=BALANCED_RS_FORMULA_VERSION,
            through_date=selected_through_date,
            first_valid_date=selected_through_date,
            candidate_count=1,
            latest_market_rs_run_id=99,
            latest_universe_hash="universe",
            feature_run_id=99,
            feature_universe_hash="universe",
            static_bundle_sha256="bundle",
            errors=(),
            artifact_policy=MarketRsActivationArtifactPolicy.LIVE_RUNTIME,
        ),
        static_staging_dir=None,
    )

    module.bootstrap_balanced_market_rs.run(
        market="HK",
        activity_lifecycle="bootstrap",
    )

    rollout.resolve_bootstrap_through_date.assert_called_once_with(
        db,
        market="HK",
        requested_through_date=requested_through_date,
    )
    assert (
        executor.execute.call_args.kwargs["request"].through_date
        == selected_through_date
    )
    assert executor.execute.call_args.kwargs["request"].artifact_policy is (
        MarketRsActivationArtifactPolicy.LIVE_RUNTIME
    )


@pytest.mark.parametrize(
    ("reason_code", "benchmark_through_date", "benchmark_lag_days"),
    [
        ("benchmark_date_unavailable", None, None),
        ("benchmark_lag_exceeds_policy", date(2026, 7, 20), 11),
        ("unexpected_resolution_state", date(2026, 7, 31), 0),
    ],
)
def test_bootstrap_balanced_market_rs_rejects_unready_through_date_resolution(
    monkeypatch,
    reason_code,
    benchmark_through_date,
    benchmark_lag_days,
):
    (
        module,
        db,
        calendar,
        executor,
        started,
        completed,
        failed,
    ) = _patch_bootstrap_rollout_dependencies(monkeypatch)
    requested_through_date = date(2026, 7, 31)
    calendar.last_completed_trading_day.return_value = requested_through_date
    rollout = MagicMock()
    rollout.resolve_bootstrap_through_date.return_value = SimpleNamespace(
        market="HK",
        requested_through_date=requested_through_date,
        selected_through_date=requested_through_date,
        benchmark_through_date=benchmark_through_date,
        benchmark_lag_days=benchmark_lag_days,
        reason_code=reason_code,
    )
    monkeypatch.setattr(module, "get_market_rs_rollout_service", lambda: rollout)

    with pytest.raises(module.MarketRsBootstrapThroughDateUnavailable) as raised:
        module.bootstrap_balanced_market_rs.run(
            market="HK",
            activity_lifecycle="bootstrap",
        )

    assert raised.value.diagnostics["reason_code"] == reason_code
    assert raised.value.diagnostics["requested_through_date"] == "2026-07-31"
    executor.execute.assert_not_called()
    started.assert_not_called()
    completed.assert_not_called()
    failed.assert_called_once()
    assert reason_code in failed.call_args.kwargs["message"]
    db.rollback.assert_called_once_with()
    db.close.assert_called_once_with()


@pytest.mark.parametrize(
    "failure",
    [
        MarketRsActivationExecutionError("static validation failed"),
        RuntimeError("adapter failed"),
        OSError("staging filesystem unavailable"),
    ],
)
def test_bootstrap_balanced_market_rs_stops_chain_on_rollout_failure(
    monkeypatch,
    failure,
):
    module, db, _calendar, executor, _started, completed, failed = (
        _patch_bootstrap_rollout_dependencies(monkeypatch)
    )
    executor.execute.side_effect = failure

    with pytest.raises(type(failure), match=str(failure)):
        module.bootstrap_balanced_market_rs.run(market="US")

    db.rollback.assert_called_once_with()
    completed.assert_not_called()
    failed.assert_called_once()
    db.close.assert_called_once_with()


def test_bootstrap_balanced_market_rs_retries_transient_connection_failure(
    monkeypatch,
):
    module, db, _calendar, executor, _started, _completed, failed = (
        _patch_bootstrap_rollout_dependencies(monkeypatch)
    )
    error = ConnectionError("database unavailable")
    executor.execute.side_effect = error
    retry = MagicMock(side_effect=RuntimeError("retry requested"))
    monkeypatch.setattr(module, "_retry_connection_failure", retry)

    with pytest.raises(RuntimeError, match="retry requested"):
        module.bootstrap_balanced_market_rs.run(market="US")

    db.rollback.assert_called_once_with()
    retry.assert_called_once()
    assert retry.call_args.args[1] is error
    failed.assert_not_called()
    db.close.assert_called_once_with()


def test_bootstrap_balanced_market_rs_marks_integrity_error_failed(monkeypatch):
    module, db, _calendar, executor, _started, completed, failed = (
        _patch_bootstrap_rollout_dependencies(monkeypatch)
    )
    error = IntegrityError(
        "insert market activity",
        {},
        Exception("not null constraint failed"),
    )
    executor.execute.side_effect = error

    with pytest.raises(IntegrityError):
        module.bootstrap_balanced_market_rs.run(market="US")

    db.rollback.assert_called_once_with()
    completed.assert_not_called()
    failed.assert_called_once()
    assert "not null constraint failed" in failed.call_args.kwargs["message"]
    db.close.assert_called_once_with()


def test_bootstrap_balanced_market_rs_retries_transient_database_error(monkeypatch):
    module, db, _calendar, executor, _started, _completed, failed = (
        _patch_bootstrap_rollout_dependencies(monkeypatch)
    )
    error = OperationalError(
        "select 1",
        {},
        Exception("database system is not yet accepting connections"),
    )
    executor.execute.side_effect = error
    retry = MagicMock(side_effect=RuntimeError("retry requested"))
    monkeypatch.setattr(module, "_retry_connection_failure", retry)

    with pytest.raises(RuntimeError, match="retry requested"):
        module.bootstrap_balanced_market_rs.run(market="US")

    db.rollback.assert_called_once_with()
    retry.assert_called_once_with(module.bootstrap_balanced_market_rs, error)
    failed.assert_not_called()
    db.close.assert_called_once_with()

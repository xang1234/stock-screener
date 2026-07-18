"""Canonical Market RS Celery task tests."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.market_rs_inputs import MarketRsInputUnavailable


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
    }
    fake_calendar.is_trading_day.assert_called_once_with("US", date(2026, 4, 10))
    fake_service.calculate.assert_called_once_with(
        fake_db,
        market="US",
        as_of_date=date(2026, 4, 10),
        formula_version=BALANCED_RS_FORMULA_VERSION,
    )
    fake_db.close.assert_called_once_with()


def test_calculate_market_rs_snapshot_returns_input_diagnostics(monkeypatch):
    module, fake_db, _fake_calendar, fake_service = _patch_task_dependencies(monkeypatch)
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

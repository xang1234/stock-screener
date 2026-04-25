from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded


def _patch_serialized_lock(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (True, False)
    fake_coordination.release_market_workload.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
    )


def _patch_calendar_service(
    monkeypatch,
    now: datetime,
    *,
    is_trading_day: bool = True,
):
    """Stub the MarketCalendarService lookup that breadth tasks use.

    The module-level ``get_market_calendar_service`` in ``breadth_tasks`` is
    patched so the task doesn't pull real exchange calendars during tests.
    """
    fake = MagicMock()
    fake.is_trading_day.return_value = is_trading_day
    fake.market_now.return_value = now
    fake.last_completed_trading_day.return_value = now.date()
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.get_market_calendar_service",
        lambda: fake,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: fake,
    )
    return fake


def test_daily_breadth_refuses_to_publish_when_same_day_warmup_incomplete(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "partial",
        "count": 9500,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
        "error": None,
    }

    fake_calculator = MagicMock()
    fake_calculator.price_cache = fake_price_cache
    fake_calculator.calculate_daily_breadth.return_value = {
        "stocks_up_4pct": 1,
        "stocks_down_4pct": 0,
        "ratio_5day": 1.0,
        "ratio_10day": 1.0,
        "stocks_up_25pct_quarter": 1,
        "stocks_down_25pct_quarter": 0,
        "stocks_up_25pct_month": 1,
        "stocks_down_25pct_month": 0,
        "stocks_up_50pct_month": 0,
        "stocks_down_50pct_month": 0,
        "stocks_up_13pct_34days": 1,
        "stocks_down_13pct_34days": 0,
        "total_stocks_scanned": 9500,
        "skipped_stocks": 500,
        "cache_miss_stocks": 25,
        "error_stocks": 0,
    }
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:fake_calculator)

    result = module.calculate_daily_breadth.run()

    assert "error" in result
    assert "warmup not complete" in result["error"].lower()
    fake_db.add.assert_not_called()
    fake_db.commit.assert_not_called()


def test_generate_trading_dates_skips_holidays_and_weekends(monkeypatch):
    import app.tasks.breadth_tasks as module

    closed_days = {
        date(2026, 1, 1),
        date(2026, 1, 3),
        date(2026, 1, 4),
    }
    fake = MagicMock()
    fake.is_trading_day.side_effect = lambda _market, d: d not in closed_days
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.get_market_calendar_service",
        lambda: fake,
    )

    trading_dates, skipped = module._generate_trading_dates(date(2026, 1, 1), date(2026, 1, 5))

    assert trading_dates == [date(2026, 1, 2), date(2026, 1, 5)]
    assert skipped == 3


def test_manual_breadth_can_force_cache_only_for_static_exports(monkeypatch):
    import app.tasks.breadth_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 4, 3, 0, 30, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda: None)

    fake_price_cache = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.price_cache = fake_price_cache
    fake_calculator.calculate_daily_breadth.return_value = {
        "stocks_up_4pct": 1,
        "stocks_down_4pct": 0,
        "ratio_5day": 1.0,
        "ratio_10day": 1.0,
        "stocks_up_25pct_quarter": 1,
        "stocks_down_25pct_quarter": 0,
        "stocks_up_25pct_month": 1,
        "stocks_down_25pct_month": 0,
        "stocks_up_50pct_month": 0,
        "stocks_down_50pct_month": 0,
        "stocks_up_13pct_34days": 1,
        "stocks_down_13pct_34days": 0,
        "total_stocks_scanned": 10000,
        "skipped_stocks": 0,
        "cache_miss_stocks": 0,
        "error_stocks": 0,
    }
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:fake_calculator)

    result = module.calculate_daily_breadth.run("2026-04-02", force_cache_only=True)

    assert result["date"] == "2026-04-02"
    assert result["indicators"]["stocks_up_4pct"] == 1
    assert result["cache_only"] is True
    fake_calculator.calculate_daily_breadth.assert_called_once_with(
        calculation_date=date(2026, 4, 2),
        cache_only=True,
    )


def test_daily_breadth_persists_non_us_record_in_market_partition(monkeypatch):
    import app.tasks.breadth_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 4, 3, 0, 30, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda: None)

    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = {
        "stocks_up_4pct": 1,
        "stocks_down_4pct": 0,
        "ratio_5day": 1.0,
        "ratio_10day": 1.0,
        "stocks_up_25pct_quarter": 1,
        "stocks_down_25pct_quarter": 0,
        "stocks_up_25pct_month": 1,
        "stocks_down_25pct_month": 0,
        "stocks_up_50pct_month": 0,
        "stocks_down_50pct_month": 0,
        "stocks_up_13pct_34days": 1,
        "stocks_down_13pct_34days": 0,
        "total_stocks_scanned": 100,
        "skipped_stocks": 0,
        "cache_miss_stocks": 0,
        "error_stocks": 0,
    }
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw: fake_calculator)

    result = module.calculate_daily_breadth.run(
        "2026-04-02",
        force_cache_only=True,
        market="HK",
    )

    assert result["date"] == "2026-04-02"
    filter_args = fake_db.query.return_value.filter.call_args.args
    assert len(filter_args) == 2
    assert filter_args[1].right.value == "HK"
    saved_record = fake_db.add.call_args.args[0]
    assert saved_record.market == "HK"
    fake_db.commit.assert_called_once()


def test_backfill_breadth_uses_service_range_with_trading_dates(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)

    monkeypatch.setattr(
        module,
        "_generate_trading_dates",
        lambda start, end, **kw: ([date(2026, 1, 2), date(2026, 1, 5)], 3),
    )

    fake_service = MagicMock()
    fake_service.backfill_range.return_value = {
        "total_dates": 2,
        "processed": 2,
        "errors": 0,
        "error_dates": [],
    }
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:fake_service)

    result = module.backfill_breadth_data.run("2026-01-01", "2026-01-05")

    fake_service.backfill_range.assert_called_once_with(
        date(2026, 1, 1),
        date(2026, 1, 5),
        trading_dates=[date(2026, 1, 2), date(2026, 1, 5)],
    )
    assert result["successful"] == 2
    assert result["failed"] == 0
    assert result["skipped_weekends"] == 3
    assert result["skipped_non_trading_days"] == 3


def test_breadth_gapfill_retries_transient_outer_failures(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", False)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(module, "calculate_daily_breadth", MagicMock(side_effect=ConnectionError("network down")))
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:MagicMock())

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.calculate_daily_breadth_with_gapfill, "retry", fake_retry)
    module.calculate_daily_breadth_with_gapfill.request.id = "task-123"
    module.calculate_daily_breadth_with_gapfill.request.retries = 0

    with pytest.raises(Retry):
        module.calculate_daily_breadth_with_gapfill.run()

    fake_db.rollback.assert_called_once()
    assert retry_calls[0]["max_retries"] == 2
    assert retry_calls[0]["countdown"] == 60
    assert module.calculate_daily_breadth_with_gapfill.soft_time_limit == 3600


def test_breadth_gapfill_retry_survives_activity_publish_failure(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", False)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(module, "calculate_daily_breadth", MagicMock(side_effect=ConnectionError("network down")))
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:MagicMock())
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        MagicMock(side_effect=RuntimeError("activity store unavailable")),
    )

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.calculate_daily_breadth_with_gapfill, "retry", fake_retry)
    module.calculate_daily_breadth_with_gapfill.request.id = "task-123"
    module.calculate_daily_breadth_with_gapfill.request.retries = 0

    with pytest.raises(Retry):
        module.calculate_daily_breadth_with_gapfill.run()

    assert retry_calls[0]["countdown"] == 60


def test_breadth_gapfill_reraises_soft_time_limit(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", False)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(module, "calculate_daily_breadth", MagicMock(side_effect=SoftTimeLimitExceeded()))
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:MagicMock())

    with pytest.raises(SoftTimeLimitExceeded):
        module.calculate_daily_breadth_with_gapfill.run()

    fake_db.rollback.assert_called_once()


def test_breadth_gapfill_publishes_market_activity(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.find_missing_dates.return_value = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", False)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:fake_calculator)
    monkeypatch.setattr(module, "calculate_daily_breadth", lambda market=None: {"date": "2026-03-20"})
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    started = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    result = module.calculate_daily_breadth_with_gapfill.run(market="US")

    assert result["today"]["date"] == "2026-03-20"
    assert started[0]["stage_key"] == "breadth"
    assert started[0]["lifecycle"] == "daily_refresh"
    assert completed[0]["stage_key"] == "breadth"


def test_breadth_gapfill_runs_for_non_us_market(monkeypatch):
    """Breadth is now computed per market; HK/JP/TW/IN proceed instead of
    returning breadth_calculation_is_us_only. The market constructor arg
    routes the calculator to the right universe.
    """
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", False)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )

    captured_init_kwargs: dict = {}

    class _CapturingCalculator:
        def __init__(self, db, price_cache, **kwargs):
            captured_init_kwargs.update(kwargs)
            self.find_missing_dates = MagicMock(return_value=[])
            self.fill_gaps = MagicMock(return_value={"total_dates": 0, "processed": 0, "errors": 0})

    monkeypatch.setattr(module, "BreadthCalculatorService", _CapturingCalculator)
    monkeypatch.setattr(
        module,
        "_calculate_daily_breadth_in_process",
        lambda market=None: {"date": "2026-03-20", "market": market},
    )

    result = module.calculate_daily_breadth_with_gapfill.run(market="HK")

    # No status='skipped' any more — the task runs for HK
    assert result.get("status") != "skipped"
    # The calculator was constructed with market='HK' so it queries the right universe
    assert captured_init_kwargs.get("market") == "HK"

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded

from app.services.breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageReport,
    BreadthOutcomeReport,
    BreadthPriceCoverage,
)


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


def _breadth_result(
    *,
    scanned: int,
    skipped: int,
    misses: int,
    errors: int = 0,
) -> BreadthCalculationResult:
    candidates = scanned + skipped
    indicators = {
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
    }
    insufficient = max(skipped - misses - errors, 0)
    return BreadthCalculationResult(
        indicators=indicators,
        coverage=BreadthCoverageReport.from_parts(
            BreadthPriceCoverage(
            candidate_stocks=candidates,
            symbols_with_cached_history=candidates - misses,
            cache_miss_stocks=misses,
            cache_miss_symbols_sample=(
                ("MISS",) if misses else ()
            ),
            cache_coverage_ratio=(
                (candidates - misses) / candidates
                if candidates
                else 0.0
            ),
            ),
            BreadthOutcomeReport(
                scanned=scanned,
                cache_misses=misses,
                insufficient=insufficient,
                errors=errors,
            ),
        ),
    )


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
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=9500,
        skipped=500,
        misses=25,
    )
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
    publish_breadth = MagicMock()
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", publish_breadth)

    fake_price_cache = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.price_cache = fake_price_cache
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=10000,
        skipped=0,
        misses=0,
    )
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw:fake_calculator)

    result = module.calculate_daily_breadth.run("2026-04-02", force_cache_only=True)

    assert result["date"] == "2026-04-02"
    assert result["indicators"]["stocks_up_4pct"] == 1
    assert result["cache_only"] is True
    call_kwargs = fake_calculator.calculate_daily_breadth.call_args.kwargs
    assert call_kwargs["calculation_date"] == date(2026, 4, 2)
    assert call_kwargs["policy"].mode.value == "strict_cache_only"
    publish_breadth.assert_called_once_with("US")


def test_guarded_historical_breadth_tolerates_cache_misses(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=60,
        skipped=40,
        misses=35,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda _market: None)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    result = module.calculate_daily_breadth.run(
        "2026-03-19",
        refresh_guarded_cache_only=True,
    )

    assert "error" not in result
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert result["cache_diagnostics"]["cache_miss_stocks"] == 35
    call_kwargs = fake_calculator.calculate_daily_breadth.call_args.kwargs
    assert call_kwargs["calculation_date"] == date(2026, 3, 19)
    assert call_kwargs["policy"].mode.value == "refresh_guarded"
    fake_db.commit.assert_called_once()


def test_guarded_historical_breadth_fails_when_no_stock_is_usable(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=0,
        skipped=100,
        misses=100,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    result = module.calculate_daily_breadth.run(
        "2026-03-19",
        refresh_guarded_cache_only=True,
    )

    assert result["cache_policy"] == "refresh_guarded"
    assert "processed no usable stocks" in result["error"].lower()
    fake_db.commit.assert_not_called()


def test_force_cache_only_wins_over_guarded_tolerance(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=60,
        skipped=40,
        misses=35,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    result = module.calculate_daily_breadth.run(
        "2026-03-19",
        force_cache_only=True,
        refresh_guarded_cache_only=True,
    )

    assert "exceeds miss tolerance" in result["error"].lower()
    assert result.get("cache_policy") != "refresh_guarded"
    fake_db.commit.assert_not_called()


def test_manual_historical_breadth_keeps_fetch_capable_behavior(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=100,
        skipped=0,
        misses=0,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda _market: None)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 4, 3, 0, 30, 0))

    result = module.calculate_daily_breadth.run("2026-04-02")

    assert result["cache_only"] is False
    assert result.get("cache_policy") is None
    call_kwargs = fake_calculator.calculate_daily_breadth.call_args.kwargs
    assert call_kwargs["calculation_date"] == date(2026, 4, 2)
    assert call_kwargs["policy"].mode.value == "auto"
    assert call_kwargs["policy"].cache_only is False
    fake_db.commit.assert_called_once()


def test_daily_breadth_persists_non_us_record_in_market_partition(monkeypatch):
    import app.tasks.breadth_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 4, 3, 0, 30, 0))
    publish_breadth = MagicMock()
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", publish_breadth)

    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_result(
        scanned=100,
        skipped=0,
        misses=0,
    )
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
    publish_breadth.assert_called_once_with("HK")


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
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", True)
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
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", True)
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
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", True)
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
    daily_calls = []
    monkeypatch.setattr(
        module,
        "calculate_daily_breadth",
        lambda **kwargs: daily_calls.append(kwargs) or {"date": "2026-03-20"},
    )
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    started = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    result = module.calculate_daily_breadth_with_gapfill.run(market="US")

    assert result["today"]["date"] == "2026-03-20"
    assert daily_calls == [{
        "market": "US",
        "calculation_date": "2026-03-20",
        "execution_policy": "auto",
    }]
    assert started[0]["stage_key"] == "breadth"
    assert started[0]["lifecycle"] == "daily_refresh"
    assert completed[0]["stage_key"] == "breadth"


def test_breadth_gapfill_uses_requested_calculation_date_for_daily_calc(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.find_missing_dates.return_value = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", True)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw: fake_calculator)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 17, 12, 0, 0))

    captured = []

    def fake_inner(
        calculation_date=None,
        market=None,
        execution_policy=None,
    ):
        captured.append((calculation_date, market, execution_policy))
        return {"date": calculation_date, "market": market}

    monkeypatch.setattr(module, "_calculate_daily_breadth_in_process", fake_inner)

    result = module.calculate_daily_breadth_with_gapfill.run(
        market="HK",
        calculation_date="2026-03-16",
    )

    assert result["today"]["date"] == "2026-03-16"
    assert captured == [("2026-03-16", "HK", "auto")]
    fake_calculator.find_missing_dates.assert_called_once_with(
        lookback_days=30,
        end_date=date(2026, 3, 16),
    )


def test_guarded_breadth_wrapper_propagates_cache_only_to_gapfill_and_target(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.find_missing_dates.return_value = [date(2026, 3, 18)]
    fake_calculator.fill_gaps.return_value = {
        "total_dates": 1,
        "processed": 0,
        "errors": 1,
        "error_dates": ["2026-03-18"],
        "cache_miss_stocks": 4,
        "cache_miss_symbols_sample": ["MISS"],
    }
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", True)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    target_call = MagicMock(return_value={
        "date": "2026-03-19",
        "cache_only": True,
        "cache_policy": "refresh_guarded",
    })
    monkeypatch.setattr(module, "_calculate_daily_breadth_in_process", target_call)

    result = module.calculate_daily_breadth_with_gapfill.run(
        market="US",
        calculation_date="2026-03-19",
        execution_policy="refresh_guarded",
    )

    fill_kwargs = fake_calculator.fill_gaps.call_args.kwargs
    assert fill_kwargs["policy"].mode.value == "refresh_guarded"
    assert fill_kwargs["policy"] is fill_kwargs["policy"].for_gap_fill()
    target_call.assert_called_once_with(
        market="US",
        calculation_date="2026-03-19",
        execution_policy="refresh_guarded",
    )
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert "error" not in result
    assert result["gap_fill"]["errors"] == 1
    assert result["gap_fill"]["cache_miss_stocks"] == 4


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
        lambda market=None, execution_policy=None: {
            "date": "2026-03-20",
            "market": market,
        },
    )

    result = module.calculate_daily_breadth_with_gapfill.run(market="HK")

    # No status='skipped' any more — the task runs for HK
    assert result.get("status") != "skipped"
    # The calculator was constructed with market='HK' so it queries the right universe
    assert captured_init_kwargs.get("market") == "HK"


def test_breadth_orchestrator_bypasses_lease_for_inner_daily_call(monkeypatch):
    """Nested daily breadth runs must not re-acquire the lease held by gap-fill."""
    import app.tasks.breadth_tasks as module
    import app.tasks.workload_coordination as wc

    seen_disabled_state: list[bool] = []

    real_task = MagicMock()
    real_task.__module__ = "app.tasks.breadth_tasks"
    real_task.request = MagicMock()

    def fake_run(market=None):
        seen_disabled_state.append(wc._SERIALIZED_MARKET_WORKLOAD_DISABLED.get())
        return {"date": "2026-03-20", "market": market}

    real_task.run = fake_run
    monkeypatch.setattr(module, "calculate_daily_breadth", real_task)

    assert wc._SERIALIZED_MARKET_WORKLOAD_DISABLED.get() is False
    result = module._calculate_daily_breadth_in_process(market="US")

    assert result == {"date": "2026-03-20", "market": "US"}
    assert seen_disabled_state == [True], (
        "The lease bypass ContextVar must be set to True while the inner task runs"
    )
    assert wc._SERIALIZED_MARKET_WORKLOAD_DISABLED.get() is False, (
        "The bypass must be reset on context exit"
    )

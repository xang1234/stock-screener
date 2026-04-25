from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded

from app.services.ibd_group_rank_service import IncompleteGroupRankingCacheError
from app.services.ibd_group_rank_service import MissingIBDIndustryMappingsError


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
    """Stub MarketCalendarService for group-rank tasks.

    Mirrors the old ``is_trading_day`` / ``get_eastern_now`` patches now that
    the task routes trading-day and "today" resolution through
    ``MarketCalendarService`` for per-market support.
    """
    fake = MagicMock()
    fake.is_trading_day.return_value = is_trading_day
    fake.market_now.return_value = now
    fake.last_completed_trading_day.return_value = now.date()
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.get_market_calendar_service",
        lambda: fake,
    )
    return fake


def test_daily_group_rankings_refuse_to_publish_when_warmup_incomplete(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "partial",
        "count": 9500,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings.run()

    assert "error" in result
    assert "warmup not complete" in result["error"].lower()
    fake_service.calculate_group_rankings.assert_not_called()


def test_daily_group_rankings_allow_in_process_same_day_bypass(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = None
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1, "num_stocks": 12}
    ]

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    with module.allow_same_day_group_rank_warmup_bypass():
        result = module.calculate_daily_group_rankings.run()

    assert result["groups_ranked"] == 1
    assert result["cache_only"] is True
    fake_service.calculate_group_rankings.assert_called_once_with(
        fake_db,
        datetime(2026, 3, 20, 17, 40, 0).date(),
        market="US",
        cache_only=True,
        require_complete_cache=True,
    )


def test_daily_group_rankings_refuse_to_publish_when_cache_only_inputs_missing(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.side_effect = IncompleteGroupRankingCacheError(
        {
            "target_symbols": 100,
            "symbols_with_prices": 99,
            "cache_miss_symbols": 1,
            "spy_cached": True,
        }
    )

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings.run()

    assert result["cache_only"] is True
    assert result["prefetch_stats"]["cache_miss_symbols"] == 1
    assert "missing cached price data" in result["error"].lower()


def test_manual_group_rankings_keep_fetch_capable_behavior(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_price_cache = MagicMock()
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1, "num_stocks": 12}
    ]

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings.run("2026-03-19")

    assert result["groups_ranked"] == 1
    assert result["cache_only"] is False
    fake_service.calculate_group_rankings.assert_called_once_with(
        fake_db,
        datetime(2026, 3, 19).date(),
        market="US",
        cache_only=False,
        require_complete_cache=False,
    )


def test_manual_group_rankings_can_force_cache_only_for_static_exports(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 4, 3, 0, 30, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    fake_service.price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1, "num_stocks": 12}
    ]

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings.run(
        "2026-04-02",
        force_cache_only=True,
    )

    assert result["groups_ranked"] == 1
    assert result["cache_only"] is True
    fake_service.calculate_group_rankings.assert_called_once_with(
        fake_db,
        datetime(2026, 4, 2).date(),
        market="US",
        cache_only=True,
        require_complete_cache=True,
    )


def test_daily_group_rankings_retries_transient_outer_failures(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.side_effect = ConnectionError("network down")

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.calculate_daily_group_rankings, "retry", fake_retry)
    module.calculate_daily_group_rankings.request.id = "task-123"
    module.calculate_daily_group_rankings.request.retries = 0

    with pytest.raises(Retry):
        module.calculate_daily_group_rankings.run()

    fake_db.rollback.assert_called_once()
    assert retry_calls[0]["max_retries"] == 2
    assert retry_calls[0]["countdown"] == 60


def test_daily_group_rankings_retry_survives_activity_publish_failure(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.side_effect = ConnectionError("network down")

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        MagicMock(side_effect=RuntimeError("activity store unavailable")),
    )

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.calculate_daily_group_rankings, "retry", fake_retry)
    module.calculate_daily_group_rankings.request.id = "task-123"
    module.calculate_daily_group_rankings.request.retries = 0

    with pytest.raises(Retry):
        module.calculate_daily_group_rankings.run()

    assert retry_calls[0]["countdown"] == 60


def test_daily_group_rankings_reraises_soft_time_limit(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.side_effect = SoftTimeLimitExceeded()

    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    with pytest.raises(SoftTimeLimitExceeded):
        module.calculate_daily_group_rankings.run()

    fake_db.rollback.assert_called_once()


def test_daily_group_rankings_publishes_market_activity(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    fake_service.price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1, "num_stocks": 12}
    ]
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    started = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks._repair_current_us_group_metadata",
        lambda **_kwargs: {"status": "ok"},
    )

    result = module.calculate_daily_group_rankings.run(market="US")

    assert result["groups_ranked"] == 1
    assert started[0]["stage_key"] == "groups"
    assert started[0]["lifecycle"] == "daily_refresh"
    assert completed[0]["stage_key"] == "groups"
    assert result["metadata_repair"] == {"status": "ok"}


def test_daily_group_rankings_run_for_non_us_market(monkeypatch):
    """Group rankings now work per market via MarketTaxonomyService; HK/JP/TW
    no longer return group_rankings_are_us_only. The market kwarg flows into
    IBDGroupRankService.calculate_group_rankings.
    """
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )

    fake_price_cache = MagicMock()
    fake_price_cache.get_warmup_metadata.return_value = None
    fake_service = MagicMock()
    fake_service.price_cache = fake_price_cache
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Banks", "avg_rs_rating": 88.0, "rank": 1, "num_stocks": 5}
    ]
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    # Use a prior date so the task doesn't route through the same-day
    # warmup-required path (we're just asserting HK is not skipped here).
    result = module.calculate_daily_group_rankings.run("2026-03-19", market="HK")

    assert result.get("status") != "skipped"
    # Service was called with market='HK' so it queries the Hong Kong taxonomy
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs.get("market") == "HK"


def test_daily_group_rankings_fail_explicitly_when_ibd_mappings_missing(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    fake_service.price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service.calculate_group_rankings.side_effect = MissingIBDIndustryMappingsError()
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings.run(market="US")

    assert "error" in result
    assert "ibd industry mappings are not loaded" in result["error"].lower()


def test_historical_group_rankings_do_not_repair_current_us_metadata(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1, "num_stocks": 12}
    ]
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    repair_calls = []
    publish_calls = []
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks._repair_current_us_group_metadata",
        lambda **kwargs: repair_calls.append(kwargs) or {"status": "ok"},
    )
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: publish_calls.append("published"))

    result = module.calculate_daily_group_rankings.run("2026-03-19", market="US")

    assert result["groups_ranked"] == 1
    assert repair_calls == []
    assert publish_calls == ["published"]
    assert result["metadata_repair"] is None


def test_non_us_market_never_invokes_us_metadata_repair(monkeypatch):
    """The US feature-store repair helper must not fire for HK/JP/TW/IN runs,
    even on same-day or bootstrap runs where the generic gate would otherwise
    trigger it.
    """
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    # Return a completed warmup so the task reaches the post-ranking repair
    # gate instead of short-circuiting on the same-day warmup-required path.
    fake_service.price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Banks", "avg_rs_rating": 90.0, "rank": 1, "num_stocks": 5}
    ]
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    repair_calls: list = []
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks._repair_current_us_group_metadata",
        lambda **kwargs: repair_calls.append(kwargs) or {"status": "ok"},
    )

    # Same-day HK run (calc_date == today_local from the calendar stub) — this
    # hits the post-ranking repair gate which only fires for market == US.
    result = module.calculate_daily_group_rankings.run("2026-03-20", market="HK")

    assert result.get("groups_ranked") == 1, f"task did not reach ranking path: {result!r}"
    assert result.get("status") != "skipped"
    assert repair_calls == [], (
        "US-scoped metadata repair must not run for non-US markets"
    )
    assert result.get("metadata_repair") is None
    # Confirm the service was invoked with market='HK', not 'US'.
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs.get("market") == "HK"


def test_gapfill_group_rankings_passes_market_to_service(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [datetime(2026, 3, 19).date()]
    fake_service.fill_gaps_optimized.return_value = {
        "processed": 1,
        "errors": 0,
    }
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.gapfill_group_rankings.run(max_days=30, market="HK")

    assert result["status"] == "complete"
    fake_service.find_missing_dates.assert_called_once_with(
        fake_db,
        lookback_days=30,
        market="HK",
    )
    fake_service.fill_gaps_optimized.assert_called_once_with(
        fake_db,
        [datetime(2026, 3, 19).date()],
        market="HK",
    )


def test_backfill_group_rankings_passes_market_to_service(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_service = MagicMock()
    fake_service.backfill_rankings_optimized.return_value = {
        "total_dates": 1,
        "deleted": 0,
        "processed": 1,
        "skipped": 0,
        "errors": 0,
    }
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.backfill_group_rankings.run("2026-03-17", "2026-03-17", market="HK")

    assert result["processed"] == 1
    fake_service.backfill_rankings_optimized.assert_called_once_with(
        fake_db,
        datetime(2026, 3, 17).date(),
        datetime(2026, 3, 17).date(),
        market="HK",
    )


def test_backfill_group_rankings_1year_passes_market_to_service(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    fake_service = MagicMock()
    fake_service.backfill_rankings_optimized.return_value = {
        "total_dates": 1,
        "deleted": 0,
        "processed": 1,
        "skipped": 0,
        "errors": 0,
    }
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.backfill_group_rankings_1year.run(market="HK")

    assert result["processed"] == 1
    fake_service.backfill_rankings_optimized.assert_called_once_with(
        fake_db,
        datetime(2025, 3, 20).date(),
        datetime(2026, 3, 20).date(),
        market="HK",
    )


def test_daily_group_rankings_fail_when_current_metadata_repair_fails(monkeypatch):
    import app.tasks.group_rank_tasks as module
    import app.services.ui_snapshot_service as snapshot_module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    fake_service.price_cache.get_warmup_metadata.return_value = {
        "status": "completed",
        "count": 10000,
        "total": 10000,
        "completed_at": datetime.now().isoformat(),
    }
    fake_service.calculate_group_rankings.return_value = [
        {"industry_group": "Software", "avg_rs_rating": 95.0, "rank": 1, "num_stocks": 12}
    ]
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks._repair_current_us_group_metadata",
        MagicMock(side_effect=RuntimeError("repair failed")),
    )
    publish_snapshot = MagicMock()
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", publish_snapshot)
    completed = MagicMock()
    monkeypatch.setattr(module, "mark_market_activity_completed", completed)

    result = module.calculate_daily_group_rankings.run(market="US")

    assert "error" in result
    assert "repair failed" in result["error"].lower()
    completed.assert_not_called()
    publish_snapshot.assert_not_called()


def test_orchestrator_gapfills_then_runs_today_on_trading_day(monkeypatch):
    """On a trading day, orchestrator gap-fills missing dates then runs the daily task."""
    import app.tasks.group_rank_tasks as module
    from datetime import date as date_cls

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        datetime(2026, 3, 20, 17, 40, 0),
        is_trading_day=True,
    )
    monkeypatch.setattr(module.settings, "group_rank_gapfill_enabled", True)
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        staticmethod(lambda db, *, market=None: ["Software", "Banks"]),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )

    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [date_cls(2026, 3, 19)]
    fake_service.fill_gaps_optimized.return_value = {
        "total_dates": 1, "processed": 1, "errors": 0,
    }
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    daily_calls: list[dict] = []
    monkeypatch.setattr(
        module,
        "_calculate_daily_group_rankings_in_process",
        lambda **kw: daily_calls.append(kw) or {"date": "2026-03-20", "groups_ranked": 2},
    )

    result = module.calculate_daily_group_rankings_with_gapfill.run(market="US")

    assert result["gap_fill"]["processed"] == 1
    assert result["today"]["date"] == "2026-03-20"
    assert daily_calls == [{"market": "US", "activity_lifecycle": "daily_refresh"}]
    fake_service.find_missing_dates.assert_called_once_with(
        fake_db, lookback_days=365, market="US",
    )
    fake_service.fill_gaps_optimized.assert_called_once_with(
        fake_db, [date_cls(2026, 3, 19)], market="US",
    )


def test_orchestrator_gapfills_but_skips_today_on_non_trading_day(monkeypatch):
    """On a non-trading day, orchestrator still gap-fills but skips the daily call."""
    import app.tasks.group_rank_tasks as module
    from datetime import date as date_cls

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        datetime(2026, 4, 25, 9, 0, 0),  # Saturday
        is_trading_day=False,
    )
    monkeypatch.setattr(module.settings, "group_rank_gapfill_enabled", True)
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        staticmethod(lambda db, *, market=None: ["Software"]),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )

    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [date_cls(2026, 4, 24)]
    fake_service.fill_gaps_optimized.return_value = {
        "total_dates": 1, "processed": 1, "errors": 0,
    }
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    daily_helper = MagicMock()
    monkeypatch.setattr(
        module, "_calculate_daily_group_rankings_in_process", daily_helper,
    )

    result = module.calculate_daily_group_rankings_with_gapfill.run(market="US")

    assert result["gap_fill"]["processed"] == 1
    assert result["today"]["skipped"] is True
    assert "not a trading day" in result["today"]["reason"].lower()
    daily_helper.assert_not_called()


def test_orchestrator_no_gaps_no_calc_when_disabled_market(monkeypatch):
    """A market disabled in runtime preferences short-circuits without DB work."""
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: False,
    )

    fake_service = MagicMock()
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings_with_gapfill.run(market="HK")

    assert result["status"] == "skipped"
    assert "disabled" in result["reason"].lower()
    fake_service.find_missing_dates.assert_not_called()
    fake_service.fill_gaps_optimized.assert_not_called()


def test_orchestrator_skips_market_with_no_taxonomy(monkeypatch):
    """An enabled market with no industry-group taxonomy is skipped gracefully."""
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        staticmethod(lambda db, *, market=None: []),
    )

    fake_service = MagicMock()
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)

    result = module.calculate_daily_group_rankings_with_gapfill.run(market="JP")

    assert result["status"] == "skipped"
    assert result["reason"] == "no_taxonomy_for_market"
    fake_service.find_missing_dates.assert_not_called()


def test_orchestrator_runs_for_non_us_market(monkeypatch):
    """HK orchestrator runs end-to-end on a trading day."""
    import app.tasks.group_rank_tasks as module
    from datetime import date as date_cls

    fake_db = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(
        monkeypatch,
        datetime(2026, 3, 20, 17, 40, 0),
        is_trading_day=True,
    )
    monkeypatch.setattr(module.settings, "group_rank_gapfill_enabled", True)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _m: True,
    )
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        staticmethod(lambda db, *, market=None: ["HK_Software"]),
    )

    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = []
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(
        module,
        "_calculate_daily_group_rankings_in_process",
        lambda **kw: {"date": "2026-03-20", "market": kw.get("market")},
    )

    result = module.calculate_daily_group_rankings_with_gapfill.run(market="HK")

    assert "status" not in result
    assert result["market"] == "HK"
    assert result["today"]["market"] == "HK"
    assert result["today"]["date"] == "2026-03-20"
    fake_service.find_missing_dates.assert_called_once_with(
        fake_db, lookback_days=365, market="HK",
    )


def test_orchestrator_bypasses_lease_for_inner_daily_call(monkeypatch):
    """The shim must engage `disable_serialized_market_workload()` so the inner
    daily task's @serialized_market_workload decorator doesn't try to re-acquire
    the per-market lease the orchestrator already holds.
    """
    import app.tasks.group_rank_tasks as module
    import app.tasks.workload_coordination as wc

    seen_disabled_state: list[bool] = []

    real_task = MagicMock()
    real_task.__module__ = "app.tasks.group_rank_tasks"
    real_task.request = MagicMock()

    def fake_run(market=None, activity_lifecycle=None):
        seen_disabled_state.append(wc._SERIALIZED_MARKET_WORKLOAD_DISABLED.get())
        return {"date": "2026-03-20", "market": market}

    real_task.run = fake_run
    monkeypatch.setattr(module, "calculate_daily_group_rankings", real_task)

    assert wc._SERIALIZED_MARKET_WORKLOAD_DISABLED.get() is False
    result = module._calculate_daily_group_rankings_in_process(
        market="US", activity_lifecycle="daily_refresh",
    )

    assert result == {"date": "2026-03-20", "market": "US"}
    assert seen_disabled_state == [True], (
        "The lease bypass ContextVar must be set to True while the inner task runs"
    )
    assert wc._SERIALIZED_MARKET_WORKLOAD_DISABLED.get() is False, (
        "The bypass must be reset on context exit"
    )

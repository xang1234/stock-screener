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

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
import pytest_asyncio
from celery.exceptions import Retry, SoftTimeLimitExceeded
from sqlalchemy.exc import OperationalError

from app.main import app


def _price_df(day: date, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [close - 1],
            "High": [close + 1],
            "Low": [close - 2],
            "Close": [close],
            "Adj Close": [close - 0.5],
            "Volume": [1_000_000],
        },
        index=pd.to_datetime([day]),
    )


def _success_result(symbol: str, close: float = 100.0) -> dict:
    return {
        "symbol": symbol,
        "price_data": _price_df(date(2026, 3, 20), close),
        "info": None,
        "fundamentals": None,
        "has_error": False,
        "error": None,
    }


def _postgres_recovery_error() -> OperationalError:
    return OperationalError(
        "select 1",
        {},
        Exception(
            "FATAL:  the database system is not yet accepting connections\n"
            "DETAIL:  Consistent recovery state has not been yet reached."
        ),
    )


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _patch_serialized_coordination(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (True, False)
    fake_coordination.release_market_workload.return_value = True
    fake_coordination.acquire_external_fetch.return_value = (True, False)
    fake_coordination.release_external_fetch.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
    )
    return fake_lock, fake_coordination


def test_daily_cache_warmup_delegates_to_smart_refresh(monkeypatch):
    import app.tasks.cache_tasks as module

    _patch_serialized_coordination(monkeypatch)

    delegated_result = {"status": "completed", "mode": "full"}
    delegated = MagicMock(return_value=delegated_result)
    monkeypatch.setattr(module, "smart_refresh_cache", delegated)

    result = module.daily_cache_warmup.run()

    assert result == delegated_result
    delegated.assert_called_once()
    assert delegated.call_args.args[0].name == module.daily_cache_warmup.name
    # After bead 9.1 daily_cache_warmup accepts a `market` kwarg and forwards it
    # to smart_refresh_cache (None → shared scope by default).
    assert delegated.call_args.kwargs == {"mode": "full", "market": None}


def test_celery_schedule_moves_orphan_cleanup_and_keeps_legacy_manual_routes():
    from app.celery_app import celery_app

    orphan_cleanup = celery_app.conf.beat_schedule["weekly-orphaned-scan-cleanup"]["schedule"]
    assert orphan_cleanup._orig_hour == 1
    assert orphan_cleanup._orig_minute == 45
    assert orphan_cleanup._orig_day_of_week == 0

    assert "app.tasks.cache_tasks.daily_cache_warmup" in celery_app.conf.task_routes
    assert "app.tasks.cache_tasks.auto_refresh_after_close" in celery_app.conf.task_routes


def test_smart_refresh_cache_reraises_soft_time_limit(monkeypatch):
    import app.tasks.cache_tasks as module

    _patch_serialized_coordination(monkeypatch)

    fake_db = MagicMock()
    fake_price_cache = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_eastern_now", lambda: SimpleNamespace(weekday=lambda: 6, hour=2, date=lambda: date(2026, 3, 22)))
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(side_effect=SoftTimeLimitExceeded()))
    monkeypatch.setattr(module, "safe_rollback", MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )

    with pytest.raises(SoftTimeLimitExceeded):
        module.smart_refresh_cache.run.__wrapped__(module.smart_refresh_cache, "full")

    module.safe_rollback.assert_called_once_with(fake_db)
    fake_price_cache.save_warmup_metadata.assert_called_once()
    fake_price_cache.complete_warmup_heartbeat.assert_called_once_with("failed", market=None)
    fake_db.close.assert_called_once()


def test_smart_refresh_cache_allows_in_process_bypass_outside_time_window(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.order_by.return_value.all.return_value = []
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 0, hour=9, date=lambda: date(2026, 3, 23)),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: MagicMock(),
    )

    with module.allow_smart_refresh_time_window_bypass():
        result = module.smart_refresh_cache.run.__wrapped__(module.smart_refresh_cache, "full")

    assert result["status"] == "completed"
    assert result["message"] == "No active symbols found in universe"
    fake_price_cache.save_warmup_metadata.assert_called_once_with("completed", 0, 0, market=None)
    fake_db.close.assert_called_once()


def test_smart_refresh_cache_non_us_market_skips_time_window_guard(monkeypatch):
    """Per-market beat entries (HK/JP/TW) fire at local-market hours in ET (e.g.
    HK at 4:30 AM ET). The legacy US-only time-window guard would reject them
    as catchup storms. Bead 9.1: when market is explicit, skip the guard."""
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.order_by.return_value.all.return_value = []
    fake_query.filter.return_value.order_by.return_value.all.return_value = []
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    # Simulate HK's schedule firing at 4:30 AM ET Tuesday — outside the US window.
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=4, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: MagicMock(),
    )

    result = module.smart_refresh_cache.run.__wrapped__(
        module.smart_refresh_cache, "full", market="HK"
    )

    # Should NOT be rejected with 'Outside refresh window'.
    assert result.get("skipped") is not True or "Outside refresh window" not in result.get("reason", "")


def test_smart_refresh_cache_publishes_failed_market_activity(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    fake_price_cache = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(side_effect=RuntimeError("benchmark unavailable")))
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: MagicMock(),
    )

    started = []
    failed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_failed", lambda *args, **kwargs: failed.append(kwargs))

    result = module.smart_refresh_cache.run.__wrapped__(module.smart_refresh_cache, "full", market="US")

    assert result["status"] == "failed"
    assert started[0]["stage_key"] == "prices"
    assert started[0]["lifecycle"] == "daily_refresh"
    assert failed[0]["stage_key"] == "prices"
    assert "benchmark unavailable" in failed[0]["message"]


def test_smart_refresh_cache_retries_transient_database_errors_from_task_body(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_lock, fake_coordination = _patch_serialized_coordination(monkeypatch)

    fake_db = MagicMock()
    fake_price_cache = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(side_effect=_postgres_recovery_error()))
    monkeypatch.setattr(module, "safe_rollback", MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )

    retry_calls = []

    def fake_retry(*, exc=None, countdown=None, max_retries=None):
        retry_calls.append(
            {
                "exc": exc,
                "countdown": countdown,
                "max_retries": max_retries,
            }
        )
        raise Retry("retry")

    monkeypatch.setattr(module.smart_refresh_cache, "retry", fake_retry)
    module.smart_refresh_cache.request.id = "task-123"
    module.smart_refresh_cache.request.retries = 0

    with pytest.raises(Retry):
        module.smart_refresh_cache.run("full")

    assert retry_calls[0]["countdown"] == 5
    assert retry_calls[0]["max_retries"] == 12
    assert "database system is not yet accepting connections" in str(retry_calls[0]["exc"])
    fake_lock.release.assert_called_once_with("task-123", market=None)
    fake_coordination.release_external_fetch.assert_called_once_with("task-123")
    fake_coordination.release_market_workload.assert_called_once_with("task-123", market=None)
    module.safe_rollback.assert_not_called()


def test_smart_refresh_cache_publishes_running_progress_per_batch(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    symbols = [SimpleNamespace(symbol=f"SYM{i}") for i in range(101)]
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.order_by.return_value.all.return_value = symbols
    fake_query.filter.return_value.order_by.return_value.all.return_value = symbols
    fake_db.query.return_value = fake_query

    fake_price_cache = MagicMock()
    fake_lock = MagicMock()
    bulk_fetcher = MagicMock()
    progress_updates: list[dict] = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        module,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(sync_from_github=lambda *_args, **_kwargs: {"status": "missing"}),
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: bulk_fetcher,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: SimpleNamespace(
            acquire_market_workload=lambda *args, **kwargs: (True, False),
            release_market_workload=lambda *args, **kwargs: True,
            acquire_external_fetch=lambda *args, **kwargs: (True, False),
            release_external_fetch=lambda *args, **kwargs: True,
        ),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_rate_limiter",
        lambda: SimpleNamespace(wait_for_market=lambda *args, **kwargs: None, wait=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(module, "_track_symbol_failures", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_fetch_with_backoff",
        lambda _fetcher, batch_symbols, **kwargs: {
            symbol: _success_result(symbol)
            for symbol in batch_symbols
        },
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_progress",
        lambda *args, **kwargs: progress_updates.append(kwargs),
    )
    monkeypatch.setattr(module.smart_refresh_cache, "update_state", lambda *args, **kwargs: None)

    result = module.smart_refresh_cache.run.__wrapped__(
        module.smart_refresh_cache, "full", market="US", activity_lifecycle="bootstrap"
    )

    assert result["status"] == "completed"
    assert progress_updates[0]["current"] == 0
    assert progress_updates[0]["total"] == 101
    assert progress_updates[0]["percent"] == 0
    assert any(update["message"] == "Batch 1/2 · refreshing prices" for update in progress_updates)
    assert any(update["message"] == "Batch 2/2 · refreshing prices" for update in progress_updates)
    assert progress_updates[-1]["current"] == 101
    assert progress_updates[-1]["total"] == 101
    assert progress_updates[-1]["percent"] == pytest.approx(100.0)


def test_smart_refresh_cache_prefers_github_daily_bundle_and_skips_live_fetch(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    symbols = [SimpleNamespace(symbol="AAPL"), SimpleNamespace(symbol="MSFT")]
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.order_by.return_value.all.return_value = symbols
    fake_query.filter.return_value.order_by.return_value.all.return_value = symbols
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 4, 21)),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: (_ for _ in ()).throw(AssertionError("live batch fetch should not run")),
    )
    monkeypatch.setattr(
        module,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(
            sync_from_github=lambda db, market, warm_redis_symbols=None: {
                "status": "success",
                "source": "github",
                "market": market,
                "as_of_date": "2026-04-21",
                "source_revision": "daily_prices_us:20260421120000",
            },
            symbols_missing_as_of=lambda db, symbols, as_of_date: [],
        ),
    )

    started = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    result = module.smart_refresh_cache.run.__wrapped__(
        module.smart_refresh_cache, "full", market="US"
    )

    assert result["status"] == "completed"
    assert result["source"] == "github"
    assert result["refreshed"] == 0
    assert started[0]["stage_key"] == "prices"
    assert completed[0]["stage_key"] == "prices"


def test_shared_smart_refresh_records_success_for_each_symbol_market(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    rows = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="0700.HK", market="HK"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value = fake_query
    fake_query.order_by.return_value = fake_query
    fake_query.all.return_value = rows
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()
    records = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: fake_price_cache)
    monkeypatch.setattr("app.services.bulk_data_fetcher.BulkDataFetcher", lambda: MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: SimpleNamespace(extend_lock=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_rate_limiter",
        lambda: SimpleNamespace(wait=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        module,
        "get_market_calendar_service",
        lambda: SimpleNamespace(
            last_completed_trading_day=lambda market: date(2026, 3, 24)
        ),
    )
    monkeypatch.setattr(module, "_track_symbol_failures", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.smart_refresh_cache, "update_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_fetch_with_backoff",
        lambda _fetcher, batch_symbols, **kwargs: {
            symbol: _success_result(symbol) for symbol in batch_symbols
        },
    )
    monkeypatch.setattr(
        module,
        "_record_market_refresh_success_safely",
        lambda _db, **kwargs: records.append(kwargs),
    )

    result = module.smart_refresh_cache.run.__wrapped__(module.smart_refresh_cache, "full")

    assert result["status"] == "completed"
    assert sorted(record["market"] for record in records) == ["HK", "US"]


def test_shared_smart_refresh_retries_failed_symbols_by_symbol_market(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    rows = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="0700.HK", market="HK"),
        SimpleNamespace(symbol="MSFT", market="US"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value = fake_query
    fake_query.order_by.return_value = fake_query
    fake_query.all.return_value = rows
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()
    retry_calls = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: fake_price_cache)
    monkeypatch.setattr("app.services.bulk_data_fetcher.BulkDataFetcher", lambda: MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: SimpleNamespace(extend_lock=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_rate_limiter",
        lambda: SimpleNamespace(wait=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(module, "_track_symbol_failures", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.smart_refresh_cache, "update_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_fetch_with_backoff",
        lambda _fetcher, batch_symbols, **kwargs: {
            "AAPL": {"has_error": True, "error": "rate limited", "price_data": None},
            "0700.HK": {"has_error": True, "error": "rate limited", "price_data": None},
            "MSFT": _success_result("MSFT"),
        },
    )
    monkeypatch.setattr(
        module,
        "_schedule_failed_symbol_retry",
        lambda symbols, *, market, attempt: retry_calls.append(
            {"symbols": symbols, "market": market, "attempt": attempt}
        ),
    )

    result = module.smart_refresh_cache.run.__wrapped__(module.smart_refresh_cache, "full")

    assert result["status"] == "partial"
    assert sorted(retry_calls, key=lambda call: call["market"]) == [
        {"symbols": ["0700.HK"], "market": "HK", "attempt": 1},
        {"symbols": ["AAPL"], "market": "US", "attempt": 1},
    ]


def test_bootstrap_explicit_market_smart_refresh_uses_github_seed(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    rows = [
        SimpleNamespace(symbol="0700.HK", market="HK"),
        SimpleNamespace(symbol="0005.HK", market="HK"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value = fake_query
    fake_query.order_by.return_value = fake_query
    fake_query.all.return_value = rows
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()
    github_service = MagicMock()
    github_service.sync_from_github.return_value = {
        "status": "success",
        "as_of_date": "2026-03-24",
        "source_revision": "sha-123",
    }
    github_service.symbols_missing_as_of.return_value = []
    retry_calls = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: fake_price_cache)
    monkeypatch.setattr(module, "get_daily_price_bundle_service", lambda: github_service)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "get_market_calendar_service",
        lambda: SimpleNamespace(
            last_completed_trading_day=lambda market: date(2026, 3, 24)
        ),
    )
    monkeypatch.setattr(
        module,
        "_fetch_with_backoff",
        lambda *args, **kwargs: pytest.fail("bootstrap current GitHub seed must not live-fetch"),
    )
    monkeypatch.setattr(
        module,
        "_schedule_failed_symbol_retry",
        lambda symbols, *, market, attempt, countdown=600: retry_calls.append(
            {"symbols": symbols, "market": market, "attempt": attempt, "countdown": countdown}
        ),
    )

    result = module.smart_refresh_cache.run.__wrapped__(
        module.smart_refresh_cache,
        mode="full",
        market="HK",
        activity_lifecycle="bootstrap",
    )

    assert result["status"] == "completed"
    assert result["source"] == "github"
    github_service.sync_from_github.assert_called_once_with(fake_db, market="HK")
    github_service.symbols_missing_as_of.assert_called_once_with(
        fake_db,
        symbols=["0700.HK", "0005.HK"],
        as_of_date="2026-03-24",
    )
    assert retry_calls == []


def test_bootstrap_smart_refresh_uses_short_failed_symbol_retry_delay(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    rows = [
        SimpleNamespace(symbol="0700.HK", market="HK"),
        SimpleNamespace(symbol="0005.HK", market="HK"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value = fake_query
    fake_query.order_by.return_value = fake_query
    fake_query.all.return_value = rows
    fake_db.query.return_value = fake_query
    fake_price_cache = MagicMock()
    retry_calls = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: fake_price_cache)
    monkeypatch.setattr(
        module,
        "get_daily_price_bundle_service",
        lambda: SimpleNamespace(sync_from_github=lambda *_args, **_kwargs: {"status": "missing"}),
    )
    monkeypatch.setattr("app.services.bulk_data_fetcher.BulkDataFetcher", lambda: MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: SimpleNamespace(extend_lock=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_rate_limiter",
        lambda: SimpleNamespace(wait_for_market=lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(module, "_track_symbol_failures", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.smart_refresh_cache, "update_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_fetch_with_backoff",
        lambda _fetcher, batch_symbols, **kwargs: {
            symbol: {"has_error": True, "error": "rate limited", "price_data": None}
            for symbol in batch_symbols
        },
    )
    monkeypatch.setattr(
        module,
        "_schedule_failed_symbol_retry",
        lambda symbols, *, market, attempt, countdown=600: retry_calls.append(
            {"symbols": symbols, "market": market, "attempt": attempt, "countdown": countdown}
        ),
    )

    result = module.smart_refresh_cache.run.__wrapped__(
        module.smart_refresh_cache,
        mode="full",
        market="HK",
        activity_lifecycle="bootstrap",
    )

    assert result["status"] == "partial"
    assert retry_calls == [
        {
            "symbols": ["0700.HK", "0005.HK"],
            "market": "HK",
            "attempt": 1,
            "countdown": 30,
        }
    ]


def test_failed_price_retry_preserves_bootstrap_retry_delay(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_price_cache = MagicMock()
    retry_calls = []

    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: fake_price_cache)
    monkeypatch.setattr("app.services.bulk_data_fetcher.BulkDataFetcher", lambda: MagicMock())
    monkeypatch.setattr(module, "_track_symbol_failures", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "_fetch_with_backoff",
        lambda _fetcher, batch_symbols, **kwargs: {
            symbol: {"has_error": True, "error": "rate limited", "price_data": None}
            for symbol in batch_symbols
        },
    )
    monkeypatch.setattr(
        module,
        "_schedule_failed_symbol_retry",
        lambda symbols, *, market, attempt, countdown=600: retry_calls.append(
            {"symbols": symbols, "market": market, "attempt": attempt, "countdown": countdown}
        ),
    )

    result = module.retry_failed_price_symbols.run.__wrapped__(
        module.retry_failed_price_symbols,
        symbols=["AAPL"],
        market="US",
        attempt=2,
        retry_countdown=30,
    )

    assert result["status"] == "partial"
    assert retry_calls == [
        {"symbols": ["AAPL"], "market": "US", "attempt": 3, "countdown": 30}
    ]


def test_smart_refresh_cache_rolls_back_before_failure_reporting(monkeypatch):
    import app.tasks.cache_tasks as module

    fake_db = MagicMock()
    fake_price_cache = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_eastern_now",
        lambda: SimpleNamespace(weekday=lambda: 1, hour=17, date=lambda: date(2026, 3, 24)),
    )
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(side_effect=RuntimeError("benchmark unavailable")))
    monkeypatch.setattr(module, "safe_rollback", MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        MagicMock(side_effect=RuntimeError("activity store unavailable")),
    )

    result = module.smart_refresh_cache.run.__wrapped__(module.smart_refresh_cache, "full", market="US")

    assert result["status"] == "failed"
    module.safe_rollback.assert_called_once_with(fake_db)
    fake_price_cache.save_warmup_metadata.assert_called_once()


def test_weekly_full_refresh_reraises_soft_time_limit(monkeypatch):
    import app.tasks.cache_tasks as module

    _patch_serialized_coordination(monkeypatch)

    fake_db = MagicMock()
    first_query = MagicMock()
    first_query.filter.return_value.order_by.return_value.all.return_value = [SimpleNamespace(symbol="AAPL")]
    fake_db.query.return_value = first_query

    fake_price_cache = MagicMock()
    fake_cache_manager = MagicMock()
    fake_cache_manager.cleanup_orphaned_cache_keys.return_value = 0

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "CacheManager", lambda db: fake_cache_manager)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(side_effect=SoftTimeLimitExceeded()))
    monkeypatch.setattr(module, "safe_rollback", MagicMock())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: MagicMock(),
    )

    with pytest.raises(SoftTimeLimitExceeded):
        module.weekly_full_refresh.run()

    module.safe_rollback.assert_called_once_with(fake_db)
    fake_price_cache.save_warmup_metadata.assert_called_once()
    fake_price_cache.complete_warmup_heartbeat.assert_called_once_with("failed", market=None)
    fake_db.close.assert_called_once()

def test_weekly_full_refresh_reraises_nested_soft_time_limit(monkeypatch):
    import app.tasks.cache_tasks as module

    _patch_serialized_coordination(monkeypatch)

    fake_db = MagicMock()
    first_query = MagicMock()
    first_query.filter.return_value.all.return_value = [SimpleNamespace(symbol="AAPL")]
    second_query = MagicMock()
    second_query.filter.return_value.order_by.return_value.all.return_value = [SimpleNamespace(symbol="AAPL")]
    fake_db.query.side_effect = [first_query, second_query]

    fake_price_cache = MagicMock()
    fake_cache_manager = MagicMock()
    fake_cache_manager.cleanup_orphaned_cache_keys.return_value = 0

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "CacheManager", lambda db: fake_cache_manager)
    monkeypatch.setattr(module, "warm_spy_cache", MagicMock(return_value={"status": "ok"}))
    monkeypatch.setattr(module, "safe_rollback", MagicMock())
    monkeypatch.setattr(module, "_fetch_with_backoff", MagicMock(side_effect=SoftTimeLimitExceeded()))
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_price_cache",
        lambda: fake_price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: MagicMock(),
    )

    with pytest.raises(SoftTimeLimitExceeded):
        module.weekly_full_refresh.run()

    module.safe_rollback.assert_called_once_with(fake_db)
    fake_price_cache.save_warmup_metadata.assert_called_once()
    fake_price_cache.complete_warmup_heartbeat.assert_called_once_with("failed", market=None)
    fake_db.close.assert_called_once()


def test_warm_price_cache_uses_batch_store(monkeypatch):
    import app.services.cache_manager as module

    price_cache = MagicMock()
    benchmark_cache = MagicMock()
    bulk_fetcher = MagicMock()
    bulk_fetcher.fetch_prices_in_batches.return_value = {
        "AAPL": _success_result("AAPL", close=120.0),
        "MSFT": _success_result("MSFT", close=130.0),
        "BAD": {
            "symbol": "BAD",
            "price_data": None,
            "info": None,
            "fundamentals": None,
            "has_error": True,
            "error": "No data",
        },
    }

    monkeypatch.setattr(module, "get_redis_client", lambda: None)
    monkeypatch.setattr(
        module,
        "BenchmarkCacheService",
        lambda redis_client, session_factory: benchmark_cache,
    )
    monkeypatch.setattr(
        module,
        "PriceCacheService",
        lambda redis_client, session_factory: price_cache,
    )
    monkeypatch.setattr(
        "app.services.bulk_data_fetcher.BulkDataFetcher",
        lambda: bulk_fetcher,
    )

    manager = module.CacheManager()
    result = manager.warm_price_cache(
        ["AAPL", "MSFT", "BAD"],
        batch_size=100,
        rate_limit=0,
        force_refresh=True,
    )

    assert result["successful"] == 2
    assert result["failed"] == 1
    price_cache.store_batch_in_cache.assert_called_once()
    assert price_cache.store_in_cache.call_count == 0
    stored_batch = price_cache.store_batch_in_cache.call_args.args[0]
    assert set(stored_batch) == {"AAPL", "MSFT"}
    assert price_cache.store_batch_in_cache.call_args.kwargs == {"also_store_db": True}


def test_task_registry_lists_daily_market_pipelines_only():
    from app.services.task_registry_service import TaskRegistryService
    from app.config import settings as app_settings

    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    tasks = TaskRegistryService().get_all_scheduled_tasks(db)
    names = {task["name"] for task in tasks}
    expected_universe_tasks = {
        f"weekly-universe-refresh-{market.lower()}"
        for market in app_settings.enabled_markets_list
    }
    expected_pipeline_tasks = {
        f"daily-market-pipeline-{market.lower()}"
        for market in app_settings.enabled_markets_list
    }

    assert expected_pipeline_tasks.issubset(names)
    assert expected_universe_tasks.issubset(names)
    assert "weekly-universe-refresh" not in names
    assert "daily-smart-refresh" not in names
    assert "daily-breadth-calculation" not in names
    assert "daily-group-ranking-calculation" not in names
    assert "daily-cache-warmup" not in names
    assert "auto-refresh-after-close" not in names

    daily_task = next(task for task in tasks if task["name"] == "daily-market-pipeline-us")
    assert daily_task["task_function"] == "app.tasks.daily_market_pipeline_tasks.queue_daily_market_pipeline"

    for market in app_settings.enabled_markets_list:
        task = next(task for task in tasks if task["name"] == f"weekly-universe-refresh-{market.lower()}")
        expected_task_function = (
            "app.tasks.universe_tasks.refresh_stock_universe"
            if market == "US"
            else "app.tasks.universe_tasks.refresh_official_market_universe"
        )
        assert task["task_function"] == expected_task_function


def test_task_registry_triggers_daily_market_pipeline_with_market(monkeypatch):
    from app.services.task_registry_service import TaskRegistryService

    service = TaskRegistryService()
    fake_task = MagicMock()
    fake_task.apply_async.return_value = SimpleNamespace(id="task-123")
    monkeypatch.setattr(service, "_get_task", lambda task_name: fake_task)

    db = MagicMock()

    def _assign_id(record):
        record.id = 99

    db.add.side_effect = _assign_id

    result = service.trigger_task("daily-market-pipeline-hk", db)

    fake_task.apply_async.assert_called_once_with(
        kwargs={"market": "HK"},
        headers={"origin": "manual"},
        queue="market_jobs_hk",
    )
    assert result["task_id"] == "task-123"
    assert result["task_name"] == "daily-market-pipeline-hk"
    assert result["execution_id"] == 99


def test_task_registry_triggers_hk_weekly_universe_refresh_on_hk_queue(monkeypatch):
    from app.services.task_registry_service import TaskRegistryService

    service = TaskRegistryService()
    fake_task = MagicMock()
    fake_task.apply_async.return_value = SimpleNamespace(id="task-hk-123")
    monkeypatch.setattr(service, "_get_task", lambda task_name: fake_task)

    db = MagicMock()

    def _assign_id(record):
        record.id = 101

    db.add.side_effect = _assign_id

    result = service.trigger_task("weekly-universe-refresh-hk", db)

    fake_task.apply_async.assert_called_once_with(
        kwargs={"market": "HK"},
        headers={"origin": "manual"},
        queue="data_fetch_shared",
    )
    assert result["task_id"] == "task-hk-123"
    assert result["task_name"] == "weekly-universe-refresh-hk"
    assert result["execution_id"] == 101


def test_task_registry_includes_supported_market_universe_entries(monkeypatch):
    import importlib
    import app.services.task_registry_service as task_registry_module

    original_enabled_markets = task_registry_module.settings.enabled_markets
    try:
        task_registry_module.settings.enabled_markets = "US,HK"
        importlib.reload(task_registry_module)

        names = set(task_registry_module.SCHEDULED_TASKS)
        expected = {
            f"weekly-universe-refresh-{market.lower()}"
            for market in task_registry_module.SUPPORTED_MARKETS
        }
        assert expected.issubset(names)
    finally:
        task_registry_module.settings.enabled_markets = original_enabled_markets
        importlib.reload(task_registry_module)


@pytest.mark.asyncio
async def test_warm_all_returns_already_running_when_refresh_active(client, monkeypatch):
    from app.api.v1 import cache as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    mock_lock = MagicMock()
    mock_lock.get_any_current_task.return_value = {
        "task_id": "running-123",
        "task_name": "smart_refresh_cache",
    }
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: mock_lock)

    apply_async = MagicMock()
    monkeypatch.setattr(module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post("/api/v1/cache/warm/all")

    assert response.status_code == 200
    assert response.json() == {
        "task_id": "running-123",
        "message": "Refresh already in progress (smart_refresh_cache)",
        "status": "already_running",
    }
    apply_async.assert_not_called()


@pytest.mark.asyncio
async def test_warm_all_queues_full_smart_refresh_when_idle(client, monkeypatch):
    from app.api.v1 import cache as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    mock_lock = MagicMock()
    mock_lock.get_any_current_task.return_value = None
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: mock_lock)

    apply_async = MagicMock(return_value=SimpleNamespace(id="queued-123"))
    monkeypatch.setattr(module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post("/api/v1/cache/warm/all")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["task_id"] == "queued-123"
    apply_async.assert_called_once_with(
        kwargs={"mode": "full"},
        headers={"origin": "manual"},
    )


@pytest.mark.asyncio
async def test_refresh_endpoint_queues_requested_mode(client, monkeypatch):
    from app.api.v1 import cache as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    mock_lock = MagicMock()
    mock_lock.get_any_current_task.return_value = None
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: mock_lock)

    apply_async = MagicMock(return_value=SimpleNamespace(id="queued-auto"))
    monkeypatch.setattr(module.smart_refresh_cache, "apply_async", apply_async)

    response = await client.post("/api/v1/cache/refresh", json={"mode": "auto"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["task_id"] == "queued-auto"
    apply_async.assert_called_once_with(
        kwargs={"mode": "auto"},
        headers={"origin": "manual"},
    )

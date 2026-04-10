from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
import pytest_asyncio
from celery.exceptions import SoftTimeLimitExceeded

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


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_daily_cache_warmup_delegates_to_smart_refresh(monkeypatch):
    import app.tasks.cache_tasks as module

    mock_lock = MagicMock()
    mock_lock.acquire.return_value = (True, False)
    mock_lock.release.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: mock_lock,
    )

    delegated_result = {"status": "completed", "mode": "full"}
    delegated = MagicMock(return_value=delegated_result)
    monkeypatch.setattr(module, "smart_refresh_cache", delegated)

    result = module.daily_cache_warmup.run()

    assert result == delegated_result
    delegated.assert_called_once()
    assert delegated.call_args.args[0].name == module.daily_cache_warmup.name
    assert delegated.call_args.kwargs == {"mode": "full"}


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

    mock_lock = MagicMock()
    mock_lock.acquire.return_value = (True, False)
    mock_lock.release.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: mock_lock,
    )

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
    fake_price_cache.complete_warmup_heartbeat.assert_called_once_with("failed")
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
    fake_price_cache.save_warmup_metadata.assert_called_once_with("completed", 0, 0)
    fake_db.close.assert_called_once()


def test_weekly_full_refresh_reraises_soft_time_limit(monkeypatch):
    import app.tasks.cache_tasks as module

    mock_lock = MagicMock()
    mock_lock.acquire.return_value = (True, False)
    mock_lock.release.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: mock_lock,
    )

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
    fake_price_cache.complete_warmup_heartbeat.assert_called_once_with("failed")
    fake_db.close.assert_called_once()

def test_weekly_full_refresh_reraises_nested_soft_time_limit(monkeypatch):
    import app.tasks.cache_tasks as module

    mock_lock = MagicMock()
    mock_lock.acquire.return_value = (True, False)
    mock_lock.release.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: mock_lock,
    )

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
    fake_price_cache.complete_warmup_heartbeat.assert_called_once_with("failed")
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


def test_task_registry_lists_daily_smart_refresh_only():
    from app.services.task_registry_service import TaskRegistryService

    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None

    tasks = TaskRegistryService().get_all_scheduled_tasks(db)
    names = {task["name"] for task in tasks}

    assert "daily-smart-refresh" in names
    assert "daily-cache-warmup" not in names
    assert "auto-refresh-after-close" not in names

    daily_task = next(task for task in tasks if task["name"] == "daily-smart-refresh")
    assert daily_task["task_function"] == "app.tasks.cache_tasks.smart_refresh_cache"


def test_task_registry_triggers_daily_smart_refresh_with_full_mode(monkeypatch):
    from app.services.task_registry_service import TaskRegistryService

    service = TaskRegistryService()
    fake_task = MagicMock()
    fake_task.apply_async.return_value = SimpleNamespace(id="task-123")
    monkeypatch.setattr(service, "_get_task", lambda task_name: fake_task)

    db = MagicMock()

    def _assign_id(record):
        record.id = 99

    db.add.side_effect = _assign_id

    result = service.trigger_task("daily-smart-refresh", db)

    fake_task.apply_async.assert_called_once_with(
        kwargs={"mode": "full"},
        headers={"origin": "manual"},
    )
    assert result["task_id"] == "task-123"
    assert result["task_name"] == "daily-smart-refresh"
    assert result["execution_id"] == 99


@pytest.mark.asyncio
async def test_warm_all_returns_already_running_when_refresh_active(client, monkeypatch):
    from app.api.v1 import cache as module
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)

    mock_lock = MagicMock()
    mock_lock.get_current_task.return_value = {
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
    mock_lock.get_current_task.return_value = None
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
    mock_lock.get_current_task.return_value = None
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

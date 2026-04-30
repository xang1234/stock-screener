from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from celery.exceptions import Retry
import pytest
import requests


def _patch_data_fetch_lock(monkeypatch):
    import app.services.provider_snapshot_service as provider_snapshot_module

    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (True, False)
    fake_coordination.release_market_workload.return_value = True
    fake_coordination.acquire_external_fetch.return_value = (True, False)
    fake_coordination.release_external_fetch.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
    )
    monkeypatch.setattr(
        provider_snapshot_module.settings,
        "market_data_source_mode",
        "live_only",
    )
    return fake_lock


def test_refresh_stock_universe_returns_original_error_when_activity_publish_fails(monkeypatch):
    import app.tasks.universe_tasks as module

    fake_db = MagicMock()
    _patch_data_fetch_lock(monkeypatch)
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "safe_rollback", MagicMock())
    monkeypatch.setattr(module, "_count_active_universe", lambda _market: 10)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mark_market_activity_progress_safely", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        MagicMock(side_effect=RuntimeError("activity store unavailable")),
    )

    class _FailingUniverseService:
        @staticmethod
        def populate_universe(db, exchange_filter=None):
            _ = db
            _ = exchange_filter
            raise RuntimeError("finviz down")

    monkeypatch.setattr(module, "get_stock_universe_service", lambda: _FailingUniverseService())

    result = module.refresh_stock_universe.run(market="US")

    assert result["status"] == "error"
    assert result["error"] == "finviz down"
    module.safe_rollback.assert_called_once_with(fake_db)


def test_refresh_official_market_universe_preserves_original_error_when_activity_publish_fails(
    monkeypatch,
):
    import app.tasks.universe_tasks as module

    fake_lock = _patch_data_fetch_lock(monkeypatch)
    activity_sessions = [MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.setattr(module, "SessionLocal", lambda: activity_sessions.pop(0))
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mark_market_activity_progress_safely", lambda **kwargs: None)
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        MagicMock(side_effect=RuntimeError("activity store unavailable")),
    )
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.OfficialMarketUniverseSourceService.fetch_market_snapshot",
        MagicMock(side_effect=RuntimeError("upstream unavailable")),
    )

    module.refresh_official_market_universe.request.id = "task-123"
    module.refresh_official_market_universe.request.retries = 0

    with pytest.raises(RuntimeError, match="upstream unavailable"):
        module.refresh_official_market_universe.run(market="TW")

    fake_lock.release.assert_called_once_with("task-123", market="TW")


def test_refresh_stock_universe_prefers_github_weekly_bundle(monkeypatch):
    import app.tasks.universe_tasks as module

    fake_db = MagicMock()
    _patch_data_fetch_lock(monkeypatch)
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "_count_active_universe", lambda _market: 10)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mark_market_activity_progress_safely", lambda **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            sync_weekly_reference_from_github=lambda db, market, hydrate_cache, hydrate_mode: {
                "status": "success",
                "source": "github",
                "market": market,
                "source_revision": "fundamentals_v1_us:20260418120000",
                "import": {"rows": 100, "universe_rows": 100},
            }
        ),
    )

    populate_calls: list[object] = []
    monkeypatch.setattr(
        module,
        "get_stock_universe_service",
        lambda: SimpleNamespace(
            populate_universe=lambda db, exchange_filter=None: populate_calls.append((db, exchange_filter))
        ),
    )

    result = module.refresh_stock_universe.run(market="US")

    assert result["status"] == "success"
    assert result["source"] == "github"
    assert not populate_calls


def test_refresh_official_market_universe_falls_back_when_github_sync_is_unsupported(monkeypatch):
    import app.tasks.universe_tasks as module

    fake_lock = _patch_data_fetch_lock(monkeypatch)
    activity_sessions = [MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.setattr(module, "SessionLocal", lambda: activity_sessions.pop(0))
    monkeypatch.setattr(module, "_count_active_universe", lambda _market: 10)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mark_market_activity_progress_safely", lambda **kwargs: None)
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            sync_weekly_reference_from_github=lambda db, market, hydrate_cache, hydrate_mode: {
                "status": "unsupported_market",
                "market": market,
            }
        ),
    )
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.OfficialMarketUniverseSourceService.fetch_market_snapshot",
        MagicMock(
            return_value=SimpleNamespace(
                market="IN",
                source_name="nse_official",
                snapshot_id="nse-20260418",
                snapshot_as_of="2026-04-18",
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "_ingest_official_snapshot",
        lambda snapshot: {
            "added": 5,
            "updated": 1,
            "total": 6,
            "source_name": snapshot.source_name,
        },
    )

    module.refresh_official_market_universe.request.id = "task-123"
    module.refresh_official_market_universe.request.retries = 0
    result = module.refresh_official_market_universe.run(market="IN")

    assert result["status"] == "success"
    assert result["market"] == "IN"
    fake_lock.release.assert_called_once_with("task-123", market="IN")


def test_refresh_official_market_universe_retries_transient_provider_failure(monkeypatch):
    import app.tasks.universe_tasks as module

    fake_lock = _patch_data_fetch_lock(monkeypatch)
    activity_sessions = [MagicMock(), MagicMock()]
    monkeypatch.setattr(module, "SessionLocal", lambda: activity_sessions.pop(0))
    monkeypatch.setattr(module, "_count_active_universe", lambda _market: 10)
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_mark_market_activity_progress_safely", lambda **kwargs: None)
    failed = MagicMock()
    monkeypatch.setattr(module, "mark_market_activity_failed", failed)
    monkeypatch.setattr(
        module,
        "get_provider_snapshot_service",
        lambda: SimpleNamespace(
            sync_weekly_reference_from_github=lambda db, market, hydrate_cache, hydrate_mode: {
                "status": "missing",
                "market": market,
            }
        ),
    )
    monkeypatch.setattr(
        "app.services.official_market_universe_source_service.OfficialMarketUniverseSourceService.fetch_market_snapshot",
        MagicMock(side_effect=requests.exceptions.ConnectionError("remote disconnected")),
    )

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.refresh_official_market_universe, "retry", fake_retry)
    module.refresh_official_market_universe.request.id = "task-123"
    module.refresh_official_market_universe.request.retries = 0

    with pytest.raises(Retry):
        module.refresh_official_market_universe.run(market="CN")

    assert retry_calls[0]["countdown"] == 300
    assert retry_calls[0]["max_retries"] == 12
    failed.assert_not_called()
    fake_lock.release.assert_called_once_with("task-123", market="CN")

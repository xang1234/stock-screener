from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _patch_data_fetch_lock(monkeypatch):
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
    activity_sessions = [MagicMock(), MagicMock()]
    monkeypatch.setattr(module, "SessionLocal", lambda: activity_sessions.pop(0))
    monkeypatch.setattr("app.services.runtime_preferences_service.is_market_enabled_now", lambda _market: True)
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: None)
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

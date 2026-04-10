from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded


def _patch_serialized_lock(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )


def test_refresh_all_fundamentals_retries_transient_outer_failures(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.all.return_value = [SimpleNamespace(symbol="AAPL")]
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: (_ for _ in ()).throw(ConnectionError("provider down")))

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.refresh_all_fundamentals, "retry", fake_retry)
    module.refresh_all_fundamentals.request.id = "task-123"
    module.refresh_all_fundamentals.request.retries = 0

    with pytest.raises(Retry):
        module.refresh_all_fundamentals.run()

    fake_db.rollback.assert_called_once()
    assert retry_calls[0]["max_retries"] == 2
    assert retry_calls[0]["countdown"] == 60
    assert module.refresh_all_fundamentals.soft_time_limit == 7200


def test_refresh_all_fundamentals_reraises_soft_time_limit(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.all.return_value = [SimpleNamespace(symbol="AAPL")]
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: (_ for _ in ()).throw(SoftTimeLimitExceeded()))

    with pytest.raises(SoftTimeLimitExceeded):
        module.refresh_all_fundamentals.run()

    fake_db.rollback.assert_called_once()


def test_refresh_all_fundamentals_reraises_nested_soft_time_limit(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.all.return_value = [SimpleNamespace(symbol="AAPL")]
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)

    fake_cache = MagicMock()
    fake_cache.get_fundamentals.side_effect = SoftTimeLimitExceeded()
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: fake_cache)

    with pytest.raises(SoftTimeLimitExceeded):
        module.refresh_all_fundamentals.run()

    fake_db.rollback.assert_called_once()

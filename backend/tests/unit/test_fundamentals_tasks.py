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
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US")
    ]
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
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US")
    ]
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
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US")
    ]
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


def test_refresh_all_fundamentals_hybrid_passes_session_factory(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US")
    ]
    fake_db.query.return_value = fake_query

    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(module.settings, "provider_snapshot_ingestion_enabled", False)
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: MagicMock())
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    captured: dict = {}

    class _HybridStub:
        def __init__(self, *args, **kwargs):
            return None

        @staticmethod
        def fetch_fundamentals_batch(*args, **kwargs):
            return {"AAPL": {"symbol": "AAPL"}}

        @staticmethod
        def store_all_caches(*args, **kwargs):
            captured["kwargs"] = kwargs
            return {
                "fundamentals_stored": 1,
                "quarterly_stored": 1,
                "failed": 0,
            }

    monkeypatch.setattr(module, "HybridFundamentalsService", _HybridStub)

    result = module.refresh_all_fundamentals_hybrid.run(include_finviz=False)

    assert result["updated"] == 1
    assert captured["kwargs"]["session_factory"] is module.SessionLocal


def test_refresh_symbols_hybrid_passes_session_factory(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: MagicMock())

    # The task now batch-resolves markets before fetch; stub SessionLocal
    # so the query doesn't try to hit a real Postgres.
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.all.return_value = [
        ("AAPL", "US"),
    ]
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)

    captured: dict = {}

    class _HybridStub:
        def __init__(self, *args, **kwargs):
            return None

        @staticmethod
        def fetch_fundamentals_batch(*args, **kwargs):
            return {"AAPL": {"symbol": "AAPL"}}

        @staticmethod
        def store_all_caches(*args, **kwargs):
            captured["kwargs"] = kwargs
            return {
                "fundamentals_stored": 1,
                "quarterly_stored": 1,
                "failed": 0,
            }

    monkeypatch.setattr(module, "HybridFundamentalsService", _HybridStub)

    result = module.refresh_symbols_hybrid.run(symbols=["AAPL"], include_finviz=False)

    assert result["updated"] == 1
    assert captured["kwargs"]["session_factory"] is module.SessionLocal

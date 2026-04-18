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
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (True, False)
    fake_coordination.release_market_workload.return_value = True
    fake_coordination.acquire_external_fetch.return_value = (True, False)
    fake_coordination.release_external_fetch.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
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


def test_refresh_all_fundamentals_retry_survives_activity_publish_failure(monkeypatch):
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
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        MagicMock(side_effect=RuntimeError("activity store unavailable")),
    )

    retry_calls = []

    def fake_retry(*args, **kwargs):
        retry_calls.append(kwargs)
        raise Retry("retry")

    monkeypatch.setattr(module.refresh_all_fundamentals, "retry", fake_retry)
    module.refresh_all_fundamentals.request.id = "task-123"
    module.refresh_all_fundamentals.request.retries = 0

    with pytest.raises(Retry):
        module.refresh_all_fundamentals.run()

    assert retry_calls[0]["countdown"] == 60


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


def test_refresh_all_fundamentals_publishes_market_activity(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US")
    ]
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US")
    ]
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(
        module,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_fundamentals=lambda *args, **kwargs: {"symbol": "AAPL"}),
    )
    monkeypatch.setattr(module, "get_ticker_validation_service", lambda: MagicMock())
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    started = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_started", lambda *args, **kwargs: started.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    result = module.refresh_all_fundamentals.run(market="US")

    assert result["updated"] == 1
    assert started[0]["stage_key"] == "fundamentals"
    assert started[0]["lifecycle"] == "weekly_refresh"
    assert completed[0]["stage_key"] == "fundamentals"
    assert completed[0]["market"] == "US"


def test_refresh_all_fundamentals_publishes_running_progress(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    stocks = [SimpleNamespace(symbol=f"SYM{i}", market="US") for i in range(30)]
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = stocks
    fake_query.filter.return_value.all.return_value = stocks
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(
        module,
        "get_fundamentals_cache",
        lambda: SimpleNamespace(get_fundamentals=lambda *args, **kwargs: {"symbol": kwargs.get("symbol", "SYM")}),
    )
    monkeypatch.setattr(module, "get_ticker_validation_service", lambda: MagicMock())
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    progress_updates = []
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: progress_updates.append(kwargs))

    result = module.refresh_all_fundamentals.run(market="US")

    assert result["updated"] == 30
    assert progress_updates
    assert progress_updates[0]["market"] == "US"
    assert progress_updates[0]["stage_key"] == "fundamentals"
    assert all(update["total"] == 30 for update in progress_updates)
    assert any(update["current"] < update["total"] for update in progress_updates)
    assert any(update["percent"] > 0 for update in progress_updates)


def test_refresh_all_fundamentals_snapshot_cutover_publishes_progress(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="MSFT", market="US"),
    ]
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="MSFT", market="US"),
    ]
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", True)
    monkeypatch.setattr(
        module,
        "_run_snapshot_pipeline",
        lambda db, publish: {
            "snapshot": {"published": True},
            "universe": {"active_symbols": 2},
            "hydrate": {"symbols_hydrated": 2},
        },
    )
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    progress_updates = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: progress_updates.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    result = module.refresh_all_fundamentals.run(market="US", activity_lifecycle="bootstrap")

    assert result["snapshot"]["published"] is True
    assert progress_updates[0]["current"] == 0
    assert progress_updates[0]["total"] == 2
    assert progress_updates[0]["percent"] == 0
    assert completed[0]["current"] == 2
    assert completed[0]["total"] == 2


def test_refresh_all_fundamentals_hybrid_snapshot_cutover_publishes_progress(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="MSFT", market="US"),
    ]
    fake_query.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="MSFT", market="US"),
    ]
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", True)
    monkeypatch.setattr(module.settings, "provider_snapshot_ingestion_enabled", False)
    monkeypatch.setattr(
        module,
        "_run_snapshot_pipeline",
        lambda db, publish: {
            "snapshot": {
                "published": True,
                "coverage": {"active_symbols": 2},
            },
            "hydrate": {"symbols_hydrated": 2},
        },
    )
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    progress_updates = []
    completed = []
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: progress_updates.append(kwargs))
    monkeypatch.setattr(module, "mark_market_activity_completed", lambda *args, **kwargs: completed.append(kwargs))

    result = module.refresh_all_fundamentals_hybrid.run(
        include_finviz=False,
        market="US",
        activity_lifecycle="bootstrap",
    )

    assert result["snapshot"]["published"] is True
    assert progress_updates[0]["current"] == 0
    assert progress_updates[0]["total"] == 2
    assert progress_updates[0]["percent"] == 0
    assert completed[0]["current"] == 2
    assert completed[0]["total"] == 2


def test_refresh_all_fundamentals_hybrid_publishes_running_progress(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    stocks = [
        SimpleNamespace(symbol="AAPL", market="US"),
        SimpleNamespace(symbol="MSFT", market="US"),
    ]
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = stocks
    fake_query.filter.return_value.all.return_value = stocks
    fake_db.query.return_value = fake_query

    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(module.settings, "provider_snapshot_ingestion_enabled", False)
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: MagicMock())
    monkeypatch.setattr(module, "get_ticker_validation_service", lambda: MagicMock())
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    progress_updates = []
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: progress_updates.append(kwargs))

    class _HybridStub:
        def __init__(self, *args, **kwargs):
            return None

        @staticmethod
        def fetch_fundamentals_batch(symbols, **kwargs):
            kwargs["progress_callback"](1, 2)
            return {symbol: {"symbol": symbol} for symbol in symbols}

        @staticmethod
        def store_all_caches(*args, **kwargs):
            return {
                "fundamentals_stored": 2,
                "quarterly_stored": 2,
                "failed": 0,
            }

    monkeypatch.setattr(module, "HybridFundamentalsService", _HybridStub)

    result = module.refresh_all_fundamentals_hybrid.run(include_finviz=False, market="US")

    assert result["updated"] == 2
    assert progress_updates
    assert progress_updates[0]["market"] == "US"
    assert progress_updates[0]["stage_key"] == "fundamentals"
    assert progress_updates[0]["current"] == 1
    assert progress_updates[0]["total"] == 2
    assert progress_updates[0]["percent"] == pytest.approx(50.0)


def test_refresh_all_fundamentals_hybrid_rolls_back_before_failure_publish(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    stocks = [SimpleNamespace(symbol="AAPL", market="US")]
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = stocks
    fake_query.filter.return_value.all.return_value = stocks
    fake_db.query.return_value = fake_query

    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)
    monkeypatch.setattr(module.settings, "provider_snapshot_ingestion_enabled", False)

    class _HybridBoom:
        def __init__(self, *args, **kwargs):
            return None

        @staticmethod
        def fetch_fundamentals_batch(*args, **kwargs):
            raise RuntimeError("hybrid fetch failed")

    monkeypatch.setattr(module, "HybridFundamentalsService", _HybridBoom)

    result = module.refresh_all_fundamentals_hybrid.run(include_finviz=False, market="US")

    fake_db.rollback.assert_called_once()
    assert result["error"] == "hybrid fetch failed"


def test_refresh_all_fundamentals_progress_counts_failed_iterations(monkeypatch):
    import app.tasks.fundamentals_tasks as module

    fake_db = MagicMock()
    stocks = [SimpleNamespace(symbol=f"SYM{i}", market="US") for i in range(30)]
    fake_query = MagicMock()
    fake_query.filter.return_value.filter.return_value.all.return_value = stocks
    fake_query.filter.return_value.all.return_value = stocks
    fake_db.query.return_value = fake_query
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "provider_snapshot_cutover_enabled", False)

    fake_cache = MagicMock()
    fake_cache.get_fundamentals.side_effect = RuntimeError("provider unavailable")
    monkeypatch.setattr(module, "get_fundamentals_cache", lambda: fake_cache)

    validation_service = MagicMock()
    validation_service.classify_error.return_value = ("provider_error", "provider unavailable")
    monkeypatch.setattr(module, "get_ticker_validation_service", lambda: validation_service)
    monkeypatch.setattr(
        module.calculate_eps_rating_percentiles,
        "delay",
        lambda: SimpleNamespace(id="eps-task-id"),
    )

    progress_updates = []
    monkeypatch.setattr(module, "mark_market_activity_progress", lambda *args, **kwargs: progress_updates.append(kwargs))

    result = module.refresh_all_fundamentals.run(market="US")

    assert result["failed"] == 30
    assert progress_updates
    assert any(update["current"] < update["total"] for update in progress_updates)
    assert progress_updates[-1]["current"] == 30
    assert progress_updates[-1]["total"] == 30

"""Regression tests for local runtime bootstrap orchestration."""

from __future__ import annotations


class _FakeSignature:
    def __init__(self, task: str, *, args=None, kwargs=None):
        self.task = task
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.queue = None

    def set(self, queue=None, **_kwargs):
        self.queue = queue
        return self


class _FakeTask:
    def __init__(self, task: str):
        self.task = task

    def si(self, *args, **kwargs):
        return _FakeSignature(self.task, args=args, kwargs=kwargs)


def test_non_us_primary_bootstrap_uses_market_scan_not_feature_snapshot(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_official_market_universe",
        _FakeTask("app.tasks.universe_tasks.refresh_official_market_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_stock_universe",
        _FakeTask("app.tasks.universe_tasks.refresh_stock_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        _FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        _FakeTask("app.tasks.fundamentals_tasks.refresh_all_fundamentals"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        _FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings",
        _FakeTask("app.tasks.group_rank_tasks.calculate_daily_group_rankings"),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        _FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )
    monkeypatch.setattr(
        module,
        "queue_market_bootstrap_scan",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.queue_market_bootstrap_scan"),
        raising=False,
    )

    signatures = module._build_market_bootstrap_signatures("HK", include_initial_scan=True)
    task_names = [signature.task for signature in signatures]

    assert "app.tasks.runtime_bootstrap_tasks.queue_market_bootstrap_scan" in task_names
    assert "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot" not in task_names
    assert [
        signature.kwargs.get("activity_lifecycle")
        for signature in signatures
        if signature.task != "app.tasks.runtime_bootstrap_tasks.queue_market_bootstrap_scan"
    ] == ["bootstrap"] * 5


def test_bootstrap_universe_name_uses_uppercase_market_code():
    from app.tasks import runtime_bootstrap_tasks as module

    assert module._bootstrap_universe_name("us") == "market:US"


def test_complete_local_runtime_bootstrap_marks_secondary_markets_queued(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeDb:
        def close(self):
            return None

    fake_db = _FakeDb()
    queued = []

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    class _FakeChain:
        def __init__(self, *signatures) -> None:
            self.signatures = signatures

        def apply_async(self):
            return _FakeAsyncResult("secondary-task-123")

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.set_bootstrap_state",
        lambda db, state: None,
    )
    monkeypatch.setattr(
        module,
        "chain",
        lambda *signatures: _FakeChain(*signatures),
    )
    monkeypatch.setattr(
        module,
        "_build_market_bootstrap_signatures",
        lambda market, include_initial_scan=False: [_FakeSignature(f"task:{market}")],
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_queued",
        lambda db, **kwargs: queued.append(kwargs),
    )

    result = module.complete_local_runtime_bootstrap("US", ["US", "HK"])

    assert result["queued_secondary"] == [{"market": "HK", "task_id": "secondary-task-123"}]
    assert queued == [
        {
            "market": "HK",
            "stage_key": "universe",
            "lifecycle": "bootstrap",
            "task_name": "runtime_bootstrap",
            "task_id": "secondary-task-123",
            "message": "Queued bootstrap for HK",
        }
    ]

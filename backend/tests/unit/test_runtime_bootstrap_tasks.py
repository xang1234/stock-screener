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

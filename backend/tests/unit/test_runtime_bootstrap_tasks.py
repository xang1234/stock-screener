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


def test_non_us_bootstrap_uses_market_feature_snapshot(monkeypatch):
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
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        _FakeTask("app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        _FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )
    signatures = module._build_market_bootstrap_signatures("HK")
    task_names = [signature.task for signature in signatures]

    assert "app.tasks.runtime_bootstrap_tasks.queue_market_bootstrap_scan" not in task_names
    assert "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot" in task_names
    assert "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill" in task_names
    assert (
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"
        in task_names
    )
    snapshot = signatures[-1]
    assert snapshot.kwargs["market"] == "HK"
    assert snapshot.kwargs["universe_name"] == "market:HK"
    assert snapshot.kwargs["publish_pointer_key"] == "latest_published_market:HK"
    assert [signature.kwargs.get("activity_lifecycle") for signature in signatures] == ["bootstrap"] * 6


def test_us_primary_bootstrap_loads_ibd_mappings_before_prices(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_stock_universe",
        _FakeTask("app.tasks.universe_tasks.refresh_stock_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_official_market_universe",
        _FakeTask("app.tasks.universe_tasks.refresh_official_market_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.industry_tasks.load_tracked_ibd_industry_groups",
        _FakeTask("app.tasks.industry_tasks.load_tracked_ibd_industry_groups"),
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
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        _FakeTask("app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        _FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )

    signatures = module._build_market_bootstrap_signatures("US")
    task_names = [signature.task for signature in signatures]

    assert task_names == [
        "app.tasks.universe_tasks.refresh_stock_universe",
        "app.tasks.industry_tasks.load_tracked_ibd_industry_groups",
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    ]
    assert signatures[1].kwargs == {
        "market": "US",
        "activity_lifecycle": "bootstrap",
    }


def test_bootstrap_universe_name_uses_uppercase_market_code():
    from app.tasks import runtime_bootstrap_tasks as module

    assert module._bootstrap_universe_name("us") == "market:US"


def test_queue_local_runtime_bootstrap_chains_all_enabled_markets_before_completion(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    captured = []

    class _FakeChain:
        def __init__(self, *signatures) -> None:
            self.signatures = signatures

        def apply_async(self):
            captured.extend(self.signatures)
            return _FakeAsyncResult("secondary-task-123")

    monkeypatch.setattr(
        module,
        "chain",
        lambda *signatures: _FakeChain(*signatures),
    )
    monkeypatch.setattr(
        module,
        "_build_market_bootstrap_signatures",
        lambda market: [_FakeSignature(f"task:{market}")],
    )

    result = module.queue_local_runtime_bootstrap(
        primary_market="US",
        enabled_markets=["HK", "US", "TW"],
    )

    assert result == "secondary-task-123"
    assert [signature.task for signature in captured] == [
        "task:US",
        "task:HK",
        "task:TW",
        "app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap",
    ]


def test_fail_local_runtime_bootstrap_preserves_active_task_owner(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeSession:
        def close(self):
            pass

    calls = []

    monkeypatch.setattr(module, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.set_bootstrap_state",
        lambda _db, _state: None,
    )
    monkeypatch.setattr(
        module,
        "mark_current_market_activity_failed",
        lambda _db, **kwargs: calls.append(kwargs),
    )

    result = module.fail_local_runtime_bootstrap.run(
        primary_market="US",
        enabled_markets=["HK"],
    )

    assert result["status"] == "failed"
    assert calls == [
        {
            "market": "HK",
            "lifecycle": "bootstrap",
            "message": "Bootstrap failed",
        }
    ]

"""Regression tests for local runtime bootstrap orchestration."""

from __future__ import annotations

import pytest


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

    def s(self, *args, **kwargs):
        return _FakeSignature(self.task, args=args, kwargs=kwargs)


def test_bootstrap_plan_uses_semantic_operations_instead_of_task_name_strings():
    from app.domain.bootstrap.plan import BootstrapOperation, build_bootstrap_plan

    market_plan = build_bootstrap_plan(primary_market="US", enabled_markets=["US"]).market_plans[0]

    assert [stage.operation for stage in market_plan.stages] == [
        BootstrapOperation.REFRESH_STOCK_UNIVERSE,
        BootstrapOperation.LOAD_TRACKED_IBD_INDUSTRY_GROUPS,
        BootstrapOperation.SMART_REFRESH_CACHE,
        BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP,
        BootstrapOperation.REFRESH_ALL_FUNDAMENTALS,
        BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL,
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL,
        BootstrapOperation.BUILD_DAILY_SNAPSHOT,
    ]
    assert all(not hasattr(stage, "task_name") for stage in market_plan.stages)


def test_non_us_bootstrap_uses_market_feature_snapshot(monkeypatch):
    from app.domain.bootstrap.plan import build_bootstrap_plan
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
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"),
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
    market_plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["HK"]).market_plans[0]
    signatures = module._build_market_bootstrap_signatures(market_plan)
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
    assert snapshot.kwargs["bootstrap_cache_only_if_covered"] is True
    assert [signature.kwargs.get("activity_lifecycle") for signature in signatures] == ["bootstrap"] * 7


def test_runtime_bootstrap_signatures_follow_bootstrap_plan(monkeypatch):
    from app.domain.bootstrap.plan import build_bootstrap_plan
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
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"),
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

    market_plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["HK"]).market_plans[0]
    signatures = module._build_market_bootstrap_signatures(market_plan)

    assert [signature.task for signature in signatures] == [
        "app.tasks.universe_tasks.refresh_official_market_universe",
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    ]
    assert signatures[2].queue == "celery"
    snapshot = signatures[-1]
    assert snapshot.kwargs["publish_pointer_key"] == "latest_published_market:HK"


def test_us_primary_bootstrap_loads_ibd_mappings_before_prices(monkeypatch):
    from app.domain.bootstrap.plan import build_bootstrap_plan
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
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"),
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

    market_plan = build_bootstrap_plan(primary_market="US", enabled_markets=["US"]).market_plans[0]
    signatures = module._build_market_bootstrap_signatures(market_plan)
    task_names = [signature.task for signature in signatures]

    assert task_names == [
        "app.tasks.universe_tasks.refresh_stock_universe",
        "app.tasks.industry_tasks.load_tracked_ibd_industry_groups",
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
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


def test_queue_local_runtime_bootstrap_splits_primary_and_background_market_chains(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    market_chains = []
    applied_chains = []
    recorded_runs = []
    events = []

    class _FakeChain:
        def __init__(self, *signatures) -> None:
            self.signatures = signatures
            market_chains.append([signature.task for signature in signatures])

        def apply_async(self, **kwargs):
            events.append(("apply", self.signatures[0].task))
            applied_chains.append(
                {
                    "tasks": [signature.task for signature in self.signatures],
                    "errback": kwargs.get("link_error"),
                }
            )
            return _FakeAsyncResult(
                "primary-task-123" if len(applied_chains) == 1 else f"background-task-{len(applied_chains)}"
            )

    monkeypatch.setattr(
        module,
        "chain",
        lambda *signatures: _FakeChain(*signatures),
    )
    monkeypatch.setattr(
        module,
        "complete_local_runtime_bootstrap",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap"),
    )
    monkeypatch.setattr(
        module,
        "complete_background_market_bootstrap",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.complete_background_market_bootstrap"),
    )
    monkeypatch.setattr(
        module,
        "fail_local_runtime_bootstrap",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.fail_local_runtime_bootstrap"),
    )
    monkeypatch.setattr(
        module,
        "fail_background_market_bootstrap",
        _FakeTask("app.tasks.runtime_bootstrap_tasks.fail_background_market_bootstrap"),
    )
    monkeypatch.setattr(
        module,
        "_build_market_bootstrap_signatures",
        lambda market_plan: [_FakeSignature(f"task:{market_plan.market}")],
    )
    monkeypatch.setattr(
        module,
        "record_runtime_bootstrap_run",
        lambda *, primary_market, enabled_markets, primary_task_id, market_task_ids, queue_state: (
            events.append(("record", queue_state, primary_task_id, dict(market_task_ids))),
            recorded_runs.append(
                {
                    "primary_market": primary_market,
                    "enabled_markets": tuple(enabled_markets),
                    "primary_task_id": primary_task_id,
                    "market_task_ids": dict(market_task_ids),
                    "queue_state": queue_state,
                }
            ),
        ),
    )

    result = module.queue_local_runtime_bootstrap(
        primary_market="US",
        enabled_markets=["HK", "US", "TW"],
    )

    assert result == "primary-task-123"
    assert events[0] == ("record", "queueing", None, {})
    assert events[-1] == (
        "record",
        "queued",
        "primary-task-123",
        {
            "US": "primary-task-123",
            "HK": "background-task-2",
            "TW": "background-task-3",
        },
    )
    assert market_chains == [
        ["task:US", "app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap"],
        ["task:HK", "app.tasks.runtime_bootstrap_tasks.complete_background_market_bootstrap"],
        ["task:TW", "app.tasks.runtime_bootstrap_tasks.complete_background_market_bootstrap"],
    ]
    assert applied_chains[0]["errback"].task == "app.tasks.runtime_bootstrap_tasks.fail_local_runtime_bootstrap"
    assert applied_chains[0]["errback"].kwargs == {
        "primary_market": "US",
    }
    assert [call["errback"].task for call in applied_chains[1:]] == [
        "app.tasks.runtime_bootstrap_tasks.fail_background_market_bootstrap",
        "app.tasks.runtime_bootstrap_tasks.fail_background_market_bootstrap",
    ]
    assert recorded_runs == [
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK", "TW"),
            "primary_task_id": None,
            "market_task_ids": {},
            "queue_state": "queueing",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK", "TW"),
            "primary_task_id": "primary-task-123",
            "market_task_ids": {
                "US": "primary-task-123",
            },
            "queue_state": "partial",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK", "TW"),
            "primary_task_id": "primary-task-123",
            "market_task_ids": {
                "US": "primary-task-123",
                "HK": "background-task-2",
            },
            "queue_state": "partial",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK", "TW"),
            "primary_task_id": "primary-task-123",
            "market_task_ids": {
                "US": "primary-task-123",
                "HK": "background-task-2",
                "TW": "background-task-3",
            },
            "queue_state": "partial",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK", "TW"),
            "primary_task_id": "primary-task-123",
            "market_task_ids": {
                "US": "primary-task-123",
                "HK": "background-task-2",
                "TW": "background-task-3",
            },
            "queue_state": "queued",
        },
    ]


def test_queue_local_runtime_bootstrap_does_not_dispatch_when_initial_manifest_fails(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    applied = []

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    def _queue(market_plan, **_kwargs):
        applied.append(market_plan.market)
        return _FakeAsyncResult(f"task-{market_plan.market.lower()}")

    monkeypatch.setattr(module, "_queue_market_bootstrap_workflow", _queue)
    monkeypatch.setattr(
        module,
        "record_runtime_bootstrap_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("manifest write failed")),
    )

    with pytest.raises(RuntimeError, match="manifest write failed"):
        module.queue_local_runtime_bootstrap(
            primary_market="US",
            enabled_markets=["US", "HK"],
        )

    assert applied == []


def test_queue_local_runtime_bootstrap_logs_late_manifest_update_failure(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    recorded_runs = []

    def _queue(market_plan, **_kwargs):
        return _FakeAsyncResult(f"task-{market_plan.market.lower()}")

    def _record(*, primary_market, enabled_markets, primary_task_id, market_task_ids, queue_state):
        recorded_runs.append(
            {
                "primary_market": primary_market,
                "enabled_markets": tuple(enabled_markets),
                "primary_task_id": primary_task_id,
                "market_task_ids": dict(market_task_ids),
                "queue_state": queue_state,
            }
        )
        if queue_state == "queued":
            raise RuntimeError("late manifest write failed")

    monkeypatch.setattr(module, "_queue_market_bootstrap_workflow", _queue)
    monkeypatch.setattr(module, "record_runtime_bootstrap_run", _record)

    result = module.queue_local_runtime_bootstrap(
        primary_market="US",
        enabled_markets=["US", "HK"],
    )

    assert result == "task-us"
    assert recorded_runs == [
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": None,
            "market_task_ids": {},
            "queue_state": "queueing",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": "task-us",
            "market_task_ids": {"US": "task-us"},
            "queue_state": "partial",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": "task-us",
            "market_task_ids": {"US": "task-us", "HK": "task-hk"},
            "queue_state": "partial",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": "task-us",
            "market_task_ids": {"US": "task-us", "HK": "task-hk"},
            "queue_state": "queued",
        }
    ]


def test_queue_local_runtime_bootstrap_records_partial_manifest_when_background_queue_fails(
    monkeypatch,
):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    recorded_runs = []

    def _queue(market_plan, **_kwargs):
        if market_plan.market == "US":
            return _FakeAsyncResult("primary-task-123")
        raise RuntimeError(f"queue failed for {market_plan.market}")

    monkeypatch.setattr(module, "_queue_market_bootstrap_workflow", _queue)
    monkeypatch.setattr(
        module,
        "record_runtime_bootstrap_run",
        lambda *, primary_market, enabled_markets, primary_task_id, market_task_ids, queue_state: recorded_runs.append(
            {
                "primary_market": primary_market,
                "enabled_markets": tuple(enabled_markets),
                "primary_task_id": primary_task_id,
                "market_task_ids": dict(market_task_ids),
                "queue_state": queue_state,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="queue failed for HK"):
        module.queue_local_runtime_bootstrap(
            primary_market="US",
            enabled_markets=["US", "HK"],
        )

    assert recorded_runs == [
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": None,
            "market_task_ids": {},
            "queue_state": "queueing",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": "primary-task-123",
            "market_task_ids": {"US": "primary-task-123"},
            "queue_state": "partial",
        },
        {
            "primary_market": "US",
            "enabled_markets": ("US", "HK"),
            "primary_task_id": "primary-task-123",
            "market_task_ids": {"US": "primary-task-123"},
            "queue_state": "dispatch_failed",
        },
    ]


def test_queue_local_runtime_bootstrap_surfaces_manifest_recording_failure(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeAsyncResult:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    def _queue(market_plan, **_kwargs):
        return _FakeAsyncResult(f"task-{market_plan.market.lower()}")

    monkeypatch.setattr(module, "_queue_market_bootstrap_workflow", _queue)
    monkeypatch.setattr(
        module,
        "record_runtime_bootstrap_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("manifest write failed")),
    )

    with pytest.raises(RuntimeError, match="manifest write failed"):
        module.queue_local_runtime_bootstrap(
            primary_market="US",
            enabled_markets=["US", "HK"],
        )


def test_apply_bootstrap_workflow_does_not_retry_without_errback_on_type_error():
    from app.tasks import runtime_bootstrap_tasks as module

    calls = []

    class _BrokenWorkflow:
        def apply_async(self, **kwargs):
            calls.append(kwargs)
            raise TypeError("bad celery signature")

    errback = object()

    with pytest.raises(TypeError, match="bad celery signature"):
        module._apply_bootstrap_workflow(_BrokenWorkflow(), errback)

    assert calls == [{"link_error": errback}]


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
    )

    assert result["status"] == "failed"
    assert calls == [
        {
            "market": "US",
            "lifecycle": "bootstrap",
            "message": "Bootstrap failed",
        }
    ]


def test_complete_local_runtime_bootstrap_evaluates_only_primary_market(monkeypatch):
    from app.services.bootstrap_readiness_service import (
        BootstrapReadiness,
        MarketBootstrapReadiness,
    )
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeReadinessService:
        def evaluate(self, db, *, enabled_markets, bootstrap_started_at=None):
            calls["evaluate"] = (db, enabled_markets, bootstrap_started_at)
            return BootstrapReadiness(
                empty_system=False,
                market_results={
                    "US": MarketBootstrapReadiness(
                        market="US",
                        core_ready=True,
                        scan_ready=True,
                    ),
                },
            )

    session = _FakeSession()
    calls = {}
    failed_markets = []

    monkeypatch.setattr(module, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        "app.services.bootstrap_readiness_service.BootstrapReadinessService",
        _FakeReadinessService,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.set_bootstrap_state",
        lambda db, state: calls.setdefault("set_bootstrap_state", (db, state)),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_preferences",
        lambda _db: type(
            "Prefs",
            (),
            {"bootstrap_started_at": "bootstrap-started-at"},
        )(),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: failed_markets.append(kwargs),
    )

    result = module.complete_local_runtime_bootstrap.run(primary_market="US")

    assert calls["evaluate"] == (session, ["US"], "bootstrap-started-at")
    assert calls["set_bootstrap_state"] == (session, "ready")
    assert result == {
        "status": "ready",
        "primary_market": "US",
        "market": "US",
    }
    assert failed_markets == []
    assert session.closed is True


def test_complete_local_runtime_bootstrap_reports_primary_readiness_failure(monkeypatch):
    from app.services.bootstrap_readiness_service import (
        BootstrapReadiness,
        MarketBootstrapReadiness,
    )
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeSession:
        def close(self):
            pass

    class _FakeReadinessService:
        def evaluate(self, db, *, enabled_markets, bootstrap_started_at=None):
            return BootstrapReadiness(
                empty_system=False,
                market_results={
                    "HK": MarketBootstrapReadiness(
                        market="HK",
                        core_ready=False,
                        scan_ready=False,
                    ),
                },
            )

    failed_markets = []

    monkeypatch.setattr(module, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(
        "app.services.bootstrap_readiness_service.BootstrapReadinessService",
        _FakeReadinessService,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_preferences",
        lambda _db: type("Prefs", (), {"bootstrap_started_at": None})(),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.set_bootstrap_state",
        lambda _db, _state: None,
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: failed_markets.append(kwargs),
    )

    result = module.complete_local_runtime_bootstrap.run(primary_market="HK")

    assert result == {
        "status": "failed",
        "primary_market": "HK",
        "market": "HK",
        "reason": "missing core market data",
    }
    assert failed_markets == [
        {
            "market": "HK",
            "stage_key": "core",
            "lifecycle": "bootstrap",
            "task_name": "runtime_bootstrap",
            "task_id": None,
            "message": "Bootstrap core data incomplete",
        },
    ]


def test_complete_local_runtime_bootstrap_uses_requested_market_readiness(monkeypatch):
    from app.services.bootstrap_readiness_service import (
        BootstrapReadiness,
        MarketBootstrapReadiness,
    )
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeSession:
        def close(self):
            pass

    class _FakeReadinessService:
        def evaluate(self, db, *, enabled_markets, bootstrap_started_at=None):
            calls["evaluate"] = (db, enabled_markets, bootstrap_started_at)
            return BootstrapReadiness(
                empty_system=False,
                market_results={
                    "HK": MarketBootstrapReadiness(
                        market="HK",
                        core_ready=False,
                        scan_ready=False,
                    ),
                    "US": MarketBootstrapReadiness(
                        market="US",
                        core_ready=True,
                        scan_ready=True,
                    ),
                },
            )

    calls = {}

    monkeypatch.setattr(module, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(
        "app.services.bootstrap_readiness_service.BootstrapReadinessService",
        _FakeReadinessService,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_preferences",
        lambda _db: type("Prefs", (), {"bootstrap_started_at": "started-at"})(),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.set_bootstrap_state",
        lambda db, state: calls.setdefault("set_bootstrap_state", (db, state)),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: calls.setdefault("mark_failed", kwargs),
    )

    result = module.complete_local_runtime_bootstrap.run(primary_market="us")

    assert calls["evaluate"][1] == ["US"]
    assert calls["set_bootstrap_state"][1] == "ready"
    assert "mark_failed" not in calls
    assert result == {
        "status": "ready",
        "primary_market": "US",
        "market": "US",
    }


def test_complete_background_market_bootstrap_marks_market_failure_without_global_state(monkeypatch):
    from app.services.bootstrap_readiness_service import (
        BootstrapReadiness,
        MarketBootstrapReadiness,
    )
    from app.tasks import runtime_bootstrap_tasks as module

    class _FakeSession:
        def close(self):
            pass

    class _FakeReadinessService:
        def evaluate(self, db, *, enabled_markets, bootstrap_started_at=None):
            calls["evaluate"] = (db, enabled_markets, bootstrap_started_at)
            return BootstrapReadiness(
                empty_system=False,
                market_results={
                    "HK": MarketBootstrapReadiness(
                        market="HK",
                        core_ready=True,
                        scan_ready=False,
                    ),
                },
            )

    session = _FakeSession()
    calls = {}
    failed_markets = []

    monkeypatch.setattr(module, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        "app.services.bootstrap_readiness_service.BootstrapReadinessService",
        _FakeReadinessService,
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.get_runtime_preferences",
        lambda _db: type("Prefs", (), {"bootstrap_started_at": "started-at"})(),
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.set_bootstrap_state",
        lambda *_args, **_kwargs: pytest.fail("background completion must not mutate global bootstrap state"),
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: failed_markets.append(kwargs),
    )

    result = module.complete_background_market_bootstrap.run(market="HK")

    assert calls["evaluate"] == (session, ["HK"], "started-at")
    assert result == {
        "status": "failed",
        "market": "HK",
        "reason": "missing published auto scan",
    }
    assert failed_markets == [
        {
            "market": "HK",
            "stage_key": "scan",
            "lifecycle": "bootstrap",
            "task_name": "runtime_bootstrap",
            "task_id": None,
            "message": "Bootstrap scan did not publish",
        }
    ]

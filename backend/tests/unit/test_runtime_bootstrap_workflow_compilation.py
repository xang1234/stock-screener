"""Runtime bootstrap workflow compilation tests."""

from __future__ import annotations

from tests.unit.runtime_bootstrap_test_fakes import FakeTask


def test_bootstrap_plan_uses_semantic_operations_instead_of_task_name_strings():
    from app.domain.bootstrap.plan import BootstrapOperation, build_bootstrap_plan

    market_plan = build_bootstrap_plan(
        primary_market="US", enabled_markets=["US"]
    ).market_plans[0]

    assert [stage.operation for stage in market_plan.stages] == [
        BootstrapOperation.REFRESH_STOCK_UNIVERSE,
        BootstrapOperation.LOAD_TRACKED_IBD_INDUSTRY_GROUPS,
        BootstrapOperation.SMART_REFRESH_CACHE,
        BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP,
        BootstrapOperation.REFRESH_ALL_FUNDAMENTALS,
        BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT,
        BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL,
        BootstrapOperation.CALCULATE_MARKET_EXPOSURE,
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS,
        BootstrapOperation.BUILD_DAILY_SNAPSHOT,
        BootstrapOperation.ENSURE_GROUP_HISTORY,
    ]
    assert all(not hasattr(stage, "task_name") for stage in market_plan.stages)


def test_non_us_bootstrap_uses_market_feature_snapshot(monkeypatch):
    from app.domain.bootstrap.plan import build_bootstrap_plan
    from app.tasks import runtime_bootstrap_tasks as module

    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_official_market_universe",
        FakeTask("app.tasks.universe_tasks.refresh_official_market_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_stock_universe",
        FakeTask("app.tasks.universe_tasks.refresh_stock_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        FakeTask("app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"),
    )
    monkeypatch.setattr(
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        FakeTask("app.tasks.fundamentals_tasks.refresh_all_fundamentals"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        FakeTask("app.tasks.market_rs_tasks.calculate_market_rs_snapshot"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_market_exposure",
        FakeTask("app.tasks.breadth_tasks.calculate_market_exposure"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        FakeTask(
            "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"
        ),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )
    market_plan = build_bootstrap_plan(
        primary_market="HK", enabled_markets=["HK"]
    ).market_plans[0]
    signatures = module._build_market_bootstrap_signatures(market_plan)
    task_names = [signature.task for signature in signatures]

    assert (
        "app.tasks.runtime_bootstrap_tasks.queue_market_bootstrap_scan"
        not in task_names
    )
    assert "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot" in task_names
    assert "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill" in task_names
    assert "app.tasks.group_rank_tasks.calculate_daily_group_rankings" in task_names
    breadth = next(
        signature
        for signature in signatures
        if signature.task
        == "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"
    )
    groups = next(
        signature
        for signature in signatures
        if signature.task == "app.tasks.group_rank_tasks.calculate_daily_group_rankings"
    )
    assert breadth.kwargs["execution_policy"] == "refresh_guarded"
    assert groups.kwargs["execution_policy"] == "refresh_guarded"
    snapshot = signatures[-2]
    assert snapshot.kwargs["market"] == "HK"
    assert snapshot.kwargs["universe_name"] == "market:HK"
    assert snapshot.kwargs["publish_pointer_key"] == "latest_published_market:HK"
    assert snapshot.kwargs["bootstrap_cache_only_if_covered"] is True
    assert [signature.kwargs.get("activity_lifecycle") for signature in signatures] == [
        "bootstrap"
    ] * 10
    assert signatures[-1].task == "app.tasks.group_history_tasks.ensure_group_history"


def test_runtime_bootstrap_signatures_follow_bootstrap_plan(monkeypatch):
    from app.domain.bootstrap.plan import build_bootstrap_plan
    from app.tasks import runtime_bootstrap_tasks as module

    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_official_market_universe",
        FakeTask("app.tasks.universe_tasks.refresh_official_market_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_stock_universe",
        FakeTask("app.tasks.universe_tasks.refresh_stock_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        FakeTask("app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"),
    )
    monkeypatch.setattr(
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        FakeTask("app.tasks.fundamentals_tasks.refresh_all_fundamentals"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        FakeTask("app.tasks.market_rs_tasks.calculate_market_rs_snapshot"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_market_exposure",
        FakeTask("app.tasks.breadth_tasks.calculate_market_exposure"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        FakeTask(
            "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"
        ),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )

    market_plan = build_bootstrap_plan(
        primary_market="HK", enabled_markets=["HK"]
    ).market_plans[0]
    signatures = module._build_market_bootstrap_signatures(market_plan)

    assert [signature.task for signature in signatures] == [
        "app.tasks.universe_tasks.refresh_official_market_universe",
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.breadth_tasks.calculate_market_exposure",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        "app.tasks.group_history_tasks.ensure_group_history",
    ]
    assert signatures[2].queue == "celery"
    snapshot = signatures[-2]
    assert snapshot.kwargs["publish_pointer_key"] == "latest_published_market:HK"


def test_us_primary_bootstrap_loads_ibd_mappings_before_prices(monkeypatch):
    from app.domain.bootstrap.plan import build_bootstrap_plan
    from app.tasks import runtime_bootstrap_tasks as module

    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_stock_universe",
        FakeTask("app.tasks.universe_tasks.refresh_stock_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.universe_tasks.refresh_official_market_universe",
        FakeTask("app.tasks.universe_tasks.refresh_official_market_universe"),
    )
    monkeypatch.setattr(
        "app.tasks.industry_tasks.load_tracked_ibd_industry_groups",
        FakeTask("app.tasks.industry_tasks.load_tracked_ibd_industry_groups"),
    )
    monkeypatch.setattr(
        "app.tasks.cache_tasks.smart_refresh_cache",
        FakeTask("app.tasks.cache_tasks.smart_refresh_cache"),
    )
    monkeypatch.setattr(
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        FakeTask("app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"),
    )
    monkeypatch.setattr(
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        FakeTask("app.tasks.fundamentals_tasks.refresh_all_fundamentals"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        FakeTask("app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill"),
    )
    monkeypatch.setattr(
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        FakeTask("app.tasks.market_rs_tasks.calculate_market_rs_snapshot"),
    )
    monkeypatch.setattr(
        "app.tasks.breadth_tasks.calculate_market_exposure",
        FakeTask("app.tasks.breadth_tasks.calculate_market_exposure"),
    )
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill",
        FakeTask(
            "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"
        ),
    )
    monkeypatch.setattr(
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        FakeTask("app.interfaces.tasks.feature_store_tasks.build_daily_snapshot"),
    )

    market_plan = build_bootstrap_plan(
        primary_market="US", enabled_markets=["US"]
    ).market_plans[0]
    signatures = module._build_market_bootstrap_signatures(market_plan)
    task_names = [signature.task for signature in signatures]

    assert task_names == [
        "app.tasks.universe_tasks.refresh_stock_universe",
        "app.tasks.industry_tasks.load_tracked_ibd_industry_groups",
        "app.tasks.cache_tasks.smart_refresh_cache",
        "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
        "app.tasks.fundamentals_tasks.refresh_all_fundamentals",
        "app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
        "app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill",
        "app.tasks.breadth_tasks.calculate_market_exposure",
        "app.tasks.group_rank_tasks.calculate_daily_group_rankings",
        "app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
        "app.tasks.group_history_tasks.ensure_group_history",
    ]
    assert signatures[1].kwargs == {
        "market": "US",
        "activity_lifecycle": "bootstrap",
    }


def test_bootstrap_universe_name_uses_uppercase_market_code():
    from app.tasks import runtime_bootstrap_tasks as module

    assert module._bootstrap_universe_name("us") == "market:US"


def test_bootstrap_includes_every_daily_pipeline_compute_step():
    # Regression guard: the scheduled daily pipeline and the first-run bootstrap
    # are two separate chains that both encode the market-compute sequence
    # (Market RS -> breadth -> exposure -> groups -> snapshot). Exposure once
    # shipped missing from the bootstrap. Assert every compute step in the daily
    # pipeline is also in the bootstrap plan, so a new step can't be half-wired.
    from datetime import date

    from app.domain.bootstrap.plan import build_bootstrap_plan
    from app.tasks.daily_market_pipeline_tasks import (
        _build_daily_market_pipeline_signatures,
    )
    from app.tasks.runtime_bootstrap_tasks import _build_market_bootstrap_signatures

    daily = {
        s.task for s in _build_daily_market_pipeline_signatures("US", date(2026, 6, 1))
    }
    daily_compute = {
        task
        for task in daily
        if "guard" not in task
        and "smart_refresh" not in task
        and "dispatch_options_after_snapshot" not in task
    }

    plan = build_bootstrap_plan(
        primary_market="US", enabled_markets=["US"]
    ).market_plans[0]
    bootstrap = {s.task for s in _build_market_bootstrap_signatures(plan)}

    if "app.tasks.group_rank_tasks.calculate_daily_group_rankings" in bootstrap:
        daily_compute.discard(
            "app.tasks.group_rank_tasks.calculate_daily_group_rankings_with_gapfill"
        )

    missing = daily_compute - bootstrap
    assert not missing, (
        f"daily-pipeline compute steps missing from bootstrap: {missing}"
    )

from __future__ import annotations

from app.domain.bootstrap.plan import BootstrapQueueKind, build_bootstrap_plan


def test_us_bootstrap_plan_includes_us_only_industry_group_seed() -> None:
    plan = build_bootstrap_plan(primary_market="US", enabled_markets=["US"])

    assert [stage.key for stage in plan.market_plans[0].stages] == [
        "universe",
        "industry_groups",
        "prices",
        "fundamentals",
        "breadth",
        "groups",
        "snapshot",
    ]
    assert plan.market_plans[0].stages[1].queue_kind == BootstrapQueueKind.MARKET_JOBS


def test_non_us_bootstrap_plan_uses_official_universe_without_industry_seed() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["HK", "US"])
    hk_plan = plan.market_plans[0]

    assert hk_plan.market == "HK"
    assert [stage.task_name for stage in hk_plan.stages] == [
        "refresh_official_market_universe",
        "smart_refresh_cache",
        "refresh_all_fundamentals",
        "calculate_daily_breadth_with_gapfill",
        "calculate_daily_group_rankings_with_gapfill",
        "build_daily_snapshot",
    ]
    assert hk_plan.stages[-1].kwargs == {
        "market": "HK",
        "universe_name": "market:HK",
        "publish_pointer_key": "latest_published_market:HK",
        "activity_lifecycle": "bootstrap",
        "bootstrap_cache_only_if_covered": True,
    }


def test_bootstrap_plan_deduplicates_primary_and_enabled_markets_in_order() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["US", "HK", "US"])

    assert [market_plan.market for market_plan in plan.market_plans] == ["HK", "US"]

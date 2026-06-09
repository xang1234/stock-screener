from __future__ import annotations

from app.domain.bootstrap.plan import (
    BootstrapOperation,
    BootstrapQueueKind,
    build_bootstrap_plan,
)


def test_us_bootstrap_plan_includes_us_only_industry_group_seed() -> None:
    plan = build_bootstrap_plan(primary_market="US", enabled_markets=["US"])

    assert [stage.key for stage in plan.market_plans[0].stages] == [
        "universe",
        "industry_groups",
        "prices",
        "price_warmup",
        "fundamentals",
        "breadth",
        "groups",
        "snapshot",
    ]
    assert plan.market_plans[0].stages[1].queue_kind == BootstrapQueueKind.MARKET_JOBS
    assert plan.market_plans[0].stages[3].queue_kind == BootstrapQueueKind.CELERY


def test_non_us_bootstrap_plan_uses_official_universe_without_industry_seed() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["HK", "US"])
    hk_plan = plan.market_plans[0]

    assert hk_plan.market == "HK"
    assert [stage.operation for stage in hk_plan.stages] == [
        BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE,
        BootstrapOperation.SMART_REFRESH_CACHE,
        BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP,
        BootstrapOperation.REFRESH_ALL_FUNDAMENTALS,
        BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL,
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL,
        BootstrapOperation.BUILD_DAILY_SNAPSHOT,
    ]
    assert hk_plan.stages[-1].kwargs == {
        "market": "HK",
        "universe_name": "market:HK",
        "publish_pointer_key": "latest_published_market:HK",
        "activity_lifecycle": "bootstrap",
        "bootstrap_cache_only_if_covered": True,
    }


def test_au_bootstrap_plan_refreshes_universe_before_prices_and_fundamentals() -> None:
    plan = build_bootstrap_plan(primary_market="AU", enabled_markets=["AU"])
    au_plan = plan.market_plans[0]

    assert au_plan.market == "AU"
    assert [stage.key for stage in au_plan.stages[:3]] == [
        "universe",
        "prices",
        "price_warmup",
    ]
    assert au_plan.stages[0].operation == BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE
    assert au_plan.stages[0].kwargs["market"] == "AU"


def test_bootstrap_plan_deduplicates_primary_and_enabled_markets_in_order() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["US", "HK", "US"])

    assert [market_plan.market for market_plan in plan.market_plans] == ["HK", "US"]

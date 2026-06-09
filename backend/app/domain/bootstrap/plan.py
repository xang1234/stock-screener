"""Pure Bootstrap workflow plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from app.domain.markets.catalog import get_market_catalog


class BootstrapQueueKind(str, Enum):
    DATA_FETCH = "data_fetch"
    MARKET_JOBS = "market_jobs"
    CELERY = "celery"


class BootstrapOperation(str, Enum):
    REFRESH_STOCK_UNIVERSE = "refresh_stock_universe"
    REFRESH_OFFICIAL_MARKET_UNIVERSE = "refresh_official_market_universe"
    LOAD_TRACKED_IBD_INDUSTRY_GROUPS = "load_tracked_ibd_industry_groups"
    SMART_REFRESH_CACHE = "smart_refresh_cache"
    WAIT_FOR_BOOTSTRAP_PRICE_WARMUP = "wait_for_bootstrap_price_warmup"
    REFRESH_ALL_FUNDAMENTALS = "refresh_all_fundamentals"
    CALCULATE_DAILY_BREADTH_WITH_GAPFILL = "calculate_daily_breadth_with_gapfill"
    CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL = "calculate_daily_group_rankings_with_gapfill"
    BUILD_DAILY_SNAPSHOT = "build_daily_snapshot"


@dataclass(frozen=True)
class BootstrapStage:
    key: str
    operation: BootstrapOperation
    queue_kind: BootstrapQueueKind
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class MarketBootstrapPlan:
    market: str
    stages: tuple[BootstrapStage, ...]


@dataclass(frozen=True)
class BootstrapPlan:
    primary_market: str
    enabled_markets: tuple[str, ...]
    market_plans: tuple[MarketBootstrapPlan, ...]


def _normalize_markets(
    primary_market: str, enabled_markets: Iterable[str]
) -> tuple[str, ...]:
    catalog = get_market_catalog()
    ordered: list[str] = []

    for raw_market in (primary_market, *tuple(enabled_markets)):
        market = catalog.get(raw_market).code
        if market not in ordered:
            ordered.append(market)

    return tuple(ordered)


def _stage(
    *,
    key: str,
    operation: BootstrapOperation,
    queue_kind: BootstrapQueueKind,
    market: str,
    **kwargs: Any,
) -> BootstrapStage:
    return BootstrapStage(
        key=key,
        operation=operation,
        queue_kind=queue_kind,
        kwargs={"market": market, "activity_lifecycle": "bootstrap", **kwargs},
    )


def _build_market_plan(market: str) -> MarketBootstrapPlan:
    stages = [
        _stage(
            key="universe",
            operation=(
                BootstrapOperation.REFRESH_STOCK_UNIVERSE
                if market == "US"
                else BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE
            ),
            queue_kind=BootstrapQueueKind.DATA_FETCH,
            market=market,
        ),
    ]

    if market == "US":
        stages.append(
            _stage(
                key="industry_groups",
                operation=BootstrapOperation.LOAD_TRACKED_IBD_INDUSTRY_GROUPS,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            )
        )

    stages.extend(
        [
            _stage(
                key="prices",
                operation=BootstrapOperation.SMART_REFRESH_CACHE,
                queue_kind=BootstrapQueueKind.DATA_FETCH,
                market=market,
                mode="bootstrap",
            ),
            _stage(
                key="price_warmup",
                operation=BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP,
                queue_kind=BootstrapQueueKind.CELERY,
                market=market,
            ),
            _stage(
                key="fundamentals",
                operation=BootstrapOperation.REFRESH_ALL_FUNDAMENTALS,
                queue_kind=BootstrapQueueKind.DATA_FETCH,
                market=market,
            ),
            _stage(
                key="breadth",
                operation=BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            ),
            _stage(
                key="groups",
                operation=BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            ),
            _stage(
                key="snapshot",
                operation=BootstrapOperation.BUILD_DAILY_SNAPSHOT,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
                universe_name=f"market:{market}",
                publish_pointer_key=f"latest_published_market:{market}",
                bootstrap_cache_only_if_covered=True,
            ),
        ]
    )

    return MarketBootstrapPlan(market=market, stages=tuple(stages))


def build_bootstrap_plan(
    *, primary_market: str, enabled_markets: Iterable[str]
) -> BootstrapPlan:
    markets = _normalize_markets(primary_market, enabled_markets)
    return BootstrapPlan(
        primary_market=markets[0],
        enabled_markets=markets,
        market_plans=tuple(_build_market_plan(market) for market in markets),
    )

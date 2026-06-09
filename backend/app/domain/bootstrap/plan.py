"""Pure Bootstrap workflow plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from app.domain.markets.catalog import get_market_catalog


class BootstrapQueueKind(str, Enum):
    DATA_FETCH = "data_fetch"
    MARKET_JOBS = "market_jobs"


@dataclass(frozen=True)
class BootstrapStage:
    key: str
    task_name: str
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
    task_name: str,
    queue_kind: BootstrapQueueKind,
    market: str,
    **kwargs: Any,
) -> BootstrapStage:
    return BootstrapStage(
        key=key,
        task_name=task_name,
        queue_kind=queue_kind,
        kwargs={"market": market, "activity_lifecycle": "bootstrap", **kwargs},
    )


def _build_market_plan(market: str) -> MarketBootstrapPlan:
    stages = [
        _stage(
            key="universe",
            task_name=(
                "refresh_stock_universe"
                if market == "US"
                else "refresh_official_market_universe"
            ),
            queue_kind=BootstrapQueueKind.DATA_FETCH,
            market=market,
        ),
    ]

    if market == "US":
        stages.append(
            _stage(
                key="industry_groups",
                task_name="load_tracked_ibd_industry_groups",
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            )
        )

    stages.extend(
        [
            _stage(
                key="prices",
                task_name="smart_refresh_cache",
                queue_kind=BootstrapQueueKind.DATA_FETCH,
                market=market,
                mode="full",
            ),
            _stage(
                key="fundamentals",
                task_name="refresh_all_fundamentals",
                queue_kind=BootstrapQueueKind.DATA_FETCH,
                market=market,
            ),
            _stage(
                key="breadth",
                task_name="calculate_daily_breadth_with_gapfill",
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            ),
            _stage(
                key="groups",
                task_name="calculate_daily_group_rankings_with_gapfill",
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            ),
            _stage(
                key="snapshot",
                task_name="build_daily_snapshot",
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

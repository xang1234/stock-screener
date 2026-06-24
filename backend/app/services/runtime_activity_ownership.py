"""Runtime activity task ownership helpers."""

from __future__ import annotations

from typing import Any

from ..tasks.market_queues import SHARED_SENTINEL, normalize_market

DEFAULT_SHARED_ACTIVITY_MARKET = "US"


def activity_market_for_runtime_task(raw_market: Any) -> str:
    normalized_market = normalize_market(raw_market or DEFAULT_SHARED_ACTIVITY_MARKET)
    if normalized_market == SHARED_SENTINEL:
        return DEFAULT_SHARED_ACTIVITY_MARKET
    return normalized_market


def cleanup_market_for_runtime_task(raw_market: Any) -> str | None:
    normalized_market = normalize_market(raw_market)
    if normalized_market == SHARED_SENTINEL:
        return None
    return normalized_market


def lock_markets_for_runtime_activity(activity_market: str) -> tuple[str | None, ...]:
    normalized_market = normalize_market(activity_market)
    if normalized_market == SHARED_SENTINEL:
        return (None,)
    if normalized_market == DEFAULT_SHARED_ACTIVITY_MARKET:
        return (normalized_market, None)
    return (normalized_market,)

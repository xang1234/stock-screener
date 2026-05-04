"""Market-scoped cache key, TTL, and freshness policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ...domain.markets import Market


DEFAULT_TTL_SECONDS_BY_NAMESPACE = {
    "benchmark": 86400,
    "fundamentals": 604800,
    "price": 604800,
}

DEFAULT_MAX_AGE_DAYS_BY_NAMESPACE = {
    "fundamentals": 7,
    "price": 1,
}


@dataclass(frozen=True, slots=True)
class MarketAwareCachePolicy:
    """Owns market-scoped cache policy shared across cache services."""

    ttl_seconds_by_namespace: dict[str, int] | None = None
    max_age_days_by_namespace: dict[str, int] | None = None

    def normalize_market(self, market: Market | str | None) -> str:
        if market is None:
            return "US"
        resolved = market if isinstance(market, Market) else Market.from_str(market)
        return resolved.code

    def key(
        self,
        namespace: str,
        symbol: str,
        *,
        market: Market | str | None = None,
        parts: tuple[str, ...] = (),
    ) -> str:
        normalized_market = self.normalize_market(market)
        normalized_symbol = str(symbol).strip().upper()
        if not namespace or not normalized_symbol:
            raise ValueError("namespace and symbol are required for cache keys")
        suffix = ":".join(str(part).strip() for part in parts if str(part).strip())
        base = f"{namespace}:{normalized_market}:{normalized_symbol}"
        return f"{base}:{suffix}" if suffix else base

    def lock_key(
        self,
        namespace: str,
        symbol: str,
        *,
        market: Market | str | None = None,
        parts: tuple[str, ...] = (),
    ) -> str:
        return f"{self.key(namespace, symbol, market=market, parts=parts)}:lock"

    def ttl_seconds(self, namespace: str, market: Market | str | None = None) -> int:
        del market
        ttl_map = self.ttl_seconds_by_namespace or DEFAULT_TTL_SECONDS_BY_NAMESPACE
        return ttl_map[namespace]

    def is_datetime_fresh(
        self,
        namespace: str,
        last_update: datetime | None,
        *,
        now: datetime | None = None,
        market: Market | str | None = None,
    ) -> bool:
        del market
        if last_update is None:
            return False
        if last_update.tzinfo is not None:
            last_update = last_update.replace(tzinfo=None)
        now = now or datetime.now()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        age_days = (now - last_update).days
        max_age_map = self.max_age_days_by_namespace or DEFAULT_MAX_AGE_DAYS_BY_NAMESPACE
        return age_days <= max_age_map[namespace]


market_cache_policy = MarketAwareCachePolicy()


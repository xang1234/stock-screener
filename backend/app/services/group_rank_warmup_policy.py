"""Policy for same-day cache-only group-rank warmup readiness."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

from app.services.cache.price_cache_warmup import evaluate_warmup_metadata
from app.services.price_coverage_policy import (
    CACHE_ONLY_MIN_PRICE_COVERAGE,
    price_coverage_policy_for_market,
)


logger = logging.getLogger(__name__)
STRICT_GROUP_RANK_CACHE_COVERAGE = CACHE_ONLY_MIN_PRICE_COVERAGE


@dataclass(frozen=True)
class SameDayGroupRankWarmupDecision:
    error: Optional[str]
    cache_coverage_min: float | None = None


def evaluate_same_day_group_rank_warmup(
    price_cache,
    market: Optional[str] = None,
) -> SameDayGroupRankWarmupDecision:
    """Decide whether same-day group rankings can use the cache-only path."""
    warmup_meta = price_cache.get_warmup_metadata(market=market) if price_cache else None
    policy = price_coverage_policy_for_market(market)
    warmup_readiness = evaluate_warmup_metadata(
        warmup_meta,
        context="same-day group ranking run",
    )
    if warmup_readiness.ready:
        return SameDayGroupRankWarmupDecision(
            error=None,
            cache_coverage_min=STRICT_GROUP_RANK_CACHE_COVERAGE,
        )

    if (
        warmup_readiness.status == "partial"
        and warmup_readiness.fresh
        and warmup_readiness.coverage_ratio is not None
        and warmup_readiness.coverage_ratio >= policy.price_min_coverage
    ):
        logger.warning(
            "Allowing same-day group rankings with partial price warmup for %s: "
            "%.1f%% >= %.1f%% price coverage threshold",
            policy.market,
            warmup_readiness.coverage_ratio * 100,
            policy.price_min_coverage * 100,
        )
        return SameDayGroupRankWarmupDecision(
            error=None,
            cache_coverage_min=policy.price_min_coverage,
        )

    return SameDayGroupRankWarmupDecision(
        error=warmup_readiness.reason,
        cache_coverage_min=None,
    )

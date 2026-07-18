"""Celery-free daily group-ranking execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import logging
import time
from typing import Any, Callable

from sqlalchemy.orm import Session

from .derived_data_execution_policy import (
    DerivedDataExecutionPolicy,
    DerivedDataValidationProfile,
)
from .group_rank_cache_policy import GroupRankCacheRequirement
from .group_rank_models import (
    GroupRankPrefetchStats,
    GroupRanking,
)
from .group_rank_warmup_policy import (
    evaluate_same_day_group_rank_warmup,
)
from .ibd_group_rank_service import IBDGroupRankService


logger = logging.getLogger(__name__)


class GroupRankWarmupIncomplete(RuntimeError):
    """Raised when same-day warmup metadata is not ready."""


class NoGroupRankingsCalculated(RuntimeError):
    """Raised when valid inputs produce no rankable groups."""

    def __init__(
        self,
        *,
        prefetch_stats: GroupRankPrefetchStats,
        duration_seconds: float,
    ) -> None:
        self.prefetch_stats = prefetch_stats
        self.duration_seconds = duration_seconds
        super().__init__(
            "No groups could be ranked (insufficient price data or all "
            "groups below 3-stock threshold)"
        )


@dataclass(frozen=True)
class DailyGroupRankRequest:
    calculation_date: date
    current_date: date
    market: str
    activity_lifecycle: str
    policy: DerivedDataExecutionPolicy


@dataclass(frozen=True)
class DailyGroupRankDependencies:
    service: IBDGroupRankService
    bump_epoch: Callable[[str], None]
    publish_snapshot: Callable[[], object]
    repair_current_us_metadata: (
        Callable[[date], object] | None
    ) = None


@dataclass(frozen=True)
class DailyGroupRankOutcome:
    calculation_date: date
    rankings: tuple[GroupRanking, ...]
    prefetch_stats: GroupRankPrefetchStats
    duration_seconds: float
    metadata_repair: object | None

    def to_task_result(
        self,
        policy: DerivedDataExecutionPolicy,
    ) -> dict[str, Any]:
        result = {
            "date": self.calculation_date.isoformat(),
            "groups_ranked": len(self.rankings),
            "top_group": self.rankings[0].industry_group,
            "top_avg_rs": self.rankings[0].avg_rs_rating,
            "calculation_duration_seconds": round(
                self.duration_seconds,
                2,
            ),
            "metadata_repair": self.metadata_repair,
            "timestamp": datetime.now().isoformat(),
        }
        policy.annotate_response(result, include_cache_only=True)
        if policy.response_cache_policy is not None:
            result["prefetch_stats"] = self.prefetch_stats.to_dict()
        return result


def run_daily_group_rankings(
    db: Session,
    request: DailyGroupRankRequest,
    dependencies: DailyGroupRankDependencies,
) -> DailyGroupRankOutcome:
    """Calculate, persist, and publish one daily group-ranking result."""
    started_at = time.perf_counter()
    cache_requirement = _cache_requirement(request, dependencies)
    calculation = dependencies.service.calculate_group_rankings(
        db,
        request.calculation_date,
        market=request.market,
        policy=request.policy,
        cache_requirement=cache_requirement,
    )
    duration_seconds = time.perf_counter() - started_at
    rankings = calculation.rankings
    if not rankings:
        raise NoGroupRankingsCalculated(
            prefetch_stats=calculation.prefetch_stats,
            duration_seconds=duration_seconds,
        )

    logger.info(
        "Successfully ranked %s groups in %.2fs",
        len(rankings),
        duration_seconds,
    )
    for ranking in rankings[:5]:
        logger.info(
            "  #%s: %s (avg RS: %.1f, %s stocks)",
            ranking.rank,
            ranking.industry_group,
            ranking.avg_rs_rating,
            ranking.num_stocks,
        )

    repair_stats = None
    if (
        request.market == "US"
        and dependencies.repair_current_us_metadata is not None
        and (
            request.activity_lifecycle == "bootstrap"
            or request.calculation_date == request.current_date
        )
    ):
        repair_stats = dependencies.repair_current_us_metadata(
            request.calculation_date
        )

    dependencies.bump_epoch(request.market)
    try:
        dependencies.publish_snapshot()
    except Exception as snapshot_error:
        logger.warning(
            "Group rankings snapshot publish failed: %s",
            snapshot_error,
        )

    return DailyGroupRankOutcome(
        calculation_date=request.calculation_date,
        rankings=rankings,
        prefetch_stats=calculation.prefetch_stats,
        duration_seconds=duration_seconds,
        metadata_repair=repair_stats,
    )


def _cache_requirement(
    request: DailyGroupRankRequest,
    dependencies: DailyGroupRankDependencies,
) -> GroupRankCacheRequirement:
    validation_profile = request.policy.validation_profile
    if (
        validation_profile
        is DerivedDataValidationProfile.STRICT_WITH_WARMUP
    ):
        decision = evaluate_same_day_group_rank_warmup(
            dependencies.service.price_cache,
            market=request.market,
        )
        if decision.error:
            raise GroupRankWarmupIncomplete(decision.error)
        return decision.cache_requirement
    if (
        validation_profile
        is DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP
    ):
        logger.info(
            "Bypassing same-day group ranking warmup metadata gate for "
            "in-process static export"
        )
        return GroupRankCacheRequirement.strict()
    return GroupRankCacheRequirement.disabled()

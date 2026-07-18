"""Celery-free daily breadth calculation, validation, and publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import logging
import time
from typing import Any, Callable, Mapping

from sqlalchemy.orm import Session

from .breadth_calculator_service import BreadthCalculatorService
from .breadth_coverage import BreadthCoverageReport
from .cache.price_cache_warmup import evaluate_warmup_metadata
from .derived_data_execution_policy import (
    DerivedDataExecutionPolicy,
    DerivedDataValidationProfile,
)


logger = logging.getLogger(__name__)
CACHE_MISS_TOLERANCE_RATIO = 0.10


class IncompleteDailyBreadth(RuntimeError):
    """Raised when a breadth result does not satisfy its cache policy."""

    def __init__(
        self,
        message: str,
        coverage: BreadthCoverageReport,
    ) -> None:
        self.coverage = coverage
        super().__init__(message)


@dataclass(frozen=True)
class DailyBreadthRequest:
    calculation_date: date
    market: str
    policy: DerivedDataExecutionPolicy


@dataclass(frozen=True)
class DailyBreadthDependencies:
    calculator: BreadthCalculatorService
    publish_snapshot: Callable[[str], object]


@dataclass(frozen=True)
class DailyBreadthOutcome:
    calculation_date: date
    indicators: Mapping[str, Any]
    coverage: BreadthCoverageReport
    duration_seconds: float

    def to_task_result(
        self,
        policy: DerivedDataExecutionPolicy,
    ) -> dict[str, Any]:
        result = {
            "date": self.calculation_date.isoformat(),
            "indicators": dict(self.indicators),
            "total_stocks_scanned": self.coverage.total_stocks_scanned,
            "calculation_duration_seconds": self.duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }
        policy.annotate_response(result, include_cache_only=True)
        if policy.response_cache_policy is not None:
            result["cache_diagnostics"] = self.coverage.to_daily_dict()
        return result


def run_daily_breadth(
    db: Session,
    request: DailyBreadthRequest,
    dependencies: DailyBreadthDependencies,
) -> DailyBreadthOutcome:
    """Calculate, validate, persist, and publish one breadth result."""
    del db  # The market-scoped calculator owns the provided session.
    started_at = time.perf_counter()
    calculation = dependencies.calculator.calculate_daily_breadth(
        calculation_date=request.calculation_date,
        policy=request.policy,
    )
    duration_seconds = time.perf_counter() - started_at
    coverage = calculation.coverage

    completeness_error = _validate_breadth(
        request,
        dependencies,
        coverage,
    )
    if completeness_error:
        raise IncompleteDailyBreadth(completeness_error, coverage)

    metrics = calculation.to_metrics_dict()
    dependencies.calculator.store_daily_breadth(
        request.calculation_date,
        metrics,
        duration_seconds=duration_seconds,
    )
    try:
        dependencies.publish_snapshot(request.market)
    except Exception as snapshot_error:
        logger.warning(
            "Breadth snapshot publish failed: %s",
            snapshot_error,
        )

    logger.info(
        "Breadth calculation completed for %s in %.2fs "
        "(scanned=%s, skipped=%s)",
        request.calculation_date,
        duration_seconds,
        coverage.total_stocks_scanned,
        coverage.skipped_stocks,
    )
    return DailyBreadthOutcome(
        calculation_date=request.calculation_date,
        indicators=dict(calculation.indicators),
        coverage=coverage,
        duration_seconds=duration_seconds,
    )


def _validate_breadth(
    request: DailyBreadthRequest,
    dependencies: DailyBreadthDependencies,
    coverage: BreadthCoverageReport,
) -> str | None:
    validation_profile = request.policy.validation_profile
    if (
        validation_profile
        is DerivedDataValidationProfile.TOLERANT_CACHE_ONLY
    ):
        return _validate_refresh_guarded_breadth(coverage)
    if (
        validation_profile
        is DerivedDataValidationProfile.STRICT_WITH_WARMUP
    ):
        return _validate_same_day_cache_only_breadth(
            dependencies.calculator.price_cache,
            coverage,
            market=request.market,
        )
    if (
        validation_profile
        is DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP
    ):
        logger.info(
            "Bypassing same-day breadth warmup metadata gate for "
            "in-process static export"
        )
        return _validate_strict_cache_only_breadth(coverage)
    return None


def _validate_same_day_cache_only_breadth(
    price_cache,
    coverage: BreadthCoverageReport,
    market: str | None = None,
) -> str | None:
    warmup_meta = (
        price_cache.get_warmup_metadata(market=market)
        if price_cache
        else None
    )
    warmup_readiness = evaluate_warmup_metadata(
        warmup_meta,
        context="same-day breadth run",
    )
    if not warmup_readiness.ready:
        return warmup_readiness.reason
    return _validate_strict_cache_only_breadth(coverage)


def _validate_strict_cache_only_breadth(
    coverage: BreadthCoverageReport,
) -> str | None:
    cache_misses = coverage.cache_miss_stocks
    errors = coverage.error_stocks
    total_attempted = (
        coverage.total_stocks_scanned + coverage.skipped_stocks
    )
    if errors > 0:
        return f"Cache-only breadth run has errors (errors={errors})"
    if total_attempted == 0:
        return "Cache-only breadth run processed no stocks"
    if coverage.total_stocks_scanned == 0:
        return "Cache-only breadth run processed no usable stocks"
    miss_ratio = cache_misses / total_attempted
    if miss_ratio > CACHE_MISS_TOLERANCE_RATIO:
        return (
            "Cache-only breadth run exceeds miss tolerance "
            f"(cache_misses={cache_misses}, total={total_attempted}, "
            f"ratio={miss_ratio:.1%}, "
            f"limit={CACHE_MISS_TOLERANCE_RATIO:.0%})"
        )
    if cache_misses > 0:
        logger.warning(
            "Cache-only breadth run has %d cache misses out of %d stocks "
            "(%.1f%%) -- within tolerance",
            cache_misses,
            total_attempted,
            miss_ratio * 100,
        )
    return None


def _validate_refresh_guarded_breadth(
    coverage: BreadthCoverageReport,
) -> str | None:
    if coverage.error_stocks > 0:
        return (
            "Refresh-guarded breadth run has calculation errors "
            f"(errors={coverage.error_stocks})"
        )
    if coverage.total_stocks_scanned == 0:
        return "Refresh-guarded breadth run processed no usable stocks"
    return None

"""Pure data-quality check functions for feature store runs.

All functions are pure: no I/O, no side effects, fully deterministic.
Each check returns a DQResult value object that captures the verdict,
severity, actual/threshold values, and a human-readable message.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from statistics import mean

from .models import DQSeverity


# ---------------------------------------------------------------------------
# Check Name Constants
# ---------------------------------------------------------------------------

DQ_ROW_COUNT = "row_count"
DQ_NULL_RATE = "null_rate"
DQ_SCORE_DISTRIBUTION = "score_distribution"
DQ_RATING_DISTRIBUTION = "rating_distribution"
DQ_SYMBOL_COVERAGE = "symbol_coverage"


# ---------------------------------------------------------------------------
# DQ Result Value Object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DQResult:
    """Outcome of a single data-quality check."""

    check_name: str
    passed: bool
    severity: DQSeverity
    actual_value: float
    threshold: float  # primary threshold for simple display
    message: str  # human-readable with full context


# ---------------------------------------------------------------------------
# Configuration Value Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DQThresholds:
    """Shared configuration for all DQ checks.

    Single source of truth for default thresholds — replaces scattered
    individual parameters on command objects.
    """

    row_count_threshold: float = 0.9
    null_max_rate: float = 0.05
    score_mean_range: tuple[float, float] = (20.0, 80.0)
    min_distinct_ratings: int = 2
    symbol_coverage_threshold: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.row_count_threshold <= 1.0:
            raise ValueError("row_count_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.null_max_rate <= 1.0:
            raise ValueError("null_max_rate must be in [0.0, 1.0]")
        low, high = self.score_mean_range
        if low >= high:
            raise ValueError(
                f"score_mean_range must be (low, high) with low < high, "
                f"got ({low}, {high})"
            )


@dataclass(frozen=True)
class DQInputs:
    """Bundles all data needed by DQ checks.

    Built either from in-memory rows (build path) or loaded from the
    database (standalone publish path).  Uses tuples for immutability.
    """

    expected_row_count: int
    actual_row_count: int
    null_score_count: int
    total_row_count: int
    scores: tuple[float, ...]
    ratings: tuple[int, ...]
    universe_symbols: tuple[str, ...]
    result_symbols: tuple[str, ...]


# ---------------------------------------------------------------------------
# Individual DQ Checks
# ---------------------------------------------------------------------------


def check_row_count(
    expected: int,
    actual: int,
    threshold: float = 0.9,
    severity: DQSeverity = DQSeverity.CRITICAL,
) -> DQResult:
    """Check that *actual* row count meets the *threshold* fraction of *expected*.

    Passes if ``actual / expected >= threshold``.  When ``expected == 0``
    the check passes (nothing was expected, nothing is missing).
    """
    if expected == 0:
        ratio = 1.0
    else:
        ratio = actual / expected

    passed = ratio >= threshold
    return DQResult(
        check_name=DQ_ROW_COUNT,
        passed=passed,
        severity=severity,
        actual_value=ratio,
        threshold=threshold,
        message=(
            f"Row count ratio {ratio:.2%} (actual={actual}, expected={expected}) "
            f"{'meets' if passed else 'below'} threshold {threshold:.0%}"
        ),
    )


def check_null_rate(
    column_nulls: int,
    total: int,
    max_rate: float = 0.05,
    severity: DQSeverity = DQSeverity.WARNING,
) -> DQResult:
    """Check that the null rate is within the acceptable *max_rate*.

    Passes if ``column_nulls / total <= max_rate``.  When ``total == 0``
    the check passes (no rows to be null).
    """
    if total == 0:
        rate = 0.0
    else:
        rate = column_nulls / total

    passed = rate <= max_rate
    return DQResult(
        check_name=DQ_NULL_RATE,
        passed=passed,
        severity=severity,
        actual_value=rate,
        threshold=max_rate,
        message=(
            f"Null rate {rate:.2%} (nulls={column_nulls}, total={total}) "
            f"{'within' if passed else 'exceeds'} limit {max_rate:.0%}"
        ),
    )


def check_score_distribution(
    scores: Sequence[float],
    expected_mean_range: tuple[float, float],
    severity: DQSeverity = DQSeverity.WARNING,
) -> DQResult:
    """Check that the mean of *scores* falls within *expected_mean_range*.

    Empty *scores* always fail.  The ``threshold`` field is set to the
    midpoint of the range for simple display; the ``message`` includes
    the full range.
    """
    low, high = expected_mean_range
    midpoint = (low + high) / 2.0

    if len(scores) == 0:
        return DQResult(
            check_name=DQ_SCORE_DISTRIBUTION,
            passed=False,
            severity=severity,
            actual_value=0.0,
            threshold=midpoint,
            message=(
                f"Score distribution check failed: no scores provided "
                f"(expected mean in [{low}, {high}])"
            ),
        )

    actual_mean = mean(scores)
    passed = low <= actual_mean <= high
    return DQResult(
        check_name=DQ_SCORE_DISTRIBUTION,
        passed=passed,
        severity=severity,
        actual_value=actual_mean,
        threshold=midpoint,
        message=(
            f"Mean score {actual_mean:.4f} "
            f"{'within' if passed else 'outside'} "
            f"expected range [{low}, {high}]"
        ),
    )


def check_rating_distribution(
    ratings: Sequence[int],
    min_distinct: int = 2,
    severity: DQSeverity = DQSeverity.WARNING,
) -> DQResult:
    """Check that ratings contain at least *min_distinct* distinct values.

    Empty ratings always fail (0 distinct < 2).
    """
    distinct = len(set(ratings))
    passed = distinct >= min_distinct
    return DQResult(
        check_name=DQ_RATING_DISTRIBUTION,
        passed=passed,
        severity=severity,
        actual_value=float(distinct),
        threshold=float(min_distinct),
        message=(
            f"Rating distribution: {distinct} distinct rating(s) "
            f"{'meets' if passed else 'below'} minimum {min_distinct}"
        ),
    )


def check_symbol_coverage(
    universe_symbols: Sequence[str],
    result_symbols: Sequence[str],
    threshold: float = 0.9,
    severity: DQSeverity = DQSeverity.WARNING,
) -> DQResult:
    """Check that result symbols cover at least *threshold* of the universe.

    Coverage = ``len(result ∩ universe) / len(universe)``.
    Empty universe always passes (ratio = 1.0).
    """
    if len(universe_symbols) == 0:
        ratio = 1.0
    else:
        overlap = len(set(result_symbols) & set(universe_symbols))
        ratio = overlap / len(universe_symbols)

    passed = ratio >= threshold
    return DQResult(
        check_name=DQ_SYMBOL_COVERAGE,
        passed=passed,
        severity=severity,
        actual_value=ratio,
        threshold=threshold,
        message=(
            f"Symbol coverage {ratio:.2%} "
            f"(results={len(result_symbols)}, universe={len(universe_symbols)}) "
            f"{'meets' if passed else 'below'} threshold {threshold:.0%}"
        ),
    )


# ---------------------------------------------------------------------------
# DQ Orchestrator
# ---------------------------------------------------------------------------


def run_all_dq_checks(
    inputs: DQInputs,
    thresholds: DQThresholds,
) -> list[DQResult]:
    """Execute all five DQ checks against the provided inputs.

    Pure function — no I/O, fully deterministic.
    """
    return [
        check_row_count(
            inputs.expected_row_count,
            inputs.actual_row_count,
            threshold=thresholds.row_count_threshold,
        ),
        check_null_rate(
            inputs.null_score_count,
            inputs.total_row_count,
            max_rate=thresholds.null_max_rate,
        ),
        check_score_distribution(
            list(inputs.scores),
            thresholds.score_mean_range,
        ),
        check_rating_distribution(
            list(inputs.ratings),
            min_distinct=thresholds.min_distinct_ratings,
        ),
        check_symbol_coverage(
            list(inputs.universe_symbols),
            list(inputs.result_symbols),
            threshold=thresholds.symbol_coverage_threshold,
        ),
    ]


# ---------------------------------------------------------------------------
# Aggregate publishability
# ---------------------------------------------------------------------------


def is_publishable(dq_results: Sequence[DQResult]) -> bool:
    """Return True iff all CRITICAL results passed (warnings don't block)."""
    return all(
        r.passed for r in dq_results if r.severity == DQSeverity.CRITICAL
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DQ_ROW_COUNT",
    "DQ_NULL_RATE",
    "DQ_SCORE_DISTRIBUTION",
    "DQ_RATING_DISTRIBUTION",
    "DQ_SYMBOL_COVERAGE",
    "DQResult",
    "DQThresholds",
    "DQInputs",
    "check_row_count",
    "check_null_rate",
    "check_score_distribution",
    "check_rating_distribution",
    "check_symbol_coverage",
    "run_all_dq_checks",
    "is_publishable",
]

"""Data requirements and incomplete-data policy for Setup Engine.

The policy is deterministic and explicit about insufficiency/degradation so
callers can generate trustworthy explain output without implicit pass/fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


class SetupEngineDataPolicyResult(TypedDict):
    """Result of evaluating minimum data requirements for Setup Engine."""

    status: Literal["ok", "degraded", "insufficient"]
    checks: dict[str, bool]
    failed_reasons: list[str]
    degradation_reasons: list[str]
    requires_weekly_exclude_current: bool


@dataclass(frozen=True)
class SetupEngineDataRequirements:
    """Minimum data requirements for pattern and readiness computation."""

    min_daily_bars: int = 252
    min_weekly_bars: int = 52
    min_benchmark_bars: int = 252
    min_completed_sessions_in_current_week: int = 5
    require_benchmark_for_rs: bool = True
    allow_degraded_without_benchmark: bool = True
    allow_incomplete_current_week: bool = False


def validate_data_requirements(requirements: SetupEngineDataRequirements) -> list[str]:
    """Return validation errors for data requirement values."""
    errors: list[str] = []

    if requirements.min_daily_bars < 60:
        errors.append("min_daily_bars must be >= 60")
    if requirements.min_weekly_bars < 20:
        errors.append("min_weekly_bars must be >= 20")
    if requirements.min_benchmark_bars < 60:
        errors.append("min_benchmark_bars must be >= 60")
    if requirements.min_completed_sessions_in_current_week < 1:
        errors.append("min_completed_sessions_in_current_week must be >= 1")

    return errors


def evaluate_setup_engine_data_policy(
    *,
    daily_bars: int | None,
    weekly_bars: int | None,
    benchmark_bars: int | None,
    current_week_sessions: int | None,
    requirements: SetupEngineDataRequirements = SetupEngineDataRequirements(),
) -> SetupEngineDataPolicyResult:
    """Evaluate sufficiency/degradation decisions for Setup Engine execution.

    Notes:
    - Missing benchmark data can be downgraded to ``degraded`` when
      ``allow_degraded_without_benchmark`` is enabled.
    - Incomplete current week never uses look-ahead assumptions; the caller
      should exclude the current week from weekly features when instructed.
    """
    req_errors = validate_data_requirements(requirements)
    if req_errors:
        raise ValueError("; ".join(req_errors))

    daily = int(daily_bars or 0)
    weekly = int(weekly_bars or 0)
    benchmark = int(benchmark_bars or 0)
    current_week = int(current_week_sessions or 0)

    checks = {
        "min_daily_bars": daily >= requirements.min_daily_bars,
        "min_weekly_bars": weekly >= requirements.min_weekly_bars,
        "benchmark_available": benchmark > 0,
        "min_benchmark_bars": benchmark >= requirements.min_benchmark_bars,
        "current_week_complete": (
            current_week >= requirements.min_completed_sessions_in_current_week
        ),
    }

    failed_reasons: list[str] = []
    degradation_reasons: list[str] = []

    if not checks["min_daily_bars"]:
        failed_reasons.append("insufficient_daily_bars")
    if not checks["min_weekly_bars"]:
        failed_reasons.append("insufficient_weekly_bars")

    if requirements.require_benchmark_for_rs and not checks["min_benchmark_bars"]:
        if requirements.allow_degraded_without_benchmark:
            degradation_reasons.append("missing_or_short_benchmark_history")
        else:
            failed_reasons.append("insufficient_benchmark_bars")

    requires_weekly_exclude_current = False
    if (not checks["current_week_complete"]) and (not requirements.allow_incomplete_current_week):
        degradation_reasons.append("current_week_incomplete_exclude_from_weekly")
        requires_weekly_exclude_current = True

    if failed_reasons:
        status: Literal["ok", "degraded", "insufficient"] = "insufficient"
    elif degradation_reasons:
        status = "degraded"
    else:
        status = "ok"

    return SetupEngineDataPolicyResult(
        status=status,
        checks=checks,
        failed_reasons=failed_reasons,
        degradation_reasons=degradation_reasons,
        requires_weekly_exclude_current=requires_weekly_exclude_current,
    )


def policy_failed_checks(result: SetupEngineDataPolicyResult) -> list[str]:
    """Convert policy result to explain-style failed checks."""
    checks = []
    if result["status"] == "insufficient":
        checks.append("insufficient_data")
    checks.extend(result["failed_reasons"])
    return checks


def policy_invalidation_flags(result: SetupEngineDataPolicyResult) -> list[str]:
    """Convert policy result to explain-style invalidation flags."""
    flags = []
    if result["status"] in {"degraded", "insufficient"}:
        flags.append(f"data_policy:{result['status']}")
    flags.extend(result["failed_reasons"])
    flags.extend(result["degradation_reasons"])
    return flags

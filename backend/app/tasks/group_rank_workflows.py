"""Shared, undecorated helpers for Group ranking task workflows."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
import logging

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.services.market_activity_service import mark_market_activity_failed


logger = logging.getLogger(__name__)


class GroupRankReasonCode:
    INVALID_DATE = "invalid_date"
    WARMUP_INCOMPLETE = "warmup_incomplete"
    MISSING_IBD_MAPPINGS = "missing_ibd_mappings"
    NO_GROUPS_RANKED = "no_groups_ranked"
    UNKNOWN = "unknown"


_ALLOW_SAME_DAY_WARMUP_BYPASS: ContextVar[bool] = ContextVar(
    "allow_same_day_group_rank_warmup_bypass",
    default=False,
)
_PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS: ContextVar[bool] = ContextVar(
    "propagate_in_process_group_rank_transient_errors",
    default=False,
)


@contextmanager
def allow_same_day_group_rank_warmup_bypass():
    token = _ALLOW_SAME_DAY_WARMUP_BYPASS.set(True)
    try:
        yield
    finally:
        _ALLOW_SAME_DAY_WARMUP_BYPASS.reset(token)


def retry_transient_failure(task, task_name: str, exc: Exception) -> None:
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(60 * (2**retries), 600)
    logger.warning(
        "Transient error in %s: %s. Retrying in %ss (attempt %s/2).",
        task_name,
        exc,
        countdown,
        retries + 1,
    )
    raise task.retry(exc=exc, countdown=countdown, max_retries=2)


def mark_market_activity_failed_safely(db, **kwargs) -> None:
    try:
        mark_market_activity_failed(db, **kwargs)
    except Exception:
        logger.warning(
            "Failed to publish market activity failure for group ranking task",
            extra={
                "market": kwargs.get("market"),
                "stage_key": kwargs.get("stage_key"),
                "task_id": kwargs.get("task_id"),
            },
            exc_info=True,
        )


def should_repair_current_us_metadata(
    *,
    calc_date: date,
    today_et: date,
    activity_lifecycle: str,
) -> bool:
    return activity_lifecycle == "bootstrap" or calc_date == today_et


def resolve_active_group_formula(db, *, market: str, group_service) -> str:
    resolved = group_service.market_rs_repository.active_formula(db, market=market)
    if resolved not in {BALANCED_RS_FORMULA_VERSION, LEGACY_RS_FORMULA_VERSION}:
        raise ValueError(f"Unsupported Group RS formula: {resolved}")
    return resolved


def coordinate_group_dates(
    db,
    *,
    dates: list[date],
    market: str,
    formula_version: str,
    coordinator,
) -> dict:
    from app.domain.relative_strength import GroupSnapshotIdentity

    report = coordinator.backfill(
        db,
        identities=tuple(
            GroupSnapshotIdentity(market, calculation_date, formula_version)
            for calculation_date in dates
        ),
        continue_on_error=True,
    )
    return {
        "total_dates": len(report.results),
        "processed": report.processed,
        "skipped": report.existing,
        "errors": report.errors,
    }


def run_daily_in_process(
    task,
    *,
    calculation_date: str | None,
    market: str | None,
    activity_lifecycle: str | None,
):
    from app.tasks.workload_coordination import disable_serialized_market_workload

    kwargs = {"market": market, "activity_lifecycle": activity_lifecycle}
    if calculation_date is not None:
        kwargs["calculation_date"] = calculation_date
    token = _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.set(True)
    try:
        if str(getattr(task, "__module__", "")).startswith("unittest.mock"):
            return task(**kwargs)
        with disable_serialized_market_workload():
            if hasattr(task, "request") and callable(getattr(task, "run", None)):
                return task.run(**kwargs)
            return task(**kwargs)
    finally:
        _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.reset(token)

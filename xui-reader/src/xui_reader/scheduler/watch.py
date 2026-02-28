"""Watch-loop orchestration on top of read and timing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timezone
from enum import IntEnum
from pathlib import Path
import time as time_module
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from xui_reader.config import RuntimeConfig
from xui_reader.errors import SchedulerError
from xui_reader.scheduler.read import MultiSourceReadResult, run_configured_read
from xui_reader.scheduler.timing import calculate_next_run, parse_shutdown_window


RunOnceFn = Callable[[], MultiSourceReadResult]
NowFn = Callable[[], datetime]
SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class WatchCycleResult:
    cycle: int
    started_at: datetime
    next_run_at: datetime | None
    sleep_seconds: float
    emitted_items: int
    succeeded_sources: int
    failed_sources: int
    auth_failed_sources: int = 0


@dataclass(frozen=True)
class WatchRunResult:
    cycles: tuple[WatchCycleResult, ...]


class WatchExitCode(IntEnum):
    """Stable watch exit code matrix for automation wrappers."""

    SUCCESS = 0
    BUDGET_STOP = 4
    AUTH_FAIL = 5
    INTERRUPTED = 130


def run_watch_loop(
    run_once: RunOnceFn,
    *,
    interval_seconds: int,
    jitter_ratio: float = 0.0,
    shutdown_start: time | None = None,
    shutdown_end: time | None = None,
    max_cycles: int,
    now_fn: NowFn | None = None,
    sleep_fn: SleepFn | None = None,
) -> WatchRunResult:
    """Run bounded watch cycles using scheduling utilities."""
    if max_cycles <= 0:
        raise SchedulerError("max_cycles must be > 0.")

    now = now_fn or (lambda: datetime.now(timezone.utc))
    sleeper = sleep_fn or time_module.sleep
    cycles: list[WatchCycleResult] = []

    for index in range(1, max_cycles + 1):
        started_at = _normalize_datetime(now())
        read_result = run_once()

        if index == max_cycles:
            auth_failed_sources = sum(
                1 for outcome in read_result.outcomes if _is_auth_related_error(outcome.error)
            )
            cycles.append(
                WatchCycleResult(
                    cycle=index,
                    started_at=started_at,
                    next_run_at=None,
                    sleep_seconds=0.0,
                    emitted_items=len(read_result.items),
                    succeeded_sources=read_result.succeeded,
                    failed_sources=read_result.failed,
                    auth_failed_sources=auth_failed_sources,
                )
            )
            break

        next_run = calculate_next_run(
            started_at,
            interval_seconds=interval_seconds,
            jitter_ratio=jitter_ratio,
            shutdown_start=shutdown_start,
            shutdown_end=shutdown_end,
        )
        sleep_seconds = max(0.0, (next_run - _normalize_datetime(now())).total_seconds())
        sleeper(sleep_seconds)
        auth_failed_sources = sum(
            1 for outcome in read_result.outcomes if _is_auth_related_error(outcome.error)
        )
        cycles.append(
            WatchCycleResult(
                cycle=index,
                started_at=started_at,
                next_run_at=next_run,
                sleep_seconds=sleep_seconds,
                emitted_items=len(read_result.items),
                succeeded_sources=read_result.succeeded,
                failed_sources=read_result.failed,
                auth_failed_sources=auth_failed_sources,
            )
        )

    return WatchRunResult(cycles=tuple(cycles))


def run_configured_watch(
    config: RuntimeConfig,
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    limit: int = 100,
    interval_seconds: int = 300,
    jitter_ratio: float = 0.0,
    shutdown_window: str | None = None,
    max_cycles: int = 1,
    now_fn: NowFn | None = None,
    sleep_fn: SleepFn | None = None,
) -> WatchRunResult:
    """Run watch mode using configured sources and runtime parameters."""
    effective_now_fn = now_fn
    if effective_now_fn is None:
        try:
            configured_zone = ZoneInfo(config.app.timezone)
        except ZoneInfoNotFoundError as exc:
            raise SchedulerError(
                f"Invalid app.timezone '{config.app.timezone}'. Use an IANA timezone like 'UTC' or 'America/New_York'."
            ) from exc
        effective_now_fn = lambda: datetime.now(configured_zone)  # noqa: E731

    shutdown_start: time | None = None
    shutdown_end: time | None = None
    if shutdown_window:
        shutdown_start, shutdown_end = parse_shutdown_window(shutdown_window)

    return run_watch_loop(
        lambda: run_configured_read(  # noqa: E731
            config,
            profile_name=profile_name,
            config_path=config_path,
            limit=limit,
        ),
        interval_seconds=interval_seconds,
        jitter_ratio=jitter_ratio,
        shutdown_start=shutdown_start,
        shutdown_end=shutdown_end,
        max_cycles=max_cycles,
        now_fn=effective_now_fn,
        sleep_fn=sleep_fn,
    )


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def determine_watch_exit_code(result: WatchRunResult, *, max_cycles: int) -> WatchExitCode:
    """Resolve watch exit code from run outcomes with deterministic precedence."""
    if max_cycles <= 0:
        raise SchedulerError("max_cycles must be > 0.")
    if _is_auth_failed_run(result):
        return WatchExitCode.AUTH_FAIL
    if max_cycles > 1 and len(result.cycles) >= max_cycles:
        return WatchExitCode.BUDGET_STOP
    return WatchExitCode.SUCCESS


def _is_auth_failed_run(result: WatchRunResult) -> bool:
    total_succeeded = sum(cycle.succeeded_sources for cycle in result.cycles)
    total_failed = sum(cycle.failed_sources for cycle in result.cycles)
    total_auth_failed = sum(cycle.auth_failed_sources for cycle in result.cycles)
    return total_succeeded == 0 and total_failed > 0 and total_auth_failed == total_failed


def _is_auth_related_error(error: str | None) -> bool:
    if error is None:
        return False
    lowered = error.lower()
    markers = (
        "missing storage_state",
        "storage_state",
        "run `xui auth login`",
        "not authenticated",
        "login wall",
        "challenge",
        "blocked session state",
    )
    return any(marker in lowered for marker in markers)

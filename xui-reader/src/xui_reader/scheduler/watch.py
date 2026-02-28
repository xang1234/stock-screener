"""Watch-loop orchestration on top of read and timing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timezone
from enum import IntEnum
import json
from pathlib import Path
import time as time_module
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from xui_reader.config import RuntimeConfig
from xui_reader.diagnostics.events import JsonlEventLogger
from xui_reader.errors import SchedulerError
from xui_reader.profiles import profiles_root
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
    seen_items: int = 0
    page_loads: int = 0
    scroll_rounds: int = 0
    succeeded_sources: int = 0
    failed_sources: int = 0
    auth_failed_sources: int = 0


@dataclass(frozen=True)
class WatchRunResult:
    cycles: tuple[WatchCycleResult, ...]
    budget_stop_reason: str | None = None
    interrupted: bool = False
    counters_state_path: str | None = None


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
    max_page_loads: int | None = None,
    max_scroll_rounds: int | None = None,
    now_fn: NowFn | None = None,
    sleep_fn: SleepFn | None = None,
    event_logger: JsonlEventLogger | None = None,
    run_id: str | None = None,
) -> WatchRunResult:
    """Run bounded watch cycles using scheduling utilities."""
    if max_cycles <= 0:
        raise SchedulerError("max_cycles must be > 0.")
    if max_page_loads is not None and max_page_loads <= 0:
        raise SchedulerError("max_page_loads must be > 0 when provided.")
    if max_scroll_rounds is not None and max_scroll_rounds <= 0:
        raise SchedulerError("max_scroll_rounds must be > 0 when provided.")

    now = now_fn or (lambda: datetime.now(timezone.utc))
    sleeper = sleep_fn or time_module.sleep
    cycles: list[WatchCycleResult] = []
    used_page_loads = 0
    used_scroll_rounds = 0
    resolved_run_id = run_id or _new_run_id("watch")

    for index in range(1, max_cycles + 1):
        started_at = _normalize_datetime(now())
        try:
            read_result = run_once()
        except KeyboardInterrupt:
            return WatchRunResult(cycles=tuple(cycles), interrupted=True)

        cycle_page_loads = read_result.total_page_loads
        cycle_scroll_rounds = read_result.total_scroll_rounds
        cycle_seen_items = read_result.total_observed_ids
        used_page_loads += cycle_page_loads
        used_scroll_rounds += cycle_scroll_rounds
        auth_failed_sources = sum(1 for outcome in read_result.outcomes if _is_auth_related_error(outcome.error))
        budget_stop_reason = _budget_stop_reason(
            used_page_loads=used_page_loads,
            used_scroll_rounds=used_scroll_rounds,
            max_page_loads=max_page_loads,
            max_scroll_rounds=max_scroll_rounds,
        )

        is_last_cycle = index == max_cycles
        next_run_at: datetime | None = None
        sleep_seconds = 0.0
        if not is_last_cycle and budget_stop_reason is None:
            next_run_at = calculate_next_run(
                started_at,
                interval_seconds=interval_seconds,
                jitter_ratio=jitter_ratio,
                shutdown_start=shutdown_start,
                shutdown_end=shutdown_end,
            )
            sleep_seconds = max(0.0, (next_run_at - _normalize_datetime(now())).total_seconds())

        cycle_result = WatchCycleResult(
            cycle=index,
            started_at=started_at,
            next_run_at=next_run_at,
            sleep_seconds=sleep_seconds,
            emitted_items=len(read_result.items),
            seen_items=cycle_seen_items,
            page_loads=cycle_page_loads,
            scroll_rounds=cycle_scroll_rounds,
            succeeded_sources=read_result.succeeded,
            failed_sources=read_result.failed,
            auth_failed_sources=auth_failed_sources,
        )
        cycles.append(cycle_result)

        if event_logger is not None:
            event_logger.append(
                "watch_cycle",
                run_id=resolved_run_id,
                payload={
                    "cycle": cycle_result.cycle,
                    "emitted_items": cycle_result.emitted_items,
                    "seen_items": cycle_result.seen_items,
                    "page_loads": cycle_result.page_loads,
                    "scroll_rounds": cycle_result.scroll_rounds,
                    "succeeded_sources": cycle_result.succeeded_sources,
                    "failed_sources": cycle_result.failed_sources,
                    "auth_failed_sources": cycle_result.auth_failed_sources,
                    "source_outcomes": [
                        {
                            "source_id": outcome.source_id,
                            "source_kind": outcome.source_kind,
                            "ok": outcome.ok,
                            "item_count": outcome.item_count,
                            "page_loads": outcome.page_loads,
                            "scroll_rounds": outcome.scroll_rounds,
                            "observed_ids": outcome.observed_ids,
                            "error": outcome.error,
                        }
                        for outcome in read_result.outcomes
                    ],
                },
            )

        if budget_stop_reason is not None:
            return WatchRunResult(cycles=tuple(cycles), budget_stop_reason=budget_stop_reason)
        if is_last_cycle:
            break
        try:
            sleeper(sleep_seconds)
        except KeyboardInterrupt:
            return WatchRunResult(cycles=tuple(cycles), interrupted=True)

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
    max_page_loads: int | None = None,
    max_scroll_rounds: int | None = None,
    now_fn: NowFn | None = None,
    sleep_fn: SleepFn | None = None,
    run_id: str | None = None,
    enable_debug_artifacts: bool = False,
    raw_html_opt_in: bool | None = None,
    event_logger: JsonlEventLogger | None = None,
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

    selected_profile = profile_name or config.app.default_profile
    resolved_run_id = run_id or _new_run_id("watch")

    shutdown_start: time | None = None
    shutdown_end: time | None = None
    if shutdown_window:
        shutdown_start, shutdown_end = parse_shutdown_window(shutdown_window)

    result = run_watch_loop(
        lambda: run_configured_read(  # noqa: E731
            config,
            profile_name=selected_profile,
            config_path=config_path,
            limit=limit,
            enable_debug_artifacts=enable_debug_artifacts,
            raw_html_opt_in=raw_html_opt_in,
            event_logger=event_logger,
        ),
        interval_seconds=interval_seconds,
        jitter_ratio=jitter_ratio,
        shutdown_start=shutdown_start,
        shutdown_end=shutdown_end,
        max_cycles=max_cycles,
        max_page_loads=max_page_loads,
        max_scroll_rounds=max_scroll_rounds,
        now_fn=effective_now_fn,
        sleep_fn=sleep_fn,
        event_logger=event_logger,
        run_id=resolved_run_id,
    )

    state_path = _counters_state_path(selected_profile, config_path)
    _persist_watch_counters(
        state_path=state_path,
        run_id=resolved_run_id,
        result=result,
        max_cycles=max_cycles,
        max_page_loads=max_page_loads,
        max_scroll_rounds=max_scroll_rounds,
    )
    return WatchRunResult(
        cycles=result.cycles,
        budget_stop_reason=result.budget_stop_reason,
        interrupted=result.interrupted,
        counters_state_path=str(state_path),
    )


def determine_watch_exit_code(result: WatchRunResult, *, max_cycles: int) -> WatchExitCode:
    """Resolve watch exit code from run outcomes with deterministic precedence."""
    if max_cycles <= 0:
        raise SchedulerError("max_cycles must be > 0.")
    if result.interrupted:
        return WatchExitCode.INTERRUPTED
    if result.budget_stop_reason is not None:
        return WatchExitCode.BUDGET_STOP
    if _is_auth_failed_run(result):
        return WatchExitCode.AUTH_FAIL
    if max_cycles > 1 and len(result.cycles) >= max_cycles:
        return WatchExitCode.BUDGET_STOP
    return WatchExitCode.SUCCESS


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _budget_stop_reason(
    *,
    used_page_loads: int,
    used_scroll_rounds: int,
    max_page_loads: int | None,
    max_scroll_rounds: int | None,
) -> str | None:
    if max_page_loads is not None and used_page_loads >= max_page_loads:
        return f"page_load_budget_exceeded:{used_page_loads}>={max_page_loads}"
    if max_scroll_rounds is not None and used_scroll_rounds >= max_scroll_rounds:
        return f"scroll_budget_exceeded:{used_scroll_rounds}>={max_scroll_rounds}"
    return None


def _counters_state_path(profile_name: str, config_path: str | Path | None) -> Path:
    return profiles_root(config_path) / profile_name / "logs" / "watch_counters.json"


def _persist_watch_counters(
    *,
    state_path: Path,
    run_id: str,
    result: WatchRunResult,
    max_cycles: int,
    max_page_loads: int | None,
    max_scroll_rounds: int | None,
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v1",
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cycles_completed": len(result.cycles),
        "interrupted": result.interrupted,
        "budget_stop_reason": result.budget_stop_reason,
        "max_cycles": max_cycles,
        "max_page_loads": max_page_loads,
        "max_scroll_rounds": max_scroll_rounds,
        "totals": {
            "page_loads": sum(cycle.page_loads for cycle in result.cycles),
            "scroll_rounds": sum(cycle.scroll_rounds for cycle in result.cycles),
            "seen_items": sum(cycle.seen_items for cycle in result.cycles),
            "emitted_items": sum(cycle.emitted_items for cycle in result.cycles),
            "succeeded_sources": sum(cycle.succeeded_sources for cycle in result.cycles),
            "failed_sources": sum(cycle.failed_sources for cycle in result.cycles),
            "auth_failed_sources": sum(cycle.auth_failed_sources for cycle in result.cycles),
        },
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _new_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"

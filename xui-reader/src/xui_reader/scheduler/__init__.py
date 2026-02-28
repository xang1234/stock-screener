"""Scheduler contracts and helpers."""

from .base import Scheduler, WatchBudget
from .merge import merge_tweet_items
from .read import (
    MultiSourceReadResult,
    SourceReadOutcome,
    collect_source_items,
    run_configured_read,
    run_multi_source_read,
    run_source_smoke_check,
)
from .timing import (
    calculate_next_run,
    clamp_to_shutdown_wakeup,
    jittered_interval_seconds,
    parse_shutdown_window,
)
from .watch import (
    WatchCycleResult,
    WatchExitCode,
    WatchRunResult,
    determine_watch_exit_code,
    run_configured_watch,
    run_watch_loop,
)

__all__ = [
    "Scheduler",
    "WatchBudget",
    "MultiSourceReadResult",
    "SourceReadOutcome",
    "WatchCycleResult",
    "WatchExitCode",
    "WatchRunResult",
    "calculate_next_run",
    "clamp_to_shutdown_wakeup",
    "collect_source_items",
    "determine_watch_exit_code",
    "jittered_interval_seconds",
    "merge_tweet_items",
    "parse_shutdown_window",
    "run_configured_read",
    "run_configured_watch",
    "run_multi_source_read",
    "run_source_smoke_check",
    "run_watch_loop",
]

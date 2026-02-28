"""Watch-loop scheduling behavior."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from xui_reader.config import AppConfig, RuntimeConfig
from xui_reader.errors import SchedulerError
from xui_reader.models import TweetItem
from xui_reader.scheduler.read import MultiSourceReadResult, SourceReadOutcome
from xui_reader.scheduler.watch import (
    WatchCycleResult,
    WatchExitCode,
    WatchRunResult,
    determine_watch_exit_code,
    run_configured_watch,
    run_watch_loop,
)


def test_run_watch_loop_uses_shutdown_clamped_next_run_for_sleep() -> None:
    tz = ZoneInfo("America/New_York")
    first_start = datetime(2026, 3, 1, 23, 55, tzinfo=tz)
    second_start = datetime(2026, 3, 2, 6, 0, tzinfo=tz)
    now_values = iter((first_start, first_start, second_start))
    sleeps: list[float] = []

    def now_fn() -> datetime:
        return next(now_values)

    def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)

    result = run_watch_loop(
        _fake_run_once,
        interval_seconds=600,
        jitter_ratio=0.0,
        shutdown_start=datetime(2026, 3, 1, 0, 0).time(),
        shutdown_end=datetime(2026, 3, 1, 6, 0).time(),
        max_cycles=2,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )

    assert len(result.cycles) == 2
    assert result.cycles[0].next_run_at == second_start
    assert sleeps == [21_900.0]
    assert result.cycles[1].sleep_seconds == 0.0
    assert result.cycles[0].auth_failed_sources == 0


def test_run_watch_loop_rejects_non_positive_cycles() -> None:
    with pytest.raises(SchedulerError, match="max_cycles"):
        run_watch_loop(
            _fake_run_once,
            interval_seconds=60,
            max_cycles=0,
        )


def test_run_configured_watch_uses_configured_timezone_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(app=AppConfig(timezone="America/New_York"))
    observed: dict[str, object] = {}

    def fake_run_watch_loop(
        _run_once: object,
        *,
        interval_seconds: int,
        jitter_ratio: float,
        shutdown_start: object,
        shutdown_end: object,
        max_cycles: int,
        now_fn: object,
        sleep_fn: object,
    ) -> WatchRunResult:
        observed["interval_seconds"] = interval_seconds
        observed["jitter_ratio"] = jitter_ratio
        observed["max_cycles"] = max_cycles
        observed["now_fn"] = now_fn
        return WatchRunResult(cycles=())

    monkeypatch.setattr("xui_reader.scheduler.watch.run_watch_loop", fake_run_watch_loop)
    monkeypatch.setattr(
        "xui_reader.scheduler.watch.run_configured_read",
        lambda *_args, **_kwargs: MultiSourceReadResult(items=(), outcomes=()),
    )

    run_configured_watch(config, interval_seconds=120, max_cycles=1)

    resolved_now = observed["now_fn"]
    assert callable(resolved_now)
    now_value = resolved_now()
    assert isinstance(now_value, datetime)
    assert getattr(now_value.tzinfo, "key", None) == "America/New_York"


def test_run_configured_watch_rejects_invalid_timezone() -> None:
    config = RuntimeConfig(app=AppConfig(timezone="Invalid/Timezone"))

    with pytest.raises(SchedulerError, match="Invalid app.timezone"):
        run_configured_watch(config, max_cycles=1)


def test_determine_watch_exit_code_returns_auth_fail_when_all_failures_are_auth_related() -> None:
    result = WatchRunResult(
        cycles=(
            WatchCycleResult(
                cycle=1,
                started_at=datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC")),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=0,
                succeeded_sources=0,
                failed_sources=2,
                auth_failed_sources=2,
            ),
        )
    )
    assert determine_watch_exit_code(result, max_cycles=1) is WatchExitCode.AUTH_FAIL


def test_determine_watch_exit_code_returns_budget_stop_for_multi_cycle_budget() -> None:
    result = WatchRunResult(
        cycles=(
            WatchCycleResult(
                cycle=1,
                started_at=datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC")),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=3,
                succeeded_sources=1,
                failed_sources=0,
                auth_failed_sources=0,
            ),
            WatchCycleResult(
                cycle=2,
                started_at=datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC")),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=2,
                succeeded_sources=1,
                failed_sources=0,
                auth_failed_sources=0,
            ),
        )
    )
    assert determine_watch_exit_code(result, max_cycles=2) is WatchExitCode.BUDGET_STOP


def test_determine_watch_exit_code_returns_success_for_single_cycle_non_auth_failures() -> None:
    result = WatchRunResult(
        cycles=(
            WatchCycleResult(
                cycle=1,
                started_at=datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC")),
                next_run_at=None,
                sleep_seconds=0.0,
                emitted_items=0,
                succeeded_sources=0,
                failed_sources=1,
                auth_failed_sources=0,
            ),
        )
    )
    assert determine_watch_exit_code(result, max_cycles=1) is WatchExitCode.SUCCESS


def _fake_run_once() -> MultiSourceReadResult:
    item = TweetItem(
        tweet_id="1",
        created_at=datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC")),
        author_handle="@a",
        text="hello",
        source_id="list:1",
    )
    return MultiSourceReadResult(
        items=(item,),
        outcomes=(
            SourceReadOutcome(
                source_id="list:1",
                source_kind="list",
                ok=True,
                item_count=1,
                error=None,
            ),
        ),
    )

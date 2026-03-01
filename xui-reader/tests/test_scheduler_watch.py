"""Watch-loop scheduling behavior."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
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
from xui_reader.testing.time_control import SleepRecorder, assert_local_day_delta, fixed_now, sequenced_now


def test_run_watch_loop_uses_shutdown_clamped_next_run_for_sleep() -> None:
    tz = ZoneInfo("America/New_York")
    first_start = datetime(2026, 3, 1, 23, 55, tzinfo=tz)
    second_start = datetime(2026, 3, 2, 6, 0, tzinfo=tz)
    now_fn = sequenced_now(first_start, first_start, second_start)
    sleep_recorder = SleepRecorder()

    result = run_watch_loop(
        _fake_run_once,
        interval_seconds=600,
        jitter_ratio=0.0,
        shutdown_start=datetime(2026, 3, 1, 0, 0).time(),
        shutdown_end=datetime(2026, 3, 1, 6, 0).time(),
        max_cycles=2,
        now_fn=now_fn,
        sleep_fn=sleep_recorder,
    )

    assert len(result.cycles) == 2
    assert result.cycles[0].next_run_at == second_start
    assert_local_day_delta(first_start, second_start, days=1)
    assert sleep_recorder.calls == [21_900.0]
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
    tmp_path: Path,
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
        max_page_loads: int | None,
        max_scroll_rounds: int | None,
        now_fn: object,
        sleep_fn: object,
        event_logger: object,
        run_id: object,
    ) -> WatchRunResult:
        observed["interval_seconds"] = interval_seconds
        observed["jitter_ratio"] = jitter_ratio
        observed["max_cycles"] = max_cycles
        observed["max_page_loads"] = max_page_loads
        observed["max_scroll_rounds"] = max_scroll_rounds
        observed["now_fn"] = now_fn
        observed["run_id"] = run_id
        return WatchRunResult(cycles=())

    monkeypatch.setattr("xui_reader.scheduler.watch.run_watch_loop", fake_run_watch_loop)
    monkeypatch.setattr(
        "xui_reader.scheduler.watch.run_configured_read",
        lambda *_args, **_kwargs: MultiSourceReadResult(items=(), outcomes=()),
    )

    run_configured_watch(
        config,
        config_path=tmp_path / "config.toml",
        interval_seconds=120,
        max_cycles=1,
    )

    resolved_now = observed["now_fn"]
    assert callable(resolved_now)
    now_value = resolved_now()
    assert isinstance(now_value, datetime)
    assert getattr(now_value.tzinfo, "key", None) == "America/New_York"


def test_run_configured_watch_rejects_invalid_timezone() -> None:
    config = RuntimeConfig(app=AppConfig(timezone="Invalid/Timezone"))

    with pytest.raises(SchedulerError, match="Invalid app.timezone"):
        run_configured_watch(config, max_cycles=1)


def test_run_watch_loop_stops_when_page_load_budget_exceeded() -> None:
    def run_once() -> MultiSourceReadResult:
        return MultiSourceReadResult(
            items=(),
            outcomes=(
                SourceReadOutcome(
                    source_id="list:1",
                    source_kind="list",
                    ok=True,
                    item_count=0,
                    page_loads=2,
                    scroll_rounds=1,
                    observed_ids=0,
                    error=None,
                ),
            ),
        )

    result = run_watch_loop(
        run_once,
        interval_seconds=60,
        max_cycles=5,
        max_page_loads=2,
        now_fn=fixed_now(datetime(2026, 3, 1, 0, 0, tzinfo=ZoneInfo("UTC"))),
        sleep_fn=lambda _seconds: None,
    )
    assert len(result.cycles) == 1
    assert result.budget_stop_reason == "page_load_budget_exceeded:2>=2"
    assert determine_watch_exit_code(result, max_cycles=5) is WatchExitCode.BUDGET_STOP


def test_run_configured_watch_persists_counter_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = RuntimeConfig()
    monkeypatch.setattr(
        "xui_reader.scheduler.watch.run_configured_read",
        lambda *_args, **_kwargs: MultiSourceReadResult(
            items=(),
            outcomes=(
                SourceReadOutcome(
                    source_id="list:1",
                    source_kind="list",
                    ok=True,
                    item_count=0,
                    page_loads=1,
                    scroll_rounds=0,
                    observed_ids=0,
                    error=None,
                ),
            ),
        ),
    )

    result = run_configured_watch(
        config,
        config_path=tmp_path / "config.toml",
        max_cycles=1,
        now_fn=fixed_now(datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC"))),
        sleep_fn=lambda _seconds: None,
    )

    assert result.counters_state_path is not None
    state_path = Path(result.counters_state_path)
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v1"
    assert payload["cycles_completed"] == 1


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


def test_determine_watch_exit_code_returns_success_for_planned_multi_cycle_completion() -> None:
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
    assert determine_watch_exit_code(result, max_cycles=2) is WatchExitCode.SUCCESS


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

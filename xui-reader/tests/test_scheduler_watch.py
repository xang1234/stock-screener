"""Watch-loop scheduling behavior."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from xui_reader.errors import SchedulerError
from xui_reader.models import TweetItem
from xui_reader.scheduler.read import MultiSourceReadResult, SourceReadOutcome
from xui_reader.scheduler.watch import run_watch_loop


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


def test_run_watch_loop_rejects_non_positive_cycles() -> None:
    with pytest.raises(SchedulerError, match="max_cycles"):
        run_watch_loop(
            _fake_run_once,
            interval_seconds=60,
            max_cycles=0,
        )


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

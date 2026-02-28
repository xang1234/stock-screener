"""Scheduler next-run calculator tests."""

from __future__ import annotations

from datetime import datetime, time, timezone
import random
from zoneinfo import ZoneInfo

import pytest

from xui_reader.errors import SchedulerError
from xui_reader.scheduler.timing import (
    calculate_next_run,
    clamp_to_shutdown_wakeup,
    jittered_interval_seconds,
    parse_shutdown_window,
)
from xui_reader.testing.time_control import assert_local_day_delta


def test_jittered_interval_seconds_is_bounded_and_repeatable() -> None:
    rng_a = random.Random(7)
    rng_b = random.Random(7)

    first = jittered_interval_seconds(60, 0.2, rng=rng_a)
    second = jittered_interval_seconds(60, 0.2, rng=rng_b)

    assert first == second
    assert 48 <= first <= 72


def test_calculate_next_run_applies_cross_midnight_shutdown_wakeup() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 3, 1, 23, 55, tzinfo=tz)

    result = calculate_next_run(
        now,
        interval_seconds=600,
        jitter_ratio=0.0,
        shutdown_start=time(0, 0),
        shutdown_end=time(6, 0),
    )

    assert result == datetime(2026, 3, 2, 6, 0, tzinfo=tz)
    assert_local_day_delta(now, result, days=1)


def test_calculate_next_run_applies_same_day_shutdown_window() -> None:
    tz = ZoneInfo("Asia/Singapore")
    now = datetime(2026, 3, 1, 11, 50, tzinfo=tz)

    result = calculate_next_run(
        now,
        interval_seconds=900,
        jitter_ratio=0.0,
        shutdown_start=time(12, 0),
        shutdown_end=time(13, 0),
    )

    assert result == datetime(2026, 3, 1, 13, 0, tzinfo=tz)
    assert_local_day_delta(now, result, days=0)


def test_clamp_to_shutdown_wakeup_returns_same_time_outside_window() -> None:
    tz = ZoneInfo("America/New_York")
    candidate = datetime(2026, 3, 1, 21, 30, tzinfo=tz)

    result = clamp_to_shutdown_wakeup(
        candidate,
        shutdown_start=time(22, 0),
        shutdown_end=time(2, 0),
    )

    assert result == candidate


def test_clamp_to_shutdown_wakeup_returns_window_end_inside_cross_midnight_window() -> None:
    tz = ZoneInfo("America/New_York")
    candidate = datetime(2026, 3, 2, 1, 30, tzinfo=tz)

    result = clamp_to_shutdown_wakeup(
        candidate,
        shutdown_start=time(22, 0),
        shutdown_end=time(2, 0),
    )

    assert result == datetime(2026, 3, 2, 2, 0, tzinfo=tz)


def test_calculate_next_run_requires_complete_shutdown_window() -> None:
    with pytest.raises(SchedulerError, match="shutdown_start and shutdown_end"):
        calculate_next_run(
            datetime(2026, 3, 1, 12, 0, tzinfo=ZoneInfo("UTC")),
            interval_seconds=60,
            shutdown_start=time(1, 0),
            shutdown_end=None,
        )


def test_jittered_interval_seconds_rejects_invalid_arguments() -> None:
    with pytest.raises(SchedulerError, match="interval_seconds"):
        jittered_interval_seconds(0)
    with pytest.raises(SchedulerError, match="jitter_ratio"):
        jittered_interval_seconds(60, jitter_ratio=1.1)


def test_calculate_next_run_handles_dst_gap_without_false_shutdown_clamp() -> None:
    tz = ZoneInfo("America/New_York")
    now = datetime(2026, 3, 8, 1, 55, tzinfo=tz)

    result = calculate_next_run(
        now,
        interval_seconds=600,
        jitter_ratio=0.0,
        shutdown_start=time(0, 0),
        shutdown_end=time(3, 0),
    )

    # 01:55 + 10m crosses spring-forward gap and should normalize to 03:05.
    assert result == datetime(2026, 3, 8, 3, 5, tzinfo=tz)


def test_clamp_to_shutdown_wakeup_rejects_aware_shutdown_times() -> None:
    with pytest.raises(SchedulerError, match="naive local time"):
        clamp_to_shutdown_wakeup(
            datetime(2026, 3, 1, 12, 0, tzinfo=ZoneInfo("UTC")),
            shutdown_start=time(1, 0, tzinfo=timezone.utc),
            shutdown_end=time(2, 0, tzinfo=timezone.utc),
        )


def test_parse_shutdown_window_parses_cross_midnight_window() -> None:
    start, end = parse_shutdown_window("22:30-05:15")
    assert start == time(22, 30)
    assert end == time(5, 15)


def test_parse_shutdown_window_rejects_invalid_strings() -> None:
    with pytest.raises(SchedulerError, match="Invalid shutdown window"):
        parse_shutdown_window("bad-value")
    with pytest.raises(SchedulerError, match="Hour"):
        parse_shutdown_window("24:00-06:00")

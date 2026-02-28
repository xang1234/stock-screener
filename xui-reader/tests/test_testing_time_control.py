"""Deterministic test utility behavior."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from xui_reader.testing.time_control import SleepRecorder, assert_local_day_delta, fixed_now, sequenced_now


def test_fixed_now_always_returns_the_same_aware_value() -> None:
    instant = datetime(2026, 3, 1, 12, 0, tzinfo=ZoneInfo("UTC"))
    now_fn = fixed_now(instant)
    assert now_fn() == instant
    assert now_fn() == instant


def test_sequenced_now_returns_values_in_order_then_fails() -> None:
    tz = ZoneInfo("Asia/Singapore")
    first = datetime(2026, 3, 1, 23, 59, tzinfo=tz)
    second = datetime(2026, 3, 2, 0, 0, tzinfo=tz)
    now_fn = sequenced_now(first, second)

    assert now_fn() == first
    assert now_fn() == second
    with pytest.raises(AssertionError, match="exhausted"):
        now_fn()


def test_sleep_recorder_tracks_seconds() -> None:
    recorder = SleepRecorder()
    recorder(1)
    recorder(2.5)
    assert recorder.calls == [1.0, 2.5]


def test_assert_local_day_delta_validates_calendar_rollover() -> None:
    tz = ZoneInfo("America/New_York")
    start = datetime(2026, 3, 1, 23, 0, tzinfo=tz)
    end = datetime(2026, 3, 2, 1, 0, tzinfo=tz)
    assert_local_day_delta(start, end, days=1)
    with pytest.raises(AssertionError, match="Expected local day delta"):
        assert_local_day_delta(start, end, days=0)

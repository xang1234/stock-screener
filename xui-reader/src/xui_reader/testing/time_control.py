"""Deterministic clock helpers reused by scheduler and budget tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

NowFn = Callable[[], datetime]
SleepFn = Callable[[float], None]


def fixed_now(moment: datetime) -> NowFn:
    """Return a clock function that always yields the same aware datetime."""
    resolved = _require_aware(moment)

    def _now() -> datetime:
        return resolved

    return _now


def sequenced_now(*moments: datetime) -> NowFn:
    """Return a clock function that yields values in order, then fails when exhausted."""
    if not moments:
        raise ValueError("sequenced_now requires at least one datetime.")
    resolved = tuple(_require_aware(moment) for moment in moments)
    index = 0

    def _now() -> datetime:
        nonlocal index
        if index >= len(resolved):
            raise AssertionError(
                f"sequenced_now exhausted after {len(resolved)} call(s); test made an unexpected time lookup."
            )
        value = resolved[index]
        index += 1
        return value

    return _now


@dataclass
class SleepRecorder:
    """Sleep function that records requested sleeps for deterministic assertions."""

    calls: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


def assert_local_day_delta(start: datetime, end: datetime, *, days: int) -> None:
    """Assert local calendar-day movement in start's timezone."""
    if days < 0:
        raise ValueError("days must be >= 0.")
    start_aware = _require_aware(start)
    end_aware = _require_aware(end)
    end_local = end_aware.astimezone(start_aware.tzinfo)
    observed = (end_local.date() - start_aware.date()).days
    if observed != days:
        raise AssertionError(
            f"Expected local day delta {days}, observed {observed} "
            f"(start={start_aware.isoformat()}, end={end_local.isoformat()})."
        )


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime value must include tzinfo for deterministic local-time assertions.")
    return value

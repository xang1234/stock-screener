"""Scheduler time calculation helpers."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import random

from xui_reader.errors import SchedulerError


def jittered_interval_seconds(
    interval_seconds: int,
    jitter_ratio: float = 0.0,
    *,
    rng: random.Random | None = None,
) -> int:
    """Return an interval with bounded +/- jitter applied."""
    if interval_seconds <= 0:
        raise SchedulerError("interval_seconds must be > 0.")
    if jitter_ratio < 0 or jitter_ratio > 1:
        raise SchedulerError("jitter_ratio must be between 0 and 1.")

    spread = int(round(interval_seconds * jitter_ratio))
    if spread <= 0:
        return interval_seconds

    chooser = rng if rng is not None else random
    offset = chooser.randint(-spread, spread)
    return max(1, interval_seconds + offset)


def calculate_next_run(
    now: datetime,
    *,
    interval_seconds: int,
    jitter_ratio: float = 0.0,
    shutdown_start: time | None = None,
    shutdown_end: time | None = None,
    rng: random.Random | None = None,
) -> datetime:
    """Compute the next run time with jitter and optional shutdown wake-up clamp."""
    normalized_now = _normalize_datetime(now)
    delay_seconds = jittered_interval_seconds(interval_seconds, jitter_ratio, rng=rng)
    candidate = normalized_now + timedelta(seconds=delay_seconds)
    if shutdown_start is None and shutdown_end is None:
        return candidate
    if shutdown_start is None or shutdown_end is None:
        raise SchedulerError("shutdown_start and shutdown_end must both be provided.")
    return clamp_to_shutdown_wakeup(candidate, shutdown_start=shutdown_start, shutdown_end=shutdown_end)


def clamp_to_shutdown_wakeup(
    candidate: datetime,
    *,
    shutdown_start: time,
    shutdown_end: time,
) -> datetime:
    """Return shutdown end boundary when candidate lands inside shutdown window."""
    normalized = _normalize_datetime(candidate)
    if _is_within_shutdown(normalized.timetz().replace(tzinfo=None), shutdown_start, shutdown_end):
        return _shutdown_window_end(normalized, shutdown_start=shutdown_start, shutdown_end=shutdown_end)
    return normalized


def _is_within_shutdown(value: time, shutdown_start: time, shutdown_end: time) -> bool:
    if shutdown_start == shutdown_end:
        return False
    if shutdown_start < shutdown_end:
        return shutdown_start <= value < shutdown_end
    return value >= shutdown_start or value < shutdown_end


def _shutdown_window_end(candidate: datetime, *, shutdown_start: time, shutdown_end: time) -> datetime:
    current = _normalize_datetime(candidate)
    local_value = current.timetz().replace(tzinfo=None)

    if shutdown_start < shutdown_end:
        return current.replace(
            hour=shutdown_end.hour,
            minute=shutdown_end.minute,
            second=shutdown_end.second,
            microsecond=shutdown_end.microsecond,
        )

    if local_value < shutdown_end:
        return current.replace(
            hour=shutdown_end.hour,
            minute=shutdown_end.minute,
            second=shutdown_end.second,
            microsecond=shutdown_end.microsecond,
        )

    next_day = current + timedelta(days=1)
    return next_day.replace(
        hour=shutdown_end.hour,
        minute=shutdown_end.minute,
        second=shutdown_end.second,
        microsecond=shutdown_end.microsecond,
    )


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(value.tzinfo)

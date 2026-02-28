"""Scheduler time calculation helpers."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import random
import re

from xui_reader.errors import SchedulerError

_SHUTDOWN_WINDOW_RE = re.compile(
    r"^\s*(?P<start_hour>\d{1,2}):(?P<start_minute>\d{2})\s*-\s*(?P<end_hour>\d{1,2}):(?P<end_minute>\d{2})\s*$"
)


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
    _validate_shutdown_time(shutdown_start, name="shutdown_start")
    _validate_shutdown_time(shutdown_end, name="shutdown_end")
    return clamp_to_shutdown_wakeup(candidate, shutdown_start=shutdown_start, shutdown_end=shutdown_end)


def clamp_to_shutdown_wakeup(
    candidate: datetime,
    *,
    shutdown_start: time,
    shutdown_end: time,
) -> datetime:
    """Return shutdown end boundary when candidate lands inside shutdown window."""
    normalized = _normalize_datetime(candidate)
    _validate_shutdown_time(shutdown_start, name="shutdown_start")
    _validate_shutdown_time(shutdown_end, name="shutdown_end")
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
    # Canonicalize around DST boundaries by round-tripping through UTC.
    zone = value.tzinfo
    return value.astimezone(timezone.utc).astimezone(zone)


def _validate_shutdown_time(value: time, *, name: str) -> None:
    if value.tzinfo is not None:
        raise SchedulerError(f"{name} must be a naive local time without timezone info.")


def parse_shutdown_window(raw_window: str) -> tuple[time, time]:
    """Parse shutdown window in HH:MM-HH:MM local-time format."""
    raw = raw_window.strip()
    match = _SHUTDOWN_WINDOW_RE.fullmatch(raw)
    if match is None:
        raise SchedulerError(
            f"Invalid shutdown window '{raw_window}'. Expected format HH:MM-HH:MM (24-hour clock)."
        )
    start = time(
        hour=_parse_hour(match.group("start_hour"), raw_window),
        minute=_parse_minute(match.group("start_minute"), raw_window),
    )
    end = time(
        hour=_parse_hour(match.group("end_hour"), raw_window),
        minute=_parse_minute(match.group("end_minute"), raw_window),
    )
    return start, end


def _parse_hour(raw: str, raw_window: str) -> int:
    value = int(raw)
    if value < 0 or value > 23:
        raise SchedulerError(
            f"Invalid shutdown window '{raw_window}'. Hour '{raw}' must be between 00 and 23."
        )
    return value


def _parse_minute(raw: str, raw_window: str) -> int:
    value = int(raw)
    if value < 0 or value > 59:
        raise SchedulerError(
            f"Invalid shutdown window '{raw_window}'. Minute '{raw}' must be between 00 and 59."
        )
    return value

"""Pure weekly bucketing shared by RRG calculations and persisted state."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta


def rrg_week_start(value: date) -> date:
    """Return the UTC Sunday-origin week key used by the RRG frontend."""
    return value - timedelta(days=(value.weekday() + 1) % 7)


def bucket_rrg_weekly(
    daily: Sequence[tuple[date, float]],
) -> list[tuple[date, float]]:
    """Keep the latest value in each RRG week, ordered by week key."""
    latest: dict[date, tuple[date, float]] = {}
    for observation_date, value in daily:
        week_start = rrg_week_start(observation_date)
        previous = latest.get(week_start)
        if previous is None or observation_date >= previous[0]:
            latest[week_start] = (observation_date, value)
    return [(week_start, latest[week_start][1]) for week_start in sorted(latest)]


__all__ = ["bucket_rrg_weekly", "rrg_week_start"]

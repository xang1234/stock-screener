"""Shared date resolution for daily task wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


class MarketCalendarLike(Protocol):
    def market_now(self, market: str) -> datetime:
        ...


@dataclass(frozen=True)
class ResolvedTaskDate:
    target_date: date
    was_explicit: bool

    def nested_daily_kwargs(self) -> dict[str, str]:
        return {"calculation_date": self.target_date.isoformat()}


def resolve_task_target_date(
    calculation_date: str | None,
    *,
    market: str,
    calendar_service: MarketCalendarLike,
) -> ResolvedTaskDate:
    """Resolve a task target date without changing nested task semantics."""
    if calculation_date:
        return ResolvedTaskDate(
            target_date=datetime.strptime(calculation_date, "%Y-%m-%d").date(),
            was_explicit=True,
        )
    return ResolvedTaskDate(
        target_date=calendar_service.market_now(market).date(),
        was_explicit=False,
    )

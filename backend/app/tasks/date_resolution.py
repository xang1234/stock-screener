"""Shared date resolution for daily task wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


class MarketCalendarLike(Protocol):
    def last_completed_trading_day(self, market: str) -> date:
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
    """Resolve implicit daily work to the latest completed market session."""
    if calculation_date:
        return ResolvedTaskDate(
            target_date=date.fromisoformat(calculation_date),
            was_explicit=True,
        )
    return ResolvedTaskDate(
        target_date=calendar_service.last_completed_trading_day(market),
        was_explicit=False,
    )

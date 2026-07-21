from __future__ import annotations

from datetime import datetime

from app.tasks.date_resolution import resolve_task_target_date


class FakeCalendar:
    def __init__(self, now: datetime, last_completed: str = "2026-03-16"):
        self.now = now
        self.market_calls: list[str] = []
        self.last_completed = last_completed
        self.last_completed_calls: list[str] = []

    def market_now(self, market: str) -> datetime:
        self.market_calls.append(market)
        return self.now

    def last_completed_trading_day(self, market: str):
        self.last_completed_calls.append(market)
        return datetime.strptime(self.last_completed, "%Y-%m-%d").date()


def test_resolve_task_target_date_uses_explicit_date_without_calendar_lookup():
    calendar = FakeCalendar(datetime(2026, 3, 17, 12, 0, 0))

    resolved = resolve_task_target_date(
        "2026-03-16",
        market="HK",
        calendar_service=calendar,
    )

    assert resolved.target_date.isoformat() == "2026-03-16"
    assert resolved.was_explicit is True
    assert resolved.nested_daily_kwargs() == {"calculation_date": "2026-03-16"}
    assert calendar.market_calls == []
    assert calendar.last_completed_calls == []


def test_resolve_task_target_date_uses_last_completed_session_for_scheduled_runs():
    calendar = FakeCalendar(
        datetime(2026, 3, 17, 9, 30, 0),
        last_completed="2026-03-16",
    )

    resolved = resolve_task_target_date(
        None,
        market="JP",
        calendar_service=calendar,
    )

    assert resolved.target_date.isoformat() == "2026-03-16"
    assert resolved.was_explicit is False
    assert resolved.nested_daily_kwargs() == {"calculation_date": "2026-03-16"}
    assert calendar.market_calls == []
    assert calendar.last_completed_calls == ["JP"]

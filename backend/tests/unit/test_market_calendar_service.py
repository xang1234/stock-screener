from datetime import datetime

import pandas as pd

from app.services.market_calendar_service import MarketCalendarService


class _FakeCalendar:
    def __init__(self):
        self.sessions = [
            pd.Timestamp("2026-04-09"),
            pd.Timestamp("2026-04-10"),
        ]
        self.schedule = pd.DataFrame(
            {
                "market_close": [
                    pd.Timestamp("2026-04-09 08:00:00+00:00"),
                    pd.Timestamp("2026-04-10 08:00:00+00:00"),
                ],
            },
            index=self.sessions,
        )

    def is_session(self, session: pd.Timestamp) -> bool:
        return any(s.date() == session.date() for s in self.sessions)

    def previous_session(self, session: pd.Timestamp) -> pd.Timestamp:
        previous = [s for s in self.sessions if s.date() < session.date()]
        return previous[-1]

    def is_open_on_minute(self, ts: pd.Timestamp, ignore_breaks: bool = False) -> bool:
        # Keep this deterministic: only one minute is considered open.
        return ts == pd.Timestamp("2026-04-10 01:30:00+00:00")


def test_market_calendar_service_uses_canonical_calendar_ids():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())

    assert service.calendar_id("US") == "XNYS"
    assert service.calendar_id("HK") == "XHKG"
    assert service.calendar_id("JP") == "XTKS"
    assert service.calendar_id("TW") == "XTAI"


def test_last_completed_trading_day_before_close_returns_previous_session():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    now_hkt = datetime.fromisoformat("2026-04-10T15:30:00+08:00")

    expected = service.last_completed_trading_day("HK", now=now_hkt)

    assert expected == pd.Timestamp("2026-04-09").date()


def test_last_completed_trading_day_after_close_returns_current_session():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    now_hkt = datetime.fromisoformat("2026-04-10T17:30:00+08:00")

    expected = service.last_completed_trading_day("HK", now=now_hkt)

    assert expected == pd.Timestamp("2026-04-10").date()


def test_is_market_open_uses_calendar_open_minute():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    open_minute_hkt = datetime.fromisoformat("2026-04-10T09:30:00+08:00")
    closed_minute_hkt = datetime.fromisoformat("2026-04-10T09:31:00+08:00")

    assert service.is_market_open("HK", now=open_minute_hkt) is True
    assert service.is_market_open("HK", now=closed_minute_hkt) is False

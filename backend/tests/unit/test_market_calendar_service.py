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
                "close": [
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


class _FallbackCalendar:
    def __init__(self):
        self.sessions = [pd.Timestamp("2026-04-10")]
        self.schedule = pd.DataFrame(
            {
                "market_open": [pd.Timestamp("2026-04-10 03:45:00+00:00")],
                "market_close": [pd.Timestamp("2026-04-10 10:00:00+00:00")],
            },
            index=self.sessions,
        )

    def is_session(self, session: pd.Timestamp) -> bool:
        return any(s.date() == session.date() for s in self.sessions)


class _ProviderCalendar:
    pass


def test_market_calendar_service_uses_canonical_calendar_ids():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())

    assert service.calendar_id("US") == "XNYS"
    assert service.calendar_id("HK") == "XHKG"
    assert service.calendar_id("IN") == "XNSE"
    assert service.calendar_id("JP") == "XTKS"
    assert service.calendar_id("KR") == "XKRX"
    assert service.calendar_id("TW") == "XTAI"


def test_last_completed_trading_day_before_close_returns_previous_session():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    now_hkt = datetime.fromisoformat("2026-04-10T15:30:00+08:00")

    expected = service.last_completed_trading_day("HK", now=now_hkt)

    assert expected == pd.Timestamp("2026-04-09").date()


def test_last_completed_trading_day_after_close_returns_current_session():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    now_hkt = datetime.fromisoformat("2026-04-10T16:30:00+08:00")

    expected = service.last_completed_trading_day("HK", now=now_hkt)

    assert expected == pd.Timestamp("2026-04-10").date()


def test_last_completed_trading_day_before_post_close_buffer_returns_previous_session():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    now_hkt = datetime.fromisoformat("2026-04-10T16:29:00+08:00")

    expected = service.last_completed_trading_day("HK", now=now_hkt)

    assert expected == pd.Timestamp("2026-04-09").date()


def test_is_market_open_uses_calendar_open_minute():
    service = MarketCalendarService(calendar_provider=lambda _: _FakeCalendar())
    open_minute_hkt = datetime.fromisoformat("2026-04-10T09:30:00+08:00")
    closed_minute_hkt = datetime.fromisoformat("2026-04-10T09:31:00+08:00")

    assert service.is_market_open("HK", now=open_minute_hkt) is True
    assert service.is_market_open("HK", now=closed_minute_hkt) is False


def test_is_market_open_schedule_fallback_treats_close_minute_as_closed():
    service = MarketCalendarService(calendar_provider=lambda _: _FallbackCalendar())
    pre_close_ist = datetime.fromisoformat("2026-04-10T15:29:00+05:30")
    close_minute_ist = datetime.fromisoformat("2026-04-10T15:30:00+05:30")

    assert service.is_market_open("IN", now=pre_close_ist) is True
    assert service.is_market_open("IN", now=close_minute_ist) is False


def test_india_pmc_lookup_uses_provider_specific_calendar_id():
    calls = []
    service = MarketCalendarService()
    service._pmc_provider = lambda calendar_id: calls.append(calendar_id) or _ProviderCalendar()
    service._xcals_provider = lambda calendar_id: calls.append(calendar_id) or _ProviderCalendar()

    service._get_calendar("IN")

    assert calls == ["NSE"]

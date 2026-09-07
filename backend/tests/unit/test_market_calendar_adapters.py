from datetime import date

import pandas as pd

from app.services import market_calendar_adapters as module
from app.services.market_calendar_adapters import RawMarketCalendarAdapter


class _FakeRedis:
    def __init__(self):
        self.values = {}
        self.setex_calls = []

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.setex_calls.append((key, ttl, value))
        self.values[key] = value


class _ScheduleCalendar:
    def __init__(self):
        self.calls = []

    def schedule(self, *, start_date: pd.Timestamp, end_date: pd.Timestamp):
        start = start_date.date()
        end = end_date.date()
        self.calls.append((start, end))
        sessions = [
            pd.Timestamp(session_day)
            for session_day in (date(2026, 1, 2), date(2026, 1, 5))
            if start <= session_day <= end
        ]
        return pd.DataFrame(index=sessions)


class _CallableLastSessionCalendar:
    def __init__(self):
        self.sessions = [pd.Timestamp("2026-01-07")]

    def last_session(self):
        return None


class _CallableFirstSessionCalendar:
    def __init__(self):
        self.sessions = [pd.Timestamp("2026-01-07")]

    def first_session(self):
        return None


def test_sessions_in_range_uses_redis_cache(monkeypatch):
    redis = _FakeRedis()
    calendar = _ScheduleCalendar()
    monkeypatch.setattr(module, "get_redis_client", lambda: redis)
    monkeypatch.setattr(module, "_session_range_cache_retry_after", 0.0)
    adapter = RawMarketCalendarAdapter(
        calendar,
        cache_namespace="exchange_calendars:XSES:XSES",
    )

    first = adapter.sessions_in_range(date(2026, 1, 1), date(2026, 1, 7))
    second = adapter.sessions_in_range(date(2026, 1, 1), date(2026, 1, 7))

    assert first == (date(2026, 1, 2), date(2026, 1, 5))
    assert second == first
    assert calendar.calls == [(date(2026, 1, 1), date(2026, 1, 7))]
    assert redis.setex_calls == [
        (
            "calendar:sessions:v1:exchange_calendars:XSES:XSES:2026-01-01:2026-01-07",
            module.SESSION_RANGE_CACHE_TTL_SECONDS,
            '["2026-01-02","2026-01-05"]',
        )
    ]


def test_public_local_calendar_opt_out_never_touches_redis(monkeypatch):
    from app.services.market_calendar_service import MarketCalendarService
    def forbidden(*args, **kwargs):
        raise AssertionError("shared cache accessed")
    monkeypatch.setattr(module, "get_redis_client", forbidden)
    monkeypatch.setattr(RawMarketCalendarAdapter, "_read_session_range_cache", forbidden)
    monkeypatch.setattr(RawMarketCalendarAdapter, "_write_session_range_cache", forbidden)
    adapter = RawMarketCalendarAdapter(_ScheduleCalendar(), use_shared_cache=False)
    assert adapter.sessions_in_range(date(2026, 1, 1), date(2026, 1, 7)) == (date(2026, 1, 2), date(2026, 1, 5))
    calendar = MarketCalendarService(use_shared_cache=False)
    assert calendar.trading_days("US", date(2026, 1, 2), date(2026, 1, 5)) == [date(2026, 1, 2), date(2026, 1, 5)]


def test_last_session_date_falls_back_when_callable_returns_none():
    adapter = RawMarketCalendarAdapter(_CallableLastSessionCalendar())

    assert adapter.last_session_date() == date(2026, 1, 7)


def test_first_session_date_falls_back_when_callable_returns_none():
    adapter = RawMarketCalendarAdapter(_CallableFirstSessionCalendar())

    assert adapter.first_session_date() == date(2026, 1, 7)

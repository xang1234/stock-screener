"""Market calendar abstraction for US/HK/JP/TW session-aware decisions."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import exchange_calendars as xcals  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    xcals = None  # type: ignore


class MarketCalendarService:
    """Unified market calendar contract backed by exchange_calendars."""

    CALENDAR_ID_BY_MARKET: dict[str, str] = {
        "US": "XNYS",
        "HK": "XHKG",
        "JP": "XTKS",
        "TW": "XTAI",
    }
    TIMEZONE_BY_MARKET: dict[str, str] = {
        "US": "America/New_York",
        "HK": "Asia/Hong_Kong",
        "JP": "Asia/Tokyo",
        "TW": "Asia/Taipei",
    }

    def __init__(self, calendar_provider=None):
        if calendar_provider is not None:
            self._calendar_provider = calendar_provider
        else:
            self._calendar_provider = xcals.get_calendar if xcals is not None else None
        self._calendar_cache: dict[str, object] = {}

    def normalize_market(self, market: str | None) -> str:
        normalized = (market or "US").strip().upper()
        if normalized not in self.CALENDAR_ID_BY_MARKET:
            raise ValueError(f"Unsupported market for calendar service: {market}")
        return normalized

    def market_timezone(self, market: str) -> ZoneInfo:
        normalized = self.normalize_market(market)
        return ZoneInfo(self.TIMEZONE_BY_MARKET[normalized])

    def market_now(self, market: str, now: datetime | None = None) -> datetime:
        tz = self.market_timezone(market)
        if now is None:
            return datetime.now(tz)
        if now.tzinfo is None:
            return now.replace(tzinfo=tz)
        return now.astimezone(tz)

    def calendar_id(self, market: str) -> str:
        normalized = self.normalize_market(market)
        return self.CALENDAR_ID_BY_MARKET[normalized]

    def _get_calendar(self, market: str):
        normalized = self.normalize_market(market)
        calendar_id = self.CALENDAR_ID_BY_MARKET[normalized]
        if self._calendar_provider is None:
            raise RuntimeError("exchange_calendars is required for MarketCalendarService")
        if calendar_id not in self._calendar_cache:
            self._calendar_cache[calendar_id] = self._calendar_provider(calendar_id)
        return self._calendar_cache[calendar_id]

    def is_trading_day(self, market: str, day: date | None = None) -> bool:
        calendar = self._get_calendar(market)
        candidate_day = day or self.market_now(market).date()
        return calendar.is_session(pd.Timestamp(candidate_day))

    def is_market_open(self, market: str, now: datetime | None = None) -> bool:
        calendar = self._get_calendar(market)
        market_now = self.market_now(market, now=now)
        current_session = pd.Timestamp(market_now.date())
        if not calendar.is_session(current_session):
            return False
        minute_utc = pd.Timestamp(market_now).tz_convert("UTC").floor("min")
        return bool(calendar.is_open_on_minute(minute_utc, ignore_breaks=False))

    def last_completed_trading_day(self, market: str, now: datetime | None = None) -> date:
        """Return the latest trading day that should already have end-of-day bars."""
        normalized = self.normalize_market(market)
        calendar = self._get_calendar(normalized)
        market_now = self.market_now(normalized, now=now)
        current_session = pd.Timestamp(market_now.date())

        if not calendar.is_session(current_session):
            return calendar.date_to_session(current_session, direction="previous").date()

        schedule = calendar.schedule.loc[current_session:current_session]
        if schedule.empty:
            return calendar.date_to_session(current_session, direction="previous").date()

        market_close = schedule.iloc[0]["close"] if "close" in schedule.columns else schedule.iloc[0]["market_close"]
        if market_close.tzinfo is None:
            market_close = market_close.tz_localize("UTC")
        close_with_buffer = (
            market_close.tz_convert(self.market_timezone(normalized)).to_pydatetime()
            + timedelta(hours=1)
        )
        if market_now >= close_with_buffer:
            return current_session.date()
        return calendar.previous_session(current_session).date()

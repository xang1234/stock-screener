"""Market calendar abstraction for US/HK/IN/JP/TW session-aware decisions."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import exchange_calendars as xcals  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    xcals = None  # type: ignore

try:
    import pandas_market_calendars as pmc  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    pmc = None  # type: ignore


class MarketCalendarService:
    """Unified market calendar contract backed by exchange_calendars."""

    CALENDAR_ID_BY_MARKET: dict[str, str] = {
        "US": "XNYS",
        "HK": "XHKG",
        "IN": "XNSE",
        "JP": "XTKS",
        "TW": "XTAI",
    }
    TIMEZONE_BY_MARKET: dict[str, str] = {
        "US": "America/New_York",
        "HK": "Asia/Hong_Kong",
        "IN": "Asia/Kolkata",
        "JP": "Asia/Tokyo",
        "TW": "Asia/Taipei",
    }
    PROVIDER_CALENDAR_ID_BY_MARKET: dict[str, str] = {
        "IN": "NSE",
    }

    def __init__(self, calendar_provider=None):
        self._calendar_provider = calendar_provider
        self._xcals_provider = xcals.get_calendar if xcals is not None else None
        self._pmc_provider = pmc.get_calendar if pmc is not None else None
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
        provider = self._calendar_provider
        uses_pmc_provider = False
        if provider is None:
            uses_pmc_provider = normalized == "IN"
            provider = self._pmc_provider if uses_pmc_provider else self._xcals_provider
        if provider is None:
            required_package = (
                "pandas_market_calendars" if normalized == "IN" else "exchange_calendars"
            )
            raise RuntimeError(f"{required_package} is required for MarketCalendarService")
        provider_calendar_id = (
            self.PROVIDER_CALENDAR_ID_BY_MARKET.get(normalized, calendar_id)
            if uses_pmc_provider
            else calendar_id
        )
        if calendar_id not in self._calendar_cache:
            self._calendar_cache[calendar_id] = provider(provider_calendar_id)
        return self._calendar_cache[calendar_id]

    @staticmethod
    def _schedule_for_range(calendar: object, *, start_day: date, end_day: date) -> pd.DataFrame:
        schedule_attr = getattr(calendar, "schedule", None)
        start_ts = pd.Timestamp(start_day)
        end_ts = pd.Timestamp(end_day)
        if callable(schedule_attr):
            return schedule_attr(start_date=start_ts, end_date=end_ts)
        if hasattr(schedule_attr, "loc"):
            return schedule_attr.loc[start_ts:end_ts]
        raise TypeError("Calendar object does not expose a usable schedule")

    def _is_session(self, calendar: object, session: pd.Timestamp) -> bool:
        is_session = getattr(calendar, "is_session", None)
        if callable(is_session):
            return bool(is_session(session))
        return not self._schedule_for_range(
            calendar,
            start_day=session.date(),
            end_day=session.date(),
        ).empty

    def _previous_session(self, calendar: object, session: pd.Timestamp) -> pd.Timestamp:
        previous_session = getattr(calendar, "previous_session", None)
        if callable(previous_session):
            return previous_session(session)

        schedule = self._schedule_for_range(
            calendar,
            start_day=(session - pd.Timedelta(days=31)).date(),
            end_day=(session - pd.Timedelta(days=1)).date(),
        )
        if schedule.empty:
            raise ValueError(f"No previous session available before {session.date().isoformat()}")
        return pd.Timestamp(schedule.index[-1])

    def is_trading_day(self, market: str, day: date | None = None) -> bool:
        calendar = self._get_calendar(market)
        candidate_day = day or self.market_now(market).date()
        return self._is_session(calendar, pd.Timestamp(candidate_day))

    def is_market_open(self, market: str, now: datetime | None = None) -> bool:
        calendar = self._get_calendar(market)
        market_now = self.market_now(market, now=now)
        current_session = pd.Timestamp(market_now.date())
        if not self._is_session(calendar, current_session):
            return False
        minute_utc = pd.Timestamp(market_now).tz_convert("UTC").floor("min")
        is_open_on_minute = getattr(calendar, "is_open_on_minute", None)
        if callable(is_open_on_minute):
            return bool(is_open_on_minute(minute_utc, ignore_breaks=False))

        schedule = self._schedule_for_range(
            calendar,
            start_day=current_session.date(),
            end_day=current_session.date(),
        )
        if schedule.empty:
            return False
        session_row = schedule.iloc[0]
        market_open = (
            session_row["market_open"] if "market_open" in session_row.index else session_row["open"]
        )
        market_close = (
            session_row["market_close"] if "market_close" in session_row.index else session_row["close"]
        )
        if market_open.tzinfo is None:
            market_open = market_open.tz_localize("UTC")
        if market_close.tzinfo is None:
            market_close = market_close.tz_localize("UTC")
        return bool(market_open.floor("min") <= minute_utc < market_close.floor("min"))

    def last_completed_trading_day(self, market: str, now: datetime | None = None) -> date:
        """Return the latest trading day that should already have end-of-day bars."""
        normalized = self.normalize_market(market)
        calendar = self._get_calendar(normalized)
        market_now = self.market_now(normalized, now=now)
        current_session = pd.Timestamp(market_now.date())

        if not self._is_session(calendar, current_session):
            date_to_session = getattr(calendar, "date_to_session", None)
            if callable(date_to_session):
                return date_to_session(current_session, direction="previous").date()
            return self._previous_session(calendar, current_session).date()

        schedule = self._schedule_for_range(
            calendar,
            start_day=current_session.date(),
            end_day=current_session.date(),
        )
        if schedule.empty:
            date_to_session = getattr(calendar, "date_to_session", None)
            if callable(date_to_session):
                return date_to_session(current_session, direction="previous").date()
            return self._previous_session(calendar, current_session).date()

        market_close = schedule.iloc[0]["close"] if "close" in schedule.columns else schedule.iloc[0]["market_close"]
        if market_close.tzinfo is None:
            market_close = market_close.tz_localize("UTC")
        close_with_buffer = (
            market_close.tz_convert(self.market_timezone(normalized)).to_pydatetime()
            + timedelta(minutes=30)
        )
        if market_now >= close_with_buffer:
            return current_session.date()
        return self._previous_session(calendar, current_session).date()

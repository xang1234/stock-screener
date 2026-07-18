"""Market calendar abstraction for supported market session-aware decisions."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from ..domain.markets.catalog import (
    MarketCatalog,
    MarketCatalogError,
    get_market_catalog,
)
from ..domain.markets.mic import MicFacts

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

    WEEKDAY_BOUNDS_FALLBACK_MARKETS = frozenset({"CN", "SG"})

    def __init__(
        self,
        calendar_provider=None,
        market_catalog: MarketCatalog | None = None,
    ):
        self._market_catalog = market_catalog or get_market_catalog()
        self._calendar_provider = calendar_provider
        self._xcals_provider = xcals.get_calendar if xcals is not None else None
        self._pmc_provider = pmc.get_calendar if pmc is not None else None
        self._calendar_cache: dict[str, object] = {}

    def normalize_market(self, market: str | None) -> str:
        try:
            normalized = self._market_catalog.get(market or "US").code
        except MarketCatalogError as exc:
            raise ValueError(
                f"Unsupported market for calendar service: {market}"
            ) from exc
        return normalized

    def _mic_facts(self, market: str | None, *, mic: str | None = None) -> MicFacts:
        normalized = self.normalize_market(market)
        return self._market_catalog.get(normalized).mic_facts_for(mic)

    def market_timezone(self, market: str, *, mic: str | None = None) -> ZoneInfo:
        return ZoneInfo(self._mic_facts(market, mic=mic).timezone)

    def market_now(
        self,
        market: str,
        now: datetime | None = None,
        *,
        mic: str | None = None,
    ) -> datetime:
        tz = self.market_timezone(market, mic=mic)
        if now is None:
            return datetime.now(tz)
        if now.tzinfo is None:
            return now.replace(tzinfo=tz)
        return now.astimezone(tz)

    def calendar_id(self, market: str, *, mic: str | None = None) -> str:
        return self._mic_facts(market, mic=mic).calendar_id

    def provider_calendar_id(
        self,
        market: str,
        *,
        mic: str | None = None,
    ) -> str | None:
        return self._mic_facts(market, mic=mic).provider_calendar_id

    def default_currency(self, market: str, *, mic: str | None = None) -> str:
        return self._mic_facts(market, mic=mic).default_currency

    def _get_calendar(self, market: str, *, mic: str | None = None):
        normalized = self.normalize_market(market)
        facts = self._mic_facts(normalized, mic=mic)
        calendar_id = facts.calendar_id
        provider_calendar_id = facts.provider_calendar_id or calendar_id
        provider = self._calendar_provider
        uses_pmc_provider = False
        if provider is None:
            uses_pmc_provider = normalized == "IN"
            provider = self._pmc_provider if uses_pmc_provider else self._xcals_provider
        if provider is None:
            required_package = (
                "pandas_market_calendars" if uses_pmc_provider else "exchange_calendars"
            )
            raise RuntimeError(f"{required_package} is required for MarketCalendarService")
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

    @staticmethod
    def _is_calendar_bounds_error(exc: Exception) -> bool:
        class_name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        return (
            "outofbounds" in class_name
            or "out of bounds" in message
            or "last session" in message
            or "first session" in message
        )

    @staticmethod
    def _is_weekday(day: date) -> bool:
        return day.weekday() < 5

    @classmethod
    def _previous_weekday(cls, day: date) -> date:
        candidate = day - timedelta(days=1)
        while not cls._is_weekday(candidate):
            candidate -= timedelta(days=1)
        return candidate

    def is_trading_day(
        self,
        market: str,
        day: date | None = None,
        *,
        mic: str | None = None,
    ) -> bool:
        normalized = self.normalize_market(market)
        candidate_day = day or self.market_now(normalized, mic=mic).date()
        try:
            calendar = self._get_calendar(normalized, mic=mic)
            return self._is_session(calendar, pd.Timestamp(candidate_day))
        except Exception as exc:
            if (
                normalized in self.WEEKDAY_BOUNDS_FALLBACK_MARKETS
                and self._is_calendar_bounds_error(exc)
            ):
                return self._is_weekday(candidate_day)
            raise

    def trading_days(
        self,
        market: str,
        start: date,
        end: date,
        *,
        mic: str | None = None,
    ) -> list[date]:
        """Trading days in ``[start, end]`` (inclusive), chronological order.

        The canonical way to enumerate sessions in a range, so callers don't
        reimplement a day-by-day loop. Preserves the per-market fallbacks in
        ``is_trading_day``.
        """
        normalized = self.normalize_market(market)
        days: list[date] = []
        day = start
        while day <= end:
            if self.is_trading_day(normalized, day, mic=mic):
                days.append(day)
            day += timedelta(days=1)
        return days

    def session_anchors(
        self,
        market: str,
        as_of_date: date,
        *,
        offsets: tuple[int, ...],
        mic: str | None = None,
    ) -> dict[int, date]:
        """Resolve exact prior Market sessions for fixed lookback offsets."""
        normalized = self.normalize_market(market)
        if not offsets or min(offsets) < 1:
            raise ValueError("session offsets must be positive")
        if not self.is_trading_day(normalized, as_of_date, mic=mic):
            raise ValueError(
                f"{as_of_date.isoformat()} is not a {normalized} trading session"
            )
        maximum = max(offsets)
        start = as_of_date - timedelta(days=maximum * 2 + 30)
        sessions = self.trading_days(normalized, start, as_of_date, mic=mic)
        if len(sessions) <= maximum:
            raise ValueError(
                f"{normalized} calendar has {len(sessions)} sessions; "
                f"{maximum + 1} required"
            )
        return {
            0: sessions[-1],
            **{offset: sessions[-1 - offset] for offset in offsets},
        }

    def is_market_open(
        self,
        market: str,
        now: datetime | None = None,
        *,
        mic: str | None = None,
    ) -> bool:
        calendar = self._get_calendar(market, mic=mic)
        market_now = self.market_now(market, now=now, mic=mic)
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

    def last_completed_trading_day(
        self,
        market: str,
        now: datetime | None = None,
        *,
        mic: str | None = None,
    ) -> date:
        """Return the latest trading day that should already have end-of-day bars."""
        normalized = self.normalize_market(market)
        market_now = self.market_now(normalized, now=now, mic=mic)
        current_session = pd.Timestamp(market_now.date())

        try:
            calendar = self._get_calendar(normalized, mic=mic)

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
                market_close.tz_convert(
                    self.market_timezone(normalized, mic=mic)
                ).to_pydatetime()
                + timedelta(minutes=30)
            )
            if market_now >= close_with_buffer:
                return current_session.date()
            return self._previous_session(calendar, current_session).date()
        except Exception as exc:
            if (
                normalized in self.WEEKDAY_BOUNDS_FALLBACK_MARKETS
                and self._is_calendar_bounds_error(exc)
            ):
                if self._is_weekday(current_session.date()) and market_now.time().hour >= 16:
                    return current_session.date()
                return self._previous_weekday(current_session.date())
            raise

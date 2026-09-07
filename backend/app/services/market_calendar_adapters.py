"""Adapters that normalize third-party market calendar APIs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from time import monotonic
from typing import Protocol

import pandas as pd

from .redis_pool import get_redis_client

try:
    from redis.exceptions import RedisError
except ModuleNotFoundError:  # pragma: no cover - runtime guard mirrors redis_pool
    _REDIS_CACHE_EXCEPTIONS: tuple[type[BaseException], ...] = (RuntimeError, OSError)
else:
    _REDIS_CACHE_EXCEPTIONS = (RedisError, RuntimeError, OSError)

logger = logging.getLogger(__name__)

SESSION_RANGE_CACHE_TTL_SECONDS = 60 * 60 * 24
_SESSION_RANGE_CACHE_RETRY_INTERVAL_SECONDS = 30.0
_session_range_cache_retry_after = 0.0


class CalendarScheduleUnavailable(RuntimeError):
    """Raised when a calendar object cannot expose a session schedule."""


class MarketCalendarAdapter(Protocol):
    """Provider-neutral calendar operations used by MarketCalendarService."""

    def is_session(self, session: pd.Timestamp) -> bool: ...

    def sessions_in_range(self, start_day: date, end_day: date) -> tuple[date, ...]: ...

    def first_session_date(self) -> date | None: ...

    def last_session_date(self) -> date | None: ...

    def session_close(self, day: date) -> pd.Timestamp | None: ...

    def session_open_ranges(
        self, day: date
    ) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...] | None: ...

    def is_open_on_minute(self, minute_utc: pd.Timestamp) -> bool | None: ...


@dataclass(slots=True)
class RawMarketCalendarAdapter:
    """Normalize exchange_calendars and pandas_market_calendars objects."""

    raw_calendar: object
    cache_namespace: str = "default"
    use_shared_cache: bool = True

    def is_session(self, session: pd.Timestamp) -> bool:
        is_session = getattr(self.raw_calendar, "is_session", None)
        if callable(is_session):
            return bool(is_session(session))
        return session.date() in self.sessions_in_range(
            session.date(),
            session.date(),
        )

    def sessions_in_range(self, start_day: date, end_day: date) -> tuple[date, ...]:
        if start_day > end_day:
            return ()
        if not self.use_shared_cache:
            return self._compute_sessions_in_range(start_day, end_day)
        cache_key = self._session_range_cache_key(start_day, end_day)
        cached_sessions = self._read_session_range_cache(cache_key)
        if cached_sessions is not None:
            return cached_sessions
        sessions = self._compute_sessions_in_range(start_day, end_day)
        self._write_session_range_cache(cache_key, sessions)
        return sessions

    def first_session_date(self) -> date | None:
        return self._session_bound_date(
            ("first_session", "first_session_date"),
            fallback_index=0,
        )

    def last_session_date(self) -> date | None:
        return self._session_bound_date(
            ("last_session", "last_session_date"),
            fallback_index=-1,
        )

    def _session_bound_date(
        self,
        attr_names: tuple[str, ...],
        *,
        fallback_index: int,
    ) -> date | None:
        for attr_name in attr_names:
            attr_value = getattr(self.raw_calendar, attr_name, None)
            if attr_value is None:
                continue
            if callable(attr_value):
                attr_value = attr_value()
            if attr_value is None:
                continue
            timestamp = pd.Timestamp(attr_value)
            if pd.isna(timestamp):
                continue
            return timestamp.date()

        sessions_attr = getattr(self.raw_calendar, "sessions", None)
        if sessions_attr is None:
            return None
        try:
            if len(sessions_attr) == 0:
                return None
            return pd.Timestamp(sessions_attr[fallback_index]).date()
        except (IndexError, KeyError, TypeError):
            sessions = tuple(sessions_attr)
            if not sessions:
                return None
            return pd.Timestamp(sessions[fallback_index]).date()

    def session_close(self, day: date) -> pd.Timestamp | None:
        schedule = self._schedule_for_range(start_day=day, end_day=day)
        if schedule.empty:
            return None
        session_row = schedule.iloc[0]
        return self._timestamp_field(
            session_row,
            ("close", "market_close"),
            day=day,
        )

    def session_open_ranges(
        self,
        day: date,
    ) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...] | None:
        schedule = self._schedule_for_range(start_day=day, end_day=day)
        if schedule.empty:
            return None
        session_row = schedule.iloc[0]
        market_open = self._timestamp_field(
            session_row,
            ("open", "market_open"),
            day=day,
        )
        market_close = self._timestamp_field(
            session_row,
            ("close", "market_close"),
            day=day,
        )
        break_start = self._optional_timestamp_field(
            session_row,
            ("break_start", "market_break_start"),
        )
        break_end = self._optional_timestamp_field(
            session_row,
            ("break_end", "market_break_end"),
        )
        if (
            break_start is not None
            and break_end is not None
            and market_open < break_start < break_end < market_close
        ):
            return (
                (market_open, break_start),
                (break_end, market_close),
            )
        return ((market_open, market_close),)

    def is_open_on_minute(self, minute_utc: pd.Timestamp) -> bool | None:
        is_open_on_minute = getattr(self.raw_calendar, "is_open_on_minute", None)
        if not callable(is_open_on_minute):
            return None
        return bool(is_open_on_minute(minute_utc, ignore_breaks=False))

    def _schedule_for_range(
        self,
        *,
        start_day: date,
        end_day: date,
    ) -> pd.DataFrame:
        schedule_attr = getattr(self.raw_calendar, "schedule", None)
        start_ts = pd.Timestamp(start_day)
        end_ts = pd.Timestamp(end_day)
        if callable(schedule_attr):
            return schedule_attr(start_date=start_ts, end_date=end_ts)
        if hasattr(schedule_attr, "loc"):
            return schedule_attr.loc[start_ts:end_ts]
        raise CalendarScheduleUnavailable(
            "Calendar object does not expose a usable schedule"
        )

    def _compute_sessions_in_range(
        self,
        start_day: date,
        end_day: date,
    ) -> tuple[date, ...]:
        sessions_in_range = getattr(self.raw_calendar, "sessions_in_range", None)
        if callable(sessions_in_range):
            return tuple(
                pd.Timestamp(session).date()
                for session in sessions_in_range(
                    pd.Timestamp(start_day),
                    pd.Timestamp(end_day),
                )
            )

        sessions_attr = getattr(self.raw_calendar, "sessions", None)
        if sessions_attr is not None:
            return tuple(
                session_day
                for session_day in (
                    pd.Timestamp(session).date() for session in sessions_attr
                )
                if start_day <= session_day <= end_day
            )

        schedule = self._schedule_for_range(
            start_day=start_day,
            end_day=end_day,
        )
        return tuple(pd.Timestamp(session).date() for session in schedule.index)

    def _session_range_cache_key(self, start_day: date, end_day: date) -> str:
        return (
            "calendar:sessions:v1:"
            f"{self.cache_namespace}:{start_day.isoformat()}:{end_day.isoformat()}"
        )

    def _read_session_range_cache(self, cache_key: str) -> tuple[date, ...] | None:
        client = self._session_range_cache_client()
        if client is None:
            return None
        try:
            raw = client.get(cache_key)
        except _REDIS_CACHE_EXCEPTIONS as exc:
            logger.debug("Calendar session-range cache read failed: %s", exc)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            return tuple(date.fromisoformat(value) for value in payload)
        except (TypeError, ValueError) as exc:
            logger.debug("Calendar session-range cache decode failed: %s", exc)
            return None

    def _write_session_range_cache(
        self,
        cache_key: str,
        sessions: tuple[date, ...],
    ) -> None:
        client = self._session_range_cache_client()
        if client is None:
            return
        try:
            client.setex(
                cache_key,
                SESSION_RANGE_CACHE_TTL_SECONDS,
                json.dumps(
                    [session.isoformat() for session in sessions],
                    separators=(",", ":"),
                ),
            )
        except _REDIS_CACHE_EXCEPTIONS as exc:
            logger.debug("Calendar session-range cache write failed: %s", exc)

    def _session_range_cache_client(self):
        return _get_session_range_cache_client()

    @classmethod
    def _timestamp_field(
        cls,
        session_row: pd.Series,
        field_names: tuple[str, ...],
        *,
        day: date,
    ) -> pd.Timestamp:
        for field_name in field_names:
            if field_name in session_row.index:
                return cls._utc_timestamp(session_row[field_name])
        raise CalendarScheduleUnavailable(
            f"Calendar schedule for {day.isoformat()} is missing all of {field_names}"
        )

    @classmethod
    def _optional_timestamp_field(
        cls,
        session_row: pd.Series,
        field_names: tuple[str, ...],
    ) -> pd.Timestamp | None:
        for field_name in field_names:
            if field_name in session_row.index:
                value = session_row[field_name]
                if pd.isna(value):
                    return None
                return cls._utc_timestamp(value)
        return None

    @staticmethod
    def _utc_timestamp(value: object) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")


def _get_session_range_cache_client():
    global _session_range_cache_retry_after

    now = monotonic()
    if now < _session_range_cache_retry_after:
        return None
    client = get_redis_client()
    if client is None:
        _session_range_cache_retry_after = (
            now + _SESSION_RANGE_CACHE_RETRY_INTERVAL_SECONDS
        )
    return client

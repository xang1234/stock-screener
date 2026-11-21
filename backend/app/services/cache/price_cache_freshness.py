"""Freshness and staleness policy for price cache hot paths.

Decisions follow each symbol's own market calendar (``MarketCalendarService``).
A cached bar is partial when it was fetched while its market's session was
open; it becomes stale once that session has completed. The verdict is derived
from ``fetch_timestamp`` rather than the stored ``needs_refresh_after_close``
flag, so metadata written by the old US-clock writer is re-judged correctly.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

from ...utils.market_hours import EASTERN, get_eastern_now, is_market_open

US_MARKET = "US"
# Matches MarketCalendarService.last_completed_trading_day: a session's bar is
# final 30 minutes after its close.
SESSION_SETTLEMENT_BUFFER = timedelta(minutes=30)
# How long a clock-based "last completed session" answer is reused. It only
# changes at session boundaries, so a few seconds of reuse is invisible.
LAST_COMPLETED_REUSE_SECONDS = 5.0

# Markets whose calendar lookups are failing. Bulk paths judge thousands of
# symbols, so an outage warns once per market until a lookup succeeds again.
# ponytail: unlocked set (each op is GIL-atomic); two threads failing at the
# same moment can log the warning twice.
_CALENDAR_OUTAGES: set[str] = set()


def warn_calendar_outage(logger, market: str, message: str, *args) -> None:
    """Log ``message % (market, *args)`` for the first failure of an outage only."""
    if market.upper() not in _CALENDAR_OUTAGES:
        _CALENDAR_OUTAGES.add(market.upper())
        logger.warning(message + " (logged once until the calendar recovers)", market, *args)


def settlement_cutoff(calendar, market: str, session_day: date) -> Optional[datetime]:
    """When ``session_day``'s bar in ``market`` is final: its close plus the
    settlement buffer. None on a non-trading day. A fetch before it may hold
    an in-progress bar for that day (lunch breaks included)."""
    if not calendar.is_trading_day(market, session_day):
        return None
    return calendar.session_close(market, session_day) + SESSION_SETTLEMENT_BUFFER


def fetched_at(meta: Mapping) -> Optional[datetime]:
    """Aware fetch time from metadata; legacy naive timestamps were written in ET."""
    raw = meta.get("fetch_timestamp")
    if not raw:
        return None
    try:
        value = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return value if value.tzinfo is not None else EASTERN.localize(value)


_UNKNOWN_FETCH_TIME = datetime.min.replace(tzinfo=timezone.utc)


def latest_fetch_metadata(metas: Iterable[Optional[Mapping]]) -> Optional[Mapping]:
    """The most recently written record among a symbol's metadata keys.

    Writers split a symbol between the US key (callers that omit the market)
    and its own market key, so the newest ``fetch_timestamp`` is the last write.
    """
    present = [meta for meta in metas if meta]
    if not present:
        return None
    return max(present, key=lambda meta: fetched_at(meta) or _UNKNOWN_FETCH_TIME)


class PriceCacheFreshnessPolicy:
    """Encapsulates staleness/freshness decisions for cached price data."""

    def __init__(
        self,
        *,
        logger,
        redis_client,
        fetch_meta_key_template: str | Sequence[str],
        get_expected_data_date: Callable[[str], Optional[date]],
        get_market_calendar: Callable[[], object],
        resolve_calendar_markets: Callable[[Mapping[str, str]], Dict[str, str]],
    ) -> None:
        self._logger = logger
        self._redis_client = redis_client
        if isinstance(fetch_meta_key_template, str):
            self._fetch_meta_key_templates = (fetch_meta_key_template,)
        else:
            self._fetch_meta_key_templates = tuple(fetch_meta_key_template)
        self._get_expected_data_date = get_expected_data_date
        self._get_market_calendar = get_market_calendar
        self._resolve_calendar_markets = resolve_calendar_markets
        # Calendar calls cost ~1-2 ms and bulk paths judge thousands of symbols.
        # A session's settlement cutoff never changes once known; the last
        # completed session only moves at session boundaries.
        self._settlement_cutoffs: Dict[tuple[str, date], Optional[datetime]] = {}
        self._last_completed_memo: Dict[str, tuple[Optional[datetime], float, date]] = {}

    def last_completed_trading_day(self, market: str, now: Optional[datetime] = None) -> date:
        """Calendar's last completed session, reused for the same ``now`` or briefly for the clock."""
        tick = monotonic()
        memo = self._last_completed_memo.get(market)
        if memo is not None:
            memo_now, memo_tick, value = memo
            if now is not None and memo_now == now:
                return value
            if now is None and memo_now is None and tick - memo_tick < LAST_COMPLETED_REUSE_SECONDS:
                return value
        value = self._get_market_calendar().last_completed_trading_day(market, now)
        _CALENDAR_OUTAGES.discard(market.upper())
        self._last_completed_memo[market] = (now, tick, value)
        return value

    def is_data_fresh(self, last_date: date | None, market: str = US_MARKET) -> bool:
        """Return True when cached data covers the market's last completed session."""
        if last_date is None:
            return False
        expected = self._get_expected_data_date(market)
        if expected is None:
            return False
        is_fresh = last_date >= expected
        if not is_fresh:
            self._logger.debug(
                "Data is stale for %s (last: %s, expected: %s)", market, last_date, expected
            )
        return is_fresh

    def partial_session_day(self, fetch_time: datetime, market: str) -> Optional[date]:
        """Trading day whose bar a fetch at ``fetch_time`` may hold only partially, else None.

        A fetch on trading day D before D's close plus the settlement buffer can
        hold an in-progress bar for D (including during a lunch break). Calendar
        errors propagate to the caller.
        """
        calendar = self._get_market_calendar()
        session_day = calendar.market_now(market, fetch_time).date()
        key = (market, session_day)
        if key not in self._settlement_cutoffs:
            self._settlement_cutoffs[key] = settlement_cutoff(calendar, market, session_day)
        final_after = self._settlement_cutoffs[key]
        return session_day if final_after is not None and fetch_time < final_after else None

    def is_fetch_metadata_stale(
        self,
        meta: Optional[Mapping],
        *,
        market: str = US_MARKET,
        now: Optional[datetime] = None,
    ) -> bool:
        """Return True when the cached bar was fetched mid-session and that session has closed."""
        if not meta:
            return False
        fetch_time = fetched_at(meta)
        if fetch_time is None:
            return bool(meta.get("needs_refresh_after_close", False))
        try:
            session_day = self.partial_session_day(fetch_time, market)
            if session_day is None:
                return False
            return self.last_completed_trading_day(market, now) >= session_day
        except Exception as exc:
            warn_calendar_outage(
                self._logger,
                market,
                "Calendar unavailable for %s freshness; using the fallback rule: %s",
                exc,
            )
            if market == US_MARKET:
                return self._legacy_us_metadata_stale(meta, now)
            return True

    @staticmethod
    def _legacy_us_metadata_stale(meta: Mapping, now: Optional[datetime]) -> bool:
        """Pre-calendar US rule, kept as the fallback when the US calendar is unavailable."""
        if not meta.get("needs_refresh_after_close", False):
            return False
        if now is None:
            now_et = get_eastern_now()
        else:
            now_et = now.astimezone(EASTERN) if now.tzinfo else now
        if is_market_open(now_et):
            return False
        return now_et.time() >= time(16, 30)

    def get_stale_intraday_symbols(self, *, now: Optional[datetime] = None) -> list[str]:
        """Return all symbols whose cached bar was fetched mid-session in a now-closed session."""
        if not self._redis_client:
            return []
        try:
            all_keys = []
            key_market_by_symbol: Dict[str, str] = {}
            seen_key_strings = set()

            for template in self._fetch_meta_key_templates:
                pattern = template.replace("{symbol}", "*")
                cursor = 0
                while True:
                    cursor, keys = self._redis_client.scan(cursor, match=pattern, count=500)
                    for key in keys:
                        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                        if key_str in seen_key_strings:
                            continue
                        seen_key_strings.add(key_str)
                        parts = key_str.split(":")
                        if len(parts) == 3:
                            symbol, key_market = parts[1], US_MARKET
                        elif len(parts) == 4:
                            symbol, key_market = parts[2], parts[1]
                        else:
                            continue
                        all_keys.append((key, symbol))
                        key_market_by_symbol.setdefault(symbol, key_market)
                    if cursor == 0:
                        break

            if not all_keys:
                return []

            pipeline = self._redis_client.pipeline()
            for key, _symbol in all_keys:
                pipeline.get(key)
            # One corrupted key must not blank the whole scan.
            meta_values = pipeline.execute(raise_on_error=False)
            calendar_markets = self._resolve_calendar_markets(key_market_by_symbol)

            metas_by_symbol: Dict[str, list] = {}
            for (_key, symbol), meta_json in zip(all_keys, meta_values):
                if not meta_json or isinstance(meta_json, Exception):
                    continue
                try:
                    metas_by_symbol.setdefault(symbol, []).append(json.loads(meta_json))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

            stale_symbols: list[str] = []
            for symbol, metas in metas_by_symbol.items():
                meta = latest_fetch_metadata(metas)
                if self.is_fetch_metadata_stale(
                    meta,
                    market=calendar_markets.get(symbol, US_MARKET),
                    now=now,
                ):
                    stale_symbols.append(symbol)

            self._logger.info(
                "Found %s symbols with stale intraday data",
                len(stale_symbols),
            )
            return stale_symbols
        except Exception as exc:
            self._logger.error("Error scanning for stale intraday symbols: %s", exc, exc_info=True)
            return []

    def get_staleness_status(self) -> dict:
        """Return aggregate staleness diagnostics payload."""
        now_et = get_eastern_now()
        stale_symbols = self.get_stale_intraday_symbols()
        return {
            "stale_intraday_count": len(stale_symbols),
            "stale_symbols": stale_symbols[:10],
            "market_is_open": is_market_open(now_et),
            "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "has_stale_data": bool(stale_symbols),
        }

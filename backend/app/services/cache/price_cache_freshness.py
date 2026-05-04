"""Freshness and staleness policy for price cache hot paths."""

from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Callable, Dict, Optional, Sequence

from ...utils.market_hours import get_eastern_now, is_market_open


class PriceCacheFreshnessPolicy:
    """Encapsulates staleness/freshness decisions for cached price data."""

    def __init__(
        self,
        *,
        logger,
        redis_client,
        fetch_meta_key_template: str | Sequence[str],
        get_expected_data_date: Callable[[], Optional[date]],
        get_fetch_metadata: Callable[[str], Optional[Dict]],
    ) -> None:
        self._logger = logger
        self._redis_client = redis_client
        if isinstance(fetch_meta_key_template, str):
            self._fetch_meta_key_templates = (fetch_meta_key_template,)
        else:
            self._fetch_meta_key_templates = tuple(fetch_meta_key_template)
        self._get_expected_data_date = get_expected_data_date
        self._get_fetch_metadata = get_fetch_metadata

    def is_data_fresh(self, last_date: date | None) -> bool:
        """Return True when cached data is fresh according to trading-day policy."""
        if last_date is None:
            return False
        expected = self._get_expected_data_date()
        if expected is None:
            return False
        is_fresh = last_date >= expected
        if not is_fresh:
            self._logger.debug("Data is stale (last: %s, expected: %s)", last_date, expected)
        return is_fresh

    def is_fetch_metadata_stale(
        self,
        meta: Optional[Dict],
        *,
        now_et: Optional[datetime] = None,
    ) -> bool:
        """Return True when same-day intraday fetch metadata is stale after close."""
        if not meta or not meta.get("needs_refresh_after_close", False):
            return False
        now_et = now_et or get_eastern_now()
        if is_market_open(now_et):
            return False
        return now_et.time() >= time(16, 30)

    def is_intraday_data_stale(self, symbol: str) -> bool:
        """Return True when symbol was fetched intraday and now requires refresh."""
        meta = self._get_fetch_metadata(symbol)
        is_stale = self.is_fetch_metadata_stale(meta)
        if is_stale:
            self._logger.debug(
                "%s: intraday data is stale (fetched during market, now after close)",
                symbol,
            )
        return is_stale

    def get_stale_intraday_symbols(self) -> list[str]:
        """Return all symbols with post-close stale intraday metadata."""
        if not self._redis_client:
            return []
        try:
            now_et = get_eastern_now()
            if is_market_open(now_et):
                return []
            if now_et.time() < time(16, 30):
                return []

            all_keys = []
            all_symbols = []
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
                            all_keys.append(key)
                            all_symbols.append(parts[1])
                        elif len(parts) == 4:
                            all_keys.append(key)
                            all_symbols.append(parts[2])
                    if cursor == 0:
                        break

            if not all_keys:
                return []

            pipeline = self._redis_client.pipeline()
            for key in all_keys:
                pipeline.get(key)
            meta_values = pipeline.execute()

            stale_symbols: list[str] = []
            for symbol, meta_json in zip(all_symbols, meta_values):
                if not meta_json:
                    continue
                try:
                    meta = json.loads(meta_json)
                    if meta.get("needs_refresh_after_close", False):
                        stale_symbols.append(symbol)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

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

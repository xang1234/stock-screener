"""Static-site daily price refresh orchestration."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

from app.models.stock_universe import StockUniverse
from app.services.bulk_data_fetcher import BulkDataFetcher
from app.services.price_history_coverage import classify_price_history
from app.services.price_refresh_planning import (
    NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
    STALE_PRICE_TOP_UP_PERIOD,
)
from app.utils.symbol_support import split_supported_price_symbols


STATIC_DAILY_PRICE_REFRESH_PERIOD = STALE_PRICE_TOP_UP_PERIOD
STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD = NO_HISTORY_PRICE_BOOTSTRAP_PERIOD
STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE = 250

# Markets where Yahoo's 429 backoff windows are long enough that a single
# refresh pass routinely leaves a tail of rate-limited symbols. For these
# markets we wait ``STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS`` after the main
# loop and replay only the symbols whose failure looks transient, in a
# smaller batch (``STATIC_RATE_LIMITED_RETRY_BATCH_SIZE``).
STATIC_RATE_LIMITED_RETRY_MARKETS = frozenset({"IN"})
STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS = 300
STATIC_RATE_LIMITED_RETRY_BATCH_SIZE = 25


def static_daily_price_refresh_batch_size(market: str | None) -> int:
    if market:
        from app.services.rate_budget_policy import get_rate_budget_policy

        return get_rate_budget_policy().get_batch_size("yfinance", market)
    return STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE


def _iter_chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def _is_rate_limit_failure(payload: dict[str, Any]) -> bool:
    if not payload.get("has_error"):
        return False
    error = str(payload.get("error") or "").lower()
    if not error:
        return False
    indicators = ("rate", "429", "too many", "limit", "throttl")
    return any(token in error for token in indicators)


class StaticDailyPriceRefreshService:
    """Refresh price rows needed by the static-site snapshot build."""

    def __init__(
        self,
        *,
        session_factory,
        price_cache,
        fetcher: BulkDataFetcher,
        batch_size_for_market: Callable[[str | None], int] = static_daily_price_refresh_batch_size,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._price_cache = price_cache
        self._fetcher = fetcher
        self._batch_size_for_market = batch_size_for_market
        if sleep is None:
            import time

            sleep = time.sleep
        self._sleep = sleep

    def refresh(self, *, as_of_date: date, market: str | None = None) -> dict[str, Any]:
        with self._session_factory() as db:
            query = (
                db.query(StockUniverse.symbol)
                .filter(StockUniverse.is_active.is_(True))
                .order_by(StockUniverse.market_cap.desc().nullslast(), StockUniverse.symbol.asc())
            )
            if market is not None:
                query = query.filter(StockUniverse.market == market)
            active_symbols = [symbol for symbol, in query.all()]
            supported_symbols, skipped_symbols = split_supported_price_symbols(active_symbols)
            coverage = classify_price_history(
                db,
                symbols=supported_symbols,
                as_of_date=as_of_date,
            )

        db_fresh_symbols = list(coverage.fresh)
        stale_symbols = list(coverage.stale)
        no_history_symbols = list(coverage.no_history)

        if not stale_symbols and not no_history_symbols:
            print(
                f"[static-daily prices] Database already has fresh price rows for "
                f"{len(db_fresh_symbols):,} supported symbols as of {as_of_date}.",
                flush=True,
            )
            return {
                "status": "skipped",
                "market": market,
                "as_of_date": as_of_date.isoformat(),
                "total_active_symbols": len(active_symbols),
                "supported_symbols": len(supported_symbols),
                "db_fresh_symbols": len(db_fresh_symbols),
                "stale_symbols": len(stale_symbols),
                "no_history_symbols": len(no_history_symbols),
                "skipped_unsupported_symbols": len(skipped_symbols),
                "yahoo_fetched_symbols": 0,
                "yahoo_failed_symbols": 0,
            }

        batch_size = self._batch_size_for_market(market)
        total_batches = (
            (len(stale_symbols) + batch_size - 1) // batch_size
            + (len(no_history_symbols) + batch_size - 1) // batch_size
        )

        print(
            f"[static-daily prices] Refreshing {len(stale_symbols):,} stale and "
            f"{len(no_history_symbols):,} no-history symbols in {total_batches} batches for {as_of_date} "
            f"(DB fresh: {len(db_fresh_symbols):,}, unsupported skipped: {len(skipped_symbols):,}).",
            flush=True,
        )

        stale_refreshed, stale_failed, stale_rate_limited = self._fetch_and_store(
            stale_symbols,
            period=STATIC_DAILY_PRICE_REFRESH_PERIOD,
            batch_size=batch_size,
            market=market,
        )
        bootstrap_refreshed, bootstrap_failed, bootstrap_rate_limited = self._fetch_and_store(
            no_history_symbols,
            period=STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
            batch_size=batch_size,
            market=market,
        )
        refreshed = stale_refreshed + bootstrap_refreshed
        failed = stale_failed + bootstrap_failed
        retry_stats = self._retry_rate_limited_failures(
            market=market,
            rate_limited_symbols_by_period={
                STATIC_DAILY_PRICE_REFRESH_PERIOD: stale_rate_limited,
                STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD: bootstrap_rate_limited,
            },
        )
        refreshed += retry_stats["recovered"]
        failed -= retry_stats["recovered"]

        return {
            "status": "completed",
            "market": market,
            "as_of_date": as_of_date.isoformat(),
            "total_active_symbols": len(active_symbols),
            "supported_symbols": len(supported_symbols),
            "db_fresh_symbols": len(db_fresh_symbols),
            "stale_symbols": len(stale_symbols),
            "no_history_symbols": len(no_history_symbols),
            "skipped_unsupported_symbols": len(skipped_symbols),
            "yahoo_fetched_symbols": refreshed,
            "yahoo_failed_symbols": failed,
            "rate_limited_retry": retry_stats,
        }

    def _fetch_and_store(
        self,
        symbols: list[str],
        *,
        period: str,
        batch_size: int,
        market: str | None,
    ) -> tuple[int, int, list[str]]:
        refreshed_count = 0
        failed_count = 0
        rate_limited: list[str] = []
        total_symbols = len(symbols)
        if not symbols:
            return 0, 0, []
        total_group_batches = (total_symbols + batch_size - 1) // batch_size
        for batch_index, batch_symbols in enumerate(
            _iter_chunks(symbols, batch_size),
            start=1,
        ):
            processed_before = refreshed_count + failed_count
            print(
                f"[static-daily prices] Batch {batch_index}/{total_group_batches}: "
                f"{processed_before:,}/{total_symbols:,} processed, fetching "
                f"{len(batch_symbols):,} symbols from Yahoo ({period}).",
                flush=True,
            )
            batch_results = self._fetcher.fetch_prices_in_batches(
                batch_symbols,
                period=period,
                start_batch_size=batch_size,
                market=market,
            )
            batch_to_store: dict[str, Any] = {}
            for symbol, payload in batch_results.items():
                price_data = payload.get("price_data")
                if not payload.get("has_error") and price_data is not None and not price_data.empty:
                    batch_to_store[symbol] = price_data
                    refreshed_count += 1
                else:
                    failed_count += 1
                    if _is_rate_limit_failure(payload):
                        rate_limited.append(symbol)
            if batch_to_store:
                self._price_cache.store_batch_in_cache(
                    batch_to_store,
                    also_store_db=True,
                    market=market,
                )
            print(
                f"[static-daily prices] Batch {batch_index}/{total_group_batches} complete: "
                f"{refreshed_count + failed_count:,}/{total_symbols:,} processed, "
                f"{refreshed_count:,} refreshed, {failed_count:,} failed.",
                flush=True,
            )
        return refreshed_count, failed_count, rate_limited

    def _retry_rate_limited_failures(
        self,
        *,
        market: str | None,
        rate_limited_symbols_by_period: dict[str, list[str]],
    ) -> dict[str, Any]:
        skipped_payload: dict[str, Any] = {
            "attempted": 0,
            "recovered": 0,
            "still_failed": 0,
            "wait_seconds": 0,
            "batch_size": STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
        }
        retry_groups = [
            (period, sorted(set(symbols)))
            for period, symbols in rate_limited_symbols_by_period.items()
            if symbols
        ]
        attempted = sum(len(symbols) for _period, symbols in retry_groups)
        if not attempted:
            return skipped_payload
        normalized = (market or "").upper()
        if normalized not in STATIC_RATE_LIMITED_RETRY_MARKETS:
            print(
                f"[static-daily prices] Skipping rate-limited retry for market={normalized or 'shared'}: "
                f"{attempted} symbols looked throttled but market is outside the retry allowlist.",
                flush=True,
            )
            return skipped_payload

        print(
            f"[static-daily prices:{normalized}] Yahoo flagged {attempted} symbols as rate-limited; "
            f"waiting {STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS}s then retrying with batch size "
            f"{STATIC_RATE_LIMITED_RETRY_BATCH_SIZE}.",
            flush=True,
        )
        self._sleep(STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS)

        recovered = 0
        for period, unique_symbols in retry_groups:
            retry_results = self._fetcher.fetch_prices_in_batches(
                unique_symbols,
                period=period,
                start_batch_size=STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
                market=market,
            )
            recovered_payload: dict[str, Any] = {}
            for symbol, payload in retry_results.items():
                price_data = payload.get("price_data")
                if not payload.get("has_error") and price_data is not None and not price_data.empty:
                    recovered_payload[symbol] = price_data
                    recovered += 1
            if recovered_payload:
                self._price_cache.store_batch_in_cache(
                    recovered_payload,
                    also_store_db=True,
                    market=market,
                )
        still_failed = attempted - recovered
        print(
            f"[static-daily prices:{normalized}] Rate-limited retry complete: "
            f"{recovered}/{attempted} recovered, {still_failed} still failed.",
            flush=True,
        )
        return {
            "attempted": attempted,
            "recovered": recovered,
            "still_failed": still_failed,
            "wait_seconds": STATIC_RATE_LIMITED_RETRY_WAIT_SECONDS,
            "batch_size": STATIC_RATE_LIMITED_RETRY_BATCH_SIZE,
        }

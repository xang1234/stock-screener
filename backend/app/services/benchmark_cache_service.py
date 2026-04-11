"""
Benchmark Cache Service for market-aware benchmark data caching.

Provides singleton-style caching of benchmark data to eliminate redundant API calls
during bulk scans. Uses Redis for hot cache with database persistence.
"""
import logging
import pickle
import time
from typing import Any, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    redis = Any  # type: ignore

from ..database import SessionLocal
from ..models.stock import StockPrice
from ..config import settings
from .redis_pool import get_redis_client, is_redis_enabled
from .market_calendar_service import MarketCalendarService
from .benchmark_registry_service import benchmark_registry

logger = logging.getLogger(__name__)


class BenchmarkCacheService:
    """
    Singleton service for caching benchmark (SPY) data.

    Strategy:
    - Store SPY in Redis with 24-hour TTL (expires after market close)
    - Persist to StockPrice table for historical reference
    - Thread-safe singleton pattern for concurrent Celery workers
    - Automatic refresh if data is stale
    """

    # Redis keys
    REDIS_KEY_PREFIX = "benchmark:"
    REDIS_LOCK_KEY_SUFFIX = ":lock"

    # TTL: expires at market close + 1 hour buffer
    CACHE_TTL_SECONDS = 86400  # 24 hours
    LOCK_TIMEOUT_SECONDS = 10  # Max time to wait for lock
    LOCK_EXPIRY_SECONDS = 30  # Lock auto-expires after 30 seconds

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable[[], Session]] = None,
    ):
        """Initialize benchmark cache service."""
        self._session_factory = session_factory or SessionLocal
        if redis_client:
            self._redis_client = redis_client
        else:
            # Use shared connection pool for efficiency
            self._redis_client = get_redis_client()
            if self._redis_client:
                logger.info("Connected to Redis for benchmark caching (using shared pool)")
            elif is_redis_enabled():
                logger.warning("Redis connection failed. Will use database fallback.")
            else:
                logger.info("Redis disabled for this runtime. Using database fallback.")
        self._market_calendar = MarketCalendarService()
        self._benchmark_registry = benchmark_registry

    def _normalize_market(self, market: str | None) -> str:
        return self._benchmark_registry.normalize_market(market)

    def get_benchmark_symbol(self, market: str = "US") -> str:
        return self._benchmark_registry.get_primary_symbol(market)

    def get_benchmark_candidates(self, market: str = "US") -> list[str]:
        return self._benchmark_registry.get_candidate_symbols(market)

    def _redis_data_key(self, benchmark_symbol: str, period: str) -> str:
        return f"{self.REDIS_KEY_PREFIX}{benchmark_symbol}:{period}"

    def _redis_lock_key(self, benchmark_symbol: str, period: str) -> str:
        return f"{self._redis_data_key(benchmark_symbol, period)}{self.REDIS_LOCK_KEY_SUFFIX}"

    def get_spy_data(
        self,
        period: str = "2y",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Compatibility wrapper for legacy SPY callers.
        """
        return self.get_benchmark_data(
            market="US",
            period=period,
            force_refresh=force_refresh,
        )

    def get_benchmark_data(
        self,
        market: str = "US",
        period: str = "2y",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Get benchmark data with market-aware caching.

        Args:
            market: Market code (US, HK, JP, TW)
            period: Time period ('1y' or '2y')
            force_refresh: Force fetch from yfinance, bypass cache

        Returns:
            DataFrame with OHLCV data for market benchmark

        Logic:
        1. Check Redis for cached benchmark
        2. If miss, acquire distributed lock
        3. Fetch from yfinance (once!)
        4. Cache in Redis + persist to database
        5. Return to all waiting callers
        """
        normalized_market = self._normalize_market(market)
        candidates = self.get_benchmark_candidates(normalized_market)

        # Pass 1: Prefer any cached candidate before triggering network fetches.
        if not force_refresh:
            for idx, benchmark_symbol in enumerate(candidates):
                role = "primary" if idx == 0 else "fallback"
                cached_data = self._get_from_redis(benchmark_symbol=benchmark_symbol, period=period)
                if cached_data is not None:
                    logger.info(
                        "Cache HIT for %s benchmark %s (%s, %s) (Redis)",
                        normalized_market,
                        benchmark_symbol,
                        period,
                        role,
                    )
                    return cached_data

                cached_data = self._get_from_database(
                    benchmark_symbol=benchmark_symbol,
                    period=period,
                    market=normalized_market,
                )
                if cached_data is not None and self._is_data_fresh(cached_data, market=normalized_market):
                    logger.info(
                        "Cache HIT for %s benchmark %s (%s, %s) (Database)",
                        normalized_market,
                        benchmark_symbol,
                        period,
                        role,
                    )
                    self._store_in_redis(benchmark_symbol=benchmark_symbol, period=period, data=cached_data)
                    return cached_data

        # Pass 2: Fetch candidates in deterministic order.
        for idx, benchmark_symbol in enumerate(candidates):
            role = "primary" if idx == 0 else "fallback"
            if force_refresh:
                logger.info(
                    "Force refresh requested for %s benchmark %s (%s, %s)",
                    normalized_market,
                    benchmark_symbol,
                    period,
                    role,
                )
                fetched = self._fetch_and_cache_benchmark(
                    benchmark_symbol=benchmark_symbol,
                    market=normalized_market,
                    period=period,
                )
                if fetched is not None and not fetched.empty:
                    return fetched
                continue

            # Cache MISS - attempt fetch for this candidate
            logger.info(
                "Cache MISS for %s benchmark %s (%s, %s) - fetching from yfinance",
                normalized_market,
                benchmark_symbol,
                period,
                role,
            )
            fetched = self._fetch_and_cache_benchmark(
                benchmark_symbol=benchmark_symbol,
                market=normalized_market,
                period=period,
            )
            if fetched is not None and not fetched.empty:
                return fetched

        logger.warning("No benchmark candidate produced data for market=%s period=%s", normalized_market, period)
        return None

    def _get_from_redis(self, benchmark_symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get cached benchmark data from Redis."""
        if not self._redis_client:
            return None

        try:
            redis_key = self._redis_data_key(benchmark_symbol, period)
            cached_bytes = self._redis_client.get(redis_key)

            if cached_bytes:
                df = pickle.loads(cached_bytes)
                logger.debug("Retrieved benchmark %s %s from Redis (%s rows)", benchmark_symbol, period, len(df))
                return df

            return None

        except Exception as e:
            logger.warning(f"Error reading from Redis: {e}")
            return None

    def _get_from_database(
        self,
        benchmark_symbol: str,
        period: str,
        market: str = "US",
    ) -> Optional[pd.DataFrame]:
        """Get cached benchmark data from database."""
        db = self._session_factory()

        try:
            # Calculate date range
            try:
                end_date = self._market_calendar.market_now(market).date()
            except Exception:
                end_date = datetime.utcnow().date()

            if period == "2y":
                start_date = end_date - timedelta(days=730)  # ~2 years
            else:  # "1y"
                start_date = end_date - timedelta(days=365)

            # Query StockPrice table
            prices = db.query(StockPrice).filter(
                StockPrice.symbol == benchmark_symbol,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date.asc()).all()

            if not prices or len(prices) < 100:  # Need substantial data
                logger.debug("Insufficient data in database for benchmark %s %s", benchmark_symbol, period)
                return None

            # Convert to DataFrame
            data = {
                'Date': [p.date for p in prices],
                'Open': [p.open for p in prices],
                'High': [p.high for p in prices],
                'Low': [p.low for p in prices],
                'Close': [p.close for p in prices],
                'Volume': [p.volume for p in prices],
            }

            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            logger.debug("Retrieved benchmark %s %s from database (%s rows)", benchmark_symbol, period, len(df))
            return df

        except Exception as e:
            logger.error(f"Error reading from database: {e}", exc_info=True)
            return None

        finally:
            db.close()

    def _fetch_from_yfinance(self, benchmark_symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch benchmark data from yfinance."""
        # Import here to avoid circular dependency
        from .yfinance_service import YFinanceService

        yfinance_service = YFinanceService()
        return yfinance_service.get_historical_data(benchmark_symbol, period=period)

    def _fetch_and_cache_benchmark(self, benchmark_symbol: str, market: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch benchmark from yfinance and cache it.

        Uses distributed lock to prevent multiple workers from fetching simultaneously.
        """
        # No Redis means no distributed coordination is possible; fetch directly
        # and persist to DB so subsequent calls can still hit local cache.
        if not self._redis_client:
            benchmark_data = self._fetch_from_yfinance(benchmark_symbol, period)
            if benchmark_data is None or benchmark_data.empty:
                logger.error("Failed to fetch benchmark %s data from yfinance", benchmark_symbol)
                return None
            self._store_in_database(benchmark_symbol=benchmark_symbol, data=benchmark_data)
            return benchmark_data

        lock_key = self._redis_lock_key(benchmark_symbol, period)
        # Try to acquire lock
        lock_acquired = False
        if self._redis_client:
            try:
                lock_acquired = self._redis_client.set(
                    lock_key,
                    "locked",
                    nx=True,  # Only set if not exists
                    ex=self.LOCK_EXPIRY_SECONDS
                )
            except Exception as e:
                logger.warning(f"Error acquiring lock: {e}")

        if not lock_acquired:
            # Another worker is fetching - wait for it to complete
            logger.info("Another worker is fetching benchmark %s (%s) - waiting...", benchmark_symbol, period)
            return self._wait_for_cache(benchmark_symbol=benchmark_symbol, period=period)

        try:
            # We have the lock - fetch from yfinance
            logger.info("Fetching benchmark %s for market %s (%s) from yfinance", benchmark_symbol, market, period)

            benchmark_data = self._fetch_from_yfinance(benchmark_symbol, period)

            if benchmark_data is None or benchmark_data.empty:
                logger.error("Failed to fetch benchmark %s data from yfinance", benchmark_symbol)
                return None

            logger.info("Fetched benchmark %s %s: %s rows", benchmark_symbol, period, len(benchmark_data))

            # Cache in Redis
            self._store_in_redis(benchmark_symbol=benchmark_symbol, period=period, data=benchmark_data)

            # Persist to database
            self._store_in_database(benchmark_symbol=benchmark_symbol, data=benchmark_data)

            return benchmark_data

        finally:
            # Release lock
            if self._redis_client and lock_acquired:
                try:
                    self._redis_client.delete(lock_key)
                except Exception as e:
                    logger.warning(f"Error releasing lock: {e}")

    def _wait_for_cache(
        self,
        benchmark_symbol: str,
        period: str,
        max_wait_seconds: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Wait for another worker to cache benchmark data.

        Polls Redis every 0.5 seconds for cached data.
        """
        if max_wait_seconds is None:
            max_wait_seconds = self.LOCK_TIMEOUT_SECONDS

        start_time = time.time()

        while (time.time() - start_time) < max_wait_seconds:
            # Check if data is now in cache
            cached_data = self._get_from_redis(benchmark_symbol=benchmark_symbol, period=period)
            if cached_data is not None:
                logger.info("Benchmark %s %s is now cached by another worker", benchmark_symbol, period)
                return cached_data

            time.sleep(0.5)

        # Timeout - fetch directly as fallback
        logger.warning("Timeout waiting for benchmark %s %s cache - fetching directly", benchmark_symbol, period)
        benchmark_data = self._fetch_from_yfinance(benchmark_symbol, period)
        if benchmark_data is not None and not benchmark_data.empty:
            # Persist fallback fetch so future calls can use DB cache even when
            # lock-holder failed to populate Redis.
            self._store_in_database(benchmark_symbol=benchmark_symbol, data=benchmark_data)
        return benchmark_data

    def _store_in_redis(self, benchmark_symbol: str, period: str, data: pd.DataFrame) -> None:
        """Store benchmark data in Redis."""
        if not self._redis_client:
            return

        try:
            redis_key = self._redis_data_key(benchmark_symbol, period)
            pickled_data = pickle.dumps(data)

            self._redis_client.setex(
                redis_key,
                self.CACHE_TTL_SECONDS,
                pickled_data
            )

            logger.info("Cached benchmark %s %s in Redis (TTL: %ss)", benchmark_symbol, period, self.CACHE_TTL_SECONDS)

        except Exception as e:
            logger.error(f"Error storing in Redis: {e}", exc_info=True)

    def _store_in_database(self, benchmark_symbol: str, data: pd.DataFrame) -> None:
        """
        Store benchmark data in database (StockPrice table).

        Uses bulk insert for efficiency (same pattern as PriceCacheService).
        """
        db = self._session_factory()

        try:
            # Reset index to get Date as a column
            df = data.reset_index()

            # Fetch all existing dates for benchmark upfront (avoid N+1 queries)
            existing_dates_query = db.query(StockPrice.date).filter(
                StockPrice.symbol == benchmark_symbol
            )
            existing_dates = set([row[0] for row in existing_dates_query.all()])

            # Prepare bulk insert data - only for dates that don't exist
            rows_to_insert = []

            for _, row in df.iterrows():
                row_date = row['Date']

                # Convert pd.Timestamp to date for proper comparison
                if isinstance(row_date, pd.Timestamp):
                    row_date = row_date.date()
                elif isinstance(row_date, datetime):
                    row_date = row_date.date()

                # Skip if already exists
                if row_date in existing_dates:
                    continue

                # Prepare row for bulk insert
                try:
                    price_dict = {
                        'symbol': benchmark_symbol,
                        'date': row_date,
                        'open': float(row.get('Open', 0)),
                        'high': float(row.get('High', 0)),
                        'low': float(row.get('Low', 0)),
                        'close': float(row.get('Close', 0)),
                        'volume': int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else 0,
                        'adj_close': float(row.get('Close', 0))  # Use close as adj_close
                    }
                    rows_to_insert.append(price_dict)

                except Exception as e:
                    logger.warning("Error preparing benchmark %s row for %s: %s", benchmark_symbol, row.get("Date"), e)
                    continue

            # Bulk insert all new rows in one operation
            if rows_to_insert:
                db.bulk_insert_mappings(StockPrice, rows_to_insert)
                db.commit()
                logger.info("Bulk inserted %s new benchmark %s rows to database", len(rows_to_insert), benchmark_symbol)
            else:
                logger.debug("No new benchmark %s rows to persist", benchmark_symbol)

        except Exception as e:
            logger.error("Error storing benchmark %s in database: %s", benchmark_symbol, e, exc_info=True)
            db.rollback()

        finally:
            db.close()

    def _is_data_fresh(self, data: pd.DataFrame, market: str = "US", max_age_hours: int = 24) -> bool:
        """
        Check if cached benchmark data is still fresh using market-aware trading-day logic.

        US continues using legacy NYSE helper for parity. Non-US uses
        exchange_calendars where available and falls back to max-age checks.
        """
        if data is None or data.empty:
            return False

        try:
            last_date = data.index[-1]
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.Timestamp(last_date)

            normalized_market = self._normalize_market(market)
            expected = None
            try:
                expected = self._market_calendar.last_completed_trading_day(normalized_market)
            except Exception:
                expected = None
            if expected is None:
                # Calendar fallback: avoid weekend/holiday false-stale by allowing data
                # through the next business day when exchange calendar lookup is unavailable.
                return self._is_data_fresh_without_calendar(last_date, max_age_hours=max_age_hours)

            is_fresh = last_date.date() >= expected
            if not is_fresh:
                logger.debug(
                    "Benchmark data stale for market %s (last: %s, expected: %s)",
                    normalized_market,
                    last_date.date(),
                    expected,
                )
            return is_fresh

        except Exception as e:
            logger.warning(f"Error checking data freshness: {e}")
            return False

    @staticmethod
    def _is_data_fresh_without_calendar(last_date: pd.Timestamp, max_age_hours: int = 24) -> bool:
        """Fallback freshness policy when exchange calendar is unavailable."""
        last_ts = pd.Timestamp(last_date)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")

        now_utc = pd.Timestamp.utcnow()
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")

        latest_allowed = now_utc - pd.Timedelta(hours=max_age_hours)
        if last_ts >= latest_allowed:
            return True

        # Allow Friday (or latest business-day) data to remain fresh until the end of
        # the next business day when calendars are unavailable (e.g., lookup failure).
        business_days_after_last = pd.bdate_range(
            start=last_ts.date() + timedelta(days=1),
            end=now_utc.date(),
        )
        return len(business_days_after_last) <= 1

    def invalidate_cache(self, period: str = None) -> None:
        """
        Invalidate cached benchmark data.

        Args:
            period: Specific period to invalidate, or None for all
        """
        if not self._redis_client:
            logger.warning("Redis not available for cache invalidation")
            return

        try:
            keys: list[str]
            symbols = set()
            for market in self._benchmark_registry.supported_markets():
                symbols.update(self._benchmark_registry.get_candidate_symbols(market))
            if period:
                keys = [
                    self._redis_data_key(symbol, period)
                    for symbol in symbols
                ]
            else:
                keys = []
                for symbol in symbols:
                    keys.append(self._redis_data_key(symbol, "1y"))
                    keys.append(self._redis_data_key(symbol, "2y"))

            if keys:
                self._redis_client.delete(*keys)
            if period:
                logger.info("Invalidated all benchmark %s cache keys", period)
            else:
                logger.info("Invalidated all benchmark cache keys")

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}", exc_info=True)

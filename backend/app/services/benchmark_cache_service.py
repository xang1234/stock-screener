"""
Benchmark Cache Service for SPY data caching.

Provides singleton caching of SPY benchmark data to eliminate redundant API calls
during bulk scans. Uses Redis for hot cache with database persistence.
"""
import logging
import pickle
import time
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import redis
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models.stock import StockPrice
from ..config import settings
from .redis_pool import get_redis_client
from ..utils.market_hours import get_eastern_now, is_trading_day, is_market_open, get_last_trading_day

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

    # Class-level singleton instance
    _instance = None
    _redis_client = None

    # Redis keys
    REDIS_KEY_PREFIX = "benchmark:"
    REDIS_KEY_SPY_2Y = "benchmark:SPY:2y"
    REDIS_KEY_SPY_1Y = "benchmark:SPY:1y"
    REDIS_LOCK_KEY = "benchmark:SPY:lock"

    # TTL: expires at market close + 1 hour buffer
    CACHE_TTL_SECONDS = 86400  # 24 hours
    LOCK_TIMEOUT_SECONDS = 10  # Max time to wait for lock
    LOCK_EXPIRY_SECONDS = 30  # Lock auto-expires after 30 seconds

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize benchmark cache service."""
        if redis_client:
            self._redis_client = redis_client
        else:
            # Use shared connection pool for efficiency
            self._redis_client = get_redis_client()
            if self._redis_client:
                logger.info("Connected to Redis for benchmark caching (using shared pool)")
            else:
                logger.warning("Redis connection failed. Will use database fallback.")

    @classmethod
    def get_instance(cls, redis_client: Optional[redis.Redis] = None):
        """
        Get singleton instance of BenchmarkCacheService.

        Thread-safe singleton pattern.
        """
        if cls._instance is None:
            cls._instance = cls(redis_client)
        return cls._instance

    def get_spy_data(
        self,
        period: str = "2y",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get SPY benchmark data with caching.

        Args:
            period: Time period ('1y' or '2y')
            force_refresh: Force fetch from yfinance, bypass cache

        Returns:
            DataFrame with OHLCV data for SPY

        Logic:
        1. Check Redis for cached SPY
        2. If miss, acquire distributed lock
        3. Fetch from yfinance (once!)
        4. Cache in Redis + persist to database
        5. Return to all waiting callers
        """
        if force_refresh:
            logger.info(f"Force refresh requested for SPY {period}")
            return self._fetch_and_cache_spy(period)

        # Try Redis cache first
        cached_data = self._get_from_redis(period)
        if cached_data is not None:
            logger.info(f"Cache HIT for SPY {period} (Redis)")
            return cached_data

        # Try database cache as fallback
        cached_data = self._get_from_database(period)
        if cached_data is not None and self._is_data_fresh(cached_data):
            logger.info(f"Cache HIT for SPY {period} (Database)")
            # Store in Redis for next time
            self._store_in_redis(period, cached_data)
            return cached_data

        # Cache MISS - need to fetch from yfinance
        logger.info(f"Cache MISS for SPY {period} - fetching from yfinance")
        return self._fetch_and_cache_spy(period)

    def _get_from_redis(self, period: str) -> Optional[pd.DataFrame]:
        """Get cached SPY data from Redis."""
        if not self._redis_client:
            return None

        try:
            redis_key = self.REDIS_KEY_SPY_2Y if period == "2y" else self.REDIS_KEY_SPY_1Y
            cached_bytes = self._redis_client.get(redis_key)

            if cached_bytes:
                df = pickle.loads(cached_bytes)
                logger.debug(f"Retrieved SPY {period} from Redis ({len(df)} rows)")
                return df

            return None

        except Exception as e:
            logger.warning(f"Error reading from Redis: {e}")
            return None

    def _get_from_database(self, period: str) -> Optional[pd.DataFrame]:
        """Get cached SPY data from database."""
        db = SessionLocal()

        try:
            # Calculate date range
            end_date = get_eastern_now().date()

            if period == "2y":
                start_date = end_date - timedelta(days=730)  # ~2 years
            else:  # "1y"
                start_date = end_date - timedelta(days=365)

            # Query StockPrice table
            prices = db.query(StockPrice).filter(
                StockPrice.symbol == "SPY",
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date.asc()).all()

            if not prices or len(prices) < 100:  # Need substantial data
                logger.debug(f"Insufficient data in database for SPY {period}")
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
            df.set_index('Date', inplace=True)

            logger.debug(f"Retrieved SPY {period} from database ({len(df)} rows)")
            return df

        except Exception as e:
            logger.error(f"Error reading from database: {e}", exc_info=True)
            return None

        finally:
            db.close()

    def _fetch_and_cache_spy(self, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch SPY from yfinance and cache it.

        Uses distributed lock to prevent multiple workers from fetching simultaneously.
        """
        # Try to acquire lock
        lock_acquired = False
        if self._redis_client:
            try:
                lock_acquired = self._redis_client.set(
                    self.REDIS_LOCK_KEY,
                    "locked",
                    nx=True,  # Only set if not exists
                    ex=self.LOCK_EXPIRY_SECONDS
                )
            except Exception as e:
                logger.warning(f"Error acquiring lock: {e}")

        if not lock_acquired:
            # Another worker is fetching - wait for it to complete
            logger.info("Another worker is fetching SPY - waiting...")
            return self._wait_for_cache(period)

        try:
            # We have the lock - fetch from yfinance
            logger.info(f"Fetching SPY {period} from yfinance")

            # Import here to avoid circular dependency
            from .yfinance_service import YFinanceService

            yfinance_service = YFinanceService()
            spy_data = yfinance_service.get_historical_data("SPY", period=period)

            if spy_data is None or spy_data.empty:
                logger.error("Failed to fetch SPY data from yfinance")
                return None

            logger.info(f"Fetched SPY {period}: {len(spy_data)} rows")

            # Cache in Redis
            self._store_in_redis(period, spy_data)

            # Persist to database
            self._store_in_database(spy_data)

            return spy_data

        finally:
            # Release lock
            if self._redis_client and lock_acquired:
                try:
                    self._redis_client.delete(self.REDIS_LOCK_KEY)
                except Exception as e:
                    logger.warning(f"Error releasing lock: {e}")

    def _wait_for_cache(self, period: str, max_wait_seconds: int = None) -> Optional[pd.DataFrame]:
        """
        Wait for another worker to cache SPY data.

        Polls Redis every 0.5 seconds for cached data.
        """
        if max_wait_seconds is None:
            max_wait_seconds = self.LOCK_TIMEOUT_SECONDS

        start_time = time.time()

        while (time.time() - start_time) < max_wait_seconds:
            # Check if data is now in cache
            cached_data = self._get_from_redis(period)
            if cached_data is not None:
                logger.info(f"SPY {period} is now cached by another worker")
                return cached_data

            time.sleep(0.5)

        # Timeout - fetch directly as fallback
        logger.warning(f"Timeout waiting for SPY {period} cache - fetching directly")
        from .yfinance_service import YFinanceService
        yfinance_service = YFinanceService()
        return yfinance_service.get_historical_data("SPY", period=period)

    def _store_in_redis(self, period: str, data: pd.DataFrame) -> None:
        """Store SPY data in Redis."""
        if not self._redis_client:
            return

        try:
            redis_key = self.REDIS_KEY_SPY_2Y if period == "2y" else self.REDIS_KEY_SPY_1Y
            pickled_data = pickle.dumps(data)

            self._redis_client.setex(
                redis_key,
                self.CACHE_TTL_SECONDS,
                pickled_data
            )

            logger.info(f"Cached SPY {period} in Redis (TTL: {self.CACHE_TTL_SECONDS}s)")

        except Exception as e:
            logger.error(f"Error storing in Redis: {e}", exc_info=True)

    def _store_in_database(self, data: pd.DataFrame) -> None:
        """
        Store SPY data in database (StockPrice table).

        Uses bulk insert for efficiency (same pattern as PriceCacheService).
        """
        db = SessionLocal()

        try:
            # Reset index to get Date as a column
            df = data.reset_index()

            # Fetch all existing dates for SPY upfront (avoid N+1 queries)
            existing_dates_query = db.query(StockPrice.date).filter(
                StockPrice.symbol == "SPY"
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
                        'symbol': "SPY",
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
                    logger.warning(f"Error preparing SPY row for {row.get('Date')}: {e}")
                    continue

            # Bulk insert all new rows in one operation
            if rows_to_insert:
                db.bulk_insert_mappings(StockPrice, rows_to_insert)
                db.commit()
                logger.info(f"Bulk inserted {len(rows_to_insert)} new SPY rows to database")
            else:
                logger.debug("No new SPY rows to persist")

        except Exception as e:
            logger.error(f"Error storing SPY in database: {e}", exc_info=True)
            db.rollback()

        finally:
            db.close()

    def _is_data_fresh(self, data: pd.DataFrame, max_age_hours: int = 24) -> bool:
        """
        Check if cached SPY data is still fresh using trading-day awareness.

        Uses market calendar to determine expected data date instead of
        naive timedelta checks that break on holidays and weekends.
        """
        if data is None or data.empty:
            return False

        try:
            last_date = data.index[-1]
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.Timestamp(last_date)

            now_et = get_eastern_now()
            today = now_et.date()

            # Determine what date we should have data for
            if is_trading_day(today) and not is_market_open(now_et) and now_et.hour >= 17:
                # After 5 PM on trading day: expect today's close
                expected = today
            else:
                # During market hours, before open, grace period, weekend, holiday:
                # last completed trading day's data is sufficient
                expected = get_last_trading_day(today - timedelta(days=1))

            is_fresh = last_date.date() >= expected
            if not is_fresh:
                logger.debug(f"SPY data stale (last: {last_date.date()}, expected: {expected})")
            return is_fresh

        except Exception as e:
            logger.warning(f"Error checking data freshness: {e}")
            return False

    def invalidate_cache(self, period: str = None) -> None:
        """
        Invalidate cached SPY data.

        Args:
            period: Specific period to invalidate, or None for all
        """
        if not self._redis_client:
            logger.warning("Redis not available for cache invalidation")
            return

        try:
            if period:
                redis_key = self.REDIS_KEY_SPY_2Y if period == "2y" else self.REDIS_KEY_SPY_1Y
                self._redis_client.delete(redis_key)
                logger.info(f"Invalidated SPY {period} cache")
            else:
                # Invalidate all
                self._redis_client.delete(self.REDIS_KEY_SPY_2Y)
                self._redis_client.delete(self.REDIS_KEY_SPY_1Y)
                logger.info("Invalidated all SPY cache")

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}", exc_info=True)

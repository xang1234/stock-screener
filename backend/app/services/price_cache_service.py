"""
Price Cache Service for stock OHLCV data caching.

Provides intelligent caching of stock price data with incremental updates
to minimize API calls. Uses Redis for hot cache and database for persistence.

Includes intraday staleness detection to handle data fetched during market
hours that becomes stale after market close.

Canonical price contract ADR:
docs/learning_loop/adr_ll2_e1_canonical_price_contract_v1.md
"""
import json
import logging
import pickle
from typing import Optional, Dict, List
from datetime import datetime, timedelta, date, time
import pandas as pd
import redis
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models.stock import StockPrice
from ..config import settings
from ..utils.market_hours import (
    is_market_open, get_eastern_now, EASTERN, MARKET_CLOSE_TIME,
    is_trading_day, get_last_trading_day
)
from .redis_pool import get_redis_client, get_bulk_redis_client

logger = logging.getLogger(__name__)

# Redis keys for warmup metadata
WARMUP_METADATA_KEY = "cache:warmup:metadata"
WARMUP_HEARTBEAT_KEY = "cache:warmup:heartbeat"


class PriceCacheService:
    """
    Service for caching stock price data with incremental updates.

    Strategy:
    - Store recent data (last 30 days) in Redis for fast access
    - Store full historical data in StockPrice table
    - Fetch only missing data (incremental updates)
    - Merge cached + new data on retrieval
    """

    # Class-level singleton instance
    _instance = None
    _redis_client = None

    # Redis keys
    REDIS_KEY_PREFIX = "price:"
    REDIS_KEY_RECENT = "price:{symbol}:recent"
    REDIS_KEY_LAST_UPDATE = "price:{symbol}:last_update"
    REDIS_KEY_FETCH_META = "price:{symbol}:fetch_meta"

    # TTL settings
    CACHE_TTL_SECONDS = 604800  # 7 days (aligned with config.cache_ttl_seconds)
    RECENT_DAYS = 1825  # Keep last 5 years in Redis (for 5-year volume analysis)

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize price cache service."""
        if redis_client:
            self._redis_client = redis_client
        else:
            # Use shared connection pool for efficiency
            self._redis_client = get_redis_client()
            if self._redis_client:
                logger.debug("Connected to Redis for price caching (using shared pool)")
            else:
                logger.warning("Redis connection failed. Will use database fallback.")

    @classmethod
    def get_instance(cls, redis_client: Optional[redis.Redis] = None):
        """
        Get singleton instance of PriceCacheService.

        Thread-safe singleton pattern.
        """
        if cls._instance is None:
            cls._instance = cls(redis_client)
        return cls._instance

    def get_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data with caching and incremental updates.

        Args:
            symbol: Stock ticker symbol
            period: Time period ('1y' or '2y')
            force_refresh: Force fetch from yfinance, bypass cache

        Returns:
            DataFrame with OHLCV data

        Logic:
        1. Check database for cached data
        2. Determine how old the data is
        3. If fresh enough, return from cache
        4. If stale, fetch only missing dates (incremental!)
        5. Merge old + new data
        6. Update cache
        """
        if force_refresh:
            logger.info(f"Force refresh requested for {symbol}")
            return self._fetch_full_and_cache(symbol, period)

        # Try to get cached data from database
        cached_data, last_date = self._get_from_database(symbol, period)

        if cached_data is not None and not cached_data.empty:
            # Check if data is fresh
            if self._is_data_fresh(last_date):
                logger.info(f"Cache HIT for {symbol} (Database, last: {last_date})")

                # Also store in Redis for faster next access
                self._store_recent_in_redis(symbol, cached_data)

                return cached_data
            else:
                # Data is stale - fetch incremental update
                logger.info(f"Cache HIT but STALE for {symbol} (last: {last_date}) - fetching incremental")
                return self._fetch_incremental_and_merge(symbol, period, cached_data, last_date)

        # No cached data - fetch full history
        logger.info(f"Cache MISS for {symbol} - fetching full history")
        return self._fetch_full_and_cache(symbol, period)

    def get_cached_only(
        self,
        symbol: str,
        period: str = "2y"
    ) -> Optional[pd.DataFrame]:
        """
        Get price data from cache ONLY - does NOT fetch from Yahoo if missing.

        Use this for operations that should only use existing cached data,
        such as signal detection where we don't want to trigger API calls.

        Args:
            symbol: Stock ticker symbol
            period: Time period ('1y', '2y', '5y')

        Returns:
            DataFrame with OHLCV data if cached, None otherwise
        """
        cached_data, last_date = self._get_from_database(symbol, period)
        if cached_data is not None and not cached_data.empty:
            logger.debug(f"Cache-only HIT for {symbol} (last: {last_date})")
            return cached_data
        logger.debug(f"Cache-only MISS for {symbol}")
        return None

    def get_many_cached_only(
        self,
        symbols: List[str],
        period: str = "2y"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get price data for multiple symbols from cache ONLY.

        Efficient bulk operation that doesn't trigger Yahoo API calls.

        Args:
            symbols: List of stock symbols
            period: Time period ('1y', '2y', '5y')

        Returns:
            Dict mapping symbol to DataFrame (or None if not cached)
        """
        results = self._get_many_from_database(symbols, period)
        return {symbol: data for symbol, (data, _) in results.items()}

    def _get_from_database(self, symbol: str, period: str) -> tuple[Optional[pd.DataFrame], Optional[date]]:
        """
        Get cached price data from database.

        Returns:
            Tuple of (DataFrame, last_date) or (None, None)
        """
        db = SessionLocal()

        try:
            # Calculate date range
            end_date = datetime.now().date()

            # Support multiple periods with backward compatibility
            period_days_map = {
                "5y": 1825,  # 5 years
                "2y": 730,   # 2 years
                "1y": 365,   # 1 year
                "max": 3650  # 10 years for max
            }
            days = period_days_map.get(period, 730)  # Default to 2y for backward compat
            start_date = end_date - timedelta(days=days)

            # Query StockPrice table
            prices = db.query(StockPrice).filter(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date.asc()).all()

            if not prices or len(prices) < 50:  # Need substantial data
                logger.debug(f"Insufficient cached data for {symbol} ({len(prices) if prices else 0} rows)")
                return None, None

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
            # Convert Date to pd.Timestamp for consistency with yfinance data
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Get last date
            last_date = prices[-1].date

            logger.debug(f"Retrieved {symbol} from database ({len(df)} rows, last: {last_date})")
            return df, last_date

        except Exception as e:
            logger.error(f"Error reading {symbol} from database: {e}", exc_info=True)
            return None, None

        finally:
            db.close()

    def _get_many_from_database(
        self,
        symbols: list[str],
        period: str
    ) -> Dict[str, tuple[Optional[pd.DataFrame], Optional[date]]]:
        """
        Bulk fetch from database for multiple symbols.

        More efficient than calling _get_from_database() repeatedly
        because it uses a single DB session for all queries.

        Args:
            symbols: List of stock symbols to fetch
            period: Time period ("1y", "2y", "5y", "max")

        Returns:
            Dict mapping symbol to (DataFrame, last_date) or (None, None)
        """
        if not symbols:
            return {}

        db = SessionLocal()
        results = {}

        try:
            # Calculate date range
            end_date = datetime.now().date()
            period_days_map = {
                "5y": 1825,
                "2y": 730,
                "1y": 365,
                "max": 3650
            }
            days = period_days_map.get(period, 730)
            start_date = end_date - timedelta(days=days)

            # Query all symbols at once using IN clause
            from sqlalchemy import and_
            all_prices = db.query(StockPrice).filter(
                and_(
                    StockPrice.symbol.in_(symbols),
                    StockPrice.date >= start_date,
                    StockPrice.date <= end_date
                )
            ).order_by(StockPrice.symbol, StockPrice.date.asc()).all()

            # Group by symbol
            symbol_prices = {}
            for price in all_prices:
                if price.symbol not in symbol_prices:
                    symbol_prices[price.symbol] = []
                symbol_prices[price.symbol].append(price)

            # Convert each symbol's prices to DataFrame
            for symbol in symbols:
                prices = symbol_prices.get(symbol, [])

                if not prices or len(prices) < 50:
                    results[symbol] = (None, None)
                    continue

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

                last_date = prices[-1].date
                results[symbol] = (df, last_date)

            logger.debug(f"Bulk DB query: {len([r for r in results.values() if r[0] is not None])} hits, "
                        f"{len([r for r in results.values() if r[0] is None])} misses")

            return results

        except Exception as e:
            logger.error(f"Error in bulk database query: {e}", exc_info=True)
            return {symbol: (None, None) for symbol in symbols}

        finally:
            db.close()

    def _fetch_full_and_cache(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch full historical data from yfinance and cache it.
        """
        try:
            # Import here to avoid circular dependency
            from .yfinance_service import YFinanceService

            yfinance_service = YFinanceService()
            # IMPORTANT: use_cache=False to avoid circular dependency
            data = yfinance_service.get_historical_data(symbol, period=period, use_cache=False)

            if data is None or data.empty:
                logger.warning(f"Failed to fetch data for {symbol}")
                return None

            logger.info(f"Fetched {symbol}: {len(data)} rows")

            # Cache in Redis (recent data only)
            self._store_recent_in_redis(symbol, data)

            # Persist to database (full data)
            self._store_in_database(symbol, data)

            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}", exc_info=True)
            return None

    def _fetch_incremental_and_merge(
        self,
        symbol: str,
        period: str,
        cached_data: pd.DataFrame,
        last_cached_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Fetch only new data since last_cached_date and merge with cached data.

        This is the key optimization - instead of fetching 2 years of data,
        we only fetch the missing days!
        """
        try:
            from .yfinance_service import YFinanceService

            # Calculate how many days we're missing
            today = datetime.now().date()
            days_missing = (today - last_cached_date).days

            if days_missing <= 0:
                logger.info(f"{symbol} cache is current (last: {last_cached_date})")
                return cached_data

            logger.info(f"{symbol} is {days_missing} days old - fetching incremental update")

            # Fetch only recent data (last 7 days to ensure overlap)
            yfinance_service = YFinanceService()
            # IMPORTANT: use_cache=False to avoid circular dependency
            new_data = yfinance_service.get_historical_data(symbol, period="7d", use_cache=False)

            if new_data is None or new_data.empty:
                logger.warning(f"Failed to fetch incremental data for {symbol}")
                return cached_data  # Return stale cache as fallback

            # Filter new_data to only dates after last_cached_date
            # Ensure timezone compatibility for comparison
            last_cached_ts = pd.Timestamp(last_cached_date)
            if new_data.index.tz is not None and last_cached_ts.tz is None:
                last_cached_ts = last_cached_ts.tz_localize(new_data.index.tz)
            new_data_filtered = new_data[new_data.index > last_cached_ts]

            if new_data_filtered.empty:
                logger.info(f"No new data available for {symbol}")
                return cached_data

            logger.info(f"Fetched {len(new_data_filtered)} new rows for {symbol}")

            # Convert cached_data index to pd.Timestamp to match new_data
            # (cached_data from DB has datetime.date index, new_data has pd.Timestamp)
            if not isinstance(cached_data.index, pd.DatetimeIndex):
                cached_data.index = pd.to_datetime(cached_data.index)

            # Ensure both are timezone-naive for consistent merging
            # Remove timezone info from both DataFrames to avoid comparison errors
            if cached_data.index.tz is not None:
                cached_data.index = cached_data.index.tz_localize(None)
            if new_data_filtered.index.tz is not None:
                new_data_filtered.index = new_data_filtered.index.tz_localize(None)

            # Merge: concatenate and remove duplicates
            merged_data = pd.concat([cached_data, new_data_filtered])
            merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
            merged_data = merged_data.sort_index()

            # Trim to requested period
            period_days_map = {
                "5y": 1825, "2y": 730, "1y": 365, "max": 3650
            }
            days = period_days_map.get(period, 730)
            cutoff_date = today - timedelta(days=days)

            merged_data = merged_data[merged_data.index >= pd.Timestamp(cutoff_date)]

            logger.info(f"Merged data for {symbol}: {len(merged_data)} total rows")

            # Update cache with merged data
            self._store_recent_in_redis(symbol, merged_data)
            self._store_in_database(symbol, new_data_filtered)  # Only persist new rows

            return merged_data

        except Exception as e:
            logger.error(f"Error fetching incremental data for {symbol}: {e}", exc_info=True)
            return cached_data  # Return stale cache as fallback

    def _store_recent_in_redis(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Store historical data (up to 5 years) in Redis for fast access.

        Stores full 5-year data to support volume breakthrough analysis
        and Minervini 200-day MA calculations without requiring database fallback.

        Also stores fetch metadata for intraday staleness detection.
        """
        if not self._redis_client:
            return

        try:
            # Keep last 5 years (1825 days) for volume analysis
            cutoff_datetime = datetime.now() - timedelta(days=self.RECENT_DAYS)
            # Convert to pd.Timestamp to ensure compatibility with pandas index
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                cutoff_date = pd.Timestamp(cutoff_datetime, tz=data.index.tz)
            else:
                cutoff_date = pd.Timestamp(cutoff_datetime)

            recent_data = data[data.index >= cutoff_date]

            if recent_data.empty:
                return

            redis_key = self.REDIS_KEY_RECENT.format(symbol=symbol)
            pickled_data = pickle.dumps(recent_data)

            self._redis_client.setex(
                redis_key,
                self.CACHE_TTL_SECONDS,
                pickled_data
            )

            # Also store last update timestamp
            last_update_key = self.REDIS_KEY_LAST_UPDATE.format(symbol=symbol)
            last_date = recent_data.index[-1].strftime('%Y-%m-%d')
            self._redis_client.setex(
                last_update_key,
                self.CACHE_TTL_SECONDS,
                last_date
            )

            # Store fetch metadata for intraday staleness detection
            self._store_fetch_metadata(symbol)

            logger.debug(f"Cached {symbol} recent data in Redis ({len(recent_data)} rows)")

        except Exception as e:
            logger.error(f"Error storing {symbol} in Redis: {e}", exc_info=True)

    def _store_fetch_metadata(self, symbol: str) -> None:
        """
        Store metadata about when data was fetched for staleness detection.

        Tracks:
        - fetch_timestamp: When data was fetched
        - market_was_open: Whether market was open at fetch time
        - data_type: 'intraday' if fetched during market hours, 'closing' otherwise
        - needs_refresh_after_close: True if this is intraday data
        """
        if not self._redis_client:
            return

        try:
            now_et = get_eastern_now()
            market_open = is_market_open(now_et)

            # Determine if this is intraday or closing data
            # Data fetched during market hours is intraday (needs refresh after close)
            # Data fetched after 4:30 PM ET is considered closing data
            post_close_buffer = time(16, 30)  # 4:30 PM ET
            is_after_close = now_et.time() >= post_close_buffer

            data_type = "intraday" if market_open else "closing"
            needs_refresh = market_open  # Only intraday data needs refresh

            fetch_meta = {
                "fetch_timestamp": now_et.isoformat(),
                "market_was_open": market_open,
                "data_type": data_type,
                "needs_refresh_after_close": needs_refresh
            }

            meta_key = self.REDIS_KEY_FETCH_META.format(symbol=symbol)
            self._redis_client.setex(
                meta_key,
                self.CACHE_TTL_SECONDS,
                json.dumps(fetch_meta)
            )

            if needs_refresh:
                logger.debug(f"Stored fetch metadata for {symbol}: intraday data, needs refresh after close")

        except Exception as e:
            logger.error(f"Error storing fetch metadata for {symbol}: {e}", exc_info=True)

    def _get_fetch_metadata(self, symbol: str) -> Optional[Dict]:
        """
        Get fetch metadata for a symbol.

        Returns:
            Dict with fetch metadata or None if not found
        """
        if not self._redis_client:
            return None

        try:
            meta_key = self.REDIS_KEY_FETCH_META.format(symbol=symbol)
            meta_json = self._redis_client.get(meta_key)

            if meta_json:
                return json.loads(meta_json)
            return None

        except Exception as e:
            logger.error(f"Error getting fetch metadata for {symbol}: {e}", exc_info=True)
            return None

    def _is_intraday_data_stale(self, symbol: str) -> bool:
        """
        Check if cached data is stale intraday data.

        Returns True if:
        - Data was fetched during market hours AND
        - Market is now closed (past 4:30 PM ET)

        This catches the case where data was fetched at 2 PM with an
        incomplete "today" bar, but user is now scanning at 6 PM.
        """
        meta = self._get_fetch_metadata(symbol)

        if not meta:
            return False  # No metadata - can't determine staleness

        # If data was marked as needing refresh after close
        if not meta.get("needs_refresh_after_close", False):
            return False  # Already closing data, not stale

        # Check if market is now closed
        now_et = get_eastern_now()
        market_open = is_market_open(now_et)

        if market_open:
            return False  # Market still open, data is current intraday

        # Market is closed - check if we're past the close buffer (4:30 PM ET)
        post_close_buffer = time(16, 30)
        if now_et.time() >= post_close_buffer:
            # We're after market close, and data was fetched during market hours
            logger.debug(f"{symbol}: intraday data is stale (fetched during market, now after close)")
            return True

        return False

    def get_stale_intraday_symbols(self) -> List[str]:
        """
        Scan Redis for all symbols with stale intraday data.

        Uses pipeline to batch-read all fetch_meta values after SCAN,
        reducing from ~N individual GETs to 1 pipeline round-trip.

        Returns:
            List of symbols that have stale intraday data
        """
        if not self._redis_client:
            return []

        try:
            # Pre-compute market state once (not per-symbol)
            now_et = get_eastern_now()
            market_open = is_market_open(now_et)

            # If market is still open, no data is "stale" yet
            if market_open:
                return []

            # Check if we're past the close buffer
            post_close_buffer = time(16, 30)
            if now_et.time() < post_close_buffer:
                return []

            # Collect all fetch_meta keys via SCAN
            pattern = "price:*:fetch_meta"
            all_keys = []
            all_symbols = []
            cursor = 0

            while True:
                cursor, keys = self._redis_client.scan(cursor, match=pattern, count=500)
                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    parts = key_str.split(':')
                    if len(parts) == 3:
                        all_keys.append(key)
                        all_symbols.append(parts[1])
                if cursor == 0:
                    break

            if not all_keys:
                return []

            # Pipeline-read all fetch_meta values at once
            pipeline = self._redis_client.pipeline()
            for key in all_keys:
                pipeline.get(key)
            meta_values = pipeline.execute()

            # Filter locally for stale intraday data
            stale_symbols = []
            for symbol, meta_json in zip(all_symbols, meta_values):
                if not meta_json:
                    continue
                try:
                    meta = json.loads(meta_json)
                    if meta.get("needs_refresh_after_close", False):
                        stale_symbols.append(symbol)
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.info(f"Found {len(stale_symbols)} symbols with stale intraday data")
            return stale_symbols

        except Exception as e:
            logger.error(f"Error scanning for stale intraday symbols: {e}", exc_info=True)
            return []

    def get_staleness_status(self) -> Dict:
        """
        Get overall staleness status for the cache.

        Returns:
            Dict with staleness info including count and market status
        """
        now_et = get_eastern_now()
        market_open = is_market_open(now_et)

        # Get stale symbols
        stale_symbols = self.get_stale_intraday_symbols()

        return {
            "stale_intraday_count": len(stale_symbols),
            "stale_symbols": stale_symbols[:10],  # Return first 10 for display
            "market_is_open": market_open,
            "current_time_et": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
            "has_stale_data": len(stale_symbols) > 0
        }

    def get_cache_health_status(self) -> Dict:
        """
        O(1) cache health check using SPY as proxy.

        Uses SPY benchmark as the health indicator because:
        1. SPY is always the first thing warmed in every cache refresh
        2. If SPY is fresh, the warmup task ran successfully
        3. Single Redis lookup vs scanning thousands of symbols

        Returns 6 possible states:
        - fresh: Cache is up to date (SPY has expected date + last warmup complete)
        - updating: Refresh task is currently running
        - stuck: Task running but no progress for >30 minutes
        - partial: Last warmup incomplete (some symbols failed)
        - stale: SPY missing expected trading date
        - error: Redis unavailable or other error

        Returns:
            Dict with:
            - status: "fresh"|"updating"|"stuck"|"partial"|"stale"|"error"
            - spy_last_date: Last date in SPY data
            - expected_date: Date cache should have
            - message: Human-readable explanation
            - can_refresh: Whether refresh is allowed
            - task_running: Task info if updating
            - last_warmup: Warmup metadata if available
        """
        try:
            # Check Redis connectivity
            if not self._redis_client:
                return {
                    "status": "error",
                    "message": "Cache unavailable - Redis not connected",
                    "can_refresh": False,
                    "spy_last_date": None,
                    "expected_date": None,
                    "task_running": None,
                    "last_warmup": None
                }

            try:
                self._redis_client.ping()
            except Exception as e:
                logger.error(f"Redis ping failed: {e}")
                return {
                    "status": "error",
                    "message": "Cache unavailable - Redis connection failed",
                    "can_refresh": False,
                    "spy_last_date": None,
                    "expected_date": None,
                    "task_running": None,
                    "last_warmup": None
                }

            # Check if a refresh task is currently running
            from ..tasks.data_fetch_lock import DataFetchLock
            lock = DataFetchLock.get_instance()
            current_holder = lock.get_current_holder()

            if current_holder and current_holder.get('task_name'):
                # Lock is held — check heartbeat to determine state
                hb_info = self._get_heartbeat_info()

                if hb_info and hb_info.get('status') in ('completed', 'failed'):
                    # Task finished but lock not yet released (brief race window).
                    # Force-release stale lock and fall through to SPY freshness check.
                    logger.info(
                        f"Task heartbeat is terminal ({hb_info['status']}), "
                        f"force-releasing stale lock"
                    )
                    lock.force_release()
                    # Fall through to SPY freshness check below

                elif hb_info and hb_info.get('status') == 'running':
                    minutes = hb_info.get('minutes')
                    if minutes is not None and minutes > 30:
                        # Running heartbeat but no progress for >30 min → stuck
                        return {
                            "status": "stuck",
                            "message": f"Task appears stuck (no progress for {int(minutes)} min)",
                            "can_refresh": True,
                            "can_force_cancel": True,
                            "spy_last_date": None,
                            "expected_date": None,
                            "task_running": {
                                "task_id": current_holder.get('task_id'),
                                "task_name": current_holder.get('task_name'),
                                "started_at": current_holder.get('started_at'),
                                "minutes_since_heartbeat": int(minutes)
                            },
                            "last_warmup": self._get_warmup_metadata()
                        }
                    else:
                        # Task is actively running with recent heartbeat
                        return {
                            "status": "updating",
                            "message": f"Cache refresh in progress ({current_holder.get('task_name')})",
                            "can_refresh": False,
                            "spy_last_date": None,
                            "expected_date": None,
                            "task_running": {
                                "task_id": current_holder.get('task_id'),
                                "task_name": current_holder.get('task_name'),
                                "started_at": current_holder.get('started_at'),
                                **self._get_task_progress()
                            },
                            "last_warmup": self._get_warmup_metadata()
                        }

                else:
                    # No heartbeat at all — use lock-age grace period
                    started_at_str = current_holder.get('started_at')
                    lock_age_minutes = None
                    if started_at_str:
                        try:
                            started_at = datetime.fromisoformat(started_at_str)
                            lock_age_minutes = (datetime.now() - started_at).total_seconds() / 60
                        except (ValueError, TypeError):
                            pass

                    if lock_age_minutes is not None and lock_age_minutes < 2:
                        # Lock acquired < 2 min ago, task is initializing
                        return {
                            "status": "updating",
                            "message": f"Cache refresh starting ({current_holder.get('task_name')})",
                            "can_refresh": False,
                            "spy_last_date": None,
                            "expected_date": None,
                            "task_running": {
                                "task_id": current_holder.get('task_id'),
                                "task_name": current_holder.get('task_name'),
                                "started_at": started_at_str,
                            },
                            "last_warmup": self._get_warmup_metadata()
                        }

                    # Lock held >= 2 min with no heartbeat.
                    # Check if warmup metadata shows completion after lock was acquired.
                    warmup_meta = self._get_warmup_metadata()
                    if warmup_meta and warmup_meta.get('completed_at') and started_at_str:
                        try:
                            completed_at = datetime.fromisoformat(warmup_meta['completed_at'])
                            started_at = datetime.fromisoformat(started_at_str)
                            if completed_at > started_at:
                                # Task completed but lock is stale — release and fall through
                                logger.info("Warmup completed after lock acquired, releasing stale lock")
                                lock.force_release()
                                # Fall through to SPY freshness check below
                            else:
                                # Old completion, task truly stuck
                                return {
                                    "status": "stuck",
                                    "message": "Task unresponsive (no heartbeat)",
                                    "can_refresh": True,
                                    "can_force_cancel": True,
                                    "spy_last_date": None,
                                    "expected_date": None,
                                    "task_running": {
                                        "task_id": current_holder.get('task_id'),
                                        "task_name": current_holder.get('task_name'),
                                        "started_at": started_at_str,
                                        "minutes_since_heartbeat": None
                                    },
                                    "last_warmup": warmup_meta
                                }
                        except (ValueError, TypeError):
                            pass

                    # Default: truly stuck
                    return {
                        "status": "stuck",
                        "message": "Task unresponsive (no heartbeat)",
                        "can_refresh": True,
                        "can_force_cancel": True,
                        "spy_last_date": None,
                        "expected_date": None,
                        "task_running": {
                            "task_id": current_holder.get('task_id'),
                            "task_name": current_holder.get('task_name'),
                            "started_at": started_at_str,
                            "minutes_since_heartbeat": None
                        },
                        "last_warmup": self._get_warmup_metadata()
                    }

            # No task running (or stale lock was released above) — check SPY freshness
            spy_last_date = self._get_spy_last_date()
            expected_date = self._get_expected_data_date()
            warmup_meta = self._get_warmup_metadata()

            if spy_last_date is None:
                return {
                    "status": "stale",
                    "message": "SPY benchmark not cached",
                    "can_refresh": True,
                    "spy_last_date": None,
                    "expected_date": str(expected_date) if expected_date else None,
                    "task_running": None,
                    "last_warmup": warmup_meta
                }

            # Compare SPY date with expected date
            is_fresh = spy_last_date >= expected_date if expected_date else True

            if is_fresh:
                # SPY is fresh — this is the authoritative signal.
                # Return "fresh" regardless of last warmup's partial status.
                return {
                    "status": "fresh",
                    "message": "Cache is up to date",
                    "can_refresh": True,
                    "spy_last_date": str(spy_last_date),
                    "expected_date": str(expected_date) if expected_date else None,
                    "task_running": None,
                    "last_warmup": warmup_meta
                }

            # SPY is stale — check if warmup metadata gives more context
            if warmup_meta and warmup_meta.get('status') == 'partial':
                return {
                    "status": "partial",
                    "message": f"Partial refresh: {warmup_meta.get('count', 0)}/{warmup_meta.get('total', 0)} symbols",
                    "can_refresh": True,
                    "spy_last_date": str(spy_last_date),
                    "expected_date": str(expected_date) if expected_date else None,
                    "task_running": None,
                    "last_warmup": warmup_meta
                }

            return {
                "status": "stale",
                "message": f"Missing data for {expected_date}",
                "can_refresh": True,
                "spy_last_date": str(spy_last_date),
                "expected_date": str(expected_date),
                "task_running": None,
                "last_warmup": warmup_meta
            }

        except Exception as e:
            logger.error(f"Error in get_cache_health_status: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error checking cache health: {str(e)}",
                "can_refresh": True,
                "spy_last_date": None,
                "expected_date": None,
                "task_running": None,
                "last_warmup": None
            }

    def _get_spy_last_date(self) -> Optional[date]:
        """
        Get the last date in SPY cache from Redis only (no API calls).

        IMPORTANT: This is called by the health endpoint (polled every 5-60s).
        It must never trigger a yfinance download. If SPY isn't in Redis,
        we return None (stale) and the user can trigger a refresh.

        Returns:
            date object of last SPY data, or None if not cached
        """
        try:
            if not self._redis_client:
                return None

            last_update_key = self.REDIS_KEY_LAST_UPDATE.format(symbol="SPY")
            last_date_str = self._redis_client.get(last_update_key)
            if last_date_str:
                decoded = last_date_str.decode() if isinstance(last_date_str, bytes) else last_date_str
                return date.fromisoformat(decoded)
            return None

        except Exception as e:
            logger.error(f"Error getting SPY last date from Redis: {e}")
            return None

    def _get_expected_data_date(self) -> Optional[date]:
        """
        Calculate the date that cache should have data for.

        Logic:
        - During market hours: Yesterday's close is sufficient
        - After 5 PM on trading day: Today's close expected
        - Before 9:30 AM on trading day: Yesterday's close is fine
        - Weekend/Holiday: Last trading day before today

        Grace period: Between 4:00-5:00 PM, don't expect today's
        close yet (data providers may have delay).

        Returns:
            date that cache should have
        """
        now_et = get_eastern_now()
        today = now_et.date()

        if is_market_open(now_et):
            # During market hours: yesterday's close is sufficient
            # (intraday data is a bonus, not required)
            return get_last_trading_day(today - timedelta(days=1))

        if is_trading_day(today):
            if now_et.hour >= 17 and now_et.minute >= 45:
                # After 5:45 PM on trading day: expect today's close
                # (scheduled refresh starts at 5:30 PM, give it 15 min to warm SPY)
                return today
            elif now_et.hour >= 18:
                # After 6 PM: definitely expect today's close
                return today
            elif now_et.hour >= 16:
                # Grace period 4:00-5:44 PM: yesterday's close is acceptable
                # Data providers may not have final close until ~4:30 PM,
                # and the scheduled refresh doesn't start until 5:30 PM
                return get_last_trading_day(today - timedelta(days=1))
            else:
                # Before market open (e.g., 7 AM Monday)
                # Yesterday's close (or Friday's if Monday) is fine
                return get_last_trading_day(today - timedelta(days=1))

        # Weekend or holiday: last trading day before today
        return get_last_trading_day(today - timedelta(days=1))

    def _get_warmup_metadata(self) -> Optional[Dict]:
        """
        Get metadata from the last warmup operation.

        Returns:
            Dict with status, count, total, completed_at or None
        """
        if not self._redis_client:
            return None

        try:
            meta_json = self._redis_client.get(WARMUP_METADATA_KEY)
            if meta_json:
                return json.loads(meta_json)
            return None
        except Exception as e:
            logger.error(f"Error getting warmup metadata: {e}")
            return None

    def save_warmup_metadata(self, status: str, count: int, total: int, error: str = None) -> None:
        """
        Save warmup operation metadata.

        Args:
            status: "completed", "partial", or "failed"
            count: Number of symbols successfully processed
            total: Total number of symbols attempted
            error: Error message if failed
        """
        if not self._redis_client:
            return

        try:
            meta = {
                "status": status,
                "count": count,
                "total": total,
                "completed_at": datetime.now().isoformat(),
                "error": error
            }
            self._redis_client.setex(
                WARMUP_METADATA_KEY,
                86400 * 7,  # 7 days TTL
                json.dumps(meta)
            )
            logger.info(f"Saved warmup metadata: {status} ({count}/{total})")
        except Exception as e:
            logger.error(f"Error saving warmup metadata: {e}")

    def update_warmup_heartbeat(self, current: int, total: int, percent: float = None) -> None:
        """
        Update heartbeat during warmup operation.

        Called periodically to indicate task is still making progress.
        Used for stuck detection.

        Args:
            current: Current symbol index
            total: Total symbols to process
            percent: Optional percentage complete
        """
        if not self._redis_client:
            return

        try:
            heartbeat = {
                "status": "running",
                "current": current,
                "total": total,
                "percent": percent or round((current / total) * 100, 1) if total > 0 else 0,
                "updated_at": datetime.now().isoformat()
            }
            self._redis_client.setex(
                WARMUP_HEARTBEAT_KEY,
                3600,  # 1 hour TTL
                json.dumps(heartbeat)
            )
        except Exception as e:
            logger.error(f"Error updating warmup heartbeat: {e}")

    def _get_heartbeat_info(self) -> Optional[Dict]:
        """
        Get heartbeat info including status and age.

        Returns:
            Dict with 'minutes' (float), 'status' (str), and raw heartbeat data,
            or None if no heartbeat found.
        """
        if not self._redis_client:
            return None

        try:
            heartbeat_json = self._redis_client.get(WARMUP_HEARTBEAT_KEY)
            if not heartbeat_json:
                return None

            heartbeat = json.loads(heartbeat_json)
            hb_status = heartbeat.get('status', 'running')

            # For terminal states, use completed_at; for running, use updated_at
            ts_field = 'completed_at' if hb_status in ('completed', 'failed') else 'updated_at'
            ts_str = heartbeat.get(ts_field, heartbeat.get('updated_at', ''))
            if ts_str:
                ts = datetime.fromisoformat(ts_str)
                minutes = (datetime.now() - ts).total_seconds() / 60
            else:
                minutes = None

            return {
                **heartbeat,
                "minutes": minutes,
                "status": hb_status,
            }
        except Exception as e:
            logger.error(f"Error getting heartbeat info: {e}")
            return None

    def _get_minutes_since_heartbeat(self) -> Optional[float]:
        """
        Get minutes since last heartbeat update.
        Backward-compatible wrapper around _get_heartbeat_info().

        Returns:
            Minutes since last heartbeat, or None if no heartbeat found
        """
        info = self._get_heartbeat_info()
        if info is None:
            return None
        return info.get("minutes")

    def _get_task_progress(self) -> Dict:
        """
        Get current task progress from heartbeat.

        Returns:
            Dict with current, total, percent or empty dict
        """
        if not self._redis_client:
            return {}

        try:
            heartbeat_json = self._redis_client.get(WARMUP_HEARTBEAT_KEY)
            if not heartbeat_json:
                return {}

            heartbeat = json.loads(heartbeat_json)
            return {
                "current": heartbeat.get('current'),
                "total": heartbeat.get('total'),
                "progress": heartbeat.get('percent')
            }
        except Exception as e:
            logger.error(f"Error getting task progress: {e}")
            return {}

    def clear_warmup_heartbeat(self) -> None:
        """Clear the warmup heartbeat. Kept for backward compatibility."""
        if not self._redis_client:
            return

        try:
            self._redis_client.delete(WARMUP_HEARTBEAT_KEY)
        except Exception as e:
            logger.error(f"Error clearing warmup heartbeat: {e}")

    def complete_warmup_heartbeat(self, status: str = "completed") -> None:
        """
        Write terminal heartbeat state instead of deleting.

        This prevents the race condition where the health endpoint polls
        between heartbeat deletion and lock release, falsely inferring "stuck".

        Args:
            status: Terminal status - "completed" or "failed"
        """
        if not self._redis_client:
            return

        try:
            heartbeat = {
                "status": status,
                "completed_at": datetime.now().isoformat(),
            }
            self._redis_client.setex(
                WARMUP_HEARTBEAT_KEY,
                3600,  # 1 hour TTL (same as running heartbeats)
                json.dumps(heartbeat)
            )
        except Exception as e:
            logger.error(f"Error completing warmup heartbeat: {e}")

    def clear_fetch_metadata(self, symbol: str) -> None:
        """
        Clear fetch metadata for a symbol (called after force refresh).
        """
        if not self._redis_client:
            return

        try:
            meta_key = self.REDIS_KEY_FETCH_META.format(symbol=symbol)
            self._redis_client.delete(meta_key)
        except Exception as e:
            logger.error(f"Error clearing fetch metadata for {symbol}: {e}", exc_info=True)

    def get_symbols_needing_refresh(self, symbols: List[str], max_age_hours: float = 4.0) -> List[str]:
        """
        Filter symbols to only those whose cache is older than max_age_hours.

        Uses Redis pipeline to batch-read fetch_meta keys for efficiency.
        Returns symbols that are either missing from cache or have a
        fetch_timestamp older than the threshold.

        Args:
            symbols: List of symbols to check
            max_age_hours: Maximum age in hours before a symbol needs refresh

        Returns:
            List of symbols that need refreshing
        """
        if not self._redis_client or not symbols:
            return symbols  # Can't check freshness without Redis — refresh all

        try:
            # Use Eastern time for cutoff since fetch_timestamp is stored in ET
            now_et = get_eastern_now()
            cutoff = now_et - timedelta(hours=max_age_hours)

            # Batch-read fetch_meta keys via pipeline
            pipeline = self._redis_client.pipeline()
            for symbol in symbols:
                meta_key = self.REDIS_KEY_FETCH_META.format(symbol=symbol)
                pipeline.get(meta_key)
            results = pipeline.execute()

            needs_refresh = []
            for symbol, meta_json in zip(symbols, results):
                if not meta_json:
                    # No metadata — never fetched or expired
                    needs_refresh.append(symbol)
                    continue

                try:
                    meta = json.loads(meta_json)
                    fetch_ts_str = meta.get("fetch_timestamp")
                    if not fetch_ts_str:
                        needs_refresh.append(symbol)
                        continue

                    fetch_ts = datetime.fromisoformat(fetch_ts_str)
                    # Compare as aware datetimes (both are Eastern)
                    # If fetch_ts somehow lost tz info, localize it to Eastern
                    if fetch_ts.tzinfo is None:
                        fetch_ts = EASTERN.localize(fetch_ts)

                    if fetch_ts < cutoff:
                        needs_refresh.append(symbol)
                except (json.JSONDecodeError, ValueError):
                    needs_refresh.append(symbol)

            return needs_refresh

        except Exception as e:
            logger.error(f"Error checking symbol freshness: {e}", exc_info=True)
            return symbols  # On error, refresh all to be safe

    def get_all_cached_symbols(self) -> List[str]:
        """
        Get all symbols that have cached price data in Redis.

        Returns:
            List of all cached symbol names
        """
        if not self._redis_client:
            return []

        symbols = []

        try:
            # Find all price:*:recent keys
            pattern = "price:*:recent"
            cursor = 0

            while True:
                cursor, keys = self._redis_client.scan(cursor, match=pattern, count=100)

                for key in keys:
                    # Extract symbol from key (price:{symbol}:recent)
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    parts = key_str.split(':')
                    if len(parts) == 3:
                        symbol = parts[1]
                        symbols.append(symbol)

                if cursor == 0:
                    break

            logger.info(f"Found {len(symbols)} cached symbols in Redis")
            return symbols

        except Exception as e:
            logger.error(f"Error scanning for cached symbols: {e}", exc_info=True)
            return []

    def _store_in_database(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Store price data in database (StockPrice table).

        Uses bulk insert to avoid N+1 query problem.
        """
        db = SessionLocal()

        try:
            # Reset index to get Date as a column
            df = data.reset_index()

            # Fetch all existing dates for this symbol upfront (avoid N+1 queries)
            existing_dates_query = db.query(StockPrice.date).filter(
                StockPrice.symbol == symbol
            )
            existing_dates = set([row[0] for row in existing_dates_query.all()])

            # Prepare bulk insert data - only for dates that don't exist
            rows_to_insert = []

            for _, row in df.iterrows():
                row_date = row['Date']

                # Convert pd.Timestamp to date for proper comparison with existing_dates
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
                        'symbol': symbol,
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
                    logger.warning(f"Error preparing row for {symbol} on {row.get('Date')}: {e}")
                    continue

            # Bulk insert all new rows in one operation
            if rows_to_insert:
                db.bulk_insert_mappings(StockPrice, rows_to_insert)
                db.commit()
                logger.info(f"Bulk inserted {len(rows_to_insert)} new rows for {symbol} to database")
            else:
                logger.debug(f"No new rows to persist for {symbol}")

        except Exception as e:
            logger.error(f"Error storing {symbol} in database: {e}", exc_info=True)
            db.rollback()

        finally:
            db.close()

    def _is_data_fresh(self, last_date: date, max_age_days: int = 1) -> bool:
        """
        Check if cached data is fresh enough using trading-day awareness.

        Delegates to _get_expected_data_date() which correctly handles all
        edge cases: market hours, grace periods, weekends, and holidays.
        """
        if last_date is None:
            return False

        expected = self._get_expected_data_date()
        if expected is None:
            return False

        is_fresh = last_date >= expected

        if not is_fresh:
            logger.debug(f"Data is stale (last: {last_date}, expected: {expected})")

        return is_fresh

    def store_in_cache(self, symbol: str, data: pd.DataFrame, also_store_db: bool = True) -> None:
        """
        Store price data in cache (Redis and optionally database).

        Public method for bulk fetching operations to populate cache.

        Args:
            symbol: Stock symbol
            data: Price data DataFrame
            also_store_db: Whether to also store in database (default True)
        """
        if data is None or data.empty:
            logger.warning(f"Cannot cache {symbol}: data is empty")
            return

        try:
            # Store in Redis
            self._store_recent_in_redis(symbol, data)
            logger.debug(f"Stored {symbol} in Redis cache ({len(data)} rows)")

            # Optionally store in database
            if also_store_db:
                self._store_in_database(symbol, data)
                logger.debug(f"Stored {symbol} in database ({len(data)} rows)")

        except Exception as e:
            logger.error(f"Error caching {symbol}: {e}")

    def store_batch_in_cache(
        self,
        batch_data: Dict[str, pd.DataFrame],
        also_store_db: bool = True
    ) -> int:
        """
        Store multiple symbols' price data in cache using Redis pipeline.

        Uses a single Redis pipeline for all symbols in the batch, reducing
        from 3 * N round-trips to 1 round-trip for N symbols.

        Args:
            batch_data: Dict mapping symbol to price DataFrame
            also_store_db: Whether to also store in database (default True)

        Returns:
            Number of symbols successfully cached
        """
        if not batch_data:
            return 0

        stored = 0

        # Batch Redis writes using pipeline
        if self._redis_client:
            try:
                now_et = get_eastern_now()
                market_open = is_market_open(now_et)
                data_type = "intraday" if market_open else "closing"
                # Pre-compute fetch metadata once (same for all symbols in batch)
                fetch_meta_json = json.dumps({
                    "fetch_timestamp": now_et.isoformat(),
                    "market_was_open": market_open,
                    "data_type": data_type,
                    "needs_refresh_after_close": market_open
                })

                pipeline = self._redis_client.pipeline()
                for symbol, data in batch_data.items():
                    if data is None or data.empty:
                        continue

                    try:
                        # Keep last 5 years for volume analysis
                        cutoff_datetime = datetime.now() - timedelta(days=self.RECENT_DAYS)
                        if hasattr(data.index, 'tz') and data.index.tz is not None:
                            cutoff_date = pd.Timestamp(cutoff_datetime, tz=data.index.tz)
                        else:
                            cutoff_date = pd.Timestamp(cutoff_datetime)
                        recent_data = data[data.index >= cutoff_date]
                        if recent_data.empty:
                            continue

                        redis_key = self.REDIS_KEY_RECENT.format(symbol=symbol)
                        pickled_data = pickle.dumps(recent_data)
                        pipeline.setex(redis_key, self.CACHE_TTL_SECONDS, pickled_data)

                        last_update_key = self.REDIS_KEY_LAST_UPDATE.format(symbol=symbol)
                        last_date = recent_data.index[-1].strftime('%Y-%m-%d')
                        pipeline.setex(last_update_key, self.CACHE_TTL_SECONDS, last_date)

                        meta_key = self.REDIS_KEY_FETCH_META.format(symbol=symbol)
                        pipeline.setex(meta_key, self.CACHE_TTL_SECONDS, fetch_meta_json)

                        stored += 1
                    except Exception as e:
                        logger.warning(f"Error preparing Redis pipeline for {symbol}: {e}")

                pipeline.execute()
                logger.debug(f"Batch stored {stored} symbols in Redis via pipeline")

            except Exception as e:
                logger.error(f"Error in batch Redis write: {e}", exc_info=True)
                # Fall back to individual writes
                for symbol, data in batch_data.items():
                    if data is not None and not data.empty:
                        self._store_recent_in_redis(symbol, data)

        # Batch DB writes
        if also_store_db:
            self._store_batch_in_database(batch_data)

        return stored

    def _store_batch_in_database(self, batch_data: Dict[str, pd.DataFrame]) -> None:
        """
        Store multiple symbols' price data in database in a single transaction.

        Queries existing dates for ALL symbols at once, then bulk inserts
        only new rows. Much more efficient than per-symbol _store_in_database().

        Args:
            batch_data: Dict mapping symbol to price DataFrame
        """
        if not batch_data:
            return

        db = SessionLocal()

        try:
            symbols = list(batch_data.keys())

            # Query all existing (symbol, date) pairs at once
            from sqlalchemy import and_, tuple_
            existing_pairs = set()
            # Query in chunks to respect SQLite's variable limit (999 max)
            for chunk_start in range(0, len(symbols), 100):
                chunk_symbols = symbols[chunk_start:chunk_start + 100]
                rows = db.query(StockPrice.symbol, StockPrice.date).filter(
                    StockPrice.symbol.in_(chunk_symbols)
                ).all()
                existing_pairs.update((r[0], r[1]) for r in rows)

            # Prepare all insert rows
            rows_to_insert = []
            for symbol, data in batch_data.items():
                if data is None or data.empty:
                    continue

                df = data.reset_index()
                for _, row in df.iterrows():
                    row_date = row['Date']
                    if isinstance(row_date, pd.Timestamp):
                        row_date = row_date.date()
                    elif isinstance(row_date, datetime):
                        row_date = row_date.date()

                    if (symbol, row_date) in existing_pairs:
                        continue

                    try:
                        rows_to_insert.append({
                            'symbol': symbol,
                            'date': row_date,
                            'open': float(row.get('Open', 0)),
                            'high': float(row.get('High', 0)),
                            'low': float(row.get('Low', 0)),
                            'close': float(row.get('Close', 0)),
                            'volume': int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else 0,
                            'adj_close': float(row.get('Close', 0))
                        })
                    except Exception as e:
                        logger.warning(f"Error preparing row for {symbol}: {e}")

            # Bulk insert in chunks (SQLite 999-variable limit / 8 columns = ~124 rows max)
            if rows_to_insert:
                chunk_size = 100
                for i in range(0, len(rows_to_insert), chunk_size):
                    chunk = rows_to_insert[i:i + chunk_size]
                    db.bulk_insert_mappings(StockPrice, chunk)
                db.commit()
                logger.info(f"Batch inserted {len(rows_to_insert)} new rows for {len(batch_data)} symbols")
            else:
                logger.debug(f"No new rows to persist for batch of {len(batch_data)} symbols")

        except Exception as e:
            logger.error(f"Error in batch database write: {e}", exc_info=True)
            db.rollback()

        finally:
            db.close()

    def get_many(self, symbols: list[str], period: str = "2y") -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get cached price data for multiple symbols using Redis pipeline.

        Uses chunked Redis pipeline operations (default 500 symbols/chunk)
        with a dedicated bulk connection pool (longer timeout) to avoid
        timeouts on large fetches. Per-chunk error handling ensures partial
        Redis failures degrade to DB fallback instead of total failure.

        IMPORTANT: Falls back to database for full historical data if Redis
        only has recent data (30 days) but caller needs more (e.g., 2y for Minervini).

        Args:
            symbols: List of stock ticker symbols
            period: Time period needed ("1y" or "2y") - used for database fallback

        Returns:
            Dict mapping symbols to their cached DataFrames (or None if not cached)
        """
        if not symbols:
            return {}

        if not self._redis_client:
            logger.warning("Redis not available for bulk get - using database fallback")
            # Fallback: query database for each symbol
            return {symbol: self.get_historical_data(symbol, period=period, force_refresh=False)
                    for symbol in symbols}

        try:
            # Use bulk Redis client with longer timeout for large pipeline operations
            bulk_client = get_bulk_redis_client() or self._redis_client
            chunk_size = getattr(settings, 'redis_pipeline_chunk_size', 500)

            # Pre-compute expected_date once (avoids per-symbol market hours calculation)
            expected_date = self._get_expected_data_date()

            # Parse results
            cached_data = {}
            redis_hits = []
            redis_misses = []
            insufficient_data = []
            stale_data = []

            # Chunked pipeline: process symbols in batches to avoid timeout on huge responses
            total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
            for chunk_idx in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[chunk_idx:chunk_idx + chunk_size]
                chunk_num = (chunk_idx // chunk_size) + 1

                try:
                    pipeline = bulk_client.pipeline()
                    for symbol in chunk_symbols:
                        redis_key = self.REDIS_KEY_RECENT.format(symbol=symbol)
                        pipeline.get(redis_key)
                    chunk_results = pipeline.execute()
                except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError, OSError) as pipe_err:
                    logger.warning(
                        f"Redis pipeline chunk {chunk_num}/{total_chunks} failed ({len(chunk_symbols)} symbols): {pipe_err}"
                    )
                    # Mark all chunk symbols as cache miss — they'll flow into DB fallback
                    for symbol in chunk_symbols:
                        cached_data[symbol] = None
                        redis_misses.append(symbol)
                    continue

                chunk_hits = 0
                for symbol, raw_data in zip(chunk_symbols, chunk_results):
                    if raw_data:
                        try:
                            df = pickle.loads(raw_data)

                            # Check if Redis data is sufficient for requested period
                            # Redis stores last 5 years (1825 days), but verify we have at least 200 days minimum
                            if len(df) >= 200:
                                # Check freshness using pre-computed expected_date (B2 optimization)
                                last_date = df.index[-1]
                                if hasattr(last_date, 'date'):
                                    last_date = last_date.date()

                                is_fresh = last_date >= expected_date if expected_date else True
                                if is_fresh:
                                    cached_data[symbol] = df
                                    redis_hits.append(symbol)
                                    chunk_hits += 1
                                    logger.debug(f"Bulk cache HIT for {symbol} (Redis, {len(df)} days, fresh)")
                                else:
                                    cached_data[symbol] = None
                                    stale_data.append(symbol)
                                    logger.debug(f"Bulk cache HIT but STALE for {symbol} (Redis, last: {last_date})")
                            else:
                                cached_data[symbol] = None
                                insufficient_data.append(symbol)
                                logger.debug(f"Bulk cache HIT but INSUFFICIENT for {symbol} (Redis has {len(df)} days, need 200+)")
                        except Exception as e:
                            logger.warning(f"Error deserializing {symbol}: {e}")
                            cached_data[symbol] = None
                            redis_misses.append(symbol)
                    else:
                        cached_data[symbol] = None
                        redis_misses.append(symbol)
                        logger.debug(f"Bulk cache MISS for {symbol} (Redis)")

                logger.info(
                    f"Redis pipeline chunk {chunk_num}/{total_chunks}: "
                    f"{chunk_hits} hits, {len(chunk_symbols) - chunk_hits} misses"
                )

            # Fallback to database for misses, insufficient data, and stale data
            needs_db_fallback = redis_misses + insufficient_data + stale_data
            if needs_db_fallback:
                logger.info(f"Fetching {len(needs_db_fallback)} symbols from database (Redis misses: {len(redis_misses)}, insufficient: {len(insufficient_data)}, stale: {len(stale_data)})")

                # STEP 1: Bulk database query (single DB session for all symbols)
                db_results = self._get_many_from_database(needs_db_fallback, period)

                # Separate DB hits from DB misses
                db_hits = []
                yfinance_needed = []

                for symbol in needs_db_fallback:
                    df, last_date = db_results.get(symbol, (None, None))
                    is_fresh = (last_date >= expected_date) if (last_date and expected_date) else False
                    if df is not None and not df.empty and is_fresh:
                        cached_data[symbol] = df
                        db_hits.append(symbol)
                        # Warm Redis for next time
                        self._store_recent_in_redis(symbol, df)
                    else:
                        yfinance_needed.append(symbol)

                logger.info(f"Database query: {len(db_hits)} hits, {len(yfinance_needed)} need yfinance")

                # STEP 2: Batch yfinance fetch for remaining symbols
                if yfinance_needed:
                    import time
                    from .bulk_data_fetcher import BulkDataFetcher

                    logger.info(f"Batch fetching {len(yfinance_needed)} symbols from yfinance...")

                    bulk_fetcher = BulkDataFetcher()
                    batch_size = getattr(settings, 'price_cache_yfinance_batch_size', 50)
                    rate_limit = getattr(settings, 'price_cache_yfinance_rate_limit', 1.0)

                    yfinance_success = 0
                    yfinance_failed = 0

                    # Process in batches with rate limiting
                    for batch_start in range(0, len(yfinance_needed), batch_size):
                        batch_symbols = yfinance_needed[batch_start:batch_start + batch_size]
                        batch_num = (batch_start // batch_size) + 1
                        total_batches = (len(yfinance_needed) + batch_size - 1) // batch_size

                        logger.info(f"yfinance batch {batch_num}/{total_batches}: fetching {len(batch_symbols)} symbols")

                        bulk_results = bulk_fetcher.fetch_batch_data(
                            batch_symbols,
                            period=period,
                            include_fundamentals=False
                        )

                        for symbol, data in bulk_results.items():
                            if not data.get('has_error') and data.get('price_data') is not None:
                                price_df = data['price_data']
                                cached_data[symbol] = price_df
                                yfinance_success += 1
                                # Store in both caches for future use
                                self._store_recent_in_redis(symbol, price_df)
                                self._store_in_database(symbol, price_df)
                            else:
                                cached_data[symbol] = None
                                yfinance_failed += 1

                        # Rate limit between batches (Redis-backed distributed limiter)
                        if batch_start + batch_size < len(yfinance_needed):
                            from .rate_limiter import rate_limiter
                            rate_limiter.wait("yfinance:batch", min_interval_s=rate_limit)

                    logger.info(f"yfinance batch fetch complete: {yfinance_success} success, {yfinance_failed} failed")

            logger.info(f"Bulk fetched {len(symbols)} symbols: {len(redis_hits)} Redis hits, {len(stale_data)} stale, {len(insufficient_data)} insufficient, {len(redis_misses)} misses")

            return cached_data

        except Exception as e:
            logger.error(f"Error in bulk get: {e}", exc_info=True)
            return {symbol: None for symbol in symbols}

    def invalidate_cache(self, symbol: str) -> None:
        """
        Invalidate cached data for a specific symbol.

        Args:
            symbol: Stock symbol to invalidate
        """
        if not self._redis_client:
            logger.warning("Redis not available for cache invalidation")
            return

        try:
            redis_key_recent = self.REDIS_KEY_RECENT.format(symbol=symbol)
            redis_key_update = self.REDIS_KEY_LAST_UPDATE.format(symbol=symbol)

            self._redis_client.delete(redis_key_recent)
            self._redis_client.delete(redis_key_update)

            logger.info(f"Invalidated cache for {symbol}")

        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}", exc_info=True)

    def get_cache_stats(self, symbol: str) -> Dict:
        """
        Get cache statistics for a symbol.

        Returns:
            Dict with cache info (last_update, cached_rows, etc.)
        """
        stats = {
            'symbol': symbol,
            'redis_cached': False,
            'db_cached': False,
            'last_update': None,
            'cached_rows': 0
        }

        # Check Redis
        if self._redis_client:
            try:
                last_update_key = self.REDIS_KEY_LAST_UPDATE.format(symbol=symbol)
                last_update = self._redis_client.get(last_update_key)

                if last_update:
                    stats['redis_cached'] = True
                    stats['last_update'] = last_update.decode('utf-8')
            except Exception as e:
                logger.debug(f"Error checking Redis stats for {symbol}: {e}")

        # Check Database
        db = SessionLocal()
        try:
            count = db.query(StockPrice).filter(StockPrice.symbol == symbol).count()

            if count > 0:
                stats['db_cached'] = True
                stats['cached_rows'] = count

                # Get last date from DB
                last_record = db.query(StockPrice).filter(
                    StockPrice.symbol == symbol
                ).order_by(StockPrice.date.desc()).first()

                if last_record and not stats['last_update']:
                    stats['last_update'] = str(last_record.date)

        except Exception as e:
            logger.debug(f"Error checking DB stats for {symbol}: {e}")
        finally:
            db.close()

        return stats

    # --- Symbol failure tracking for auto-deactivation of delisted symbols ---

    SYMBOL_FAILURE_KEY = "cache:symbol_failures:{symbol}"
    SYMBOL_FAILURE_THRESHOLD = 3
    SYMBOL_FAILURE_TTL = 86400 * 30  # 30 days

    def record_symbol_failure(self, symbol: str) -> int:
        """
        Increment persistent failure counter for a symbol.

        Returns the new failure count. When count >= SYMBOL_FAILURE_THRESHOLD,
        the caller should deactivate the symbol in stock_universe.
        """
        if not self._redis_client:
            return 0

        try:
            key = self.SYMBOL_FAILURE_KEY.format(symbol=symbol)
            count = self._redis_client.incr(key)
            self._redis_client.expire(key, self.SYMBOL_FAILURE_TTL)
            return count
        except Exception as e:
            logger.error(f"Error recording failure for {symbol}: {e}")
            return 0

    def clear_symbol_failure(self, symbol: str) -> None:
        """Clear failure counter on successful fetch."""
        if not self._redis_client:
            return

        try:
            key = self.SYMBOL_FAILURE_KEY.format(symbol=symbol)
            self._redis_client.delete(key)
        except Exception as e:
            logger.error(f"Error clearing failure for {symbol}: {e}")

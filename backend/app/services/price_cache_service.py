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
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta, date, time
import pandas as pd
from sqlalchemy.orm import Session

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    redis = Any  # type: ignore

from ..database import SessionLocal
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from ..config import settings
from ..utils.market_hours import (
    is_market_open, get_eastern_now, EASTERN, MARKET_CLOSE_TIME,
    is_trading_day, get_last_trading_day
)
from .cache.price_cache_failure_telemetry import PriceCacheFailureTelemetry
from .cache.price_cache_freshness import PriceCacheFreshnessPolicy
from .cache.market_cache_policy import MarketAwareCachePolicy, market_cache_policy
from .cache.price_cache_warmup import PriceCacheWarmupStore
from .errors import CacheRefreshError
from .redis_pool import get_redis_client, get_bulk_redis_client, is_redis_enabled

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

    # Redis keys
    REDIS_KEY_PREFIX = "price:"
    REDIS_KEY_RECENT = "price:{symbol}:recent"
    REDIS_KEY_LAST_UPDATE = "price:{symbol}:last_update"
    REDIS_KEY_FETCH_META = "price:{symbol}:fetch_meta"

    # TTL settings
    CACHE_TTL_SECONDS = 604800  # 7 days (aligned with config.cache_ttl_seconds)
    RECENT_DAYS = 1825  # Keep last 5 years in Redis (for 5-year volume analysis)

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable[[], Session]] = None,
        cache_policy: MarketAwareCachePolicy = market_cache_policy,
    ):
        """Initialize price cache service."""
        self._session_factory = session_factory or SessionLocal
        self._cache_policy = cache_policy
        if redis_client:
            self._redis_client = redis_client
        else:
            # Use shared connection pool for efficiency
            self._redis_client = get_redis_client()
            if self._redis_client:
                logger.debug("Connected to Redis for price caching (using shared pool)")
            elif is_redis_enabled():
                logger.warning("Redis connection failed. Will use database fallback.")
            else:
                logger.info("Redis disabled for this runtime. Using database fallback.")

        self._freshness_policy = PriceCacheFreshnessPolicy(
            logger=logger,
            redis_client=self._redis_client,
            fetch_meta_key_template="price:*:*:fetch_meta",
            get_expected_data_date=self._get_expected_data_date,
            get_fetch_metadata=self._get_fetch_metadata,
        )
        self._warmup_store = PriceCacheWarmupStore(
            logger=logger,
            redis_client=self._redis_client,
            metadata_key=WARMUP_METADATA_KEY,
            heartbeat_key=WARMUP_HEARTBEAT_KEY,
        )
        self._failure_telemetry = PriceCacheFailureTelemetry(
            logger=logger,
            redis_client=self._redis_client,
            key_template=self.SYMBOL_FAILURE_KEY,
            ttl_seconds=self.SYMBOL_FAILURE_TTL,
        )

    def get_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        force_refresh: bool = False,
        market: str | None = None,
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
            return self._fetch_full_and_cache(symbol, period, market=market)

        # Try to get cached data from database
        cached_data, last_date = self._get_from_database(symbol, period)

        if cached_data is not None and not cached_data.empty:
            # Check if data is fresh
            intraday_stale = self._is_intraday_data_stale(symbol, market=market)
            if self._is_data_fresh(last_date) and not intraday_stale:
                logger.info(f"Cache HIT for {symbol} (Database, last: {last_date})")

                # Also store in Redis for faster next access
                self._store_recent_in_redis(symbol, cached_data, market=market)

                return cached_data
            else:
                # Data is stale - fetch incremental update
                if intraday_stale:
                    logger.info(
                        "Cache HIT but STALE for %s (intraday bar requires after-close refresh) - fetching incremental",
                        symbol,
                    )
                else:
                    logger.info(
                        f"Cache HIT but STALE for {symbol} (last: {last_date}) - fetching incremental"
                    )
                return self._fetch_incremental_and_merge(
                    symbol,
                    period,
                    cached_data,
                    last_date,
                    force_same_day_refresh=intraday_stale,
                    market=market,
                )

        # No cached data - fetch full history
        logger.info(f"Cache MISS for {symbol} - fetching full history")
        return self._fetch_full_and_cache(symbol, period, market=market)

    def _redis_recent_key(self, symbol: str, market: str | None = None) -> str:
        return self._cache_policy.key("price", symbol, market=market, parts=("recent",))

    def _redis_last_update_key(self, symbol: str, market: str | None = None) -> str:
        return self._cache_policy.key("price", symbol, market=market, parts=("last_update",))

    def _redis_fetch_meta_key(self, symbol: str, market: str | None = None) -> str:
        return self._cache_policy.key("price", symbol, market=market, parts=("fetch_meta",))

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

    def get_cached_only_fresh(
        self,
        symbol: str,
        period: str = "2y"
    ) -> Optional[pd.DataFrame]:
        """
        Get cache-only price data when the cached row is still fresh enough.

        Returns None for stale same-day/intraday rows so callers can treat the
        symbol as a cache miss without triggering Yahoo fetches.
        """
        cached_data, last_date = self._get_from_database(symbol, period)
        if cached_data is None or cached_data.empty:
            logger.debug(f"Fresh cache-only MISS for {symbol}")
            return None

        if not self._is_data_fresh(last_date):
            logger.debug(f"Fresh cache-only STALE for {symbol} (last: {last_date})")
            return None

        if self._is_intraday_data_stale(symbol):
            logger.debug(f"Fresh cache-only INTRADAY_STALE for {symbol}")
            return None

        logger.debug(f"Fresh cache-only HIT for {symbol} (last: {last_date})")
        return cached_data

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

    def get_many_cached_only_fresh(
        self,
        symbols: List[str],
        period: str = "2y"
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get fresh-enough cached price data for multiple symbols without Yahoo fetches.

        Symbols with stale or missing database rows return None so callers can
        distinguish safe cached-only reads from symbols that still lack current
        technical input data.
        """
        results = self._get_many_from_database(symbols, period)
        fresh_results: Dict[str, Optional[pd.DataFrame]] = {}

        for symbol, (data, last_date) in results.items():
            if (
                data is not None
                and not data.empty
                and self._is_data_fresh(last_date)
                and not self._is_intraday_data_stale(symbol)
            ):
                fresh_results[symbol] = data
            else:
                fresh_results[symbol] = None

        return fresh_results

    def _get_from_database(self, symbol: str, period: str) -> tuple[Optional[pd.DataFrame], Optional[date]]:
        """
        Get cached price data from database.

        Returns:
            Tuple of (DataFrame, last_date) or (None, None)
        """
        db = self._session_factory()

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

        db = self._session_factory()
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

            chunk_size = max(1, int(getattr(settings, "price_cache_db_chunk_size", 250) or 250))
            total_chunks = (len(symbols) + chunk_size - 1) // chunk_size

            # Query symbols in bounded chunks. A full US cache-only group ranking
            # can touch thousands of symbols and millions of price rows; loading
            # that as ORM entities in one query can spike worker RSS enough for
            # the container OOM killer to terminate the Celery child.
            from sqlalchemy import and_

            for chunk_idx in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[chunk_idx:chunk_idx + chunk_size]
                chunk_num = (chunk_idx // chunk_size) + 1
                rows = db.query(
                    StockPrice.symbol,
                    StockPrice.date,
                    StockPrice.open,
                    StockPrice.high,
                    StockPrice.low,
                    StockPrice.close,
                    StockPrice.volume,
                ).filter(
                    and_(
                        StockPrice.symbol.in_(chunk_symbols),
                        StockPrice.date >= start_date,
                        StockPrice.date <= end_date
                    )
                ).order_by(StockPrice.symbol, StockPrice.date.asc()).all()

                # Group by symbol within this bounded result set.
                symbol_prices = {}
                for row in rows:
                    if row.symbol not in symbol_prices:
                        symbol_prices[row.symbol] = []
                    symbol_prices[row.symbol].append(row)

                for symbol in chunk_symbols:
                    prices = symbol_prices.get(symbol, [])

                    if not prices or len(prices) < 50:
                        results[symbol] = (None, None)
                        continue

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

                logger.debug(
                    "Bulk DB query chunk %d/%d: %d symbols, %d rows",
                    chunk_num,
                    total_chunks,
                    len(chunk_symbols),
                    len(rows),
                )

            logger.debug(f"Bulk DB query: {len([r for r in results.values() if r[0] is not None])} hits, "
                        f"{len([r for r in results.values() if r[0] is None])} misses")

            return results

        except Exception as e:
            logger.error(f"Error in bulk database query: {e}", exc_info=True)
            return {symbol: (None, None) for symbol in symbols}

        finally:
            db.close()

    def _fetch_full_and_cache(
        self,
        symbol: str,
        period: str,
        market: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch full historical data from the market provider and cache it.
        """
        try:
            data = self._fetch_direct_historical_data(symbol, period=period)

            if data is None or data.empty:
                logger.warning(f"Failed to fetch data for {symbol}")
                return None

            logger.info(f"Fetched {symbol}: {len(data)} rows")

            # Cache in Redis (recent data only)
            self._store_recent_in_redis(symbol, data, market=market)

            # Persist to database (full data)
            self._store_in_database(symbol, data)

            return data

        except Exception as exc:
            error = CacheRefreshError(
                f"Full cache refresh failed for {symbol}: {exc}",
                error_code="price_cache_full_refresh_failed",
            )
            logger.error(
                "%s",
                error,
                extra={
                    "event": "cache_refresh_failed",
                    "path": "price_cache_service._fetch_full_and_cache",
                    "pipeline": None,
                    "run_id": None,
                    "symbol": symbol,
                    "error_code": error.error_code,
                },
                exc_info=exc,
            )
            return None

    @staticmethod
    def _is_kr_price_symbol(symbol: str) -> bool:
        normalized = str(symbol or "").strip().upper()
        return normalized.endswith(".KS") or normalized.endswith(".KQ")

    @staticmethod
    def _is_cn_price_symbol(symbol: str) -> bool:
        normalized = str(symbol or "").strip().upper()
        return normalized.endswith(".SS") or normalized.endswith(".SZ") or normalized.endswith(".BJ")

    def _fetch_kr_historical_data(self, symbol: str, *, period: str) -> Optional[pd.DataFrame]:
        try:
            from .kr_market_data_service import KrxPriceService
            from .security_master_service import security_master_resolver

            identity = security_master_resolver.resolve_identity(symbol=symbol, market="KR")
            local_code = str(identity.local_code or "").strip()
            if not local_code.isdigit():
                return None
            return KrxPriceService().daily_ohlcv_dataframe(local_code, period=period)
        except Exception as exc:  # pragma: no cover - provider/network variability
            logger.warning("KRX historical fetch failed for %s: %s", symbol, exc)
            return None

    def _fetch_cn_historical_data(self, symbol: str, *, period: str) -> Optional[pd.DataFrame]:
        try:
            from .cn_market_data_service import CnMarketDataService
            from .security_master_service import security_master_resolver

            identity = security_master_resolver.resolve_identity(symbol=symbol, market="CN")
            local_code = str(identity.local_code or "").strip()
            if not local_code.isdigit():
                return None
            return CnMarketDataService().daily_ohlcv_dataframe(local_code, period=period)
        except Exception as exc:  # pragma: no cover - provider/network variability
            logger.warning("CN historical fetch failed for %s: %s", symbol, exc)
            return None

    def _fetch_direct_historical_data(self, symbol: str, *, period: str) -> Optional[pd.DataFrame]:
        if self._is_kr_price_symbol(symbol):
            krx_data = self._fetch_kr_historical_data(symbol, period=period)
            if krx_data is not None and not krx_data.empty:
                return krx_data
        if self._is_cn_price_symbol(symbol):
            cn_data = self._fetch_cn_historical_data(symbol, period=period)
            if cn_data is not None and not cn_data.empty:
                return cn_data
            if str(symbol or "").strip().upper().endswith(".BJ"):
                return None

        from .yfinance_service import YFinanceService

        yfinance_service = YFinanceService()
        # IMPORTANT: use_cache=False to avoid circular dependency
        return yfinance_service.get_historical_data(symbol, period=period, use_cache=False)

    def _fetch_incremental_and_merge(
        self,
        symbol: str,
        period: str,
        cached_data: pd.DataFrame,
        last_cached_date: date,
        force_same_day_refresh: bool = False,
        market: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch only new data since last_cached_date and merge with cached data.

        This is the key optimization - instead of fetching 2 years of data,
        we only fetch the missing days!
        """
        try:
            # Calculate how many days we're missing
            today = datetime.now().date()
            days_missing = (today - last_cached_date).days

            if days_missing <= 0 and not force_same_day_refresh:
                logger.info(f"{symbol} cache is current (last: {last_cached_date})")
                return cached_data

            if force_same_day_refresh and days_missing <= 0:
                logger.info(
                    "%s has a same-day intraday bar that needs after-close refresh - fetching overlap update",
                    symbol,
                )
            else:
                logger.info(f"{symbol} is {days_missing} days old - fetching incremental update")

            # Fetch only recent data (last 7 days to ensure overlap)
            new_data = self._fetch_direct_historical_data(symbol, period="7d")

            if new_data is None or new_data.empty:
                logger.warning(f"Failed to fetch incremental data for {symbol}")
                return cached_data  # Return stale cache as fallback

            # Filter new_data to only dates after last_cached_date
            # Ensure timezone compatibility for comparison
            last_cached_ts = pd.Timestamp(last_cached_date)
            if new_data.index.tz is not None and last_cached_ts.tz is None:
                last_cached_ts = last_cached_ts.tz_localize(new_data.index.tz)
            if force_same_day_refresh:
                new_data_filtered = new_data[new_data.index >= last_cached_ts]
            else:
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
            self._store_recent_in_redis(symbol, merged_data, market=market)
            self._store_in_database(symbol, new_data_filtered)  # Only persist new/updated rows

            return merged_data

        except Exception as exc:
            error = CacheRefreshError(
                f"Incremental cache refresh failed for {symbol}: {exc}",
                error_code="price_cache_incremental_refresh_failed",
            )
            logger.error(
                "%s",
                error,
                extra={
                    "event": "cache_refresh_failed",
                    "path": "price_cache_service._fetch_incremental_and_merge",
                    "pipeline": None,
                    "run_id": None,
                    "symbol": symbol,
                    "error_code": error.error_code,
                },
                exc_info=exc,
            )
            return cached_data  # Return stale cache as fallback

    def _store_recent_in_redis(
        self,
        symbol: str,
        data: pd.DataFrame,
        market: str | None = None,
    ) -> None:
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

            redis_key = self._redis_recent_key(symbol, market=market)
            pickled_data = pickle.dumps(recent_data)

            self._redis_client.setex(
                redis_key,
                self._cache_policy.ttl_seconds("price", market=market),
                pickled_data
            )

            # Also store last update timestamp
            last_update_key = self._redis_last_update_key(symbol, market=market)
            last_date = recent_data.index[-1].strftime('%Y-%m-%d')
            self._redis_client.setex(
                last_update_key,
                self._cache_policy.ttl_seconds("price", market=market),
                last_date
            )

            # Store fetch metadata for intraday staleness detection
            self._store_fetch_metadata(symbol, market=market)

            logger.debug(f"Cached {symbol} recent data in Redis ({len(recent_data)} rows)")

        except Exception as e:
            logger.error(f"Error storing {symbol} in Redis: {e}", exc_info=True)

    def _store_fetch_metadata(self, symbol: str, market: str | None = None) -> None:
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

            meta_key = self._redis_fetch_meta_key(symbol, market=market)
            self._redis_client.setex(
                meta_key,
                self._cache_policy.ttl_seconds("price", market=market),
                json.dumps(fetch_meta)
            )

            if needs_refresh:
                logger.debug(f"Stored fetch metadata for {symbol}: intraday data, needs refresh after close")

        except Exception as e:
            logger.error(f"Error storing fetch metadata for {symbol}: {e}", exc_info=True)

    def _get_fetch_metadata(self, symbol: str, market: str | None = None) -> Optional[Dict]:
        """
        Get fetch metadata for a symbol.

        Returns:
            Dict with fetch metadata or None if not found
        """
        if not self._redis_client:
            return None

        try:
            meta_key = self._redis_fetch_meta_key(symbol, market=market)
            meta_json = self._redis_client.get(meta_key)

            if meta_json:
                return json.loads(meta_json)
            return None

        except Exception as e:
            logger.error(f"Error getting fetch metadata for {symbol}: {e}", exc_info=True)
            return None

    def _is_fetch_metadata_stale(
        self,
        meta: Optional[Dict],
        *,
        now_et: Optional[datetime] = None,
    ) -> bool:
        """Return True when a fetch-meta record marks a same-day bar stale after close."""
        return self._freshness_policy.is_fetch_metadata_stale(meta, now_et=now_et)

    def _is_intraday_data_stale(self, symbol: str, market: str | None = None) -> bool:
        """
        Check if cached data is stale intraday data.

        Returns True if:
        - Data was fetched during market hours AND
        - Market is now closed (past 4:30 PM ET)

        This catches the case where data was fetched at 2 PM with an
        incomplete "today" bar, but user is now scanning at 6 PM.
        """
        if market is None:
            return self._freshness_policy.is_intraday_data_stale(symbol)
        meta = self._get_fetch_metadata(symbol, market=market)
        is_stale = self._freshness_policy.is_fetch_metadata_stale(meta)
        if is_stale:
            logger.debug(
                "%s: intraday data is stale for market %s (fetched during market, now after close)",
                symbol,
                market,
            )
        return is_stale

    def get_stale_intraday_symbols(self) -> List[str]:
        """
        Scan Redis for all symbols with stale intraday data.

        Uses pipeline to batch-read all fetch_meta values after SCAN,
        reducing from ~N individual GETs to 1 pipeline round-trip.

        Returns:
            List of symbols that have stale intraday data
        """
        return self._freshness_policy.get_stale_intraday_symbols()

    def get_staleness_status(self) -> Dict:
        """
        Get overall staleness status for the cache.

        Returns:
            Dict with staleness info including count and market status
        """
        return self._freshness_policy.get_staleness_status()

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

            # Check if a refresh task is currently running across all market scopes.
            from ..wiring.bootstrap import get_data_fetch_lock
            lock = get_data_fetch_lock()
            current_holder = lock.get_any_current_holder()

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
                    lock.force_release_all()
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
                                # Task completed but lock is stale — release and fall through.
                                # Use force_release_all() to clear whichever market key is held.
                                logger.info("Warmup completed after lock acquired, releasing stale lock")
                                lock.force_release_all()
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

            last_update_key = self._redis_last_update_key("SPY", market="US")
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
            if now_et.hour > 16 or (now_et.hour == 16 and now_et.minute >= 30):
                # After 4:30 PM on trading day: expect today's close.
                return today
            elif now_et.hour >= 16:
                # Grace period 4:00-4:29 PM: yesterday's close is acceptable
                # while data providers finalize the close.
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
        return self._warmup_store.get_warmup_metadata()

    def get_warmup_metadata(self, market: Optional[str] = None) -> Optional[Dict]:
        """Public wrapper for last warmup metadata used by downstream scheduled tasks."""
        return self._warmup_store.get_warmup_metadata(market=market)

    def save_warmup_metadata(
        self,
        status: str,
        count: int,
        total: int,
        error: str = None,
        market: Optional[str] = None,
    ) -> None:
        """Save warmup operation metadata, scoped per market."""
        self._warmup_store.save_warmup_metadata(
            status=status, count=count, total=total, error=error, market=market,
        )

    def update_warmup_heartbeat(
        self,
        current: int,
        total: int,
        percent: float = None,
        market: Optional[str] = None,
    ) -> None:
        """Update heartbeat during warmup, scoped per market."""
        self._warmup_store.update_warmup_heartbeat(
            current=current, total=total, percent=percent, market=market,
        )

    def _get_heartbeat_info(self, market: Optional[str] = None) -> Optional[Dict]:
        """Get heartbeat info including status and age."""
        return self._warmup_store.get_heartbeat_info(market=market)

    def _get_minutes_since_heartbeat(self, market: Optional[str] = None) -> Optional[float]:
        """Get minutes since last heartbeat update for the given market scope."""
        return self._warmup_store.get_minutes_since_heartbeat(market=market)

    def _get_task_progress(self, market: Optional[str] = None) -> Dict:
        """Get current task progress from heartbeat for the given market scope."""
        return self._warmup_store.get_task_progress(market=market)

    def clear_warmup_heartbeat(self, market: Optional[str] = None) -> None:
        """Clear the warmup heartbeat for the given market scope."""
        self._warmup_store.clear_warmup_heartbeat(market=market)

    def complete_warmup_heartbeat(
        self,
        status: str = "completed",
        market: Optional[str] = None,
    ) -> None:
        """Write terminal heartbeat state instead of deleting (per market)."""
        self._warmup_store.complete_warmup_heartbeat(status=status, market=market)

    def clear_fetch_metadata(self, symbol: str, market: str | None = None) -> None:
        """
        Clear fetch metadata for a symbol (called after force refresh).
        """
        if not self._redis_client:
            return

        try:
            meta_key = self._redis_fetch_meta_key(symbol, market=market)
            self._redis_client.delete(meta_key)
        except Exception as e:
            logger.error(f"Error clearing fetch metadata for {symbol}: {e}", exc_info=True)

    def get_symbols_needing_refresh(
        self,
        symbols: List[str],
        max_age_hours: float = 4.0,
        market: str | None = None,
    ) -> List[str]:
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
                meta_key = self._redis_fetch_meta_key(symbol, market=market)
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
            # Find all market-scoped price:*:*:recent keys plus legacy price:*:recent keys.
            pattern = "price:*:recent"
            cursor = 0

            while True:
                cursor, keys = self._redis_client.scan(cursor, match=pattern, count=100)

                for key in keys:
                    # Extract symbol from price:US:AAPL:recent or legacy price:AAPL:recent.
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    parts = key_str.split(':')
                    if len(parts) == 4:
                        symbol = parts[2]
                        symbols.append(symbol)
                    elif len(parts) == 3:
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

        Uses insert for historical rows and upsert/replace for the latest row so
        intraday partial bars can be corrected after the close.
        """
        db = self._session_factory()

        try:
            # Reset index to get Date as a column
            df = data.reset_index()
            if 'Date' not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: 'Date'})
            if df.empty:
                return

            normalized_dates = []
            for _, row in df.iterrows():
                row_date = row["Date"]
                if isinstance(row_date, pd.Timestamp):
                    row_date = row_date.date()
                elif isinstance(row_date, datetime):
                    row_date = row_date.date()
                normalized_dates.append(row_date)

            latest_row_date = max(normalized_dates)
            existing_rows = {
                record.date: record.id
                for record in db.query(StockPrice.id, StockPrice.date).filter(
                    StockPrice.symbol == symbol,
                    StockPrice.date.in_(normalized_dates),
                ).all()
            }

            rows_to_insert = []
            rows_to_update = []

            for _, row in df.iterrows():
                row_date = row['Date']

                # Convert pd.Timestamp to date for proper comparison with existing_dates
                if isinstance(row_date, pd.Timestamp):
                    row_date = row_date.date()
                elif isinstance(row_date, datetime):
                    row_date = row_date.date()

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
                        'adj_close': float(row.get('Adj Close', row.get('Close', 0))),
                    }
                    existing_id = existing_rows.get(row_date)
                    if existing_id is None:
                        rows_to_insert.append(price_dict)
                    elif row_date == latest_row_date:
                        price_dict["id"] = existing_id
                        rows_to_update.append(price_dict)

                except Exception as e:
                    logger.warning(f"Error preparing row for {symbol} on {row.get('Date')}: {e}")
                    continue

            # Bulk insert historical rows, overwrite the latest day if it already exists.
            if rows_to_insert:
                db.bulk_insert_mappings(StockPrice, rows_to_insert)
            if rows_to_update:
                db.bulk_update_mappings(StockPrice, rows_to_update)

            db.commit()
            if rows_to_insert or rows_to_update:
                logger.info(
                    "Persisted %s price rows for %s (%d inserts, %d latest-day updates)",
                    len(rows_to_insert) + len(rows_to_update),
                    symbol,
                    len(rows_to_insert),
                    len(rows_to_update),
                )
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
        del max_age_days
        return self._freshness_policy.is_data_fresh(last_date)

    def store_in_cache(
        self,
        symbol: str,
        data: pd.DataFrame,
        also_store_db: bool = True,
        market: str | None = None,
    ) -> None:
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
            self._store_recent_in_redis(symbol, data, market=market)
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
        also_store_db: bool = True,
        market: str | None = None,
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

                        redis_key = self._redis_recent_key(symbol, market=market)
                        pickled_data = pickle.dumps(recent_data)
                        pipeline.setex(
                            redis_key,
                            self._cache_policy.ttl_seconds("price", market=market),
                            pickled_data,
                        )

                        last_update_key = self._redis_last_update_key(symbol, market=market)
                        last_date = recent_data.index[-1].strftime('%Y-%m-%d')
                        pipeline.setex(
                            last_update_key,
                            self._cache_policy.ttl_seconds("price", market=market),
                            last_date,
                        )

                        meta_key = self._redis_fetch_meta_key(symbol, market=market)
                        pipeline.setex(
                            meta_key,
                            self._cache_policy.ttl_seconds("price", market=market),
                            fetch_meta_json,
                        )

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
                        self._store_recent_in_redis(symbol, data, market=market)

        # Batch DB writes
        if also_store_db:
            self._store_batch_in_database(batch_data)

        return stored

    def _store_batch_in_database(self, batch_data: Dict[str, pd.DataFrame]) -> None:
        """
        Store multiple symbols' price data in database in a single transaction.

        Queries existing dates for ALL symbols at once, bulk inserts historical
        rows, and replaces the latest row when it already exists.

        Args:
            batch_data: Dict mapping symbol to price DataFrame
        """
        if not batch_data:
            return

        db = self._session_factory()

        try:
            symbols = list(batch_data.keys())

            symbol_dates: Dict[str, set] = {}
            latest_dates: Dict[str, date] = {}
            for symbol, data in batch_data.items():
                if data is None or data.empty:
                    continue
                normalized = set()
                latest = None
                for raw_date in data.reset_index()["Date"]:
                    row_date = raw_date
                    if isinstance(row_date, pd.Timestamp):
                        row_date = row_date.date()
                    elif isinstance(row_date, datetime):
                        row_date = row_date.date()
                    normalized.add(row_date)
                    latest = row_date if latest is None or row_date > latest else latest
                if normalized:
                    symbol_dates[symbol] = normalized
                    latest_dates[symbol] = latest

            existing_pairs: Dict[tuple[str, date], int] = {}
            for chunk_start in range(0, len(symbols), 100):
                chunk_symbols = symbols[chunk_start:chunk_start + 100]
                rows = db.query(StockPrice.id, StockPrice.symbol, StockPrice.date).filter(
                    StockPrice.symbol.in_(chunk_symbols)
                ).all()
                for record_id, record_symbol, record_date in rows:
                    target_dates = symbol_dates.get(record_symbol)
                    if target_dates and record_date in target_dates:
                        existing_pairs[(record_symbol, record_date)] = record_id

            rows_to_insert = []
            rows_to_update = []
            for symbol, data in batch_data.items():
                if data is None or data.empty:
                    continue

                df = data.reset_index()
                if 'Date' not in df.columns and len(df.columns) > 0:
                    df = df.rename(columns={df.columns[0]: 'Date'})
                for _, row in df.iterrows():
                    row_date = row['Date']
                    if isinstance(row_date, pd.Timestamp):
                        row_date = row_date.date()
                    elif isinstance(row_date, datetime):
                        row_date = row_date.date()

                    try:
                        price_dict = {
                            'symbol': symbol,
                            'date': row_date,
                            'open': float(row.get('Open', 0)),
                            'high': float(row.get('High', 0)),
                            'low': float(row.get('Low', 0)),
                            'close': float(row.get('Close', 0)),
                            'volume': int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else 0,
                            'adj_close': float(row.get('Adj Close', row.get('Close', 0))),
                        }
                        existing_id = existing_pairs.get((symbol, row_date))
                        if existing_id is None:
                            rows_to_insert.append(price_dict)
                        elif row_date == latest_dates.get(symbol):
                            price_dict["id"] = existing_id
                            rows_to_update.append(price_dict)
                    except Exception as e:
                        logger.warning(f"Error preparing row for {symbol}: {e}")

            # Bulk insert in conservative chunks to keep statement size bounded.
            if rows_to_insert:
                chunk_size = 100
                for i in range(0, len(rows_to_insert), chunk_size):
                    chunk = rows_to_insert[i:i + chunk_size]
                    db.bulk_insert_mappings(StockPrice, chunk)
            if rows_to_update:
                chunk_size = 100
                for i in range(0, len(rows_to_update), chunk_size):
                    chunk = rows_to_update[i:i + chunk_size]
                    db.bulk_update_mappings(StockPrice, chunk)

            if rows_to_insert or rows_to_update:
                db.commit()
                logger.info(
                    "Batch persisted %d price rows for %d symbols (%d inserts, %d latest-day updates)",
                    len(rows_to_insert) + len(rows_to_update),
                    len(batch_data),
                    len(rows_to_insert),
                    len(rows_to_update),
                )
            else:
                logger.debug(f"No new rows to persist for batch of {len(batch_data)} symbols")

        except Exception as e:
            logger.error(f"Error in batch database write: {e}", exc_info=True)
            db.rollback()

        finally:
            db.close()

    @staticmethod
    def _market_for_symbol(
        symbol: str,
        *,
        market: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
    ) -> str | None:
        if market_by_symbol is not None and symbol in market_by_symbol:
            return market_by_symbol[symbol]
        return market

    def _store_batch_in_cache_for_market(
        self,
        batch_data: Dict[str, pd.DataFrame],
        *,
        also_store_db: bool,
        market: str | None,
    ) -> int:
        if market is None:
            return self.store_batch_in_cache(batch_data, also_store_db=also_store_db)
        try:
            return self.store_batch_in_cache(batch_data, also_store_db=also_store_db, market=market)
        except TypeError as exc:
            if "market" not in str(exc):
                raise
            return self.store_batch_in_cache(batch_data, also_store_db=also_store_db)

    def get_many(
        self,
        symbols: list[str],
        period: str = "2y",
        market: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
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
            market: Optional market for homogeneous batches.
            market_by_symbol: Optional per-symbol market map for mixed batches.

        Returns:
            Dict mapping symbols to their cached DataFrames (or None if not cached)
        """
        if not symbols:
            return {}

        expected_date = self._get_expected_data_date()
        now_et = get_eastern_now()

        if not self._redis_client:
            logger.info("Redis unavailable for bulk get - using database and batch-fetch fallback")
            return self._resolve_bulk_fallback(
                symbols,
                period=period,
                expected_date=expected_date,
                now_et=now_et,
                market=market,
                market_by_symbol=market_by_symbol,
            )

        try:
            # Use bulk Redis client with longer timeout for large pipeline operations
            bulk_client = get_bulk_redis_client() or self._redis_client
            chunk_size = getattr(settings, 'redis_pipeline_chunk_size', 500)

            # Parse results
            cached_data = {}
            redis_hits = []
            redis_misses = []
            insufficient_data = []
            stale_data = []
            stale_intraday_data = []
            fetch_meta_by_symbol: Dict[str, Optional[Dict]] = {}

            # Chunked pipeline: process symbols in batches to avoid timeout on huge responses
            total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
            for chunk_idx in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[chunk_idx:chunk_idx + chunk_size]
                chunk_num = (chunk_idx // chunk_size) + 1

                try:
                    pipeline = bulk_client.pipeline()
                    for symbol in chunk_symbols:
                        symbol_market = self._market_for_symbol(
                            symbol,
                            market=market,
                            market_by_symbol=market_by_symbol,
                        )
                        redis_key = self._redis_recent_key(symbol, market=symbol_market)
                        pipeline.get(redis_key)
                        pipeline.get(self._redis_fetch_meta_key(symbol, market=symbol_market))
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
                recent_results = chunk_results[0::2]
                meta_results = chunk_results[1::2]
                for symbol, raw_data, meta_raw in zip(chunk_symbols, recent_results, meta_results):
                    meta = None
                    if meta_raw:
                        try:
                            meta = json.loads(meta_raw)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            meta = None
                    fetch_meta_by_symbol[symbol] = meta
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
                                meta_is_stale = self._is_fetch_metadata_stale(meta, now_et=now_et)
                                if meta_is_stale:
                                    is_fresh = False
                                if is_fresh:
                                    cached_data[symbol] = df
                                    redis_hits.append(symbol)
                                    chunk_hits += 1
                                    logger.debug(f"Bulk cache HIT for {symbol} (Redis, {len(df)} days, fresh)")
                                else:
                                    cached_data[symbol] = None
                                    if meta_is_stale:
                                        stale_intraday_data.append(symbol)
                                        logger.debug(
                                            "Bulk cache HIT but STALE_INTRADAY for %s (Redis, last: %s)",
                                            symbol,
                                            last_date,
                                        )
                                    else:
                                        stale_data.append(symbol)
                                        logger.debug(
                                            f"Bulk cache HIT but STALE for {symbol} (Redis, last: {last_date})"
                                        )
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
            needs_db_fallback = redis_misses + insufficient_data + stale_data + stale_intraday_data
            if needs_db_fallback:
                logger.info(
                    "Fetching %d symbols from database (Redis misses: %d, insufficient: %d, stale: %d, stale intraday: %d)",
                    len(needs_db_fallback),
                    len(redis_misses),
                    len(insufficient_data),
                    len(stale_data),
                    len(stale_intraday_data),
                )
                cached_data.update(
                    self._resolve_bulk_fallback(
                        needs_db_fallback,
                        period=period,
                        expected_date=expected_date,
                        now_et=now_et,
                        fetch_meta_by_symbol=fetch_meta_by_symbol,
                        market=market,
                        market_by_symbol=market_by_symbol,
                    )
                )

            logger.info(
                "Bulk fetched %d symbols: %d Redis hits, %d stale, %d stale intraday, %d insufficient, %d misses",
                len(symbols),
                len(redis_hits),
                len(stale_data),
                len(stale_intraday_data),
                len(insufficient_data),
                len(redis_misses),
            )

            return cached_data

        except Exception as e:
            logger.error(f"Error in bulk get: {e}", exc_info=True)
            return {symbol: None for symbol in symbols}

    def _resolve_bulk_fallback(
        self,
        symbols: list[str],
        *,
        period: str,
        expected_date: Optional[date],
        now_et: datetime,
        fetch_meta_by_symbol: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        market: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Resolve a multi-symbol cache miss via one DB query and one optional batch fetch.

        This keeps the non-Redis path efficient and reuses the same freshness logic
        as the Redis-assisted bulk path.
        """
        if not symbols:
            return {}

        fetch_meta_by_symbol = fetch_meta_by_symbol or {}
        cached_data: Dict[str, Optional[pd.DataFrame]] = {}

        db_results = self._get_many_from_database(symbols, period)
        db_hits = []
        yfinance_needed = []

        for symbol in symbols:
            df, last_date = db_results.get(symbol, (None, None))
            symbol_market = self._market_for_symbol(
                symbol,
                market=market,
                market_by_symbol=market_by_symbol,
            )
            is_fresh = (last_date >= expected_date) if (last_date and expected_date) else False
            if self._is_fetch_metadata_stale(fetch_meta_by_symbol.get(symbol), now_et=now_et):
                is_fresh = False
            if df is not None and not df.empty and is_fresh:
                cached_data[symbol] = df
                db_hits.append(symbol)
                if self._redis_client:
                    self._store_recent_in_redis(symbol, df, market=symbol_market)
            else:
                yfinance_needed.append(symbol)

        logger.info("Database query: %d hits, %d need yfinance", len(db_hits), len(yfinance_needed))

        if not yfinance_needed:
            return cached_data

        from .bulk_data_fetcher import BulkDataFetcher

        active_query_db = self._session_factory()
        try:
            active_market_by_symbol = {
                row[0]: row[1]
                for row in active_query_db.query(StockUniverse.symbol, StockUniverse.market).filter(
                    StockUniverse.symbol.in_(yfinance_needed),
                    StockUniverse.active_filter(),
                ).all()
            }
        finally:
            active_query_db.close()

        inactive_symbols = [symbol for symbol in yfinance_needed if symbol not in active_market_by_symbol]
        active_yfinance_needed = [symbol for symbol in yfinance_needed if symbol in active_market_by_symbol]

        logger.info(
            "Batch fetching %d active symbols from market price providers (%d inactive skipped)",
            len(active_yfinance_needed),
            len(inactive_symbols),
        )

        if inactive_symbols:
            for symbol in inactive_symbols:
                df, _ = db_results.get(symbol, (None, None))
                cached_data[symbol] = df

        yfinance_success = 0
        yfinance_failed = 0

        if active_yfinance_needed:
            bulk_fetcher = BulkDataFetcher()
            bulk_results = {}
            market_groups: dict[str | None, list[str]] = {}
            for symbol in active_yfinance_needed:
                symbol_market = self._market_for_symbol(
                    symbol,
                    market=market,
                    market_by_symbol=market_by_symbol,
                )
                if symbol_market is None:
                    symbol_market = active_market_by_symbol.get(symbol)
                market_groups.setdefault(symbol_market, []).append(symbol)
            for group_market, market_symbols in market_groups.items():
                try:
                    provider_results = bulk_fetcher.fetch_prices_in_batches(
                        market_symbols,
                        period=period,
                        start_batch_size=getattr(settings, 'price_cache_yfinance_batch_size', 100),
                        market=group_market,
                    )
                except TypeError as exc:
                    if "market" not in str(exc):
                        raise
                    provider_results = bulk_fetcher.fetch_prices_in_batches(
                        market_symbols,
                        period=period,
                        start_batch_size=getattr(settings, 'price_cache_yfinance_batch_size', 100),
                    )
                bulk_results.update(provider_results)
            batch_to_store_by_market: dict[str | None, Dict[str, pd.DataFrame]] = {}
            for symbol, data in bulk_results.items():
                if not data.get('has_error') and data.get('price_data') is not None:
                    price_df = data['price_data']
                    cached_data[symbol] = price_df
                    yfinance_success += 1
                    symbol_market = self._market_for_symbol(
                        symbol,
                        market=market,
                        market_by_symbol=market_by_symbol,
                    )
                    if symbol_market is None:
                        symbol_market = active_market_by_symbol.get(symbol)
                    batch_to_store_by_market.setdefault(symbol_market, {})[symbol] = price_df
                else:
                    cached_data[symbol] = None
                    yfinance_failed += 1
            for group_market, batch_to_store in batch_to_store_by_market.items():
                self._store_batch_in_cache_for_market(
                    batch_to_store,
                    also_store_db=True,
                    market=group_market,
                )

        logger.info("yfinance batch fetch complete: %d success, %d failed", yfinance_success, yfinance_failed)
        return cached_data

    def invalidate_cache(self, symbol: str, market: str | None = None) -> None:
        """
        Invalidate cached data for a specific symbol.

        Args:
            symbol: Stock symbol to invalidate
            market: Market for the market-scoped cache key. Defaults to US for legacy callers.
        """
        if not self._redis_client:
            logger.warning("Redis not available for cache invalidation")
            return

        try:
            redis_key_recent = self._redis_recent_key(symbol, market=market)
            redis_key_update = self._redis_last_update_key(symbol, market=market)

            self._redis_client.delete(redis_key_recent)
            self._redis_client.delete(redis_key_update)

            logger.info(f"Invalidated cache for {symbol}")

        except Exception as e:
            logger.error(f"Error invalidating cache for {symbol}: {e}", exc_info=True)

    def get_cache_stats(self, symbol: str, market: str | None = None) -> Dict:
        """
        Get cache statistics for a symbol.

        Returns:
            Dict with cache info (last_update, cached_rows, etc.)
        """
        stats = {
            'symbol': symbol,
            'market': self._cache_policy.normalize_market(market),
            'redis_cached': False,
            'db_cached': False,
            'last_update': None,
            'cached_rows': 0
        }

        # Check Redis
        if self._redis_client:
            try:
                last_update_key = self._redis_last_update_key(symbol, market=market)
                last_update = self._redis_client.get(last_update_key)

                if last_update:
                    stats['redis_cached'] = True
                    stats['last_update'] = last_update.decode('utf-8')
            except Exception as e:
                logger.debug(f"Error checking Redis stats for {symbol}: {e}")

        # Check Database
        db = self._session_factory()
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
    SYMBOL_FAILURE_THRESHOLD = 5
    SYMBOL_FAILURE_TTL = 86400 * 30  # 30 days

    def record_symbol_failure(self, symbol: str) -> int:
        """
        Increment persistent failure counter for a symbol.

        Returns the new failure count. When count >= SYMBOL_FAILURE_THRESHOLD,
        the caller should deactivate the symbol in stock_universe.
        """
        return self._failure_telemetry.record_symbol_failure(symbol)

    def clear_symbol_failure(self, symbol: str) -> None:
        """Clear failure counter on successful fetch."""
        self._failure_telemetry.clear_symbol_failure(symbol)

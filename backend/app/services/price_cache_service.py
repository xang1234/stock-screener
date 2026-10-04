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
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Dict, List, Callable, Mapping
from datetime import datetime, timedelta, date
import pandas as pd
from sqlalchemy.orm import Session

try:
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in desktop packaging
    redis = Any  # type: ignore

from ..database import SessionLocal
from ..domain.markets.cn_symbols import cn_price_symbol_for_native_provider
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse, UNIVERSE_STATUS_ACTIVE
from ..config import settings
from ..utils.market_hours import (
    is_market_open, get_eastern_now, EASTERN, MARKET_CLOSE_TIME,
    is_trading_day, get_last_trading_day
)
from .cache.price_cache_failure_telemetry import PriceCacheFailureTelemetry
from .cache.price_cache_freshness import PriceCacheFreshnessPolicy, latest_fetch_metadata
from .cache.market_cache_policy import MarketAwareCachePolicy, market_cache_policy
from .cache.price_cache_warmup import PriceCacheWarmupStore
from .cache.redis_codec import decode_frame, encode_frame
from .errors import CacheRefreshError
from .price_row_normalization import (
    normalize_price_batch,
    normalize_price_frame,
    STOCK_PRICE_ROW_COLUMNS,
    stock_price_frame,
    stock_price_frames_by_symbol,
    stock_price_row_from_ohlcv,
)
from .stock_price_persistence import persist_stock_price_mappings
from .redis_pool import get_redis_client, get_bulk_redis_client, is_redis_enabled

if TYPE_CHECKING:
    from .market_calendar_service import MarketCalendarService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _registered_instrument_markets() -> Dict[str, str]:
    """Trading market of key-market instruments fetched outside ``stock_universe`` (e.g. ^HSI)."""
    from ..domain.markets.key_markets import KEY_MARKET_INSTRUMENTS_BY_MARKET

    return {
        instrument.data_symbol.upper(): instrument.market
        for instruments in KEY_MARKET_INSTRUMENTS_BY_MARKET.values()
        for instrument in instruments
    }

# Redis keys for warmup metadata
WARMUP_METADATA_KEY = "cache:warmup:metadata"
WARMUP_HEARTBEAT_KEY = "cache:warmup:heartbeat"

# Calendar days of history per requested period, shared by every cache tier.
PERIOD_DAYS: Dict[str, int] = {
    "5y": 1825,  # 5 years
    "2y": 730,   # 2 years
    "1y": 365,   # 1 year
    "7d": 7,     # live delta top-up
    "max": 3650  # 10 years for max
}
DEFAULT_PERIOD_DAYS = 730  # Unknown periods read as 2y for backward compat


def _period_days(period: str) -> int:
    return PERIOD_DAYS.get(period, DEFAULT_PERIOD_DAYS)


def _rejected_latest_date(
    raw: Optional[pd.DataFrame],
    normalized: Optional[pd.DataFrame],
) -> Optional[date]:
    """Date of the latest fetched bar when normalization dropped it, else None."""
    if raw is None or raw.empty:
        return None
    latest = raw.index.max()
    if normalized is not None and not normalized.empty and normalized.index.max() >= latest:
        return None
    return pd.Timestamp(latest).date()


# DataFrame.attrs key on a Redis frame: the window (calendar days) it was cut from.
# A 5y read that returns 250 bars is complete for 5y; a 2y read of 500 bars is not,
# and row counts cannot tell the two apart.
_COVERAGE_ATTR = "price_cache_period_days"


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
        market_calendar: Optional["MarketCalendarService"] = None,
    ):
        """Initialize price cache service."""
        self._session_factory = session_factory or SessionLocal
        self._cache_policy = cache_policy
        self._market_calendar = market_calendar
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
            fetch_meta_key_template=("price:*:*:fetch_meta", "price:*:fetch_meta"),
            get_expected_data_date=lambda market: self._get_expected_data_date(market),
            get_market_calendar=self._calendar,
            resolve_calendar_markets=lambda key_markets: self._calendar_markets(
                list(key_markets), fallback_by_symbol=key_markets
            ),
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
            calendar_market = self._calendar_markets([symbol], market=market)[symbol]
            intraday_stale = self._is_intraday_data_stale(
                symbol, market=market, calendar_market=calendar_market
            )
            if self._is_data_fresh(last_date, market=calendar_market) and not intraday_stale:
                logger.info(f"Cache HIT for {symbol} (Database, last: {last_date})")

                # Also store in Redis for faster next access
                self._store_recent_in_redis(
                    symbol,
                    cached_data,
                    market=market,
                    stamp_fetch_metadata=False,
                    period=period,
                    from_database=True,
                )

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

    @staticmethod
    def _contains_required_as_of_date(
        data: Optional[pd.DataFrame],
        required_as_of_date: date | None,
    ) -> bool:
        if required_as_of_date is None:
            return True
        if data is None or data.empty:
            return False
        return any(
            pd.Timestamp(index_value).date() == required_as_of_date
            for index_value in data.index
        )

    def get_cached_only_fresh(
        self,
        symbol: str,
        period: str = "2y",
        *,
        required_as_of_date: date | None = None,
        market: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get cache-only price data when the cached row is still fresh enough.

        Returns None for stale same-day/intraday rows so callers can treat the
        symbol as a cache miss without triggering Yahoo fetches. Freshness uses
        the symbol's own market calendar.
        """
        cached_data, last_date = self._get_from_database(symbol, period)
        if cached_data is None or cached_data.empty:
            logger.debug(f"Fresh cache-only MISS for {symbol}")
            return None

        if not self._contains_required_as_of_date(
            cached_data,
            required_as_of_date,
        ):
            logger.debug(
                "Fresh cache-only TARGET_DATE_MISS for %s (required: %s)",
                symbol,
                required_as_of_date,
            )
            return None

        calendar_market = self._calendar_markets([symbol], market=market)[symbol]
        if required_as_of_date is None and not self._is_data_fresh(last_date, market=calendar_market):
            logger.debug(f"Fresh cache-only STALE for {symbol} (last: {last_date})")
            return None

        if self._is_intraday_data_stale(symbol, market=market, calendar_market=calendar_market):
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
        period: str = "2y",
        *,
        required_as_of_date: date | None = None,
        minimum_rows: int = 50,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Get fresh-enough cached price data for multiple symbols without Yahoo fetches.

        Symbols with stale or missing database rows return None so callers can
        distinguish safe cached-only reads from symbols that still lack current
        technical input data. ``minimum_rows`` lets formula-specific consumers
        admit shorter but structurally valid histories.
        """
        results = self._get_many_from_database(
            symbols,
            period,
            minimum_rows=minimum_rows,
        )
        fresh_results: Dict[str, Optional[pd.DataFrame]] = {}
        calendar_markets = self._calendar_markets(list(results))

        # Date checks first; only survivors need their fetch metadata.
        candidates = {
            symbol: calendar_markets[symbol]
            for symbol, (data, last_date) in results.items()
            if data is not None
            and not data.empty
            and (
                required_as_of_date is not None
                or self._is_data_fresh(last_date, market=calendar_markets[symbol])
            )
            and self._contains_required_as_of_date(data, required_as_of_date)
        }
        meta_by_symbol = self._latest_fetch_metadata_many(candidates)

        for symbol, (data, _last_date) in results.items():
            calendar_market = candidates.get(symbol)
            if calendar_market is not None and not self._is_fetch_metadata_stale(
                meta_by_symbol.get(symbol), market=calendar_market
            ):
                fresh_results[symbol] = data
            else:
                fresh_results[symbol] = None

        return fresh_results

    def _latest_fetch_metadata_many(
        self,
        calendar_market_by_symbol: Mapping[str, str],
    ) -> Dict[str, Optional[Dict]]:
        """Newest fetch metadata per symbol across its key namespaces, in one pipeline.

        Reads the same keys as ``_is_intraday_data_stale`` (unscoped caller key),
        so a Redis error means "no metadata" exactly as the per-key reads did.
        """
        if not self._redis_client or not calendar_market_by_symbol:
            return {}
        keys_by_symbol = {
            symbol: [
                self._redis_fetch_meta_key(symbol, market=key_market)
                for key_market in self._metadata_key_markets(None, calendar_market)
            ]
            for symbol, calendar_market in calendar_market_by_symbol.items()
        }
        # Full-universe reads use the long-timeout bulk client in bounded chunks,
        # like get_many; a failing chunk loses only its own symbols' metadata.
        client = get_bulk_redis_client() or self._redis_client
        chunk_size = max(1, int(getattr(settings, "redis_pipeline_chunk_size", 500) or 500))
        symbols = list(keys_by_symbol)
        meta_by_symbol: Dict[str, Optional[Dict]] = {}
        for start in range(0, len(symbols), chunk_size):
            chunk = symbols[start:start + chunk_size]
            try:
                pipeline = client.pipeline()
                for symbol in chunk:
                    for key in keys_by_symbol[symbol]:
                        pipeline.get(key)
                # Per-command errors (e.g. WRONGTYPE on one corrupted key) come back
                # in place, so only that key reads as missing, as per-key reads did.
                raw_results = iter(pipeline.execute(raise_on_error=False))
            except Exception as exc:
                logger.error("Error batch-reading fetch metadata: %s", exc, exc_info=True)
                continue
            for symbol in chunk:
                meta_by_symbol[symbol] = latest_fetch_metadata(
                    self._parse_fetch_metadata(next(raw_results))
                    for _ in keys_by_symbol[symbol]
                )
        return meta_by_symbol

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

            start_date = end_date - timedelta(days=_period_days(period))

            # Column select: several times cheaper than loading ORM entities (#418).
            prices = db.query(
                *(getattr(StockPrice, column) for column in STOCK_PRICE_ROW_COLUMNS[1:])
            ).filter(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date.asc()).all()

            if not prices or len(prices) < 50:  # Need substantial data
                logger.debug(f"Insufficient cached data for {symbol} ({len(prices) if prices else 0} rows)")
                return None, None

            df = normalize_price_frame(stock_price_frame(prices, include_adj_close=True), min_rows=50)
            if df is None:
                logger.debug(f"Insufficient finite cached data for {symbol}")
                return None, None

            # Get last date
            last_date = df.index[-1].date()

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
        period: str,
        *,
        minimum_rows: int = 50,
        raise_on_error: bool = False,
    ) -> Dict[str, tuple[Optional[pd.DataFrame], Optional[date]]]:
        """
        Bulk fetch from database for multiple symbols.

        More efficient than calling _get_from_database() repeatedly
        because it uses a single DB session for all queries.

        Args:
            symbols: List of stock symbols to fetch
            period: Time period ("1y", "2y", "5y", "max")
            minimum_rows: Fewest structurally valid rows required per symbol
            raise_on_error: Raise a query failure instead of reporting every
                symbol as missing (read paths fall back; write paths must not
                mistake an outage for an empty history)

        Returns:
            Dict mapping symbol to (DataFrame, last_date) or (None, None)
        """
        if not symbols:
            return {}

        minimum_rows = max(1, int(minimum_rows))

        db = self._session_factory()
        results = {}

        try:
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=_period_days(period))

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
                    *(getattr(StockPrice, column) for column in STOCK_PRICE_ROW_COLUMNS)
                ).filter(
                    and_(
                        StockPrice.symbol.in_(chunk_symbols),
                        StockPrice.date >= start_date,
                        StockPrice.date <= end_date
                    )
                ).order_by(StockPrice.symbol, StockPrice.date.asc()).all()

                # One frame per chunk, cut by symbol (#418); rows are ordered by symbol.
                frames = stock_price_frames_by_symbol(rows, include_adj_close=True)

                for symbol in chunk_symbols:
                    frame = frames.get(symbol)

                    if frame is None or len(frame) < minimum_rows:
                        results[symbol] = (None, None)
                        continue

                    df = normalize_price_frame(frame, min_rows=minimum_rows)
                    if df is None:
                        results[symbol] = (None, None)
                        continue

                    last_date = df.index[-1].date()
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
            if raise_on_error:
                raise
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
            raw = self._fetch_direct_historical_data(symbol, period=period)

            data = normalize_price_frame(raw)
            if data is None:
                logger.warning("Failed to fetch finite price data for %s", symbol)
                return None

            logger.info(f"Fetched {symbol}: {len(data)} rows")

            # Persist first: fetch metadata must only vouch for committed rows.
            if self._store_in_database(symbol, data) and not self._unreplaced_rejected_rows(
                {symbol: raw}, {symbol: data}
            ):
                self._store_recent_in_redis(symbol, data, market=market, period=period)

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
            provider_symbol = cn_price_symbol_for_native_provider(
                symbol,
                local_code=identity.local_code,
                canonical_symbol=identity.canonical_symbol,
            )
            if provider_symbol is None:
                return None
            return CnMarketDataService().daily_ohlcv_dataframe(
                provider_symbol,
                period=period,
            )
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
            cached_data = normalize_price_frame(cached_data)
            if cached_data is None:
                return self._fetch_full_and_cache(symbol, period, market=market)

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
            raw_new_data = self._fetch_direct_historical_data(symbol, period="7d")

            new_data = normalize_price_frame(raw_new_data)
            if new_data is None:
                logger.warning(f"Failed to fetch finite incremental data for {symbol}")
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

            new_data_filtered = normalize_price_frame(new_data_filtered)
            if new_data_filtered is None:
                logger.info(f"No finite new data available for {symbol}")
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
            cutoff_date = today - timedelta(days=_period_days(period))

            merged_data = merged_data[merged_data.index >= pd.Timestamp(cutoff_date)]
            merged_data = normalize_price_frame(merged_data)
            if merged_data is None:
                logger.warning("Merged %s data produced no finite close rows", symbol)
                return None

            logger.info(f"Merged data for {symbol}: {len(merged_data)} total rows")

            # Persist only new/updated rows first; fetch metadata must only
            # vouch for committed rows, so Redis is updated after the DB.
            # The merged history is the database window for ``period`` plus the top-up.
            if self._store_in_database(symbol, new_data_filtered) and not self._unreplaced_rejected_rows(
                {symbol: raw_new_data}, {symbol: new_data}
            ):
                self._store_recent_in_redis(
                    symbol, merged_data, market=market, period=period, from_database=True
                )

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
        *,
        stamp_fetch_metadata: bool = True,
        period: str | None = None,
        from_database: bool = False,
    ) -> None:
        """
        Store historical data (up to 5 years) in Redis for fast access.

        Stores full 5-year data to support volume breakthrough analysis
        and Minervini 200-day MA calculations without requiring database fallback.

        ``period`` is the window ``data`` was read or fetched with, when known,
        and ``from_database`` says its history is a database read of that window;
        ``get_many`` only serves the frame to requests it covers (see
        ``_mark_coverage``).

        Also stores fetch metadata for intraday staleness detection, unless the
        frame is a warm copy of DB rows (``stamp_fetch_metadata=False``): that is
        not a provider fetch, so it must not vouch for the DB row later.
        """
        if not self._redis_client:
            return

        try:
            data = normalize_price_frame(data)
            if data is None:
                return

            # Keep last 5 years (1825 days) for volume analysis
            cutoff_datetime = datetime.now() - timedelta(days=self.RECENT_DAYS)
            # Convert to pd.Timestamp to ensure compatibility with pandas index
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                cutoff_date = pd.Timestamp(cutoff_datetime, tz=data.index.tz)
            else:
                cutoff_date = pd.Timestamp(cutoff_datetime)

            recent_data = self._mark_coverage(
                data[data.index >= cutoff_date],
                period,
                from_database=from_database,
            )

            if recent_data.empty:
                return

            redis_key = self._redis_recent_key(symbol, market=market)

            self._redis_client.setex(
                redis_key,
                self._cache_policy.ttl_seconds("price", market=market),
                encode_frame(recent_data),
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
            if stamp_fetch_metadata:
                self._store_fetch_metadata(symbol, market=market)

            logger.debug(f"Cached {symbol} recent data in Redis ({len(recent_data)} rows)")

        except Exception as e:
            logger.error(f"Error storing {symbol} in Redis: {e}", exc_info=True)

    def _fetch_metadata_payload(self, calendar_market: str, now_et: datetime) -> Dict[str, Any]:
        """Fetch metadata for a fetch at ``now_et``, judged by ``calendar_market``'s session.

        Readers derive staleness from ``fetch_timestamp``; the flags are kept for
        diagnostics and for readers still on the pre-calendar rule during rollout.
        """
        try:
            partial = self._freshness_policy.partial_session_day(now_et, calendar_market) is not None
        except Exception:
            if calendar_market != "US":
                logger.warning("Calendar unavailable for %s; marking fetch intraday", calendar_market)
            partial = is_market_open(now_et) if calendar_market == "US" else True
        return {
            "fetch_timestamp": now_et.isoformat(),
            "market": calendar_market,
            "market_was_open": partial,
            "data_type": "intraday" if partial else "closing",
            "needs_refresh_after_close": partial,
        }

    def _store_fetch_metadata(self, symbol: str, market: str | None = None) -> None:
        """
        Store metadata about when data was fetched for staleness detection.

        Tracks:
        - fetch_timestamp: When data was fetched
        - market_was_open: Whether market was open at fetch time
        - data_type: 'intraday' if fetched during market hours, 'closing' otherwise
        - needs_refresh_after_close: True if this is intraday data
        - market: the market whose session the flags describe
        """
        if not self._redis_client:
            return

        try:
            calendar_market = self._calendar_markets([symbol], market=market)[symbol]
            fetch_meta = self._fetch_metadata_payload(calendar_market, get_eastern_now())

            meta_key = self._redis_fetch_meta_key(symbol, market=market)
            self._redis_client.setex(
                meta_key,
                self._cache_policy.ttl_seconds("price", market=market),
                json.dumps(fetch_meta)
            )

            if fetch_meta["needs_refresh_after_close"]:
                logger.debug(f"Stored fetch metadata for {symbol}: intraday data, needs refresh after close")

        except Exception as e:
            logger.error(f"Error storing fetch metadata for {symbol}: {e}", exc_info=True)

    @staticmethod
    def _parse_fetch_metadata(raw: Any) -> Optional[Dict]:
        if not raw or isinstance(raw, Exception):
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

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
        market: str = "US",
        now: Optional[datetime] = None,
    ) -> bool:
        """Return True when the cached bar was fetched mid-session and that session has closed."""
        return self._freshness_policy.is_fetch_metadata_stale(meta, market=market, now=now)

    def _is_intraday_data_stale(
        self,
        symbol: str,
        market: str | None = None,
        *,
        calendar_market: str | None = None,
    ) -> bool:
        """
        Check if cached data holds a partial bar from a now-completed session.

        ``market`` selects the metadata key (``None`` -> US key, as written);
        ``calendar_market`` is the symbol's own market, resolved when omitted.
        This catches data fetched mid-session (e.g. 2 PM) whose "today" bar is
        incomplete once that market's session has closed.
        """
        calendar_market = calendar_market or self._calendar_markets([symbol], market=market)[symbol]
        # Writers split between the caller's key (None -> US) and the symbol's
        # own market key; judge the most recent write.
        meta = latest_fetch_metadata(
            self._get_fetch_metadata(symbol, market=key_market)
            for key_market in self._metadata_key_markets(market, calendar_market)
        )
        if not meta:
            return False
        is_stale = self._is_fetch_metadata_stale(meta, market=calendar_market)
        if is_stale:
            logger.debug(
                "%s: intraday data is stale for market %s (fetched mid-session, session now closed)",
                symbol,
                calendar_market,
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
            expected_date = self._get_expected_data_date("US")
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

    def _get_expected_data_date(self, market: str | None = None) -> Optional[date]:
        """Latest session the cache must cover: the market's last completed trading day.

        ``MarketCalendarService`` counts a session complete 30 minutes after its
        close. Calendar failure falls back to the pre-calendar US rule for US and
        to ``None`` (stale) elsewhere.
        """
        market = (market or "US").upper()
        try:
            return self._freshness_policy.last_completed_trading_day(market)
        except Exception as exc:
            if market == "US":
                return self._legacy_us_expected_data_date()
            logger.warning("Calendar unavailable for %s expected session: %s", market, exc)
            return None

    def _legacy_us_expected_data_date(self) -> Optional[date]:
        """
        Calculate the date that cache should have data for (US clock only).

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

    def _unreplaced_rejected_rows(
        self,
        raw_by_symbol: Mapping[str, Optional[pd.DataFrame]],
        normalized_by_symbol: Mapping[str, Optional[pd.DataFrame]],
    ) -> set[str]:
        """Symbols whose dropped latest bar left a stored row for that date in place.

        Normalization drops non-finite bars (a provider returns today's bar with
        a NaN close while data is delayed). If the DB already holds a row for
        that date, usually a partial mid-session bar, this fetch did not replace
        it, so its metadata must not vouch for it. Without a stored row (an
        illiquid ticker's no-trade day) there is nothing to vouch for.
        """
        rejected = {
            symbol: day
            for symbol, raw in raw_by_symbol.items()
            if (day := _rejected_latest_date(raw, normalized_by_symbol.get(symbol))) is not None
        }
        if not rejected:
            return set()
        db = None
        try:
            db = self._session_factory()
            rows = db.query(StockPrice.symbol, StockPrice.date).filter(
                StockPrice.symbol.in_(list(rejected)),
                StockPrice.date.in_(set(rejected.values())),
            ).all()
            return {symbol for symbol, day in rows if rejected.get(symbol) == day}
        except Exception as exc:
            # Cannot tell whether a partial row is stored: do not vouch for it.
            logger.warning("Could not check stored rows for dropped latest bars: %s", exc)
            return set(rejected)
        finally:
            if db is not None:
                db.close()

    def _store_in_database(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store price data in database (StockPrice table).

        Uses insert for historical rows and upsert/replace for the latest row so
        intraday partial bars can be corrected after the close.

        Returns True once the rows are committed (or already present); callers
        stamp fetch metadata only then, so it never vouches for a failed write.
        """
        db = self._session_factory()

        try:
            data = normalize_price_frame(data)
            if data is None:
                return False
            # Reset index to get Date as a column
            df = data.reset_index()
            if 'Date' not in df.columns and len(df.columns) > 0:
                df = df.rename(columns={df.columns[0]: 'Date'})
            if df.empty:
                return False

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
                    price_dict = stock_price_row_from_ohlcv(
                        symbol=symbol,
                        row_date=row_date,
                        row=row,
                    )
                    if price_dict is None:
                        continue
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
            return True

        except Exception as e:
            logger.error(f"Error storing {symbol} in database: {e}", exc_info=True)
            db.rollback()
            return False

        finally:
            db.close()

    def _is_data_fresh(
        self,
        last_date: date,
        max_age_days: int = 1,
        *,
        market: str = "US",
    ) -> bool:
        """
        Check if cached data covers ``market``'s last completed session.

        Delegates to _get_expected_data_date(market), which follows the
        market's calendar: sessions, holidays and early closes.
        """
        del max_age_days
        return self._freshness_policy.is_data_fresh(last_date, market)

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
            also_store_db: Whether to also store in database (default True).
                With False the caller owns the DB write and must call this only
                after it succeeded, because the Redis write stamps fetch metadata.
        """
        if data is None or data.empty:
            logger.warning(f"Cannot cache {symbol}: data is empty")
            return
        raw = data
        data = normalize_price_frame(raw)
        if data is None:
            logger.warning(f"Cannot cache {symbol}: no finite close rows")
            return

        try:
            # Persist first: fetch metadata must only vouch for committed rows.
            if also_store_db:
                if not self._store_in_database(symbol, data):
                    return
                logger.debug(f"Stored {symbol} in database ({len(data)} rows)")

            if self._unreplaced_rejected_rows({symbol: raw}, {symbol: data}):
                return
            self._store_recent_in_redis(symbol, data, market=market)
            logger.debug(f"Stored {symbol} in Redis cache ({len(data)} rows)")

        except Exception as e:
            logger.error(f"Error caching {symbol}: {e}")

    def store_batch_in_cache(
        self,
        batch_data: Dict[str, pd.DataFrame],
        also_store_db: bool = True,
        market: str | None = None,
        period: str | None = None,
        from_database: bool = False,
    ) -> int:
        """
        Store multiple symbols' price data in cache using Redis pipeline.

        Uses a single Redis pipeline for all symbols in the batch, reducing
        from 3 * N round-trips to 1 round-trip for N symbols.

        Args:
            batch_data: Dict mapping symbol to price DataFrame
            also_store_db: Whether to also store in database (default True).
                With False the caller owns the DB write and must call this only
                after it succeeded, because the Redis write stamps fetch metadata.
            period: Period the frames were fetched with. Pass it when it can be
                shorter than 2y, so ``get_many`` does not serve them to longer
                requests.
            from_database: The frames are committed rows read back for ``period``,
                so they vouch for that whole window (up to 5y), not just 2y.

        Returns:
            Number of symbols successfully cached (0 when the DB write failed)
        """
        if not batch_data:
            return 0
        raw_batch = batch_data
        batch_data = normalize_price_batch(raw_batch)
        if not batch_data:
            return 0

        # Persist first: fetch metadata must only vouch for committed rows. The
        # batch is one transaction; a failure raises before Redis is touched, so
        # the older frame + metadata pair stays in place for every symbol.
        if also_store_db:
            self._store_batch_in_database(batch_data)

        # Keep the older pair for symbols whose dropped latest bar left a stored
        # (partial) row in place; everything else is cached as usual.
        unreplaced = self._unreplaced_rejected_rows(raw_batch, batch_data)
        if unreplaced:
            batch_data = {s: d for s, d in batch_data.items() if s not in unreplaced}
            if not batch_data:
                return 0

        stored = 0

        # Batch Redis writes using pipeline
        if self._redis_client:
            try:
                now_et = get_eastern_now()
                # Pre-compute fetch metadata once per market in the batch
                calendar_markets = self._calendar_markets(list(batch_data), market=market)
                meta_json_by_market = {
                    calendar_market: json.dumps(self._fetch_metadata_payload(calendar_market, now_et))
                    for calendar_market in set(calendar_markets.values())
                }

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
                        recent_data = self._mark_coverage(
                            data[data.index >= cutoff_date], period, from_database=from_database
                        )
                        if recent_data.empty:
                            continue

                        redis_key = self._redis_recent_key(symbol, market=market)
                        pipeline.setex(
                            redis_key,
                            self._cache_policy.ttl_seconds("price", market=market),
                            encode_frame(recent_data),
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
                            meta_json_by_market[calendar_markets[symbol]],
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
                        self._store_recent_in_redis(
                            symbol, data, market=market, period=period, from_database=from_database
                        )

        return stored

    def store_refreshed_batch(
        self,
        batch_data: Dict[str, pd.DataFrame],
        *,
        period: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
    ) -> int:
        """Store a price refresh under every key namespace readers use.

        Writers split symbols between the US key (callers that omit the market)
        and the symbol's own market key, and a stale partial bar may sit under
        either. Overwrite both so neither keeps serving it; write the DB once,
        first (a failure raises before Redis is touched).

        A top-up fetched for less than 2y (``period``, e.g. the 7d delta; an
        unknown period counts as short) holds too few bars for readers, so
        caching it would replace the full history with a frame every read
        rejects. Cache the committed 5y database history (the Redis window)
        for those symbols instead. A symbol the database returns nothing for
        keeps its fetched frame, stamped with the short period it covers; a
        failed re-read writes nothing, leaving readers to fall back to the
        database rows just committed.
        """
        if not batch_data:
            return 0
        self._store_batch_in_database(batch_data)
        if period is None or PERIOD_DAYS.get(period, 0) >= DEFAULT_PERIOD_DAYS:
            return self._cache_in_reader_namespaces(batch_data, period, market_by_symbol)

        normalized = normalize_price_batch(batch_data)
        unreplaced = self._unreplaced_rejected_rows(batch_data, normalized)
        # Normalized, so a fetch with no usable rows (a suspended stock's NaN
        # bars) stored nothing and gets no fetch metadata vouching for it.
        fetched = {s: frame for s, frame in normalized.items() if s not in unreplaced}
        if not fetched:
            return 0
        # ponytail: re-reads the committed window rather than merging into the
        # Redis frame; that also picks up the persistence policy's corrections.
        try:
            history = self._get_many_from_database(
                list(fetched), "5y", minimum_rows=1, raise_on_error=True
            )
        except Exception as exc:
            logger.error(
                "Committed %d refreshed symbols but could not re-read their history; "
                "leaving their Redis frames for the database fallback: %s",
                len(fetched),
                exc,
                exc_info=True,
            )
            return 0
        full = {s: history[s][0] for s in fetched if history.get(s, (None, None))[0] is not None}
        short = {s: frame for s, frame in fetched.items() if s not in full}
        stored = 0
        if full:
            stored += self._cache_in_reader_namespaces(
                full, "5y", market_by_symbol, from_database=True
            )
        if short:
            stored += self._cache_in_reader_namespaces(short, period, market_by_symbol)
        return stored

    def _cache_in_reader_namespaces(
        self,
        frames: Dict[str, pd.DataFrame],
        period: str | None,
        market_by_symbol: Dict[str, str | None] | None,
        *,
        from_database: bool = False,
    ) -> int:
        """Cache committed frames under the US key and each non-US symbol's own key."""
        stored = self.store_batch_in_cache(
            frames, also_store_db=False, period=period, from_database=from_database
        )
        non_us: Dict[str, Dict[str, pd.DataFrame]] = {}
        calendar_markets = self._calendar_markets(list(frames), market_by_symbol=market_by_symbol)
        for symbol, calendar_market in calendar_markets.items():
            if calendar_market != "US":
                non_us.setdefault(calendar_market, {})[symbol] = frames[symbol]
        for calendar_market, group in non_us.items():
            self.store_batch_in_cache(
                group,
                also_store_db=False,
                market=calendar_market,
                period=period,
                from_database=from_database,
            )
        return stored

    def _store_batch_in_database(self, batch_data: Dict[str, pd.DataFrame]) -> None:
        """
        Store multiple symbols' price data in database in a single transaction.

        Queries existing dates for ALL symbols at once, bulk inserts historical
        rows, and replaces the latest row when it already exists.

        Args:
            batch_data: Dict mapping symbol to price DataFrame

        Raises:
            The database error after rolling back. It is one transaction, so a
            failure means nothing in the batch was persisted.
        """
        if not batch_data:
            return
        batch_data = normalize_price_batch(batch_data)
        if not batch_data:
            return

        db = self._session_factory()

        try:
            price_rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
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
                        price_dict = stock_price_row_from_ohlcv(
                            symbol=symbol,
                            row_date=row_date,
                            row=row,
                        )
                        if price_dict is None:
                            continue
                        price_rows_by_symbol.setdefault(symbol, []).append(price_dict)
                    except (KeyError, TypeError, ValueError, OverflowError) as e:
                        logger.warning(f"Error preparing row for {symbol}: {e}")

            result = persist_stock_price_mappings(db, price_rows_by_symbol, chunk_size=100)
            inserted = result["inserted"]
            updated = result["updated"]
            if inserted or updated:
                db.commit()
                logger.info(
                    "Batch persisted %d price rows for %d symbols (%d inserts, %d latest-day updates)",
                    inserted + updated,
                    len(batch_data),
                    inserted,
                    updated,
                )
            else:
                logger.debug(f"No new rows to persist for batch of {len(batch_data)} symbols")

        except Exception as e:
            logger.error(f"Error in batch database write: {e}", exc_info=True)
            db.rollback()
            # Refresh runners classify this (retrying transient DB errors) and
            # mark the batch failed; swallowing it reported lost writes as done.
            raise

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

    @staticmethod
    def _metadata_key_markets(key_market: str | None, calendar_market: str) -> tuple[str, ...]:
        """Key namespaces that may hold a symbol's fetch metadata, the caller's key first.

        Writers use either no market (the US key, e.g. the daily refresh) or the
        symbol's own market (bulk fallback), so DB-freshness checks read all of
        them; the first entry is the one paired with the caller's Redis frame.
        """
        return tuple(dict.fromkeys((str(key_market or "US").upper(), calendar_market, "US")))

    def _calendar(self) -> "MarketCalendarService":
        if self._market_calendar is None:
            from ..wiring.bootstrap import get_market_calendar_service

            try:
                self._market_calendar = get_market_calendar_service()
            except RuntimeError:
                # Scripts that never initialized runtime services.
                from .market_calendar_service import MarketCalendarService

                self._market_calendar = MarketCalendarService()
        return self._market_calendar

    def _calendar_markets(
        self,
        symbols: list[str],
        *,
        market: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
        fallback_by_symbol: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Market whose calendar judges each symbol's freshness.

        Cache keys keep the caller's market (``None`` -> US key), but the calendar
        must be the symbol's own: explicit caller market, else its active
        universe market, else ``fallback_by_symbol``, else US.
        """
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        for symbol in symbols:
            explicit = self._market_for_symbol(
                symbol, market=market, market_by_symbol=market_by_symbol
            )
            if explicit:
                resolved[symbol] = str(explicit).upper()
            else:
                unresolved.append(symbol)
        if unresolved:
            # ponytail: one indexed universe lookup per call; memoize per process
            # if per-symbol get_historical_data loops show up in profiles.
            try:
                universe = self._active_market_by_symbol(unresolved)
            except Exception:
                logger.debug("Universe market lookup failed; using fallback markets", exc_info=True)
                universe = {}
            registered = _registered_instrument_markets()
            fallback = fallback_by_symbol or {}
            for symbol in unresolved:
                resolved[symbol] = str(
                    universe.get(symbol)
                    or registered.get(str(symbol).upper())
                    or fallback.get(symbol)
                    or "US"
                ).upper()
        return resolved

    def _active_market_by_symbol(self, symbols: list[str]) -> dict[str, str | None]:
        if not symbols:
            return {}
        db = self._session_factory()
        try:
            return {
                row[0]: row[1]
                for row in db.query(StockUniverse.symbol, StockUniverse.market).filter(
                    StockUniverse.symbol.in_(symbols),
                    StockUniverse.active_filter(),
                ).all()
            }
        finally:
            db.close()

    def _store_batch_in_cache_for_market(
        self,
        batch_data: Dict[str, pd.DataFrame],
        *,
        also_store_db: bool,
        market: str | None,
        period: str | None = None,
    ) -> int:
        # Only a sub-2y fetch needs its period recorded; omitting the keyword
        # otherwise keeps two-argument test doubles working, as with ``market``.
        kwargs: Dict[str, Any] = {"also_store_db": also_store_db}
        if PERIOD_DAYS.get(period, DEFAULT_PERIOD_DAYS) < DEFAULT_PERIOD_DAYS:
            kwargs["period"] = period
        if market is None:
            return self.store_batch_in_cache(batch_data, **kwargs)
        try:
            return self.store_batch_in_cache(batch_data, market=market, **kwargs)
        except TypeError as exc:
            if "market" not in str(exc):
                raise
            return self.store_batch_in_cache(batch_data, **kwargs)

    @classmethod
    def _trim_to_period(cls, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Cut a Redis frame (up to RECENT_DAYS) to the window the DB tier returns.

        Relies on the ascending index the freshness check (``df.index[-1]``)
        already assumes; a binary search is several times cheaper than a mask.
        """
        days = _period_days(period)
        if days >= cls.RECENT_DAYS:
            return df
        cutoff = pd.Timestamp(
            datetime.now().date() - timedelta(days=days),
            tz=getattr(df.index, "tz", None),
        )
        start = df.index.searchsorted(cutoff)
        if start == 0:
            return df
        # .copy() releases the 5y block and keeps later column writes warning-free.
        trimmed = df.iloc[start:].copy()
        # The cut frame covers only this window, whatever the frame it came from claimed.
        trimmed.attrs = {**trimmed.attrs, _COVERAGE_ATTR: days}
        return trimmed

    @classmethod
    def _mark_coverage(
        cls,
        frame: pd.DataFrame,
        period: str | None,
        *,
        from_database: bool = False,
    ) -> pd.DataFrame:
        """Stamp a frame about to be cached with the window it was cut from.

        A database read vouches for its whole window. A provider fetch can return
        less than it was asked for, so it may lower the claim below the 2y that
        unstamped frames get (a 1y fetch) but never raise it (a 5y fetch).
        A stamp the frame already carries (it was read from the cache) is never
        raised either, so storing it again cannot widen what it claims.
        """
        claims = [frame.attrs.get(_COVERAGE_ATTR)]
        if period in PERIOD_DAYS:
            limit = cls.RECENT_DAYS if from_database else DEFAULT_PERIOD_DAYS
            claims.append(min(PERIOD_DAYS[period], limit))
        known = [days for days in claims if days is not None]
        if known:
            frame.attrs = {**frame.attrs, _COVERAGE_ATTR: min(known)}
        return frame

    @staticmethod
    def _covers_period(df: pd.DataFrame, period: str) -> bool:
        """Whether a Redis frame was cut from a window at least as long as ``period``."""
        # Unstamped frames (writers that do not know their period, and frames
        # stored before stamping existed) are trusted up to 2y, as before.
        return df.attrs.get(_COVERAGE_ATTR, DEFAULT_PERIOD_DAYS) >= _period_days(period)

    def get_many(
        self,
        symbols: list[str],
        period: str = "2y",
        market: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
        *,
        cache_only: bool = False,
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
            period: Time period needed ("1y", "2y", "5y") - Redis hits are trimmed
                to it and the database fallback queries it, so both tiers return
                the same window
            market: Optional market for homogeneous batches.
            market_by_symbol: Optional per-symbol market map for mixed batches.
            cache_only: Never call a price provider. Symbols Redis cannot serve
                get whatever the database holds, however short or stale, or
                None. Manual scans set this; freshness is decided before the
                scan starts (``services/market_data_freshness.py``).

        Returns:
            Dict mapping symbols to their cached DataFrames (or None if not cached)
        """
        if not symbols:
            return {}

        now_et = get_eastern_now()
        calendar_markets = self._calendar_markets(
            symbols, market=market, market_by_symbol=market_by_symbol
        )
        expected_by_market = {
            calendar_market: self._get_expected_data_date(calendar_market)
            for calendar_market in set(calendar_markets.values())
        }

        if not self._redis_client:
            logger.info("Redis unavailable for bulk get - using database and batch-fetch fallback")
            return self._resolve_bulk_fallback(
                symbols,
                period=period,
                calendar_markets=calendar_markets,
                expected_by_market=expected_by_market,
                now_et=now_et,
                market=market,
                market_by_symbol=market_by_symbol,
                cache_only=cache_only,
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
                    meta_key_counts = []
                    for symbol in chunk_symbols:
                        symbol_market = self._market_for_symbol(
                            symbol,
                            market=market,
                            market_by_symbol=market_by_symbol,
                        )
                        redis_key = self._redis_recent_key(symbol, market=symbol_market)
                        pipeline.get(redis_key)
                        # Every namespace a writer may have used; the caller's key first.
                        meta_key_markets = self._metadata_key_markets(
                            symbol_market, calendar_markets[symbol]
                        )
                        for meta_key_market in meta_key_markets:
                            pipeline.get(self._redis_fetch_meta_key(symbol, market=meta_key_market))
                        meta_key_counts.append(len(meta_key_markets))
                    # One corrupted key (e.g. WRONGTYPE) must only miss its own symbol.
                    chunk_results = pipeline.execute(raise_on_error=False)
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
                results = iter(chunk_results)
                for symbol, meta_key_count in zip(chunk_symbols, meta_key_counts):
                    raw_data = next(results)
                    if isinstance(raw_data, Exception):
                        raw_data = None  # per-key error: treat as a cache miss
                    metas = [self._parse_fetch_metadata(next(results)) for _ in range(meta_key_count)]
                    # The Redis frame is judged by the metadata written with it
                    # (same key); the shared DB row by the latest write anywhere.
                    meta = metas[0]
                    fetch_meta_by_symbol[symbol] = latest_fetch_metadata(metas)
                    if raw_data:
                        try:
                            # A legacy (pickled) payload decodes to None: a miss.
                            df = normalize_price_frame(decode_frame(raw_data))
                            if df is None:
                                cached_data[symbol] = None
                                redis_misses.append(symbol)
                                continue

                            # Check if Redis data is sufficient for requested period:
                            # at least 200 days, cut from a window no shorter than the request
                            if len(df) >= 200 and self._covers_period(df, period):
                                # Check freshness using the pre-computed per-market expected session (B2 optimization)
                                last_date = df.index[-1]
                                if hasattr(last_date, 'date'):
                                    last_date = last_date.date()

                                calendar_market = calendar_markets[symbol]
                                expected_date = expected_by_market[calendar_market]
                                is_fresh = last_date >= expected_date if expected_date else False
                                meta_is_stale = self._is_fetch_metadata_stale(
                                    meta, market=calendar_market, now=now_et
                                )
                                if meta_is_stale:
                                    is_fresh = False
                                if is_fresh:
                                    cached_data[symbol] = self._trim_to_period(df, period)
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
                                logger.debug(
                                    f"Bulk cache HIT but INSUFFICIENT for {symbol} "
                                    f"(Redis has {len(df)} days, need 200+ covering {period})"
                                )
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
                        calendar_markets=calendar_markets,
                        expected_by_market=expected_by_market,
                        now_et=now_et,
                        fetch_meta_by_symbol=fetch_meta_by_symbol,
                        market=market,
                        market_by_symbol=market_by_symbol,
                        cache_only=cache_only,
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
        calendar_markets: Dict[str, str],
        expected_by_market: Dict[str, Optional[date]],
        now_et: datetime,
        fetch_meta_by_symbol: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
        market: str | None = None,
        market_by_symbol: Dict[str, str | None] | None = None,
        cache_only: bool = False,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Resolve a multi-symbol cache miss via one DB query and one optional batch fetch.

        This keeps the non-Redis path efficient and reuses the same freshness logic
        as the Redis-assisted bulk path. With ``cache_only`` there is no fetch:
        see ``get_many``.
        """
        if not symbols:
            return {}

        fetch_meta_by_symbol = fetch_meta_by_symbol or {}
        cached_data: Dict[str, Optional[pd.DataFrame]] = {}

        if cache_only:
            # A new listing has fewer than the loader's default 50 bars; the
            # scanner decides what is enough history, not the cache.
            db_results = self._get_many_from_database(symbols, period, minimum_rows=1)
        else:
            db_results = self._get_many_from_database(symbols, period)
        db_hits = []
        yfinance_needed = []
        caller_market_by_symbol = {
            symbol: self._market_for_symbol(
                symbol,
                market=market,
                market_by_symbol=market_by_symbol,
            )
            for symbol in symbols
        }
        active_market_by_symbol = self._active_market_by_symbol(
            [symbol for symbol, symbol_market in caller_market_by_symbol.items() if symbol_market is None]
        )

        for symbol in symbols:
            df, last_date = db_results.get(symbol, (None, None))
            symbol_market = caller_market_by_symbol[symbol] or active_market_by_symbol.get(symbol)
            calendar_market = calendar_markets[symbol]
            expected_date = expected_by_market[calendar_market]
            is_fresh = (last_date >= expected_date) if (last_date and expected_date) else False
            if self._is_fetch_metadata_stale(
                fetch_meta_by_symbol.get(symbol), market=calendar_market, now=now_et
            ):
                is_fresh = False
            if df is not None and not df.empty and is_fresh:
                cached_data[symbol] = df
                db_hits.append(symbol)
                if self._redis_client:
                    self._store_recent_in_redis(
                        symbol,
                        df,
                        market=symbol_market,
                        stamp_fetch_metadata=False,
                        period=period,
                        from_database=True,
                    )
            else:
                yfinance_needed.append(symbol)

        logger.info("Database query: %d hits, %d need yfinance", len(db_hits), len(yfinance_needed))

        if cache_only:
            for symbol in yfinance_needed:
                cached_data[symbol] = db_results.get(symbol, (None, None))[0]
            return cached_data

        if not yfinance_needed:
            return cached_data

        from .bulk_data_fetcher import BulkDataFetcher

        missing_active_lookup = [symbol for symbol in yfinance_needed if symbol not in active_market_by_symbol]
        active_market_by_symbol.update(self._active_market_by_symbol(missing_active_lookup))
        # Key-market instruments (e.g. ^HSI) are refreshed outside stock_universe;
        # without this a stale row for them would be served instead of refetched.
        registered = _registered_instrument_markets()
        for symbol in missing_active_lookup:
            registered_market = registered.get(str(symbol).upper())
            if symbol not in active_market_by_symbol and registered_market:
                active_market_by_symbol[symbol] = registered_market

        inactive_symbols =[symbol for symbol in yfinance_needed if symbol not in active_market_by_symbol]
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
                fetch_kwargs = {"period": period}
                if group_market is None:
                    fetch_kwargs["start_batch_size"] = getattr(
                        settings,
                        'price_cache_yfinance_batch_size',
                        100,
                    )
                else:
                    fetch_kwargs["market"] = group_market
                provider_results = bulk_fetcher.fetch_prices_in_batches(
                    market_symbols,
                    **fetch_kwargs,
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
                # A read path: the fetched frames are already in cached_data, so
                # a failed write only means they are not cached for next time.
                try:
                    self._store_batch_in_cache_for_market(
                        batch_to_store,
                        also_store_db=True,
                        market=group_market,
                        period=period,
                    )
                except Exception as exc:
                    logger.error(
                        "Could not persist %d fetched price frames (%s): %s",
                        len(batch_to_store),
                        group_market or "US",
                        exc,
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

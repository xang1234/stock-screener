"""
Cache Manager for orchestrating all caching operations.

Provides centralized cache coordination, warming, and statistics.
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date, datetime
from sqlalchemy.orm import Session

from ..database import SessionLocal
from .benchmark_cache_service import BenchmarkCacheService
from .benchmark_registry_service import benchmark_registry
from .benchmark_resolution import latest_benchmark_data_date
from .cache.market_cache_policy import market_cache_policy
from .price_cache_service import PriceCacheService
from .redis_pool import get_redis_client, is_redis_enabled
from ..utils.market_hours import is_market_open, format_market_status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkWarmResult:
    """One benchmark period's warm.

    ``ok`` only when a candidate carries a bar for ``required_date``, the
    market's last completed session. ``latest_date`` is the newest bar on or
    before it that any candidate reached.
    """

    ok: bool
    required_date: date | None
    latest_date: date | None = None
    reason: str | None = None

    def describe(self) -> str:
        latest = self.latest_date.isoformat() if self.latest_date else "none"
        required = self.required_date.isoformat() if self.required_date else "unknown"
        return f"{self.reason}: latest {latest}, required {required}"


class CacheManager:
    """
    Centralized cache management and orchestration.

    Coordinates:
    - Market benchmark caching (BenchmarkCacheService)
    - Stock price caching (PriceCacheService)
    - Cache warming operations
    - Cache statistics and monitoring
    """

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize cache manager.

        Args:
            db: Database session (optional)
        """
        # Use shared connection pool for efficiency
        self.redis_client = get_redis_client()
        if self.redis_client:
            logger.info("Cache manager connected to Redis (using shared pool)")
        elif is_redis_enabled():
            logger.warning("Redis connection failed. Cache manager will use fallback mode.")
        else:
            logger.info("Redis disabled for this runtime. Cache manager will use fallback mode.")

        # Initialize cache services (they'll use their own pool connections)
        self.benchmark_cache = BenchmarkCacheService(
            redis_client=self.redis_client,
            session_factory=SessionLocal,
        )
        self.price_cache = PriceCacheService(
            redis_client=self.redis_client,
            session_factory=SessionLocal,
        )

        self.db = db

    def warm_benchmark_cache(self, period: str = "2y", market: str = "US") -> BenchmarkWarmResult:
        """Warm one period of the market's benchmark cache.

        The warm requires a bar for the market's last completed session, so a
        provider history that ends a session earlier (Yahoo publishes a close
        late, or drops it) is a failed warm, not a current cache. During a
        session that is the prior day: today's partial bar is neither
        required nor counted. A calendar that cannot name that session fails
        the warm too; warming blind is what let stale data count as fresh.
        """
        from ..wiring.bootstrap import get_market_calendar_service

        required_date: date | None = None
        try:
            required_date = get_market_calendar_service().last_completed_trading_day(market)
            benchmark_symbol = self.benchmark_cache.get_benchmark_symbol(market)
            logger.info(
                "Warming %s benchmark cache (%s, %s) through %s...",
                market, benchmark_symbol, period, required_date,
            )
            start_time = time.time()

            resolution = self.benchmark_cache.resolve_benchmark_bundle(
                market=market,
                period=period,
                force_refresh=False,
                required_as_of_date=required_date,
            )
            if resolution.bundle is None:
                reason = resolution.error or "no_benchmark_data"
                latest_date = max(
                    (status.latest_date for status in resolution.candidate_statuses if status.latest_date),
                    default=None,
                )
                logger.warning(
                    "Failed to warm %s benchmark cache (%s): %s; candidates: %s",
                    market, period, reason, resolution.candidate_diagnostics(),
                )
                return BenchmarkWarmResult(
                    ok=False, required_date=required_date, latest_date=latest_date, reason=reason,
                )

            elapsed = time.time() - start_time
            logger.info(
                "✓ %s benchmark cache warmed: %s rows (%s) in %.2fs",
                market, len(resolution.bundle.data), resolution.bundle.benchmark_symbol, elapsed,
            )
            return BenchmarkWarmResult(
                ok=True,
                required_date=required_date,
                latest_date=latest_benchmark_data_date(resolution.bundle.data, as_of_date=required_date),
            )

        except Exception as e:
            logger.error(f"Error warming {market} benchmark cache: {e}", exc_info=True)
            return BenchmarkWarmResult(ok=False, required_date=required_date, reason=f"error: {e}")

    def warm_price_cache(
        self,
        symbols: List[str],
        period: str = "2y",
        batch_size: int = 50,
        rate_limit: float = 1.0,
        force_refresh: bool = True
    ) -> Dict:
        """
        Warm price cache for multiple symbols using bulk fetching (Phase 2 optimization).

        Uses the shared ``yf.download()`` batch adapter for scheduled price refreshes.

        Args:
            symbols: List of stock symbols to warm
            period: Time period to cache
            batch_size: Symbols per bulk fetch batch (default 50)
            rate_limit: Seconds to wait between batch requests
            force_refresh: Force fetch even if cached (default True for warming)

        Returns:
            Dict with warming statistics
        """
        stats = {
            'total': len(symbols),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': time.time(),
            'errors': []
        }

        logger.info(f"Warming price cache for {len(symbols)} symbols using bulk fetching (batch_size={batch_size})...")

        # PHASE 2: Use bulk fetcher for efficient batch operations
        from ..services.bulk_data_fetcher import BulkDataFetcher
        bulk_fetcher = BulkDataFetcher()

        # If not forcing refresh, filter out already-cached symbols
        symbols_to_fetch = symbols
        if not force_refresh:
            symbols_to_fetch = []
            for symbol in symbols:
                cache_stats = self.price_cache.get_cache_stats(symbol)
                if cache_stats['redis_cached']:
                    logger.debug(f"Skipping {symbol} - already in Redis cache")
                    stats['skipped'] += 1
                else:
                    symbols_to_fetch.append(symbol)

            logger.info(f"After cache check: {len(symbols_to_fetch)} symbols need fetching, {stats['skipped']} already cached")

        if not symbols_to_fetch:
            logger.info("All symbols already cached")
            return stats

        # Process in batches using bulk fetcher
        total_batches = (len(symbols_to_fetch) + batch_size - 1) // batch_size
        consecutive_backoffs = 0  # Track consecutive rate-limited batches

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(symbols_to_fetch))
            batch = symbols_to_fetch[start_idx:end_idx]

            logger.info(f"Batch {batch_num + 1}/{total_batches}: Bulk fetching {len(batch)} symbols...")

            fetch_failures: set[str] = set()
            try:
                bulk_data = bulk_fetcher.fetch_prices_in_batches(
                    batch,
                    period=period,
                )

                batch_to_store = {}
                batch_failed = 0
                for symbol, data in bulk_data.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        price_df = data['price_data']
                        if price_df is not None and not price_df.empty:
                            batch_to_store[symbol] = price_df
                            logger.debug(f"✓ {symbol}: Fetched {len(price_df)} rows")
                        else:
                            stats['failed'] += 1
                            batch_failed += 1
                            fetch_failures.add(symbol)
                            stats['errors'].append(f"{symbol}: Empty data")
                            logger.debug(f"✗ {symbol}: Empty data")
                    else:
                        stats['failed'] += 1
                        batch_failed += 1
                        fetch_failures.add(symbol)
                        error_msg = data.get('error', 'No data')
                        stats['errors'].append(f"{symbol}: {error_msg}")
                        logger.debug(f"✗ {symbol}: {error_msg}")

                if batch_to_store:
                    self.price_cache.store_batch_in_cache(
                        batch_to_store,
                        also_store_db=True,
                        period=period,
                    )
                # Count only once stored: a store that raises persisted nothing.
                stats['successful'] += len(batch_to_store)

                # Progress logging
                logger.info(
                    f"Batch {batch_num + 1}/{total_batches} completed: "
                    f"{stats['successful']}/{len(symbols_to_fetch)} successful so far"
                )

                # Batch-level backoff: if >50% of batch failed, likely rate-limited
                batch_failure_rate = batch_failed / len(batch) if batch else 0
                if batch_failure_rate > 0.5:
                    consecutive_backoffs += 1
                    backoff_time = min(60 * (2 ** (consecutive_backoffs - 1)), 480)  # 60, 120, 240, 480 max
                    logger.warning(
                        f"High failure rate ({batch_failed}/{len(batch)}) in batch {batch_num + 1}. "
                        f"Backing off {backoff_time}s before next batch."
                    )
                    time.sleep(backoff_time)
                else:
                    consecutive_backoffs = 0  # Reset on successful batch
                    # Normal rate limiting between batches (Redis-backed distributed limiter)
                    if rate_limit > 0 and batch_num < total_batches - 1:
                        from ..wiring.bootstrap import get_rate_limiter

                        get_rate_limiter().wait("yfinance:batch", min_interval_s=rate_limit)

            except Exception as e:
                logger.error(f"Error warming batch {batch_num + 1}: {e}", exc_info=True)
                # Mark the rest of this batch failed (fetch failures are counted)
                for symbol in batch:
                    if symbol in fetch_failures:
                        continue
                    stats['failed'] += 1
                    stats['errors'].append(f"{symbol}: Batch error - {str(e)}")

        stats['end_time'] = time.time()
        stats['duration'] = stats['end_time'] - stats['start_time']

        logger.info(
            f"✓ Price cache warming complete: "
            f"{stats['successful']} successful, {stats['failed']} failed, "
            f"{stats['skipped']} skipped in {stats['duration']:.1f}s"
        )

        return stats

    def get_cache_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.

        Returns:
            Dict with cache statistics for benchmarks, prices, and fundamentals
        """
        stats = {
            'redis_connected': self.redis_client is not None,
            'market_status': format_market_status(),
            'benchmark_cache': {},
            # Backward compatibility key expected by existing response schema/clients.
            'spy_cache': {},
            'price_cache': {
                'total_keys': 0,
                'symbols_cached': 0
            },
            'fundamentals_cache': {
                'total_keys': 0,
                'symbols_cached': 0
            },
            'benchmark_registry': {
                'version': benchmark_registry.TABLE_VERSION,
                'table': benchmark_registry.mapping_table(),
            },
            'redis_memory': None
        }

        if not self.redis_client:
            return stats

        try:
            # Benchmark cache stats by market
            benchmark_stats: Dict[str, Dict[str, Optional[int] | bool | str]] = {}
            for market in benchmark_registry.supported_markets():
                entry = benchmark_registry.get_entry(market)
                primary_symbol = entry.primary_symbol
                fallback_symbol = entry.fallback_symbol
                key_2y = market_cache_policy.key("benchmark", primary_symbol, market=market, parts=("2y",))
                key_1y = market_cache_policy.key("benchmark", primary_symbol, market=market, parts=("1y",))
                fallback_key_2y = (
                    market_cache_policy.key("benchmark", fallback_symbol, market=market, parts=("2y",))
                    if fallback_symbol
                    else None
                )
                fallback_key_1y = (
                    market_cache_policy.key("benchmark", fallback_symbol, market=market, parts=("1y",))
                    if fallback_symbol
                    else None
                )
                has_2y = self.redis_client.exists(key_2y) > 0
                has_1y = self.redis_client.exists(key_1y) > 0
                fallback_has_2y = self.redis_client.exists(fallback_key_2y) > 0 if fallback_key_2y else False
                fallback_has_1y = self.redis_client.exists(fallback_key_1y) > 0 if fallback_key_1y else False
                benchmark_stats[market] = {
                    'symbol': primary_symbol,
                    'fallback_symbol': fallback_symbol,
                    '2y_cached': has_2y,
                    '1y_cached': has_1y,
                    '2y_ttl': self.redis_client.ttl(key_2y) if has_2y else None,
                    '1y_ttl': self.redis_client.ttl(key_1y) if has_1y else None,
                    'fallback_2y_cached': fallback_has_2y,
                    'fallback_1y_cached': fallback_has_1y,
                    'fallback_2y_ttl': self.redis_client.ttl(fallback_key_2y) if fallback_has_2y and fallback_key_2y else None,
                    'fallback_1y_ttl': self.redis_client.ttl(fallback_key_1y) if fallback_has_1y and fallback_key_1y else None,
                }
            stats['benchmark_cache'] = benchmark_stats
            # Backward compatibility for existing consumers.
            stats['spy_cache'] = benchmark_stats.get("US", {})

            # Price cache stats
            price_keys = self.redis_client.keys("price:*")
            stats['price_cache']['total_keys'] = len(price_keys)

            # Count unique price symbols
            price_symbols = set()
            for key in price_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Extract symbol from "price:US:AAPL:recent" and legacy "price:AAPL:recent".
                parts = key_str.split(':')
                if len(parts) >= 4:
                    price_symbols.add(parts[2])
                elif len(parts) >= 2:
                    price_symbols.add(parts[1])

            stats['price_cache']['symbols_cached'] = len(price_symbols)

            # Fundamentals cache stats
            fundamentals_keys = self.redis_client.keys("fundamentals:*")
            stats['fundamentals_cache']['total_keys'] = len(fundamentals_keys)

            # Count unique fundamentals symbols
            fundamentals_symbols = set()
            for key in fundamentals_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Extract symbol from "fundamentals:US:AAPL" and legacy "fundamentals:AAPL".
                parts = key_str.split(':')
                if len(parts) >= 3:
                    fundamentals_symbols.add(parts[2])
                elif len(parts) >= 2:
                    fundamentals_symbols.add(parts[1])

            stats['fundamentals_cache']['symbols_cached'] = len(fundamentals_symbols)

            # Redis memory info
            info = self.redis_client.info('memory')
            stats['redis_memory'] = {
                'used_memory_human': info.get('used_memory_human'),
                'used_memory_peak_human': info.get('used_memory_peak_human'),
            }

        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")

        return stats

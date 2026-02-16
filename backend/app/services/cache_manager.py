"""
Cache Manager for orchestrating all caching operations.

Provides centralized cache coordination, warming, and statistics.
"""
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime
import redis
from sqlalchemy.orm import Session

from .benchmark_cache_service import BenchmarkCacheService
from .price_cache_service import PriceCacheService
from .redis_pool import get_redis_client
from ..config import settings
from ..utils.market_hours import is_market_open, format_market_status

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Centralized cache management and orchestration.

    Coordinates:
    - SPY benchmark caching (BenchmarkCacheService)
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
        else:
            logger.warning("Redis connection failed. Cache manager will use fallback mode.")

        # Initialize cache services (they'll use their own pool connections)
        self.benchmark_cache = BenchmarkCacheService.get_instance(self.redis_client)
        self.price_cache = PriceCacheService.get_instance(self.redis_client)

        self.db = db

    def warm_benchmark_cache(self, period: str = "2y") -> bool:
        """
        Warm SPY benchmark cache.

        Args:
            period: Time period to cache

        Returns:
            True if successful
        """
        try:
            logger.info(f"Warming SPY benchmark cache ({period})...")
            start_time = time.time()

            spy_data = self.benchmark_cache.get_spy_data(period=period, force_refresh=False)

            if spy_data is not None and not spy_data.empty:
                elapsed = time.time() - start_time
                logger.info(f"✓ SPY cache warmed: {len(spy_data)} rows in {elapsed:.2f}s")
                return True
            else:
                logger.warning("Failed to warm SPY cache")
                return False

        except Exception as e:
            logger.error(f"Error warming SPY cache: {e}", exc_info=True)
            return False

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

        Uses yfinance.Tickers() to fetch multiple stocks in batches, which is
        much faster than individual yf.Ticker() calls.

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

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(symbols_to_fetch))
            batch = symbols_to_fetch[start_idx:end_idx]

            logger.info(f"Batch {batch_num + 1}/{total_batches}: Bulk fetching {len(batch)} symbols...")

            try:
                # Bulk fetch this batch using yfinance.Tickers()
                bulk_data = bulk_fetcher.fetch_batch_data(
                    batch,
                    period=period,
                    include_fundamentals=False  # Only warming price cache
                )

                # Store each symbol's data in cache
                for symbol, data in bulk_data.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        # Store in cache
                        self.price_cache.store_in_cache(symbol, data['price_data'], also_store_db=True)
                        stats['successful'] += 1
                        logger.debug(f"✓ {symbol}: Cached {len(data['price_data'])} rows")
                    else:
                        stats['failed'] += 1
                        error_msg = data.get('error', 'No data')
                        stats['errors'].append(f"{symbol}: {error_msg}")
                        logger.debug(f"✗ {symbol}: {error_msg}")

                # Progress logging
                logger.info(
                    f"Batch {batch_num + 1}/{total_batches} completed: "
                    f"{stats['successful']}/{len(symbols_to_fetch)} successful so far"
                )

                # Rate limiting between batches (Redis-backed distributed limiter)
                if rate_limit > 0 and batch_num < total_batches - 1:
                    from .rate_limiter import rate_limiter
                    rate_limiter.wait("yfinance:batch", min_interval_s=rate_limit)

            except Exception as e:
                logger.error(f"Error warming batch {batch_num + 1}: {e}", exc_info=True)
                # Mark all symbols in this batch as failed
                for symbol in batch:
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

    def warm_all_caches(self, symbols: List[str], force_refresh: bool = True) -> Dict:
        """
        Warm all caches (SPY + stock prices).

        Args:
            symbols: List of stock symbols
            force_refresh: Force fetch even if cached (default True)

        Returns:
            Combined statistics with warmed, failed, already_cached counts
        """
        logger.info("=" * 60)
        logger.info("WARMING ALL CACHES")
        logger.info("=" * 60)

        start_time = time.time()

        # 1. Warm SPY benchmark cache
        spy_success = self.warm_benchmark_cache()

        # 2. Warm price cache for all symbols
        price_stats = self.warm_price_cache(
            symbols,
            rate_limit=settings.scan_rate_limit,
            force_refresh=force_refresh
        )

        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(
            f"✓ All caches warmed in {total_time:.1f}s"
        )
        logger.info("=" * 60)

        # Return summary in format expected by prewarm_scan_cache task
        return {
            'warmed': price_stats['successful'],
            'failed': price_stats['failed'],
            'already_cached': price_stats['skipped'],
            'spy_warmed': spy_success,
            'total_time': total_time,
            'details': price_stats
        }

    def get_cache_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.

        Returns:
            Dict with cache statistics for SPY, prices, and fundamentals
        """
        stats = {
            'redis_connected': self.redis_client is not None,
            'market_status': format_market_status(),
            'spy_cache': {},
            'price_cache': {
                'total_keys': 0,
                'symbols_cached': 0
            },
            'fundamentals_cache': {
                'total_keys': 0,
                'symbols_cached': 0
            },
            'redis_memory': None
        }

        if not self.redis_client:
            return stats

        try:
            # SPY cache stats
            spy_key_2y = "benchmark:SPY:2y"
            spy_key_1y = "benchmark:SPY:1y"

            stats['spy_cache'] = {
                '2y_cached': self.redis_client.exists(spy_key_2y) > 0,
                '1y_cached': self.redis_client.exists(spy_key_1y) > 0,
                '2y_ttl': self.redis_client.ttl(spy_key_2y) if self.redis_client.exists(spy_key_2y) else None,
                '1y_ttl': self.redis_client.ttl(spy_key_1y) if self.redis_client.exists(spy_key_1y) else None,
            }

            # Price cache stats
            price_keys = self.redis_client.keys("price:*")
            stats['price_cache']['total_keys'] = len(price_keys)

            # Count unique price symbols
            price_symbols = set()
            for key in price_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Extract symbol from key like "price:AAPL:recent"
                parts = key_str.split(':')
                if len(parts) >= 2:
                    price_symbols.add(parts[1])

            stats['price_cache']['symbols_cached'] = len(price_symbols)

            # Fundamentals cache stats
            fundamentals_keys = self.redis_client.keys("fundamentals:*")
            stats['fundamentals_cache']['total_keys'] = len(fundamentals_keys)

            # Count unique fundamentals symbols
            fundamentals_symbols = set()
            for key in fundamentals_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                # Extract symbol from key like "fundamentals:AAPL"
                parts = key_str.split(':')
                if len(parts) >= 2:
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

    def invalidate_all_caches(self) -> Dict:
        """
        Invalidate all caches (use with caution!).

        Returns:
            Dict with deletion statistics
        """
        if not self.redis_client:
            logger.warning("Redis not available for cache invalidation")
            return {'deleted': 0}

        try:
            logger.warning("Invalidating ALL caches...")

            # Get all cache keys
            benchmark_keys = self.redis_client.keys("benchmark:*")
            price_keys = self.redis_client.keys("price:*")

            total_keys = len(benchmark_keys) + len(price_keys)

            if total_keys > 0:
                # Delete all keys
                if benchmark_keys:
                    self.redis_client.delete(*benchmark_keys)
                if price_keys:
                    self.redis_client.delete(*price_keys)

                logger.warning(f"✓ Invalidated {total_keys} cache keys")

            return {
                'deleted': total_keys,
                'benchmark_keys': len(benchmark_keys),
                'price_keys': len(price_keys)
            }

        except Exception as e:
            logger.error(f"Error invalidating caches: {e}", exc_info=True)
            return {'deleted': 0, 'error': str(e)}

    def invalidate_symbol_cache(self, symbol: str) -> bool:
        """
        Invalidate cache for a specific symbol.

        Args:
            symbol: Stock symbol to invalidate

        Returns:
            True if successful
        """
        try:
            self.price_cache.invalidate_cache(symbol)
            logger.info(f"✓ Invalidated cache for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error invalidating {symbol} cache: {e}")
            return False

    def get_cache_hit_rate(self, window_minutes: int = 60) -> Optional[float]:
        """
        Calculate cache hit rate from Redis stats.

        Args:
            window_minutes: Time window to analyze

        Returns:
            Hit rate percentage (0-100) or None if unavailable
        """
        if not self.redis_client:
            return None

        try:
            info = self.redis_client.info('stats')

            keyspace_hits = info.get('keyspace_hits', 0)
            keyspace_misses = info.get('keyspace_misses', 0)

            total_requests = keyspace_hits + keyspace_misses

            if total_requests == 0:
                return None

            hit_rate = (keyspace_hits / total_requests) * 100
            return round(hit_rate, 2)

        except Exception as e:
            logger.warning(f"Error calculating hit rate: {e}")
            return None

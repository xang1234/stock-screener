"""
Celery tasks for cache warming and maintenance.

Provides background tasks for:
- Daily cache warming after market close
- Weekly full cache refresh
- On-demand cache warming

All data-fetching tasks use the @serialized_data_fetch decorator
to ensure only one task fetches external data at a time.
"""
import logging
from typing import List, Optional
from datetime import datetime

from ..celery_app import celery_app
from ..database import SessionLocal
from ..services.cache_manager import CacheManager
from ..config import settings
from ..utils.market_hours import is_market_open, is_trading_day, get_eastern_now, format_market_status
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


@celery_app.task(name='app.tasks.cache_tasks.warm_spy_cache')
def warm_spy_cache():
    """
    Warm SPY benchmark cache.

    This task is typically run after market close to ensure
    the next day's scans have fresh SPY data cached.

    Returns:
        Dict with task results
    """
    logger.info("=" * 60)
    logger.info("TASK: Warming SPY Benchmark Cache")
    logger.info(f"Market status: {format_market_status()}")
    logger.info("=" * 60)

    try:
        cache_manager = CacheManager()

        # Warm both 1y and 2y periods
        results = {
            '2y': cache_manager.warm_benchmark_cache(period="2y"),
            '1y': cache_manager.warm_benchmark_cache(period="1y"),
            'market_status': format_market_status(),
            'timestamp': datetime.now().isoformat()
        }

        logger.info("✓ SPY cache warming task completed")
        return results

    except Exception as e:
        logger.error(f"Error in warm_spy_cache task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(name='app.tasks.cache_tasks.warm_top_symbols')
def warm_top_symbols(symbols: Optional[List[str]] = None, count: Optional[int] = None):
    """
    Warm cache for symbols.

    Args:
        symbols: List of symbols to warm (optional, will use universe if None)
        count: Number of symbols to warm (None = all active stocks)

    Returns:
        Dict with warming statistics
    """
    # If count is None, fetch ALL active stocks (no limit)
    # If count is provided, limit to that many
    fetch_all = (count is None)

    logger.info("=" * 60)
    if fetch_all:
        logger.info(f"TASK: Warming ALL Active Symbols from Universe")
    else:
        logger.info(f"TASK: Warming Top {count} Symbols")
    logger.info(f"Market status: {format_market_status()}")
    logger.info("=" * 60)

    db = SessionLocal()

    try:
        # If no symbols provided, get from universe
        if symbols is None:
            from ..models.stock_universe import StockUniverse

            # Build query for active stocks
            query = db.query(StockUniverse).filter(
                StockUniverse.is_active == True
            )

            # Only apply limit if count is specified
            if not fetch_all and count is not None:
                query = query.limit(count)

            universe_records = query.all()
            symbols = [record.symbol for record in universe_records]

            if not symbols:
                # Fallback to S&P 500 top stocks
                logger.warning("No active symbols found in universe, using fallback list")
                symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                    'META', 'TSLA', 'BRK.B', 'V', 'JPM',
                    'WMT', 'MA', 'JNJ', 'PG', 'XOM',
                    'UNH', 'HD', 'CVX', 'ABBV', 'KO',
                    'LLY', 'AVGO', 'MRK', 'PEP', 'COST',
                ]
                if not fetch_all and count is not None:
                    symbols = symbols[:count]

        logger.info(f"Warming {len(symbols)} symbols: {', '.join(symbols[:10])}...")

        cache_manager = CacheManager(db)

        # Warm all caches
        results = cache_manager.warm_all_caches(symbols)

        logger.info("✓ Symbol cache warming task completed")
        return results

    except Exception as e:
        logger.error(f"Error in warm_top_symbols task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.cache_tasks.daily_cache_warmup')
@serialized_data_fetch('daily_cache_warmup')
def daily_cache_warmup(self):
    """
    Daily cache warmup task.

    Runs after market close to prepare cache for next trading day.
    Warms SPY + ALL active symbols in the universe.

    Returns:
        Dict with combined results
    """
    logger.info("=" * 80)
    logger.info("TASK: Daily Cache Warmup")
    logger.info(f"Market status: {format_market_status()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Skip on non-trading days (weekends, holidays)
    today = get_eastern_now().date()
    if not is_trading_day(today):
        logger.info(f"Skipping daily cache warmup - {today} is not a trading day")
        return {'skipped': True, 'reason': 'Not a trading day', 'date': today.isoformat()}

    # Check if market is still open (task should run after close)
    if is_market_open():
        logger.warning("Market is still open - cache warmup may fetch incomplete data")

    results = {
        'spy': None,
        'symbols': None,
        'started_at': datetime.now().isoformat()
    }

    try:
        # 1. Warm SPY benchmark cache
        logger.info("\n[1/2] Warming SPY benchmark cache...")
        results['spy'] = warm_spy_cache()

        # 2. Warm ALL active symbols from universe
        logger.info(f"\n[2/2] Warming ALL active symbols from universe...")
        results['symbols'] = warm_top_symbols(count=None)  # None = fetch all active stocks

        results['completed_at'] = datetime.now().isoformat()

        logger.info("=" * 80)
        logger.info("✓ Daily cache warmup completed successfully")
        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.error(f"Error in daily_cache_warmup task: {e}", exc_info=True)
        results['error'] = str(e)
        results['completed_at'] = datetime.now().isoformat()
        return results


@celery_app.task(bind=True, name='app.tasks.cache_tasks.weekly_full_refresh')
@serialized_data_fetch('weekly_full_refresh')
def weekly_full_refresh(self):
    """
    Weekly full cache refresh.

    Cleans up orphaned cache keys (delisted/deactivated symbols),
    warms SPY, then performs a full universe refresh inline.
    Typically runs Sunday morning before markets open.

    NOTE: Previously this task called smart_refresh_cache.delay(mode="full"),
    which spawned a duplicate task on the data_fetch queue. Now the full
    refresh logic is inlined to prevent task pileup.

    Returns:
        Dict with refresh results
    """
    import time
    from ..services.price_cache_service import PriceCacheService
    from ..services.bulk_data_fetcher import BulkDataFetcher

    logger.info("=" * 80)
    logger.info("TASK: Weekly Full Cache Refresh")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    db = SessionLocal()
    refreshed = 0
    failed = 0
    failed_symbols = []

    try:
        from ..models.stock_universe import StockUniverse
        cache_manager = CacheManager(db)
        price_cache = PriceCacheService.get_instance()
        bulk_fetcher = BulkDataFetcher()

        # 1. Clean up orphaned cache keys (symbols no longer in active universe)
        logger.info("\n[1/4] Cleaning up orphaned cache entries...")
        active_symbols = set(
            r.symbol for r in db.query(StockUniverse.symbol)
            .filter(StockUniverse.is_active == True).all()
        )
        orphan_count = cache_manager.cleanup_orphaned_cache_keys(active_symbols)
        logger.info(f"✓ Cleaned up {orphan_count} orphaned cache entries")

        # 2. Warm SPY cache
        logger.info("\n[2/4] Re-warming SPY benchmark cache...")
        spy_results = warm_spy_cache()

        # 3. Get all active symbols ordered by market cap
        logger.info(f"\n[3/4] Preparing full universe refresh ({len(active_symbols)} symbols)...")
        symbols = [
            r.symbol for r in db.query(StockUniverse.symbol)
            .filter(StockUniverse.is_active == True)
            .order_by(StockUniverse.market_cap.desc().nullslast())
            .all()
        ]

        if not symbols:
            logger.warning("No active symbols found in universe")
            price_cache.save_warmup_metadata("completed", 0, 0)
            return {
                'orphans_cleaned': orphan_count,
                'spy': spy_results,
                'refreshed': 0,
                'universe_size': 0,
                'completed_at': datetime.now().isoformat()
            }

        total = len(symbols)

        # 4. Batch fetch all symbols (inline, no child task)
        logger.info(f"\n[4/4] Fetching {total} symbols...")
        from .data_fetch_lock import DataFetchLock
        lock = DataFetchLock.get_instance()
        task_id = self.request.id or 'unknown'

        batch_size = 100
        for batch_start in range(0, total, batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: Fetching {len(batch_symbols)} symbols")

            try:
                batch_results = _fetch_with_backoff(bulk_fetcher, batch_symbols, period="2y")
                batch_to_store = {}
                for symbol, data in batch_results.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        price_df = data['price_data']
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            refreshed += 1
                        else:
                            failed += 1
                            failed_symbols.append(symbol)
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                # Batch store in Redis (pipeline) + DB (single transaction)
                if batch_to_store:
                    price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)
            except Exception as e:
                logger.error(f"Batch {batch_num} error: {e}")
                failed += len(batch_symbols)
                failed_symbols.extend(batch_symbols)

            # Update progress
            progress = min((batch_start + batch_size), total)
            percent = (progress / total) * 100
            self.update_state(state='PROGRESS', meta={
                'current': progress, 'total': total, 'percent': percent,
                'refreshed': refreshed, 'failed': failed
            })
            price_cache.update_warmup_heartbeat(progress, total, percent)

            # Extend lock TTL to prevent expiry during long runs (A5)
            lock.extend_lock(task_id, 3600)

            # Rate limit between batches
            if batch_start + batch_size < total:
                from ..services.rate_limiter import rate_limiter
                rate_limiter.wait("yfinance:batch", min_interval_s=settings.yfinance_batch_rate_limit_interval)

        # Save final metadata
        success_rate = refreshed / total if total > 0 else 0
        status = "completed" if success_rate >= 0.95 else "partial"
        price_cache.save_warmup_metadata(status, refreshed, total)
        price_cache.clear_warmup_heartbeat()

        logger.info("=" * 80)
        logger.info(f"✓ Weekly full refresh completed:")
        logger.info(f"  Orphans cleaned: {orphan_count}")
        logger.info(f"  Refreshed: {refreshed}/{total}")
        logger.info(f"  Failed: {failed}")
        logger.info("=" * 80)

        return {
            'orphans_cleaned': orphan_count,
            'spy': spy_results,
            'status': status,
            'refreshed': refreshed,
            'failed': failed,
            'total': total,
            'universe_size': len(active_symbols),
            'failed_symbols': failed_symbols[:20],
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in weekly_full_refresh task: {e}", exc_info=True)
        price_cache = PriceCacheService.get_instance()
        price_cache.save_warmup_metadata("failed", refreshed, locals().get('total', 0), str(e))
        price_cache.clear_warmup_heartbeat()
        return {
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.cache_tasks.invalidate_cache')
def invalidate_cache(symbol: Optional[str] = None):
    """
    Invalidate cache for a specific symbol or all caches.

    Args:
        symbol: Symbol to invalidate (None = invalidate all)

    Returns:
        Dict with invalidation results
    """
    logger.info(f"TASK: Invalidate cache for {symbol if symbol else 'ALL'}")

    try:
        cache_manager = CacheManager()

        if symbol:
            success = cache_manager.invalidate_symbol_cache(symbol)
            return {
                'symbol': symbol,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
        else:
            results = cache_manager.invalidate_all_caches()
            results['timestamp'] = datetime.now().isoformat()
            return results

    except Exception as e:
        logger.error(f"Error in invalidate_cache task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(name='app.tasks.cache_tasks.get_cache_stats')
def get_cache_stats():
    """
    Get comprehensive cache statistics.

    Returns:
        Dict with cache statistics
    """
    try:
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()
        stats['timestamp'] = datetime.now().isoformat()
        return stats

    except Exception as e:
        logger.error(f"Error in get_cache_stats task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def _prewarm_scan_cache_impl(task, symbol_list: List[str], priority: str = 'normal', db=None) -> dict:
    """
    Core implementation of cache pre-warming.

    Extracted so it can be called from both the Celery task (with lock)
    and from parent tasks like prewarm_all_active_symbols (without
    re-acquiring the lock).

    Args:
        task: Celery task instance (for update_state), or None
        symbol_list: Symbols to warm
        priority: Priority level ('low', 'normal', 'high')
        db: Optional SQLAlchemy session (avoids double-open when called from another task)

    Returns:
        Dict with warming statistics
    """
    total = len(symbol_list)

    logger.info("=" * 80)
    logger.info(f"TASK: Pre-warming Cache for Scan ({total} symbols)")
    logger.info(f"Priority: {priority}")
    logger.info(f"Market status: {format_market_status()}")
    logger.info("=" * 80)

    owns_db = db is None
    if owns_db:
        db = SessionLocal()
    warmed = 0
    failed = 0
    cached = 0

    try:
        cache_manager = CacheManager(db)

        # Process in chunks for progress tracking
        chunk_size = 50
        chunks = [symbol_list[i:i + chunk_size] for i in range(0, total, chunk_size)]

        for chunk_num, chunk in enumerate(chunks, 1):
            chunk_start = datetime.now()

            # Warm this chunk
            result = cache_manager.warm_all_caches(chunk)

            # Update counters
            warmed += result.get('warmed', 0)
            failed += result.get('failed', 0)
            cached += result.get('already_cached', 0)

            # Calculate progress
            completed = min(chunk_num * chunk_size, total)
            progress = (completed / total) * 100

            # Update Celery task state for progress tracking
            if task is not None:
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'current': completed,
                        'total': total,
                        'percent': progress,
                        'warmed': warmed,
                        'failed': failed,
                        'cached': cached
                    }
                )

            # Progress logging
            elapsed = (datetime.now() - chunk_start).total_seconds()
            logger.info(
                f"Chunk {chunk_num}/{len(chunks)}: {len(chunk)} symbols "
                f"in {elapsed:.1f}s | Progress: {completed}/{total} ({progress:.1f}%)"
            )

        logger.info("=" * 80)
        logger.info(f"✓ Cache pre-warming completed:")
        logger.info(f"  Warmed: {warmed}, Failed: {failed}, Already Cached: {cached}")
        logger.info("=" * 80)

        return {
            'warmed': warmed,
            'failed': failed,
            'already_cached': cached,
            'total': total,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in prewarm_scan_cache: {e}", exc_info=True)
        return {
            'error': str(e),
            'warmed': warmed,
            'failed': failed,
            'total': total,
            'timestamp': datetime.now().isoformat()
        }

    finally:
        if owns_db:
            db.close()


@celery_app.task(name='app.tasks.cache_tasks.prewarm_scan_cache', bind=True)
@serialized_data_fetch('prewarm_scan_cache')
def prewarm_scan_cache(self, symbol_list: List[str], priority: str = 'normal'):
    """
    Pre-warm cache for upcoming scan.

    Fetches data in background at controlled rate, so cache is ready
    before scan starts. This allows scans to run from cache without
    hitting API rate limits.

    Args:
        symbol_list: List of symbols to warm
        priority: Priority level ('low', 'normal', 'high')

    Returns:
        Dict with warming statistics
    """
    return _prewarm_scan_cache_impl(self, symbol_list, priority)


@celery_app.task(bind=True, name='app.tasks.cache_tasks.prewarm_all_active_symbols')
@serialized_data_fetch('prewarm_all_active_symbols')
def prewarm_all_active_symbols(self):
    """
    Warm cache for all active symbols after market close.

    Runs at 5:30 PM ET daily, takes ~2-3 hours for 5000 stocks.
    Next day's scans will be instant (served from cache).

    This is the nightly cache warming job that ensures all active
    symbols have fresh data cached for the next trading day.

    Returns:
        Dict with warming statistics
    """
    logger.info("=" * 80)
    logger.info("TASK: Pre-warming All Active Symbols (Nightly Job)")
    logger.info(f"Market status: {format_market_status()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Skip on non-trading days (weekends, holidays)
    today = get_eastern_now().date()
    if not is_trading_day(today):
        logger.info(f"Skipping nightly warmup - {today} is not a trading day")
        return {'skipped': True, 'reason': 'Not a trading day', 'date': today.isoformat()}

    db = SessionLocal()

    try:
        # Get all active symbols from stock universe
        from ..models.stock_universe import StockUniverse

        universe_records = db.query(StockUniverse).filter(
            StockUniverse.is_active == True
        ).all()

        symbols = [record.symbol for record in universe_records]

        logger.info(f"Found {len(symbols)} active symbols to warm")

        if not symbols:
            logger.warning("No active symbols found in stock universe")
            return {
                'warmed': 0,
                'symbols_found': 0,
                'timestamp': datetime.now().isoformat()
            }

        # Call impl directly to avoid re-acquiring the serialized_data_fetch lock
        result = _prewarm_scan_cache_impl(self, symbols, priority='low', db=db)

        logger.info("=" * 80)
        logger.info("✓ Nightly cache warming completed")
        logger.info(f"  Total symbols: {len(symbols)}")
        logger.info(f"  Warmed: {result.get('warmed', 0)}")
        logger.info(f"  Failed: {result.get('failed', 0)}")
        logger.info("=" * 80)

        return {
            **result,
            'symbols_found': len(symbols),
            'job_type': 'nightly_warmup',
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in prewarm_all_active_symbols task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


def _is_rate_limit_error(error_str: str) -> bool:
    """Check if an error string indicates a rate limit."""
    lower = error_str.lower()
    return any(indicator in lower for indicator in ("rate", "429", "too many", "limit", "throttl"))


def _fetch_with_backoff(bulk_fetcher, symbols: List[str], period: str = "2y", max_retries: int = 3):
    """
    Fetch batch data with exponential backoff on rate limit errors.

    yfinance can return 429 errors or rate limit blocks in two ways:
    1. Raising an exception (caught in the except block)
    2. Returning per-symbol errors with has_error=True (the common case)

    This function detects both and retries with exponential backoff
    (60s, 120s, 240s) before giving up.

    Args:
        bulk_fetcher: BulkDataFetcher instance
        symbols: List of symbols to fetch
        period: Data period (default "2y")
        max_retries: Maximum retry attempts (default 3)

    Returns:
        Dict of symbol -> data, or empty dict if all retries fail
    """
    import time
    base_delay = 60  # Start with 60 second wait

    for attempt in range(max_retries):
        try:
            # Use yf.download()-based batch fetch (single HTTP request per batch)
            # instead of per-ticker fetch_batch_data() which sleeps between each symbol
            results = bulk_fetcher.fetch_batch_prices(
                symbols,
                period=period,
            )

            # Check for per-symbol rate limit errors that fetch_batch_data
            # swallows (returns as has_error=True instead of raising)
            if results:
                rate_limit_failures = sum(
                    1 for data in results.values()
                    if data.get('has_error') and _is_rate_limit_error(data.get('error', ''))
                )
                total = len(results)
                failure_rate = rate_limit_failures / total if total > 0 else 0

                if failure_rate > 0.5 and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # 60, 120, 240 seconds
                    logger.warning(
                        f"Rate limited: {rate_limit_failures}/{total} symbols hit rate limits. "
                        f"Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue  # Retry the whole batch

            return results

        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit indicators in the exception
            if _is_rate_limit_error(error_str):
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # 60, 120, 240 seconds
                    logger.warning(
                        f"Rate limited by yfinance (exception). Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Rate limit persists after {max_retries} retries. "
                        f"Skipping batch of {len(symbols)} symbols."
                    )
                    return {}
            else:
                # Re-raise non-rate-limit errors
                raise
    return {}


def _force_refresh_stale_intraday_impl(task, symbols: Optional[List[str]] = None) -> dict:
    """
    Core implementation of stale intraday data refresh.

    Extracted so it can be called from both the Celery task (with lock)
    and from parent tasks like auto_refresh_after_close (without
    re-acquiring the lock).

    Args:
        task: Celery task instance (for update_state), or None
        symbols: Symbols to refresh, or None for auto-detect

    Returns:
        Dict with refresh statistics
    """
    import time
    from ..services.price_cache_service import PriceCacheService
    from ..services.bulk_data_fetcher import BulkDataFetcher

    logger.info("=" * 80)
    logger.info("TASK: Force Refresh Stale Intraday Data (Batch Mode)")
    logger.info(f"Market status: {format_market_status()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    price_cache = PriceCacheService.get_instance()
    bulk_fetcher = BulkDataFetcher()

    try:
        # Get symbols to refresh
        if symbols is None:
            symbols = price_cache.get_stale_intraday_symbols()
            logger.info(f"Auto-detected {len(symbols)} symbols with stale intraday data")
        else:
            logger.info(f"Refreshing {len(symbols)} specified symbols")

        if not symbols:
            logger.info("No symbols to refresh - all data is fresh")
            return {
                'refreshed': 0,
                'failed': 0,
                'symbols': [],
                'message': 'No stale intraday data detected',
                'completed_at': datetime.now().isoformat()
            }

        total = len(symbols)
        refreshed = 0
        failed = 0
        failed_symbols = []

        # Process symbols in batches using yfinance.Tickers()
        batch_size = 100  # Reduced from 200 to avoid rate limiting

        for batch_start in range(0, total, batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: Fetching {len(batch_symbols)} symbols using yfinance.Tickers()")

            try:
                # Batch fetch using yf.download() with rate limit backoff
                batch_results = _fetch_with_backoff(bulk_fetcher, batch_symbols, period="2y")

                # Separate successes from failures, then batch-store
                batch_to_store = {}
                for symbol, data in batch_results.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        price_df = data['price_data']
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            refreshed += 1
                            logger.debug(f"✓ {symbol}: {len(price_df)} rows refreshed")
                        else:
                            failed += 1
                            failed_symbols.append(symbol)
                            logger.warning(f"✗ {symbol}: Empty data returned")
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                        error_msg = data.get('error', 'Unknown error')
                        logger.warning(f"✗ {symbol}: {error_msg}")

                # Batch store in Redis (pipeline) + DB (single transaction)
                if batch_to_store:
                    price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)

            except Exception as e:
                logger.error(f"Batch {batch_num} error: {e}")
                failed += len(batch_symbols)
                failed_symbols.extend(batch_symbols)

            # Update task state for progress tracking
            progress = min((batch_start + batch_size), total)
            if task is not None:
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'current': progress,
                        'total': total,
                        'percent': (progress / total) * 100,
                        'refreshed': refreshed,
                        'failed': failed
                    }
                )

            # Rate limit between batches (Redis-backed distributed limiter)
            if batch_start + batch_size < total:
                from ..services.rate_limiter import rate_limiter
                rate_limiter.wait("yfinance:batch", min_interval_s=settings.yfinance_batch_rate_limit_interval)

        logger.info("=" * 80)
        logger.info(f"✓ Force refresh completed:")
        logger.info(f"  Refreshed: {refreshed}")
        logger.info(f"  Failed: {failed}")
        if failed_symbols:
            logger.info(f"  Failed symbols: {failed_symbols[:10]}...")
        logger.info("=" * 80)

        return {
            'refreshed': refreshed,
            'failed': failed,
            'total': total,
            'failed_symbols': failed_symbols[:20],  # Limit to 20 for response size
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in force_refresh_stale_intraday: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, name='app.tasks.cache_tasks.force_refresh_stale_intraday')
@serialized_data_fetch('force_refresh_stale_intraday')
def force_refresh_stale_intraday(self, symbols: Optional[List[str]] = None):
    """
    Force refresh symbols with stale intraday data.

    This task refreshes price data for symbols that were fetched during
    market hours and are now stale (market has closed). The "today" bar
    that was incomplete during market hours is replaced with actual
    closing data.

    Uses yfinance.Tickers() for efficient batch fetching.

    Args:
        symbols: List of symbols to refresh. If None, auto-detects
                 all symbols with stale intraday data.

    Returns:
        Dict with refresh statistics
    """
    return _force_refresh_stale_intraday_impl(self, symbols)


@celery_app.task(bind=True, name='app.tasks.cache_tasks.auto_refresh_after_close')
@serialized_data_fetch('auto_refresh_after_close')
def auto_refresh_after_close(self):
    """
    Automatic post-close refresh of stale intraday data.

    Scheduled to run at 4:45 PM ET daily (after market close).
    Detects and refreshes any symbols with stale intraday data.

    This ensures that any data fetched during market hours is
    automatically updated with actual closing prices.

    Returns:
        Dict with refresh statistics
    """
    logger.info("=" * 80)
    logger.info("TASK: Auto Refresh After Market Close")
    logger.info(f"Market status: {format_market_status()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Skip on non-trading days (weekends, holidays)
    today = get_eastern_now().date()
    if not is_trading_day(today):
        logger.info(f"Skipping auto refresh - {today} is not a trading day")
        return {'skipped': True, 'reason': 'Not a trading day', 'date': today.isoformat()}

    # Check if market is still open (shouldn't be, but safety check)
    if is_market_open():
        logger.warning("Market is still open - skipping auto refresh")
        return {
            'skipped': True,
            'reason': 'Market still open',
            'timestamp': datetime.now().isoformat()
        }

    # Call impl directly to avoid re-acquiring the serialized_data_fetch lock
    result = _force_refresh_stale_intraday_impl(self, symbols=None)

    result['job_type'] = 'auto_refresh_after_close'
    return result


@celery_app.task(bind=True, name='app.tasks.cache_tasks.smart_refresh_cache')
@serialized_data_fetch('smart_refresh_cache')
def smart_refresh_cache(self, mode: str = "auto"):
    """
    Smart cache refresh with market cap prioritization.

    This is the unified refresh task that replaces the confusing split
    between daily_cache_warmup and force_refresh_stale_intraday.

    Key features:
    1. Always warms SPY first (required for RS calculations)
    2. Fetches symbols in market cap order (high cap first)
    3. Updates heartbeat for stuck detection
    4. Saves warmup metadata for partial completion tracking

    Args:
        mode: Refresh mode
            - "auto": Full universe, skip symbols refreshed within 4h (smart refresh)
            - "full": Full universe, force re-fetch everything regardless of freshness

    Returns:
        Dict with refresh statistics
    """
    import time
    from ..services.price_cache_service import PriceCacheService
    from ..services.bulk_data_fetcher import BulkDataFetcher
    from ..models.stock_universe import StockUniverse

    logger.info("=" * 80)
    logger.info(f"TASK: Smart Cache Refresh (mode={mode})")
    logger.info(f"Market status: {format_market_status()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Time-window guard: reject Beat-scheduled full mode outside expected windows.
    # This prevents catchup storms (Beat replaying missed schedules on restart).
    # Manual API calls set headers={'origin': 'manual'} to bypass this guard.
    is_manual = getattr(self.request, 'headers', None) and self.request.headers.get('origin') == 'manual'
    if mode == "full" and not is_manual:
        now_et = get_eastern_now()
        weekday = now_et.weekday()  # 0=Mon, 6=Sun
        hour = now_et.hour

        # Allow: weekdays 4PM-midnight (daily refresh window)
        # Allow: Sunday 1AM-6AM (weekly refresh window)
        in_weekday_window = weekday < 5 and 16 <= hour < 24
        in_sunday_window = weekday == 6 and 1 <= hour < 6

        if not in_weekday_window and not in_sunday_window:
            logger.warning(
                f"Rejecting Beat-scheduled full refresh outside time window "
                f"(weekday={weekday}, hour={hour}). Likely a catchup storm."
            )
            return {
                'skipped': True,
                'reason': f'Outside refresh window (weekday={weekday}, hour={hour})',
                'mode': mode,
                'timestamp': datetime.now().isoformat()
            }

    # Skip on non-trading days in auto mode (full mode can be on-demand)
    if mode == "auto":
        today = get_eastern_now().date()
        if not is_trading_day(today):
            logger.info(f"Skipping smart refresh (auto) - {today} is not a trading day")
            return {'skipped': True, 'reason': 'Not a trading day', 'date': today.isoformat(), 'mode': mode}

    price_cache = PriceCacheService.get_instance()
    bulk_fetcher = BulkDataFetcher()
    db = SessionLocal()

    refreshed = 0
    failed = 0
    failed_symbols = []

    try:
        # Step 1: Always warm SPY first (required for RS calculations)
        logger.info("[1/3] Warming SPY benchmark...")
        spy_result = warm_spy_cache()
        if spy_result.get('error'):
            logger.error(f"SPY warmup failed: {spy_result.get('error')}")

        # Step 2: Get symbols to refresh, ordered by market cap
        logger.info(f"[2/3] Determining symbols to refresh (mode={mode})...")

        if mode == "full":
            # Full refresh: ALL active symbols, ordered by market cap DESC
            symbols = [
                r.symbol for r in db.query(StockUniverse.symbol)
                .filter(StockUniverse.is_active == True)
                .order_by(StockUniverse.market_cap.desc().nullslast())
                .all()
            ]
            logger.info(f"Full refresh: {len(symbols)} symbols (market cap order)")
        else:
            # Auto mode: Full universe, skip recently-refreshed symbols
            symbols = [
                r.symbol for r in db.query(StockUniverse.symbol)
                .filter(StockUniverse.is_active == True)
                .order_by(StockUniverse.market_cap.desc().nullslast())
                .all()
            ]
            logger.info(f"Auto refresh: {len(symbols)} active symbols (full universe, market cap order)")

            # Filter out symbols refreshed within skip window
            original_count = len(symbols)
            symbols = price_cache.get_symbols_needing_refresh(symbols, max_age_hours=settings.refresh_skip_hours)
            skipped = original_count - len(symbols)
            if skipped > 0:
                logger.info(f"Skipping {skipped} recently-refreshed symbols (fresh within {settings.refresh_skip_hours}h)")

        if not symbols:
            message = (
                "All symbols recently refreshed - nothing to do"
                if mode == "auto" else
                "No active symbols found in universe"
            )
            price_cache.save_warmup_metadata("completed", 0, 0)
            return {
                "status": "completed",
                "refreshed": 0,
                "failed": 0,
                "total": 0,
                "message": message,
                "mode": mode,
                "completed_at": datetime.now().isoformat()
            }

        total = len(symbols)

        # Step 3: Batch fetch with progress tracking
        logger.info(f"[3/3] Fetching {total} symbols...")

        batch_size = 100

        for batch_start in range(0, total, batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: Fetching {len(batch_symbols)} symbols")

            try:
                # Batch fetch using yf.download() (single HTTP request per batch)
                batch_results = _fetch_with_backoff(bulk_fetcher, batch_symbols, period="2y")

                # Separate successes from failures, then batch-store all successes
                batch_to_store = {}
                for symbol, data in batch_results.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        price_df = data['price_data']
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            refreshed += 1
                        else:
                            failed += 1
                            failed_symbols.append(symbol)
                    else:
                        failed += 1
                        failed_symbols.append(symbol)

                # Batch store in Redis (pipeline) + DB (single transaction)
                if batch_to_store:
                    price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)

            except Exception as e:
                logger.error(f"Batch {batch_num} error: {e}")
                failed += len(batch_symbols)
                failed_symbols.extend(batch_symbols)

            # Update progress for UI and stuck detection
            progress = min((batch_start + batch_size), total)
            percent = (progress / total) * 100

            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': progress,
                    'total': total,
                    'percent': percent,
                    'refreshed': refreshed,
                    'failed': failed
                }
            )

            # Update heartbeat for stuck detection
            price_cache.update_warmup_heartbeat(progress, total, percent)

            # Extend lock TTL to prevent expiry during long-running tasks (A5)
            task_id = self.request.id or 'unknown'
            from .data_fetch_lock import DataFetchLock
            lock = DataFetchLock.get_instance()
            lock.extend_lock(task_id, 3600)

            # Rate limit between batches (Redis-backed distributed limiter)
            if batch_start + batch_size < total:
                from ..services.rate_limiter import rate_limiter
                rate_limiter.wait("yfinance:batch", min_interval_s=settings.yfinance_batch_rate_limit_interval)

        # Save final warmup metadata (treat >95% success as "completed")
        success_rate = refreshed / total if total > 0 else 0
        status = "completed" if success_rate >= 0.95 else "partial"
        price_cache.save_warmup_metadata(status, refreshed, total)
        price_cache.clear_warmup_heartbeat()

        logger.info("=" * 80)
        logger.info(f"✓ Smart refresh completed ({mode} mode):")
        logger.info(f"  Refreshed: {refreshed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total: {total}")
        if failed_symbols:
            logger.info(f"  Failed symbols: {failed_symbols[:10]}...")
        logger.info("=" * 80)

        return {
            "status": status,
            "refreshed": refreshed,
            "failed": failed,
            "total": total,
            "failed_symbols": failed_symbols[:20],
            "mode": mode,
            "completed_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in smart_refresh_cache task: {e}", exc_info=True)
        # Save partial progress
        price_cache.save_warmup_metadata("failed", refreshed, locals().get('total', 0), str(e))
        price_cache.clear_warmup_heartbeat()
        return {
            "status": "failed",
            "error": str(e),
            "refreshed": refreshed,
            "failed": failed,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.cache_tasks.cleanup_old_price_data')
def cleanup_old_price_data(self, keep_years: int = 5):
    """
    Clean up price data older than the specified retention period.

    This task removes historical price data that is no longer needed,
    preventing unbounded database growth. For stock scanning, typically
    only 1-2 years of data is needed, but we default to keeping 5 years
    for analysis purposes.

    Args:
        keep_years: Number of years of price data to retain (default: 5)

    Returns:
        Dict with cleanup statistics
    """
    from datetime import date, timedelta
    from ..models.stock import StockPrice
    from sqlalchemy import func

    logger.info("=" * 80)
    logger.info(f"TASK: Cleanup Old Price Data (keeping {keep_years} years)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    db = SessionLocal()

    try:
        # Calculate cutoff date
        cutoff_date = date.today() - timedelta(days=keep_years * 365)

        logger.info(f"Cutoff date: {cutoff_date}")

        # Get count of records to delete
        delete_count = db.query(func.count(StockPrice.id)).filter(
            StockPrice.date < cutoff_date
        ).scalar() or 0

        if delete_count == 0:
            logger.info("No old price data to delete")
            return {
                'deleted_rows': 0,
                'cutoff_date': str(cutoff_date),
                'message': 'No data older than cutoff date',
                'completed_at': datetime.now().isoformat()
            }

        logger.info(f"Found {delete_count} rows older than {cutoff_date}")

        # Delete in batches to avoid locking
        batch_size = 10000
        total_deleted = 0

        while True:
            # Delete a batch
            deleted = db.query(StockPrice).filter(
                StockPrice.date < cutoff_date
            ).limit(batch_size).delete(synchronize_session=False)

            if deleted == 0:
                break

            total_deleted += deleted
            db.commit()

            # Update progress
            progress = (total_deleted / delete_count) * 100 if delete_count > 0 else 100
            self.update_state(
                state='PROGRESS',
                meta={
                    'deleted': total_deleted,
                    'total': delete_count,
                    'percent': progress
                }
            )

            logger.info(f"Deleted batch: {deleted} rows (total: {total_deleted}/{delete_count})")

        logger.info("=" * 80)
        logger.info(f"✓ Price data cleanup completed:")
        logger.info(f"  Deleted: {total_deleted} rows")
        logger.info(f"  Cutoff date: {cutoff_date}")
        logger.info("=" * 80)

        return {
            'deleted_rows': total_deleted,
            'cutoff_date': str(cutoff_date),
            'keep_years': keep_years,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in cleanup_old_price_data task: {e}", exc_info=True)
        db.rollback()
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.cache_tasks.prewarm_chart_cache_for_scan')
def prewarm_chart_cache_for_scan(self, scan_id: str, top_n: int = 50):
    """
    Pre-warm chart cache for top results after a scan completes.

    This task is triggered after a scan completes to warm the price
    cache for the top N results, so charts load instantly when
    users view the scan results.

    Args:
        scan_id: UUID of completed scan
        top_n: Number of top results to warm (default: 50)

    Returns:
        Dict with warming statistics
    """
    from ..models.scan_result import ScanResult
    from ..services.price_cache_service import PriceCacheService

    logger.info("=" * 80)
    logger.info(f"TASK: Pre-warming Chart Cache for Scan {scan_id}")
    logger.info(f"Top {top_n} results will be warmed")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    db = SessionLocal()

    try:
        # Get top N results ordered by composite score
        top_results = db.query(ScanResult.symbol).filter(
            ScanResult.scan_id == scan_id
        ).order_by(
            ScanResult.composite_score.desc()
        ).limit(top_n).all()

        symbols = [r.symbol for r in top_results]

        if not symbols:
            logger.warning(f"No results found for scan {scan_id}")
            return {
                'scan_id': scan_id,
                'warmed': 0,
                'message': 'No scan results found',
                'completed_at': datetime.now().isoformat()
            }

        logger.info(f"Warming chart cache for {len(symbols)} symbols: {symbols[:5]}...")

        # Use PriceCacheService to warm the cache
        price_cache = PriceCacheService.get_instance()
        warmed = 0
        failed = 0
        already_cached = 0

        # Process symbols with rate limiting to avoid overwhelming yfinance
        for i, symbol in enumerate(symbols):
            try:
                # Check if already cached
                cache_stats = price_cache.get_cache_stats(symbol)
                if cache_stats.get('redis_cached'):
                    already_cached += 1
                    logger.debug(f"✓ {symbol}: Already cached")
                    continue

                # Fetch and cache (6mo period for charts)
                data = price_cache.get_price_data(symbol, period='6mo')
                if data is not None and not data.empty:
                    warmed += 1
                    logger.debug(f"✓ {symbol}: Warmed ({len(data)} rows)")
                else:
                    failed += 1
                    logger.warning(f"✗ {symbol}: No data returned")

            except Exception as e:
                failed += 1
                logger.warning(f"✗ {symbol}: Error - {e}")

            # Update progress
            progress = ((i + 1) / len(symbols)) * 100
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': i + 1,
                    'total': len(symbols),
                    'percent': progress,
                    'warmed': warmed,
                    'failed': failed,
                    'already_cached': already_cached
                }
            )

        logger.info("=" * 80)
        logger.info(f"✓ Chart cache warming completed for scan {scan_id}")
        logger.info(f"  Warmed: {warmed}, Already Cached: {already_cached}, Failed: {failed}")
        logger.info("=" * 80)

        return {
            'scan_id': scan_id,
            'warmed': warmed,
            'failed': failed,
            'already_cached': already_cached,
            'total': len(symbols),
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in prewarm_chart_cache_for_scan task: {e}", exc_info=True)
        return {
            'scan_id': scan_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.cache_tasks.cleanup_orphaned_scans')
def cleanup_orphaned_scans(self):
    """
    Clean up orphaned scan data across all universes.

    This task cleans up:
    1. All cancelled scans and their results (no value)
    2. All stale running/queued scans older than 1 hour (will never complete)

    Should be run periodically to prevent scan data buildup.

    Returns:
        Dict with cleanup statistics
    """
    from datetime import timedelta
    from ..models.scan_result import Scan, ScanResult
    from sqlalchemy import func

    logger.info("=" * 80)
    logger.info("TASK: Cleanup Orphaned Scans")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    db = SessionLocal()

    try:
        total_deleted_scans = 0
        total_deleted_results = 0

        # 1. Delete all cancelled scans
        logger.info("[1/2] Deleting cancelled scans...")
        cancelled_scan_ids = [s.scan_id for s in db.query(Scan).filter(Scan.status == "cancelled").all()]
        if cancelled_scan_ids:
            cancelled_results = db.query(ScanResult).filter(
                ScanResult.scan_id.in_(cancelled_scan_ids)
            ).delete(synchronize_session=False)
            cancelled_scans = db.query(Scan).filter(Scan.status == "cancelled").delete(synchronize_session=False)
            total_deleted_scans += cancelled_scans
            total_deleted_results += cancelled_results
            logger.info(f"  Deleted {cancelled_scans} cancelled scans, {cancelled_results} results")
        else:
            logger.info("  No cancelled scans found")

        # 2. Delete stale running/queued scans (older than 1 hour)
        logger.info("[2/2] Deleting stale running/queued scans...")
        stale_cutoff = datetime.utcnow() - timedelta(hours=1)
        stale_scan_ids = [
            s.scan_id for s in db.query(Scan).filter(
                Scan.status.in_(["running", "queued"]),
                Scan.started_at < stale_cutoff
            ).all()
        ]
        if stale_scan_ids:
            stale_results = db.query(ScanResult).filter(
                ScanResult.scan_id.in_(stale_scan_ids)
            ).delete(synchronize_session=False)
            stale_scans = db.query(Scan).filter(
                Scan.status.in_(["running", "queued"]),
                Scan.started_at < stale_cutoff
            ).delete(synchronize_session=False)
            total_deleted_scans += stale_scans
            total_deleted_results += stale_results
            logger.info(f"  Deleted {stale_scans} stale scans, {stale_results} results")
        else:
            logger.info("  No stale running/queued scans found")

        db.commit()

        logger.info("=" * 80)
        logger.info(f"✓ Orphaned scan cleanup completed:")
        logger.info(f"  Deleted scans: {total_deleted_scans}")
        logger.info(f"  Deleted results: {total_deleted_results}")
        logger.info("=" * 80)

        return {
            'deleted_scans': total_deleted_scans,
            'deleted_results': total_deleted_results,
            'completed_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in cleanup_orphaned_scans task: {e}", exc_info=True)
        db.rollback()
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()

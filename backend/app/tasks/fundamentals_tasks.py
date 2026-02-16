"""
Celery tasks for fundamental data updates.

Provides background tasks for:
- Weekly fundamental refresh for all stocks (standard and hybrid modes)
- On-demand fundamental refresh
- Initial cache population

All data-fetching tasks use the @serialized_data_fetch decorator
to ensure only one task fetches external data at a time.
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime
import time

from ..celery_app import celery_app
from ..database import SessionLocal
from ..models.stock_universe import StockUniverse
from ..services.fundamentals_cache_service import FundamentalsCacheService
from ..services.hybrid_fundamentals_service import HybridFundamentalsService
from ..services.ticker_validation_service import ticker_validation_service, TickerValidationService
from ..config import settings
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='app.tasks.fundamentals_tasks.refresh_all_fundamentals')
@serialized_data_fetch('refresh_all_fundamentals')
def refresh_all_fundamentals(self):
    """
    Weekly task to refresh fundamental data for all stocks in universe.

    Fetches and updates fundamental data (PE, EPS, revenue, margins, etc.)
    for all active stocks. Designed to run once weekly (e.g., Friday 6 PM ET).

    Returns:
        Dict with refresh statistics:
        {
            'total_stocks': int,
            'updated': int,
            'failed': int,
            'skipped': int,
            'duration_seconds': float,
            'timestamp': str
        }
    """
    logger.info("=" * 60)
    logger.info("TASK: Weekly Fundamental Data Refresh")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        # Get all active stocks from universe
        universe_stocks = db.query(StockUniverse).filter(
            StockUniverse.is_active == True
        ).all()

        if not universe_stocks:
            logger.warning("No active stocks found in universe")
            return {
                'error': 'No active stocks found',
                'timestamp': datetime.now().isoformat()
            }

        total_stocks = len(universe_stocks)
        logger.info(f"Found {total_stocks} active stocks to refresh")

        # Initialize cache service
        cache = FundamentalsCacheService.get_instance()

        # Statistics
        stats = {
            'updated': 0,
            'failed': 0,
            'skipped': 0,
            'failed_symbols': []
        }

        # Process each stock with rate limiting (2 seconds between calls)
        for i, stock in enumerate(universe_stocks):
            try:
                symbol = stock.symbol

                # Log progress every 50 stocks
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{total_stocks} ({(i+1)/total_stocks*100:.1f}%)")

                # Fetch fundamental data (force_refresh=True to get fresh data)
                fundamental_data = cache.get_fundamentals(
                    symbol,
                    force_refresh=True  # Always get fresh data for weekly refresh
                )

                if fundamental_data:
                    stats['updated'] += 1
                    logger.debug(f"✓ Updated fundamentals for {symbol}")
                else:
                    stats['skipped'] += 1
                    stats['failed_symbols'].append(symbol)
                    logger.warning(f"⚠ No data returned for {symbol}")
                    # Log validation failure for reporting
                    ticker_validation_service.log_validation_failure(
                        db=db,
                        symbol=symbol,
                        error_type=TickerValidationService.ERROR_NO_DATA,
                        error_message='No data returned from fundamentals API',
                        data_source=TickerValidationService.SOURCE_YFINANCE,
                        triggered_by=TickerValidationService.TRIGGER_FUNDAMENTALS_REFRESH,
                        task_id=self.request.id if self.request else None,
                    )

                # Rate limiting handled internally by cache.get_fundamentals()
                # → DataSourceService → finviz_service / yfinance_service
                # (each uses Redis-backed distributed rate limiter)

            except Exception as e:
                stats['failed'] += 1
                stats['failed_symbols'].append(symbol)
                logger.error(f"✗ Error updating {symbol}: {e}")
                # Log validation failure with exception details
                error_type, error_msg = ticker_validation_service.classify_error(exception=e)
                ticker_validation_service.log_validation_failure(
                    db=db,
                    symbol=symbol,
                    error_type=error_type,
                    error_message=error_msg,
                    data_source=TickerValidationService.SOURCE_YFINANCE,
                    triggered_by=TickerValidationService.TRIGGER_FUNDAMENTALS_REFRESH,
                    task_id=self.request.id if self.request else None,
                )
                continue

        duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Weekly Fundamental Refresh Complete!")
        logger.info(f"Total stocks: {total_stocks}")
        logger.info(f"Updated: {stats['updated']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Duration: {duration:.2f}s ({duration/60:.1f} minutes)")
        logger.info(f"Average: {duration/total_stocks:.2f}s per stock")
        if stats['failed_symbols'][:5]:
            logger.info(f"Failed symbols (first 5): {', '.join(stats['failed_symbols'][:5])}")
        logger.info("=" * 60)

        # Chain EPS rating percentiles calculation after fundamentals refresh
        logger.info("Queuing EPS Rating Percentiles calculation...")
        eps_task = calculate_eps_rating_percentiles.delay()
        logger.info(f"EPS Rating Percentiles task queued: {eps_task.id}")

        return {
            'total_stocks': total_stocks,
            'updated': stats['updated'],
            'failed': stats['failed'],
            'skipped': stats['skipped'],
            'failed_symbols': stats['failed_symbols'][:10],  # Only return first 10
            'duration_seconds': round(duration, 2),
            'duration_minutes': round(duration / 60, 1),
            'timestamp': datetime.now().isoformat(),
            'eps_rating_task_id': eps_task.id
        }

    except Exception as e:
        logger.error(f"Fatal error in fundamental refresh: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.fundamentals_tasks.refresh_symbol_fundamentals')
@serialized_data_fetch('refresh_symbol_fundamentals')
def refresh_symbol_fundamentals(self, symbol: str):
    """
    On-demand task to refresh fundamentals for a single stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dict with fundamental data or error
    """
    logger.info(f"TASK: Refresh Fundamentals for {symbol}")

    try:
        cache = FundamentalsCacheService.get_instance()
        data = cache.get_fundamentals(symbol.upper(), force_refresh=True)

        if data:
            logger.info(f"✓ Successfully refreshed fundamentals for {symbol}")
            # Count populated fields
            populated_fields = len([v for v in data.values() if v is not None])
            return {
                'symbol': symbol,
                'success': True,
                'populated_fields': populated_fields,
                'timestamp': datetime.now().isoformat()
            }
        else:
            logger.warning(f"⚠ No data returned for {symbol}")
            return {
                'symbol': symbol,
                'success': False,
                'error': 'No data returned',
                'timestamp': datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"✗ Error refreshing {symbol}: {e}", exc_info=True)
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, name='app.tasks.fundamentals_tasks.populate_initial_cache')
@serialized_data_fetch('populate_initial_cache')
def populate_initial_cache(self, limit: Optional[int] = None):
    """
    One-time task to populate fundamental cache immediately after deployment.

    This is the same logic as refresh_all_fundamentals() but with different logging
    to indicate it's an initial population rather than a scheduled refresh.

    Args:
        limit: Optional limit on number of stocks to populate (for testing)

    Returns:
        Dict with population statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Initial Fundamental Cache Population")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if limit:
        logger.info(f"Limit: {limit} stocks (testing mode)")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        # Get all active stocks from universe
        query = db.query(StockUniverse).filter(StockUniverse.is_active == True)

        if limit:
            query = query.limit(limit)

        universe_stocks = query.all()

        if not universe_stocks:
            logger.warning("No active stocks found in universe")
            return {
                'error': 'No active stocks found',
                'timestamp': datetime.now().isoformat()
            }

        total_stocks = len(universe_stocks)
        logger.info(f"Populating cache for {total_stocks} active stocks")
        logger.info(f"Estimated time: {total_stocks * 0.5 / 3600:.1f} hours (at 0.5s per stock)")

        # Initialize cache service
        cache = FundamentalsCacheService.get_instance()

        # Statistics
        stats = {
            'updated': 0,
            'failed': 0,
            'skipped': 0,
            'failed_symbols': []
        }

        # Process each stock with rate limiting (2 seconds between calls)
        for i, stock in enumerate(universe_stocks):
            try:
                symbol = stock.symbol

                # Log progress every 100 stocks for initial population
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (total_stocks - i - 1)
                    logger.info(
                        f"Progress: {i + 1}/{total_stocks} ({(i+1)/total_stocks*100:.1f}%) | "
                        f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m"
                    )

                # Fetch fundamental data (force_refresh=True to populate fresh data)
                fundamental_data = cache.get_fundamentals(
                    symbol,
                    force_refresh=True
                )

                if fundamental_data:
                    stats['updated'] += 1
                    logger.debug(f"✓ Populated fundamentals for {symbol}")
                else:
                    stats['skipped'] += 1
                    stats['failed_symbols'].append(symbol)
                    logger.warning(f"⚠ No data returned for {symbol}")
                    # Log validation failure for reporting
                    ticker_validation_service.log_validation_failure(
                        db=db,
                        symbol=symbol,
                        error_type=TickerValidationService.ERROR_NO_DATA,
                        error_message='No data returned from fundamentals API',
                        data_source=TickerValidationService.SOURCE_YFINANCE,
                        triggered_by=TickerValidationService.TRIGGER_CACHE_WARMUP,
                        task_id=self.request.id if self.request else None,
                    )

                # Rate limiting handled internally by cache.get_fundamentals()
                # → DataSourceService → finviz_service / yfinance_service
                # (each uses Redis-backed distributed rate limiter)

            except Exception as e:
                stats['failed'] += 1
                stats['failed_symbols'].append(symbol)
                logger.error(f"✗ Error populating {symbol}: {e}")
                # Log validation failure with exception details
                error_type, error_msg = ticker_validation_service.classify_error(exception=e)
                ticker_validation_service.log_validation_failure(
                    db=db,
                    symbol=symbol,
                    error_type=error_type,
                    error_message=error_msg,
                    data_source=TickerValidationService.SOURCE_YFINANCE,
                    triggered_by=TickerValidationService.TRIGGER_CACHE_WARMUP,
                    task_id=self.request.id if self.request else None,
                )
                continue

        duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Initial Cache Population Complete!")
        logger.info(f"Total stocks: {total_stocks}")
        logger.info(f"Updated: {stats['updated']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Duration: {duration:.2f}s ({duration/60:.1f} minutes / {duration/3600:.2f} hours)")
        logger.info(f"Average: {duration/total_stocks:.2f}s per stock")
        if stats['failed_symbols'][:5]:
            logger.info(f"Failed symbols (first 5): {', '.join(stats['failed_symbols'][:5])}")
        logger.info("=" * 60)

        # Chain EPS rating percentiles calculation after initial cache population
        logger.info("Queuing EPS Rating Percentiles calculation...")
        eps_task = calculate_eps_rating_percentiles.delay()
        logger.info(f"EPS Rating Percentiles task queued: {eps_task.id}")

        return {
            'total_stocks': total_stocks,
            'updated': stats['updated'],
            'failed': stats['failed'],
            'skipped': stats['skipped'],
            'failed_symbols': stats['failed_symbols'][:20],  # Return first 20
            'duration_seconds': round(duration, 2),
            'duration_minutes': round(duration / 60, 1),
            'duration_hours': round(duration / 3600, 2),
            'timestamp': datetime.now().isoformat(),
            'eps_rating_task_id': eps_task.id
        }

    except Exception as e:
        logger.error(f"Fatal error in initial cache population: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.fundamentals_tasks.get_cache_stats')
def get_cache_stats(symbols: Optional[List[str]] = None):
    """
    Get cache statistics for fundamental data.

    Args:
        symbols: Optional list of symbols to check (defaults to all active stocks)

    Returns:
        Dict with cache statistics
    """
    logger.info("TASK: Get Fundamental Cache Statistics")

    db = SessionLocal()

    try:
        # Get symbols to check
        if symbols is None:
            universe_stocks = db.query(StockUniverse).filter(
                StockUniverse.is_active == True
            ).limit(100).all()  # Check first 100 for quick stats
            symbols = [s.symbol for s in universe_stocks]

        cache = FundamentalsCacheService.get_instance()

        # Collect stats
        total = len(symbols)
        redis_cached = 0
        db_cached = 0
        fresh = 0
        stale = 0

        for symbol in symbols:
            stats = cache.get_cache_stats(symbol)

            if stats['redis_cached']:
                redis_cached += 1
            if stats['db_cached']:
                db_cached += 1
            if stats['age_days'] is not None:
                if stats['age_days'] <= 7:
                    fresh += 1
                else:
                    stale += 1

        return {
            'total_checked': total,
            'redis_cached': redis_cached,
            'db_cached': db_cached,
            'fresh': fresh,
            'stale': stale,
            'redis_hit_rate': round(redis_cached / total * 100, 1) if total > 0 else 0,
            'db_hit_rate': round(db_cached / total * 100, 1) if total > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.fundamentals_tasks.refresh_all_fundamentals_hybrid')
@serialized_data_fetch('refresh_all_fundamentals_hybrid')
def refresh_all_fundamentals_hybrid(
    self,
    include_finviz: bool = True,
    yfinance_batch_size: int = 50
):
    """
    HYBRID fundamental refresh - optimized for speed.

    Uses hybrid approach:
    1. yfinance batch fetching (50 symbols at a time) - ~25 min
    2. Technical calculations from cached price data - ~10 min
    3. finviz-only fields (short interest, etc.) - ~1-1.5 hours

    Target: ~1.5-2 hours vs ~4 hours for traditional approach.

    Args:
        include_finviz: Whether to fetch finviz-only fields (default True)
        yfinance_batch_size: Symbols per yfinance batch (default 50)

    Returns:
        Dict with refresh statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Hybrid Fundamental Data Refresh (Optimized)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Include finviz: {include_finviz}")
    logger.info(f"yfinance batch size: {yfinance_batch_size}")
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()

    try:
        # Get all active stocks from universe
        universe_stocks = db.query(StockUniverse).filter(
            StockUniverse.is_active == True
        ).all()

        if not universe_stocks:
            logger.warning("No active stocks found in universe")
            return {
                'error': 'No active stocks found',
                'timestamp': datetime.now().isoformat()
            }

        symbols = [s.symbol for s in universe_stocks]
        total_stocks = len(symbols)
        logger.info(f"Found {total_stocks} active stocks to refresh")

        # Estimate time
        if include_finviz:
            est_time = (total_stocks / 50 * 10) / 60 + 10 + (total_stocks * 2) / 60  # yf + tech + finviz
            logger.info(f"Estimated time: ~{est_time:.0f} minutes (hybrid with finviz)")
        else:
            est_time = (total_stocks / 50 * 10) / 60 + 10  # yf + tech only
            logger.info(f"Estimated time: ~{est_time:.0f} minutes (yfinance + technicals only)")

        # Initialize hybrid service
        hybrid_service = HybridFundamentalsService(
            include_finviz=include_finviz,
            yfinance_batch_size=yfinance_batch_size
        )

        # Initialize cache for storage
        cache = FundamentalsCacheService.get_instance()

        def progress_callback(current, total):
            """Log progress updates."""
            pct = current / total * 100 if total > 0 else 0
            elapsed = time.time() - start_time
            if current > 0:
                eta = (elapsed / current) * (total - current) / 60
                logger.info(f"Hybrid progress: {current}/{total} ({pct:.1f}%), ETA: {eta:.1f} min")

        # Fetch all fundamentals using hybrid approach
        all_data = hybrid_service.fetch_fundamentals_batch(
            symbols,
            include_technicals=True,
            include_finviz=include_finviz,
            progress_callback=progress_callback
        )

        # Store in all caches (fundamentals + quarterly for CANSLIM compatibility)
        logger.info("Storing results in cache...")
        storage_stats = hybrid_service.store_all_caches(
            all_data,
            cache,
            include_quarterly=True
        )

        stats = {
            'updated': storage_stats['fundamentals_stored'],
            'quarterly_stored': storage_stats['quarterly_stored'],
            'failed': storage_stats['failed'],
            'skipped': len([d for d in all_data.values() if not d or d.get('has_error')]),
            'failed_symbols': [s for s, d in all_data.items() if not d or d.get('has_error')]
        }

        # Log validation failures for failed symbols
        for symbol in stats['failed_symbols']:
            data = all_data.get(symbol, {})
            error_msg = data.get('error', 'No data returned') if data else 'No data returned'
            ticker_validation_service.log_validation_failure(
                db=db,
                symbol=symbol,
                error_type=TickerValidationService.ERROR_NO_DATA,
                error_message=error_msg,
                data_source=TickerValidationService.SOURCE_BOTH if include_finviz else TickerValidationService.SOURCE_YFINANCE,
                triggered_by=TickerValidationService.TRIGGER_FUNDAMENTALS_REFRESH,
                task_id=self.request.id if self.request else None,
            )

        duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Hybrid Fundamental Refresh Complete!")
        logger.info(f"Total stocks: {total_stocks}")
        logger.info(f"Updated: {stats['updated']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Duration: {duration:.2f}s ({duration/60:.1f} minutes)")
        logger.info(f"Average: {duration/total_stocks:.2f}s per stock")
        if stats['failed_symbols'][:5]:
            logger.info(f"Failed symbols (first 5): {', '.join(stats['failed_symbols'][:5])}")
        logger.info("=" * 60)

        # Chain EPS rating percentiles calculation after fundamentals refresh
        logger.info("Queuing EPS Rating Percentiles calculation...")
        eps_task = calculate_eps_rating_percentiles.delay()
        logger.info(f"EPS Rating Percentiles task queued: {eps_task.id}")

        return {
            'mode': 'hybrid',
            'include_finviz': include_finviz,
            'total_stocks': total_stocks,
            'updated': stats['updated'],
            'quarterly_stored': stats.get('quarterly_stored', 0),
            'failed': stats['failed'],
            'skipped': stats['skipped'],
            'failed_symbols': stats['failed_symbols'][:10],
            'duration_seconds': round(duration, 2),
            'duration_minutes': round(duration / 60, 1),
            'timestamp': datetime.now().isoformat(),
            'eps_rating_task_id': eps_task.id
        }

    except Exception as e:
        logger.error(f"Fatal error in hybrid fundamental refresh: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.fundamentals_tasks.refresh_fundamentals_yfinance_only')
def refresh_fundamentals_yfinance_only(yfinance_batch_size: int = 50):
    """
    Fast fundamental refresh using ONLY yfinance + technical calculations.

    Skips finviz entirely for maximum speed (~30-40 minutes).
    Use when finviz-only fields (short interest, etc.) aren't needed.

    Args:
        yfinance_batch_size: Symbols per batch (default 50)

    Returns:
        Dict with refresh statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Fast Fundamental Refresh (yfinance + technicals only)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Use hybrid task with finviz disabled
    return refresh_all_fundamentals_hybrid(
        include_finviz=False,
        yfinance_batch_size=yfinance_batch_size
    )


@celery_app.task(bind=True, name='app.tasks.fundamentals_tasks.refresh_symbols_hybrid')
@serialized_data_fetch('refresh_symbols_hybrid')
def refresh_symbols_hybrid(
    self,
    symbols: List[str],
    include_finviz: bool = True
):
    """
    Refresh fundamentals for a specific list of symbols using hybrid approach.

    Useful for:
    - Refreshing a watchlist
    - Updating specific sectors
    - Testing the hybrid approach on a subset

    Args:
        symbols: List of stock ticker symbols
        include_finviz: Whether to include finviz-only fields

    Returns:
        Dict with refresh statistics
    """
    logger.info(f"TASK: Hybrid refresh for {len(symbols)} symbols")
    start_time = time.time()

    try:
        hybrid_service = HybridFundamentalsService(include_finviz=include_finviz)
        cache = FundamentalsCacheService.get_instance()

        # Fetch fundamentals
        all_data = hybrid_service.fetch_fundamentals_batch(
            symbols,
            include_technicals=True,
            include_finviz=include_finviz
        )

        # Store in all caches (fundamentals + quarterly for CANSLIM compatibility)
        storage_stats = hybrid_service.store_all_caches(
            all_data,
            cache,
            include_quarterly=True
        )

        duration = time.time() - start_time

        return {
            'mode': 'hybrid',
            'symbols_requested': len(symbols),
            'updated': storage_stats['fundamentals_stored'],
            'quarterly_stored': storage_stats['quarterly_stored'],
            'failed': storage_stats['failed'],
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in hybrid refresh: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(bind=True, name='app.tasks.fundamentals_tasks.calculate_eps_rating_percentiles')
def calculate_eps_rating_percentiles(self):
    """
    Calculate EPS Rating percentiles across all stocks in the universe.

    This task should be run after fundamentals refresh to compute
    the 0-99 percentile ranking for each stock based on raw EPS scores.

    The formula for raw EPS score is:
        raw_score = 0.40 * CAGR_5yr + 0.50 * avg(Q1_YoY, Q2_YoY) + 0.10 * (Q1_YoY - Q2_YoY)

    Returns:
        Dict with calculation statistics:
        {
            'total_stocks': int,
            'stocks_with_score': int,
            'stocks_updated': int,
            'duration_seconds': float,
            'timestamp': str
        }
    """
    logger.info("=" * 60)
    logger.info("TASK: Calculate EPS Rating Percentiles")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    from ..models.stock import StockFundamental
    from ..services.eps_rating_service import eps_rating_service

    db = SessionLocal()
    start_time = time.time()

    try:
        # Get all fundamentals with raw scores
        fundamentals = db.query(StockFundamental).filter(
            StockFundamental.eps_raw_score.isnot(None)
        ).all()

        if not fundamentals:
            logger.warning("No stocks with EPS raw scores found")
            return {
                'total_stocks': 0,
                'stocks_with_score': 0,
                'stocks_updated': 0,
                'timestamp': datetime.now().isoformat()
            }

        # Build raw scores dict
        raw_scores = {f.symbol: f.eps_raw_score for f in fundamentals}
        logger.info(f"Found {len(raw_scores)} stocks with EPS raw scores")

        # Calculate percentile ranks
        percentile_ranks = eps_rating_service.calculate_percentile_ranks(raw_scores)
        logger.info(f"Calculated percentile ranks for {len(percentile_ranks)} stocks")

        # Update database with eps_rating values
        updated_count = 0
        batch_size = 100
        symbols_to_update = list(percentile_ranks.keys())

        for i in range(0, len(symbols_to_update), batch_size):
            batch = symbols_to_update[i:i + batch_size]

            for symbol in batch:
                eps_rating = percentile_ranks[symbol]

                # Update the fundamental record
                db.query(StockFundamental).filter(
                    StockFundamental.symbol == symbol
                ).update(
                    {'eps_rating': eps_rating},
                    synchronize_session=False
                )
                updated_count += 1

            db.commit()
            logger.debug(f"Updated batch {i // batch_size + 1}, {updated_count} stocks total")

        # Also update ScanResult records with new eps_rating values
        # This ensures existing scan results reflect the latest EPS ratings
        from ..models.scan_result import ScanResult

        scan_updated_count = 0
        for i in range(0, len(symbols_to_update), batch_size):
            batch = symbols_to_update[i:i + batch_size]

            for symbol in batch:
                eps_rating = percentile_ranks[symbol]

                # Update all ScanResult records for this symbol
                rows_updated = db.query(ScanResult).filter(
                    ScanResult.symbol == symbol
                ).update(
                    {'eps_rating': eps_rating},
                    synchronize_session=False
                )
                scan_updated_count += rows_updated

            db.commit()

        logger.info(f"Updated {scan_updated_count} ScanResult records with EPS ratings")

        # Invalidate Redis cache for all updated symbols
        # This forces the next cache read to hit the database and get fresh eps_rating
        logger.info("Invalidating Redis cache for updated symbols...")
        cache = FundamentalsCacheService.get_instance()
        cache_invalidated_count = 0

        for i in range(0, len(symbols_to_update), batch_size):
            batch = symbols_to_update[i:i + batch_size]

            for symbol in batch:
                try:
                    cache.invalidate_cache(symbol)
                    cache_invalidated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to invalidate cache for {symbol}: {e}")

        logger.info(f"Invalidated Redis cache for {cache_invalidated_count} symbols")

        duration = time.time() - start_time

        logger.info("=" * 60)
        logger.info("EPS Rating Percentiles Calculation Complete!")
        logger.info(f"Total stocks in universe: {len(fundamentals)}")
        logger.info(f"Stocks with raw score: {len(raw_scores)}")
        logger.info(f"Fundamentals updated: {updated_count}")
        logger.info(f"Scan results updated: {scan_updated_count}")
        logger.info(f"Redis cache invalidated: {cache_invalidated_count}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info("=" * 60)

        return {
            'total_stocks': len(fundamentals),
            'stocks_with_score': len(raw_scores),
            'stocks_updated': updated_count,
            'scan_results_updated': scan_updated_count,
            'cache_invalidated': cache_invalidated_count,
            'duration_seconds': round(duration, 2),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating EPS rating percentiles: {e}", exc_info=True)
        db.rollback()
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()

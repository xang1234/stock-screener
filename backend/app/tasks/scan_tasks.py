"""
Celery tasks for bulk stock scanning.

Handles background processing of large-scale stock scans.

All data-fetching tasks use the @serialized_data_fetch decorator
to ensure only one task fetches external data at a time.
"""
import logging
import time
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..celery_app import celery_app
from ..database import SessionLocal
from ..models.scan_result import Scan, ScanResult
# MinerviniScanner removed - use MinerviniScannerV2 via orchestrator instead
# NOTE: get_scan_orchestrator and get_stock_data_provider are imported lazily
# inside functions to avoid a circular import through wiring.bootstrap →
# infra.tasks.dispatcher → scan_tasks → wiring.bootstrap.
from ..config import settings
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    from datetime import datetime, date

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (datetime, date)):
        # Convert datetime/date to ISO format string
        return obj.isoformat()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif obj is None:
        return None
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj


def update_scan_status(
    db: Session,
    scan_id: str,
    status: str,
    **kwargs
) -> None:
    """
    Update scan record in database.

    Args:
        db: Database session
        scan_id: UUID of scan
        status: New status (queued, running, completed, failed)
        **kwargs: Additional fields to update (total_stocks, passed_stocks, etc.)
    """
    try:
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()

        if scan:
            scan.status = status

            if 'total_stocks' in kwargs:
                scan.total_stocks = kwargs['total_stocks']
            if 'passed_stocks' in kwargs:
                scan.passed_stocks = kwargs['passed_stocks']
            if status == 'completed':
                scan.completed_at = datetime.utcnow()

            db.commit()
            logger.info(f"Updated scan {scan_id} status to {status}")
        else:
            logger.warning(f"Scan {scan_id} not found for status update")

    except Exception as e:
        logger.error(f"Error updating scan status: {e}", exc_info=True)
        db.rollback()


def cleanup_old_scans(db: Session, universe_key: str, keep_count: int = 3) -> None:
    """
    Delete old scans, keeping only the most recent `keep_count` per universe_key.

    Called after successful scan completion to maintain retention policy.
    Uses universe_key (canonical identifier) instead of the legacy universe string,
    so different exchange/custom scans get separate retention buckets.

    Handles:
    1. Cancelled scans - deleted immediately (no value)
    2. Stale running/queued scans - deleted if older than 1 hour (orphaned)
    3. Completed scans - keep only the most recent `keep_count`

    Args:
        db: Database session
        universe_key: Canonical universe key (e.g., "all", "exchange:NYSE", "custom:<hash>")
        keep_count: Number of recent scans to keep (default: 3)
    """
    from datetime import timedelta

    try:
        total_deleted_scans = 0
        total_deleted_results = 0

        # 1. Delete all cancelled scans for this universe (they have no value)
        cancelled_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status == "cancelled"
        ).all()

        if cancelled_scans:
            logger.info(f"Cleaning up {len(cancelled_scans)} cancelled scans for universe_key '{universe_key}'")
            for scan in cancelled_scans:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1
                logger.debug(f"Deleted cancelled scan {scan.scan_id} ({deleted_results} results)")

        # 2. Delete stale running/queued scans (older than 1 hour - they will never complete)
        stale_cutoff = datetime.utcnow() - timedelta(hours=1)
        stale_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status.in_(["running", "queued"]),
            Scan.started_at < stale_cutoff
        ).all()

        if stale_scans:
            logger.info(f"Cleaning up {len(stale_scans)} stale running/queued scans for universe_key '{universe_key}'")
            for scan in stale_scans:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1
                logger.debug(f"Deleted stale scan {scan.scan_id} (status={scan.status}, {deleted_results} results)")

        # 3. Keep only the last `keep_count` completed scans (existing logic)
        completed_scans = db.query(Scan).filter(
            Scan.universe_key == universe_key,
            Scan.status == "completed"
        ).order_by(Scan.completed_at.desc()).all()

        scans_to_delete = completed_scans[keep_count:]

        if scans_to_delete:
            logger.info(f"Cleaning up {len(scans_to_delete)} old completed scans for universe_key '{universe_key}'")
            for scan in scans_to_delete:
                deleted_results = db.query(ScanResult).filter(
                    ScanResult.scan_id == scan.scan_id
                ).delete()
                db.delete(scan)
                total_deleted_results += deleted_results
                total_deleted_scans += 1
                logger.debug(f"Deleted old scan {scan.scan_id} ({deleted_results} results)")

        if total_deleted_scans > 0:
            db.commit()
            logger.info(
                f"Cleanup complete for '{universe_key}': deleted {total_deleted_scans} scans, "
                f"{total_deleted_results} results. Kept {keep_count} most recent completed scans."
            )

    except Exception as e:
        logger.error(f"Error cleaning up old scans: {e}", exc_info=True)
        db.rollback()


def save_scan_result(
    db: Session,
    scan_id: str,
    symbol: str,
    result: Dict,
    ibd_group: Optional[str] = None,
    gics_data: Optional[Dict] = None,
    ibd_group_rank: Optional[int] = None
) -> None:
    """
    Save individual stock scan result to database.

    Supports both single screener (backward compatible) and multi-screener results.

    Args:
        db: Database session
        scan_id: UUID of scan
        symbol: Stock symbol
        result: Scan result dict from orchestrator or MinerviniScanner
        ibd_group: Optional pre-fetched IBD industry group (avoids DB query)
        gics_data: Optional pre-fetched GICS data dict with 'sector' and 'industry' keys
        ibd_group_rank: Optional pre-fetched IBD group rank (1=best, higher=worse)
    """
    try:
        # Convert numpy types to native Python types
        result_clean = convert_numpy_types(result)

        # Extract scores (handle both old and new formats)
        composite_score = result_clean.get('composite_score', result_clean.get('minervini_score', 0))
        minervini_score = result_clean.get('minervini_score')
        canslim_score = result_clean.get('canslim_score')
        ipo_score = result_clean.get('ipo_score')
        custom_score = result_clean.get('custom_score')
        volume_breakthrough_score = result_clean.get('volume_breakthrough_score')

        # Get rating (from orchestrator or calculate)
        rating = result_clean.get('rating')
        if not rating:
            # Backward compatibility: calculate rating
            passes_template = result_clean.get('passes_template', False)
            if passes_template and composite_score >= 80:
                rating = "Strong Buy"
            elif passes_template:
                rating = "Buy"
            elif composite_score >= 60:
                rating = "Watch"
            else:
                rating = "Pass"

        # Phase 3.3: Extract indexed fields from details
        stage = result_clean.get('stage')
        rs_rating = result_clean.get('rs_rating')
        rs_rating_1m = result_clean.get('rs_rating_1m')
        rs_rating_3m = result_clean.get('rs_rating_3m')
        rs_rating_12m = result_clean.get('rs_rating_12m')
        eps_growth_qq = result_clean.get('eps_growth_qq')
        sales_growth_qq = result_clean.get('sales_growth_qq')
        eps_growth_yy = result_clean.get('eps_growth_yy')
        sales_growth_yy = result_clean.get('sales_growth_yy')
        peg_ratio = result_clean.get('peg_ratio')
        adr_percent = result_clean.get('adr_percent')
        eps_rating = result_clean.get('eps_rating')  # IBD-style EPS Rating (0-99)
        ipo_date = result_clean.get('ipo_date')  # IPO date in YYYY-MM-DD format

        # Beta and Beta-Adjusted RS metrics
        beta = result_clean.get('beta')
        beta_adj_rs = result_clean.get('beta_adj_rs')
        beta_adj_rs_1m = result_clean.get('beta_adj_rs_1m')
        beta_adj_rs_3m = result_clean.get('beta_adj_rs_3m')
        beta_adj_rs_12m = result_clean.get('beta_adj_rs_12m')

        # RS Sparkline data (30-day stock/SPY ratio trend)
        rs_sparkline_data = result_clean.get('rs_sparkline_data')
        rs_trend = result_clean.get('rs_trend')

        # Price Sparkline data (30-day normalized price trend)
        price_sparkline_data = result_clean.get('price_sparkline_data')
        price_change_1d = result_clean.get('price_change_1d')
        price_trend = result_clean.get('price_trend')

        # Performance metrics (new technical filters)
        perf_week = result_clean.get('perf_week')
        perf_month = result_clean.get('perf_month')

        # Qullamaggie extended performance metrics
        perf_3m = result_clean.get('perf_3m')
        perf_6m = result_clean.get('perf_6m')

        # Episodic Pivot metrics
        gap_percent = result_clean.get('gap_percent')
        volume_surge = result_clean.get('volume_surge')

        # EMA distances (new technical filters)
        ema_10_distance = result_clean.get('ema_10_distance')
        ema_20_distance = result_clean.get('ema_20_distance')
        ema_50_distance = result_clean.get('ema_50_distance')

        # 52-week distances (promoted from details JSON)
        week_52_high_distance = result_clean.get('from_52w_high_pct')
        week_52_low_distance = result_clean.get('above_52w_low_pct')

        # Phase 4: Fetch industry classifications
        from ..services.ibd_industry_service import ibd_industry_service
        from ..models.stock import StockIndustry
        from ..services.data_fetcher import DataFetcher

        # Get IBD industry group (use pre-fetched if available, else query)
        if ibd_group is None:
            ibd_group = ibd_industry_service.get_industry_group(db, symbol)

        # Get GICS classifications (use pre-fetched if available, else query)
        gics_sector = None
        gics_industry = None

        if gics_data is not None:
            # Use pre-fetched GICS data (bulk query optimization)
            gics_sector = gics_data.get('sector')
            gics_industry = gics_data.get('industry')
        else:
            # Fallback: individual query (for backward compatibility)
            try:
                # Check if already cached
                gics_record = db.query(StockIndustry).filter(
                    StockIndustry.symbol == symbol
                ).first()

                if gics_record:
                    gics_sector = gics_record.sector
                    gics_industry = gics_record.industry
                else:
                    # Fetch from yfinance and cache it
                    data_fetcher = DataFetcher(db)
                    industry_info = data_fetcher.get_industry_classification(symbol)
                    if industry_info:
                        gics_sector = industry_info.get('sector')
                        gics_industry = industry_info.get('industry')
            except Exception as e:
                logger.warning(f"Could not fetch GICS data for {symbol}: {e}")

        # Create scan result record
        scan_result = ScanResult(
            scan_id=scan_id,
            symbol=symbol,
            composite_score=composite_score,
            minervini_score=minervini_score,
            canslim_score=canslim_score,
            ipo_score=ipo_score,
            custom_score=custom_score,
            volume_breakthrough_score=volume_breakthrough_score,
            rating=rating,
            price=result_clean.get('current_price'),
            volume=result_clean.get('avg_dollar_volume'),
            market_cap=result_clean.get('market_cap'),
            stage=stage,  # Phase 3.3: Indexed for fast filtering
            rs_rating=rs_rating,  # Phase 3.3: Indexed for fast filtering
            rs_rating_1m=rs_rating_1m,  # Multi-period RS for filtering
            rs_rating_3m=rs_rating_3m,  # Multi-period RS for filtering
            rs_rating_12m=rs_rating_12m,  # Multi-period RS for filtering
            eps_growth_qq=eps_growth_qq,  # QoQ EPS growth for filtering
            sales_growth_qq=sales_growth_qq,  # QoQ sales growth for filtering
            eps_growth_yy=eps_growth_yy,  # YoY EPS growth for filtering
            sales_growth_yy=sales_growth_yy,  # YoY sales growth for filtering
            peg_ratio=peg_ratio,  # PEG ratio for valuation filtering
            adr_percent=adr_percent,  # ADR % for volatility filtering
            eps_rating=eps_rating,  # IBD-style EPS Rating (0-99)
            ibd_industry_group=ibd_group,  # Phase 4: IBD industry group
            ibd_group_rank=ibd_group_rank,  # IBD group rank (1=best)
            gics_sector=gics_sector,  # Phase 4: GICS sector
            gics_industry=gics_industry,  # Phase 4: GICS industry
            rs_sparkline_data=rs_sparkline_data,  # RS sparkline (30-day ratio trend)
            rs_trend=rs_trend,  # RS trend direction (-1, 0, 1)
            price_sparkline_data=price_sparkline_data,  # Price sparkline (30-day trend)
            price_change_1d=price_change_1d,  # 1-day percentage change
            price_trend=price_trend,  # Price trend direction (-1, 0, 1)
            # Technical filter columns
            perf_week=perf_week,  # 5-day % change
            perf_month=perf_month,  # 21-day % change
            # Qullamaggie extended performance metrics
            perf_3m=perf_3m,  # 67-day % change
            perf_6m=perf_6m,  # 126-day % change
            # Episodic Pivot metrics
            gap_percent=gap_percent,  # Gap up %
            volume_surge=volume_surge,  # Volume ratio vs 50-day avg
            ema_10_distance=ema_10_distance,  # % from EMA10
            ema_20_distance=ema_20_distance,  # % from EMA20
            ema_50_distance=ema_50_distance,  # % from EMA50
            week_52_high_distance=week_52_high_distance,  # % below 52-week high
            week_52_low_distance=week_52_low_distance,  # % above 52-week low
            ipo_date=ipo_date,  # IPO date for age filtering
            # Beta and Beta-Adjusted RS metrics
            beta=beta,
            beta_adj_rs=beta_adj_rs,
            beta_adj_rs_1m=beta_adj_rs_1m,
            beta_adj_rs_3m=beta_adj_rs_3m,
            beta_adj_rs_12m=beta_adj_rs_12m,
            details=result_clean,  # Store full result (numpy types converted)
        )

        db.add(scan_result)
        # Batch commits now handled in run_bulk_scan loop

    except Exception as e:
        logger.error(f"Error saving scan result for {symbol}: {e}", exc_info=True)
        db.rollback()


def compute_industry_peer_metrics(db: Session, scan_id: str):
    """
    Compute aggregate metrics for each industry group in scan.
    Called after scan completion.

    Args:
        db: Database session
        scan_id: Scan ID
    """
    from ..models.industry import IBDGroupPeerCache
    from sqlalchemy import func

    try:
        logger.info(f"Computing industry peer metrics for scan {scan_id}...")

        # Query all industry groups in this scan
        groups = db.query(
            ScanResult.ibd_industry_group,
            func.count(ScanResult.symbol).label('total_stocks'),
            func.avg(ScanResult.rs_rating_1m).label('avg_rs_1m'),
            func.avg(ScanResult.rs_rating_3m).label('avg_rs_3m'),
            func.avg(ScanResult.rs_rating_12m).label('avg_rs_12m'),
            func.avg(ScanResult.minervini_score).label('avg_minervini_score'),
            func.avg(ScanResult.composite_score).label('avg_composite_score'),
        ).filter(
            ScanResult.scan_id == scan_id,
            ScanResult.ibd_industry_group.isnot(None)
        ).group_by(
            ScanResult.ibd_industry_group
        ).all()

        # Save cache records
        for group in groups:
            # Find top performer in this group
            top = db.query(ScanResult).filter(
                ScanResult.scan_id == scan_id,
                ScanResult.ibd_industry_group == group.ibd_industry_group
            ).order_by(
                ScanResult.composite_score.desc()
            ).first()

            cache = IBDGroupPeerCache(
                scan_id=scan_id,
                industry_group=group.ibd_industry_group,
                total_stocks=group.total_stocks,
                avg_rs_1m=group.avg_rs_1m,
                avg_rs_3m=group.avg_rs_3m,
                avg_rs_12m=group.avg_rs_12m,
                avg_minervini_score=group.avg_minervini_score,
                avg_composite_score=group.avg_composite_score,
                top_symbol=top.symbol if top else None,
                top_score=top.composite_score if top else None
            )
            db.add(cache)

        db.commit()
        logger.info(f"Computed peer metrics for {len(groups)} industry groups in scan {scan_id}")

    except Exception as e:
        logger.error(f"Error computing peer metrics: {e}", exc_info=True)
        db.rollback()


def _run_parallel_scan(
    task_instance,
    db: Session,
    scan_id: str,
    symbol_list: List[str],
    screener_types: List[str],
    composite_method: str,
    criteria: dict
) -> Dict:
    """
    Run scan in parallel by splitting into batches.

    Phase 3.1 optimization: Distributes stocks across batches and processes
    batches with internal parallelism.

    Args:
        task_instance: Celery task instance
        db: Database session
        scan_id: Scan ID
        symbol_list: All symbols to scan
        screener_types: List of screeners to run
        composite_method: Score combination method
        criteria: Scan criteria

    Returns:
        Dict with completion stats
    """
    import time

    total = len(symbol_list)
    batch_size = settings.scan_parallel_batch_size

    # Split symbols into batches
    batches = [
        symbol_list[i:i + batch_size]
        for i in range(0, total, batch_size)
    ]

    logger.info(
        f"Parallel scan: {total} stocks split into {len(batches)} batches "
        f"of ~{batch_size} stocks each"
    )

    # Process batches sequentially (can be parallelized further if needed)
    total_passed = 0
    total_failed = 0
    total_scanned = 0

    # Track scan start time for throughput-based ETA
    scan_start_time = time.time()

    for batch_num, batch_symbols in enumerate(batches):
        try:
            # Call internal batch scanning function with progress callback
            result = _scan_stock_batch_internal(
                scan_id=scan_id,
                symbols=batch_symbols,
                screener_types=screener_types,
                composite_method=composite_method,
                criteria=criteria,
                batch_num=batch_num + 1,
                task_instance=task_instance,
                total_stocks=total,
                stocks_completed_so_far=total_scanned,
                scan_start_time=scan_start_time
            )

            total_scanned += result.get('scanned', 0)
            total_passed += result.get('passed', 0)
            total_failed += result.get('failed', 0)

            # Calculate throughput and ETA
            elapsed = time.time() - scan_start_time
            stocks_per_second = total_scanned / elapsed if elapsed > 0 else 0
            remaining_stocks = total - total_scanned
            estimated_remaining_seconds = remaining_stocks / stocks_per_second if stocks_per_second > 0 else 0

            # Update task progress with accurate ETA
            progress = (total_scanned / total) * 100
            task_instance.update_state(
                state='PROGRESS',
                meta={
                    'current': total_scanned,
                    'total': total,
                    'percent': progress,
                    'passed': total_passed,
                    'failed': total_failed,
                    'throughput': round(stocks_per_second, 2),
                    'eta_seconds': round(estimated_remaining_seconds)
                }
            )

            # Log throughput metrics
            logger.info(
                f"Progress: {total_scanned}/{total} ({progress:.1f}%) | "
                f"Throughput: {stocks_per_second:.2f} stocks/sec | "
                f"ETA: {estimated_remaining_seconds/60:.1f} min | "
                f"Passed: {total_passed}"
            )

        except Exception as e:
            logger.error(f"Error processing batch {batch_num + 1}: {e}", exc_info=True)
            total_failed += len(batch_symbols)
            total_scanned += len(batch_symbols)

    # Update final scan status
    update_scan_status(db, scan_id, "completed", passed_stocks=total_passed)

    # Phase 4: Compute industry peer metrics
    compute_industry_peer_metrics(db, scan_id)

    # Phase 5: Cleanup old scans (keep only last 3 per universe_key)
    scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
    if scan and scan.universe_key:
        cleanup_old_scans(db, scan.universe_key)

    # Phase 6: Pre-warm chart cache for top 50 results (runs async)
    try:
        from .cache_tasks import prewarm_chart_cache_for_scan
        prewarm_chart_cache_for_scan.delay(scan_id, top_n=50)
        logger.info(f"Triggered chart cache warming for scan {scan_id}")
    except Exception as e:
        logger.warning(f"Could not trigger chart cache warming: {e}")

    logger.info(
        f"Parallel scan {scan_id} completed: {total_scanned} scanned, "
        f"{total_passed} passed, {total_failed} failed"
    )

    return {
        'scan_id': scan_id,
        'completed': total_scanned,
        'passed': total_passed,
        'failed': total_failed,
        'status': 'completed'
    }


def _scan_stock_batch_internal(
    scan_id: str,
    symbols: List[str],
    screener_types: List[str],
    composite_method: str,
    criteria: dict,
    batch_num: int,
    task_instance=None,
    total_stocks: int = 0,
    stocks_completed_so_far: int = 0,
    scan_start_time: float = None
) -> Dict:
    """
    Internal function to scan a batch of stocks in parallel.

    Phase 2 & 3 optimizations:
    - Phase 2: Bulk data fetching with yfinance.Tickers() for cache misses
    - Phase 3: Parallel processing with adaptive rate limiting

    Args:
        scan_id: UUID of scan
        symbols: List of symbols for this batch
        screener_types: List of screener names to run
        composite_method: Method for combining scores
        criteria: Scan criteria dict
        batch_num: Batch number for logging
        task_instance: Celery task instance for progress updates (optional)
        total_stocks: Total number of stocks in entire scan (for progress %)
        stocks_completed_so_far: Number of stocks already completed in previous batches
        scan_start_time: Unix timestamp when scan started (for ETA calculation)

    Returns:
        Dict with batch stats: {scanned, passed, failed}
    """
    import time
    db = SessionLocal()

    # Track progress within this batch
    batch_stocks_completed = 0

    try:
        logger.info(f"Batch {batch_num}: Starting scan of {len(symbols)} stocks")

        # PHASE 2 OPTIMIZATION: Bulk pre-fetch price data for entire batch
        # This reduces API overhead by using yfinance.Tickers() for cache misses
        if settings.use_bulk_fetching:
            from ..services.bulk_data_fetcher import BulkDataFetcher
            from ..services.price_cache_service import PriceCacheService

            logger.info(f"Batch {batch_num}: Pre-fetching data for {len(symbols)} symbols")

            # Get cache service
            price_cache = PriceCacheService.get_instance()

            # Check which symbols are NOT in Redis cache
            cache_misses = []
            for symbol in symbols:
                cache_stats = price_cache.get_cache_stats(symbol)
                if not cache_stats['redis_cached']:
                    cache_misses.append(symbol)

            if cache_misses:
                logger.info(
                    f"Batch {batch_num}: {len(cache_misses)} cache misses, "
                    f"{len(symbols) - len(cache_misses)} cache hits - bulk fetching {len(cache_misses)} symbols"
                )

                # Bulk fetch missing data
                bulk_fetcher = BulkDataFetcher()
                bulk_data = bulk_fetcher.fetch_batch_data(
                    cache_misses,
                    period='2y',
                    include_fundamentals=False  # Skip fundamentals for now (separate cache)
                )

                # Store fetched data in cache for use during scanning
                for symbol, data in bulk_data.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        # Store in cache so prepare_data() will use it
                        price_cache.store_in_cache(symbol, data['price_data'])
                        logger.debug(f"Batch {batch_num}: Cached {symbol} ({len(data['price_data'])} rows)")
            else:
                logger.info(f"Batch {batch_num}: All {len(symbols)} symbols found in cache")

        # PHASE 3 OPTIMIZATION: Bulk cache lookups using Redis pipelines
        # Pre-fetch ALL cache data for the batch using pipeline operations
        # This reduces Redis round trips from N*3 calls to just 3 pipeline calls
        logger.info(f"Batch {batch_num}: Phase 3 - Pre-fetching cache data using Redis pipelines")

        from ..scanners.base_screener import DataRequirements
        from ..scanners.screener_registry import screener_registry
        from ..wiring.bootstrap import get_stock_data_provider, get_scan_orchestrator

        # Get provider via wiring bootstrap
        provider = get_stock_data_provider()

        # Get screeners from registry and merge their data requirements
        # This ensures we only fetch what's actually needed
        screeners = screener_registry.get_multiple(screener_types)
        requirements = DataRequirements.merge_all([
            screener.get_data_requirements(criteria)
            for screener in screeners.values()
        ])

        logger.info(
            f"Batch {batch_num}: Merged data requirements - "
            f"fundamentals={requirements.needs_fundamentals}, "
            f"quarterly={requirements.needs_quarterly_growth}, "
            f"benchmark={requirements.needs_benchmark}"
        )

        # Bulk cache pre-fetch using Redis pipelines
        # This makes 3 pipeline calls (price, fundamentals, quarterly) instead of N*3 individual calls
        # For a 50-stock batch: 3 calls instead of 150 calls = 50x reduction!
        bulk_stock_data = provider.prepare_data_bulk(symbols, requirements)

        logger.info(
            f"Batch {batch_num}: Phase 3 bulk cache fetch completed - "
            f"{len(bulk_stock_data)} stocks prepared"
        )

        # Now proceed with normal scanning (will use pre-populated cache)

        # Always use orchestrator (it handles all screeners including minervini)
        orchestrator = get_scan_orchestrator()

        # Rate limiting state (adaptive)
        rate_limiter = Semaphore(settings.scan_parallel_workers)
        current_delay = settings.scan_parallel_rate_limit
        error_count = 0

        # Results collection
        results = []

        def scan_with_rate_limit(symbol: str) -> tuple[str, Optional[Dict]]:
            """Scan a single stock with rate limiting."""
            nonlocal current_delay, error_count

            # Scan the stock (semaphore controls concurrency)
            with rate_limiter:
                try:
                    # Scan the stock using orchestrator
                    # Pass pre-fetched data to avoid redundant fetching per stock (CRITICAL for performance!)
                    pre_fetched = bulk_stock_data.get(symbol.upper())
                    result = orchestrator.scan_stock_multi(
                        symbol=symbol.upper(),
                        screener_names=screener_types,
                        criteria=criteria,
                        composite_method=composite_method,
                        pre_merged_requirements=requirements,
                        pre_fetched_data=pre_fetched  # Use bulk pre-fetched data!
                    )

                    # Track success for adaptive rate limiting
                    if error_count > 0:
                        error_count = max(0, error_count - 1)
                        # Gradually reduce delay back to normal
                        if current_delay > settings.scan_parallel_rate_limit:
                            current_delay = max(
                                current_delay * 0.9,
                                settings.scan_parallel_rate_limit
                            )

                except Exception as e:
                    # Adaptive rate limiting: slow down on errors
                    error_count += 1
                    if settings.scan_rate_limit_adaptive and error_count >= 3:
                        old_delay = current_delay
                        current_delay = min(
                            current_delay * 1.5,
                            settings.scan_rate_limit_max
                        )
                        logger.warning(
                            f"Batch {batch_num}: Adaptive rate limiting - "
                            f"increased delay from {old_delay:.2f}s to {current_delay:.2f}s "
                            f"(errors: {error_count})"
                        )

                    logger.error(f"Batch {batch_num}: Error scanning {symbol}: {e}")
                    result = {'error': str(e)}

            # OPTIMIZATION: Skip rate limiting when data is cached
            # Phase 3 bulk pre-fetch ensures data is cached, so no API calls = no rate limiting needed
            # Only sleep if we had recent errors (adaptive rate limiting)
            if current_delay > 0 and error_count > 0:
                time.sleep(current_delay)

            return (symbol, result)

        # Execute scans in parallel
        with ThreadPoolExecutor(max_workers=settings.scan_parallel_workers) as executor:
            # Submit all stock scans
            futures = [executor.submit(scan_with_rate_limit, sym) for sym in symbols]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    symbol, result = future.result(timeout=60)
                    results.append((symbol, result))

                    # Update progress after each stock completes (real-time updates)
                    batch_stocks_completed += 1
                    if task_instance and total_stocks > 0:
                        total_completed = stocks_completed_so_far + batch_stocks_completed
                        progress = (total_completed / total_stocks) * 100

                        # Calculate ETA based on throughput
                        if scan_start_time:
                            elapsed = time.time() - scan_start_time
                            stocks_per_second = total_completed / elapsed if elapsed > 0 else 0
                            remaining_stocks = total_stocks - total_completed
                            eta_seconds = remaining_stocks / stocks_per_second if stocks_per_second > 0 else 0
                        else:
                            stocks_per_second = 0
                            eta_seconds = 0

                        # Update task state with current progress
                        task_instance.update_state(
                            state='PROGRESS',
                            meta={
                                'current': total_completed,
                                'total': total_stocks,
                                'percent': progress,
                                'throughput': round(stocks_per_second, 2),
                                'eta_seconds': round(eta_seconds)
                            }
                        )

                        # Log every 10 stocks to avoid log spam
                        if batch_stocks_completed % 10 == 0:
                            logger.info(
                                f"Batch {batch_num}: Progress {total_completed}/{total_stocks} "
                                f"({progress:.1f}%) | {stocks_per_second:.2f} stocks/sec"
                            )

                except Exception as e:
                    logger.error(f"Batch {batch_num}: Future failed: {e}")

        # Save all results to database in bulk
        passed = 0
        failed = 0

        # Bulk fetch industry data to eliminate N+1 queries (performance optimization)
        from ..services.ibd_industry_service import ibd_industry_service
        from ..models.stock import StockIndustry

        symbols_in_batch = [symbol for symbol, _ in results]

        # Bulk fetch IBD industry groups
        ibd_lookup = ibd_industry_service.get_bulk_industry_groups(db, symbols_in_batch)

        # Bulk fetch GICS data
        gics_lookup = {}
        if symbols_in_batch:
            gics_records = db.query(StockIndustry).filter(
                StockIndustry.symbol.in_(symbols_in_batch)
            ).all()
            gics_lookup = {
                record.symbol: {'sector': record.sector, 'industry': record.industry}
                for record in gics_records
            }

        # Bulk fetch latest IBD group ranks
        from ..models.industry import IBDGroupRank
        from sqlalchemy import desc

        ibd_rank_lookup = {}
        latest_rank_date = db.query(IBDGroupRank.date).order_by(desc(IBDGroupRank.date)).first()
        if latest_rank_date:
            for r in db.query(IBDGroupRank).filter(IBDGroupRank.date == latest_rank_date[0]).all():
                ibd_rank_lookup[r.industry_group] = r.rank

        # Save results with pre-fetched industry data
        for symbol, result in results:
            if result and 'error' not in result:
                ibd_group = ibd_lookup.get(symbol)
                gics_data = gics_lookup.get(symbol)
                ibd_group_rank = ibd_rank_lookup.get(ibd_group) if ibd_group else None
                save_scan_result(db, scan_id, symbol, result, ibd_group=ibd_group, gics_data=gics_data, ibd_group_rank=ibd_group_rank)
                if result.get('passes_template'):
                    passed += 1
            else:
                failed += 1

        # Batch commit all results
        db.commit()
        logger.info(
            f"Batch {batch_num}: Completed - {len(results)} scanned, "
            f"{passed} passed, {failed} failed"
        )

        return {
            'batch_num': batch_num,
            'scanned': len(results),
            'passed': passed,
            'failed': failed
        }

    except Exception as e:
        logger.error(f"Batch {batch_num}: Fatal error: {e}", exc_info=True)
        db.rollback()
        raise

    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.scan_tasks.scan_stock_batch')
@serialized_data_fetch('scan_stock_batch')
def scan_stock_batch(
    self,
    scan_id: str,
    symbols: List[str],
    screener_types: List[str],
    composite_method: str,
    criteria: dict,
    batch_num: int
) -> Dict:
    """
    Celery task wrapper for batch stock scanning.

    This is a Celery task that can be called asynchronously.
    For synchronous execution, use _scan_stock_batch_internal() directly.
    """
    return _scan_stock_batch_internal(
        scan_id=scan_id,
        symbols=symbols,
        screener_types=screener_types,
        composite_method=composite_method,
        criteria=criteria,
        batch_num=batch_num
    )


@celery_app.task(bind=True, name='app.tasks.scan_tasks.run_bulk_scan')
@serialized_data_fetch('run_bulk_scan')
def run_bulk_scan(self, scan_id: str, symbol_list: List[str], criteria: dict = None):
    """
    Scan multiple stocks in background.

    This is the main Celery task that processes bulk stock scans.

    Args:
        self: Celery task instance (bind=True)
        scan_id: UUID of scan record
        symbol_list: List of symbols to scan
        criteria: Scan criteria dict (include_vcp, min_rs_rating, etc.)

    Returns:
        Dict with completion stats
    """
    db = SessionLocal()

    try:
        total = len(symbol_list)
        logger.info(f"Starting bulk scan {scan_id} for {total} stocks")

        # Update scan status to running
        update_scan_status(db, scan_id, "running", total_stocks=total)

        # Get scan config from database
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        screener_types = scan.screener_types if scan and scan.screener_types else ["minervini"]
        composite_method = scan.composite_method if scan and scan.composite_method else "weighted_average"

        # Always use orchestrator for consistency
        # MinerviniScannerV2 is registered and will be used by orchestrator
        from ..wiring.bootstrap import get_scan_orchestrator
        orchestrator = get_scan_orchestrator()

        criteria = criteria or {}
        include_vcp = criteria.get('include_vcp', False)  # Default False for speed

        # Phase 3.1: Check feature flag for parallel scanning
        if settings.use_parallel_scanning:
            logger.info(f"Using PARALLEL scanning mode for {total} stocks")
            return _run_parallel_scan(
                self,
                db,
                scan_id,
                symbol_list,
                screener_types,
                composite_method,
                criteria
            )

        # Track progress
        completed = 0
        passed = 0
        failed = 0

        # Process stocks one by one (LEGACY MODE)
        logger.info(f"Using SEQUENTIAL scanning mode for {total} stocks")
        for i, symbol in enumerate(symbol_list):
            try:
                # Check if scan has been cancelled every 10 stocks
                if i % 10 == 0:
                    db.refresh(db.query(Scan).filter(Scan.scan_id == scan_id).first())
                    current_scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
                    if current_scan and current_scan.status == "cancelled":
                        logger.info(f"Scan {scan_id} was cancelled. Stopping at {completed}/{total} stocks.")
                        update_scan_status(db, scan_id, "cancelled", passed_stocks=passed)
                        return {
                            'scan_id': scan_id,
                            'completed': completed,
                            'passed': passed,
                            'failed': failed,
                            'status': 'cancelled'
                        }

                # Scan individual stock
                logger.debug(f"Scanning {symbol} ({i+1}/{total})")

                # Use orchestrator (handles all screeners including minervini)
                result = orchestrator.scan_stock_multi(
                    symbol=symbol.upper(),
                    screener_names=screener_types,
                    criteria=criteria,
                    composite_method=composite_method
                )

                # Check if scan returned valid result
                if result and 'error' not in result:
                    # Save result to database
                    save_scan_result(db, scan_id, symbol, result)

                    if result.get('passes_template'):
                        passed += 1
                else:
                    logger.warning(f"No valid result for {symbol}: {result.get('error', 'unknown error')}")
                    failed += 1

                completed += 1

                # Batch commit every 50 stocks for performance
                if completed % 50 == 0 or completed == total:
                    try:
                        db.commit()
                        logger.info(f"Batch committed at {completed}/{total} stocks")
                    except Exception as commit_error:
                        logger.error(f"Batch commit failed at {completed}/{total}: {commit_error}")
                        db.rollback()
                        raise

                # Update progress
                progress = (completed / total) * 100
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': completed,
                        'total': total,
                        'percent': progress,
                        'passed': passed,
                        'failed': failed
                    }
                )

                # Rate limiting (1 req/sec to respect yfinance limits)
                time.sleep(settings.scan_rate_limit)

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
                failed += 1
                completed += 1
                # Continue with next stock

        # Mark scan as completed
        update_scan_status(db, scan_id, "completed", passed_stocks=passed)

        # Phase 4: Compute industry peer metrics
        compute_industry_peer_metrics(db, scan_id)

        # Phase 5: Cleanup old scans (keep only last 3 per universe_key)
        scan = db.query(Scan).filter(Scan.scan_id == scan_id).first()
        if scan and scan.universe_key:
            cleanup_old_scans(db, scan.universe_key)

        # Phase 6: Pre-warm chart cache for top 50 results (runs async)
        try:
            from .cache_tasks import prewarm_chart_cache_for_scan
            prewarm_chart_cache_for_scan.delay(scan_id, top_n=50)
            logger.info(f"Triggered chart cache warming for scan {scan_id}")
        except Exception as e:
            logger.warning(f"Could not trigger chart cache warming: {e}")

        logger.info(
            f"Bulk scan {scan_id} completed: {completed} scanned, "
            f"{passed} passed, {failed} failed"
        )

        return {
            'scan_id': scan_id,
            'completed': completed,
            'passed': passed,
            'failed': failed,
            'status': 'completed'
        }

    except Exception as e:
        logger.error(f"Fatal error in bulk scan {scan_id}: {e}", exc_info=True)

        # Mark scan as failed
        try:
            update_scan_status(db, scan_id, "failed")
        except:
            pass

        # Re-raise exception so Celery marks task as failed
        raise

    finally:
        db.close()


@celery_app.task(name='app.tasks.scan_tasks.test_celery')
def test_celery():
    """
    Simple test task to verify Celery is working.

    Returns:
        Success message
    """
    logger.info("Test Celery task executed successfully")
    return {'status': 'success', 'message': 'Celery is working!'}

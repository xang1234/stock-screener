"""
Celery tasks for fundamental data updates.

Provides background tasks for:
- Weekly fundamental refresh for all stocks

All data-fetching tasks use the @serialized_data_fetch_task decorator
to ensure only one task fetches external data at a time.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
import time

from celery.exceptions import SoftTimeLimitExceeded

from ..celery_app import celery_app
from ..database import SessionLocal
from ..models.stock_universe import StockUniverse
from ..wiring.bootstrap import (
    get_eps_rating_service,
    get_fundamentals_cache,
    get_provider_snapshot_service,
    get_ticker_validation_service,
)
from ..services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_progress,
    mark_market_activity_started,
)
from ..services.ticker_validation_service import TickerValidationService
from .data_fetch_lock import serialized_data_fetch_task
from .transient_database import raise_if_transient_database_error

logger = logging.getLogger(__name__)
TRANSIENT_TASK_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
PROGRESS_PUBLISH_EVERY_STOCKS = 25
PROGRESS_PUBLISH_EVERY_SECONDS = 2.0
_GITHUB_SYNC_SUCCESS_STATUSES = frozenset({"success", "up_to_date"})


def _retry_transient_failure(task, task_name: str, exc: Exception) -> None:
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(60 * (2 ** retries), 600)
    logger.warning(
        "Transient error in %s: %s. Retrying in %ss (attempt %s/2).",
        task_name,
        exc,
        countdown,
        retries + 1,
    )
    raise task.retry(exc=exc, countdown=countdown, max_retries=2)


def _mark_market_activity_failed_safely(db, **kwargs) -> None:
    try:
        mark_market_activity_failed(db, **kwargs)
    except Exception:
        logger.warning(
            "Failed to publish market activity failure for fundamentals task",
            extra={
                "market": kwargs.get("market"),
                "stage_key": kwargs.get("stage_key"),
                "task_id": kwargs.get("task_id"),
            },
            exc_info=True,
        )


def _mark_market_activity_progress_safely(db, **kwargs) -> None:
    try:
        mark_market_activity_progress(db, **kwargs)
    except Exception:
        logger.warning(
            "Failed to publish market activity progress for fundamentals task",
            extra={
                "market": kwargs.get("market"),
                "stage_key": kwargs.get("stage_key"),
                "task_id": kwargs.get("task_id"),
            },
            exc_info=True,
        )


def _maybe_publish_fundamentals_progress(
    db,
    *,
    market: str,
    lifecycle: str,
    task_name: str,
    task_id: str | None,
    current: int,
    total: int,
    message: str,
    progress_state: dict[str, float | int],
    force: bool = False,
) -> None:
    if total <= 0:
        return

    now = time.monotonic()
    last_current = int(progress_state.get("last_current") or 0)
    last_time = float(progress_state.get("last_time") or 0.0)
    should_publish = force or current >= total
    if not should_publish:
        should_publish = (
            (last_current == 0 and current > 0)
            or
            (current - last_current) >= PROGRESS_PUBLISH_EVERY_STOCKS
            or (now - last_time) >= PROGRESS_PUBLISH_EVERY_SECONDS
        )
    if not should_publish:
        return

    _mark_market_activity_progress_safely(
        db,
        market=market,
        stage_key="fundamentals",
        lifecycle=lifecycle,
        task_name=task_name,
        task_id=task_id,
        current=current,
        total=total,
        percent=round((current / total) * 100, 1),
        message=message,
    )
    progress_state["last_current"] = current
    progress_state["last_time"] = now


def _runtime_market_scope() -> tuple[str, frozenset[str]] | None:
    """(primary market, enabled markets) from runtime preferences, or None if unreadable.

    None fails open (no market filter, the behaviour before #414) rather than
    silently stopping fundamentals refreshes when preferences can't be read.
    """
    from ..services.runtime_preferences_service import runtime_preferences_now

    try:
        prefs = runtime_preferences_now()
    except Exception:
        logger.warning(
            "Runtime preferences unreadable; fundamentals refresh is not market-filtered",
            exc_info=True,
        )
        return None
    return prefs.primary_market, frozenset(prefs.enabled_markets)


def _market_enabled(market: str, enabled_markets: frozenset[str] | None) -> bool:
    """Whether a scoped refresh may run; unreadable preferences (None) fail open."""
    return enabled_markets is None or market in enabled_markets


def _sync_weekly_bundles(
    db,
    *,
    effective_market: str,
    scoped_market: str | None,
    enabled_markets: frozenset[str] | None,
) -> tuple[dict, frozenset[str]]:
    """Sync the GitHub weekly bundle for every market in the refresh scope.

    Returns the primary (``effective_market``) sync result and the markets whose
    bundle did not sync. The GitHub fast path may complete the refresh only
    when nothing is pending; otherwise the run continues for the pending ones.
    """
    if scoped_market is not None or enabled_markets is None:
        markets = [effective_market]
    else:
        markets = [effective_market, *sorted(enabled_markets - {effective_market})]
    service = get_provider_snapshot_service()
    results = {
        sync_market: service.sync_weekly_reference_from_github(
            db,
            market=sync_market,
            hydrate_cache=True,
            hydrate_mode="static",
        )
        for sync_market in markets
    }
    pending = frozenset(
        sync_market
        for sync_market, result in results.items()
        if result.get("status") not in _GITHUB_SYNC_SUCCESS_STATUSES
    )
    return results[effective_market], pending


def _load_active_universe_stocks(
    db,
    *,
    market: str | None = None,
    enabled_markets: frozenset[str] | None = None,
) -> list[StockUniverse]:
    """Active universe rows for ``market``, or for every enabled market when unscoped."""
    query = db.query(StockUniverse).filter(StockUniverse.is_active)
    if market is not None:
        query = query.filter(StockUniverse.market == market)
    elif enabled_markets is not None:
        query = query.filter(StockUniverse.market.in_(sorted(enabled_markets)))
    return query.all()


def _resolve_refresh_scope(market: str | None) -> tuple[frozenset[str] | None, str, str | None]:
    """Return ``(enabled_markets, effective_market, scoped_market)`` for a weekly refresh.

    Unscoped runs cover every enabled market; activity and the GitHub bundle
    use the primary market (US unless the install runs non-US markets only).
    """
    from .market_queues import normalize_market

    scope = _runtime_market_scope()
    enabled_markets = scope[1] if scope is not None else None
    effective_market = (
        normalize_market(market) if market is not None
        else scope[0] if scope is not None else "US"
    )
    scoped_market = effective_market if market is not None else None
    return enabled_markets, effective_market, scoped_market


def _disabled_market_response(effective_market: str) -> Dict:
    return {
        'status': 'skipped',
        'reason': f'market {effective_market} is disabled in local runtime preferences',
        'market': effective_market,
        'timestamp': datetime.now().isoformat(),
    }


@dataclass
class _RefreshActivity:
    """Market-activity bookkeeping for the weekly fundamentals refresh."""

    db: Any
    market: str
    lifecycle: str
    task_name: str
    task_id: Optional[str]
    progress_state: dict = field(
        default_factory=lambda: {"last_current": 0, "last_time": time.monotonic()}
    )

    def _identity(self) -> dict:
        return {
            "market": self.market,
            "stage_key": "fundamentals",
            "lifecycle": self.lifecycle,
            "task_name": self.task_name,
            "task_id": self.task_id,
        }

    def started(self) -> None:
        mark_market_activity_started(self.db, **self._identity(), message="Refreshing fundamentals")

    def progress(self, current: int, total: int, *, force: bool = False) -> None:
        _maybe_publish_fundamentals_progress(
            self.db,
            market=self.market,
            lifecycle=self.lifecycle,
            task_name=self.task_name,
            task_id=self.task_id,
            current=current,
            total=total,
            message="Refreshing fundamentals",
            progress_state=self.progress_state,
            **({"force": True} if force else {}),
        )

    def completed(
        self,
        current: int | None,
        total: int | None,
        message: str = "Fundamentals refresh completed",
    ) -> None:
        mark_market_activity_completed(
            self.db, **self._identity(), current=current, total=total, message=message
        )

    def failed(self, message: str) -> None:
        mark_market_activity_failed(self.db, **self._identity(), message=message)

    def failed_safely(self, message: str, **counts) -> None:
        _mark_market_activity_failed_safely(self.db, **self._identity(), message=message, **counts)


def _complete_from_github_bundle(
    activity: _RefreshActivity,
    github_sync: Dict,
    *,
    scoped_market: str | None,
    enabled_markets: frozenset[str] | None,
    start_time: float,
) -> Dict:
    """Finish a refresh whose every in-scope market was synced from its GitHub bundle."""
    total_stocks = len(_load_active_universe_stocks(
        activity.db, market=scoped_market, enabled_markets=enabled_markets
    ))
    if total_stocks > 0:
        activity.progress(total_stocks, total_stocks, force=True)
    duration = time.time() - start_time
    eps_task = calculate_eps_rating_percentiles.delay()
    activity.completed(
        total_stocks or None,
        total_stocks or None,
        message="Fundamentals refresh completed from GitHub bundle",
    )
    return {
        "status": "success",
        "source": "github",
        "github_sync_status": github_sync.get("status"),
        "market": activity.market,
        "source_revision": github_sync.get("source_revision"),
        "import": github_sync.get("import"),
        "eps_rating_task_id": eps_task.id,
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
    }


def _log_refresh_summary(title: str, total_stocks: int, stats: Dict, duration: float) -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info(f"Total stocks: {total_stocks}")
    logger.info(f"Updated: {stats['updated']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Duration: {duration:.2f}s ({duration/60:.1f} minutes)")
    logger.info(f"Average: {duration/total_stocks:.2f}s per stock")
    if stats['failed_symbols'][:5]:
        logger.info(f"Failed symbols (first 5): {', '.join(stats['failed_symbols'][:5])}")
    logger.info("=" * 60)


@serialized_data_fetch_task(
    celery_app,
    "refresh_all_fundamentals",
    name='app.tasks.fundamentals_tasks.refresh_all_fundamentals',
    soft_time_limit=7200,
    max_retries=2,
)
def refresh_all_fundamentals(
    self,
    market: str | None = None,
    activity_lifecycle: str | None = None,
):
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
    from .market_queues import market_tag, log_extra
    _log_extra = log_extra(market)
    logger.info("=" * 60)
    logger.info("TASK: Weekly Fundamental Data Refresh %s", market_tag(market), extra=_log_extra)
    logger.info("Timestamp: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), extra=_log_extra)
    logger.info("=" * 60)

    enabled_markets, effective_market, scoped_market = _resolve_refresh_scope(market)
    if market is not None and not _market_enabled(effective_market, enabled_markets):
        logger.info("Skipping fundamentals refresh for disabled market %s", market, extra=_log_extra)
        return _disabled_market_response(effective_market)

    db = SessionLocal()
    start_time = time.time()
    ticker_validation_service = get_ticker_validation_service()
    activity = _RefreshActivity(
        db=db,
        market=effective_market,
        lifecycle=activity_lifecycle or "weekly_refresh",
        task_name=getattr(self, "name", "refresh_all_fundamentals"),
        task_id=getattr(getattr(self, "request", None), "id", None),
    )
    processed = 0
    total_stocks = 0

    try:
        activity.started()
        github_sync, pending_markets = _sync_weekly_bundles(
            db,
            effective_market=effective_market,
            scoped_market=scoped_market,
            enabled_markets=enabled_markets,
        )
        if not pending_markets:
            return _complete_from_github_bundle(
                activity,
                github_sync,
                scoped_market=scoped_market,
                enabled_markets=enabled_markets,
                start_time=start_time,
            )
        if scoped_market is None and enabled_markets is not None:
            # Markets refreshed from their GitHub bundle are done; continue with the rest.
            enabled_markets = pending_markets

        # Get all active stocks from universe (market-filtered when scoped)
        universe_stocks = _load_active_universe_stocks(
            db, market=scoped_market, enabled_markets=enabled_markets
        )

        if not universe_stocks:
            logger.warning("No active stocks found in universe", extra=_log_extra)
            activity.failed("No active stocks found")
            return {
                'error': 'No active stocks found',
                'timestamp': datetime.now().isoformat()
            }

        total_stocks = len(universe_stocks)
        logger.info(f"Found {total_stocks} active stocks to refresh")
        activity.progress(0, total_stocks, force=True)

        # Initialize cache service
        cache = get_fundamentals_cache()

        # Statistics
        stats = {
            'updated': 0,
            'failed': 0,
            'skipped': 0,
            'failed_symbols': []
        }

        for i, stock in enumerate(universe_stocks):
            symbol = stock.symbol
            try:
                # Log progress every 50 stocks
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{total_stocks} ({(i+1)/total_stocks*100:.1f}%)")

                # Fetch fundamental data (force_refresh=True to get fresh data).
                # Pass market from the already-loaded universe row so the cache
                # service doesn't re-query the DB for it per symbol.
                fundamental_data = cache.get_fundamentals(
                    symbol,
                    force_refresh=True,
                    market=stock.market,
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

            except SoftTimeLimitExceeded:
                raise
            except Exception as e:
                raise_if_transient_database_error(e)
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
            finally:
                processed = i + 1
                activity.progress(i + 1, total_stocks)

        activity.progress(total_stocks, total_stocks, force=True)

        duration = time.time() - start_time
        _log_refresh_summary("Weekly Fundamental Refresh Complete!", total_stocks, stats, duration)

        # Chain EPS rating percentiles calculation after fundamentals refresh
        logger.info("Queuing EPS Rating Percentiles calculation...")
        eps_task = calculate_eps_rating_percentiles.delay()
        logger.info(f"EPS Rating Percentiles task queued: {eps_task.id}")
        activity.completed(total_stocks, total_stocks)

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

    except SoftTimeLimitExceeded:
        db.rollback()
        logger.error("Soft time limit exceeded in refresh_all_fundamentals", exc_info=True)
        activity.failed_safely("Soft time limit exceeded", current=processed, total=total_stocks)
        raise
    except TRANSIENT_TASK_EXCEPTIONS as e:
        db.rollback()
        activity.failed_safely(str(e), current=processed, total=total_stocks)
        _retry_transient_failure(self, "refresh_all_fundamentals", e)
    except Exception as e:
        raise_if_transient_database_error(e)
        db.rollback()
        logger.error(f"Fatal error in fundamental refresh: {e}", exc_info=True)
        activity.failed_safely(str(e), current=processed, total=total_stocks)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

    finally:
        db.close()


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

    db = SessionLocal()
    start_time = time.time()
    eps_rating_service = get_eps_rating_service()

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
        cache = get_fundamentals_cache()
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

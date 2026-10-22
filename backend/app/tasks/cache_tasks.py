"""
Celery tasks for cache warming and maintenance.

Provides background tasks for:
- Daily cache warming after market close
- Weekly full cache refresh
- On-demand cache warming

All data-fetching tasks use the @serialized_data_fetch_task decorator
to ensure only one task fetches external data at a time.
"""
import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

from ..celery_app import celery_app
from ..config import settings
from ..database import SessionLocal, is_corruption_error, safe_rollback
from ..services.benchmark_registry_service import benchmark_registry
from ..services.cache_manager import CacheManager
from ..services.failed_price_retry_runner import (
    FailedPriceRetryRunner,
    FailedPriceRetryRunnerDependencies,
)
from ..services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_progress,
    mark_market_activity_started,
)
from ..services.price_fetch_failures import (
    is_no_data_price_failure,
)
from ..services.price_fetch_failures import (
    is_rate_limit_error as is_price_rate_limit_error,
)
from ..services.price_refresh_activity import (
    PriceRefreshActivityDependencies,
    PriceRefreshActivityReporter,
)
from ..services.price_refresh_live_runner import (
    LivePriceRefreshRunner,
    LivePriceRefreshRunnerDependencies,
    PriceRefreshRetryScheduler,
)
from ..services.price_refresh_plan_builder import build_market_price_refresh_plan
from ..services.price_refresh_workflow import (
    PriceRefreshMarketGateway,
    PriceRefreshWorkflow,
    PriceRefreshWorkflowDependencies,
)
from ..services.security_master_service import security_master_resolver
from ..utils.market_hours import (
    format_market_status,
    get_eastern_now,
    is_trading_day,
)
from ..wiring.bootstrap import (
    get_daily_price_bundle_service,
    get_market_calendar_service,
    get_rate_limiter,
    get_stock_universe_service,
)
from .data_fetch_lock import serialized_data_fetch_task
from .runtime_activity_failure_hooks import RuntimeActivityTrackedTask
from .transient_database import raise_if_transient_database_error

logger = logging.getLogger(__name__)

def _mark_market_activity_failed_safely(db, **kwargs) -> None:
    try:
        mark_market_activity_failed(db, **kwargs)
    except Exception:
        logger.warning(
            "Failed to publish market activity failure for cache task",
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
            "Failed to publish market activity progress for cache task",
            extra={
                "market": kwargs.get("market"),
                "stage_key": kwargs.get("stage_key"),
                "task_id": kwargs.get("task_id"),
            },
            exc_info=True,
        )


def _active_benchmark_markets(db, scope_market: Optional[str] = None) -> List[str]:
    """Return distinct active markets that have configured benchmarks.

    When ``scope_market`` is supplied, only that market's benchmark is
    returned. Prevents 4× redundant benchmark warmups when 4 per-market
    beat entries call warm_spy_cache in parallel. Unknown scopes fall
    back to US (safe default that keeps legacy SPY consumers working).
    """
    from ..models.stock_universe import StockUniverse

    supported = set(benchmark_registry.supported_markets())
    if scope_market is not None:
        scoped = scope_market.upper()
        return [scoped] if scoped in supported else ["US"]

    rows = (
        db.query(
            StockUniverse.market,
            StockUniverse.exchange,
            StockUniverse.symbol,
        )
        .filter(StockUniverse.is_active == True)
        .all()
    )
    markets = []
    for market, exchange, symbol in rows:
        normalized = security_master_resolver.normalize_market(market) or security_master_resolver.infer_market(
            symbol=symbol or "",
            exchange=exchange,
        )
        if normalized in supported:
            markets.append(normalized)
    # Preserve US benchmark warmups for legacy SPY consumers until migration is complete.
    markets.append("US")
    return sorted(set(markets))


def run_orphaned_scan_cleanup(
    *,
    session_factory=SessionLocal,
    now_utc: Optional[datetime] = None,
):
    """Delete cancelled and stale unfinished scans synchronously.

    This is the operator-safe implementation used both by the Celery task and
    by the manual maintenance script. It performs the cleanup itself rather
    than only queueing work for a background worker.
    """
    from datetime import timedelta

    from ..models.scan_result import Scan, ScanResult

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    logger.info("=" * 80)
    logger.info("Cleanup Orphaned Scans")
    logger.info("Timestamp: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)

    db = session_factory()

    try:
        total_deleted_scans = 0
        total_deleted_results = 0

        def _locked_scan_ids(*criteria):
            return [
                scan.scan_id
                for scan in (
                    db.query(Scan)
                    .filter(*criteria)
                    .with_for_update()
                    .all()
                )
            ]

        logger.info("[1/2] Deleting cancelled scans...")
        cancelled_scan_ids = _locked_scan_ids(Scan.status == "cancelled")
        if cancelled_scan_ids:
            cancelled_results = db.query(ScanResult).filter(
                ScanResult.scan_id.in_(cancelled_scan_ids)
            ).delete(synchronize_session=False)
            cancelled_scans = db.query(Scan).filter(
                Scan.scan_id.in_(cancelled_scan_ids)
            ).delete(synchronize_session=False)
            total_deleted_scans += cancelled_scans
            total_deleted_results += cancelled_results
            logger.info(
                "  Deleted %s cancelled scans, %s results",
                cancelled_scans,
                cancelled_results,
            )
        else:
            logger.info("  No cancelled scans found")

        logger.info("[2/2] Deleting stale running/queued scans...")
        stale_cutoff = now_utc - timedelta(hours=1)
        stale_scan_ids = _locked_scan_ids(
            Scan.status.in_(["running", "queued"]),
            Scan.started_at < stale_cutoff,
        )
        if stale_scan_ids:
            stale_results = db.query(ScanResult).filter(
                ScanResult.scan_id.in_(stale_scan_ids)
            ).delete(synchronize_session=False)
            stale_scans = db.query(Scan).filter(
                Scan.scan_id.in_(stale_scan_ids),
            ).delete(synchronize_session=False)
            total_deleted_scans += stale_scans
            total_deleted_results += stale_results
            logger.info(
                "  Deleted %s stale scans, %s results",
                stale_scans,
                stale_results,
            )
        else:
            logger.info("  No stale running/queued scans found")

        db.commit()

        logger.info("=" * 80)
        logger.info("Orphaned scan cleanup completed")
        logger.info("  Deleted scans: %s", total_deleted_scans)
        logger.info("  Deleted results: %s", total_deleted_results)
        logger.info("=" * 80)

        return {
            "deleted_scans": total_deleted_scans,
            "deleted_results": total_deleted_results,
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        if is_corruption_error(e):
            logger.critical(
                "DATABASE CORRUPTION in cleanup_orphaned_scans: %s — inspect the database "
                "and restore from backup or rerun migrations as appropriate",
                e,
            )
        else:
            logger.error("Error in cleanup_orphaned_scans task: %s", e, exc_info=True)
        safe_rollback(db)
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

    finally:
        db.close()


@celery_app.task(name='app.tasks.cache_tasks.refresh_universe_weights')
def refresh_universe_weights():
    """Recompute per-market universe weights for the rate-budget policy.

    Runs weekly via Beat (Sunday). The RateBudgetPolicy in-process cache has
    a 14-day stale fallback, but recomputing weekly keeps weights tracking
    universe-size shifts (new index memberships, delistings) before that
    fallback kicks in.
    """
    try:
        from ..services.rate_budget_policy import get_rate_budget_policy
        policy = get_rate_budget_policy()
        policy.invalidate_weights_cache()
        weights = policy._universe_weights(force_refresh=True)
        logger.info("Refreshed rate-budget universe weights: %s", weights)
        return {"status": "ok", "weights": weights}
    except Exception as exc:
        logger.warning("refresh_universe_weights failed: %s", exc)
        return {"status": "error", "error": str(exc)}


@celery_app.task(name='app.tasks.cache_tasks.warm_spy_cache')
def warm_spy_cache(market: Optional[str] = None):
    """Warm benchmark cache for active markets.

    Args:
        market: Optional market scope (US/HK/JP/TW). When supplied, only
            that market's benchmark is warmed — prevents 4× redundant work
            when called by per-market weekly_full_refresh tasks running in
            parallel. When None, warms benchmarks for all active markets
            (legacy behavior).
    """
    logger.info("=" * 60)
    logger.info("TASK: Warming Market Benchmark Cache (scope=%s)", market or "all")
    logger.info(f"Market status: {format_market_status()}")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        cache_manager = CacheManager()
        markets = _active_benchmark_markets(db, scope_market=market)
        logger.info(f"Warming benchmark cache for markets: {markets}")

        # Warm both 1y and 2y periods for each active market
        by_market = {}
        warmed_symbols: dict[str, set[str | None]] = {}
        failed_periods: list[str] = []
        for market in markets:
            outcomes = {
                period: cache_manager.warm_benchmark_cache(period=period, market=market)
                for period in ("2y", "1y")
            }
            by_market[market] = {period: outcome.ok for period, outcome in outcomes.items()}
            warmed_symbols[market] = {outcome.benchmark_symbol for outcome in outcomes.values() if outcome.ok}
            failed_periods.extend(
                f"{market}:{period} ({outcome.describe()})"
                for period, outcome in outcomes.items()
                if not outcome.ok
            )

        results = {
            'by_market': by_market,
            # Backward compatibility aliases for legacy SPY consumers.
            '2y': by_market.get("US", {}).get("2y", False),
            '1y': by_market.get("US", {}).get("1y", False),
            'market_status': format_market_status(),
            'timestamp': datetime.now().isoformat()
        }

        if failed_periods:
            results["error"] = "Benchmark warm failed for " + ", ".join(failed_periods)

        # Per-market telemetry (bead asia.10.1): record warms of the primary
        # only, so a failed warm, or one only the fallback served, doesn't
        # reset the primary's "age" gauge to zero.
        try:
            from ..services.benchmark_registry_service import benchmark_registry
            from ..services.telemetry import get_telemetry
            telemetry = get_telemetry()
            for warmed_market, symbols in warmed_symbols.items():
                primary = benchmark_registry.get_primary_symbol(warmed_market)
                if primary in symbols:
                    telemetry.record_benchmark_age(warmed_market, benchmark_symbol=primary)
        except Exception as exc:
            logger.debug("telemetry: benchmark_age emit failed (%s)", exc)

        logger.info("✓ Market benchmark cache warming task completed")
        return results

    except Exception as e:
        logger.error(f"Error in warm_spy_cache task: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    finally:
        db.close()


def _universe_rows(db) -> list[tuple[str, bool, str]]:
    """(SYMBOL, is_active, MARKET) for every universe row in every market."""
    from ..models.stock_universe import StockUniverse

    return [
        (str(symbol).strip().upper(), bool(is_active), str(market or "US").strip().upper())
        for symbol, is_active, market in db.query(
            StockUniverse.symbol, StockUniverse.is_active, StockUniverse.market
        )
    ]


_PRICE_KEY_SUFFIXES = ("recent", "last_update", "fetch_meta")


def _former_universe_symbols(db) -> set[str]:
    """Symbols with universe lifecycle history.

    A snapshot import replaces a market's rows (it deletes them and inserts
    only active ones), so a delisted symbol can vanish from stock_universe
    instead of turning inactive. Its status events survive: they have no
    foreign key to the universe row.
    """
    from ..models.stock_universe import StockUniverseStatusEvent

    return {
        str(symbol).strip().upper()
        for (symbol,) in db.query(StockUniverseStatusEvent.symbol).distinct()
    }


def _cleanup_orphaned_price_cache(
    universe_rows: list[tuple[str, bool, str]], former_symbols: set[str]
) -> int:
    """Delete Redis price keys of delisted symbols; returns symbols cleaned.

    An orphan is a symbol the universe holds, or once held (former_symbols),
    with no active row now. Symbols it
    never held (SPY, COT instruments, futures) are cached on purpose and kept,
    which also makes an empty or broken universe delete nothing. Keys are
    matched by name, ``price:<MARKET>:<SYMBOL>:<suffix>`` or the legacy
    ``price:<SYMBOL>:<suffix>``, so every market's keys go; deleting by symbol
    alone would target only the US keys. Redis errors propagate.
    """
    from ..services.redis_pool import get_redis_client

    active = {symbol for symbol, is_active, _ in universe_rows if is_active}
    # A market with no active rows (empty, unreadable, or mid-replacement)
    # says nothing about its delistings; with history alone, every symbol it
    # ever held would read as orphaned. Its keys are left alone.
    active_markets = {market for _, is_active, market in universe_rows if is_active}
    if not active_markets:
        logger.warning("Orphan cleanup skipped: the universe has no active symbols")
        return 0
    known = {symbol for symbol, _, _ in universe_rows} | set(former_symbols)
    orphaned = known - active
    if not orphaned:
        return 0
    redis_client = get_redis_client()
    if redis_client is None:
        raise RuntimeError("Redis is unavailable for the orphaned price-cache cleanup")
    doomed, cleaned = [], set()
    for key in redis_client.scan_iter(match="price:*", count=500):
        parts = (key.decode() if isinstance(key, bytes) else key).split(":")
        if parts[-1] not in _PRICE_KEY_SUFFIXES or len(parts) not in (3, 4):
            continue
        # Legacy unscoped keys are US keys (the cache policy's default market).
        market, symbol = (parts[1], parts[2]) if len(parts) == 4 else ("US", parts[1])
        if symbol in orphaned and market in active_markets:
            doomed.append(key)
            cleaned.add(symbol)
    for start in range(0, len(doomed), 500):
        redis_client.delete(*doomed[start:start + 500])
    return len(cleaned)


@celery_app.task(bind=True, name='app.tasks.cache_tasks.weekly_global_maintenance')
def weekly_global_maintenance(self):
    """Market-independent weekly cleanup (#615).

    Per-market weekly_full_refresh runs skip both steps (one market's symbols
    would make every other market's cache look orphaned), and Beat only runs
    the per-market refreshes, so this is their scheduled home.
    """
    from ..services.telemetry import get_telemetry

    # Independent of the cache, so it runs even if the Redis step fails.
    telemetry_deleted = get_telemetry().cleanup_old_events()
    db = SessionLocal()
    try:
        orphans = _cleanup_orphaned_price_cache(_universe_rows(db), _former_universe_symbols(db))
    finally:
        db.close()
    logger.info(
        "Weekly global maintenance: %d delisted symbols' cache removed, %d expired telemetry events",
        orphans, telemetry_deleted,
    )
    return {
        'orphans_cleaned': orphans,
        'telemetry_events_deleted': telemetry_deleted,
        'timestamp': datetime.now().isoformat(),
    }


@serialized_data_fetch_task(
    celery_app,
    "weekly_full_refresh",
    name='app.tasks.cache_tasks.weekly_full_refresh',
    soft_time_limit=14400,
)
def weekly_full_refresh(self, market: str | None = None):
    """
    Weekly full cache refresh.

    Cleans up orphaned cache keys (delisted/deactivated symbols),
    warms required benchmarks, then performs a full universe refresh inline.
    Typically runs Sunday morning before markets open.

    NOTE: Previously this task called smart_refresh_cache.delay(mode="full"),
    which spawned a duplicate task on the data_fetch queue. Now the full
    refresh logic is inlined to prevent task pileup.

    Returns:
        Dict with refresh results
    """
    import time

    from ..services.bulk_data_fetcher import BulkDataFetcher
    from ..services.runtime_preferences_service import is_market_enabled_now
    from ..wiring.bootstrap import get_price_cache
    from .market_queues import log_extra, market_tag, normalize_market

    _log_extra = log_extra(market)
    price_cache = get_price_cache()
    logger.info("=" * 80)
    logger.info("TASK: Weekly Full Cache Refresh %s", market_tag(market), extra=_log_extra)
    logger.info("Timestamp: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), extra=_log_extra)
    logger.info("=" * 80)

    if market is not None and not is_market_enabled_now(normalize_market(market)):
        logger.info("Skipping weekly full refresh for disabled market %s", market, extra=_log_extra)
        return {
            'status': 'skipped',
            'reason': f'market {normalize_market(market)} is disabled in local runtime preferences',
            'market': normalize_market(market),
            'timestamp': datetime.now().isoformat(),
        }

    db = SessionLocal()
    refreshed = 0
    processed = 0
    failed = 0
    failed_symbols = []

    try:
        from ..models.stock_universe import StockUniverse
        cache_manager = CacheManager(db)
        bulk_fetcher = BulkDataFetcher()

        # 1. Clean up delisted symbols' cache keys. The cleanup is global, so
        #    only the unscoped run does it; per-market runs leave it to
        #    weekly_global_maintenance (Beat, Sunday) rather than repeat it
        #    once per market.
        if market is None:
            logger.info("\n[1/4] Cleaning up orphaned cache entries...", extra=_log_extra)
            universe_rows = _universe_rows(db)
            active_symbols = {symbol for symbol, is_active, _ in universe_rows if is_active}
            orphan_count = _cleanup_orphaned_price_cache(universe_rows, _former_universe_symbols(db))
            logger.info("✓ Cleaned up %d orphaned cache entries", orphan_count, extra=_log_extra)
        else:
            logger.info(
                "\n[1/4] Skipping orphan cleanup for market-scoped run (%s) — "
                "weekly_global_maintenance does it for every market.",
                normalize_market(market), extra=_log_extra,
            )
            orphan_count = 0
            active_symbols = set()  # not used for cleanup in per-market path

        # 2. Warm benchmark cache (scoped to this market to avoid 4× redundant
        # work when 4 per-market refreshes run in parallel).
        logger.info("\n[2/4] Re-warming market benchmark cache...", extra=_log_extra)
        benchmark_results = warm_spy_cache(market=market)

        # 3. Get all active symbols ordered by market cap (market-filtered)
        _sym_q = db.query(StockUniverse.symbol).filter(StockUniverse.is_active)
        if market is not None:
            _sym_q = _sym_q.filter(StockUniverse.market == normalize_market(market))
        symbols = [
            r.symbol for r in _sym_q.order_by(StockUniverse.market_cap.desc().nullslast()).all()
        ]
        logger.info(
            "\n[3/4] Preparing full universe refresh (%d symbols)...",
            len(symbols), extra=_log_extra,
        )

        if not symbols:
            logger.warning("No active symbols found in universe")
            price_cache.save_warmup_metadata("completed", 0, 0, market=market)
            return {
                'orphans_cleaned': orphan_count,
                'benchmark': benchmark_results,
                'spy': benchmark_results.get("by_market", {}).get("US", {}),
                'refreshed': 0,
                'universe_size': 0,
                'completed_at': datetime.now().isoformat()
            }

        total = len(symbols)

        # Write initial heartbeat before batch loop
        price_cache.update_warmup_heartbeat(0, total, 0.0, market=market)

        # 4. Batch fetch all symbols (inline, no child task)
        logger.info(f"\n[4/4] Fetching {total} symbols...")
        from ..wiring.bootstrap import get_data_fetch_lock
        lock = get_data_fetch_lock()
        task_id = self.request.id or 'unknown'

        batch_size = 100
        for batch_start in range(0, total, batch_size):
            batch_symbols = symbols[batch_start:batch_start + batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (total + batch_size - 1) // batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: Fetching {len(batch_symbols)} symbols")

            batch_successes = []
            batch_failures = []
            failure_details = {}

            try:
                batch_results = _fetch_with_backoff(bulk_fetcher, batch_symbols, period="2y", market=market)
                batch_to_store = {}
                for symbol, data in batch_results.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        price_df = data['price_data']
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            batch_successes.append(symbol)
                        else:
                            failed += 1
                            failed_symbols.append(symbol)
                            batch_failures.append(symbol)
                            failure_details[symbol] = "Empty data returned"
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                        batch_failures.append(symbol)
                        failure_details[symbol] = data.get('error', 'Unknown error')
                # Batch store in Redis (pipeline) + DB (single transaction)
                if batch_to_store:
                    price_cache.store_batch_in_cache(batch_to_store, also_store_db=True)
                # Count only once stored: a store that raises persisted nothing.
                refreshed += len(batch_successes)
            except SoftTimeLimitExceeded:
                raise
            except Exception as e:
                raise_if_transient_database_error(e)
                logger.error(f"Batch {batch_num} error: {e}")
                # Fetch failures in this batch are already counted.
                newly_failed = [s for s in batch_symbols if s not in batch_failures]
                batch_successes.clear()
                failed += len(newly_failed)
                failed_symbols.extend(newly_failed)
                batch_failures.extend(newly_failed)
                failure_details.update({symbol: str(e) for symbol in newly_failed})

            # Track symbol failures for auto-deactivation
            _track_symbol_failures(
                price_cache,
                batch_successes,
                batch_failures,
                db,
                failure_details=failure_details,
            )

            # Update progress
            progress = min((batch_start + batch_size), total)
            percent = (progress / total) * 100
            self.update_state(state='PROGRESS', meta={
                'current': progress, 'total': total, 'percent': percent,
                'refreshed': refreshed, 'failed': failed
            })
            price_cache.update_warmup_heartbeat(progress, total, percent, market=market)

            # Extend lock TTL to prevent expiry during long runs (capped at 2h)
            lock.extend_lock(task_id, 300, market=market)

            # Rate limit between batches (per-market key when scoped)
            if batch_start + batch_size < total:
                if market is not None:
                    get_rate_limiter().wait_for_market("yfinance:batch", market)
                else:
                    get_rate_limiter().wait(
                        "yfinance:batch",
                        min_interval_s=settings.yfinance_batch_rate_limit_interval,
                    )

        # Save final metadata
        success_rate = refreshed / total if total > 0 else 0
        status = "completed" if success_rate >= 0.95 else "partial"
        price_cache.save_warmup_metadata(status, refreshed, total, market=market)
        price_cache.complete_warmup_heartbeat("completed", market=market)

        # Per-market telemetry (bead asia.10.1): freshness gauge + completeness
        # distribution for this market, plus opportunistic 15d retention cleanup.
        try:
            from ..services.telemetry import get_telemetry
            telemetry = get_telemetry()
            telemetry.record_freshness(
                market, source="prices", symbols_refreshed=refreshed,
            )
            telemetry.record_completeness_from_db(market)
            # Field-coverage snapshot (bead asia.10.5). Per-market only; the
            # SHARED scope has no field-capability semantics (policy chains
            # differ by market), so the emit method no-ops when market is None.
            if market is not None:
                telemetry.record_field_coverage_from_registry(market)
            # Cleanup runs once per weekly_full_refresh invocation; cheap on a
            # 15d-retained event log. Only the full-universe pass triggers it
            # to avoid 4× redundant deletes when 4 per-market refreshes run in
            # parallel.
            if market is None:
                deleted = telemetry.cleanup_old_events()
                if deleted:
                    logger.info("telemetry: cleaned %d old events", deleted)
        except Exception as exc:
            logger.debug("telemetry: emit/cleanup failed (%s)", exc)

        logger.info("=" * 80)
        logger.info(f"✓ Weekly full refresh completed:")
        logger.info(f"  Orphans cleaned: {orphan_count}")
        logger.info(f"  Refreshed: {refreshed}/{total}")
        logger.info(f"  Failed: {failed}")
        logger.info("=" * 80)

        return {
            'orphans_cleaned': orphan_count,
            'benchmark': benchmark_results,
            'spy': benchmark_results.get("by_market", {}).get("US", {}),
            'status': status,
            'refreshed': refreshed,
            'failed': failed,
            'total': total,
            'universe_size': len(symbols) if market is not None else len(active_symbols),
            'failed_symbols': failed_symbols[:20],
            'completed_at': datetime.now().isoformat()
        }

    except SoftTimeLimitExceeded:
        logger.error("Soft time limit exceeded in weekly_full_refresh", exc_info=True)
        safe_rollback(db)
        price_cache.save_warmup_metadata(
            "failed",
            refreshed,
            locals().get("total", 0),
            "Soft time limit exceeded",
            market=market,
        )
        price_cache.complete_warmup_heartbeat("failed", market=market)
        raise
    except Exception as e:
        raise_if_transient_database_error(e)
        logger.error(f"Error in weekly_full_refresh task: {e}", exc_info=True)
        price_cache.save_warmup_metadata("failed", refreshed, locals().get('total', 0), str(e), market=market)
        price_cache.complete_warmup_heartbeat("failed", market=market)
        return {
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        }

    finally:
        db.close()


def _is_rate_limit_error(error_str: str) -> bool:
    """Check if an error string indicates a rate limit."""
    return is_price_rate_limit_error(error_str)


class RateLimitExhaustedError(RuntimeError):
    """Raised when provider rate limiting persists after all backoff attempts."""

    def __init__(self, symbols: List[str], market: str | None) -> None:
        self.symbols = list(symbols)
        self.market = market
        super().__init__(
            f"rate_limit_exhausted market={market or 'shared'} symbols={len(symbols)}"
        )


def _fetch_with_backoff(
    bulk_fetcher,
    symbols: List[str],
    period: str = "2y",
    max_retries: int = 3,
    market: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
):
    """
    Fetch batch data with exponential backoff on rate limit errors.

    yfinance can return 429 errors or rate limit blocks in two ways:
    1. Raising an exception (caught in the except block)
    2. Returning per-symbol errors with has_error=True (the common case)

    This function detects both and retries with exponential backoff
    (60s, 120s, 240s) before giving up.

    When ``market`` is supplied, ``fetch_prices_in_batches`` uses the
    per-market batch size from ProviderDataPlanRegistry instead of the
    hard-coded default, so the configured market plan actually applies.

    Args:
        bulk_fetcher: BulkDataFetcher instance
        symbols: List of symbols to fetch
        period: Data period (default "2y")
        max_retries: Maximum retry attempts (default 3)
        market: Optional market scope — when set, defers batch sizing to
            ProviderDataPlanRegistry (pass None for legacy global behaviour)

    Returns:
        Dict of symbol -> data, or empty dict if all retries fail
    """
    for attempt in range(max_retries):
        try:
            # When market is set, omit start_batch_size so fetch_prices_in_batches
            # uses the per-market provider-plan batch size. For shared/None runs
            # keep the hard-coded default to preserve pre-9.1 behaviour.
            if market is not None:
                results = bulk_fetcher.fetch_prices_in_batches(
                    symbols,
                    period=period,
                    market=market,
                    progress_callback=progress_callback,
                )
            else:
                results = bulk_fetcher.fetch_prices_in_batches(
                    symbols,
                    period=period,
                    start_batch_size=min(
                        len(symbols),
                        getattr(bulk_fetcher, "DEFAULT_PRICE_BATCH_SIZE", 100),
                    ),
                    progress_callback=progress_callback,
                )

            return results

        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit indicators in the exception
            if _is_rate_limit_error(error_str):
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limited by yfinance (exception). Retrying via shared batch adapter "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                else:
                    logger.error(
                        f"Rate limit persists after {max_retries} retries. "
                        f"Raising tagged failure for batch of {len(symbols)} symbols."
                    )
                    raise RateLimitExhaustedError(symbols, market)
            else:
                # Re-raise non-rate-limit errors
                raise
    raise RateLimitExhaustedError(symbols, market)


def _classify_no_data_failure(error_message: str) -> bool:
    """Heuristic classifier for delisted/no-data provider failures."""
    return is_no_data_price_failure(error_message)


def _track_symbol_failures(
    price_cache,
    successes: List[str],
    failures: List[str],
    db=None,
    failure_details: Optional[dict[str, str]] = None,
):
    """
    Track per-symbol fetch failures and auto-deactivate persistently failing symbols.

    Symbols that fail 5+ consecutive refreshes with no-data/delisted indicators
    transition to inactive_no_data to stop wasting provider traffic.

    Args:
        price_cache: PriceCacheService instance
        successes: Symbols that fetched successfully this batch
        failures: Symbols that failed this batch
        db: Optional SQLAlchemy session (if None, opens a new one)
    """
    if not successes and not failures:
        return

    failure_details = failure_details or {}
    owns_db = db is None
    if owns_db:
        db = SessionLocal()

    try:
        # Clear failure counters for successful symbols
        stock_universe_service = get_stock_universe_service()

        skipped_corrupt_symbols: List[str] = []

        for symbol in successes:
            price_cache.clear_symbol_failure(symbol)
            try:
                with db.begin_nested():
                    stock_universe_service.record_fetch_success(db, symbol)
                    db.flush()
            except Exception as exc:
                if not is_corruption_error(exc):
                    raise
                skipped_corrupt_symbols.append(symbol)
                db.expire_all()

        deactivated = []
        for symbol in failures:
            count = price_cache.record_symbol_failure(symbol)
            try:
                with db.begin_nested():
                    details = stock_universe_service.record_fetch_failure(
                        db,
                        symbol,
                        reason=(
                            "Repeated no-data provider responses"
                            if _classify_no_data_failure(failure_details.get(symbol, ""))
                            else "Repeated provider fetch failures"
                        ),
                        trigger_source="price_refresh",
                        no_data=_classify_no_data_failure(failure_details.get(symbol, "")),
                        deactivate_threshold=price_cache.SYMBOL_FAILURE_THRESHOLD,
                    )
                    db.flush()
            except Exception as exc:
                if not is_corruption_error(exc):
                    raise
                skipped_corrupt_symbols.append(symbol)
                db.expire_all()
                continue
            if count >= price_cache.SYMBOL_FAILURE_THRESHOLD and details.get("deactivated"):
                deactivated.append(symbol)

        try:
            db.commit()
        except Exception as exc:
            if not is_corruption_error(exc):
                raise
            logger.warning(
                "Skipping stock_universe fetch tracking commit for %d symbols due to database "
                "corruption signature: %s",
                len(successes) + len(failures),
                exc,
            )
            safe_rollback(db)
            return

        if skipped_corrupt_symbols:
            logger.warning(
                "Skipped stock_universe fetch-tracking writes for %d symbols due to database "
                "corruption signatures. Sample symbols: %s",
                len(skipped_corrupt_symbols),
                skipped_corrupt_symbols[:10],
            )
        if deactivated:
            logger.info(
                "Auto-deactivated %d persistently failing symbols: %s",
                len(deactivated),
                deactivated,
            )

    except Exception as e:
        logger.error(f"Error auto-deactivating symbols: {e}", exc_info=True)
        db.rollback()

    finally:
        if owns_db:
            db.close()


def _record_market_refresh_success_safely(
    db,
    *,
    market: str | None,
    trading_day,
    success_rate: float,
) -> None:
    if market is None:
        return
    try:
        from ..services.market_refresh_state_service import (
            record_market_refresh_success,
        )

        record_market_refresh_success(
            db,
            market=market,
            trading_day=trading_day,
            success_rate=success_rate,
        )
    except Exception:
        logger.warning(
            "Failed to record market refresh state for %s",
            market,
            exc_info=True,
        )


def _schedule_failed_symbol_retry(
    symbols: list[str],
    *,
    market: str | None,
    attempt: int = 1,
    countdown: int = 600,
) -> None:
    if not symbols or market is None or attempt > 3:
        return
    try:
        from .market_queues import data_fetch_queue_for_market, normalize_market

        normalized_market = normalize_market(market)
        retry_failed_price_symbols.apply_async(
            kwargs={
                "symbols": list(dict.fromkeys(symbols)),
                "market": normalized_market,
                "attempt": attempt,
                "retry_countdown": countdown,
            },
            countdown=countdown,
            queue=data_fetch_queue_for_market(normalized_market),
        )
    except Exception:
        logger.warning(
            "Failed to schedule price retry for %s failed symbols",
            len(symbols),
            exc_info=True,
        )


def _group_symbols_by_market(symbols: List[str]) -> List[tuple]:
    """Group symbols by active-universe market, preserving order within each group.

    Provider plans are chosen per market (CN AKShare/BaoStock, KR KRX, HK close
    repair); an unscoped fetch defaults to the US plan. Falls back to a single
    unscoped group when the universe lookup fails.
    """
    from ..models.stock_universe import StockUniverse

    try:
        db = SessionLocal()
        try:
            rows = (
                db.query(StockUniverse.symbol, StockUniverse.market)
                .filter(StockUniverse.symbol.in_(symbols))
                .all()
            )
        finally:
            db.close()
    except Exception:
        logger.warning("Market lookup failed for stale refresh; fetching unscoped", exc_info=True)
        return [(None, list(symbols))]

    market_by_symbol = {
        symbol: security_master_resolver.normalize_market(market) for symbol, market in rows
    }
    groups: dict = {}
    for symbol in symbols:
        groups.setdefault(market_by_symbol.get(symbol), []).append(symbol)
    return list(groups.items())


def _filter_active_symbols(symbols: List[str]) -> List[str]:
    """Limit stale-refresh batches to active universe symbols only."""
    if not symbols:
        return []

    try:
        stock_universe_service = get_stock_universe_service()
    except RuntimeError:
        # Helper is used directly in tests/scripts outside full runtime bootstrapping.
        from ..services.stock_universe_service import StockUniverseService

        stock_universe_service = StockUniverseService()

    db = SessionLocal()
    try:
        filtered = stock_universe_service.filter_active_symbols(db, symbols)
        filtered_set = set(filtered)
        dropped = [symbol for symbol in symbols if symbol.upper() not in filtered_set]
        if dropped:
            logger.info(
                "Skipping %d inactive/unknown symbols during stale refresh: %s",
                len(dropped),
                ", ".join(dropped[:10]),
            )
        return filtered
    finally:
        db.close()


@serialized_data_fetch_task(
    celery_app,
    "retry_failed_price_symbols",
    name="app.tasks.cache_tasks.retry_failed_price_symbols",
    soft_time_limit=3600,
)
def retry_failed_price_symbols(
    self,
    symbols: list[str],
    market: str,
    attempt: int = 1,
    retry_countdown: int = 600,
) -> dict:
    from ..services.bulk_data_fetcher import BulkDataFetcher
    from ..wiring.bootstrap import get_price_cache

    deduped_symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols if symbol))
    if not deduped_symbols:
        return {
            "status": "completed",
            "market": market,
            "attempt": attempt,
            "refreshed": 0,
            "failed": 0,
        }

    price_cache = get_price_cache()
    bulk_fetcher = BulkDataFetcher()
    result = FailedPriceRetryRunner(
        FailedPriceRetryRunnerDependencies(
            fetch_with_backoff=_fetch_with_backoff,
            track_symbol_failures=_track_symbol_failures,
            raise_if_transient_database_error=raise_if_transient_database_error,
            schedule_failed_symbol_retry=_schedule_failed_symbol_retry,
        )
    ).run(
        price_cache=price_cache,
        bulk_fetcher=bulk_fetcher,
        symbols=deduped_symbols,
        market=market,
        attempt=attempt,
        retry_countdown=retry_countdown,
        batch_size=settings.price_refresh_live_batch_size,
    )
    payload = {
        "status": result.status,
        "market": market,
        "attempt": attempt,
        "refreshed": result.refreshed,
        "failed": result.failed,
        "failed_symbols": list(result.failed_symbols[:20]),
        "completed_at": datetime.now().isoformat(),
    }
    if result.error is not None:
        payload["error"] = result.error
    return payload


def _force_refresh_stale_intraday_impl(task, symbols: Optional[List[str]] = None) -> dict:
    """
    Core implementation of stale intraday data refresh.

    Args:
        task: Celery task instance (for update_state), or None
        symbols: Symbols to refresh, or None for auto-detect

    Returns:
        Dict with refresh statistics
    """
    import time

    from ..services.bulk_data_fetcher import BulkDataFetcher
    from ..wiring.bootstrap import get_price_cache

    logger.info("=" * 80)
    logger.info("TASK: Force Refresh Stale Intraday Data (Batch Mode)")
    logger.info(f"Market status: {format_market_status()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    price_cache = get_price_cache()
    bulk_fetcher = BulkDataFetcher()

    try:
        # Get symbols to refresh
        if symbols is None:
            symbols = price_cache.get_stale_intraday_symbols()
            logger.info(f"Auto-detected {len(symbols)} symbols with stale intraday data")
        else:
            logger.info(f"Refreshing {len(symbols)} specified symbols")

        symbols = _filter_active_symbols(symbols)
        logger.info(f"Stale intraday refresh scoped to {len(symbols)} active symbols")

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

        # Process symbols in per-market batches so each market's provider plan applies
        batch_size = 100  # Reduced from 200 to avoid rate limiting
        batches = [
            (batch_market, group[start:start + batch_size])
            for batch_market, group in _group_symbols_by_market(symbols)
            for start in range(0, len(group), batch_size)
        ]
        total_batches = len(batches)
        processed = 0

        for batch_num, (batch_market, batch_symbols) in enumerate(batches, 1):
            logger.info(
                "Batch %d/%d: Fetching %d active %s symbols",
                batch_num,
                total_batches,
                len(batch_symbols),
                batch_market or "unscoped",
            )

            try:
                # Batch fetch with rate limit backoff
                batch_results = _fetch_with_backoff(
                    bulk_fetcher, batch_symbols, period="2y", market=batch_market
                )

                # Separate successes from failures, then batch-store
                batch_to_store = {}
                for symbol, data in batch_results.items():
                    if not data.get('has_error') and data.get('price_data') is not None:
                        price_df = data['price_data']
                        if not price_df.empty:
                            batch_to_store[symbol] = price_df
                            logger.debug(f"✓ {symbol}: {len(price_df)} rows fetched")
                        else:
                            failed += 1
                            failed_symbols.append(symbol)
                            logger.warning(f"✗ {symbol}: Empty data returned")
                    else:
                        failed += 1
                        failed_symbols.append(symbol)
                        error_msg = data.get('error', 'Unknown error')
                        logger.warning(f"✗ {symbol}: {error_msg}")

                # Batch store in Redis (pipeline, every key namespace) + DB (single transaction)
                if batch_to_store:
                    price_cache.store_refreshed_batch(batch_to_store)
                # Count only once stored: a store that raises persisted nothing.
                refreshed += len(batch_to_store)

            except Exception as e:
                raise_if_transient_database_error(e)
                logger.error(f"Batch {batch_num} error: {e}")
                # Fetch failures in this batch are already counted.
                newly_failed = [s for s in batch_symbols if s not in failed_symbols]
                failed += len(newly_failed)
                failed_symbols.extend(newly_failed)

            # Update task state for progress tracking
            processed += len(batch_symbols)
            progress = processed
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
            if batch_num < total_batches:
                get_rate_limiter().wait(
                    "yfinance:batch",
                    min_interval_s=settings.yfinance_batch_rate_limit_interval,
                )

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
        raise_if_transient_database_error(e)
        logger.error(f"Error in force_refresh_stale_intraday: {e}", exc_info=True)
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@serialized_data_fetch_task(
    celery_app,
    "force_refresh_stale_intraday",
    name="app.tasks.cache_tasks.force_refresh_stale_intraday",
)
def force_refresh_stale_intraday(self, symbols: Optional[List[str]] = None):
    """
    Force refresh symbols with stale intraday data.

    This task refreshes price data for symbols that were fetched during
    market hours and are now stale (market has closed). The "today" bar
    that was incomplete during market hours is replaced with actual
    closing data.

    Uses the shared yf.download() batch adapter for efficient fetching.

    Args:
        symbols: List of symbols to refresh. If None, auto-detects
                 all symbols with stale intraday data.

    Returns:
        Dict with refresh statistics
    """
    return _force_refresh_stale_intraday_impl(self, symbols)


def run_smart_price_refresh(
    *,
    task,
    mode: str,
    market: str | None,
    activity_lifecycle: str | None,
    ensure_group_history: bool = False,
) -> dict:
    from ..services.bulk_data_fetcher import BulkDataFetcher
    from ..services.runtime_preferences_service import is_market_enabled_now
    from ..wiring.bootstrap import get_data_fetch_lock, get_price_cache
    from .market_queues import log_extra, market_tag, normalize_market

    def build_refresh_plan(
        db,
        *,
        mode,
        market,
        effective_market,
        recently_refreshed_filter,
        ensure_group_history,
    ):
        return build_market_price_refresh_plan(
            db,
            mode=mode,
            market=market,
            effective_market=effective_market,
            normalize_market=normalize_market,
            market_calendar_service=get_market_calendar_service(),
            sync_github_seed=lambda sync_db, *, market, allow_stale: (
                get_daily_price_bundle_service().sync_from_github(
                    sync_db,
                    market=market,
                    allow_stale=allow_stale,
                )
            ),
            recently_refreshed_filter=recently_refreshed_filter,
            ensure_group_history=ensure_group_history,
        )

    activity_reporter = PriceRefreshActivityReporter(
        PriceRefreshActivityDependencies(
            record_market_refresh_success=_record_market_refresh_success_safely,
            mark_market_activity_started=mark_market_activity_started,
            mark_market_activity_completed=mark_market_activity_completed,
            mark_market_activity_progress_safely=_mark_market_activity_progress_safely,
            mark_market_activity_failed_safely=_mark_market_activity_failed_safely,
        )
    )
    live_runner = LivePriceRefreshRunner(
        LivePriceRefreshRunnerDependencies(
            fetch_with_backoff=_fetch_with_backoff,
            track_symbol_failures=_track_symbol_failures,
            data_fetch_lock_factory=get_data_fetch_lock,
            raise_if_transient_database_error=raise_if_transient_database_error,
        )
    )
    dependencies = PriceRefreshWorkflowDependencies(
        session_factory=SessionLocal,
        price_cache_factory=get_price_cache,
        bulk_fetcher_factory=BulkDataFetcher,
        warm_benchmarks=warm_spy_cache,
        build_refresh_plan=build_refresh_plan,
        last_completed_trading_day=lambda market_code: (
            get_market_calendar_service().last_completed_trading_day(market_code)
        ),
        activity_reporter=activity_reporter,
        live_runner=live_runner,
        retry_scheduler=PriceRefreshRetryScheduler(_schedule_failed_symbol_retry),
        market_gateway=PriceRefreshMarketGateway(
            normalize_market=normalize_market,
            market_tag=market_tag,
            log_extra=log_extra,
            get_eastern_now=get_eastern_now,
            is_trading_day=is_trading_day,
            format_market_status=format_market_status,
            is_market_enabled_now=is_market_enabled_now,
        ),
        raise_if_transient_database_error=raise_if_transient_database_error,
        safe_rollback=safe_rollback,
    )
    return PriceRefreshWorkflow(dependencies).run(
        task=task,
        mode=mode,
        market=market,
        activity_lifecycle=activity_lifecycle,
        ensure_group_history=ensure_group_history,
    )


@serialized_data_fetch_task(
    celery_app,
    "smart_refresh_cache",
    base=RuntimeActivityTrackedTask,
    name='app.tasks.cache_tasks.smart_refresh_cache',
    soft_time_limit=14400,
)
def smart_refresh_cache(
    self,
    mode: str = "auto",
    market: str | None = None,
    activity_lifecycle: str | None = None,
    ensure_group_history: bool = False,
):
    """Run the smart price refresh workflow behind the Celery data-fetch lock."""
    return run_smart_price_refresh(
        task=self,
        mode=mode,
        market=market,
        activity_lifecycle=activity_lifecycle,
        ensure_group_history=ensure_group_history,
    )


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

    from sqlalchemy import func

    from ..models.stock import StockPrice
    from ..models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse

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
        active_symbol_subquery = db.query(StockUniverse.symbol).filter(
            StockUniverse.active_filter()
        )
        delete_count = db.query(func.count(StockPrice.id)).filter(
            StockPrice.date < cutoff_date,
            StockPrice.symbol.in_(active_symbol_subquery),
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
            ids_to_delete = [
                row[0]
                for row in db.query(StockPrice.id).filter(
                    StockPrice.date < cutoff_date,
                    StockPrice.symbol.in_(active_symbol_subquery),
                ).limit(batch_size).all()
            ]

            if not ids_to_delete:
                break

            deleted = db.query(StockPrice).filter(
                StockPrice.id.in_(ids_to_delete)
            ).delete(synchronize_session=False)

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
        if is_corruption_error(e):
            logger.critical(
                "DATABASE CORRUPTION in cleanup_old_price_data: %s — inspect the database "
                "and restore from backup or rerun migrations as appropriate",
                e,
            )
        else:
            logger.error("Error in cleanup_old_price_data task: %s", e, exc_info=True)
        safe_rollback(db)
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
    from ..wiring.bootstrap import get_price_cache

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
            ScanResult.composite_score.desc().nullslast()
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
        price_cache = get_price_cache()
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
    """Queueable Celery entrypoint for orphaned scan cleanup."""
    return run_orphaned_scan_cleanup()

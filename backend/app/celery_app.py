"""
Celery configuration for background task processing.

Handles async bulk scanning of stocks and cache warming.
"""
import logging
import os

# Disable MPS/Metal before any PyTorch imports to avoid fork() issues on macOS
# Must be set at the very start before any libraries that use PyTorch are imported
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable macOS Objective-C fork safety check
# Keeps worker startup stable for libraries that initialize Objective-C runtime
# after Celery forks worker processes.
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init, worker_ready, worker_shutting_down
from .config import settings
from .domain.scanning.defaults import get_default_scan_profile


def _offset_schedule(hour: int, minute: int, offset_minutes: int) -> tuple[int, int]:
    """Compute (hour, minute) with proper carry when offset pushes minute past 59."""
    total_minutes = minute + offset_minutes
    return (hour + total_minutes // 60) % 24, total_minutes % 60

# Import scanners to trigger registration
# This ensures all screeners are registered with the registry before tasks run
import app.scanners  # noqa: F401

# Create Celery application instance
celery_app = Celery(
    "stock_scanner",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'app.tasks.scan_tasks',
        'app.tasks.cache_tasks',  # Cache warming tasks
        'app.tasks.breadth_tasks',  # Market breadth tasks
        'app.tasks.fundamentals_tasks',  # Fundamental data caching tasks
        'app.tasks.group_rank_tasks',  # IBD group ranking tasks
        'app.tasks.theme_discovery_tasks',  # Theme discovery pipeline tasks
        'app.tasks.universe_tasks',  # Stock universe management tasks
        'app.tasks.telemetry_tasks',  # Weekly telemetry governance audit (asia.10.4)
        'app.tasks.runtime_bootstrap_tasks',  # Local-default first-run bootstrap orchestration
        'app.interfaces.tasks.feature_store_tasks',  # Daily feature snapshot
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone=settings.celery_timezone,
    enable_utc=True,
    task_track_started=True,  # Track when tasks start
    task_time_limit=86400,  # 24 hours max per task (for very large scans like 9650 stocks)
    task_soft_time_limit=82800,  # Soft limit at 23 hours
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
    broker_connection_retry_on_startup=True,  # Retry connecting to broker on startup
)

_logger = logging.getLogger(__name__)
_default_scan_profile = get_default_scan_profile()


def _ensure_worker_runtime_services(*, force_rebuild: bool = False):
    """Create/bind process-scoped runtime services for this Celery worker process."""
    runtime_pid = getattr(celery_app, "runtime_services_pid", None)
    current_pid = os.getpid()
    pid_changed = runtime_pid is not None and runtime_pid != current_pid
    should_force_rebuild = force_rebuild or pid_changed
    from .wiring.bootstrap import initialize_process_runtime_services

    runtime_services = initialize_process_runtime_services(force=should_force_rebuild)
    celery_app.runtime_services = runtime_services
    celery_app.runtime_services_pid = current_pid
    return runtime_services


@celery_app.on_after_configure.connect
def _log_timezone(sender, **kwargs):
    _logger.info("Celery timezone: %s (enable_utc=%s)", settings.celery_timezone, True)
    if not settings.setup_engine_enabled:
        _logger.warning("SetupEngine scanner disabled via SETUP_ENGINE_ENABLED=false")


@worker_ready.connect
def _clear_stale_data_fetch_lock(sender, **kwargs):
    """Clear stale data fetch locks when a data-fetch worker starts.

    ``datafetch-global@host`` clears all per-market lock keys because it is the
    only worker allowed to consume external-fetch queues. Legacy
    ``datafetch-<market>@host`` or ``datafetch-shared@host`` names still clear
    only their scoped lock so older local setups remain safe.

    When containers restart, Python finally blocks don't execute,
    leaving the Redis lock key with its 2-hour TTL. This blocks all
    new data_fetch tasks until the TTL expires.
    """
    try:
        _ensure_worker_runtime_services()
        hostname = getattr(sender, 'hostname', '') or ''
        if not hostname.startswith('datafetch'):
            return

        from .wiring.bootstrap import get_data_fetch_lock
        from .tasks.market_queues import SUPPORTED_MARKETS, normalize_market

        # Parse "datafetch-hk@host" -> market="HK"; "datafetch-shared@host" -> None;
        # "datafetch-global@host" clears every market lock.
        prefix = hostname.split('@', 1)[0]  # e.g. "datafetch-hk"
        parts = prefix.split('-', 1)
        worker_market: str | None = None
        clear_all_markets = False
        if len(parts) == 2:
            suffix = parts[1].upper()
            if suffix in SUPPORTED_MARKETS:
                worker_market = suffix
            elif suffix == "SHARED":
                worker_market = None
            elif suffix == "GLOBAL":
                clear_all_markets = True
            else:
                # Unknown suffix — treat as shared to be safe (legacy "datafetch" name).
                worker_market = None
        else:
            # Legacy "datafetch" worker name (no suffix) -> shared scope.
            worker_market = None

        lock = get_data_fetch_lock()
        if clear_all_markets:
            released = lock.force_release_all()
            if released:
                _logger.warning(
                    "Cleared %s stale data fetch lock key(s) on global worker startup",
                    released,
                )
            else:
                _logger.info("No stale data fetch locks found on global worker startup")
        else:
            holder = lock.get_current_holder(market=worker_market)
            if holder:
                _logger.warning(
                    "Clearing stale data fetch lock on worker startup "
                    "(market=%s, was held by %s, task_id=%s)",
                    normalize_market(worker_market).lower(),
                    holder.get('task_name', 'unknown'),
                    holder.get('task_id', 'unknown'),
                )
                lock.force_release(market=worker_market)
            else:
                _logger.info(
                    "No stale data fetch lock found on startup (market=%s)",
                    normalize_market(worker_market).lower(),
                )

        # Also clear the unsuffixed legacy key on the shared/global worker only,
        # so old pre-9.1 deployments upgrading in place don't leave it stuck.
        if worker_market is None or clear_all_markets:
            from .tasks.data_fetch_lock import LOCK_KEY as _LEGACY_LOCK_KEY
            if lock.redis.exists(_LEGACY_LOCK_KEY) > 0:
                lock.redis.delete(_LEGACY_LOCK_KEY)
                _logger.warning("Cleared legacy unsuffixed %s key on startup", _LEGACY_LOCK_KEY)
    except Exception as e:
        _logger.warning("Failed to check/clear stale lock on startup: %s", e)


@worker_process_init.connect
def _dispose_engine_after_fork(sender=None, **kwargs):
    """Dispose inherited SQLAlchemy engine after Celery prefork.

    When using prefork pool, the child process inherits the parent's
    engine and open file descriptors. Disposing forces each child to
    create fresh connections instead of reusing inherited DB state.
    """
    try:
        from .database import engine
        engine.dispose()
        _ensure_worker_runtime_services(force_rebuild=True)
        _logger.debug("Disposed inherited DB engine after fork")
    except Exception as e:
        _logger.warning("Failed to dispose engine after fork (non-fatal): %s", e)


@worker_shutting_down.connect
def _graceful_db_shutdown(sender=None, **kwargs):
    """Close DB connections on worker shutdown."""
    try:
        from .database import engine
        from .wiring.bootstrap import clear_runtime_services

        clear_runtime_services()
        if hasattr(celery_app, "runtime_services"):
            delattr(celery_app, "runtime_services")
        if hasattr(celery_app, "runtime_services_pid"):
            delattr(celery_app, "runtime_services_pid")
        engine.dispose()
        _logger.info("Worker shutdown: DB connections closed")
    except Exception as e:
        _logger.warning("Worker shutdown DB cleanup failed (non-fatal): %s", e)


# Task routing: set conservative defaults so any caller that omits an explicit
# queue still lands on a safe shared lane. Beat entries and runtime bootstrap
# override these defaults with market-specific queues.
from .tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SHARED_USER_SCANS_QUEUE,
    SUPPORTED_MARKETS,
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
)

_MARKET_SCOPED_DATA_FETCH_TASKS = (
    'app.tasks.cache_tasks.prewarm_all_active_symbols',
    'app.tasks.cache_tasks.weekly_full_refresh',
    'app.tasks.cache_tasks.warm_spy_cache',
    'app.tasks.cache_tasks.warm_top_symbols',
    'app.tasks.cache_tasks.force_refresh_stale_intraday',
    'app.tasks.cache_tasks.smart_refresh_cache',
    'app.tasks.cache_tasks.daily_cache_warmup',
    'app.tasks.cache_tasks.auto_refresh_after_close',
    'app.tasks.fundamentals_tasks.refresh_all_fundamentals',
    'app.tasks.fundamentals_tasks.refresh_all_fundamentals_hybrid',
    'app.tasks.fundamentals_tasks.refresh_symbol_fundamentals',
    'app.tasks.fundamentals_tasks.populate_initial_cache',
    'app.tasks.fundamentals_tasks.refresh_fundamentals_yfinance_only',
    'app.tasks.fundamentals_tasks.refresh_symbols_hybrid',
    'app.tasks.universe_tasks.refresh_stock_universe',
    'app.tasks.universe_tasks.refresh_official_market_universe',
    'app.tasks.universe_tasks.refresh_sp500_membership',
)

_MARKET_JOB_TASKS = (
    'app.tasks.breadth_tasks.calculate_daily_breadth',
    'app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill',
    'app.tasks.group_rank_tasks.calculate_daily_group_rankings',
    'app.tasks.group_rank_tasks.gapfill_group_rankings',
    'app.tasks.group_rank_tasks.backfill_group_rankings',
    'app.tasks.group_rank_tasks.backfill_group_rankings_1year',
    'app.interfaces.tasks.feature_store_tasks.build_daily_snapshot',
)

# Default route = shared queue. Beat entries and .apply_async() callers override
# this with the market-specific queue via `options={'queue': ...}` / `queue=`.
celery_app.conf.task_routes = {
    task_name: {'queue': SHARED_DATA_FETCH_QUEUE}
    for task_name in _MARKET_SCOPED_DATA_FETCH_TASKS
}
celery_app.conf.task_routes.update({
    task_name: {'queue': market_jobs_queue_for_market("US")}
    for task_name in _MARKET_JOB_TASKS
})

# User scans: same default-to-shared pattern; API layer sets the queue explicitly.
celery_app.conf.task_routes['app.tasks.scan_tasks.run_bulk_scan'] = {
    'queue': SHARED_USER_SCANS_QUEUE
}

# Optional: Configure result expiration
celery_app.conf.result_expires = 86400  # Results expire after 24 hours

# Celery Beat Schedule - Periodic Tasks
if settings.cache_warmup_enabled:
    _enabled_markets = list(SUPPORTED_MARKETS)
    beat_schedule: dict = {}

    for _market in _enabled_markets:
        _warm_h, _warm_m = settings.cache_warm_schedule_for(_market)
        _qname = data_fetch_queue_for_market(_market)
        _m_lower = _market.lower()

        # Daily smart cache refresh after local market close.
        beat_schedule[f'daily-smart-refresh-{_m_lower}'] = {
            'task': 'app.tasks.cache_tasks.smart_refresh_cache',
            'schedule': crontab(
                hour=_warm_h,
                minute=_warm_m,
                day_of_week='1-5'  # Monday-Friday only
            ),
            'options': {'queue': _qname},
            'kwargs': {'mode': 'full', 'market': _market},
        }

        # Daily feature snapshot (close +15m).
        _fh, _fm = _offset_schedule(_warm_h, _warm_m, 15)
        beat_schedule[f'daily-feature-snapshot-{_m_lower}'] = {
            'task': 'app.interfaces.tasks.feature_store_tasks.build_daily_snapshot',
            'schedule': crontab(hour=_fh, minute=_fm, day_of_week='1-5'),
            'options': {'queue': market_jobs_queue_for_market(_market)},
            'kwargs': {
                'screener_names': _default_scan_profile['screeners'],
                'criteria': _default_scan_profile['criteria'],
                'composite_method': _default_scan_profile['composite_method'],
                'universe_name': _default_scan_profile['universe'],
                'market': _market,
            },
        }

        # Weekly full refresh (market-local, Sunday morning).
        beat_schedule[f'weekly-full-refresh-{_m_lower}'] = {
            'task': 'app.tasks.cache_tasks.weekly_full_refresh',
            'schedule': crontab(
                hour=settings.cache_weekly_hour,
                minute=0,
                day_of_week=settings.cache_weekly_day,
            ),
            'options': {'queue': _qname},
            'kwargs': {'market': _market},
        }

        # Weekly fundamental refresh (Saturday morning) — per market.
        beat_schedule[f'weekly-fundamental-refresh-{_m_lower}'] = {
            'task': 'app.tasks.fundamentals_tasks.refresh_all_fundamentals',
            'schedule': crontab(
                hour=settings.fundamental_refresh_hour,
                minute=0,
                day_of_week=settings.fundamental_refresh_day,
            ),
            'options': {'queue': _qname},
            'kwargs': {'market': _market},
        }

        # Weekly stock universe refresh (Sunday 3 AM ET) — per market.
        beat_schedule[f'weekly-universe-refresh-{_m_lower}'] = {
            'task': (
                'app.tasks.universe_tasks.refresh_stock_universe'
                if _market == 'US'
                else 'app.tasks.universe_tasks.refresh_official_market_universe'
            ),
            'schedule': crontab(hour=3, minute=0, day_of_week=0),
            'options': {'queue': _qname},
            'kwargs': {'market': _market},
        }

    _warm_h, _warm_m = settings.cache_warm_schedule_for("US")
    _bh, _bm = _offset_schedule(_warm_h, _warm_m, 5)
    beat_schedule['daily-breadth-calculation-us'] = {
        'task': 'app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill',
        'schedule': crontab(hour=_bh, minute=_bm, day_of_week='1-5'),
        'options': {'queue': market_jobs_queue_for_market("US")},
        'kwargs': {'market': 'US'},
    }
    _gh, _gm = _offset_schedule(_warm_h, _warm_m, 10)
    beat_schedule['daily-group-ranking-calculation-us'] = {
        'task': 'app.tasks.group_rank_tasks.calculate_daily_group_rankings',
        'schedule': crontab(hour=_gh, minute=_gm, day_of_week='1-5'),
        'options': {'queue': market_jobs_queue_for_market("US")},
        'kwargs': {'market': 'US'},
    }

    # --- Market-agnostic / shared beat entries below ---
    # These tasks are not market-scoped (theme discovery, cleanup jobs, etc.)
    # so they run on the shared data_fetch queue with shared lock scope.
    #
    # NOTE: auto-refresh-after-close (4:45 PM) was removed to eliminate
    # daily schedule overlap. The per-market daily-smart-refresh entries above
    # already do a full refresh that supersedes the stale-intraday refresh.
    # The task function remains available for manual invocation via the API.
    _shared_entries = {
        # Weekly cleanup of orphaned scans (cancelled, stale running/queued).
        # Runs Sunday at 1:45 AM ET, before weekly-full-refresh at 2:00 AM.
        'weekly-orphaned-scan-cleanup': {
            'task': 'app.tasks.cache_tasks.cleanup_orphaned_scans',
            'schedule': crontab(
                hour=1,  # 1:45 AM ET
                minute=45,
                day_of_week=0  # Sunday
            ),
        },

        # Weekly refresh of per-market universe weights for RateBudgetPolicy.
        # Runs Sunday 0:30 AM ET — early in the weekly window so subsequent
        # weekly refreshes use freshly-computed budget splits.
        'weekly-rate-budget-weights-refresh': {
            'task': 'app.tasks.cache_tasks.refresh_universe_weights',
            'schedule': crontab(hour=0, minute=30, day_of_week=0),
        },

        # Monthly cleanup of old price data (keep 5 years)
        # Runs 1st of each month at 1 AM ET
        # Low priority - prevents unbounded database growth over time
        'monthly-price-data-cleanup': {
            'task': 'app.tasks.cache_tasks.cleanup_old_price_data',
            'schedule': crontab(
                hour=1,  # 1 AM ET
                minute=0,
                day_of_month=1  # 1st of month
            ),
            'kwargs': {'keep_years': 5},
        },

        # Daily candidate promotion pass for theme lifecycle state machine.
        # Runs at 4:30 AM ET after nightly data refresh jobs.
        'daily-theme-candidate-promotion': {
            'task': 'app.tasks.theme_discovery_tasks.promote_candidate_themes',
            'schedule': crontab(
                hour=4,
                minute=30,
            ),
        },

        # Daily dormancy/reactivation lifecycle pass for stale/resurgent themes.
        # Runs immediately after candidate promotion.
        'daily-theme-lifecycle-policies': {
            'task': 'app.tasks.theme_discovery_tasks.apply_lifecycle_policies',
            'schedule': crontab(
                hour=4,
                minute=45,
            ),
        },

        # Poll content sources based on their fetch_interval_minutes
        # Runs every 15 minutes to check which sources are due for refresh
        # Sources are only fetched if their last_fetched_at + fetch_interval_minutes < now
        'poll-content-sources': {
            'task': 'app.tasks.theme_discovery_tasks.poll_due_sources',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes
        },

        # Retry failed-retryable items before extracting new ones
        # Runs at :05 past the hour so retries are picked up each cycle
        'hourly-reprocess-failed-themes': {
            'task': 'app.tasks.theme_discovery_tasks.reprocess_failed_themes',
            'schedule': crontab(minute=5),  # :05 every hour
            'kwargs': {'limit': 200},
        },

        # Extract themes from pending content items via LLM
        # Runs at :10 and :40, staggered 5 min after reprocess_failed_themes
        # and 10 min after poll_due_sources to let ingestion complete
        'periodic-theme-extraction': {
            'task': 'app.tasks.theme_discovery_tasks.extract_themes',
            'schedule': crontab(minute='10,40'),  # Twice per hour
            'kwargs': {'limit': 200},
        },

        # Recalculate theme metrics (rankings, velocity, correlations)
        # Runs at :20 and :50, after extraction batches have completed
        'periodic-theme-metrics': {
            'task': 'app.tasks.theme_discovery_tasks.calculate_theme_metrics',
            'schedule': crontab(minute='20,50'),
        },

        # Weekly telemetry governance audit (bead asia.10.4).
        # Sunday 5 AM ET — after weekly-full-refresh (2 AM) and universe refresh
        # (3 AM) have emitted their final telemetry for the week. Produces a signed report at
        # data/governance/telemetry_audit/YYYY-MM-DD.{json,md,sha256}.
        'weekly-telemetry-governance-audit': {
            'task': 'app.tasks.telemetry_tasks.weekly_telemetry_audit',
            'schedule': crontab(
                hour=5,
                minute=0,
                day_of_week=0,  # Sunday
            ),
        },
    }

    # Merge shared entries into the fanned-out schedule and install.
    beat_schedule.update(_shared_entries)
    celery_app.conf.beat_schedule = beat_schedule

if __name__ == '__main__':
    celery_app.start()

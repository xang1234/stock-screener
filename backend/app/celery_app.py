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
# Required for libraries like curl_cffi (used by Sotwe scraper) that trigger
# Objective-C initialization after Celery forks worker processes
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready
from .config import settings


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


@celery_app.on_after_configure.connect
def _log_timezone(sender, **kwargs):
    _logger.info("Celery timezone: %s (enable_utc=%s)", settings.celery_timezone, True)
    if not settings.setup_engine_enabled:
        _logger.warning("SetupEngine scanner disabled via SETUP_ENGINE_ENABLED=false")


@worker_ready.connect
def _clear_stale_data_fetch_lock(sender, **kwargs):
    """Clear stale data fetch lock when the datafetch worker starts.

    When containers restart, Python finally blocks don't execute,
    leaving the Redis lock key with its 2-hour TTL. This blocks all
    new data_fetch tasks until the TTL expires.
    """
    # Only clear for the datafetch worker — the general worker must
    # NOT clear a lock that the datafetch worker may legitimately hold
    hostname = getattr(sender, 'hostname', '') or ''
    if not hostname.startswith('datafetch'):
        return

    try:
        from .tasks.data_fetch_lock import DataFetchLock
        lock = DataFetchLock.get_instance()
        holder = lock.get_current_holder()
        if holder:
            _logger.warning(
                "Clearing stale data fetch lock on worker startup "
                "(was held by %s, task_id=%s)",
                holder.get('task_name', 'unknown'),
                holder.get('task_id', 'unknown'),
            )
            lock.force_release()
        else:
            _logger.info("No stale data fetch lock found on startup")
    except Exception as e:
        _logger.warning("Failed to check/clear stale lock on startup: %s", e)


# Task routing: route background tasks to dedicated queues
# Run workers with: celery -A app.celery_app worker -Q celery,data_fetch,user_scans -c 1
celery_app.conf.task_routes = {
    # Cache tasks (yfinance)
    'app.tasks.cache_tasks.daily_cache_warmup': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.prewarm_all_active_symbols': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.weekly_full_refresh': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.warm_spy_cache': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.warm_top_symbols': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.force_refresh_stale_intraday': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.auto_refresh_after_close': {'queue': 'data_fetch'},
    'app.tasks.cache_tasks.smart_refresh_cache': {'queue': 'data_fetch'},
    # Breadth tasks (yfinance on cache miss)
    'app.tasks.breadth_tasks.calculate_daily_breadth': {'queue': 'data_fetch'},
    'app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill': {'queue': 'data_fetch'},
    # Group rank tasks (yfinance on cache miss)
    'app.tasks.group_rank_tasks.calculate_daily_group_rankings': {'queue': 'data_fetch'},
    'app.tasks.group_rank_tasks.gapfill_group_rankings': {'queue': 'data_fetch'},
    'app.tasks.group_rank_tasks.backfill_group_rankings': {'queue': 'data_fetch'},
    'app.tasks.group_rank_tasks.backfill_group_rankings_1year': {'queue': 'data_fetch'},
    # Fundamentals tasks (finviz + yfinance)
    'app.tasks.fundamentals_tasks.refresh_all_fundamentals': {'queue': 'data_fetch'},
    'app.tasks.fundamentals_tasks.refresh_all_fundamentals_hybrid': {'queue': 'data_fetch'},
    'app.tasks.fundamentals_tasks.refresh_symbol_fundamentals': {'queue': 'data_fetch'},
    'app.tasks.fundamentals_tasks.populate_initial_cache': {'queue': 'data_fetch'},
    'app.tasks.fundamentals_tasks.refresh_fundamentals_yfinance_only': {'queue': 'data_fetch'},
    'app.tasks.fundamentals_tasks.refresh_symbols_hybrid': {'queue': 'data_fetch'},
    # User-initiated scan tasks (isolated from maintenance queue)
    'app.tasks.scan_tasks.run_bulk_scan': {'queue': 'user_scans'},
    # Universe tasks (finviz)
    'app.tasks.universe_tasks.refresh_stock_universe': {'queue': 'data_fetch'},
    'app.tasks.universe_tasks.refresh_sp500_membership': {'queue': 'data_fetch'},
    # Feature store tasks (multi-screener scan)
    'app.interfaces.tasks.feature_store_tasks.build_daily_snapshot': {'queue': 'data_fetch'},
}

# Optional: Configure result expiration
celery_app.conf.result_expires = 86400  # Results expire after 24 hours

# Celery Beat Schedule - Periodic Tasks
# All data-fetching tasks route to 'data_fetch' queue for serialization
if settings.cache_warmup_enabled:
    celery_app.conf.beat_schedule = {
        # Daily smart cache refresh after market close
        # Uses smart_refresh_cache which has correct rate limiting,
        # exponential backoff on 429s, market-cap-prioritized ordering,
        # and heartbeat monitoring for stuck detection.
        'daily-smart-refresh': {
            'task': 'app.tasks.cache_tasks.smart_refresh_cache',
            'schedule': crontab(
                hour=settings.cache_warm_hour,
                minute=settings.cache_warm_minute,
                day_of_week='1-5'  # Monday-Friday only
            ),
            'options': {'queue': 'data_fetch'},
            'kwargs': {'mode': 'full'},
        },

        # Weekly full refresh (Sunday morning)
        'weekly-full-refresh': {
            'task': 'app.tasks.cache_tasks.weekly_full_refresh',
            'schedule': crontab(
                hour=settings.cache_weekly_hour,
                minute=0,
                day_of_week=settings.cache_weekly_day
            ),
            'options': {'queue': 'data_fetch'}
        },

        # Daily breadth calculation with automatic gap-fill
        # (queued after cache warmup via serialized queue)
        'daily-breadth-calculation': {
            'task': 'app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill',
            'schedule': crontab(
                hour=_offset_schedule(settings.cache_warm_hour, settings.cache_warm_minute, 5)[0],
                minute=_offset_schedule(settings.cache_warm_hour, settings.cache_warm_minute, 5)[1],
                day_of_week='1-5'  # Monday-Friday only
            ),
            'options': {'queue': 'data_fetch'}
        },

        # Weekly fundamental refresh (Saturday morning - avoids Friday warmup collision)
        'weekly-fundamental-refresh': {
            'task': 'app.tasks.fundamentals_tasks.refresh_all_fundamentals',
            'schedule': crontab(
                hour=settings.fundamental_refresh_hour,  # 8 AM ET
                minute=0,
                day_of_week=settings.fundamental_refresh_day  # Saturday = 6
            ),
            'options': {'queue': 'data_fetch'}
        },

        # Daily IBD group ranking calculation (queued after cache warmup via serialized queue)
        'daily-group-ranking-calculation': {
            'task': 'app.tasks.group_rank_tasks.calculate_daily_group_rankings',
            'schedule': crontab(
                hour=_offset_schedule(settings.cache_warm_hour, settings.cache_warm_minute, 10)[0],
                minute=_offset_schedule(settings.cache_warm_hour, settings.cache_warm_minute, 10)[1],
                day_of_week='1-5'  # Monday-Friday only
            ),
            'options': {'queue': 'data_fetch'}
        },

        # Weekly stock universe refresh (Sunday 3 AM ET - after weekly-full-refresh)
        # Adds new stocks, deactivates removed stocks, updates metadata
        'weekly-universe-refresh': {
            'task': 'app.tasks.universe_tasks.refresh_stock_universe',
            'schedule': crontab(
                hour=3,  # 3 AM ET
                minute=0,
                day_of_week=0  # Sunday
            ),
            'options': {'queue': 'data_fetch'}
        },

        # Daily feature snapshot (queued after group rankings via serialized queue)
        'daily-feature-snapshot': {
            'task': 'app.interfaces.tasks.feature_store_tasks.build_daily_snapshot',
            'schedule': crontab(
                hour=_offset_schedule(settings.cache_warm_hour, settings.cache_warm_minute, 15)[0],
                minute=_offset_schedule(settings.cache_warm_hour, settings.cache_warm_minute, 15)[1],
                day_of_week='1-5',
            ),
            'options': {'queue': 'data_fetch'},
            'kwargs': {
                'screener_names': ['minervini', 'canslim'],
                'universe_name': 'active',
            },
        },

        # NOTE: auto-refresh-after-close (4:45 PM) was removed to eliminate
        # daily schedule overlap. The 5:30 PM daily-smart-refresh already does
        # a full refresh that supersedes the stale-intraday refresh. The task
        # function remains available for manual invocation via the API.

        # Weekly cleanup of orphaned scans (cancelled, stale running/queued)
        # Runs Sunday at 2 AM ET, before weekly-full-refresh
        'weekly-orphaned-scan-cleanup': {
            'task': 'app.tasks.cache_tasks.cleanup_orphaned_scans',
            'schedule': crontab(
                hour=2,  # 2 AM ET
                minute=0,
                day_of_week=0  # Sunday
            ),
        },

        # Daily WAL checkpoint to flush WAL file and keep size bounded
        # Runs at 1:30 AM ET daily (quiet period between price cleanup and orphan cleanup)
        'daily-wal-checkpoint': {
            'task': 'app.tasks.cache_tasks.wal_checkpoint',
            'schedule': crontab(
                hour=1,
                minute=30,
            ),
        },

        # Daily database integrity check (PRAGMA quick_check)
        # Runs at 2:00 AM ET, 30 min after WAL checkpoint, to detect corruption
        # early. Logs warnings instead of blocking the health check endpoint.
        'daily-integrity-check': {
            'task': 'app.tasks.cache_tasks.daily_integrity_check',
            'schedule': crontab(
                hour=2,
                minute=0,
            ),
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

        # Weekly theme consolidation - identifies and merges duplicate themes
        # Runs Sunday at 4 AM ET (after weekly-universe-refresh)
        # Uses embedding similarity + LLM verification for high-quality merges
        'weekly-theme-consolidation': {
            'task': 'app.tasks.theme_discovery_tasks.consolidate_themes',
            'schedule': crontab(
                hour=4,  # 4 AM ET
                minute=0,
                day_of_week=0  # Sunday
            ),
            'kwargs': {'dry_run': False},
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

        # Daily semantic relationship edge inference for theme analysis UX.
        # Runs after lifecycle updates and merge suggestion generation windows.
        'daily-theme-relationship-inference': {
            'task': 'app.tasks.theme_discovery_tasks.infer_theme_relationships',
            'schedule': crontab(
                hour=5,
                minute=0,
            ),
        },

        # Poll content sources based on their fetch_interval_minutes
        # Runs every 15 minutes to check which sources are due for refresh
        # Sources are only fetched if their last_fetched_at + fetch_interval_minutes < now
        'poll-content-sources': {
            'task': 'app.tasks.theme_discovery_tasks.poll_due_sources',
            'schedule': crontab(minute='*/15'),  # Every 15 minutes
        },
    }

if __name__ == '__main__':
    celery_app.start()

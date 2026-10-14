"""
Service for managing and tracking scheduled Celery tasks.

Provides task metadata, execution history, and manual triggering capabilities.
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from celery.result import AsyncResult
from sqlalchemy import desc, inspect
from sqlalchemy.orm import Session

from ..celery_app import celery_app
from ..database import SessionLocal
from ..models.task_execution import TaskExecutionHistory
from ..config import settings
from .errors import TaskCooldownError, TaskRunBlockedError
from ..tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SUPPORTED_MARKETS,
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
)

logger = logging.getLogger(__name__)


def _coerce_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _weekly_universe_task_definitions() -> Dict[str, Dict]:
    """Build task-registry entries that match the per-market beat fanout."""
    entries: Dict[str, Dict] = {}
    for market in SUPPORTED_MARKETS:
        entries[f'weekly-universe-refresh-{market.lower()}'] = {
            'task_function': (
                'app.tasks.universe_tasks.refresh_stock_universe'
                if market == 'US'
                else 'app.tasks.universe_tasks.refresh_official_market_universe'
            ),
            'display_name': f'Weekly Universe Refresh ({market})',
            'description': (
                'Adds new US stocks, deactivates removed, updates metadata'
                if market == 'US'
                else f'Refreshes the official {market} universe snapshot from exchange sources'
            ),
            'schedule_description': f'Sunday 3:00 AM ET ({market})',
            'manual_dispatch_kwargs': {'market': market},
            'manual_dispatch_headers': {'origin': 'manual'},
            'manual_dispatch_options': {'queue': SHARED_DATA_FETCH_QUEUE},
        }
    return entries


_CRON_DAY_NAMES = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')


def _weekly_at(day: int, hour: int) -> str:
    """'Saturday 8:00 AM ET' from crontab day_of_week (0 = Sunday) and hour."""
    return f"{_CRON_DAY_NAMES[day % 7]} {hour % 12 or 12}:00 {'AM' if hour < 12 else 'PM'} ET"


def _weekly_refresh_task_definitions() -> Dict[str, Dict]:
    """One entry per market, matching the per-market Beat fanout, so each
    market's run is its own last run: a failed US run is no longer hidden by
    a later HK success (#591)."""
    entries: Dict[str, Dict] = {}
    for market in SUPPORTED_MARKETS:
        common = {
            'manual_dispatch_kwargs': {'market': market},
            'manual_dispatch_headers': {'origin': 'manual'},
            'manual_dispatch_options': {'queue': data_fetch_queue_for_market(market)},
        }
        entries[f'weekly-full-refresh-{market.lower()}'] = {
            'task_function': 'app.tasks.cache_tasks.weekly_full_refresh',
            'display_name': f'Weekly Full Refresh ({market})',
            'description': 'Clears the market\'s Redis caches and re-fetches fresh data',
            'schedule_description': (
                f'{_weekly_at(settings.cache_weekly_day, settings.cache_weekly_hour)} ({market})'
            ),
            **common,
        }
        entries[f'weekly-fundamental-refresh-{market.lower()}'] = {
            'task_function': 'app.tasks.fundamentals_tasks.refresh_all_fundamentals',
            'display_name': f'Weekly Fundamental Refresh ({market})',
            'description': 'Fetches PE, EPS, revenue, margins for the market\'s stocks',
            'schedule_description': (
                f'{_weekly_at(settings.fundamental_refresh_day, settings.fundamental_refresh_hour)} ({market})'
            ),
            **common,
        }
    return entries


def _daily_market_pipeline_task_definitions() -> Dict[str, Dict]:
    entries: Dict[str, Dict] = {}
    for market in SUPPORTED_MARKETS:
        warm_hour, warm_minute = settings.cache_warm_schedule_for(market)
        entries[f"daily-market-pipeline-{market.lower()}"] = {
            "task_function": "app.tasks.daily_market_pipeline_tasks.queue_daily_market_pipeline",
            "display_name": f"Daily Market Pipeline ({market})",
            "description": "Runs price refresh, breadth, group rankings, and market scan in order",
            "schedule_description": f"{warm_hour}:{warm_minute:02d} ET, Mon-Fri ({market})",
            "manual_dispatch_kwargs": {"market": market},
            "manual_dispatch_headers": {"origin": "manual"},
            "manual_dispatch_options": {"queue": market_jobs_queue_for_market(market)},
            # Pre-flight: why a Run would only skip, checked before queueing.
            "run_blockers": "app.tasks.daily_market_pipeline_tasks.pipeline_run_blockers",
        }
    return entries


# Task definitions with metadata
SCHEDULED_TASKS = {
    'social-signal-refresh': {
        'task_function': 'app.interfaces.tasks.social_signal_tasks.refresh_social_signals',
        'display_name': 'Refresh Social Signals',
        'description': 'Collects configured X lists and refreshes the Social Signal Queue',
        'schedule_description': 'Every six hours at minute 17',
        'manual_dispatch_kwargs': {'origin': 'manual'},
        'manual_dispatch_options': {'queue': 'social_ingestion'},
        'db_runtime_social': True,
        'admin_only': True,
    },
    # ===== SUNDAY (Off-Market Maintenance) =====
    'weekly-orphaned-scan-cleanup': {
        'task_function': 'app.tasks.cache_tasks.cleanup_orphaned_scans',
        'display_name': 'Weekly Scan Cleanup',
        'description': 'Deletes cancelled and stale scans',
        'schedule_description': 'Sunday 2:00 AM ET',
    },
    # ===== WEEKDAYS (After Market Close) =====
    **_daily_market_pipeline_task_definitions(),
    'daily-us-options-analytics': {
        'task_function': (
            'app.interfaces.tasks.options_analytics_tasks.refresh_options_analytics'
        ),
        'display_name': 'Daily US Options Analytics',
        'description': 'Refreshes the bounded Options Command Center cohort',
        'schedule_description': 'After the published US daily equity snapshot',
        'manual_dispatch_kwargs': {'market': 'US'},
        'manual_dispatch_headers': {'origin': 'manual'},
        'manual_dispatch_options': {'queue': 'data_fetch_us'},
        'enabled_setting': 'options_analytics_enabled',
    },

    # ===== WEEKLY, PER MARKET (cache: Sunday; fundamentals: Saturday) =====
    **_weekly_refresh_task_definitions(),
    # Manual all-markets refresh. It also runs the orphaned-cache and telemetry
    # cleanups (weekly_full_refresh does both only when market is None), which
    # weekly-global-maintenance now runs on schedule (#615). Per-market runs
    # match their own entries first in _registry_entry_for, so only
    # market=None runs land here.
    'weekly-full-refresh': {
        'task_function': 'app.tasks.cache_tasks.weekly_full_refresh',
        'display_name': 'Weekly Full Refresh (all markets)',
        'description': (
            'Refreshes every market and also removes orphaned cache keys and '
            'expired telemetry events, which per-market runs skip'
        ),
        'schedule_description': 'Manual only',
    },

    'weekly-global-maintenance': {
        'task_function': 'app.tasks.cache_tasks.weekly_global_maintenance',
        'display_name': 'Weekly Cache & Telemetry Cleanup',
        'description': (
            'Removes cached prices of delisted symbols in every market '
            'and telemetry events past retention'
        ),
        'schedule_description': 'Sunday 6:00 AM ET',
        # A shared Beat entry: installed even when cache warming is off.
        'always_enabled': True,
    },

    # ===== MONTHLY =====
    'monthly-price-data-cleanup': {
        'task_function': 'app.tasks.cache_tasks.cleanup_old_price_data',
        'display_name': 'Monthly Price Data Cleanup',
        'description': 'Deletes price data older than 5 years',
        'schedule_description': '1st of month, 1:00 AM ET',
    },
    **_weekly_universe_task_definitions(),
}


_REGISTRY_FUNCTIONS = frozenset(info['task_function'] for info in SCHEDULED_TASKS.values())
# Finished run outcomes and the Celery state each one ends in. skipped and
# dispatched runs succeeded as Celery tasks; the outcome says what they did.
_CELERY_STATE_FOR = {
    'completed': 'SUCCESS',
    'skipped': 'SUCCESS',
    'dispatched': 'SUCCESS',
    'failed': 'FAILURE',
    'cancelled': 'REVOKED',
}
_STATUS_FOR_CELERY_STATE = {'SUCCESS': 'completed', 'FAILURE': 'failed', 'REVOKED': 'cancelled'}
_FINISHED_STATUSES = frozenset(_CELERY_STATE_FOR)
# Celery kills a task at the global task_time_limit and stores FAILURE for
# it; the margin leaves the reconcile job time to record that. A run still
# open after this was never started, or its worker died without a stored
# outcome (or ran under --pool=solo, where time limits are not enforced), so
# its result is unknown.
RUN_LOST_AFTER = timedelta(seconds=celery_app.conf.task_time_limit) + timedelta(hours=1)
# Callers that tag the origin kwarg outside these dispatch the task ad hoc
# (e.g. per-attachment social refreshes), not as the scheduled job.
_TRACKED_ORIGINS = frozenset({'manual', 'scheduled'})


_BROKER_TIMEOUT_SECONDS = 2
_BROKER_CHECK_TTL_SECONDS = 60
_broker_checks: Dict[str, tuple] = {}
_BROKER_SCAN_BATCH = 200
_BROKER_SCAN_BUDGET_SECONDS = 2.0


class _BrokerScanBudgetExceeded(Exception):
    pass


def _broker_messages(broker, queue_names, deadline):
    """Every message in the queue lists, then the ``unacked`` hash, read in
    batches under one overall deadline.

    Paging from the head is safe while workers consume: kombu pushes at the
    head and pops from the tail, so a page can repeat a message but not skip one.
    """
    def _check_deadline():
        if time.monotonic() > deadline:
            raise _BrokerScanBudgetExceeded()

    for name in queue_names:
        start = 0
        while True:
            _check_deadline()
            batch = broker.lrange(name, start, start + _BROKER_SCAN_BATCH - 1)
            yield from batch
            if len(batch) < _BROKER_SCAN_BATCH:
                break
            start += _BROKER_SCAN_BATCH
    cursor = 0
    while True:
        _check_deadline()
        cursor, entries = broker.hscan('unacked', cursor, count=_BROKER_SCAN_BATCH)
        yield from entries.values()
        if not cursor:
            break


def _message_task_id(raw: bytes) -> Optional[str]:
    """The task id in a queued message, or in an ``unacked`` entry (a JSON
    ``[message, exchange, routing_key]``). Not parent_id/root_id: a child's
    message must not count as its parent's."""
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    return (payload.get('headers') or {}).get('id')


def still_in_broker(
    task_ids: List[Optional[str]], *, cache: bool = True
) -> Dict[str, Optional[bool]]:
    """For each never-started run, whether its message is still in the broker:
    in a queue list, or prefetched by a worker that stopped (kombu's
    ``unacked`` hash; a live worker restores it to its queue). None if the
    broker is unreadable or the scan runs out of budget.

    All ids share one client, one pass and one deadline. Answers are cached
    briefly: the Schedules list is polled, and a lost run can stay a job's
    last run until its next scheduled run. A missing id maps to False: it can
    never be found or cancelled, so blocking Run would be permanent.
    ``cache=False`` reads the broker fresh and does not cache the answers.
    """
    now = datetime.now(timezone.utc).timestamp()
    answers: Dict[str, Optional[bool]] = {}
    pending = set()
    for task_id in task_ids:
        if not task_id:
            continue
        cached = _broker_checks.get(task_id) if cache else None
        if cached and now - cached[0] < _BROKER_CHECK_TTL_SECONDS:
            answers[task_id] = cached[1]
        else:
            pending.add(task_id)
    if not pending:
        return answers
    # Lazy: operations_job_service imports this module.
    import redis

    from .operations_job_service import OperationsJobService

    found: set = set()
    unknown = False
    broker = None
    try:
        broker = redis.from_url(
            settings.celery_broker_url,
            socket_timeout=_BROKER_TIMEOUT_SECONDS,
            socket_connect_timeout=_BROKER_TIMEOUT_SECONDS,
        )
        queue_names = [*OperationsJobService.broker_queue_names(), 'social_ingestion']
        deadline = time.monotonic() + _BROKER_SCAN_BUDGET_SECONDS
        # Two passes under the one deadline: a message kombu moves between a
        # queue and ``unacked`` mid-pass (a worker taking it, or a restore)
        # can slip past both reads of one pass, but not two passes in a row.
        for _ in range(2):
            for raw in _broker_messages(broker, queue_names, deadline):
                message_id = _message_task_id(raw)
                if message_id in pending:
                    found.add(message_id)
                    if found == pending:
                        break
            if found == pending:
                break
    except _BrokerScanBudgetExceeded:
        # A backlog too big to read in budget: ids not yet found are unknown,
        # so Run stays blocked (the message is likely still waiting anyway).
        logger.info("Broker scan for %d lost runs exceeded its budget", len(pending))
        unknown = True
    except Exception:
        # Any broker failure means "unknown"; Run stays blocked (it would
        # fail to publish to the same broker anyway).
        logger.warning("Could not read the broker for %d lost runs", len(pending), exc_info=True)
        unknown = True
    finally:
        if broker is not None:
            broker.close()
    for task_id in pending:
        answer = True if task_id in found else (None if unknown else False)
        if cache:
            _broker_checks[task_id] = (now, answer)
        answers[task_id] = answer
    return answers


def _run_blockers(entries: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """{entry name: why a manual Run would only skip, or None}.

    Entries with a ``run_blockers`` hook (a function taking a list of markets)
    are checked in one call per hook, so the polled list does one bounded
    round of reads. Never raises; a failed check must not stop a Run (the
    task re-checks).
    """
    by_hook: Dict[str, List[tuple]] = {}
    for name, info in entries.items():
        if info.get('run_blockers'):
            market = str((info.get('manual_dispatch_kwargs') or {}).get('market', '')).upper()
            by_hook.setdefault(info['run_blockers'], []).append((name, market))
    answers: Dict[str, Optional[Dict[str, Any]]] = {}
    for path, items in by_hook.items():
        try:
            module_path, func_name = path.rsplit('.', 1)
            check = getattr(__import__(module_path, fromlist=[func_name]), func_name)
            by_market = check([market for _, market in items])
        except Exception:
            logger.warning("Pre-flight check %s failed", path, exc_info=True)
            by_market = {}
        for name, market in items:
            answers[name] = by_market.get(market)
    return answers


def _result_summary(result: Any, task_id: str) -> Optional[Dict[str, Any]]:
    """A task's dict result as JSON, or None (PostgreSQL JSON rejects NaN).

    Dropping the summary must never also drop the run's outcome.
    """
    if not isinstance(result, dict):
        return None
    try:
        return json.loads(json.dumps(result, default=str, allow_nan=False))
    except ValueError:
        logger.warning("Dropping non-JSON result summary for run %s", task_id)
        return None


def _reported_error(result: Any) -> Optional[str]:
    """The error a task returned instead of raising, if any.

    Several registry tasks (weekly_full_refresh, cleanup_old_price_data,
    refresh_all_fundamentals, refresh_stock_universe) catch their exception
    and return ``{'error': ...}`` or ``{'status': 'error'}``, so Celery
    reports SUCCESS. Other statuses (skipped, ok, the Options task's
    failed_quality gate) are finished runs, not failures.
    """
    if not isinstance(result, dict):
        return None
    if result.get('error'):
        return str(result['error'])
    if result.get('status') in ('error', 'failed'):
        return f"Task reported status '{result['status']}'"
    return None


def _outcome(status: str, result: Any, error: Optional[str] = None) -> tuple[str, Optional[str]]:
    """What a successful task actually did, from the result it returned.

    Celery reports SUCCESS whenever the function returns, so a run that
    caught its own error, did nothing, or only queued other tasks would all
    read as completed. ``queued`` is queue_daily_market_pipeline handing its
    stage chain off; the stages are still ahead.
    """
    if status != 'completed' or not isinstance(result, dict):
        return status, error
    if reported := _reported_error(result):
        return 'failed', reported
    # refresh_social_signals returns asdict(SocialRunResult): processing_status
    # instead of status. 'deferred' collected and resumes later, so it stays
    # completed.
    processing = result.get('processing_status')
    codes = result.get('reason_codes') or ()
    if processing == 'failed':
        return 'failed', f"Social Signals analysis failed ({', '.join(map(str, codes)) or 'no reason given'})"
    if processing == 'blocked' and 'source_participation_failed' in codes:
        return 'failed', 'Social source collection failed (source_participation_failed)'
    # 'skipped' is True only for a skipped run; refresh_all_fundamentals
    # reports a count of skipped symbols under the same key.
    if result.get('status') == 'skipped' or result.get('skipped') is True or processing in ('skipped', 'blocked'):
        return 'skipped', None
    if result.get('status') == 'queued':
        return 'dispatched', None
    return status, error


def _skip_detail(summary: Any) -> Optional[str]:
    """Why a run skipped, from its stored result.

    reason (pipeline, universe), reason_codes (options, social), or message
    (serialized_data_fetch_task's 'Data fetch already in progress').
    """
    if not isinstance(summary, dict):
        return None
    if summary.get('reason'):
        return str(summary['reason'])
    codes = summary.get('reason_codes')
    if codes:
        return ', '.join(str(code) for code in codes)
    return str(summary['message']) if summary.get('message') else None


def _registry_entry_for(task_function: str, kwargs: Dict[str, Any]) -> Optional[str]:
    """Map a Celery task to its registry entry, preferring a per-market entry."""
    market = str(kwargs.get('market') or '').upper()
    fallback = None
    for name, info in SCHEDULED_TASKS.items():
        if info['task_function'] != task_function:
            continue
        entry_market = (info.get('manual_dispatch_kwargs') or {}).get('market')
        if entry_market is None:
            fallback = fallback or name
        elif entry_market == market:
            return name
    return fallback


def record_task_run(
    task_id: Optional[str],
    task_function: Optional[str],
    kwargs: Optional[Dict[str, Any]],
    status: str,
    *,
    result: Any = None,
    error: Optional[str] = None,
) -> None:
    """Record a registry task's lifecycle from the worker (best-effort).

    Called from Celery task signals so history no longer depends on a browser
    polling the status endpoint. Manual triggers insert their row before
    dispatch; scheduled runs get their row here. Failures only log a warning:
    run bookkeeping must never fail the task it describes.
    """
    kwargs = kwargs or {}
    origin = kwargs.get('origin')
    if not task_id or task_function not in _REGISTRY_FUNCTIONS:
        return
    if isinstance(origin, str) and origin not in _TRACKED_ORIGINS:
        return
    task_name = _registry_entry_for(task_function, kwargs)
    if task_name is None:
        return
    status, error = _outcome(status, result, error)
    try:
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            row = db.query(TaskExecutionHistory).filter_by(task_id=task_id).first()
            if row is None:
                # Manual triggers pre-insert their row, so this is Beat or
                # another automatic dispatch; only a tagged origin says which.
                row = TaskExecutionHistory(
                    task_name=task_name,
                    task_function=task_function,
                    task_id=task_id,
                    started_at=now,
                    triggered_by=origin if isinstance(origin, str) else 'auto',
                )
                db.add(row)
            if status == 'running' and row.status == 'queued':
                # Start the clock when a worker first picks up a manual run; a
                # retry's prerun (row already running) keeps the first start.
                row.started_at = now
            row.status = status
            if status in _FINISHED_STATUSES:
                row.completed_at = now
                started_at = _coerce_utc_datetime(row.started_at)
                if started_at:
                    row.duration_seconds = (now - started_at).total_seconds()
                row.result_summary = _result_summary(result, task_id)
                if error:
                    row.error_message = error
            db.commit()
    except Exception:
        logger.warning("Failed to record %s run %s as %s", task_name, task_id, status, exc_info=True)


class TaskRegistryService:
    """
    Service for managing scheduled Celery tasks.

    Provides:
    - Task metadata and schedule information
    - Last run status from execution history
    - Manual task triggering
    - Task status polling via AsyncResult
    """

    def __init__(self, *, social_gate=None):
        """Initialize the service with task imports."""
        self._task_imports = {}
        self._social_gate = social_gate

    def _gate(self):
        if self._social_gate is None:
            from app.services.redis_pool import get_redis_client
            from app.services.social_signal_runtime_gate import RedisSocialSignalGate
            self._social_gate = RedisSocialSignalGate(get_redis_client())
        return self._social_gate

    def _get_task(self, task_name: str):
        """Lazy-load and cache task imports."""
        if task_name not in self._task_imports:
            task_info = SCHEDULED_TASKS.get(task_name)
            if not task_info:
                raise ValueError(f"Unknown task: {task_name}")

            # Dynamic import of the task
            task_function = task_info['task_function']
            module_path, func_name = task_function.rsplit('.', 1)
            module = __import__(module_path, fromlist=[func_name])
            self._task_imports[task_name] = getattr(module, func_name)

        return self._task_imports[task_name]

    def get_all_scheduled_tasks(self, db: Session, *, include_run_blockers: bool = False) -> List[Dict]:
        """
        Get all scheduled tasks with their metadata and last run info.

        Args:
            db: Database session

        Returns:
            List of task dictionaries with schedule and last run details
        """
        result = []
        # Opt-in: live checks (calendar, Redis) only the Run UI needs, in one
        # bounded batch for all entries.
        blockers = _run_blockers(SCHEDULED_TASKS) if include_run_blockers else {}

        for task_name, task_info in SCHEDULED_TASKS.items():
            history_task_names = task_info.get('history_task_names', [task_name])
            # Get last execution from history
            last_run = db.query(TaskExecutionHistory).filter(
                TaskExecutionHistory.task_name.in_(history_task_names)
            ).order_by(desc(TaskExecutionHistory.started_at)).first()

            if task_info.get('db_runtime_social'):
                from app.infra.db.models.social_signals import SocialSourceRegistry
                bind = db.get_bind()
                if inspect(bind).has_table(SocialSourceRegistry.__tablename__):
                    runtime = db.get(SocialSourceRegistry, 1)
                    is_enabled = bool(
                        runtime
                        and runtime.mode in {'validation', 'live'}
                        and runtime.provider != 'disabled'
                    )
                else:
                    # Older/partial databases cannot run Social Signals, but
                    # must still expose the rest of the task registry.
                    is_enabled = False
            elif task_info.get('always_enabled'):
                # Scheduled by Beat regardless of cache warming.
                is_enabled = True
            else:
                is_enabled = bool(getattr(
                    settings, task_info.get('enabled_setting', 'cache_warmup_enabled')
                ))
            task_data = {
                'name': task_name,
                'display_name': task_info['display_name'],
                'task_function': task_info['task_function'],
                'description': task_info['description'],
                'schedule_description': task_info['schedule_description'],
                'is_enabled': is_enabled,
                'market': task_info.get('manual_dispatch_kwargs', {}).get('market'),
                'last_run': None,
            }
            if include_run_blockers:
                task_data['run_blocker'] = blockers.get(task_name)

            if last_run:
                status = last_run.status
                started_at = _coerce_utc_datetime(last_run.started_at)
                lost_from = still_queued = None
                if (
                    status not in _FINISHED_STATUSES
                    and started_at
                    and datetime.now(timezone.utc) - started_at > RUN_LOST_AFTER
                ):
                    # A queued row was never picked up (a worker flips it to
                    # running at prerun). Its message may still be waiting,
                    # and a second Run would then run the job twice.
                    lost_from = 'queued' if status == 'queued' else 'running'
                    status = 'lost'
                task_data['last_run'] = {
                    'id': last_run.id,
                    'task_id': last_run.task_id,
                    'status': status,
                    'started_at': last_run.started_at.isoformat() if last_run.started_at else None,
                    'completed_at': last_run.completed_at.isoformat() if last_run.completed_at else None,
                    'duration_seconds': last_run.duration_seconds,
                    'triggered_by': last_run.triggered_by,
                    'error_message': last_run.error_message,
                    'detail': _skip_detail(last_run.result_summary) if status == 'skipped' else None,
                    'lost_from': lost_from,
                    'still_queued': still_queued,
                }

            result.append(task_data)

        # One broker pass for every run lost from queued, under one budget,
        # so several such runs (or a broker outage) cost the request once.
        lost_queued = [
            task['last_run'] for task in result
            if task['last_run'] and task['last_run']['lost_from'] == 'queued'
        ]
        if lost_queued:
            answers = still_in_broker([run['task_id'] for run in lost_queued])
            absent = [run['task_id'] for run in lost_queued if answers.get(run['task_id'], False) is False]
            # A worker that took the message after the history read has acked
            # it (so it is gone) and flipped the row to running: not safe to
            # rerun. The next poll shows it Running; until then, unknown.
            picked_up = {
                task_id for (task_id,) in db.query(TaskExecutionHistory.task_id).filter(
                    TaskExecutionHistory.task_id.in_([t for t in absent if t]),
                    TaskExecutionHistory.status != 'queued',
                )
            } if any(absent) else set()
            for run in lost_queued:
                answer = answers.get(run['task_id'], False)
                run['still_queued'] = None if run['task_id'] in picked_up else answer

        return result

    def reconcile_open_runs(self, db: Session) -> int:
        """Close open runs from Celery's stored outcome when no signal did.

        A hard time limit or a lost worker kills the child before
        task_postrun, and revokes are announced in the worker's parent
        process; Celery still stores FAILURE/REVOKED in the result backend.
        Runs in a worker (see task_history_tasks) because the result backend
        retries for minutes while Redis is down. Returns the number closed.
        """
        cutoff = datetime.now(timezone.utc) - RUN_LOST_AFTER
        open_runs = db.query(TaskExecutionHistory).filter(
            TaskExecutionHistory.status.in_(('queued', 'running')),
            TaskExecutionHistory.task_id.isnot(None),
            TaskExecutionHistory.started_at >= cutoff,
        ).all()
        closed = 0
        for run in open_runs:
            task_id = run.task_id
            try:
                outcome = AsyncResult(task_id)
                status = _STATUS_FOR_CELERY_STATE.get(outcome.state)
                if status is None:
                    continue
                self._update_execution_completed(
                    db,
                    task_id,
                    status,
                    result=outcome.result if status == 'completed' else None,
                    error=str(outcome.result) if status == 'failed' else None,
                )
                closed += 1
            except Exception:
                db.rollback()
                logger.warning("Could not read Celery state for run %s", task_id, exc_info=True)
        return closed

    def trigger_task(self, task_name: str, db: Session) -> Dict:
        """
        Manually trigger a scheduled task.

        Args:
            task_name: Name of the task to trigger
            db: Database session

        Returns:
            Dictionary with task_id and status
        """
        if task_name not in SCHEDULED_TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        task_info = SCHEDULED_TASKS[task_name]
        blocker = _run_blockers({task_name: task_info}).get(task_name)
        if blocker:
            # Refuse before queueing: the run would only read Skipped.
            raise TaskRunBlockedError(blocker)
        logger.info(f"Manually triggering task: {task_name}")

        cooldown_owner = None
        if task_info.get('db_runtime_social'):
            cooldown_owner = f"manual:{datetime.now(timezone.utc).isoformat()}"
            accepted, retry_after = self._gate().acquire_manual_cooldown(
                cooldown_owner,
                settings.social_manual_refresh_cooldown_seconds,
            )
            if not accepted:
                raise TaskCooldownError(retry_after)

        # Commit the run row before publishing so the worker's start/finish
        # signals always find it, however quickly the task is picked up.
        task_id = str(uuid4())
        committed = False
        try:
            task_func = self._get_task(task_name)
            execution = TaskExecutionHistory(
                task_name=task_name,
                task_function=task_info['task_function'],
                task_id=task_id,
                status='queued',
                started_at=datetime.now(timezone.utc),
                triggered_by='manual',
            )
            db.add(execution)
            db.commit()
            committed = True
            task_func.apply_async(
                kwargs=task_info.get('manual_dispatch_kwargs') or {},
                headers=task_info.get('manual_dispatch_headers'),
                task_id=task_id,
                **(task_info.get('manual_dispatch_options') or {}),
            )
        except Exception as exc:
            if cooldown_owner is not None:
                self._gate().release_manual_cooldown(cooldown_owner)
            if committed:
                # Only a still-queued row: if the publish landed after all, a
                # worker may already have advanced it.
                try:
                    db.query(TaskExecutionHistory).filter_by(
                        task_id=task_id, status='queued'
                    ).update({
                        'status': 'failed',
                        'completed_at': datetime.now(timezone.utc),
                        'error_message': f"Dispatch failed: {exc}",
                    }, synchronize_session=False)
                    db.commit()
                except Exception:
                    db.rollback()
                    logger.warning("Could not mark manual run %s failed", task_id, exc_info=True)
            raise

        logger.info(f"Task {task_name} queued with ID: {task_id}")

        return {
            'task_id': task_id,
            'task_name': task_name,
            'status': 'queued',
            'execution_id': execution.id,
            'message': f'Task {task_info["display_name"]} queued for execution',
        }

    def get_task_status(self, task_name: str, task_id: str, db: Session) -> Dict:
        """
        Get the current status of a task execution.

        Args:
            task_name: Name of the task
            task_id: Celery task ID
            db: Database session

        Returns:
            Dictionary with current status and progress
        """
        # The worker's task signals record finished runs; the ledger owns
        # their status. This endpoint reports task execution: skipped and
        # dispatched runs completed as tasks, and callers (the Options page)
        # read result.status for what they did.
        run = db.query(TaskExecutionHistory).filter_by(task_id=task_id).first()
        if run is not None and run.status in _FINISHED_STATUSES:
            celery_state = _CELERY_STATE_FOR[run.status]
            response = {
                'task_id': task_id,
                'task_name': task_name,
                'status': _STATUS_FOR_CELERY_STATE[celery_state],
                'celery_state': celery_state,
            }
            if celery_state == 'SUCCESS':
                response['result'] = run.result_summary
                if run.result_summary is None:
                    response['result'] = self._celery_success_result(task_id)
            elif run.error_message:
                response['error'] = run.error_message
            return response

        # Get status from Celery
        result = AsyncResult(task_id)
        celery_state = result.state

        # Map Celery states to our states
        state_mapping = {
            'PENDING': 'queued',
            'STARTED': 'running',
            'PROGRESS': 'running',
            'SUCCESS': 'completed',
            'FAILURE': 'failed',
            'REVOKED': 'cancelled',
        }

        status = state_mapping.get(celery_state, celery_state.lower())
        if status == 'queued' and run is not None and run.status == 'running':
            # task_prerun fires before Celery stores STARTED; never step back.
            status = 'running'

        response = {
            'task_id': task_id,
            'task_name': task_name,
            'status': status,
            'celery_state': celery_state,
        }

        # Add progress info if available
        if celery_state == 'PROGRESS' and result.info:
            response['progress'] = result.info.get('percent', 0)
            response['message'] = result.info.get('message', '')
            response['current'] = result.info.get('current', 0)
            response['total'] = result.info.get('total', 0)
        elif celery_state == 'SUCCESS':
            response['result'] = result.result
            # Same classifier as the recorded outcome; skipped and dispatched
            # runs still report completed (execution status, Options contract).
            outcome, reported = _outcome('completed', result.result)
            if outcome == 'failed':
                status = response['status'] = 'failed'
                response['error'] = reported
        elif celery_state == 'FAILURE':
            response['error'] = str(result.result) if result.result else 'Unknown error'

        # Fallback writer for runs whose signal write was lost.
        if status in _FINISHED_STATUSES:
            self._update_execution_completed(
                db,
                task_id,
                status,
                result=response.get('result'),
                error=response.get('error'),
            )

        return response

    @staticmethod
    def _celery_success_result(task_id: str) -> Any:
        """Celery's stored result for a run the ledger finished without a
        summary (non-JSON result). Enrichment only: past result_expires or
        with the backend down there is nothing to add, and the ledger's
        completed status stands either way."""
        try:
            outcome = AsyncResult(task_id)
            return outcome.result if outcome.state == 'SUCCESS' else None
        except Exception:
            logger.warning("Could not read Celery result for run %s", task_id, exc_info=True)
            return None

    def _update_execution_completed(
        self,
        db: Session,
        task_id: str,
        status: str,
        result: Dict = None,
        error: str = None
    ):
        """Update execution record when task completes."""
        status, error = _outcome(status, result, error)
        execution = db.query(TaskExecutionHistory).filter(
            TaskExecutionHistory.task_id == task_id
        ).first()

        if execution:
            execution.status = status
            execution.completed_at = datetime.now(timezone.utc)
            if execution.started_at:
                started_at = _coerce_utc_datetime(execution.started_at)
                completed_at = _coerce_utc_datetime(execution.completed_at)
                if started_at and completed_at:
                    execution.duration_seconds = (completed_at - started_at).total_seconds()
            if result:
                execution.result_summary = _result_summary(result, task_id)
            if error:
                execution.error_message = error
            db.commit()

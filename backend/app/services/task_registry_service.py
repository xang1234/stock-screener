"""
Service for managing and tracking scheduled Celery tasks.

Provides task metadata, execution history, and manual triggering capabilities.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from celery.result import AsyncResult
from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..models.task_execution import TaskExecutionHistory
from ..config import settings
from ..tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    SUPPORTED_MARKETS,
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
        }
    return entries


# Task definitions with metadata
SCHEDULED_TASKS = {
    # ===== SUNDAY (Off-Market Maintenance) =====
    'weekly-orphaned-scan-cleanup': {
        'task_function': 'app.tasks.cache_tasks.cleanup_orphaned_scans',
        'display_name': 'Weekly Scan Cleanup',
        'description': 'Deletes cancelled and stale scans',
        'schedule_description': 'Sunday 2:00 AM ET',
    },
    'weekly-full-refresh': {
        'task_function': 'app.tasks.cache_tasks.weekly_full_refresh',
        'display_name': 'Weekly Full Refresh',
        'description': 'Clears all Redis caches and re-fetches fresh data',
        'schedule_description': f'Sunday {settings.cache_weekly_hour}:00 AM ET',
    },
    # ===== WEEKDAYS (After Market Close) =====
    **_daily_market_pipeline_task_definitions(),

    # ===== FRIDAY =====
    'weekly-fundamental-refresh': {
        'task_function': 'app.tasks.fundamentals_tasks.refresh_all_fundamentals',
        'display_name': 'Weekly Fundamental Refresh',
        'description': 'Fetches PE, EPS, revenue, margins for all stocks',
        'schedule_description': f'Friday {settings.fundamental_refresh_hour}:00 PM ET',
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


class TaskRegistryService:
    """
    Service for managing scheduled Celery tasks.

    Provides:
    - Task metadata and schedule information
    - Last run status from execution history
    - Manual task triggering
    - Task status polling via AsyncResult
    """

    def __init__(self):
        """Initialize the service with task imports."""
        self._task_imports = {}

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

    def get_all_scheduled_tasks(self, db: Session) -> List[Dict]:
        """
        Get all scheduled tasks with their metadata and last run info.

        Args:
            db: Database session

        Returns:
            List of task dictionaries with schedule and last run details
        """
        result = []

        for task_name, task_info in SCHEDULED_TASKS.items():
            history_task_names = task_info.get('history_task_names', [task_name])
            # Get last execution from history
            last_run = db.query(TaskExecutionHistory).filter(
                TaskExecutionHistory.task_name.in_(history_task_names)
            ).order_by(desc(TaskExecutionHistory.started_at)).first()

            task_data = {
                'name': task_name,
                'display_name': task_info['display_name'],
                'task_function': task_info['task_function'],
                'description': task_info['description'],
                'schedule_description': task_info['schedule_description'],
                'is_enabled': settings.cache_warmup_enabled,
                'last_run': None,
            }

            if last_run:
                task_data['last_run'] = {
                    'id': last_run.id,
                    'task_id': last_run.task_id,
                    'status': last_run.status,
                    'started_at': last_run.started_at.isoformat() if last_run.started_at else None,
                    'completed_at': last_run.completed_at.isoformat() if last_run.completed_at else None,
                    'duration_seconds': last_run.duration_seconds,
                    'triggered_by': last_run.triggered_by,
                    'error_message': last_run.error_message,
                }

            result.append(task_data)

        return result

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
        logger.info(f"Manually triggering task: {task_name}")

        # Get the task function and dispatch it
        task_func = self._get_task(task_name)
        manual_dispatch_kwargs = task_info.get('manual_dispatch_kwargs')
        manual_dispatch_headers = task_info.get('manual_dispatch_headers')
        manual_dispatch_options = task_info.get('manual_dispatch_options') or {}
        if manual_dispatch_kwargs or manual_dispatch_headers or manual_dispatch_options:
            celery_task = task_func.apply_async(
                kwargs=manual_dispatch_kwargs or {},
                headers=manual_dispatch_headers,
                **manual_dispatch_options,
            )
        else:
            celery_task = task_func.delay()

        # Record the execution in history
        execution = TaskExecutionHistory(
            task_name=task_name,
            task_function=task_info['task_function'],
            task_id=celery_task.id,
            status='queued',
            started_at=datetime.now(timezone.utc),
            triggered_by='manual',
        )
        db.add(execution)
        db.commit()

        logger.info(f"Task {task_name} queued with ID: {celery_task.id}")

        return {
            'task_id': celery_task.id,
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
            'REVOKED': 'failed',
        }

        status = state_mapping.get(celery_state, celery_state.lower())

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
            # Update database record
            self._update_execution_completed(db, task_id, 'completed', result.result)
        elif celery_state == 'FAILURE':
            response['error'] = str(result.result) if result.result else 'Unknown error'
            # Update database record
            self._update_execution_completed(db, task_id, 'failed', error=str(result.result))

        return response

    def _update_execution_completed(
        self,
        db: Session,
        task_id: str,
        status: str,
        result: Dict = None,
        error: str = None
    ):
        """Update execution record when task completes."""
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
                execution.result_summary = result
            if error:
                execution.error_message = error
            db.commit()

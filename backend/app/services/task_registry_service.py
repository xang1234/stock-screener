"""
Service for managing and tracking scheduled Celery tasks.

Provides task metadata, execution history, and manual triggering capabilities.
"""
import logging
from typing import Optional, Dict, List
from datetime import datetime
from celery.result import AsyncResult
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models.task_execution import TaskExecutionHistory
from ..config import settings

logger = logging.getLogger(__name__)


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
    'weekly-universe-refresh': {
        'task_function': 'app.tasks.universe_tasks.refresh_stock_universe',
        'display_name': 'Weekly Universe Refresh',
        'description': 'Adds new stocks, deactivates removed, updates metadata',
        'schedule_description': 'Sunday 3:00 AM ET',
    },
    'weekly-theme-consolidation': {
        'task_function': 'app.tasks.theme_discovery_tasks.consolidate_themes',
        'display_name': 'Weekly Theme Consolidation',
        'description': 'Merges duplicate themes via embeddings + LLM',
        'schedule_description': 'Sunday 4:00 AM ET',
    },
    'daily-theme-stale-embedding-recompute': {
        'task_function': 'app.tasks.theme_discovery_tasks.recompute_stale_theme_embeddings',
        'display_name': 'Daily Stale Embedding Refresh',
        'description': 'Incrementally recomputes stale theme embeddings in bounded batches',
        'schedule_description': 'Daily 5:10 AM ET',
    },

    # ===== WEEKDAYS (After Market Close) =====
    'auto-refresh-after-close': {
        'task_function': 'app.tasks.cache_tasks.auto_refresh_after_close',
        'display_name': 'Auto Refresh After Close',
        'description': 'Refreshes symbols with stale intraday data',
        'schedule_description': '4:45 PM ET, Mon-Fri',
    },
    'daily-cache-warmup': {
        'task_function': 'app.tasks.cache_tasks.daily_cache_warmup',
        'display_name': 'Daily Cache Warmup',
        'description': 'Warms SPY benchmark and all active symbols cache',
        'schedule_description': f'{settings.cache_warm_hour}:{settings.cache_warm_minute:02d} ET, Mon-Fri',
    },
    'daily-breadth-calculation': {
        'task_function': 'app.tasks.breadth_tasks.calculate_daily_breadth_with_gapfill',
        'display_name': 'Daily Breadth Calculation',
        'description': 'Calculates 13 market breadth indicators (with automatic gap-fill)',
        'schedule_description': f'{settings.cache_warm_hour}:{settings.cache_warm_minute + 5:02d} ET, Mon-Fri',
    },
    'daily-group-ranking-calculation': {
        'task_function': 'app.tasks.group_rank_tasks.calculate_daily_group_rankings',
        'display_name': 'Daily Group Rankings',
        'description': 'Calculates IBD industry group rankings',
        'schedule_description': f'{settings.cache_warm_hour}:{settings.cache_warm_minute + 10:02d} ET, Mon-Fri',
    },

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

    _instance = None

    def __init__(self):
        """Initialize the service with task imports."""
        self._task_imports = {}

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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
            # Get last execution from history
            last_run = db.query(TaskExecutionHistory).filter(
                TaskExecutionHistory.task_name == task_name
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
        celery_task = task_func.delay()

        # Record the execution in history
        execution = TaskExecutionHistory(
            task_name=task_name,
            task_function=task_info['task_function'],
            task_id=celery_task.id,
            status='queued',
            started_at=datetime.now(),
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
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.duration_seconds = (
                    execution.completed_at - execution.started_at
                ).total_seconds()
            if result:
                execution.result_summary = result
            if error:
                execution.error_message = error
            db.commit()

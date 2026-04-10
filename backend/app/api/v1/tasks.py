"""
API endpoints for managing scheduled Celery tasks.

Provides access to task schedules, execution history,
and manual task triggering.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.task import (
    TaskListResponse,
    ScheduledTaskResponse,
    TriggerTaskResponse,
    TaskStatusResponse,
)
from ...services.task_registry_service import SCHEDULED_TASKS
from ...wiring.bootstrap import get_task_registry_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/scheduled", response_model=TaskListResponse)
async def get_scheduled_tasks(db: Session = Depends(get_db)):
    """
    Get all scheduled Celery tasks with their schedules and last run info.

    Returns a list of all configured scheduled tasks including:
    - Task name and description
    - Schedule (human-readable)
    - Last execution status and timing
    """
    service = get_task_registry_service()
    tasks = service.get_all_scheduled_tasks(db)

    return TaskListResponse(
        tasks=[ScheduledTaskResponse(**t) for t in tasks],
        total_tasks=len(tasks)
    )


@router.post("/{task_name}/run", response_model=TriggerTaskResponse)
async def trigger_task(
    task_name: str,
    db: Session = Depends(get_db)
):
    """
    Manually trigger a scheduled task.

    This queues the task for immediate execution.
    Use the status endpoint to poll for completion.

    Args:
        task_name: Name of the task to trigger (e.g., 'daily-smart-refresh')

    Returns:
        Task ID for status polling
    """
    if task_name not in SCHEDULED_TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task: {task_name}. Available tasks: {list(SCHEDULED_TASKS.keys())}"
        )

    try:
        service = get_task_registry_service()
        result = service.trigger_task(task_name, db)
        return TriggerTaskResponse(**result)
    except Exception as e:
        logger.error(f"Error triggering task {task_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger task: {str(e)}"
        )


@router.get("/{task_name}/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_name: str,
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the current status of a task execution.

    Use this endpoint to poll for task completion after triggering.

    Args:
        task_name: Name of the task
        task_id: Celery task ID from the trigger response

    Returns:
        Current status, progress (if available), and result/error
    """
    if task_name not in SCHEDULED_TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task: {task_name}"
        )

    try:
        service = get_task_registry_service()
        result = service.get_task_status(task_name, task_id, db)
        return TaskStatusResponse(**result)
    except Exception as e:
        logger.error(f"Error getting task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )

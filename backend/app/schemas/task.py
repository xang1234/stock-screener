"""Pydantic schemas for Scheduled Tasks API endpoints"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class LastRunInfo(BaseModel):
    """Information about the last execution of a task"""

    id: int = Field(..., description="Execution history record ID")
    task_id: Optional[str] = Field(None, description="Celery task ID")
    status: str = Field(..., description="Status: queued, running, completed, failed")
    started_at: Optional[str] = Field(None, description="Start time (ISO format)")
    completed_at: Optional[str] = Field(None, description="Completion time (ISO format)")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    triggered_by: str = Field(..., description="Trigger source: schedule, manual")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ScheduledTaskResponse(BaseModel):
    """Response model for a single scheduled task"""

    name: str = Field(..., description="Task identifier (e.g., 'daily-smart-refresh')")
    display_name: str = Field(..., description="Human-readable task name")
    task_function: str = Field(..., description="Full task function path")
    description: str = Field(..., description="Description of what the task does")
    schedule_description: str = Field(..., description="Human-readable schedule")
    is_enabled: bool = Field(..., description="Whether scheduling is enabled")
    last_run: Optional[LastRunInfo] = Field(None, description="Last execution details")


class TaskListResponse(BaseModel):
    """Response model for list of scheduled tasks"""

    tasks: List[ScheduledTaskResponse] = Field(..., description="List of scheduled tasks")
    total_tasks: int = Field(..., description="Total number of tasks")


class TriggerTaskResponse(BaseModel):
    """Response model when triggering a task manually"""

    task_id: str = Field(..., description="Celery task ID for status polling")
    task_name: str = Field(..., description="Name of the triggered task")
    status: str = Field(..., description="Initial status (queued)")
    execution_id: int = Field(..., description="Execution history record ID")
    message: str = Field(..., description="Human-readable message")


class TaskStatusResponse(BaseModel):
    """Response model for task execution status"""

    task_id: str = Field(..., description="Celery task ID")
    task_name: str = Field(..., description="Name of the task")
    status: str = Field(..., description="Status: queued, running, completed, failed")
    celery_state: str = Field(..., description="Raw Celery state")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Progress message")
    current: Optional[int] = Field(None, description="Current item being processed")
    total: Optional[int] = Field(None, description="Total items to process")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result when completed")
    error: Optional[str] = Field(None, description="Error message when failed")

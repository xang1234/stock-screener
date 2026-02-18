"""Shared Pydantic schemas used across multiple routers."""

from pydantic import BaseModel


class TaskResponse(BaseModel):
    """Generic task response model.

    Shared by cache and fundamentals routers for Celery task dispatch.
    """

    task_id: str
    message: str
    status: str

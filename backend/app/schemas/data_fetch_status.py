"""Pydantic schemas for data-fetch job status API endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class DataFetchStatusResponse(BaseModel):
    """Response model for data-fetch status."""

    is_running: bool
    current_task: Optional[Dict[str, Any]] = None


class ForceReleaseLockResponse(BaseModel):
    """Response model for force-release lock."""

    success: bool
    message: str

"""
API endpoints for data-fetch job status visibility.

Provides endpoints to check which data-fetching task is currently running,
and to force-release a stuck lock if needed.
"""
from fastapi import APIRouter, HTTPException

from ...schemas.data_fetch_status import DataFetchStatusResponse, ForceReleaseLockResponse
from ...wiring.bootstrap import get_data_fetch_lock

router = APIRouter(prefix="/data-fetch", tags=["data-fetch"])


@router.get("/status", response_model=DataFetchStatusResponse)
async def get_data_fetch_status():
    """
    Get current data-fetch job status.

    Returns information about whether a data-fetching task is currently
    running, and if so, which task and when it started.

    Returns:
        DataFetchStatusResponse with:
        - is_running: Whether a data-fetch task is currently running
        - current_task: Info about the running task (task_name, task_id, started_at, ttl_seconds)
    """
    lock = get_data_fetch_lock()
    # Use any-market checks so per-market Beat tasks are visible, not just
    # the legacy shared key (which no scheduled task writes to after 9.1).
    holder = lock.get_any_current_holder()

    return DataFetchStatusResponse(
        is_running=lock.is_any_locked(),
        current_task=holder
    )


@router.post("/force-release-lock", response_model=ForceReleaseLockResponse)
async def force_release_lock():
    """
    Force release a stuck data-fetch lock.

    Use this endpoint only if a task has crashed without releasing the lock.
    The lock has a TTL (default 2 hours), so it will auto-expire eventually.

    WARNING: Use with caution. Only force-release if you're sure no task is running.

    Returns:
        ForceReleaseLockResponse with success status and message
    """
    lock = get_data_fetch_lock()

    # Check across all market scopes (not just :shared).
    if not lock.is_any_locked():
        return ForceReleaseLockResponse(
            success=False,
            message="No lock to release - no data-fetch task is currently running"
        )

    # Get current holder info before releasing
    holder = lock.get_any_current_holder()
    task_name = holder.get('task_name', 'unknown') if holder else 'unknown'

    # Force-release all market lock keys (one per active market that might be held).
    count = lock.force_release_all()

    if count > 0:
        return ForceReleaseLockResponse(
            success=True,
            message=f"Lock force-released ({count} key(s)). Previous holder: {task_name}"
        )
    else:
        return ForceReleaseLockResponse(
            success=False,
            message="Failed to release lock"
        )

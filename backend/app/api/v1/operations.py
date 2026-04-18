"""Operations job console endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.operations import OperationsCancelJobResponse, OperationsJobsResponse
from ...services.operations_job_service import OperationsJobService

router = APIRouter(prefix="/operations", tags=["operations"])

_service = OperationsJobService()


@router.get("/jobs", response_model=OperationsJobsResponse)
def get_operations_jobs(db: Session = Depends(get_db)) -> OperationsJobsResponse:
    """Return queue, worker, and lease state for the Operations console."""
    return OperationsJobsResponse(**_service.list_jobs(db))


@router.post("/jobs/{task_id}/cancel", response_model=OperationsCancelJobResponse)
def cancel_operations_job(
    task_id: str,
    db: Session = Depends(get_db),
) -> OperationsCancelJobResponse:
    """Safely cancel or revoke a job when the strategy supports it."""
    return OperationsCancelJobResponse(**_service.cancel_job(db, task_id))

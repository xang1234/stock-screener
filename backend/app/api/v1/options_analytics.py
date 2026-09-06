"""Protected, read-mostly Options Analytics API."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.options_analytics import (
    OptionsCommandCenterResponse,
    OptionsRefreshAcceptedResponse,
    OptionsRefreshRequest,
    OptionsRunDiagnosticsResponse,
    OptionsSymbolDetailResponse,
)
from app.schemas.task import TaskStatusResponse
from app.wiring.bootstrap import (
    get_options_analytics_queries,
    get_task_registry_service,
)

router = APIRouter()


def _unavailable(*reason_codes: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "code": "options_analytics_unavailable",
            "reason_codes": list(reason_codes),
        },
    )


def dispatch_options_refresh(
    *, source_run_id: int | None, force: bool
) -> dict[str, Any]:
    from app.interfaces.tasks.options_analytics_tasks import refresh_options_analytics
    from app.tasks.market_queues import data_fetch_queue_for_market

    task = refresh_options_analytics.apply_async(
        kwargs={"source_run_id": source_run_id, "market": "US", "force": force},
        queue=data_fetch_queue_for_market("US"),
    )
    return {"task_id": task.id, "run_id": None, "source_run_id": source_run_id}


@router.get("/command-center", response_model=OptionsCommandCenterResponse)
def get_command_center(db: Annotated[Session, Depends(get_db)]):
    queries = get_options_analytics_queries(db)
    run = queries.get_published_command_center("US")
    if run is None:
        raise _unavailable("published_run_unavailable")
    return OptionsCommandCenterResponse.from_run(
        run, stale=queries.is_stale(run, "US")
    )


@router.get("/symbols/{symbol}", response_model=OptionsSymbolDetailResponse)
def get_symbol_detail(symbol: str, db: Annotated[Session, Depends(get_db)]):
    queries = get_options_analytics_queries(db)
    result = queries.get_published_symbol_detail(symbol, "US")
    if result is None:
        raise _unavailable("published_symbol_unavailable")
    return OptionsSymbolDetailResponse.from_result(
        result, stale=queries.is_stale(result.run, "US")
    )


@router.get(
    "/runs/{run_id}/diagnostics", response_model=OptionsRunDiagnosticsResponse
)
def get_run_diagnostics(run_id: int, db: Annotated[Session, Depends(get_db)]):
    run = get_options_analytics_queries(db).get_run_diagnostics(run_id)
    if run is None:
        raise _unavailable("run_unavailable")
    return OptionsRunDiagnosticsResponse.from_run(run)


@router.post(
    "/refresh",
    response_model=OptionsRefreshAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def refresh_options(request: OptionsRefreshRequest):
    dispatched = dispatch_options_refresh(
        source_run_id=request.source_run_id,
        force=request.force,
    )
    return OptionsRefreshAcceptedResponse(
        task_id=dispatched["task_id"],
        run_id=dispatched.get("run_id"),
        source_run_id=dispatched.get("source_run_id"),
    )


@router.get("/refresh/{task_id}/status", response_model=TaskStatusResponse)
def get_options_refresh_status(
    task_id: str,
    db: Annotated[Session, Depends(get_db)],
):
    result = get_task_registry_service().get_task_status(
        "daily-us-options-analytics",
        task_id,
        db,
    )
    return TaskStatusResponse(**result)

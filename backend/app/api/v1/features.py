"""Feature store API endpoints.

Provides monitoring and comparison capabilities for the feature store:
- GET /features/runs — list feature runs with row counts
- GET /features/compare — compare two feature runs side-by-side
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...domain.common.errors import (
    EntityNotFoundError,
    ValidationError as DomainValidationError,
)
from ...infra.db.uow import SqlUnitOfWork
from ...schemas.feature_store import (
    CompareRunsResponse,
    FeatureRunResponse,
    ListRunsResponse,
)
from ...use_cases.feature_store.compare_runs import (
    CompareFeatureRunsUseCase,
    CompareRunsQuery,
)
from ...use_cases.feature_store.list_runs import (
    ListFeatureRunsUseCase,
    ListRunsQuery,
)
from ...wiring.bootstrap import (
    get_compare_feature_runs_use_case,
    get_list_feature_runs_use_case,
    get_uow,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/runs", response_model=ListRunsResponse)
async def list_runs(
    status: Optional[str] = Query(None, description="Filter by run status"),
    date_from: Optional[date] = Query(None, description="Start date (inclusive)"),
    date_to: Optional[date] = Query(None, description="End date (inclusive)"),
    limit: int = Query(50, ge=1, le=200, description="Max runs to return"),
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: ListFeatureRunsUseCase = Depends(get_list_feature_runs_use_case),
):
    """List feature runs with row counts and publish status."""
    try:
        query = ListRunsQuery(
            status=status,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
        result = use_case.execute(uow, query)
    except DomainValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    runs = [FeatureRunResponse.from_domain(r) for r in result.runs]
    return ListRunsResponse(runs=runs)


@router.get("/compare", response_model=CompareRunsResponse)
async def compare_runs(
    run_a: int = Query(..., description="First run ID (baseline)"),
    run_b: int = Query(..., description="Second run ID (comparison)"),
    limit: int = Query(50, ge=1, le=500, description="Max movers to return"),
    uow: SqlUnitOfWork = Depends(get_uow),
    use_case: CompareFeatureRunsUseCase = Depends(get_compare_feature_runs_use_case),
):
    """Compare two feature runs: added/removed symbols and score movers."""
    try:
        query = CompareRunsQuery(run_a=run_a, run_b=run_b, limit=limit)
        result = use_case.execute(uow, query)
    except DomainValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return CompareRunsResponse.from_domain(result)

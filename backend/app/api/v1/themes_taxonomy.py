"""Themes API routes for L1/L2 taxonomy workflows."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...database import get_db
from ...schemas.theme import (
    L1CategoriesResponse,
    L1CategoryItem,
    L1ChildItem,
    L1ChildrenResponse,
    L1ThemeDetail,
    L1ThemeRankingItem,
    L1ThemeRankingsResponse,
    L2ReassignRequest,
    TaxonomyAssignmentRequest,
    UnassignedThemeItem,
    UnassignedThemesResponse,
)

router = APIRouter()


@router.get("/taxonomy/l1", response_model=L1ThemeRankingsResponse)
async def get_l1_rankings(
    pipeline: str = Query("technical", description="Pipeline filter"),
    category: Optional[str] = Query(None, description="Filter by L1 category"),
    sort_by: str = Query("momentum_score", description="Sort field: momentum_score, mentions_7d, num_constituents, basket_return_1w, basket_rs_vs_spy, display_name, rank"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get L1 theme rankings with aggregated metrics."""
    from ...services.theme_taxonomy_service import ThemeTaxonomyService

    service = ThemeTaxonomyService(db, pipeline=pipeline)
    rankings, total = service.get_l1_themes(
        category_filter=category,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return L1ThemeRankingsResponse(
        total=total,
        pipeline=pipeline,
        rankings=[L1ThemeRankingItem(**row) for row in rankings],
    )


@router.get("/taxonomy/l1/{l1_id}/children", response_model=L1ChildrenResponse)
async def get_l1_children(
    l1_id: int,
    sort_by: str = Query("momentum_score", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get L2 children of an L1 theme."""
    from ...services.theme_taxonomy_service import ThemeTaxonomyService

    service = ThemeTaxonomyService(db)
    result = service.get_l1_with_children(
        l1_id,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"L1 theme {l1_id} not found")
    return L1ChildrenResponse(
        l1=L1ThemeDetail(**result["l1"]),
        children=[L1ChildItem(**child) for child in result["children"]],
        total_children=result["total_children"],
    )


@router.post("/taxonomy/assign")
async def run_taxonomy_assignment(
    request: TaxonomyAssignmentRequest,
    db: Session = Depends(get_db),
):
    """Run full taxonomy assignment pipeline (rules -> clustering -> LLM naming)."""
    from ...services.theme_taxonomy_service import ThemeTaxonomyService

    service = ThemeTaxonomyService(db, pipeline=request.pipeline)
    report = service.run_full_taxonomy_assignment(dry_run=request.dry_run)
    if not request.dry_run:
        from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

        safe_publish_themes_bootstrap_variants(request.pipeline)
    return report


@router.post("/taxonomy/assign/async")
async def run_taxonomy_assignment_async(
    pipeline: str = Query("technical"),
    dry_run: bool = Query(False),
    db: Session = Depends(get_db),
):
    """Run taxonomy assignment asynchronously via Celery."""
    del db
    from ...tasks.theme_discovery_tasks import run_taxonomy_assignment as taxonomy_task

    task = taxonomy_task.delay(pipeline=pipeline, dry_run=dry_run)
    return {"task_id": task.id, "status": "queued"}


@router.put("/taxonomy/{l2_id}/reassign")
async def reassign_l2_to_l1(
    l2_id: int,
    request: L2ReassignRequest,
    db: Session = Depends(get_db),
):
    """Manually reassign an L2 theme to a different L1 parent."""
    from ...services.theme_taxonomy_service import ThemeTaxonomyService

    service = ThemeTaxonomyService(db)
    success = service.assign_l2_to_l1(l2_id, request.l1_id, method="manual", confidence=1.0)
    if not success:
        raise HTTPException(status_code=404, detail="L2 or L1 theme not found")
    db.commit()
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants()
    return {"success": True, "l2_id": l2_id, "l1_id": request.l1_id}


@router.get("/taxonomy/unassigned", response_model=UnassignedThemesResponse)
async def get_unassigned_themes(
    pipeline: str = Query("technical"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get L2 themes without L1 parent assignment."""
    from ...services.theme_taxonomy_service import ThemeTaxonomyService

    service = ThemeTaxonomyService(db, pipeline=pipeline)
    themes, total = service.get_unassigned_themes(limit=limit, offset=offset)
    return UnassignedThemesResponse(
        total=total,
        themes=[UnassignedThemeItem(**theme) for theme in themes],
    )


@router.get("/taxonomy/categories", response_model=L1CategoriesResponse)
async def get_l1_categories(
    pipeline: str = Query("technical"),
    db: Session = Depends(get_db),
):
    """List available L1 categories with theme counts."""
    from ...services.theme_taxonomy_service import ThemeTaxonomyService

    service = ThemeTaxonomyService(db, pipeline=pipeline)
    categories = service.get_categories()
    return L1CategoriesResponse(
        categories=[L1CategoryItem(**category) for category in categories],
    )


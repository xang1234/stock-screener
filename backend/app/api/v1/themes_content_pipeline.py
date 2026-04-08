"""Themes API routes for pipeline execution and content browser reads."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from importlib import import_module
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.theme import ContentItem, ContentItemPipelineState, ContentSource
from ...schemas.theme import ContentItemsListResponse, ThemePipelineObservabilityResponse
from ...services.theme_pipeline_state_service import (
    compute_pipeline_observability,
    compute_pipeline_state_health,
)
from ...theme_platform.content_browser_queries import render_content_items_csv_chunk
from ...theme_platform.contracts import PipelineRunStatusPayload
from .themes_common import _VALID_THEME_PIPELINES, resolve_source_ids_for_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


def _themes_api_module():
    """Load themes aggregate module to preserve monkeypatch-compatible call sites."""
    return import_module("app.api.v1.themes")


@router.post("/pipeline/run")
def run_pipeline_async(
    pipeline: Optional[str] = Query(None, description="Pipeline: technical, fundamental, or None for both"),
    lookback_days: Optional[int] = Query(None, ge=1, le=30, description="Re-fetch articles from the last N days (backfill mode)"),
    db: Session = Depends(get_db),
):
    """Start full theme discovery pipeline asynchronously."""
    from ...models.theme import ThemePipelineRun
    from ...tasks.theme_discovery_tasks import run_full_pipeline

    if pipeline is not None and pipeline not in _VALID_THEME_PIPELINES:
        raise HTTPException(status_code=400, detail="pipeline must be technical, fundamental, or omitted")

    run_id = str(uuid.uuid4())
    pipeline_run = ThemePipelineRun(
        run_id=run_id,
        pipeline=pipeline,
        status="created",
    )
    db.add(pipeline_run)
    db.commit()

    try:
        task = run_full_pipeline.delay(run_id=run_id, pipeline=pipeline, lookback_days=lookback_days)
    except Exception as exc:
        pipeline_run.status = "failed"
        pipeline_run.error_message = f"Failed to queue pipeline task: {exc}"
        db.commit()
        raise HTTPException(status_code=503, detail="Failed to queue theme discovery pipeline") from exc

    pipeline_run.task_id = task.id
    pipeline_run.status = "queued"
    db.commit()

    pipeline_desc = pipeline if pipeline else "both (technical + fundamental)"
    logger.info("Theme pipeline %s queued for %s with task ID: %s", run_id, pipeline_desc, task.id)

    return {
        "run_id": run_id,
        "task_id": task.id,
        "status": "queued",
        "pipeline": pipeline,
        "message": f"Theme discovery pipeline queued for {pipeline_desc}",
    }


@router.get("/pipeline/{run_id}/status")
def get_pipeline_status(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Get current status of a pipeline run."""
    from celery.result import AsyncResult
    from ...models.theme import ThemePipelineRun

    pipeline_run = db.query(ThemePipelineRun).filter(
        ThemePipelineRun.run_id == run_id
    ).first()

    if not pipeline_run:
        raise HTTPException(status_code=404, detail=f"Pipeline run {run_id} not found")

    response: PipelineRunStatusPayload = {
        "run_id": run_id,
        "task_id": pipeline_run.task_id,
        "status": pipeline_run.status,
        "current_step": pipeline_run.current_step,
        "step_number": 0,
        "total_steps": 5,
        "percent": 0.0,
        "message": None,
        "ingestion_result": None,
        "reprocessing_result": None,
        "extraction_result": None,
        "metrics_result": None,
        "alerts_result": None,
        "started_at": pipeline_run.started_at.isoformat() if pipeline_run.started_at else None,
        "completed_at": pipeline_run.completed_at.isoformat() if pipeline_run.completed_at else None,
        "error_message": pipeline_run.error_message,
    }

    if pipeline_run.status in ["completed", "failed"]:
        if pipeline_run.status == "completed":
            response["percent"] = 100.0
            response["step_number"] = 5
            response["current_step"] = "completed"
        return response

    if pipeline_run.task_id:
        task_result = AsyncResult(pipeline_run.task_id)

        if task_result.state == "PROGRESS" and task_result.info:
            info = task_result.info
            response["current_step"] = info.get("current_step")
            response["step_number"] = info.get("step_number", 0)
            response["percent"] = info.get("percent", 0.0)
            response["message"] = info.get("message")
            response["ingestion_result"] = info.get("ingestion_result")
            response["reprocessing_result"] = info.get("reprocessing_result")
            response["extraction_result"] = info.get("extraction_result")
            response["metrics_result"] = info.get("metrics_result")
            response["status"] = "running"
        elif task_result.state == "SUCCESS":
            response["status"] = "completed"
            response["percent"] = 100.0
            response["step_number"] = 5
            response["current_step"] = "completed"
            if task_result.result:
                response["alerts_result"] = task_result.result.get("steps", {}).get("alerts")
        elif task_result.state == "FAILURE":
            response["status"] = "failed"
            response["error_message"] = str(task_result.result) if task_result.result else "Unknown error"

    return response


@router.get("/pipeline/runs")
def list_pipeline_runs(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """List recent pipeline runs."""
    from ...models.theme import ThemePipelineRun

    runs = db.query(ThemePipelineRun).order_by(
        ThemePipelineRun.started_at.desc()
    ).limit(limit).all()

    return {
        "total": len(runs),
        "runs": [
            {
                "run_id": run.run_id,
                "status": run.status,
                "current_step": run.current_step,
                "items_ingested": run.items_ingested,
                "items_reprocessed": run.items_reprocessed,
                "themes_updated": run.themes_updated,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            }
            for run in runs
        ],
    }


@router.get("/pipeline/failed-count")
def get_failed_items_count(
    pipeline: Optional[str] = Query(None, description="Filter by pipeline"),
    db: Session = Depends(get_db),
):
    """Count content items with extraction errors eligible for reprocessing."""
    from sqlalchemy import func

    cutoff = datetime.utcnow() - timedelta(days=30)

    query = db.query(func.count(ContentItemPipelineState.id)).join(
        ContentItem,
        ContentItem.id == ContentItemPipelineState.content_item_id,
    ).filter(
        ContentItemPipelineState.status == "failed_retryable",
        ContentItem.published_at >= cutoff,
    )

    if pipeline:
        if pipeline not in _VALID_THEME_PIPELINES:
            raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

        query = query.filter(ContentItemPipelineState.pipeline == pipeline)
        source_ids = resolve_source_ids_for_pipeline(db, pipeline)
        if source_ids:
            query = query.filter(ContentItem.source_id.in_(source_ids))
        else:
            return {"failed_count": 0, "max_age_days": 30}

    count = query.scalar()
    return {"failed_count": count, "max_age_days": 30}


@router.get("/pipeline/state-health")
def get_pipeline_state_health(
    pipeline: Optional[str] = Query(None, description="Pipeline: technical or fundamental"),
    window_days: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    db: Session = Depends(get_db),
):
    """Return pipeline-state health and drift metrics for operational triage."""
    if pipeline and pipeline not in _VALID_THEME_PIPELINES:
        raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

    return compute_pipeline_state_health(
        db=db,
        pipeline=pipeline,
        max_age_days=window_days,
    )


@router.get("/pipeline/observability", response_model=ThemePipelineObservabilityResponse)
def get_pipeline_observability(
    pipeline: str = Query(..., description="Pipeline: technical or fundamental"),
    window_days: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    db: Session = Depends(get_db),
):
    """Return dashboard summary plus alert policies for pipeline health."""
    if pipeline not in _VALID_THEME_PIPELINES:
        raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")
    return compute_pipeline_observability(
        db=db,
        pipeline=pipeline,
        max_age_days=window_days,
    )


@router.get("/content", response_model=ContentItemsListResponse)
def list_content_items(
    search: Optional[str] = Query(None, description="Search in title, source_name, tickers"),
    source_type: Optional[str] = Query(None, description="Filter: substack, twitter, news, reddit"),
    sentiment: Optional[str] = Query(None, description="Filter: bullish, bearish, neutral"),
    date_from: Optional[str] = Query(None, description="From date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="To date (YYYY-MM-DD)"),
    pipeline: Optional[str] = Query(None, description="Pipeline: technical or fundamental"),
    limit: int = Query(50, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: str = Query("published_at", description="Sort column"),
    sort_order: str = Query("desc", description="asc or desc"),
    db: Session = Depends(get_db),
):
    """List content items with associated themes, sentiments, and tickers."""
    if pipeline and pipeline not in _VALID_THEME_PIPELINES:
        raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

    themes_api = _themes_api_module()
    items, total = themes_api._fetch_content_items_with_themes_with_recovery(
        db,
        search=search,
        source_type=source_type,
        sentiment=sentiment,
        date_from=date_from,
        date_to=date_to,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
        pipeline=pipeline,
    )

    return ContentItemsListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


@router.get("/content/export")
def export_content_items(
    search: Optional[str] = Query(None, description="Search in title, source_name, tickers"),
    source_type: Optional[str] = Query(None, description="Filter: substack, twitter, news, reddit"),
    sentiment: Optional[str] = Query(None, description="Filter: bullish, bearish, neutral"),
    date_from: Optional[str] = Query(None, description="From date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="To date (YYYY-MM-DD)"),
    pipeline: Optional[str] = Query(None, description="Pipeline: technical or fundamental"),
    sort_by: str = Query("published_at", description="Sort column"),
    sort_order: str = Query("desc", description="asc or desc"),
    db: Session = Depends(get_db),
):
    """Export all content items matching filters as a CSV file."""
    if pipeline and pipeline not in _VALID_THEME_PIPELINES:
        raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

    themes_api = _themes_api_module()
    page_size = 500

    def _csv_stream():
        offset = 0
        include_header = True
        include_bom = True

        while True:
            items, total = themes_api._fetch_content_items_with_themes_with_recovery(
                db,
                search=search,
                source_type=source_type,
                sentiment=sentiment,
                date_from=date_from,
                date_to=date_to,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=page_size,
                offset=offset,
                pipeline=pipeline,
            )

            chunk = render_content_items_csv_chunk(
                items,
                include_header=include_header,
                include_bom=include_bom,
            )
            if chunk:
                yield chunk

            if not items:
                break

            include_header = False
            include_bom = False
            offset += len(items)
            if offset >= total:
                break

    filename = f"theme_articles_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        _csv_stream(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

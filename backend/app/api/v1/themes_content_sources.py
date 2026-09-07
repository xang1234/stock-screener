"""Themes API routes for source management and ingestion/extraction."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.theme import ContentSource
from ...infra.db.models.social_signals import SocialSourceConfiguration
from ...schemas.theme import (
    ContentSourceCreate,
    ContentSourceResponse,
    ContentSourceUpdate,
    ExtractionResponse,
    IngestionResponse,
)
from ...services.content_ingestion_service import ContentIngestionService, seed_default_sources
from ...services.theme_correlation_service import ThemeCorrelationService
from ...services.theme_discovery_service import ThemeDiscoveryService
from ...services.theme_extraction_service import ThemeExtractionService
from ...services.theme_pipeline_state_service import (
    normalize_pipelines,
    reconcile_source_pipeline_change,
    validate_pipeline_selection,
)
from .themes_common import detect_source_type_from_url

logger = logging.getLogger(__name__)

router = APIRouter()


_SOCIAL_SOURCE_DETAIL = {
    "code": "social_source_managed_elsewhere",
    "message": "Manage this list in Operations → Social Sources.",
}


def _social_list_id(value: str | None) -> str | None:
    """Recognize accepted legacy spellings without accepting them for creation."""
    if not value:
        return None
    import re
    match = re.fullmatch(
        r"https://(?:x|twitter)\.com/i/lists/([0-9]{1,32})/?", value.strip()
    )
    return str(int(match.group(1))) if match else None


def _reject_social_owned(db: Session, *, source_id=None, url=None) -> None:
    query = db.query(SocialSourceConfiguration)
    if source_id is not None and query.filter_by(content_source_id=source_id).first():
        raise HTTPException(status_code=409, detail=_SOCIAL_SOURCE_DETAIL)
    list_id = _social_list_id(url)
    if list_id and query.filter_by(x_list_id=list_id).first():
        raise HTTPException(status_code=409, detail=_SOCIAL_SOURCE_DETAIL)


@router.get("/sources", response_model=list[ContentSourceResponse])
def list_content_sources(
    active_only: bool = Query(True),
    pipeline: Optional[str] = Query(None, description="Filter by pipeline: technical or fundamental"),
    db: Session = Depends(get_db),
):
    """List all configured content sources, optionally filtered by pipeline assignment."""
    query = db.query(ContentSource)
    if active_only:
        query = query.filter(ContentSource.is_active == True)

    sources = query.order_by(ContentSource.priority.desc()).all()

    if pipeline:
        sources = [
            source
            for source in sources
            if pipeline in normalize_pipelines(source.pipelines)
        ]

    return [ContentSourceResponse.model_validate(source) for source in sources]


@router.post("/sources", response_model=ContentSourceResponse)
def add_content_source(
    source: ContentSourceCreate,
    db: Session = Depends(get_db),
):
    """Add a new content source for theme extraction."""
    _reject_social_owned(db, url=source.url)
    detected_type = detect_source_type_from_url(source.url or source.name, source.source_type)

    if detected_type != source.source_type:
        logger.info(
            "Auto-correcting source type for '%s': %s -> %s",
            source.name,
            source.source_type,
            detected_type,
        )

    try:
        pipelines = validate_pipeline_selection(source.pipelines)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    new_source = ContentSource(
        name=source.name,
        source_type=detected_type,
        url=source.url,
        priority=source.priority,
        fetch_interval_minutes=source.fetch_interval_minutes,
        pipelines=pipelines,
        is_active=True,
        total_items_fetched=0,
    )
    db.add(new_source)
    db.commit()
    db.refresh(new_source)
    return ContentSourceResponse.model_validate(new_source)


@router.put("/sources/{source_id}", response_model=ContentSourceResponse)
def update_content_source(
    source_id: int,
    source: ContentSourceUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing content source and reconcile pipeline state assignments."""
    _reject_social_owned(db, source_id=source_id, url=source.url)
    existing = db.query(ContentSource).filter(ContentSource.id == source_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Source not found")

    old_pipelines = normalize_pipelines(existing.pipelines)
    new_pipelines = old_pipelines

    if source.name is not None:
        existing.name = source.name
    if source.source_type is not None:
        existing.source_type = source.source_type
    if source.url is not None:
        existing.url = source.url
    if source.priority is not None:
        existing.priority = source.priority
    if source.fetch_interval_minutes is not None:
        existing.fetch_interval_minutes = source.fetch_interval_minutes
    if source.is_active is not None:
        existing.is_active = source.is_active

    try:
        if source.pipelines is not None:
            new_pipelines = validate_pipeline_selection(source.pipelines)
            existing.pipelines = new_pipelines

        db.flush()

        if source.pipelines is not None and set(old_pipelines) != set(new_pipelines):
            reconcile_summary = reconcile_source_pipeline_change(
                db=db,
                source_id=source_id,
                old_pipelines=old_pipelines,
                new_pipelines=new_pipelines,
                commit_each_chunk=False,
            )
            logger.info("Source %s pipeline change reconciled: %s", source_id, reconcile_summary)

        db.commit()
        db.refresh(existing)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        raise

    return ContentSourceResponse.model_validate(existing)


@router.delete("/sources/{source_id}")
def delete_content_source(
    source_id: int,
    db: Session = Depends(get_db),
):
    """Deactivate a content source."""
    _reject_social_owned(db, source_id=source_id)
    source = db.query(ContentSource).filter(ContentSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    source.is_active = False
    db.commit()

    return {"status": "deactivated", "source": source.name}


@router.post("/sources/seed-defaults")
def seed_default_sources_endpoint(
    db: Session = Depends(get_db),
):
    """Seed default content sources."""
    seed_default_sources(db)
    return {"status": "success", "message": "Default sources seeded"}


@router.post("/ingest", response_model=IngestionResponse)
def run_ingestion(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Fetch new content from all active sources."""
    del background_tasks
    service = ContentIngestionService(db)
    result = service.fetch_all_active_sources()
    return IngestionResponse(**result)


@router.post("/extract", response_model=ExtractionResponse)
def run_extraction(
    limit: int = Query(50, ge=1, le=200, description="Max items to process"),
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$", description="Pipeline: technical or fundamental"),
    db: Session = Depends(get_db),
):
    """Extract themes from unprocessed content using LLM."""
    service = ThemeExtractionService(db, pipeline=pipeline)
    result = service.process_batch(limit=limit)
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants(pipeline)
    return ExtractionResponse(**result)


@router.post("/calculate-metrics")
def calculate_theme_metrics(
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$", description="Pipeline: technical or fundamental"),
    db: Session = Depends(get_db),
):
    """Calculate/update metrics for all active themes in a pipeline."""
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    result = service.update_all_theme_metrics()
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants(pipeline)
    return result


@router.post("/validate-all")
def validate_all_themes(
    min_correlation: float = Query(0.5, ge=0.2, le=0.9),
    db: Session = Depends(get_db),
):
    """Run validation on all active themes."""
    service = ThemeCorrelationService(db)
    result = service.run_full_validation(min_correlation)
    return result

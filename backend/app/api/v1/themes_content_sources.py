"""Themes API routes for source management and ingestion/extraction."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.theme import ContentSource
from ...schemas.theme import (
    ContentSourceCreate,
    ContentSourceResponse,
    ContentSourceUpdate,
    ExtractionResponse,
    IngestionResponse,
    TwitterSessionChallengeResponse,
    TwitterSessionImportRequest,
    TwitterSessionStatusResponse,
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
from ...services.xui_session_bridge_service import XUISessionBridgeError, XUISessionBridgeService
from .themes_common import detect_source_type_from_url

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/twitter/session", response_model=TwitterSessionStatusResponse)
def get_twitter_session_status() -> TwitterSessionStatusResponse:
    """Get current XUI auth/session status for twitter ingestion."""
    service = XUISessionBridgeService()
    try:
        status = service.get_auth_status()
    except XUISessionBridgeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return TwitterSessionStatusResponse.model_validate(status, from_attributes=True)


@router.post("/twitter/session/challenge", response_model=TwitterSessionChallengeResponse)
def create_twitter_session_challenge(request: Request) -> TwitterSessionChallengeResponse:
    """Issue one-time challenge token for browser-extension session import."""
    service = XUISessionBridgeService()
    origin = request.headers.get("origin")
    client_key = request.client.host if request.client else "unknown"
    try:
        challenge = service.create_import_challenge(origin=origin, client_key=client_key)
    except XUISessionBridgeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return TwitterSessionChallengeResponse.model_validate(challenge, from_attributes=True)


@router.post("/twitter/session/import", response_model=TwitterSessionStatusResponse)
def import_twitter_session_from_browser(
    payload: TwitterSessionImportRequest,
    request: Request,
) -> TwitterSessionStatusResponse:
    """Import x.com/twitter.com cookies from extension and persist XUI storage state."""
    service = XUISessionBridgeService()
    origin = request.headers.get("x-xui-bridge-origin") or request.headers.get("origin")
    client_key = request.client.host if request.client else "unknown"
    try:
        status = service.import_browser_cookies(
            challenge_id=payload.challenge_id,
            challenge_token=payload.challenge_token,
            cookies=[cookie.model_dump() for cookie in payload.cookies],
            origin=origin,
            client_key=client_key,
            browser=payload.browser,
            extension_version=payload.extension_version,
        )
    except XUISessionBridgeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return TwitterSessionStatusResponse.model_validate(status, from_attributes=True)


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

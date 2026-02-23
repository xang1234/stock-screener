"""
API endpoints for Theme Discovery.

Provides access to:
- Theme rankings and metrics
- Emerging theme discovery
- Content source management
- Theme validation and correlation analysis
- Alerts
"""
import csv
import io
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from ...database import get_db
from ...models.theme import (
    ContentSource,
    ContentItem,
    ContentItemPipelineState,
    ThemeMention,
    ThemeCluster,
    ThemeConstituent,
    ThemeMetrics,
    ThemeAlert,
)
from ...schemas.theme import (
    ContentSourceCreate,
    ContentSourceUpdate,
    ContentSourceResponse,
    ThemeClusterResponse,
    ThemeDetailResponse,
    ThemeConstituentResponse,
    ThemeMetricsResponse,
    ThemeRankingsResponse,
    ThemeRankingItem,
    EmergingThemesResponse,
    EmergingThemeResponse,
    AlertsResponse,
    ThemeAlertResponse,
    ThemeMentionsResponse,
    ThemeMentionDetailResponse,
    CorrelationDiscoveryResponse,
    CorrelationClusterResponse,
    CrossIndustryPairResponse,
    ThemeValidationResponse,
    NewEntrantResponse,
    IngestionResponse,
    ExtractionResponse,
    ThemeMergeSuggestionResponse,
    ThemeMergeSuggestionsResponse,
    ThemeMergeHistoryListResponse,
    SimilarThemesResponse,
    SimilarThemeResponse,
    ConsolidationResultResponse,
    MergeActionResponse,
    ContentItemsListResponse,
    ContentItemWithThemesResponse,
    ThemeReference,
)
from ...services.content_ingestion_service import ContentIngestionService, seed_default_sources
from ...services.theme_extraction_service import ThemeExtractionService
from ...services.theme_discovery_service import ThemeDiscoveryService
from ...services.theme_correlation_service import ThemeCorrelationService
from ...services.theme_merging_service import ThemeMergingService
from ...services.theme_pipeline_state_service import (
    compute_pipeline_state_health,
    normalize_pipelines,
    reconcile_source_pipeline_change,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def detect_source_type_from_url(url: str, provided_type: str | None) -> str:
    """
    Auto-detect source type from URL if not provided or if it looks incorrect.

    Returns the detected source type or the provided type if detection fails.
    """
    if not url:
        return provided_type or "news"

    url_lower = url.lower()

    # Twitter/X detection
    if 'twitter.com' in url_lower or 'x.com' in url_lower or url_lower.startswith('@'):
        return 'twitter'
    # Reddit detection
    elif 'reddit.com' in url_lower or url_lower.startswith('r/'):
        return 'reddit'
    # Substack detection
    elif 'substack.com' in url_lower:
        return 'substack'
    # RSS feed detection
    elif url_lower.endswith('/feed') or url_lower.endswith('.rss') or url_lower.endswith('.xml'):
        return 'substack'  # RSS feeds are treated as substack type

    # If provided type exists, use it; otherwise default to news
    return provided_type or "news"


# ==================== Theme Rankings ====================

@router.get("/pipelines")
async def get_available_pipelines():
    """
    Get list of available theme pipelines.

    Returns technical and fundamental pipeline configurations.
    """
    from ...config.pipeline_config import get_all_pipelines
    return {"pipelines": get_all_pipelines()}


@router.get("/rankings", response_model=ThemeRankingsResponse)
async def get_theme_rankings(
    limit: int = Query(20, ge=1, le=500, description="Number of themes to return"),
    offset: int = Query(0, ge=0, description="Number of themes to skip for pagination"),
    status: Optional[str] = Query(None, description="Filter by status: emerging, trending, fading, dormant"),
    source_types: Optional[str] = Query(None, description="Comma-separated source types: substack,twitter,news,reddit"),
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    recalculate: bool = Query(False, description="Force recalculation of metrics"),
    db: Session = Depends(get_db)
):
    """
    Get current theme rankings sorted by momentum score.

    Returns themes ranked by composite momentum score which combines:
    - Mention velocity (social signal strength)
    - Relative strength vs SPY
    - Breadth (% above 50MA)
    - Internal correlation (theme cohesiveness)
    - Screener quality (% passing Minervini)

    Rankings are calculated separately per pipeline.
    """
    service = ThemeDiscoveryService(db, pipeline=pipeline)

    # Check if metrics need updating (themes without metrics or forced recalc)
    if recalculate:
        service.update_all_theme_metrics()
    else:
        # Auto-calculate metrics for themes that don't have any
        from sqlalchemy import func
        themes_without_metrics = db.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ~ThemeCluster.id.in_(
                db.query(ThemeMetrics.theme_cluster_id).distinct()
            )
        ).count()
        if themes_without_metrics > 0:
            logger.info(f"Auto-calculating metrics for {themes_without_metrics} themes without metrics")
            service.update_all_theme_metrics()

    # Parse source_types from comma-separated string
    source_types_list = None
    if source_types:
        source_types_list = [t.strip() for t in source_types.split(",") if t.strip()]

    rankings, total_count = service.get_theme_rankings(
        limit=limit,
        status_filter=status,
        source_types_filter=source_types_list,
        offset=offset
    )

    if not rankings:
        return ThemeRankingsResponse(
            date=None,
            total_themes=total_count,
            rankings=[]
        )

    return ThemeRankingsResponse(
        date=datetime.utcnow().date().isoformat(),
        total_themes=total_count,
        pipeline=pipeline,
        rankings=[ThemeRankingItem(**r) for r in rankings]
    )


@router.get("/emerging", response_model=EmergingThemesResponse)
async def get_emerging_themes(
    min_velocity: float = Query(1.5, description="Minimum mention velocity"),
    min_mentions: int = Query(3, description="Minimum mentions in 7 days"),
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    db: Session = Depends(get_db)
):
    """
    Discover newly emerging themes.

    Returns themes first seen in last 7 days with accelerating mentions.
    Filtered by pipeline.
    """
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    themes = service.discover_emerging_themes(
        min_velocity=min_velocity,
        min_mentions=min_mentions
    )

    return EmergingThemesResponse(
        count=len(themes),
        themes=[EmergingThemeResponse(**t) for t in themes]
    )


# ==================== Alerts (MUST be before /{theme_id} to avoid route conflict) ====================

@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Get theme alerts (excludes dismissed alerts).
    """
    query = db.query(ThemeAlert).filter(ThemeAlert.is_dismissed == False)

    if unread_only:
        query = query.filter(ThemeAlert.is_read == False)

    query = query.order_by(ThemeAlert.triggered_at.desc()).limit(limit)
    alerts = query.all()

    unread_count = db.query(ThemeAlert).filter(
        ThemeAlert.is_read == False,
        ThemeAlert.is_dismissed == False
    ).count()

    return AlertsResponse(
        total=len(alerts),
        unread=unread_count,
        alerts=[ThemeAlertResponse.model_validate(a) for a in alerts]
    )


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """Dismiss (soft delete) an alert."""
    alert = db.query(ThemeAlert).filter(ThemeAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_dismissed = True
    db.commit()
    return {"status": "dismissed", "alert_id": alert_id}


# ==================== Content Sources (MUST be before /{theme_id}) ====================

@router.get("/sources", response_model=list[ContentSourceResponse])
async def list_content_sources(
    active_only: bool = Query(True),
    pipeline: Optional[str] = Query(None, description="Filter by pipeline: technical or fundamental"),
    db: Session = Depends(get_db)
):
    """
    List all configured content sources.

    Optionally filter by pipeline assignment.
    """
    query = db.query(ContentSource)
    if active_only:
        query = query.filter(ContentSource.is_active == True)

    sources = query.order_by(ContentSource.priority.desc()).all()

    # Filter by pipeline if specified
    if pipeline:
        import json
        filtered_sources = []
        for source in sources:
            source_pipelines = source.pipelines or ["technical", "fundamental"]
            if isinstance(source_pipelines, str):
                try:
                    source_pipelines = json.loads(source_pipelines)
                except:
                    source_pipelines = ["technical", "fundamental"]
            if pipeline in source_pipelines:
                filtered_sources.append(source)
        sources = filtered_sources

    return [ContentSourceResponse.model_validate(s) for s in sources]


@router.post("/sources", response_model=ContentSourceResponse)
async def add_content_source(
    source: ContentSourceCreate,
    db: Session = Depends(get_db)
):
    """
    Add a new content source for theme extraction.

    The pipelines field determines which pipelines this source feeds into.
    Source type is auto-detected from URL if it appears incorrect (e.g., Twitter URL with substack type).
    """
    # Auto-detect source type from URL/name if it looks like it might be wrong
    detected_type = detect_source_type_from_url(source.url or source.name, source.source_type)

    # Log if we're correcting the source type
    if detected_type != source.source_type:
        logger.info(f"Auto-correcting source type for '{source.name}': {source.source_type} -> {detected_type}")

    # Create source directly since ContentIngestionService.add_source may not support pipelines yet
    new_source = ContentSource(
        name=source.name,
        source_type=detected_type,
        url=source.url,
        priority=source.priority,
        fetch_interval_minutes=source.fetch_interval_minutes,
        pipelines=source.pipelines,
        is_active=True,
        total_items_fetched=0,
    )
    db.add(new_source)
    db.commit()
    db.refresh(new_source)
    return ContentSourceResponse.model_validate(new_source)


@router.put("/sources/{source_id}", response_model=ContentSourceResponse)
async def update_content_source(
    source_id: int,
    source: ContentSourceUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an existing content source.

    When the pipelines field changes, existing ThemeMention records from this
    source are updated to reflect the new pipeline classification.
    """
    existing = db.query(ContentSource).filter(ContentSource.id == source_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Source not found")

    # Capture old pipelines BEFORE update for comparison
    old_pipelines = normalize_pipelines(existing.pipelines)
    new_pipelines = old_pipelines

    # Update fields if provided
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
    if source.pipelines is not None:
        new_pipelines = normalize_pipelines(source.pipelines)
        existing.pipelines = new_pipelines

    db.commit()
    db.refresh(existing)

    # Reconcile pipeline-state rows if assignments changed (non-destructive)
    if source.pipelines is not None and set(old_pipelines) != set(new_pipelines):
        reconcile_summary = reconcile_source_pipeline_change(
            db=db,
            source_id=source_id,
            old_pipelines=old_pipelines,
            new_pipelines=new_pipelines,
        )
        logger.info("Source %s pipeline change reconciled: %s", source_id, reconcile_summary)

    return ContentSourceResponse.model_validate(existing)


@router.delete("/sources/{source_id}")
async def delete_content_source(
    source_id: int,
    db: Session = Depends(get_db)
):
    """
    Deactivate a content source.
    """
    source = db.query(ContentSource).filter(ContentSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    source.is_active = False
    db.commit()

    return {"status": "deactivated", "source": source.name}


@router.post("/sources/seed-defaults")
async def seed_default_sources_endpoint(
    db: Session = Depends(get_db)
):
    """
    Seed the database with default content sources.
    """
    seed_default_sources(db)
    return {"status": "success", "message": "Default sources seeded"}


# ==================== Ingestion & Extraction (MUST be before /{theme_id}) ====================

@router.post("/ingest", response_model=IngestionResponse)
async def run_ingestion(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Fetch new content from all active sources.

    This pulls content from RSS feeds, Twitter, news APIs, etc.
    """
    service = ContentIngestionService(db)
    result = service.fetch_all_active_sources()

    return IngestionResponse(**result)


@router.post("/extract", response_model=ExtractionResponse)
async def run_extraction(
    limit: int = Query(50, ge=1, le=200, description="Max items to process"),
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    db: Session = Depends(get_db)
):
    """
    Extract themes from unprocessed content using LLM.

    Processes content items that haven't been analyzed yet,
    extracting themes, tickers, and sentiment.

    Pipeline parameter determines which content sources to process and
    which pipeline-specific extraction prompt to use.
    """
    service = ThemeExtractionService(db, pipeline=pipeline)
    result = service.process_batch(limit=limit)

    return ExtractionResponse(**result)


@router.post("/calculate-metrics")
async def calculate_theme_metrics(
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    db: Session = Depends(get_db)
):
    """
    Calculate/update metrics for all active themes in a pipeline.

    This updates mention velocity, price performance, correlations,
    and rankings for all themes in the specified pipeline.
    Uses pipeline-specific scoring weights.
    """
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    result = service.update_all_theme_metrics()

    return result


@router.post("/validate-all")
async def validate_all_themes(
    min_correlation: float = Query(0.5, ge=0.2, le=0.9),
    db: Session = Depends(get_db)
):
    """
    Run validation on all active themes.

    Checks internal correlations and identifies invalid themes.
    """
    service = ThemeCorrelationService(db)
    result = service.run_full_validation(min_correlation)

    return result


# ==================== Async Pipeline (Celery-based) ====================

@router.post("/pipeline/run")
async def run_pipeline_async(
    pipeline: Optional[str] = Query(None, description="Pipeline: technical, fundamental, or None for both"),
    db: Session = Depends(get_db)
):
    """
    Start the full theme discovery pipeline asynchronously.

    Queues a Celery task to run:
    1. Content ingestion from all active sources
    2. Reprocess previously failed extractions (retry)
    3. Theme extraction via LLM (with pipeline-specific prompts)
    4. Metrics calculation for all themes (with pipeline-specific weights)
    5. Alert generation

    If pipeline is None, runs for both technical and fundamental sequentially.

    Returns a run_id for polling status.
    """
    import uuid
    from ...tasks.theme_discovery_tasks import run_full_pipeline
    from ...models.theme import ThemePipelineRun

    # Generate unique run ID
    run_id = str(uuid.uuid4())

    # Queue Celery task with pipeline parameter
    task = run_full_pipeline.delay(run_id=run_id, pipeline=pipeline)

    # Create pipeline run record for tracking
    pipeline_run = ThemePipelineRun(
        run_id=run_id,
        task_id=task.id,
        status="queued"
    )
    db.add(pipeline_run)
    db.commit()

    pipeline_desc = pipeline if pipeline else "both (technical + fundamental)"
    logger.info(f"Theme pipeline {run_id} queued for {pipeline_desc} with task ID: {task.id}")

    return {
        "run_id": run_id,
        "task_id": task.id,
        "status": "queued",
        "pipeline": pipeline,
        "message": f"Theme discovery pipeline queued for {pipeline_desc}"
    }


@router.get("/pipeline/{run_id}/status")
async def get_pipeline_status(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current status of a pipeline run.

    Polls Celery task state for real-time progress updates.
    Falls back to database record if task state unavailable.
    """
    from celery.result import AsyncResult
    from ...models.theme import ThemePipelineRun

    # Get pipeline run from database
    pipeline_run = db.query(ThemePipelineRun).filter(
        ThemePipelineRun.run_id == run_id
    ).first()

    if not pipeline_run:
        raise HTTPException(status_code=404, detail=f"Pipeline run {run_id} not found")

    # Default response from DB
    response = {
        'run_id': run_id,
        'task_id': pipeline_run.task_id,
        'status': pipeline_run.status,
        'current_step': pipeline_run.current_step,
        'step_number': 0,
        'total_steps': 5,
        'percent': 0.0,
        'message': None,
        'ingestion_result': None,
        'reprocessing_result': None,
        'extraction_result': None,
        'metrics_result': None,
        'alerts_result': None,
        'started_at': pipeline_run.started_at.isoformat() if pipeline_run.started_at else None,
        'completed_at': pipeline_run.completed_at.isoformat() if pipeline_run.completed_at else None,
        'error_message': pipeline_run.error_message,
    }

    # If completed or failed, return DB state
    if pipeline_run.status in ['completed', 'failed']:
        if pipeline_run.status == 'completed':
            response['percent'] = 100.0
            response['step_number'] = 5
            response['current_step'] = 'completed'
        return response

    # Get real-time progress from Celery task
    if pipeline_run.task_id:
        task_result = AsyncResult(pipeline_run.task_id)

        if task_result.state == 'PROGRESS' and task_result.info:
            info = task_result.info
            response['current_step'] = info.get('current_step')
            response['step_number'] = info.get('step_number', 0)
            response['percent'] = info.get('percent', 0.0)
            response['message'] = info.get('message')
            response['ingestion_result'] = info.get('ingestion_result')
            response['reprocessing_result'] = info.get('reprocessing_result')
            response['extraction_result'] = info.get('extraction_result')
            response['metrics_result'] = info.get('metrics_result')
            response['status'] = 'running'
        elif task_result.state == 'SUCCESS':
            response['status'] = 'completed'
            response['percent'] = 100.0
            response['step_number'] = 5
            response['current_step'] = 'completed'
            if task_result.result:
                response['alerts_result'] = task_result.result.get('steps', {}).get('alerts')
        elif task_result.state == 'FAILURE':
            response['status'] = 'failed'
            response['error_message'] = str(task_result.result) if task_result.result else 'Unknown error'

    return response


@router.get("/pipeline/runs")
async def list_pipeline_runs(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    List recent pipeline runs.
    """
    from ...models.theme import ThemePipelineRun

    runs = db.query(ThemePipelineRun).order_by(
        ThemePipelineRun.started_at.desc()
    ).limit(limit).all()

    return {
        'total': len(runs),
        'runs': [
            {
                'run_id': r.run_id,
                'status': r.status,
                'current_step': r.current_step,
                'items_ingested': r.items_ingested,
                'items_reprocessed': r.items_reprocessed,
                'themes_updated': r.themes_updated,
                'started_at': r.started_at.isoformat() if r.started_at else None,
                'completed_at': r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in runs
        ]
    }


@router.get("/pipeline/failed-count")
async def get_failed_items_count(
    pipeline: Optional[str] = Query(None, description="Filter by pipeline"),
    db: Session = Depends(get_db),
):
    """
    Count content items with extraction errors eligible for reprocessing.

    Returns the number of items that will be retried on the next pipeline run.
    """
    from datetime import timedelta
    from sqlalchemy import func

    cutoff = datetime.utcnow() - timedelta(days=30)

    query = db.query(func.count(ContentItemPipelineState.id)).join(
        ContentItem,
        ContentItem.id == ContentItemPipelineState.content_item_id,
    ).filter(
        ContentItemPipelineState.status == "failed_retryable",
        ContentItem.published_at >= cutoff,
    )

    # Filter by pipeline and active source assignments if specified
    if pipeline:
        if pipeline not in {"technical", "fundamental"}:
            raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

        query = query.filter(ContentItemPipelineState.pipeline == pipeline)
        import json as json_lib
        source_ids = []
        sources = db.query(ContentSource).filter(ContentSource.is_active == True).all()
        for source in sources:
            source_pipelines = source.pipelines or ["technical", "fundamental"]
            if isinstance(source_pipelines, str):
                try:
                    source_pipelines = json_lib.loads(source_pipelines)
                except Exception:
                    source_pipelines = ["technical", "fundamental"]
            if pipeline in source_pipelines:
                source_ids.append(source.id)
        if source_ids:
            query = query.filter(ContentItem.source_id.in_(source_ids))
        else:
            return {"failed_count": 0, "max_age_days": 30}

    count = query.scalar()
    return {"failed_count": count, "max_age_days": 30}


@router.get("/pipeline/state-health")
async def get_pipeline_state_health(
    pipeline: Optional[str] = Query(None, description="Pipeline: technical or fundamental"),
    window_days: int = Query(30, ge=1, le=365, description="Lookback window in days"),
    db: Session = Depends(get_db),
):
    """
    Pipeline-state health and drift metrics for operational triage.

    Includes:
    - pending/in-progress/processed/retryable/terminal counts
    - pending age percentiles
    - retry queue growth (last 24h vs previous 24h)
    - parse-failure rate
    - processed-without-mentions ratio
    """
    if pipeline and pipeline not in {"technical", "fundamental"}:
        raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

    return compute_pipeline_state_health(
        db=db,
        pipeline=pipeline,
        max_age_days=window_days,
    )


# ==================== Content Item Browser (MUST be before /{theme_id}) ====================


def _fetch_content_items_with_themes(
    db: Session,
    search: Optional[str] = None,
    source_type: Optional[str] = None,
    sentiment: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort_by: str = "published_at",
    sort_order: str = "desc",
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> tuple[list[ContentItemWithThemesResponse], int]:
    """
    Shared query logic for content items with theme/sentiment/ticker aggregation.

    Returns (items, total_count) where items are ContentItemWithThemesResponse objects.
    When limit/offset are None, returns all matching items (used by export).
    """
    from sqlalchemy import or_, desc, asc
    from datetime import datetime as dt

    # Base query - only processed items from active sources
    base_query = db.query(ContentItem).join(
        ContentSource, ContentItem.source_id == ContentSource.id
    ).filter(
        ContentItem.is_processed == True,
        ContentSource.is_active == True
    )

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        base_query = base_query.filter(
            or_(
                ContentItem.title.ilike(search_term),
                ContentItem.source_name.ilike(search_term),
            )
        )

    # Apply source_type filter
    if source_type:
        base_query = base_query.filter(ContentItem.source_type == source_type)

    # Apply date filters
    if date_from:
        try:
            from_date = dt.strptime(date_from, "%Y-%m-%d")
            base_query = base_query.filter(ContentItem.published_at >= from_date)
        except ValueError:
            pass

    if date_to:
        try:
            to_date = dt.strptime(date_to, "%Y-%m-%d")
            from datetime import timedelta
            to_date = to_date + timedelta(days=1)
            base_query = base_query.filter(ContentItem.published_at < to_date)
        except ValueError:
            pass

    # Get total count before pagination
    total = base_query.count()

    # Apply sorting
    sort_column = getattr(ContentItem, sort_by, ContentItem.published_at)
    if sort_order.lower() == "asc":
        base_query = base_query.order_by(asc(sort_column))
    else:
        base_query = base_query.order_by(desc(sort_column))

    # Apply pagination only if provided
    if limit is not None and offset is not None:
        content_items = base_query.offset(offset).limit(limit).all()
    else:
        content_items = base_query.all()

    if not content_items:
        return [], total

    # Get all content item IDs for batch fetching mentions
    content_ids = [item.id for item in content_items]

    # Fetch all theme mentions for these content items
    mentions = db.query(
        ThemeMention.content_item_id,
        ThemeMention.theme_cluster_id,
        ThemeMention.sentiment,
        ThemeMention.tickers,
        ThemeCluster.id.label("cluster_id"),
        ThemeCluster.name.label("cluster_name")
    ).outerjoin(
        ThemeCluster, ThemeMention.theme_cluster_id == ThemeCluster.id
    ).filter(
        ThemeMention.content_item_id.in_(content_ids)
    ).all()

    # Apply sentiment filter if specified - filter the content_ids based on mentions
    if sentiment:
        filtered_content_ids = set()
        for mention in mentions:
            if mention.sentiment and mention.sentiment.lower() == sentiment.lower():
                filtered_content_ids.add(mention.content_item_id)

        if not filtered_content_ids:
            return [], 0

        content_items = [item for item in content_items if item.id in filtered_content_ids]
        total = len(filtered_content_ids)

    # Also apply search to tickers if provided
    if search:
        search_upper = search.upper()
        tickers_matched_ids = set()
        for mention in mentions:
            if mention.tickers:
                for ticker in mention.tickers:
                    if search_upper in ticker.upper():
                        tickers_matched_ids.add(mention.content_item_id)
                        break

        if tickers_matched_ids:
            additional_items = db.query(ContentItem).filter(
                ContentItem.id.in_(tickers_matched_ids),
                ContentItem.is_processed == True
            ).all()
            existing_ids = {item.id for item in content_items}
            for item in additional_items:
                if item.id not in existing_ids:
                    content_items.append(item)

    # Aggregate mentions by content_item_id
    mentions_by_content = {}
    for mention in mentions:
        content_id = mention.content_item_id
        if content_id not in mentions_by_content:
            mentions_by_content[content_id] = {
                "themes": [],
                "sentiments": [],
                "tickers": set()
            }

        if mention.cluster_id and mention.cluster_name:
            theme_ref = {"id": mention.cluster_id, "name": mention.cluster_name}
            if theme_ref not in mentions_by_content[content_id]["themes"]:
                mentions_by_content[content_id]["themes"].append(theme_ref)

        if mention.sentiment:
            if mention.sentiment not in mentions_by_content[content_id]["sentiments"]:
                mentions_by_content[content_id]["sentiments"].append(mention.sentiment)

        if mention.tickers:
            mentions_by_content[content_id]["tickers"].update(mention.tickers)

    # Build response items
    items = []
    for content in content_items:
        mention_data = mentions_by_content.get(content.id, {"themes": [], "sentiments": [], "tickers": set()})

        sentiments_list = mention_data["sentiments"]
        primary_sentiment = None
        if sentiments_list:
            sentiment_counts = {}
            for s in sentiments_list:
                sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
            primary_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        items.append(ContentItemWithThemesResponse(
            id=content.id,
            title=content.title,
            content=content.content,
            url=content.url,
            source_type=content.source_type,
            source_name=content.source_name,
            author=content.author,
            published_at=content.published_at,
            themes=[ThemeReference(**t) for t in mention_data["themes"]],
            sentiments=sentiments_list,
            primary_sentiment=primary_sentiment,
            tickers=sorted(list(mention_data["tickers"]))
        ))

    return items, total


@router.get("/content", response_model=ContentItemsListResponse)
async def list_content_items(
    search: Optional[str] = Query(None, description="Search in title, source_name, tickers"),
    source_type: Optional[str] = Query(None, description="Filter: substack, twitter, news, reddit"),
    sentiment: Optional[str] = Query(None, description="Filter: bullish, bearish, neutral"),
    date_from: Optional[str] = Query(None, description="From date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="To date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=200, description="Items per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    sort_by: str = Query("published_at", description="Sort column"),
    sort_order: str = Query("desc", description="asc or desc"),
    db: Session = Depends(get_db)
):
    """
    List all content items with their associated themes, sentiments, and tickers.

    Aggregates data from ContentItem and ThemeMention tables to provide a unified
    view of all ingested content with their extracted metadata.
    """
    items, total = _fetch_content_items_with_themes(
        db, search=search, source_type=source_type, sentiment=sentiment,
        date_from=date_from, date_to=date_to, sort_by=sort_by,
        sort_order=sort_order, limit=limit, offset=offset,
    )

    return ContentItemsListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items
    )


@router.get("/content/export")
async def export_content_items(
    search: Optional[str] = Query(None, description="Search in title, source_name, tickers"),
    source_type: Optional[str] = Query(None, description="Filter: substack, twitter, news, reddit"),
    sentiment: Optional[str] = Query(None, description="Filter: bullish, bearish, neutral"),
    date_from: Optional[str] = Query(None, description="From date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="To date (YYYY-MM-DD)"),
    sort_by: str = Query("published_at", description="Sort column"),
    sort_order: str = Query("desc", description="asc or desc"),
    db: Session = Depends(get_db)
):
    """
    Export all content items matching filters as a CSV file.

    Returns all matching items (no pagination) with full article body text.
    CSV includes: ID, Title, Content, URL, Themes, Sentiment, Tickers,
    Published Date, Source Type, Source Name, Author.
    """
    items, total = _fetch_content_items_with_themes(
        db, search=search, source_type=source_type, sentiment=sentiment,
        date_from=date_from, date_to=date_to, sort_by=sort_by,
        sort_order=sort_order,
    )

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header row
    writer.writerow([
        "ID", "Title", "Content", "URL", "Themes", "Sentiment",
        "Tickers", "Published Date", "Source Type", "Source Name", "Author",
    ])

    # Data rows
    for item in items:
        writer.writerow([
            item.id,
            item.title or "",
            item.content or "",
            item.url or "",
            "; ".join(t.name for t in item.themes),
            item.primary_sentiment or "",
            "; ".join(item.tickers),
            item.published_at.strftime("%Y-%m-%d %H:%M") if item.published_at else "",
            item.source_type or "",
            item.source_name or "",
            item.author or "",
        ])

    # UTF-8 BOM for Excel compatibility
    csv_bytes = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
    filename = f"theme_articles_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ==================== Theme Merging (MUST be before /{theme_id}) ====================

@router.get("/merge-suggestions", response_model=ThemeMergeSuggestionsResponse)
async def get_merge_suggestions(
    status: Optional[str] = Query(None, description="Filter by status: pending, approved, rejected, auto_merged"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Get theme merge suggestions from the queue.

    Returns pairs of themes that may be duplicates, with embedding similarity
    scores and LLM verification results.
    """
    service = ThemeMergingService(db)
    suggestions = service.get_merge_suggestions(status=status, limit=limit)

    return ThemeMergeSuggestionsResponse(
        total=len(suggestions),
        suggestions=[ThemeMergeSuggestionResponse(**s) for s in suggestions]
    )


@router.post("/merge-suggestions/{suggestion_id}/approve", response_model=MergeActionResponse)
async def approve_merge_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db)
):
    """
    Approve a merge suggestion and execute the merge.

    The source theme will be merged INTO the target theme:
    - Aliases are combined
    - Constituents are reassigned
    - Mentions are reassigned
    - Source theme is deactivated
    """
    service = ThemeMergingService(db)
    result = service.approve_suggestion(suggestion_id)
    return MergeActionResponse(**result)


@router.post("/merge-suggestions/{suggestion_id}/reject")
async def reject_merge_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db)
):
    """
    Reject a merge suggestion.

    The suggestion is marked as rejected and will not be suggested again.
    """
    service = ThemeMergingService(db)
    result = service.reject_suggestion(suggestion_id)
    return result


@router.get("/merge-history", response_model=ThemeMergeHistoryListResponse)
async def get_merge_history(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Get history of theme merges.

    Returns audit trail of all merges (auto and manual).
    """
    service = ThemeMergingService(db)
    history = service.get_merge_history(limit=limit)

    return ThemeMergeHistoryListResponse(
        total=len(history),
        history=history
    )


@router.post("/consolidate", response_model=ConsolidationResultResponse)
async def run_theme_consolidation(
    dry_run: bool = Query(True, description="If true, only report what would happen without executing merges"),
    db: Session = Depends(get_db)
):
    """
    Run theme consolidation pipeline.

    Steps:
    1. Update embeddings for all themes
    2. Find similar pairs via cosine similarity
    3. Verify with LLM
    4. Auto-merge high confidence pairs, queue others for review

    Use dry_run=true first to preview changes.
    """
    service = ThemeMergingService(db)
    result = service.run_consolidation(dry_run=dry_run)
    return ConsolidationResultResponse(**result)


@router.post("/consolidate/async")
async def run_theme_consolidation_async(
    dry_run: bool = Query(False, description="If true, only report what would happen"),
    db: Session = Depends(get_db)
):
    """
    Run theme consolidation pipeline asynchronously via Celery.

    Returns a task ID for polling status.
    """
    from ...tasks.theme_discovery_tasks import consolidate_themes

    task = consolidate_themes.delay(dry_run=dry_run)

    return {
        "task_id": task.id,
        "status": "queued",
        "message": f"Theme consolidation queued (dry_run={dry_run})"
    }


# ==================== Theme Details (parameterized routes - MUST be after static ones) ====================

@router.get("/{theme_id}", response_model=ThemeDetailResponse)
async def get_theme_detail(
    theme_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific theme.

    Includes constituents, metrics, and historical data.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    # Get constituents
    constituents = db.query(ThemeConstituent).filter(
        ThemeConstituent.theme_cluster_id == theme_id,
        ThemeConstituent.is_active == True
    ).order_by(ThemeConstituent.mention_count.desc()).all()

    # Get latest metrics
    latest_metrics = db.query(ThemeMetrics).filter(
        ThemeMetrics.theme_cluster_id == theme_id
    ).order_by(ThemeMetrics.date.desc()).first()

    return ThemeDetailResponse(
        theme=ThemeClusterResponse.model_validate(cluster),
        constituents=[ThemeConstituentResponse.model_validate(c) for c in constituents],
        metrics=ThemeMetricsResponse.model_validate(latest_metrics) if latest_metrics else None
    )


@router.get("/{theme_id}/history")
async def get_theme_history(
    theme_id: int,
    days: int = Query(30, ge=1, le=180, description="Days of history"),
    db: Session = Depends(get_db)
):
    """
    Get historical metrics for a theme.
    """
    from datetime import timedelta

    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    cutoff_date = datetime.utcnow().date() - timedelta(days=days)

    metrics = db.query(ThemeMetrics).filter(
        ThemeMetrics.theme_cluster_id == theme_id,
        ThemeMetrics.date >= cutoff_date
    ).order_by(ThemeMetrics.date).all()

    return {
        "theme": cluster.name,
        "history": [
            {
                "date": m.date.isoformat(),
                "rank": m.rank,
                "momentum_score": m.momentum_score,
                "mention_velocity": m.mention_velocity,
                "basket_rs_vs_spy": m.basket_rs_vs_spy,
                "status": m.status,
            }
            for m in metrics
        ]
    }


@router.get("/{theme_id}/mentions", response_model=ThemeMentionsResponse)
async def get_theme_mentions(
    theme_id: int,
    limit: int = Query(50, ge=1, le=200, description="Max mentions to return"),
    db: Session = Depends(get_db)
):
    """
    Get news sources (content items) that mention this theme.

    Returns content items with their associated mention details including
    excerpt, sentiment, confidence, and source information.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    # Query mentions joined with content items
    mentions_query = db.query(ThemeMention, ContentItem).join(
        ContentItem, ThemeMention.content_item_id == ContentItem.id
    ).filter(
        ThemeMention.theme_cluster_id == theme_id
    ).order_by(
        ThemeMention.mentioned_at.desc()
    ).limit(limit)

    mentions = mentions_query.all()

    return ThemeMentionsResponse(
        theme_name=cluster.name,
        theme_id=theme_id,
        total_count=len(mentions),
        mentions=[
            ThemeMentionDetailResponse(
                mention_id=mention.id,
                content_title=content.title,
                content_url=content.url,
                author=content.author,
                published_at=content.published_at,
                excerpt=mention.excerpt,
                sentiment=mention.sentiment,
                confidence=mention.confidence,
                tickers=mention.tickers or [],
                source_type=mention.source_type,
                source_name=mention.source_name or content.source_name,
            )
            for mention, content in mentions
        ]
    )


# ==================== Correlation Discovery ====================

@router.get("/correlation/clusters", response_model=CorrelationDiscoveryResponse)
async def discover_correlation_clusters(
    correlation_threshold: float = Query(0.6, ge=0.3, le=0.95),
    min_cluster_size: int = Query(3, ge=2, le=20),
    db: Session = Depends(get_db)
):
    """
    Discover hidden themes via correlation clustering.

    Finds groups of stocks that move together regardless of industry classification.
    """
    service = ThemeCorrelationService(db)

    # Get correlation clusters
    clusters = service.discover_correlation_clusters(
        correlation_threshold=correlation_threshold,
        min_cluster_size=min_cluster_size
    )

    # Get cross-industry correlations
    cross_industry = service.find_cross_industry_correlations(
        min_correlation=correlation_threshold + 0.1
    )

    return CorrelationDiscoveryResponse(
        correlation_clusters=[CorrelationClusterResponse(**c) for c in clusters],
        cross_industry_pairs=[CrossIndustryPairResponse(**p) for p in cross_industry.get("cross_industry_pairs", [])],
        hub_stocks=cross_industry.get("hub_stocks", [])
    )


@router.get("/{theme_id}/validate", response_model=ThemeValidationResponse)
async def validate_theme(
    theme_id: int,
    min_correlation: float = Query(0.5, ge=0.2, le=0.9),
    db: Session = Depends(get_db)
):
    """
    Validate a theme by checking internal correlations.

    A valid theme has high average correlation between constituents.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    service = ThemeCorrelationService(db)
    result = service.validate_theme(theme_id, min_correlation)

    return ThemeValidationResponse(
        theme=cluster.name,
        **result
    )


@router.get("/{theme_id}/entrants")
async def find_theme_entrants(
    theme_id: int,
    correlation_threshold: float = Query(0.6, ge=0.4, le=0.9),
    db: Session = Depends(get_db)
):
    """
    Find stocks that may be joining this theme.

    Identifies stocks with high correlation to the theme basket
    that aren't yet constituents.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    service = ThemeCorrelationService(db)
    entrants = service.find_new_theme_entrants(
        theme_id,
        correlation_threshold=correlation_threshold
    )

    return {
        "theme": cluster.name,
        "potential_entrants": entrants
    }


@router.get("/{theme_id}/similar", response_model=SimilarThemesResponse)
async def find_similar_themes(
    theme_id: int,
    threshold: float = Query(0.75, ge=0.5, le=0.99, description="Minimum similarity threshold"),
    db: Session = Depends(get_db)
):
    """
    Find themes similar to the given theme using embedding similarity.

    Uses sentence-transformers embeddings and cosine similarity.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    service = ThemeMergingService(db)
    similar = service.find_similar_themes(theme_id, threshold=threshold)

    return SimilarThemesResponse(
        source_theme_id=theme_id,
        source_theme_name=cluster.name,
        threshold=threshold,
        similar_themes=[SimilarThemeResponse(**s) for s in similar]
    )


# ==================== Alert Operations ====================

@router.post("/alerts/check")
async def check_for_alerts(
    db: Session = Depends(get_db)
):
    """
    Check for and generate new alerts.
    """
    service = ThemeDiscoveryService(db)
    alerts = service.check_for_alerts()

    return {
        "new_alerts": len(alerts),
        "alerts": [{"type": a.alert_type, "title": a.title} for a in alerts]
    }


@router.post("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    Mark an alert as read.
    """
    alert = db.query(ThemeAlert).filter(ThemeAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_read = True
    alert.read_at = datetime.utcnow()
    db.commit()

    return {"status": "marked as read"}


# ==================== Theme Management ====================

@router.post("/create-from-cluster")
async def create_theme_from_cluster(
    name: str,
    symbols: list[str],
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Create a new theme from a correlation cluster.
    """
    service = ThemeCorrelationService(db)
    cluster = service.create_theme_from_cluster(
        symbols=symbols,
        name=name,
        description=description
    )

    return {
        "status": "created",
        "theme_id": cluster.id,
        "name": cluster.name,
        "num_constituents": len(symbols)
    }


@router.post("/{theme_id}/add-constituents")
async def add_theme_constituents(
    theme_id: int,
    symbols: list[str],
    db: Session = Depends(get_db)
):
    """
    Add stocks to a theme manually.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    added = 0
    for symbol in symbols:
        existing = db.query(ThemeConstituent).filter(
            ThemeConstituent.theme_cluster_id == theme_id,
            ThemeConstituent.symbol == symbol
        ).first()

        if not existing:
            constituent = ThemeConstituent(
                theme_cluster_id=theme_id,
                symbol=symbol,
                source="manual",
                confidence=1.0,
                mention_count=0,
                first_mentioned_at=datetime.utcnow(),
                last_mentioned_at=datetime.utcnow(),
            )
            db.add(constituent)
            added += 1

    db.commit()

    return {"status": "success", "added": added}


@router.delete("/{theme_id}")
async def deactivate_theme(
    theme_id: int,
    db: Session = Depends(get_db)
):
    """
    Deactivate a theme.
    """
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    cluster.is_active = False
    db.commit()

    return {"status": "deactivated", "theme": cluster.name}

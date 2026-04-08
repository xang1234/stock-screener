"""Themes API routes for merge/review workflows and write operations."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.theme import ThemeAlert, ThemeCluster, ThemeConstituent, ThemeMergeSuggestion
from ...schemas.theme import (
    CandidateThemeQueueItemResponse,
    CandidateThemeQueueResponse,
    CandidateThemeQueueSummaryBandResponse,
    CandidateThemeReviewItemResult,
    CandidateThemeReviewRequest,
    CandidateThemeReviewResponse,
    ConsolidationResultResponse,
    EmbeddingRefreshCampaignResponse,
    ManualReviewWaveRequest,
    ManualReviewWaveResponse,
    MergeActionResponse,
    MergePlanConfidenceBucketResponse,
    MergePlanDryRunResponse,
    MergePlanGroupResponse,
    MergePlanPairResponse,
    MergePlanWaveResponse,
    StrictAutoMergeWaveResponse,
    ThemeMergeHistoryListResponse,
    ThemeMergeSuggestionResponse,
    ThemeMergeSuggestionsResponse,
    ThemeRelationshipGraphEdgeResponse,
    ThemeRelationshipGraphNodeResponse,
    ThemeRelationshipGraphResponse,
)
from ...services.theme_correlation_service import ThemeCorrelationService
from ...services.theme_discovery_service import ThemeDiscoveryService
from ...services.theme_merging_service import ThemeMergingService

router = APIRouter()


@router.get("/merge-suggestions", response_model=ThemeMergeSuggestionsResponse)
async def get_merge_suggestions(
    status: Optional[str] = Query(None, description="Filter by status: pending, approved, rejected, auto_merged"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get theme merge suggestions from the queue."""
    service = ThemeMergingService(db)
    suggestions = service.get_merge_suggestions(status=status, limit=limit)
    return ThemeMergeSuggestionsResponse(
        total=len(suggestions),
        suggestions=[ThemeMergeSuggestionResponse(**suggestion) for suggestion in suggestions],
    )


@router.post("/merge-suggestions/{suggestion_id}/approve", response_model=MergeActionResponse)
async def approve_merge_suggestion(
    suggestion_id: int,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
):
    """Approve merge suggestion and execute merge."""
    suggestion = db.query(ThemeMergeSuggestion).filter(ThemeMergeSuggestion.id == suggestion_id).first()
    source_pipeline = None
    if suggestion is not None:
        source_pipeline = db.query(ThemeCluster.pipeline).filter(
            ThemeCluster.id == suggestion.source_cluster_id
        ).scalar()
    service = ThemeMergingService(db)
    result = service.approve_suggestion(suggestion_id, idempotency_key=idempotency_key)
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants(source_pipeline)
    return MergeActionResponse(**result)


@router.post("/merge-suggestions/{suggestion_id}/reject")
async def reject_merge_suggestion(
    suggestion_id: int,
    db: Session = Depends(get_db),
):
    """Reject a merge suggestion."""
    suggestion = db.query(ThemeMergeSuggestion).filter(ThemeMergeSuggestion.id == suggestion_id).first()
    source_pipeline = None
    if suggestion is not None:
        source_pipeline = db.query(ThemeCluster.pipeline).filter(
            ThemeCluster.id == suggestion.source_cluster_id
        ).scalar()
    service = ThemeMergingService(db)
    result = service.reject_suggestion(suggestion_id)
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants(source_pipeline)
    return result


@router.get("/merge-history", response_model=ThemeMergeHistoryListResponse)
async def get_merge_history(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get history of theme merges."""
    service = ThemeMergingService(db)
    history = service.get_merge_history(limit=limit)
    return ThemeMergeHistoryListResponse(
        total=len(history),
        history=history,
    )


@router.post("/consolidate", response_model=ConsolidationResultResponse)
async def run_theme_consolidation(
    dry_run: bool = Query(True, description="If true, report only without executing merges"),
    db: Session = Depends(get_db),
):
    """Run theme consolidation pipeline."""
    service = ThemeMergingService(db)
    result = service.run_consolidation(dry_run=dry_run)
    if not dry_run:
        from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

        safe_publish_themes_bootstrap_variants()
    return ConsolidationResultResponse(**result)


@router.post("/consolidate/async")
async def run_theme_consolidation_async(
    dry_run: bool = Query(False, description="If true, only report what would happen"),
    db: Session = Depends(get_db),
):
    """Run theme consolidation pipeline asynchronously via Celery."""
    del db
    from ...tasks.theme_discovery_tasks import consolidate_themes

    task = consolidate_themes.delay(dry_run=dry_run)
    return {
        "task_id": task.id,
        "status": "queued",
        "message": f"Theme consolidation queued (dry_run={dry_run})",
    }


@router.get("/merge-plan/dry-run", response_model=MergePlanDryRunResponse)
async def get_merge_plan_dry_run(
    limit_pairs: int = Query(120, ge=1, le=500, description="Max similar pairs to analyze"),
    pipeline: Optional[str] = Query(None, pattern="^(technical|fundamental)$"),
    db: Session = Depends(get_db),
):
    """Generate non-mutating duplicate-analysis merge plan."""
    service = ThemeMergingService(db)
    result = service.generate_dry_run_merge_plan(limit_pairs=limit_pairs, pipeline=pipeline)
    return MergePlanDryRunResponse(
        timestamp=result["timestamp"],
        total_pairs_analyzed=result["total_pairs_analyzed"],
        confidence_distribution=[MergePlanConfidenceBucketResponse(**row) for row in result["confidence_distribution"]],
        merge_groups=[MergePlanGroupResponse(**row) for row in result["merge_groups"]],
        waves=[MergePlanWaveResponse(**row) for row in result["waves"]],
        ambiguity_clusters=[MergePlanPairResponse(**row) for row in result["ambiguity_clusters"]],
        do_not_merge=[MergePlanPairResponse(**row) for row in result["do_not_merge"]],
        manual_review_recommendations=result["manual_review_recommendations"],
    )


@router.post("/embeddings/refresh-campaign", response_model=EmbeddingRefreshCampaignResponse)
async def run_embedding_refresh_campaign(
    pipeline: Optional[str] = Query(None, pattern="^(technical|fundamental)$"),
    refresh_batch_size: int = Query(100, ge=1, le=1000, description="Missing/outdated refresh batch size per pass"),
    stale_batch_size: int = Query(100, ge=1, le=1000, description="Stale recompute batch size"),
    stale_max_batches_per_pass: int = Query(10, ge=1, le=100, description="Max stale batches executed per pass"),
    max_passes: int = Query(6, ge=1, le=50, description="Max bounded passes for campaign"),
    min_coverage_ratio: float = Query(0.98, ge=0.0, le=1.0, description="Pre-merge coverage gate threshold"),
    min_freshness_ratio: float = Query(0.95, ge=0.0, le=1.0, description="Pre-merge freshness gate threshold"),
    db: Session = Depends(get_db),
):
    """Execute bounded embedding refresh + stale recompute campaign with gate reporting."""
    service = ThemeMergingService(db)
    result = service.run_embedding_refresh_campaign(
        pipeline=pipeline,
        refresh_batch_size=refresh_batch_size,
        stale_batch_size=stale_batch_size,
        stale_max_batches_per_pass=stale_max_batches_per_pass,
        max_passes=max_passes,
        min_coverage_ratio=min_coverage_ratio,
        min_freshness_ratio=min_freshness_ratio,
    )
    return EmbeddingRefreshCampaignResponse(**result)


@router.post("/merge-wave/strict-auto", response_model=StrictAutoMergeWaveResponse)
async def run_strict_auto_merge_wave(
    pipeline: Optional[str] = Query(None, pattern="^(technical|fundamental)$"),
    limit_pairs: int = Query(200, ge=1, le=1000, description="Maximum candidate pairs to evaluate"),
    dry_run: bool = Query(False, description="If true, evaluate strict eligibility without mutating data"),
    db: Session = Depends(get_db),
):
    """Run Wave-1 strict auto-merge for only highest-confidence identical pairs."""
    service = ThemeMergingService(db)
    result = service.run_strict_auto_merge_wave(
        pipeline=pipeline,
        limit_pairs=limit_pairs,
        dry_run=dry_run,
    )
    if not dry_run:
        from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

        safe_publish_themes_bootstrap_variants(pipeline)
    return StrictAutoMergeWaveResponse(**result)


@router.post("/merge-wave/manual-review", response_model=ManualReviewWaveResponse)
async def run_manual_review_wave(
    payload: ManualReviewWaveRequest,
    pipeline: Optional[str] = Query(None, pattern="^(technical|fundamental)$"),
    db: Session = Depends(get_db),
):
    """Run Wave-2 moderated manual-review processing."""
    service = ThemeMergingService(db)
    result = service.run_manual_review_wave(
        decisions=[row.model_dump() for row in payload.decisions],
        pipeline=pipeline,
        sla_target_hours=payload.sla_target_hours,
        queue_limit=payload.queue_limit,
        dry_run=payload.dry_run,
    )
    if not payload.dry_run:
        from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

        safe_publish_themes_bootstrap_variants(pipeline)
    return ManualReviewWaveResponse(**result)


@router.get("/candidates/queue", response_model=CandidateThemeQueueResponse)
async def get_candidate_theme_queue(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$"),
    db: Session = Depends(get_db),
):
    """List candidate themes queued for analyst lifecycle review."""
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    queue_rows, total_count = service.get_candidate_theme_queue(limit=limit, offset=offset)
    confidence_bands = [
        CandidateThemeQueueSummaryBandResponse(**row)
        for row in service.get_candidate_theme_confidence_bands()
    ]
    return CandidateThemeQueueResponse(
        total=total_count,
        items=[CandidateThemeQueueItemResponse(**row) for row in queue_rows],
        confidence_bands=confidence_bands,
    )


@router.post("/candidates/review", response_model=CandidateThemeReviewResponse)
async def review_candidate_themes(
    payload: CandidateThemeReviewRequest,
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$"),
    db: Session = Depends(get_db),
):
    """Bulk analyst triage for candidate themes."""
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    result = service.review_candidate_themes(
        theme_cluster_ids=payload.theme_cluster_ids,
        action=payload.action,
        actor=payload.actor or "analyst",
        note=payload.note,
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Candidate review failed"))
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants(pipeline)
    return CandidateThemeReviewResponse(
        success=True,
        action=result["action"],
        updated=result["updated"],
        skipped=result["skipped"],
        results=[CandidateThemeReviewItemResult(**row) for row in result["results"]],
    )


@router.get("/relationship-graph", response_model=ThemeRelationshipGraphResponse)
async def get_relationship_graph(
    theme_cluster_id: int = Query(..., ge=1, description="Root theme cluster id"),
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$"),
    limit: int = Query(120, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Return relationship graph nodes/edges centered on the requested theme."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_cluster_id).first()
    if cluster is None:
        raise HTTPException(status_code=404, detail="Theme not found")
    if (cluster.pipeline or "technical") != pipeline:
        raise HTTPException(status_code=404, detail="Theme not found for selected pipeline")
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    graph = service.get_theme_relationship_graph(theme_cluster_id, limit=limit)
    return ThemeRelationshipGraphResponse(
        theme_cluster_id=theme_cluster_id,
        total_nodes=len(graph["nodes"]),
        total_edges=len(graph["edges"]),
        nodes=[ThemeRelationshipGraphNodeResponse(**node) for node in graph["nodes"]],
        edges=[ThemeRelationshipGraphEdgeResponse(**edge) for edge in graph["edges"]],
    )


@router.post("/alerts/check")
async def check_for_alerts(
    db: Session = Depends(get_db),
):
    """Check for and generate new alerts."""
    service = ThemeDiscoveryService(db)
    alerts = service.check_for_alerts()
    return {
        "new_alerts": len(alerts),
        "alerts": [{"type": alert.alert_type, "title": alert.title} for alert in alerts],
    }


@router.post("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: int,
    db: Session = Depends(get_db),
):
    """Mark an alert as read."""
    alert = db.query(ThemeAlert).filter(ThemeAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.is_read = True
    alert.read_at = datetime.utcnow()
    db.commit()
    return {"status": "marked as read"}


@router.post("/create-from-cluster")
async def create_theme_from_cluster(
    name: str,
    symbols: list[str],
    description: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Create a new theme from a correlation cluster."""
    service = ThemeCorrelationService(db)
    cluster = service.create_theme_from_cluster(
        symbols=symbols,
        name=name,
        description=description,
    )
    return {
        "status": "created",
        "theme_id": cluster.id,
        "name": cluster.display_name,
        "num_constituents": len(symbols),
    }


@router.post("/{theme_id}/add-constituents")
async def add_theme_constituents(
    theme_id: int,
    symbols: list[str],
    db: Session = Depends(get_db),
):
    """Add stocks to a theme manually."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    added = 0
    for symbol in symbols:
        existing = db.query(ThemeConstituent).filter(
            ThemeConstituent.theme_cluster_id == theme_id,
            ThemeConstituent.symbol == symbol,
        ).first()
        if existing:
            continue

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
    db: Session = Depends(get_db),
):
    """Deactivate a theme."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    cluster.is_active = False
    db.commit()
    return {"status": "deactivated", "theme": cluster.display_name}


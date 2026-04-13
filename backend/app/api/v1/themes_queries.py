"""Themes API routes for read/query surfaces."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.theme import (
    ContentItem,
    ThemeAlert,
    ThemeCluster,
    ThemeConstituent,
    ThemeMention,
    ThemeMetrics,
)
from ...schemas.theme import (
    AlertsResponse,
    CorrelationClusterResponse,
    CorrelationDiscoveryResponse,
    CrossIndustryPairResponse,
    EmergingThemeResponse,
    EmergingThemesResponse,
    NewEntrantResponse,
    SimilarThemeResponse,
    SimilarThemesResponse,
    ThemeAlertResponse,
    ThemeConstituentResponse,
    ThemeDetailResponse,
    ThemeLifecycleTransitionHistoryResponse,
    ThemeLifecycleTransitionResponse,
    ThemeMatchBandBucketResponse,
    ThemeMatchDecisionReasonDistributionResponse,
    ThemeMatchMethodDistributionResponse,
    ThemeMatchTelemetryResponse,
    ThemeMatchTelemetrySliceResponse,
    ThemeMentionsResponse,
    ThemeMentionDetailResponse,
    ThemeMetricsResponse,
    ThemeRankingItem,
    ThemeRankingsResponse,
    ThemeReference,
    ThemeRelationshipResponse,
    ThemeValidationResponse,
)
from ...schemas.ui_view_snapshot import UISnapshotEnvelope
from ...services.theme_correlation_service import ThemeCorrelationService
from ...services.theme_discovery_service import ThemeDiscoveryService
from ...services.theme_merging_service import ThemeMergingService
from ...wiring.bootstrap import get_ui_snapshot_service
from .themes_common import parse_csv_values, safe_theme_cluster_response

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/bootstrap", response_model=UISnapshotEnvelope)
def get_themes_bootstrap(
    pipeline: str = Query("technical", pattern="^(technical|fundamental)$"),
    theme_view: str = Query("grouped", pattern="^(grouped|flat)$"),
    snapshot_service=Depends(get_ui_snapshot_service),
):
    """Return published themes bootstrap snapshot."""
    snapshot = snapshot_service.get_themes_bootstrap(pipeline=pipeline, theme_view=theme_view)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No published themes bootstrap snapshot is available")
    return UISnapshotEnvelope(**snapshot.to_dict())


@router.get("/pipelines")
def get_available_pipelines():
    """Return available theme pipelines."""
    from ...config.pipeline_config import get_all_pipelines

    return {"pipelines": get_all_pipelines()}


@router.get("/rankings", response_model=ThemeRankingsResponse)
def get_theme_rankings(
    limit: int = Query(20, ge=1, le=500, description="Number of themes to return"),
    offset: int = Query(0, ge=0, description="Number of themes to skip for pagination"),
    status: Optional[str] = Query(None, description="Filter by status: emerging, trending, fading, dormant"),
    source_types: Optional[str] = Query(None, description="Comma-separated source types: substack,twitter,news,reddit"),
    lifecycle_states: Optional[str] = Query(None, description="Comma-separated lifecycle states"),
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    recalculate: bool = Query(False, description="Force recalculation of metrics"),
    db: Session = Depends(get_db),
):
    """Get current theme rankings sorted by momentum score."""
    service = ThemeDiscoveryService(db, pipeline=pipeline)

    if recalculate:
        service.update_all_theme_metrics()
    else:
        themes_without_metrics = db.query(ThemeCluster).filter(
            ThemeCluster.is_active == True,
            ThemeCluster.is_l1 == False,
            ~ThemeCluster.id.in_(db.query(ThemeMetrics.theme_cluster_id).distinct()),
        ).count()
        if themes_without_metrics > 0:
            logger.info("Auto-calculating metrics for %s themes without metrics", themes_without_metrics)
            service.update_all_theme_metrics()

    source_types_list = parse_csv_values(source_types)
    lifecycle_states_list = parse_csv_values(lifecycle_states)

    rankings, total_count = service.get_theme_rankings(
        limit=limit,
        status_filter=status,
        source_types_filter=source_types_list,
        lifecycle_states_filter=lifecycle_states_list,
        offset=offset,
    )

    if not rankings:
        return ThemeRankingsResponse(
            date=None,
            total_themes=total_count,
            pipeline=pipeline,
            rankings=[],
        )

    return ThemeRankingsResponse(
        date=datetime.utcnow().date().isoformat(),
        total_themes=total_count,
        pipeline=pipeline,
        rankings=[ThemeRankingItem(**ranking) for ranking in rankings],
    )


@router.get("/emerging", response_model=EmergingThemesResponse)
def get_emerging_themes(
    min_velocity: float = Query(1.5, description="Minimum mention velocity"),
    min_mentions: int = Query(3, description="Minimum mentions in 7 days"),
    lifecycle_states: Optional[str] = Query(None, description="Comma-separated lifecycle states"),
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    db: Session = Depends(get_db),
):
    """Discover newly emerging themes."""
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    lifecycle_states_list = parse_csv_values(lifecycle_states)
    themes = service.discover_emerging_themes(
        min_velocity=min_velocity,
        min_mentions=min_mentions,
        lifecycle_states_filter=lifecycle_states_list,
    )
    return EmergingThemesResponse(
        count=len(themes),
        themes=[EmergingThemeResponse(**theme) for theme in themes],
    )


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get theme alerts (excluding dismissed alerts)."""
    query = db.query(ThemeAlert).filter(ThemeAlert.is_dismissed == False)
    if unread_only:
        query = query.filter(ThemeAlert.is_read == False)
    query = query.order_by(ThemeAlert.triggered_at.desc()).limit(limit)
    alerts = query.all()

    unread_count = db.query(ThemeAlert).filter(
        ThemeAlert.is_read == False,
        ThemeAlert.is_dismissed == False,
    ).count()

    return AlertsResponse(
        total=len(alerts),
        unread=unread_count,
        alerts=[ThemeAlertResponse.model_validate(alert) for alert in alerts],
    )


@router.get("/lifecycle-transitions", response_model=ThemeLifecycleTransitionHistoryResponse)
def get_lifecycle_transitions(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    pipeline: str = Query("technical", description="Pipeline: technical or fundamental"),
    theme_cluster_id: Optional[int] = Query(None, description="Optional theme cluster ID filter"),
    to_state: Optional[str] = Query(None, description="Optional target lifecycle state filter"),
    db: Session = Depends(get_db),
):
    """Get lifecycle transition audit history with decision context."""
    service = ThemeDiscoveryService(db, pipeline=pipeline)
    history, total_count = service.get_lifecycle_transition_history(
        limit=limit,
        offset=offset,
        theme_cluster_id=theme_cluster_id,
        to_state=to_state,
    )
    return ThemeLifecycleTransitionHistoryResponse(
        total=total_count,
        transitions=[ThemeLifecycleTransitionResponse(**row) for row in history],
    )


@router.post("/alerts/{alert_id}/dismiss")
def dismiss_alert(
    alert_id: int,
    db: Session = Depends(get_db),
):
    """Dismiss (soft delete) an alert."""
    alert = db.query(ThemeAlert).filter(ThemeAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.is_dismissed = True
    db.commit()
    from ...services.ui_snapshot_service import safe_publish_themes_bootstrap_variants

    safe_publish_themes_bootstrap_variants()
    return {"status": "dismissed", "alert_id": alert_id}


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _confidence_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.40:
        return "0.00-0.39"
    if value < 0.70:
        return "0.40-0.69"
    if value < 0.85:
        return "0.70-0.84"
    return "0.85-1.00"


def _score_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "0.00-0.24"
    if value < 0.50:
        return "0.25-0.49"
    if value < 0.75:
        return "0.50-0.74"
    if value < 0.90:
        return "0.75-0.89"
    return "0.90-1.00"


def _build_band_buckets(
    rows: list[dict[str, object]],
    *,
    band_selector,
) -> list[ThemeMatchBandBucketResponse]:
    if not rows:
        return []

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[band_selector(row)].append(row)

    ordered_bands = sorted(grouped.keys())
    output: list[ThemeMatchBandBucketResponse] = []
    total = len(rows)
    for band in ordered_bands:
        bucket_rows = grouped[band]
        bucket_total = len(bucket_rows)
        new_cluster_count = sum(1 for row in bucket_rows if row["match_method"] == "create_new_cluster")
        attach_count = sum(
            1
            for row in bucket_rows
            if row["match_method"] not in {"create_new_cluster", "unknown"}
        )
        output.append(
            ThemeMatchBandBucketResponse(
                band=band,
                count=bucket_total,
                pct=_safe_rate(bucket_total, total),
                new_cluster_rate=_safe_rate(new_cluster_count, bucket_total),
                attach_rate=_safe_rate(attach_count, bucket_total),
            )
        )
    return output


def _build_match_slice(key: str, rows: list[dict[str, object]]) -> ThemeMatchTelemetrySliceResponse:
    total = len(rows)
    new_cluster_count = sum(1 for row in rows if row["match_method"] == "create_new_cluster")
    attach_count = sum(
        1
        for row in rows
        if row["match_method"] not in {"create_new_cluster", "unknown"}
    )

    method_counter = Counter(str(row["match_method"] or "unknown") for row in rows)
    method_distribution = [
        ThemeMatchMethodDistributionResponse(
            method=method,
            count=count,
            pct=_safe_rate(count, total),
        )
        for method, count in sorted(method_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    reason_counter = Counter(str(row["match_fallback_reason"] or "none") for row in rows)
    reason_distribution = [
        ThemeMatchDecisionReasonDistributionResponse(
            reason=reason,
            count=count,
            pct=_safe_rate(count, total),
        )
        for reason, count in sorted(reason_counter.items(), key=lambda item: (-item[1], item[0]))
    ]

    confidence_bands = _build_band_buckets(rows, band_selector=lambda row: _confidence_band(row["confidence"]))
    score_bands = _build_band_buckets(rows, band_selector=lambda row: _score_band(row["match_score"]))

    return ThemeMatchTelemetrySliceResponse(
        key=key,
        total_mentions=total,
        new_cluster_count=new_cluster_count,
        attach_count=attach_count,
        new_cluster_rate=_safe_rate(new_cluster_count, total),
        attach_rate=_safe_rate(attach_count, total),
        method_distribution=method_distribution,
        decision_reason_distribution=reason_distribution,
        confidence_bands=confidence_bands,
        score_bands=score_bands,
    )


@router.get("/matching/telemetry", response_model=ThemeMatchTelemetryResponse)
def get_matching_telemetry(
    days: Annotated[int, Query(ge=1, le=365, description="Rolling window in days")] = 30,
    pipeline: Annotated[Optional[str], Query(description="Filter by pipeline: technical or fundamental")] = None,
    source_type: Annotated[Optional[str], Query(description="Filter by source type")] = None,
    threshold_version: Annotated[Optional[str], Query(description="Filter by threshold version")] = None,
    db: Session = Depends(get_db),
):
    """Aggregate matcher telemetry for ops/model tuning dashboards."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    query = db.query(
        ThemeMention.match_method,
        ThemeMention.match_fallback_reason,
        ThemeMention.match_score,
        ThemeMention.confidence,
        ThemeMention.threshold_version,
        ThemeMention.source_type,
        ThemeMention.mentioned_at,
    ).filter(ThemeMention.mentioned_at >= cutoff)

    if pipeline:
        query = query.filter(ThemeMention.pipeline == pipeline)
    if source_type:
        query = query.filter(ThemeMention.source_type == source_type)
    if threshold_version:
        if threshold_version == "unknown":
            query = query.filter(ThemeMention.threshold_version.is_(None))
        else:
            query = query.filter(ThemeMention.threshold_version == threshold_version)

    records = query.all()
    rows: list[dict[str, object]] = [
        {
            "match_method": record.match_method or "unknown",
            "match_fallback_reason": record.match_fallback_reason,
            "match_score": record.match_score,
            "confidence": record.confidence,
            "threshold_version": record.threshold_version or "unknown",
            "source_type": record.source_type,
            "mentioned_at": record.mentioned_at,
        }
        for record in records
    ]

    if not rows:
        return ThemeMatchTelemetryResponse(
            window_days=days,
            start_at=None,
            end_at=None,
            pipeline=pipeline,
            source_type=source_type,
            threshold_version=threshold_version,
            total_mentions=0,
            new_cluster_count=0,
            attach_count=0,
            new_cluster_rate=0.0,
            attach_rate=0.0,
            method_distribution=[],
            decision_reason_distribution=[],
            confidence_bands=[],
            score_bands=[],
            by_threshold_version=[],
            by_source_type=[],
        )

    overall = _build_match_slice("overall", rows)
    by_threshold: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_threshold[str(row["threshold_version"])].append(row)
        by_source[str(row["source_type"])].append(row)

    by_threshold_version = [
        _build_match_slice(key, grouped_rows)
        for key, grouped_rows in sorted(by_threshold.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    by_source_type = [
        _build_match_slice(key, grouped_rows)
        for key, grouped_rows in sorted(by_source.items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    mentioned_at_values = [row["mentioned_at"] for row in rows if row["mentioned_at"] is not None]
    start_at = min(mentioned_at_values) if mentioned_at_values else None
    end_at = max(mentioned_at_values) if mentioned_at_values else None

    return ThemeMatchTelemetryResponse(
        window_days=days,
        start_at=start_at,
        end_at=end_at,
        pipeline=pipeline,
        source_type=source_type,
        threshold_version=threshold_version,
        total_mentions=overall.total_mentions,
        new_cluster_count=overall.new_cluster_count,
        attach_count=overall.attach_count,
        new_cluster_rate=overall.new_cluster_rate,
        attach_rate=overall.attach_rate,
        method_distribution=overall.method_distribution,
        decision_reason_distribution=overall.decision_reason_distribution,
        confidence_bands=overall.confidence_bands,
        score_bands=overall.score_bands,
        by_threshold_version=by_threshold_version,
        by_source_type=by_source_type,
    )


@router.get("/{theme_id}", response_model=ThemeDetailResponse)
def get_theme_detail(
    theme_id: int,
    db: Session = Depends(get_db),
):
    """Get detailed information for a specific theme."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    constituents = db.query(ThemeConstituent).filter(
        ThemeConstituent.theme_cluster_id == theme_id,
        ThemeConstituent.is_active == True,
    ).order_by(ThemeConstituent.mention_count.desc()).all()

    latest_metrics = db.query(ThemeMetrics).filter(
        ThemeMetrics.theme_cluster_id == theme_id
    ).order_by(ThemeMetrics.date.desc()).first()

    service = ThemeDiscoveryService(db, pipeline=cluster.pipeline or "technical")
    relationships = service.get_theme_relationships(theme_id, limit=50)

    return ThemeDetailResponse(
        theme=safe_theme_cluster_response(cluster),
        constituents=[ThemeConstituentResponse.model_validate(c) for c in constituents],
        metrics=ThemeMetricsResponse.model_validate(latest_metrics) if latest_metrics else None,
        relationships=[ThemeRelationshipResponse(**row) for row in relationships],
    )


@router.get("/{theme_id}/history")
def get_theme_history(
    theme_id: int,
    days: int = Query(30, ge=1, le=180, description="Days of history"),
    db: Session = Depends(get_db),
):
    """Get historical metrics for a theme."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    cutoff_date = datetime.utcnow().date() - timedelta(days=days)
    metrics = db.query(ThemeMetrics).filter(
        ThemeMetrics.theme_cluster_id == theme_id,
        ThemeMetrics.date >= cutoff_date,
    ).order_by(ThemeMetrics.date).all()

    return {
        "theme": cluster.display_name,
        "history": [
            {
                "date": metric.date.isoformat(),
                "rank": metric.rank,
                "momentum_score": metric.momentum_score,
                "mention_velocity": metric.mention_velocity,
                "basket_rs_vs_spy": metric.basket_rs_vs_spy,
                "status": metric.status,
            }
            for metric in metrics
        ],
    }


@router.get("/{theme_id}/mentions", response_model=ThemeMentionsResponse)
def get_theme_mentions(
    theme_id: int,
    limit: int = Query(50, ge=1, le=200, description="Max mentions to return"),
    db: Session = Depends(get_db),
):
    """Get content items that mention this theme."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    mentions = db.query(ThemeMention, ContentItem).join(
        ContentItem, ThemeMention.content_item_id == ContentItem.id
    ).filter(
        ThemeMention.theme_cluster_id == theme_id
    ).order_by(
        ThemeMention.mentioned_at.desc()
    ).limit(limit).all()

    return ThemeMentionsResponse(
        theme_name=cluster.display_name,
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
                source_language=content.source_language,
                translated_excerpt=mention.translated_excerpt,
                translated_raw_theme=mention.translated_raw_theme,
                # Explicit None check: the detection-only path (T7.2) may
                # write {} on the mention to signal "translation not yet
                # done", and Python's `or` would silently swap that for
                # stale content-level metadata.
                translation_metadata=(
                    mention.translation_metadata
                    if mention.translation_metadata is not None
                    else content.translation_metadata
                ),
            )
            for mention, content in mentions
        ],
    )


@router.get("/correlation/clusters", response_model=CorrelationDiscoveryResponse)
def discover_correlation_clusters(
    correlation_threshold: float = Query(0.6, ge=0.3, le=0.95),
    min_cluster_size: int = Query(3, ge=2, le=20),
    db: Session = Depends(get_db),
):
    """Discover hidden themes via correlation clustering."""
    service = ThemeCorrelationService(db)
    clusters = service.discover_correlation_clusters(
        correlation_threshold=correlation_threshold,
        min_cluster_size=min_cluster_size,
    )
    cross_industry = service.find_cross_industry_correlations(
        min_correlation=correlation_threshold + 0.1
    )

    return CorrelationDiscoveryResponse(
        correlation_clusters=[CorrelationClusterResponse(**cluster) for cluster in clusters],
        cross_industry_pairs=[
            CrossIndustryPairResponse(**pair)
            for pair in cross_industry.get("cross_industry_pairs", [])
        ],
        hub_stocks=cross_industry.get("hub_stocks", []),
    )


@router.get("/{theme_id}/validate", response_model=ThemeValidationResponse)
def validate_theme(
    theme_id: int,
    min_correlation: float = Query(0.5, ge=0.2, le=0.9),
    db: Session = Depends(get_db),
):
    """Validate a theme by checking internal correlations."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    service = ThemeCorrelationService(db)
    result = service.validate_theme(theme_id, min_correlation)
    return ThemeValidationResponse(
        theme=cluster.display_name,
        **result,
    )


@router.get("/{theme_id}/entrants")
def find_theme_entrants(
    theme_id: int,
    correlation_threshold: float = Query(0.6, ge=0.4, le=0.9),
    db: Session = Depends(get_db),
):
    """Find stocks that may be joining this theme."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    service = ThemeCorrelationService(db)
    entrants = service.find_new_theme_entrants(
        theme_id,
        correlation_threshold=correlation_threshold,
    )

    return {
        "theme": cluster.display_name,
        "potential_entrants": entrants,
    }


@router.get("/{theme_id}/similar", response_model=SimilarThemesResponse)
def find_similar_themes(
    theme_id: int,
    threshold: float = Query(0.75, ge=0.5, le=0.99, description="Minimum similarity threshold"),
    db: Session = Depends(get_db),
):
    """Find themes similar to the given theme using embedding similarity."""
    cluster = db.query(ThemeCluster).filter(ThemeCluster.id == theme_id).first()
    if not cluster:
        raise HTTPException(status_code=404, detail="Theme not found")

    service = ThemeMergingService(db)
    similar = service.find_similar_themes(theme_id, threshold=threshold)

    return SimilarThemesResponse(
        source_theme_id=theme_id,
        source_theme_name=cluster.display_name,
        threshold=threshold,
        similar_themes=[SimilarThemeResponse(**similar_row) for similar_row in similar],
    )

"""Helpers for pipeline-scoped state reconciliation and observability."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session, aliased

from ..models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ThemeMention,
    ThemeCluster,
    ThemeMergeSuggestion,
)

VALID_PIPELINES = ("technical", "fundamental")
PIPELINE_OBSERVABILITY_RUNBOOK_URL = "/docs/theme_identity/e8_t4_pipeline_observability_runbook.md"


def normalize_pipelines(value: Any) -> list[str]:
    """Normalize source pipeline payloads into known pipeline names."""
    if value is None:
        return list(VALID_PIPELINES)

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [segment.strip() for segment in value.split(",") if segment.strip()]

    if not isinstance(value, (list, tuple, set)):
        return list(VALID_PIPELINES)

    seen: set[str] = set()
    normalized: list[str] = []
    for raw in value:
        pipeline = str(raw).strip().lower()
        if pipeline in VALID_PIPELINES and pipeline not in seen:
            normalized.append(pipeline)
            seen.add(pipeline)

    return normalized or list(VALID_PIPELINES)


def validate_pipeline_selection(value: Any) -> list[str]:
    """
    Validate operator-provided pipeline assignments.

    Raises ValueError when the provided value is empty or includes unknown
    pipeline names so API callers fail fast instead of silently broadening scope.
    """
    if value is None:
        raise ValueError("pipelines is required")

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            value = [segment.strip() for segment in value.split(",") if segment.strip()]

    if not isinstance(value, (list, tuple, set)):
        raise ValueError("pipelines must be a list of pipeline names")

    seen: set[str] = set()
    normalized: list[str] = []
    invalid: list[str] = []
    for raw in value:
        pipeline = str(raw).strip().lower()
        if pipeline in VALID_PIPELINES:
            if pipeline not in seen:
                normalized.append(pipeline)
                seen.add(pipeline)
        else:
            invalid.append(str(raw))

    if invalid:
        raise ValueError(
            f"Invalid pipelines: {invalid}. Allowed values are: {list(VALID_PIPELINES)}"
        )
    if not normalized:
        raise ValueError("At least one valid pipeline must be provided")

    return normalized


def reconcile_source_pipeline_change(
    db: Session,
    source_id: int,
    old_pipelines: list[str],
    new_pipelines: list[str],
    chunk_size: int = 500,
    commit_each_chunk: bool = True,
) -> dict[str, int | list[str]]:
    """
    Reconcile pipeline-state rows when source pipeline assignments change.

    Behavior is intentionally non-destructive:
    - Existing theme mentions are preserved unchanged.
    - Added pipelines receive missing pending state rows.
    - Removed pipelines mark stale in-progress rows retryable so they do not
      remain indefinitely in_progress.
    """
    old_norm = normalize_pipelines(old_pipelines)
    new_norm = normalize_pipelines(new_pipelines)
    old_set = set(old_norm)
    new_set = set(new_norm)

    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)

    if not added and not removed:
        return {
            "added_pipelines": [],
            "removed_pipelines": [],
            "created_pending_rows": 0,
            "existing_conflicts": 0,
            "removed_in_progress_rows": 0,
        }

    created_pending = 0
    existing_conflicts = 0
    removed_in_progress = 0

    cursor = 0
    while True:
        chunk_rows = db.query(ContentItem.id).filter(
            ContentItem.source_id == source_id,
            ContentItem.id > cursor,
        ).order_by(ContentItem.id.asc()).limit(chunk_size).all()
        if not chunk_rows:
            break

        item_ids = [row[0] for row in chunk_rows]
        cursor = item_ids[-1]

        if added:
            existing_pairs = {
                (row[0], row[1])
                for row in db.query(
                    ContentItemPipelineState.content_item_id,
                    ContentItemPipelineState.pipeline,
                ).filter(
                    ContentItemPipelineState.content_item_id.in_(item_ids),
                    ContentItemPipelineState.pipeline.in_(added),
                ).all()
            }
            for item_id in item_ids:
                for pipeline in added:
                    key = (item_id, pipeline)
                    if key in existing_pairs:
                        existing_conflicts += 1
                        continue
                    db.add(
                        ContentItemPipelineState(
                            content_item_id=item_id,
                            pipeline=pipeline,
                            status="pending",
                            attempt_count=0,
                        )
                    )
                    created_pending += 1

        if removed:
            removed_in_progress += db.query(ContentItemPipelineState).filter(
                ContentItemPipelineState.content_item_id.in_(item_ids),
                ContentItemPipelineState.pipeline.in_(removed),
                ContentItemPipelineState.status == "in_progress",
            ).update(
                {
                    ContentItemPipelineState.status: "failed_retryable",
                    ContentItemPipelineState.error_code: "source_pipeline_removed",
                    ContentItemPipelineState.error_message: (
                        "Source pipeline assignment removed while item was in progress."
                    ),
                    ContentItemPipelineState.last_attempt_at: datetime.utcnow(),
                },
                synchronize_session=False,
            )

        if commit_each_chunk:
            db.commit()

    if not commit_each_chunk:
        db.commit()

    return {
        "added_pipelines": added,
        "removed_pipelines": removed,
        "created_pending_rows": created_pending,
        "existing_conflicts": existing_conflicts,
        "removed_in_progress_rows": removed_in_progress,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0}
    sorted_values = sorted(values)

    def _pct(p: float) -> float:
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        idx = p * (len(sorted_values) - 1)
        low = int(idx)
        high = min(low + 1, len(sorted_values) - 1)
        frac = idx - low
        return float(sorted_values[low] * (1 - frac) + sorted_values[high] * frac)

    return {"p50": round(_pct(0.5), 2), "p95": round(_pct(0.95), 2)}


def compute_pipeline_state_health(
    db: Session,
    pipeline: Optional[str] = None,
    max_age_days: int = 30,
) -> dict[str, Any]:
    """
    Compute per-pipeline health and drift metrics for pipeline state.
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(days=max_age_days)
    pipelines = [pipeline] if pipeline else list(VALID_PIPELINES)

    health_rows: list[dict[str, Any]] = []
    for p in pipelines:
        status_counts = {
            row[0]: int(row[1])
            for row in db.query(
                ContentItemPipelineState.status,
                func.count(ContentItemPipelineState.id),
            ).join(
                ContentItem,
                ContentItem.id == ContentItemPipelineState.content_item_id,
            ).filter(
                ContentItemPipelineState.pipeline == p,
                ContentItem.published_at >= cutoff,
            ).group_by(
                ContentItemPipelineState.status
            ).all()
        }

        pending_count = status_counts.get("pending", 0)
        retryable_count = status_counts.get("failed_retryable", 0)
        processed_count = status_counts.get("processed", 0)
        failed_terminal_count = status_counts.get("failed_terminal", 0)
        in_progress_count = status_counts.get("in_progress", 0)

        pending_created = [
            row[0]
            for row in db.query(ContentItemPipelineState.created_at).join(
                ContentItem,
                ContentItem.id == ContentItemPipelineState.content_item_id,
            ).filter(
                ContentItemPipelineState.pipeline == p,
                ContentItemPipelineState.status == "pending",
                ContentItem.published_at >= cutoff,
                ContentItemPipelineState.created_at.isnot(None),
            ).all()
            if row[0] is not None
        ]
        pending_ages_hours = [
            max((now - created_at).total_seconds() / 3600.0, 0.0)
            for created_at in pending_created
        ]
        pending_age = _percentiles(pending_ages_hours)

        retryable_recent = db.query(func.count(ContentItemPipelineState.id)).join(
            ContentItem,
            ContentItem.id == ContentItemPipelineState.content_item_id,
        ).filter(
            ContentItemPipelineState.pipeline == p,
            ContentItemPipelineState.status == "failed_retryable",
            ContentItem.published_at >= cutoff,
            ContentItemPipelineState.updated_at >= now - timedelta(hours=24),
        ).scalar() or 0

        retryable_previous = db.query(func.count(ContentItemPipelineState.id)).join(
            ContentItem,
            ContentItem.id == ContentItemPipelineState.content_item_id,
        ).filter(
            ContentItemPipelineState.pipeline == p,
            ContentItemPipelineState.status == "failed_retryable",
            ContentItem.published_at >= cutoff,
            ContentItemPipelineState.updated_at >= now - timedelta(hours=48),
            ContentItemPipelineState.updated_at < now - timedelta(hours=24),
        ).scalar() or 0

        failed_total = retryable_count + failed_terminal_count
        parse_failures = db.query(func.count(ContentItemPipelineState.id)).join(
            ContentItem,
            ContentItem.id == ContentItemPipelineState.content_item_id,
        ).filter(
            ContentItemPipelineState.pipeline == p,
            ContentItem.published_at >= cutoff,
            ContentItemPipelineState.status.in_(["failed_retryable", "failed_terminal"]),
            ContentItemPipelineState.error_code.isnot(None),
            (
                func.lower(ContentItemPipelineState.error_code).like("%json%")
            ) | (
                func.lower(ContentItemPipelineState.error_code).like("%parse%")
            ) | (
                func.lower(ContentItemPipelineState.error_code).like("%schema%")
            ),
        ).scalar() or 0

        processed_without_mentions = db.query(func.count(ContentItemPipelineState.id)).join(
            ContentItem,
            ContentItem.id == ContentItemPipelineState.content_item_id,
        ).outerjoin(
            ThemeMention,
            and_(
                ThemeMention.content_item_id == ContentItemPipelineState.content_item_id,
                ThemeMention.pipeline == ContentItemPipelineState.pipeline,
            ),
        ).filter(
            ContentItemPipelineState.pipeline == p,
            ContentItemPipelineState.status == "processed",
            ContentItem.published_at >= cutoff,
            ThemeMention.id.is_(None),
        ).scalar() or 0

        processed_without_mentions_ratio = (
            float(processed_without_mentions) / float(processed_count)
            if processed_count > 0
            else 0.0
        )

        retryable_growth_delta = int(retryable_recent - retryable_previous)
        retryable_growth_ratio = (
            float(retryable_recent) / float(retryable_previous)
            if retryable_previous > 0
            else (1.0 if retryable_recent > 0 else 0.0)
        )

        health_rows.append(
            {
                "pipeline": p,
                "window_days": max_age_days,
                "counts": {
                    "pending": pending_count,
                    "in_progress": in_progress_count,
                    "processed": processed_count,
                    "failed_retryable": retryable_count,
                    "failed_terminal": failed_terminal_count,
                    "parse_error": int(parse_failures),
                    "processed_without_mentions": int(processed_without_mentions),
                },
                "pending_age_hours": pending_age,
                "retryable_growth": {
                    "last_24h": int(retryable_recent),
                    "previous_24h": int(retryable_previous),
                    "delta": retryable_growth_delta,
                    "ratio": round(retryable_growth_ratio, 3),
                },
                "rates": {
                    "parse_failure_rate": round(
                        float(parse_failures) / float(failed_total),
                        4,
                    ) if failed_total > 0 else 0.0,
                    "processed_without_mentions_ratio": round(
                        processed_without_mentions_ratio,
                        4,
                    ),
                },
            }
        )

    return {
        "generated_at": now.isoformat(),
        "window_days": max_age_days,
        "pipelines": health_rows,
    }


def compute_pipeline_observability(
    db: Session,
    pipeline: str,
    max_age_days: int = 30,
) -> dict[str, Any]:
    """
    Build dashboard-ready observability metrics and thresholded alerts.
    """
    if pipeline not in VALID_PIPELINES:
        raise ValueError(f"pipeline must be one of {list(VALID_PIPELINES)}")

    health = compute_pipeline_state_health(db=db, pipeline=pipeline, max_age_days=max_age_days)
    pipeline_health = health["pipelines"][0] if health["pipelines"] else {
        "counts": {},
        "rates": {},
        "retryable_growth": {},
    }
    counts = pipeline_health.get("counts", {})
    rates = pipeline_health.get("rates", {})
    retryable_growth = pipeline_health.get("retryable_growth", {})
    now = datetime.utcnow()
    cutoff = now - timedelta(days=max_age_days)

    method_rows = db.query(
        ThemeMention.match_method,
        func.count(ThemeMention.id),
    ).filter(
        ThemeMention.pipeline == pipeline,
        ThemeMention.mentioned_at >= cutoff,
    ).group_by(
        ThemeMention.match_method
    ).all()
    total_mentions = int(sum(int(row[1]) for row in method_rows))
    method_mix = {
        str((row[0] or "unknown")): round(
            float(row[1]) / float(total_mentions),
            4,
        )
        for row in method_rows
        if total_mentions > 0
    }
    new_cluster_count = int(
        sum(int(row[1]) for row in method_rows if (row[0] or "unknown") == "create_new_cluster")
    )
    new_cluster_rate = (
        round(float(new_cluster_count) / float(total_mentions), 4)
        if total_mentions > 0
        else 0.0
    )

    source_cluster = aliased(ThemeCluster)
    target_cluster = aliased(ThemeCluster)
    merge_rows = db.query(
        ThemeMergeSuggestion.status,
        func.count(ThemeMergeSuggestion.id),
    ).join(
        source_cluster,
        source_cluster.id == ThemeMergeSuggestion.source_cluster_id,
    ).join(
        target_cluster,
        target_cluster.id == ThemeMergeSuggestion.target_cluster_id,
    ).filter(
        source_cluster.pipeline == pipeline,
        target_cluster.pipeline == pipeline,
        ThemeMergeSuggestion.created_at >= cutoff,
    ).group_by(
        ThemeMergeSuggestion.status
    ).all()
    merge_status_counts = {str(row[0]): int(row[1]) for row in merge_rows}
    merge_pending_count = int(merge_status_counts.get("pending", 0))
    merge_reviewed = (
        int(merge_status_counts.get("approved", 0))
        + int(merge_status_counts.get("rejected", 0))
        + int(merge_status_counts.get("auto_merged", 0))
    )
    merge_positive = int(merge_status_counts.get("approved", 0)) + int(merge_status_counts.get("auto_merged", 0))
    merge_precision_proxy = (
        round(float(merge_positive) / float(merge_reviewed), 4)
        if merge_reviewed > 0
        else 1.0
    )

    metrics = {
        "parse_failure_rate": float(rates.get("parse_failure_rate", 0.0) or 0.0),
        "processed_without_mentions_ratio": float(rates.get("processed_without_mentions_ratio", 0.0) or 0.0),
        "new_cluster_rate": float(new_cluster_rate),
        "total_mentions": int(total_mentions),
        "new_cluster_count": int(new_cluster_count),
        "failed_retryable_count": int(counts.get("failed_retryable", 0) or 0),
        "retryable_growth_ratio": float(retryable_growth.get("ratio", 0.0) or 0.0),
        "retryable_growth_delta": int(retryable_growth.get("delta", 0) or 0),
        "merge_pending_count": int(merge_pending_count),
        "merge_reviewed_count": int(merge_reviewed),
        "merge_precision_proxy": float(merge_precision_proxy),
        "match_method_mix": method_mix,
        "merge_status_counts": merge_status_counts,
    }

    thresholds = {
        "parse_failure_rate_max": 0.25,
        "processed_without_mentions_ratio_max": 0.20,
        "new_cluster_rate_max": 0.45,
        "retryable_growth_ratio_max": 2.0,
        "retryable_growth_delta_max": 10,
        "merge_pending_count_max": 50,
        "merge_precision_proxy_min": 0.55,
    }
    runbook_base = PIPELINE_OBSERVABILITY_RUNBOOK_URL
    alerts: list[dict[str, Any]] = []

    if metrics["parse_failure_rate"] > thresholds["parse_failure_rate_max"]:
        alerts.append(
            {
                "key": "parse_failure_rate_high",
                "severity": "warning",
                "title": "Parse failures elevated",
                "description": "Parser/schema failures are above policy threshold for this pipeline.",
                "metric": "parse_failure_rate",
                "value": round(metrics["parse_failure_rate"], 4),
                "threshold": thresholds["parse_failure_rate_max"],
                "runbook_url": f"{runbook_base}#parse-failure-rate",
                "likely_causes": [
                    "LLM output drift or malformed JSON payloads",
                    "Provider/model changes without prompt/schema tuning",
                ],
            }
        )

    if metrics["processed_without_mentions_ratio"] > thresholds["processed_without_mentions_ratio_max"]:
        alerts.append(
            {
                "key": "processed_without_mentions_ratio_high",
                "severity": "warning",
                "title": "Processed-without-mentions ratio elevated",
                "description": "Too many items are marked processed but produced no mentions.",
                "metric": "processed_without_mentions_ratio",
                "value": round(metrics["processed_without_mentions_ratio"], 4),
                "threshold": thresholds["processed_without_mentions_ratio_max"],
                "runbook_url": f"{runbook_base}#processed-without-mentions-ratio",
                "likely_causes": [
                    "Extraction prompt too strict for current source mix",
                    "Silent parse/validation errors bypassing mention persistence",
                ],
            }
        )

    if metrics["new_cluster_rate"] > thresholds["new_cluster_rate_max"] and metrics["total_mentions"] >= 20:
        alerts.append(
            {
                "key": "new_cluster_rate_high",
                "severity": "info",
                "title": "New-cluster rate drift",
                "description": "Matcher is creating clusters at an unusually high rate.",
                "metric": "new_cluster_rate",
                "value": round(metrics["new_cluster_rate"], 4),
                "threshold": thresholds["new_cluster_rate_max"],
                "runbook_url": f"{runbook_base}#new-cluster-rate",
                "likely_causes": [
                    "Alias coverage stale or under-trained",
                    "Matching thresholds too strict for current language variance",
                ],
            }
        )

    if (
        metrics["retryable_growth_ratio"] > thresholds["retryable_growth_ratio_max"]
        and metrics["retryable_growth_delta"] >= thresholds["retryable_growth_delta_max"]
    ):
        alerts.append(
            {
                "key": "retryable_growth_high",
                "severity": "warning",
                "title": "Retry queue growth accelerating",
                "description": "Retryable failures are growing faster than expected over the last 24h.",
                "metric": "retryable_growth_ratio",
                "value": round(metrics["retryable_growth_ratio"], 4),
                "threshold": thresholds["retryable_growth_ratio_max"],
                "runbook_url": f"{runbook_base}#retry-queue-growth",
                "likely_causes": [
                    "Upstream provider instability or quota pressure",
                    "New extraction failure mode not handled by current retry policy",
                ],
            }
        )

    if metrics["merge_pending_count"] > thresholds["merge_pending_count_max"]:
        alerts.append(
            {
                "key": "merge_pending_backlog_high",
                "severity": "info",
                "title": "Merge-review backlog high",
                "description": "Pending merge suggestions exceed review queue target.",
                "metric": "merge_pending_count",
                "value": int(metrics["merge_pending_count"]),
                "threshold": int(thresholds["merge_pending_count_max"]),
                "runbook_url": f"{runbook_base}#merge-review-backlog",
                "likely_causes": [
                    "Insufficient reviewer throughput for incoming suggestions",
                    "Similarity thresholds too permissive for auto-queueing",
                ],
            }
        )

    if (
        metrics["merge_reviewed_count"] >= 10
        and metrics["merge_precision_proxy"] < thresholds["merge_precision_proxy_min"]
    ):
        alerts.append(
            {
                "key": "merge_precision_proxy_low",
                "severity": "warning",
                "title": "Merge precision proxy low",
                "description": "Approved/auto-merged share of reviewed suggestions fell below target.",
                "metric": "merge_precision_proxy",
                "value": round(metrics["merge_precision_proxy"], 4),
                "threshold": thresholds["merge_precision_proxy_min"],
                "runbook_url": f"{runbook_base}#merge-precision-proxy",
                "likely_causes": [
                    "Embedding similarity threshold too loose for current universe",
                    "LLM merge verifier calibration drift",
                ],
            }
        )

    return {
        "generated_at": now.isoformat(),
        "window_days": max_age_days,
        "pipeline": pipeline,
        "metrics": metrics,
        "alerts": alerts,
    }

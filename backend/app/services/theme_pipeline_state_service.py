"""Helpers for pipeline-scoped state reconciliation and observability."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ..models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ThemeMention,
)

VALID_PIPELINES = ("technical", "fundamental")


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


def reconcile_source_pipeline_change(
    db: Session,
    source_id: int,
    old_pipelines: list[str],
    new_pipelines: list[str],
    chunk_size: int = 500,
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

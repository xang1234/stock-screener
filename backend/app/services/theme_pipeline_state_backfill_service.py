"""Backfill service for content_item_pipeline_state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.theme import (
    ContentItem,
    ContentItemPipelineState,
    ContentSource,
    ThemeMention,
)
from .theme_pipeline_state_service import normalize_pipelines


@dataclass
class BackfillChunkResult:
    rows_read: int = 0
    rows_written: int = 0
    conflicts: int = 0
    next_cursor: int = 0
    done: bool = False
    writes_by_pipeline_status: dict[str, int] = field(default_factory=dict)


class ThemePipelineStateBackfillService:
    """Chunked, idempotent backfill worker for pipeline-state rows."""

    def __init__(self, db: Session):
        self.db = db

    def _infer_state(
        self,
        item: ContentItem,
        pipeline: str,
        mention_counts: dict[str, int],
    ) -> dict[str, Any]:
        mention_count = mention_counts.get(pipeline, 0)
        now = datetime.utcnow()

        if mention_count > 0:
            return {
                "status": "processed",
                "attempt_count": 1,
                "processed_at": item.processed_at,
                "last_attempt_at": item.processed_at,
                "error_code": None,
                "error_message": None,
            }

        if item.extraction_error:
            return {
                "status": "failed_retryable",
                "attempt_count": 1,
                "processed_at": None,
                "last_attempt_at": item.processed_at or now,
                "error_code": "legacy_extraction_error",
                "error_message": str(item.extraction_error)[:4000],
            }

        if item.is_processed:
            return {
                "status": "failed_retryable",
                "attempt_count": 1,
                "processed_at": None,
                "last_attempt_at": item.processed_at or now,
                "error_code": "legacy_processed_without_mentions",
                "error_message": (
                    "Legacy content row marked processed without pipeline-specific mention evidence."
                ),
            }

        return {
            "status": "pending",
            "attempt_count": 0,
            "processed_at": None,
            "last_attempt_at": None,
            "error_code": None,
            "error_message": None,
        }

    def process_chunk(
        self,
        after_content_item_id: int,
        limit: int,
        dry_run: bool = False,
    ) -> BackfillChunkResult:
        rows = self.db.query(ContentItem).filter(
            ContentItem.id > after_content_item_id,
        ).order_by(ContentItem.id.asc()).limit(limit).all()

        if not rows:
            return BackfillChunkResult(
                rows_read=0,
                rows_written=0,
                conflicts=0,
                next_cursor=after_content_item_id,
                done=True,
                writes_by_pipeline_status={},
            )

        item_ids = [item.id for item in rows]
        source_ids = sorted({item.source_id for item in rows if item.source_id is not None})

        source_pipeline_map = {
            row[0]: normalize_pipelines(row[1])
            for row in self.db.query(ContentSource.id, ContentSource.pipelines).filter(
                ContentSource.id.in_(source_ids)
            ).all()
        }

        mention_rows = self.db.query(
            ThemeMention.content_item_id,
            ThemeMention.pipeline,
            func.count(ThemeMention.id),
        ).filter(
            ThemeMention.content_item_id.in_(item_ids)
        ).group_by(
            ThemeMention.content_item_id,
            ThemeMention.pipeline,
        ).all()
        mention_count_map: dict[int, dict[str, int]] = {}
        for content_item_id, pipeline, count in mention_rows:
            mention_count_map.setdefault(content_item_id, {})[pipeline] = int(count)

        existing_pairs = {
            (row[0], row[1])
            for row in self.db.query(
                ContentItemPipelineState.content_item_id,
                ContentItemPipelineState.pipeline,
            ).filter(
                ContentItemPipelineState.content_item_id.in_(item_ids)
            ).all()
        }

        result = BackfillChunkResult(
            rows_read=len(rows),
            rows_written=0,
            conflicts=0,
            next_cursor=item_ids[-1],
            done=False,
            writes_by_pipeline_status={},
        )

        for item in rows:
            source_pipelines = source_pipeline_map.get(item.source_id, ["technical", "fundamental"])
            mention_counts = mention_count_map.get(item.id, {})
            target_pipelines = sorted(set(source_pipelines) | set(mention_counts.keys()))

            if not target_pipelines:
                target_pipelines = list(source_pipelines)

            for pipeline in target_pipelines:
                key = (item.id, pipeline)
                if key in existing_pairs:
                    result.conflicts += 1
                    continue

                inferred = self._infer_state(item, pipeline, mention_counts)
                if not dry_run:
                    self.db.add(
                        ContentItemPipelineState(
                            content_item_id=item.id,
                            pipeline=pipeline,
                            status=inferred["status"],
                            attempt_count=inferred["attempt_count"],
                            error_code=inferred["error_code"],
                            error_message=inferred["error_message"],
                            last_attempt_at=inferred["last_attempt_at"],
                            processed_at=inferred["processed_at"],
                        )
                    )

                status_key = f"{pipeline}:{inferred['status']}"
                result.writes_by_pipeline_status[status_key] = (
                    result.writes_by_pipeline_status.get(status_key, 0) + 1
                )
                result.rows_written += 1
                existing_pairs.add(key)

        if not dry_run:
            self.db.commit()
        else:
            self.db.rollback()

        return result

    def summary_counts(self) -> dict[str, Any]:
        rows = self.db.query(
            ContentItemPipelineState.pipeline,
            ContentItemPipelineState.status,
            func.count(ContentItemPipelineState.id),
        ).group_by(
            ContentItemPipelineState.pipeline,
            ContentItemPipelineState.status,
        ).all()

        by_pipeline: dict[str, dict[str, int]] = {}
        for pipeline, status, count in rows:
            by_pipeline.setdefault(pipeline, {})[status] = int(count)

        return {"by_pipeline_status": by_pipeline}

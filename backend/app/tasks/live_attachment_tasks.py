"""Durable live attachment preparation; pending DB rows are the work queue."""

import os
from datetime import datetime, timezone

from sqlalchemy import func, or_, select, update

from app.celery_app import celery_app
from app.services.live_attachment_eligibility import (
    attachment_eligibility,
    authorize_attachment,
)


@celery_app.task(name="app.tasks.live_attachment_tasks.refresh_attachment_themes")
def refresh_attachment_themes(item_ids):
    from app.database import SessionLocal
    from app.services.theme_extraction_service import ThemeExtractionService
    from app.tasks.theme_discovery_tasks import _theme_automation_gate_result

    ids = sorted({int(value) for value in item_ids})[:100]
    results = []
    with SessionLocal() as db:
        skip = _theme_automation_gate_result(db)
        if skip is not None:
            return skip
        for pipeline in ("fundamental", "technical"):
            results.append(
                ThemeExtractionService(db, pipeline=pipeline).process_batch(
                    item_ids=ids
                )
            )
    return {"results": results}


@celery_app.task(name="app.tasks.live_attachment_tasks.prepare_live_attachments")
def prepare_live_attachments():
    from app.database import SessionLocal
    from app.models.theme import (
        ContentAttachment,
        ContentItem,
        ContentItemPipelineState,
    )
    from app.services.live_attachment_preparation import LiveAttachmentPreparer
    from app.services.live_attachment_service import reconcile_attachment_revisions
    from app.services.live_attachment_worker import prepare_pending_attachment

    if os.environ.get("LIVE_ATTACHMENT_PREPARATION_ENABLED", "false").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return {"status": "disabled", "processed": 0}
    key = os.environ.get("OPENCODE_GO_API_KEY", "").strip()
    if not key:
        return {
            "status": "blocked",
            "reason": "opencode_go_api_key_required",
            "processed": 0,
        }
    started = datetime.now(timezone.utc)
    preparer = LiveAttachmentPreparer(key)
    count = 0
    for _ in range(2):
        if not prepare_pending_attachment(
            SessionLocal,
            preparer,
            eligibility=attachment_eligibility,
            authorize=authorize_attachment,
        ):
            break
        count += 1
    with SessionLocal.begin() as db:
        # Include pending stale rows again so a lost targeted dispatch is retried.
        stale = db.scalars(
            select(ContentItem.id)
            .join(
                ContentItemPipelineState,
                ContentItemPipelineState.content_item_id == ContentItem.id,
            )
            .where(
                attachment_eligibility(ContentItem.id),
                ContentItem.attachment_revision.is_not(None),
                ContentItemPipelineState.status.in_(
                    ["pending", "processed", "failed_retryable", "failed_terminal"]
                ),
                or_(
                    ContentItemPipelineState.evidence_revision.is_(None),
                    ContentItemPipelineState.evidence_revision
                    != ContentItem.attachment_revision,
                ),
            )
            .group_by(ContentItem.id)
            .order_by(func.min(ContentItemPipelineState.updated_at), ContentItem.id)
            .limit(100)
        ).all()
        for item_id in stale:
            reconcile_attachment_revisions(db, item_id)
        if stale:
            # Rotate recovery after a dispatch attempt instead of repeatedly
            # selecting the same low IDs when the extraction queue is delayed.
            db.execute(update(ContentItemPipelineState).where(
                ContentItemPipelineState.content_item_id.in_(stale),
                ContentItemPipelineState.status != 'in_progress'
            ).values(updated_at=started))
        prepared_ids = list(
            db.scalars(
                select(ContentAttachment.content_item_id)
                .where(
                    attachment_eligibility(ContentAttachment.content_item_id),
                    ContentAttachment.prepared_at >= started,
                )
                .distinct()
                .limit(100)
            )
        )
    if stale or prepared_ids:
        refresh_attachment_themes.delay(list(dict.fromkeys([*prepared_ids, *stale]))[:100])
    if prepared_ids:
        from app.interfaces.tasks.social_signal_tasks import refresh_social_signals

        refresh_social_signals.apply_async(
            kwargs={"origin": "attachment-evidence"}, queue="social_ingestion"
        )
    return {"status": "processed", "processed": count}

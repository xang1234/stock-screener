"""Lease-fenced preparation: external I/O outside database transactions."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import or_, select, update

from app.models.theme import ContentAttachment
from app.services.theme_evaluation.preparation_failures import PreparationFailure

MAX_ATTEMPTS = 3
LEASE_DURATION = timedelta(minutes=15)


def prepare_pending_attachment(
    session_factory,
    preparer,
    *,
    now=None,
    eligible_item_ids=None,
    eligibility=None,
    authorize=None,
):
    fixed_clock = now
    now = now or datetime.now(timezone.utc)
    token = str(uuid4())
    with session_factory.begin() as db:
        # Exhausted crashed leases must not remain "processing" forever.
        db.execute(
            update(ContentAttachment)
            .where(
                ContentAttachment.status == "processing",
                ContentAttachment.lease_until <= now,
                ContentAttachment.attempt_count >= MAX_ATTEMPTS,
            )
            .values(
                status="failed",
                error_code="preparation_lease_exhausted",
                lease_token=None,
                lease_until=None,
            )
        )
        due = or_(
            (ContentAttachment.status.in_(["pending", "failed"]))
            & (
                or_(
                    ContentAttachment.next_attempt_at.is_(None),
                    ContentAttachment.next_attempt_at <= now,
                )
            ),
            (ContentAttachment.status == "processing")
            & (ContentAttachment.lease_until <= now),
            (ContentAttachment.status == "partial")
            & (ContentAttachment.next_attempt_at <= now)
            & or_(
                ContentAttachment.lease_until.is_(None),
                ContentAttachment.lease_until <= now,
            ),
        )
        query = select(ContentAttachment).where(
            due, ContentAttachment.attempt_count < MAX_ATTEMPTS
        )
        if eligibility is not None:
            query = query.where(eligibility(ContentAttachment.content_item_id))
        if eligible_item_ids is not None:
            query = query.where(
                ContentAttachment.content_item_id.in_(eligible_item_ids)
            )
        row = db.scalar(
            query.order_by(ContentAttachment.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if row is None:
            return False
        if authorize is not None and not authorize(db, row.content_item_id):
            # SQL is a fast eligibility filter; the precise alias-aware check
            # can still reject a candidate. Defer it so it cannot block the queue.
            row.next_attempt_at = now + timedelta(minutes=15)
            return True
        if row.status != "partial":
            row.status = "processing"
        row.lease_token, row.lease_until = token, now + LEASE_DURATION
        row.attempt_count += 1
        row_id, kind, url = row.id, row.kind, row.url
        # Successful records themselves are the cache. A matching URL on a
        # different post is not a content hash and may now serve different bytes.
        result = None
    failure = None
    if result is None:
        try:
            prepared = preparer(kind, url)
            if prepared.status not in {"complete", "partial"} or not prepared.text:
                raise ValueError("empty_prepared_attachment")
            result = {
                "prepared_text": prepared.text,
                "original_text": prepared.original_text,
                "content_sha256": prepared.content_sha256,
                "final_url": prepared.final_url,
                "status": prepared.status,
                "provenance": prepared.provenance,
            }
        except (
            PreparationFailure,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as exc:
            failure = exc
    finished_at = fixed_clock or datetime.now(timezone.utc)
    with session_factory.begin() as db:
        row = db.scalar(
            select(ContentAttachment)
            .where(
                ContentAttachment.id == row_id, ContentAttachment.lease_token == token
            )
            .with_for_update()
        )
        if row is None:
            return True  # A newer lease owns this row; discard this stale result.
        row.lease_token, row.lease_until = None, None
        if authorize is not None and not authorize(db, row.content_item_id):
            row.status = "partial" if row.prepared_text else "pending"
            return True
        if failure is not None:
            row.status = "partial" if row.prepared_text else "failed"
            row.error_code = (
                failure.code[:100]
                if isinstance(failure, PreparationFailure)
                else "attachment_preparation_failed"
            )
            retryable = (
                failure.retryable
                if isinstance(failure, PreparationFailure)
                else isinstance(failure, OSError)
            )
            if not retryable:
                row.attempt_count = MAX_ATTEMPTS
            retry_after = getattr(failure, "retry_after_seconds", None) or 0
            row.next_attempt_at = finished_at + timedelta(
                seconds=max(60 * 2**row.attempt_count, retry_after)
            )
        else:
            for name, value in result.items():
                setattr(row, name, value)
            row.prepared_at = finished_at
            row.next_attempt_at, row.error_code = None, None
            retry = (row.provenance or {}).get("translation", {}).get("retry", {})
            if retry.get("needed") and row.attempt_count < MAX_ATTEMPTS:
                row.next_attempt_at = finished_at + timedelta(
                    seconds=60 * 2**row.attempt_count
                )
            from app.models.theme import ContentItem
            from app.services.live_attachment_service import attachment_snapshot

            item = db.scalar(
                select(ContentItem)
                .where(ContentItem.id == row.content_item_id)
                .with_for_update()
            )
            db.flush()
            item.attachment_revision = attachment_snapshot(db, row.content_item_id)[
                "revision"
            ]
    return True

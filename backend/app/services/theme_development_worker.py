"""Bounded event queue with retries, revision fencing and no model calls in ingestion."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import exists, func, or_

from app.models.theme import ContentItem, ThemeMention
from app.models.theme_intelligence import ThemeDevelopmentWork
from app.services.theme_development_preparation import generate_facts, input_bundle
from app.services.theme_development_service import record_developments
from app.services.theme_evidence_eligibility_service import legacy_eligibility_exists


def enqueue(db, item_id, pipeline):
    db.query(ContentItem).filter_by(id=item_id).with_for_update().one()
    bundle = input_bundle(db, item_id, pipeline)
    row = (
        db.query(ThemeDevelopmentWork)
        .filter_by(
            content_item_id=item_id, pipeline=pipeline, revision=bundle["revision"]
        )
        .first()
    )
    if row is None:
        row = ThemeDevelopmentWork(
            content_item_id=item_id,
            pipeline=pipeline,
            revision=bundle["revision"],
            source_marker=bundle["source_marker"],
            status="pending",
            attempts=0,
        )
        db.query(ThemeDevelopmentWork).filter(
            ThemeDevelopmentWork.content_item_id == item_id,
            ThemeDevelopmentWork.pipeline == pipeline,
            ThemeDevelopmentWork.status.in_(
                ["pending", "retry", "failed", "processing"]
            ),
        ).update(
            {"status": "superseded", "claim_token": None, "lease_until": None},
            synchronize_session=False,
        )
        db.add(row)
        db.flush()
    else:
        row.source_marker = max(row.source_marker, bundle["source_marker"])
    return row


def discover(db, limit=50, item_ids=None):
    latest = (
        db.query(
            ThemeMention.content_item_id.label("item"),
            ThemeMention.pipeline.label("pipeline"),
            func.max(ThemeMention.id).label("marker"),
        )
        .filter(
            ThemeMention.pipeline.in_(["technical", "fundamental"]),
            ThemeMention.social_work_id.is_(None),
            legacy_eligibility_exists(
                ThemeMention.content_item_id, ThemeMention.pipeline, active_only=True
            ),
        )
        .group_by(ThemeMention.content_item_id, ThemeMention.pipeline)
        .subquery()
    )
    query = db.query(latest).join(ContentItem, ContentItem.id == latest.c.item)
    if item_ids is None:
        # Automatic discovery is for recent live ingestion; older backfill is explicit.
        query = query.filter(
            ContentItem.fetched_at >= datetime.now(timezone.utc) - timedelta(days=2)
        )
    else:
        query = query.filter(latest.c.item.in_(item_ids))
    query = query.filter(
        ~exists().where(
            ThemeDevelopmentWork.content_item_id == latest.c.item,
            ThemeDevelopmentWork.pipeline == latest.c.pipeline,
            ThemeDevelopmentWork.source_marker == latest.c.marker,
        )
    )
    rows = query.order_by(latest.c.marker).limit(min(limit, 100)).all()
    for row in rows:
        enqueue(db, row.item, row.pipeline)
    return len(rows)


def process_one(sessions, generate=generate_facts):
    now = datetime.now(timezone.utc)
    token = uuid4().hex
    with sessions.begin() as db:
        db.query(ThemeDevelopmentWork).filter(
            ThemeDevelopmentWork.status == "processing",
            ThemeDevelopmentWork.attempts >= 3,
            ThemeDevelopmentWork.lease_until <= now,
        ).update(
            {
                "status": "failed",
                "error_code": "development_lease_expired",
                "claim_token": None,
                "lease_until": None,
            },
            synchronize_session=False,
        )
        row = (
            db.query(ThemeDevelopmentWork)
            .filter(
                ThemeDevelopmentWork.status.in_(["pending", "retry", "processing"]),
                ThemeDevelopmentWork.attempts < 3,
                or_(
                    ThemeDevelopmentWork.next_attempt_at.is_(None),
                    ThemeDevelopmentWork.next_attempt_at <= now,
                ),
                or_(
                    ThemeDevelopmentWork.lease_until.is_(None),
                    ThemeDevelopmentWork.lease_until <= now,
                ),
            )
            .order_by(ThemeDevelopmentWork.id)
            .with_for_update(skip_locked=True)
            .first()
        )
        if row is None:
            return False
        row.status, row.claim_token = "processing", token
        row.attempts += 1
        row.lease_until = now + timedelta(minutes=5)
        work_id, item_id, pipeline, revision = (
            row.id,
            row.content_item_id,
            row.pipeline,
            row.revision,
        )
    try:
        with sessions() as db:
            bundle = input_bundle(db, item_id, pipeline)
            values = (
                generate(pipeline, db, bundle)
                if bundle["revision"] == revision
                else None
            )
        with sessions.begin() as db:
            # Parent then work is the same lock order used by enqueue.
            db.query(ContentItem).filter_by(id=item_id).with_for_update().one()
            row = (
                db.query(ThemeDevelopmentWork)
                .filter_by(id=work_id)
                .with_for_update()
                .one()
            )
            if row.claim_token != token:
                return True
            current = input_bundle(db, item_id, pipeline)
            if values is None or current["revision"] != revision:
                row.status = "superseded"
            else:
                record_developments(
                    db,
                    item=current["item"],
                    pipeline=pipeline,
                    revision=revision,
                    theme_ids=current["theme_ids"],
                    sources=current["sources"],
                    source_urls=current["source_urls"],
                    observations=values,
                    available_at=datetime.now(timezone.utc),
                )
                row.status = "complete"
            row.lease_until, row.claim_token, row.error_code = None, None, None
    except Exception:  # noqa: BLE001 -- Provider failures become bounded, visible retries.
        with sessions.begin() as db:
            row = (
                db.query(ThemeDevelopmentWork)
                .filter_by(id=work_id, claim_token=token)
                .with_for_update()
                .first()
            )
            if row:
                row.status = "failed" if row.attempts >= 3 else "retry"
                row.error_code = "development_preparation_failed"
                row.next_attempt_at = datetime.now(timezone.utc) + timedelta(
                    minutes=2**row.attempts
                )
                row.lease_until, row.claim_token = None, None
    return True

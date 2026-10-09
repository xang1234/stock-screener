"""Bounded event queue with retries, revision fencing and no model calls in ingestion."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, or_

from app.models.economic_taxonomy_runtime import TaxonomyAuthority
from app.models.theme import ContentItem, ThemeMention
from app.models.theme_intelligence import ThemeDevelopmentWork
from app.services.economic_taxonomy_fence import producer_write
from app.services.theme_development_facts import normalize_batch
from app.services.theme_development_preparation import (
    development_bundle,
    economic_authority,
    generate_facts,
)
from app.services.theme_development_service import record_developments
from app.services.theme_evidence_eligibility_service import legacy_eligibility_exists


def enqueue(db, item_id, pipeline):
    db.query(ContentItem).filter_by(id=item_id).with_for_update().one()
    bundle = development_bundle(db, item_id, pipeline)
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
            checked_at=datetime.now(timezone.utc),
            status="pending",
            attempts=0,
        )
        db.add(row)
        db.flush()
    else:
        row.checked_at = datetime.now(timezone.utc)
        if row.status == "superseded":
            row.status, row.attempts, row.next_attempt_at = "pending", 0, None
    db.query(ThemeDevelopmentWork).filter(
        ThemeDevelopmentWork.content_item_id == item_id,
        ThemeDevelopmentWork.pipeline == pipeline,
        ThemeDevelopmentWork.revision != bundle["revision"],
        ThemeDevelopmentWork.status.in_(
            ["pending", "retry", "failed", "processing", "complete"]
        ),
    ).update(
        {"status": "superseded", "claim_token": None, "lease_until": None},
        synchronize_session=False,
    )
    return row


CHANNELS = ("technical", "fundamental")


def _economic_candidate_query(item_ids):
    """Lineages with a completed, non-empty classification.

    No DISTINCT and no JSON in the select list: PostgreSQL cannot compare its
    json type. Recent classifications only for automatic discovery, so a new
    taxonomy version reclassifying old items does not re-run the model.
    """
    from sqlalchemy import exists, select

    from app.models.economic_taxonomy_runtime import (
        ClaimAssignment,
        ClassificationAttempt,
        ProcessingRequest,
    )

    query = (
        select(ProcessingRequest.source_lineage_id)
        .join(
            ClassificationAttempt,
            ClassificationAttempt.processing_request_id == ProcessingRequest.id,
        )
        .where(
            ClassificationAttempt.result_status == "completed",
            exists().where(ClaimAssignment.classification_attempt_id == ClassificationAttempt.id),
        )
    )
    if item_ids is None:
        query = query.where(
            ClassificationAttempt.created_at >= datetime.now(timezone.utc) - timedelta(days=2)
        )
    return query


def _economic_candidates(db, item_ids):
    """(item, channel) pairs with classified economic evidence (#513).

    The classified packet may be a later capture of the same source (Social
    supersedes an X capture, #500), so items are found through any
    content-ingestion packet in the lineage; channels come from the lineage's
    effective packet.
    """
    from sqlalchemy import select

    from app.models.economic_taxonomy_runtime import EvidencePacket
    from app.services.economic_source_admission import (
        CONTENT_INGESTION_ROUTE,
        EconomicSourceAdmissionService,
    )

    lineages = set(db.execute(_economic_candidate_query(item_ids)).scalars())
    if not lineages:
        return []
    items_by_lineage = {}
    for lineage_id, metadata in db.execute(
        select(EvidencePacket.source_lineage_id, EvidencePacket.source_metadata).where(
            EvidencePacket.source_lineage_id.in_(lineages),
            EvidencePacket.capture_route == CONTENT_INGESTION_ROUTE,
        )
    ):
        item = (metadata or {}).get("content_item_id")
        if item is not None:
            items_by_lineage.setdefault(lineage_id, set()).add(int(item))
    wanted = {item for items in items_by_lineage.values() for item in items}
    if item_ids is not None:
        wanted &= set(item_ids)
    elif wanted:
        wanted = {
            row.id
            for row in db.query(ContentItem.id).filter(
                ContentItem.id.in_(wanted),
                ContentItem.fetched_at >= datetime.now(timezone.utc) - timedelta(days=2),
            )
        }
    admission = EconomicSourceAdmissionService(db)
    pairs = set()
    for lineage_id, items in items_by_lineage.items():
        items &= wanted
        if not items:
            continue
        packet = admission.effective_packet(lineage_id)
        channels = admission.latest_channels(packet.id) if packet is not None else set()
        pairs.update((item, channel) for item in items for channel in CHANNELS if channel in channels)
    return sorted(pairs)


def _least_recently_checked(db, pairs, limit):
    """The legacy rotation: never-checked pairs first, then the oldest check."""
    if not pairs:
        return []
    checked = {
        (row.item, row.pipeline): row.checked_at
        for row in db.query(
            ThemeDevelopmentWork.content_item_id.label("item"),
            ThemeDevelopmentWork.pipeline.label("pipeline"),
            func.max(ThemeDevelopmentWork.checked_at).label("checked_at"),
        )
        .filter(ThemeDevelopmentWork.content_item_id.in_({item for item, _ in pairs}))
        .group_by(ThemeDevelopmentWork.content_item_id, ThemeDevelopmentWork.pipeline)
    }

    def key(pair):
        at = checked.get(pair)
        return (at is not None, at.timestamp() if at is not None else 0.0, pair)

    return sorted(pairs, key=key)[: min(limit, 100)]


def discover(db, limit=50, item_ids=None):
    if economic_authority(db):
        queued = 0
        for item, pipeline in _least_recently_checked(db, _economic_candidates(db, item_ids), limit):
            work = enqueue(db, item, pipeline)
            queued += work.status == "pending"
        return queued
    latest = (
        db.query(
            ThemeMention.content_item_id.label("item"),
            ThemeMention.pipeline.label("pipeline"),
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
    checked = (
        db.query(
            ThemeDevelopmentWork.content_item_id.label("item"),
            ThemeDevelopmentWork.pipeline.label("pipeline"),
            func.max(ThemeDevelopmentWork.checked_at).label("checked_at"),
        )
        .group_by(ThemeDevelopmentWork.content_item_id, ThemeDevelopmentWork.pipeline)
        .subquery()
    )
    query = query.outerjoin(
        checked,
        (checked.c.item == latest.c.item) & (checked.c.pipeline == latest.c.pipeline),
    )
    rows = (
        query.order_by(
            checked.c.checked_at.asc().nullsfirst(), latest.c.item, latest.c.pipeline
        )
        .limit(min(limit, 100))
        .all()
    )
    queued = 0
    for row in rows:
        work = enqueue(db, row.item, row.pipeline)
        queued += work.status == "pending"
    return queued


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
            bundle = development_bundle(db, item_id, pipeline)
            values = (
                generate(pipeline, db, bundle)
                if bundle["revision"] == revision
                else None
            )
            prepared = (
                normalize_batch(
                    values,
                    item_id=item_id,
                    theme_ids=set(bundle["theme_ids"]),
                    sources=bundle["sources"],
                )
                if values is not None
                else None
            )
        with sessions.begin() as db:
            authority = db.get(TaxonomyAuthority, 1)
            expected_epoch = authority.authority_epoch if authority is not None else 1
            with producer_write(
                db,
                expected_epoch=expected_epoch,
                allowed_modes={"legacy", "shadow", "dual", "economic"},
            ) as locked_authority:
                # Fence and authority precede the established parent/work lock order.
                db.query(ContentItem).filter_by(id=item_id).with_for_update().one()
                row = (
                    db.query(ThemeDevelopmentWork)
                    .filter_by(id=work_id)
                    .with_for_update()
                    .one()
                )
                if row.claim_token != token:
                    return True
                current = development_bundle(db, item_id, pipeline)
                # Economic evidence not classified for this channel is "not
                # ready", never an empty revision: recording supersedes every
                # other revision of the item, which would wipe its history.
                not_ready = current["kind"] == "economic" and not current["theme_ids"]
                if values is None or current["revision"] != revision:
                    row.status = "superseded"
                    if not not_ready:
                        enqueue(db, item_id, pipeline)
                elif not_ready:
                    row.status = "complete"
                else:
                    economic = current["kind"] == "economic"
                    record_developments(
                        db,
                        item=current["item"],
                        pipeline=pipeline,
                        revision=revision,
                        # Economic authority links each event to its own
                        # economic themes and writes no legacy links (#513).
                        theme_ids=[] if economic else current["theme_ids"],
                        economic_theme_refs=current["economic_refs"] if economic else None,
                        sources=current["sources"],
                        source_urls=current["source_urls"],
                        observations=values,
                        prepared_observations=prepared,
                        available_at=datetime.now(timezone.utc),
                        authority_fenced=True,
                        authority_epoch=locked_authority.authority_epoch,
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

"""Live child evidence storage and bounded, versioned extraction snapshots."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from urllib.parse import urlsplit

from sqlalchemy import select

from app.models.theme import ContentAttachment, ContentItem, ContentItemPipelineState
from app.services.theme_company_context import build_company_context
from app.services.theme_grounding_context import (
    MAX_RELATED_CHARACTERS,
    GroundingContext,
    GroundingEvidence,
)

POLICY_VERSION = "live-attachment-v1"
MAX_ATTACHMENTS = 10


def utc(value):
    return (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )


def digest(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def record_attachments(db, item, refs, observed_at=None):
    """Persist references inside ingestion's transaction; a sweep owns delivery."""
    if not refs:
        return
    # Serialize duplicate ingestion for this parent on databases with row locks.
    db.execute(
        select(ContentItem.id).where(ContentItem.id == item.id).with_for_update()
    )
    existing = set(
        db.scalars(
            select(ContentAttachment.reference_key).where(
                ContentAttachment.content_item_id == item.id
            )
        )
    )
    for ref in refs:
        ref = asdict(ref) if is_dataclass(ref) else ref
        if not isinstance(ref, dict):
            continue
        kind, url = ref.get("kind"), ref.get("url")
        if (
            kind not in {"image", "article"}
            or not isinstance(url, str)
            or len(url) > 4096
        ):
            continue
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                continue
            if parsed.port not in (None, 80, 443):
                continue
        except ValueError:
            continue
        url = parsed._replace(fragment="").geturl()
        key = digest([kind, url])
        if key in existing or len(existing) >= MAX_ATTACHMENTS:
            continue
        db.add(
            ContentAttachment(
                content_item_id=item.id,
                kind=kind,
                url=url,
                reference_key=key,
                policy_version=POLICY_VERSION,
                status="pending",
                observed_at=observed_at or datetime.now(timezone.utc),
                attempt_count=0,
            )
        )
        existing.add(key)
    db.flush()


def attachment_snapshot(db, item_id, as_of=None):
    """A derivative snapshot, limited to evidence actually available at as_of."""
    rows = db.scalars(
        select(ContentAttachment)
        .where(ContentAttachment.content_item_id == item_id)
        .order_by(ContentAttachment.id)
    ).all()
    cutoff = utc(as_of) if as_of else datetime.now(timezone.utc)
    rows = [r for r in rows if utc(r.observed_at) <= cutoff]
    evidence, summary = [], []
    remaining = MAX_RELATED_CHARACTERS
    seen_content = set()
    for row in rows:
        available = row.prepared_at and utc(row.prepared_at) <= cutoff
        status = (
            row.status
            if available or row.status not in {"complete", "partial"}
            else "pending"
        )
        summary.append(
            {
                "kind": row.kind,
                "url": row.url,
                "status": status,
                "error_code": row.error_code,
                "warnings": (row.provenance or {}).get("warnings", []),
            }
        )
        if (
            not available
            or row.status not in {"complete", "partial"}
            or not row.prepared_text
            or remaining <= 0
        ):
            continue
        identity = (row.kind, row.content_sha256)
        if identity in seen_content:
            continue
        seen_content.add(identity)
        text = row.prepared_text[:remaining]
        remaining -= len(text)
        provenance = dict(row.provenance or {})
        provenance["truncated"] = len(text) < len(row.prepared_text) or bool(
            provenance.get("truncated")
        )
        provenance["source_family_id"] = str(item_id)
        evidence.append(
            {
                "id": digest(
                    [
                        str(item_id),
                        row.reference_key,
                        row.content_sha256,
                        row.policy_version,
                    ]
                ),
                "kind": row.kind,
                "url": row.final_url or row.url,
                "text": text,
                # This is the complete prepared representation before context clipping;
                # the captured original has its own provenance hash.
                "original_text_sha256": sha256(row.prepared_text.encode()).hexdigest(),
                "text_sha256": sha256(text.encode()).hexdigest(),
                "available_at": utc(row.prepared_at).isoformat(),
                "provenance": {
                    **provenance,
                    "captured_original_text_sha256": sha256(
                        (row.original_text or "").encode()
                    ).hexdigest(),
                    "content_sha256": row.content_sha256,
                    "policy_version": row.policy_version,
                },
            }
        )
    states = [r["status"] for r in summary]
    if not states:
        status = "none"
    elif all(s == "complete" for s in states) and remaining > 0:
        status = "complete"
    elif all(s == "failed" for s in states):
        status = "failed"
    elif not evidence and any(s in {"pending", "processing"} for s in states):
        status = "pending"
    else:
        status = "partial"
    return {
        "revision": digest(evidence),
        "status": status,
        "evidence": evidence,
        "attachments": summary,
    }


def build_live_grounding(db, item, *, now=None, snapshot=None):
    now = now or datetime.now(timezone.utc)
    snapshot = (
        snapshot
        if snapshot is not None
        else attachment_snapshot(db, item.id, as_of=now)
    )
    company_text = "\n".join(
        [item.content or "", *(e["text"] for e in snapshot["evidence"])]
    )
    facts = build_company_context(db, company_text, as_of=now)
    evidence = [
        GroundingEvidence(
            input_id=e["id"],
            source_id=str(item.id),
            source_url=e["url"],
            input_kind="image_observation" if e["kind"] == "image" else "article_text",
            relation="attached_image" if e["kind"] == "image" else "linked_article",
            text=e["text"],
            original_text_sha256=e["original_text_sha256"],
            text_sha256=e["text_sha256"],
            available_at=e["available_at"],
            truncated=e["provenance"].get("truncated", False),
            warnings=e["provenance"].get("warnings", [])[:100],
        )
        for e in snapshot["evidence"]
    ]
    return GroundingContext(
        context_available_at=now,
        companies=facts["companies"],
        evidence=evidence,
        warnings=(
            facts["warnings"]
            + (
                [f"attachments_{snapshot['status']}"]
                if snapshot["status"] not in {"none", "complete"}
                else []
            )
        )[:100],
    )


def reconcile_attachment_revisions(db, item_id):
    """Don't steal an in-flight claim. The next sweep catches a stale completion."""
    snapshot = attachment_snapshot(db, item_id)
    if not snapshot["evidence"]:
        return
    states = db.scalars(
        select(ContentItemPipelineState)
        .where(
            ContentItemPipelineState.content_item_id == item_id,
            ContentItemPipelineState.status.in_(
                ["processed", "failed_terminal", "failed_retryable"]
            ),
        )
        .with_for_update()
    ).all()
    for state in states:
        if state.evidence_revision != snapshot["revision"]:
            state.status = "pending"
            state.error_code = None
            state.error_message = None
            # Keep old mentions visible until a successful replacement commits.


# Compatibility import keeps persistence readers independent of task registration.
from app.services.live_attachment_worker import (
    prepare_pending_attachment,  # noqa: F401
)

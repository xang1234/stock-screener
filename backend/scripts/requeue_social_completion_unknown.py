"""Return social extraction work stranded as completion_unknown, or refused as
provider_rejected, to the backlog.

Work qualifies when no extraction attempt for it was billed (reconciled) or is
possibly still in flight, and its post is inside the analysis window the backlog claims
from. Work whose attempt is uncertain is requeued deliberately, and a few of
those requests may have been billed. --since and --until keep only work with an
attempt created in that window, read as UTC. --model moves the requeued work, and
work still waiting on another model, to that model; a row whose post already has
work for it is left alone. A dry run prints the counts;
--apply writes them in one transaction. Running it again finds nothing new to
requeue.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ERROR_CODES = ("completion_unknown", "provider_rejected")
# Claim expiry marks a running request uncertain after 10 minutes, and extraction calls
# time out after 120 s (REQUEST_TIMEOUT_SECONDS), so an attempt this recent may still return a result.
IN_FLIGHT_GRACE = timedelta(hours=1)


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


def _classify(db, now, error_code, since, until):
    from sqlalchemy import select

    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt

    attempt_states, latest, in_window = {}, {}, set()
    for attempt in db.scalars(select(SocialLLMAttempt).where(
        SocialLLMAttempt.operation_kind == "social_extraction"
    )):
        created = _utc(attempt.created_at)
        for work_id in attempt.work_ids:
            attempt_states.setdefault(work_id, set()).add(attempt.state)
            latest[work_id] = max(latest.get(work_id, created), created)
            if (since is None or since <= created) and (until is None or created < until):
                in_window.add(work_id)
    rows = db.scalars(select(SocialExtractionWork).where(
        SocialExtractionWork.state == "failed_terminal",
        SocialExtractionWork.error_code == error_code,
    ).order_by(SocialExtractionWork.id)).all()
    for row in rows:
        if (since, until) != (None, None) and row.id not in in_window:
            continue
        states = attempt_states.get(row.id, set())
        published = datetime.fromisoformat(row.input_snapshot_json["created_at"])
        if "reconciled" in states:
            yield "billed", row
        elif states & {"reserved", "dispatched"} or latest.get(row.id, now - IN_FLIGHT_GRACE) > now - IN_FLIGHT_GRACE:
            yield "in_flight", row
        elif published < now - timedelta(days=14) and not row.requested_by_admin:
            yield "outside_window", row
        else:
            yield "requeue", row


def _taken(db, row, model):
    from sqlalchemy import select

    from app.infra.db.models.social_analysis import SocialExtractionWork

    return db.scalar(select(SocialExtractionWork.id).where(
        SocialExtractionWork.id != row.id,
        SocialExtractionWork.content_item_id == row.content_item_id,
        SocialExtractionWork.input_hash == row.input_hash,
        SocialExtractionWork.prompt_version == row.prompt_version,
        SocialExtractionWork.schema_version == row.schema_version,
        SocialExtractionWork.selected_model == model,
    )) is not None


def _plan(db, now, error_code, since, until, model):
    from sqlalchemy import select

    from app.infra.db.models.social_analysis import SocialExtractionWork
    from app.services.social_signal_backlog_service import NO_SIGNAL_GATE

    plan, assigned = [], set()

    def moved(row):
        """Whether row can take model: no other row holds it, nor one this plan moves first."""
        if row.selected_model == model:
            return True
        identity = (row.content_item_id, row.input_hash, row.prompt_version, row.schema_version)
        if identity in assigned or _taken(db, row, model):
            return False
        assigned.add(identity)
        return True

    for kind, row in _classify(db, now, error_code, since, until):
        plan.append(("retarget_duplicate" if kind == "requeue" and model and not moved(row) else kind, row))
    if model:
        for row in db.scalars(select(SocialExtractionWork).where(
            SocialExtractionWork.state.in_(("pending", "waiting_budget", "failed_retryable")),
            SocialExtractionWork.selected_model.not_in((model, NO_SIGNAL_GATE)),
        ).order_by(SocialExtractionWork.id)):
            plan.append(("retarget" if moved(row) else "retarget_duplicate", row))
    return plan


def requeue(session_factory, now, *, apply=False, error_code="completion_unknown", since=None, until=None,
            model=None):
    from app.services.social_llm_budget_service import social_analysis_transaction

    since, until = _utc(since), _utc(until)
    if not apply:
        with session_factory() as db:
            return Counter(kind for kind, _ in _plan(db, now, error_code, since, until, model))
    counts = Counter()
    with social_analysis_transaction(session_factory) as db:
        for kind, row in _plan(db, now, error_code, since, until, model):
            counts[kind] += 1
            if kind == "requeue":
                row.state, row.error_code = "pending", None
                row.claim_token = row.claim_expires_at = None
                row.updated_at = now
            if kind in {"requeue", "retarget"} and model:
                row.selected_model = model
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the requeue instead of only counting")
    parser.add_argument("--error-code", choices=ERROR_CODES, default="completion_unknown")
    parser.add_argument("--since", type=datetime.fromisoformat, help="earliest attempt created_at, inclusive")
    parser.add_argument("--until", type=datetime.fromisoformat, help="latest attempt created_at, exclusive")
    parser.add_argument("--model", help="move requeued and waiting work to this social extraction model")
    args = parser.parse_args(argv)
    from app.database import SessionLocal

    counts = requeue(SessionLocal, datetime.now(timezone.utc), apply=args.apply,
                     error_code=args.error_code, since=args.since, until=args.until, model=args.model)
    verb = "requeued" if args.apply else "would requeue"
    print(f"{verb} {counts['requeue']} {args.error_code} work rows")
    if args.model:
        print(f"{'moved' if args.apply else 'would move'} {counts['retarget']} waiting work rows to {args.model}")
    for kind in ("billed", "in_flight", "outside_window", "retarget_duplicate"):
        print(f"left {kind}: {counts[kind]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

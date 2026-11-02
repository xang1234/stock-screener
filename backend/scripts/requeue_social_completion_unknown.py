"""Return social extraction work stranded as completion_unknown to the backlog.

Work qualifies when no extraction attempt for it was billed (reconciled) or is
still in flight, and its post is inside the analysis window the backlog claims
from. A dry run prints the counts; --apply writes them in one transaction.
Running it again finds nothing new to requeue.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _classify(db, now):
    from sqlalchemy import select

    from app.infra.db.models.social_analysis import SocialExtractionWork, SocialLLMAttempt

    attempt_states = {}
    for attempt in db.scalars(select(SocialLLMAttempt).where(
        SocialLLMAttempt.operation_kind == "social_extraction"
    )):
        for work_id in attempt.work_ids:
            attempt_states.setdefault(work_id, set()).add(attempt.state)
    rows = db.scalars(select(SocialExtractionWork).where(
        SocialExtractionWork.state == "failed_terminal",
        SocialExtractionWork.error_code == "completion_unknown",
    ).order_by(SocialExtractionWork.id)).all()
    for row in rows:
        states = attempt_states.get(row.id, set())
        published = datetime.fromisoformat(row.input_snapshot_json["created_at"])
        if "reconciled" in states:
            yield "billed", row
        elif states & {"reserved", "dispatched"}:
            yield "in_flight", row
        elif published < now - timedelta(days=14) and not row.requested_by_admin:
            yield "outside_window", row
        else:
            yield "requeue", row


def requeue(session_factory, now, *, apply=False):
    from app.services.social_llm_budget_service import social_analysis_transaction

    if not apply:
        with session_factory() as db:
            return Counter(kind for kind, _ in _classify(db, now))
    counts = Counter()
    with social_analysis_transaction(session_factory) as db:
        for kind, row in _classify(db, now):
            counts[kind] += 1
            if kind == "requeue":
                row.state, row.error_code = "pending", None
                row.claim_token = row.claim_expires_at = None
                row.updated_at = now
    return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the requeue instead of only counting")
    args = parser.parse_args(argv)
    from app.database import SessionLocal

    counts = requeue(SessionLocal, datetime.now(timezone.utc), apply=args.apply)
    verb = "requeued" if args.apply else "would requeue"
    print(f"{verb} {counts['requeue']} completion_unknown work rows")
    for kind in ("billed", "in_flight", "outside_window"):
        print(f"left {kind}: {counts[kind]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

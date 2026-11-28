"""Return Economic Taxonomy processing requests that a provider failure made
terminal to the work queue.

A request qualifies when it is terminal_failure with one of the given
completion codes. It moves to retryable through the work repository's retry
transition, under the same producer fence and authority epoch check as the
task loop, which records a retry_scheduled event. Its terminal revision stays
in the append-only revision log: it is a routine attention item, the request's
next outcome appends a revision of its own, and a requeued request reserves
Social budget under a new logical operation so an uncertain earlier attempt
keeps its charge without blocking the retry. A dry run prints the counts;
--apply writes them in one transaction. Running it again finds nothing new to
requeue.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMPLETION_CODES = (
    "provider_outcome_uncertain",
    "invalid_result_status",
    "invalid_provider_json",
)


def requeue(session_factory, now, *, apply=False, codes=COMPLETION_CODES, limit=None):
    from sqlalchemy import select

    from app.infra.db.repositories.economic_taxonomy_work_repo import (
        OPERATOR_REQUEUE_REASON,
        EconomicTaxonomyWorkRepository,
    )
    from app.models.economic_taxonomy_runtime import ProcessingRequest, TaxonomyAuthority
    from app.services.economic_taxonomy_fence import producer_write

    query = (
        select(ProcessingRequest.id, ProcessingRequest.completion_code)
        .where(
            ProcessingRequest.status == "terminal_failure",
            ProcessingRequest.completion_code.in_(codes),
        )
        .order_by(ProcessingRequest.created_at, ProcessingRequest.id)
        .limit(limit)
    )
    with session_factory() as db:
        if not apply:
            return Counter(code for _, code in db.execute(query))
        authority = db.get(TaxonomyAuthority, 1)
        if authority is None:
            raise RuntimeError("taxonomy_authority_missing")
        with producer_write(
            db,
            expected_epoch=authority.authority_epoch,
            allowed_modes={"shadow", "dual", "economic"},
        ):
            work = EconomicTaxonomyWorkRepository(db)
            counts = Counter()
            for request_id, code in db.execute(query.with_for_update()).all():
                work.retry(request_id, available_at=now, reason=OPERATOR_REQUEUE_REASON)
                counts[code] += 1
            db.commit()
        return counts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the requeue instead of only counting")
    parser.add_argument(
        "--completion-code",
        action="append",
        dest="codes",
        help=f"requeue this completion code; repeat for several (default: {', '.join(COMPLETION_CODES)})",
    )
    parser.add_argument("--limit", type=int, help="requeue at most this many requests, oldest first")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    from app.database import SessionLocal

    counts = requeue(
        SessionLocal,
        datetime.now(timezone.utc),
        apply=args.apply,
        codes=tuple(args.codes or COMPLETION_CODES),
        limit=args.limit,
    )
    verb = "requeued" if args.apply else "would requeue"
    print(f"{verb} {sum(counts.values())} terminal processing requests")
    for code, count in sorted(counts.items()):
        print(f"  {code}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

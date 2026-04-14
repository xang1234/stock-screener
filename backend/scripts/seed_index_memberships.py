#!/usr/bin/env python
"""CLI wrapper for seeding stock_universe_index_membership from a CSV.

Usage:
    python scripts/seed_index_memberships.py \\
        --csv app/data/hsi_constituents_2025-05.csv \\
        --index HSI \\
        --as-of 2025-05-01

Use ``--dry-run`` to report added/updated/unchanged counts without writing.
The ``--index`` choices track the ``IndexName`` enum via import so adding a
new index value lands in one place.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.schemas.universe import IndexName  # noqa: E402
from app.services.index_membership_seeder import seed_from_csv  # noqa: E402
from app.wiring.bootstrap import (  # noqa: E402
    get_session_factory,
    initialize_process_runtime_services,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed stock_universe_index_membership from a constituent CSV.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the constituent CSV (columns: symbol,name).",
    )
    parser.add_argument(
        "--index",
        required=True,
        choices=[e.value for e in IndexName],
        help="Index name — must match an IndexName enum value.",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        help="Constituent snapshot date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--source",
        default="seed_v1",
        help="Source label written to membership rows (default: seed_v1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the CSV and report counts without committing writes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.csv.exists() or not args.csv.is_file():
        print(f"ERROR: CSV not found or is not a file: {args.csv}", file=sys.stderr)
        return 2

    initialize_process_runtime_services()
    session = get_session_factory()()
    try:
        counts = seed_from_csv(
            session,
            args.csv,
            index_name=args.index,
            as_of_date=args.as_of,
            source=args.source,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        # Typically raised by seed_from_csv on missing ``symbol`` header.
        # Return a distinct exit code so shell scripts can branch on it.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        session.close()

    mode = "DRY RUN (no writes)" if args.dry_run else "COMMITTED"
    print(
        f"[{mode}] {args.index} from {args.csv.name}: "
        f"added={counts.added} updated={counts.updated} "
        f"unchanged={counts.unchanged} skipped={counts.skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

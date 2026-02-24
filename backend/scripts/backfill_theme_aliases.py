#!/usr/bin/env python3
"""One-shot backfill for theme_aliases from legacy clusters and mentions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import SessionLocal  # noqa: E402
from app.services.theme_alias_backfill_service import ThemeAliasBackfillService  # noqa: E402


def _default_report_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "theme_alias_backfill_report.json"


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill theme_aliases from existing clusters and mentions.")
    parser.add_argument("--dry-run", action="store_true", help="Build report without writing alias rows")
    parser.add_argument(
        "--mention-limit",
        type=int,
        default=0,
        help="Optional cap on scanned theme_mentions rows (0 = all)",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(_default_report_path()),
        help="JSON report output path",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip interactive confirmation")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report_path = Path(args.report_file).expanduser()

    if not args.yes and not args.dry_run:
        answer = input("Backfill theme_aliases for historical rows. Continue? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return

    db = SessionLocal()
    try:
        service = ThemeAliasBackfillService(db)
        report = service.run(
            dry_run=bool(args.dry_run),
            mention_limit=int(args.mention_limit or 0),
        )
        _save_json(report_path, report)

        totals = report["totals"]
        print("Theme alias backfill summary")
        print(f"  dry_run: {report['dry_run']}")
        print(f"  clusters_scanned: {totals['clusters_scanned']}")
        print(f"  mentions_scanned: {totals['mentions_scanned']}")
        print(f"  candidate_groups: {totals['candidate_groups']}")
        print(f"  planned_inserts: {totals['planned_inserts']}")
        print(f"  inserted: {totals['inserted']}")
        print(f"  collisions_total: {totals['collisions_total']}")
        print("  collision_buckets:")
        for bucket, count in report["collisions"]["by_bucket"].items():
            print(f"    {bucket}: {count}")
        print(f"  report_file: {report_path}")
    except Exception as exc:
        db.rollback()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()

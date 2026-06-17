"""CLI: sync the latest IBD classification bundle from the GitHub release.

Thin wrapper around
``app.services.ibd_classification_bundle.sync_ibd_classification_from_github`` —
the same service the live-site Celery task uses. Imports the per-market bundle
published to the ``ibd-classification-data`` release (validating sha256),
preserving authoritative CSV/manual rows. Used by the daily static-site build
after the curated CSV is loaded.

Usage:
    python -m app.scripts.sync_ibd_classification_from_github --market SG [--allow-stale]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.services.ibd_classification_bundle import (
    NON_FATAL_SYNC_STATUSES,
    sync_ibd_classification_from_github,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--output-dir", default=str(Path(".tmp") / "ibd-classification"))
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()

    prepare_runtime()
    with SessionLocal() as db:
        result = sync_ibd_classification_from_github(
            db,
            market=args.market,
            allow_stale=args.allow_stale,
            output_dir=args.output_dir,
        )

    status = result["status"]
    print(f"IBD classification GitHub sync: status={status}")
    if status == "success":
        print("Imported:")
        for key, value in result["imported"].items():
            print(f"  - {key}: {value}")
        return 0

    if result.get("reason"):
        print(f"  - reason: {result['reason']}")
    # live_only / up_to_date are non-fatal; anything else is a soft failure.
    return 0 if status in NON_FATAL_SYNC_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())

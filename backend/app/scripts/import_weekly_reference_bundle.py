"""Import a weekly fundamentals reference bundle into the local database."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.wiring.bootstrap import get_provider_snapshot_service


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a weekly-reference bundle (.json or .json.gz).",
    )
    args = parser.parse_args()

    prepare_runtime()
    provider_snapshot_service = get_provider_snapshot_service()

    with SessionLocal() as db:
        stats = provider_snapshot_service.import_weekly_reference_bundle(
            db,
            input_path=Path(args.input),
        )

    print("Weekly reference import complete:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

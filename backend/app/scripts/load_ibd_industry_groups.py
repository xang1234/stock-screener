"""Load the tracked IBD industry mapping CSV into the local database."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.ibd_industry_service import IBDIndustryService


def _default_csv_path() -> Path:
    return repo_root() / "data" / "IBD_industry_group.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(_default_csv_path()),
        help="Path to the tracked IBD industry group CSV.",
    )
    args = parser.parse_args()

    prepare_runtime()

    with SessionLocal() as db:
        loaded = IBDIndustryService.load_from_csv(db, csv_path=args.csv)

    print("IBD industry group load complete:")
    print(f"  - csv: {Path(args.csv)}")
    print(f"  - loaded: {loaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

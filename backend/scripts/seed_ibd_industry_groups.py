#!/usr/bin/env python3
"""Seed the ibd_industry_groups table from the bundled canonical CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.services.ibd_industry_service import IBDIndustryService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed ibd_industry_groups from a CSV file.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional override path for the source CSV.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace any existing ibd_industry_groups rows instead of only seeding empty tables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with SessionLocal() as db:
        if args.replace:
            loaded = IBDIndustryService.load_from_csv(db, args.csv_path)
            print(f"Replaced IBD industry mappings with {loaded} rows")
            return 0

        loaded = IBDIndustryService.seed_if_empty(db, args.csv_path)
        if loaded > 0:
            print(f"Seeded IBD industry mappings with {loaded} rows")
        else:
            print("IBD industry mappings already present; no changes made")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

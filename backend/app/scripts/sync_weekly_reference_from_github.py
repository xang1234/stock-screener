"""Sync the latest weekly GitHub reference bundle into the local runtime database."""

from __future__ import annotations

import argparse

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.services.provider_snapshot_service import ProviderSnapshotService
from app.wiring.bootstrap import get_provider_snapshot_service


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market",
        required=True,
        choices=list(ProviderSnapshotService.SNAPSHOT_KEY_FUNDAMENTALS_BY_MARKET),
        help="Market code to sync from the GitHub weekly reference release.",
    )
    parser.add_argument(
        "--no-hydrate-cache",
        action="store_true",
        help="Import snapshot/universe rows without hydrating stock_fundamentals or Redis.",
    )
    parser.add_argument(
        "--hydrate-mode",
        choices=("static", "full"),
        default="static",
        help="How to hydrate imported fundamentals into local cache/database.",
    )
    args = parser.parse_args()

    prepare_runtime()
    provider_snapshot_service = get_provider_snapshot_service()

    with SessionLocal() as db:
        result = provider_snapshot_service.sync_weekly_reference_from_github(
            db,
            market=args.market,
            hydrate_cache=not args.no_hydrate_cache,
            hydrate_mode=args.hydrate_mode,
        )

    print("Weekly GitHub sync result:")
    for key, value in result.items():
        print(f"  - {key}: {value}")

    return 0 if result.get("status") in {"success", "up_to_date", "live_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

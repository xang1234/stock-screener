"""Sync the latest daily GitHub price bundle into the local runtime database."""

from __future__ import annotations

import argparse

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.wiring.bootstrap import get_daily_price_bundle_service


NON_FATAL_SYNC_STATUSES = {"success", "up_to_date", "live_only", "missing_manifest"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market",
        required=True,
        choices=list(DailyPriceBundleService.DAILY_PRICE_SUPPORTED_MARKETS),
        help="Market code to sync from the GitHub daily price release.",
    )
    parser.add_argument(
        "--warm-redis-symbols",
        type=int,
        default=None,
        help="Override the number of imported symbols to warm back into Redis immediately.",
    )
    args = parser.parse_args()

    prepare_runtime()
    service = get_daily_price_bundle_service()

    with SessionLocal() as db:
        result = service.sync_from_github(
            db,
            market=args.market,
            warm_redis_symbols=args.warm_redis_symbols,
        )

    print("Daily GitHub price sync result:")
    for key, value in result.items():
        print(f"  - {key}: {value}")

    return 0 if result.get("status") in NON_FATAL_SYNC_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build market-scoped daily price bundles for durable GitHub release publishing."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime, repo_root
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.wiring.bootstrap import get_daily_price_bundle_service


def _default_output_dir() -> Path:
    return repo_root() / ".tmp" / "daily-price"


def _default_bundle_name(market: str, as_of_date: date) -> str:
    return f"daily-price-{market.lower()}-{as_of_date.isoformat().replace('-', '')}.json.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--market",
        required=True,
        choices=list(DailyPriceBundleService.DAILY_PRICE_SUPPORTED_MARKETS),
        help="Market code to build the daily price bundle for.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Directory to receive the generated bundle and latest manifest.",
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Bundle asset filename. Defaults to daily-price-<market>-<YYYYMMDD>.json.gz",
    )
    parser.add_argument(
        "--latest-manifest-name",
        default=None,
        help="Filename for the latest-pointer manifest JSON.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Override the bundle as-of date (YYYY-MM-DD). Defaults to the latest completed trading day.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any active market symbol is missing price rows through the bundle as-of date.",
    )
    args = parser.parse_args()

    prepare_runtime()
    service = get_daily_price_bundle_service()

    market = args.market.upper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_as_of_date = (
        date.fromisoformat(args.as_of_date)
        if args.as_of_date
        else service.market_calendar.last_completed_trading_day(market)
    )
    if args.as_of_date:
        latest_completed = service.market_calendar.last_completed_trading_day(market)
        if resolved_as_of_date > latest_completed:
            raise SystemExit(
                f"{market} as-of date {resolved_as_of_date.isoformat()} is after "
                f"the latest completed trading day {latest_completed.isoformat()}"
            )
        if not service.market_calendar.is_trading_day(market, resolved_as_of_date):
            raise SystemExit(
                f"{market} as-of date {resolved_as_of_date.isoformat()} is not a trading day"
            )
    bundle_name = args.bundle_name or _default_bundle_name(market, resolved_as_of_date)
    latest_manifest_name = (
        args.latest_manifest_name or service.latest_manifest_name_for_market(market)
    )

    with SessionLocal() as db:
        stats = service.export_daily_price_bundle(
            db,
            market=market,
            output_path=output_dir / bundle_name,
            bundle_asset_name=bundle_name,
            latest_manifest_path=output_dir / latest_manifest_name,
            as_of_date=resolved_as_of_date,
            require_complete=args.require_complete,
        )

    print(f"Daily price bundle complete for {market}:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

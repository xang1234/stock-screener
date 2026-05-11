"""Import a daily price bundle into the local runtime database."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.services.daily_price_bundle_service import DailyPriceBundleService
from app.wiring.bootstrap import get_daily_price_bundle_service


def verify_daily_price_bundle_manifest(
    *,
    input_path: Path,
    manifest_path: Path,
    expected_market: str | None = None,
    expected_as_of_date: str | None = None,
    expected_bundle_asset_name: str | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_schema = DailyPriceBundleService.DAILY_PRICE_MANIFEST_SCHEMA_VERSION
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(
            "Unsupported daily price manifest schema version: "
            f"{manifest.get('schema_version')!r}"
        )

    bundle_asset_name = expected_bundle_asset_name or input_path.name
    expected_values = {
        "bundle_asset_name": bundle_asset_name,
        "bar_period": DailyPriceBundleService.DAILY_PRICE_BAR_PERIOD,
    }
    if expected_market is not None:
        expected_values["market"] = expected_market.upper()
    if expected_as_of_date is not None:
        expected_values["as_of_date"] = expected_as_of_date

    for key, expected_value in expected_values.items():
        actual_value = manifest.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Daily price manifest {key} mismatch: "
                f"{actual_value!r} != {expected_value!r}"
            )

    expected_sha256 = str(manifest.get("sha256") or "").strip()
    if not expected_sha256:
        raise ValueError("Daily price manifest is missing sha256")
    actual_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Daily price bundle checksum mismatch for {input_path.name}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a daily price bundle (.json or .json.gz).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest JSON to validate before importing the bundle.",
    )
    parser.add_argument(
        "--expected-market",
        default=None,
        help="Expected market code when validating --manifest.",
    )
    parser.add_argument(
        "--expected-as-of-date",
        default=None,
        help="Expected as-of date (YYYY-MM-DD) when validating --manifest.",
    )
    parser.add_argument(
        "--expected-bundle-asset-name",
        default=None,
        help="Expected bundle_asset_name when validating --manifest.",
    )
    parser.add_argument(
        "--warm-redis-symbols",
        type=int,
        default=0,
        help="Number of imported symbols to warm into Redis. Defaults to 0 for CI bootstrap jobs.",
    )
    args = parser.parse_args()

    prepare_runtime()
    service = get_daily_price_bundle_service()
    input_path = Path(args.input)
    if args.manifest:
        verify_daily_price_bundle_manifest(
            input_path=input_path,
            manifest_path=Path(args.manifest),
            expected_market=args.expected_market,
            expected_as_of_date=args.expected_as_of_date,
            expected_bundle_asset_name=args.expected_bundle_asset_name,
        )

    with SessionLocal() as db:
        stats = service.import_daily_price_bundle(
            db,
            input_path=input_path,
            warm_redis_symbols=args.warm_redis_symbols,
        )

    print("Daily price bundle import complete:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

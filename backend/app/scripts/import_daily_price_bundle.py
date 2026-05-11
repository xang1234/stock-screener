"""Import a daily price bundle into the local runtime database."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.wiring.bootstrap import get_daily_price_bundle_service


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a daily price bundle (.json or .json.gz).",
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

    with SessionLocal() as db:
        stats = service.import_daily_price_bundle(
            db,
            input_path=Path(args.input),
            warm_redis_symbols=args.warm_redis_symbols,
        )

    print("Daily price bundle import complete:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

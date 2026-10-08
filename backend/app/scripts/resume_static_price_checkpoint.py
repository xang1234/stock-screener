"""Import a compatible static price-stage checkpoint before the export (#502).

Runs after the daily-bundle seed. A compatible checkpoint for the session the
export will use is imported in checkpoint mode, so the refresh fetches only
the symbols it still lacks. Any other outcome (missing, incompatible,
invalid) leaves the database as seeded and the export takes the normal path,
so this step never fails the job.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from app.database import SessionLocal
from app.scripts._runtime import prepare_runtime
from app.scripts.export_static_site import _resolve_latest_completed_trading_date
from app.services.static_price_checkpoint import resume_price_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    args = parser.parse_args(argv)
    market = args.market.strip().upper()

    prepare_runtime()
    as_of_date = _resolve_latest_completed_trading_date(market)
    with SessionLocal() as db:
        result = resume_price_checkpoint(db, market=market, as_of_date=as_of_date)
    print(
        f"[price checkpoint] {market} {as_of_date.isoformat()} "
        f"checkpoint status={result.get('status')} {result}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

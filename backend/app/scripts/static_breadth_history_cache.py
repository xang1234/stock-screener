"""Carry a market's ``market_breadth`` rows between ephemeral static-site builds.

Each CI market build starts on an empty Postgres, so the breadth coordinator
recomputed the whole ~150-session exposure window every run (~32 min for US).
``export`` saves the market's rows after a successful build; ``import``
restores them before the next one.

Only aggregate rows travel. Import drops the newest ``RECOMPUTED_TAIL_SESSIONS``
sessions so the coordinator recomputes them. That covers late provider bars,
and it regenerates the contributor snapshots, which are only retained (and
exported) for those newest sessions anyway. Older rows are reused only while
their ``eligibility_signature`` still matches the current universe; the
coordinator recomputes the rest.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.market_breadth import MarketBreadth
from app.scripts.export_options_history import write_history_bundle
from app.services.breadth import (
    CONTRIBUTOR_RETENTION_SESSIONS,
    CURRENT_BREADTH_CALCULATION_REVISION,
)
from app.services.static_breadth_history_coordinator import (
    STATIC_BREADTH_RATIO_RECOMPUTE_TRADING_DAYS,
)

SCHEMA = "static-breadth-history-v1"
RECOMPUTED_TAIL_SESSIONS = max(
    CONTRIBUTOR_RETENTION_SESSIONS, STATIC_BREADTH_RATIO_RECOMPUTE_TRADING_DAYS
)
_COLUMNS = tuple(
    column.name
    for column in MarketBreadth.__table__.columns
    if column.name not in {"id", "created_at"}
)


def export_bundle(db: Session, *, market: str) -> dict[str, Any]:
    rows = (
        db.query(MarketBreadth)
        .filter(
            MarketBreadth.market == market,
            MarketBreadth.calculation_revision == CURRENT_BREADTH_CALCULATION_REVISION,
        )
        .order_by(MarketBreadth.date)
        .all()
    )
    return {
        "schema": SCHEMA,
        "market": market,
        "calculation_revision": CURRENT_BREADTH_CALCULATION_REVISION,
        "rows": [{name: _jsonable(getattr(row, name)) for name in _COLUMNS} for row in rows],
    }


def _jsonable(value: Any) -> Any:
    return value.isoformat() if isinstance(value, date) else value


def import_bundle(db: Session, bundle: dict[str, Any], *, market: str) -> dict[str, Any]:
    if bundle.get("schema") != SCHEMA or bundle.get("market") != market:
        return {"status": "skipped", "reason": "incompatible_bundle"}
    if bundle.get("calculation_revision") != CURRENT_BREADTH_CALCULATION_REVISION:
        return {"status": "skipped", "reason": "calculation_revision_changed"}
    rows = bundle.get("rows") or []
    # A row without a current column would restore it as NULL and never be
    # recomputed, so a bundle from before a column was added is discarded.
    if any(not set(_COLUMNS) <= row.keys() for row in rows):
        return {"status": "skipped", "reason": "columns_changed"}
    dates = sorted({row["date"] for row in rows})
    reusable = set(dates[:-RECOMPUTED_TAIL_SESSIONS])
    existing = {
        value.isoformat()
        for (value,) in db.query(MarketBreadth.date).filter(MarketBreadth.market == market)
    }
    restored = 0
    for row in rows:
        if row["date"] not in reusable or row["date"] in existing:
            continue
        values = {name: row.get(name) for name in _COLUMNS}
        values["date"] = date.fromisoformat(row["date"])
        db.add(MarketBreadth(**values))
        restored += 1
    db.commit()
    return {
        "status": "restored",
        "restored_dates": restored,
        "dropped_tail_dates": len(dates) - len(reusable),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("--market", required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args(argv)
    market = args.market.strip().upper()
    if args.action == "import" and not args.path.is_file():
        print(json.dumps({"status": "missing"}))
        return 0

    from app.database import SessionLocal
    from app.scripts._runtime import prepare_runtime

    prepare_runtime()
    with SessionLocal() as db:
        if args.action == "export":
            bundle = export_bundle(db, market=market)
            write_history_bundle(args.path, bundle)
            result = {"status": "exported", "dates": len(bundle["rows"])}
        else:
            with gzip.open(args.path, "rt", encoding="utf-8") as handle:
                result = import_bundle(db, json.load(handle), market=market)
    print(json.dumps({"market": market, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

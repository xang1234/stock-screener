#!/usr/bin/env python
"""Seed the German (DE) stock universe and DAX/MDAX/SDAX index memberships.

Usage:
    python scripts/seed_de_universe.py \\
        --csv data/de_dax40_constituents.csv \\
        --as-of 2026-05-09

CSV schema: ``symbol,name,exchange,index`` (header row required).
- ``symbol``: yfinance ticker, e.g. ``SAP.DE`` or ``ALV.F``
- ``name``: human-readable company name
- ``exchange``: ``XETR`` (Xetra) or ``XFRA`` (Frankfurt)
- ``index``: optional index code — one of ``DAX``, ``MDAX``, ``SDAX``. Leave
  blank to insert the row into stock_universe without index membership.

The script:
  1. Inserts/updates rows into ``stock_universe`` via ``ingest_de_from_csv``
     (deterministic canonicalization through the DE adapter).
  2. Seeds ``stock_universe_index_membership`` rows for each non-blank
     ``index`` value, grouped by index name.

Use ``--dry-run`` to walk the CSV without committing writes.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.index_membership_seeder import seed_from_csv  # noqa: E402
from app.services.stock_universe_service import StockUniverseService  # noqa: E402
from app.wiring.bootstrap import (  # noqa: E402
    get_session_factory,
    initialize_process_runtime_services,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the DE constituent CSV (columns: symbol,name,exchange,index).",
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        required=True,
        help="Constituent snapshot date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--source",
        default="de_manual_csv",
        help="Source label written to ingestion + membership rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk the CSV and report counts without committing writes.",
    )
    return parser


def _split_universe_and_index_csvs(csv_path: Path) -> tuple[str, dict[str, str]]:
    """Read the master CSV; return (universe_csv_text, {index_name: index_csv_text})."""
    rows: list[dict[str, str]] = []
    by_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV {csv_path} is empty or missing a header row.")
        required = {"symbol"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            row = {k: (v or "").strip() for k, v in raw.items() if k}
            if not row.get("symbol"):
                continue
            rows.append(row)
            index_name = (row.get("index") or "").strip().upper()
            if index_name:
                by_index[index_name].append(row)

    universe_buffer = io.StringIO()
    universe_writer = csv.DictWriter(
        universe_buffer,
        fieldnames=["symbol", "name", "exchange"],
    )
    universe_writer.writeheader()
    for row in rows:
        universe_writer.writerow({
            "symbol": row.get("symbol", ""),
            "name": row.get("name", ""),
            "exchange": row.get("exchange") or "XETR",
        })

    index_csvs: dict[str, str] = {}
    for index_name, index_rows in by_index.items():
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["symbol", "name"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow({"symbol": row.get("symbol", ""), "name": row.get("name", "")})
        index_csvs[index_name] = buffer.getvalue()

    return universe_buffer.getvalue(), index_csvs


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("seed_de_universe")

    if not args.csv.exists() or not args.csv.is_file():
        print(f"ERROR: CSV not found or is not a file: {args.csv}", file=sys.stderr)
        return 2

    universe_csv_text, index_csvs = _split_universe_and_index_csvs(args.csv)

    initialize_process_runtime_services()
    session_factory = get_session_factory()

    universe_service = StockUniverseService()

    with session_factory() as db:
        if args.dry_run:
            # Dry-run: parse rows through the DE adapter without writing.
            rows = StockUniverseService._parse_de_csv_rows(universe_csv_text)
            canonicalized = universe_service._de_ingestion.canonicalize_rows(
                rows,
                source_name=args.source,
                snapshot_id=f"de:dryrun:{args.as_of}",
                snapshot_as_of=args.as_of,
            )
            log.info(
                "DE universe dry-run: canonical=%d rejected=%d",
                len(canonicalized.canonical_rows),
                len(canonicalized.rejected_rows),
            )
            for index_name, index_csv in index_csvs.items():
                line_count = sum(1 for _ in csv.reader(io.StringIO(index_csv))) - 1
                log.info("DE index dry-run: %s rows=%d", index_name, line_count)
            return 0

        result = universe_service.ingest_de_from_csv(
            db,
            universe_csv_text,
            source_name=args.source,
            snapshot_as_of=args.as_of,
            strict=False,
        )
        log.info(
            "DE universe ingest: added=%d updated=%d total=%d rejected=%d",
            result["added"],
            result["updated"],
            result["total"],
            result["rejected"],
        )

    # Seed index memberships in their own session — index membership writes
    # use a different commit boundary from universe ingest. seed_from_csv
    # accepts a Path; use a NamedTemporaryFile so the script doesn't depend
    # on the input CSV's parent directory being writable.
    for index_name, index_csv in index_csvs.items():
        with session_factory() as db:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=f".de.{index_name.lower()}.csv",
                encoding="utf-8",
                delete=False,
            ) as tmp_handle:
                tmp_handle.write(index_csv)
                tmp_path = Path(tmp_handle.name)
            try:
                counts = seed_from_csv(
                    db,
                    tmp_path,
                    index_name=index_name,
                    as_of_date=args.as_of,
                    source=args.source,
                    dry_run=False,
                )
                log.info(
                    "DE index seed: %s added=%d updated=%d unchanged=%d skipped=%d",
                    index_name,
                    counts.added,
                    counts.updated,
                    counts.unchanged,
                    counts.skipped,
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

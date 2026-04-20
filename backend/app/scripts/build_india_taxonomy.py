"""Build the tracked India taxonomy CSV from the Nexus map plus NSE/BSE listings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.config.settings import get_project_root
from app.services.official_market_universe_source_service import (
    OfficialMarketUniverseSourceService,
)

_OUTPUT_FIELDS = [
    "Symbol",
    "Exchange",
    "Industry (Sector)",
    "Subgroup (Theme)",
    "Sub-industry",
    "Source Symbol",
    "Match Type",
    "ISIN",
    "Company Name",
    "Primary Company Name",
]
_MATCH_PRIORITY = {
    "primary_local": 0,
    "nse_local": 1,
    "bse_security_id": 2,
    "bse_code": 3,
}


def _default_nexus_csv() -> Path:
    return get_project_root() / "data" / "nexus_market_map_stocks.csv"


def _default_output_csv() -> Path:
    return get_project_root() / "data" / "india-deep.csv"


def _read_nexus_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_nse_rows(
    *,
    service: OfficialMarketUniverseSourceService,
    csv_path: Path | None,
) -> list[dict[str, Any]]:
    if csv_path is None:
        return list(service.fetch_nse_snapshot().rows)
    return service.parse_nse_rows(csv_path.read_bytes())


def _load_bse_rows(
    *,
    service: OfficialMarketUniverseSourceService,
    json_path: Path | None,
) -> list[dict[str, Any]]:
    if json_path is None:
        return list(service.fetch_bse_snapshot().rows)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return service.parse_bse_rows(json.dumps(payload).encode("utf-8"))


def build_india_taxonomy_rows(
    *,
    nexus_rows: list[dict[str, str]],
    nse_rows: list[dict[str, Any]],
    bse_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], Counter]:
    primary_by_isin: dict[str, dict[str, str]] = {}
    venues_by_isin: dict[str, list[dict[str, str]]] = {}
    for row in nse_rows:
        isin = str(row["isin"])
        venue = {
            "symbol": str(row["symbol"]),
            "exchange": str(row["exchange"]),
            "isin": isin,
            "name": str(row["name"]),
        }
        primary_by_isin[isin] = venue
        venues_by_isin.setdefault(isin, []).append(venue)
    for row in bse_rows:
        isin = str(row["isin"])
        venue = {
            "symbol": str(row["symbol"]),
            "exchange": str(row["exchange"]),
            "isin": isin,
            "name": str(row["name"]),
        }
        primary_by_isin.setdefault(isin, venue)
        venues_by_isin.setdefault(isin, []).append(venue)

    primary_by_local = {
        entry["symbol"].split(".", 1)[0]: entry
        for entry in primary_by_isin.values()
    }
    nse_by_local = {
        str(row["symbol"]).split(".", 1)[0]: row
        for row in nse_rows
    }
    bse_by_security_id = {
        str(row.get("security_id") or "").strip().upper(): row
        for row in bse_rows
        if str(row.get("security_id") or "").strip()
    }
    bse_by_code = {
        str(row["symbol"]).split(".", 1)[0]: row
        for row in bse_rows
    }

    counts: Counter = Counter()
    selected_by_symbol: dict[str, dict[str, str]] = {}

    for source_row in nexus_rows:
        token = str(source_row.get("Symbol") or "").strip().upper()
        if not token:
            continue

        match_type: str | None = None
        primary: dict[str, str] | None = None
        if token in primary_by_local:
            primary = primary_by_local[token]
            match_type = "primary_local"
        elif token in nse_by_local:
            primary = primary_by_isin[str(nse_by_local[token]["isin"])]
            match_type = "nse_local"
        elif token in bse_by_security_id:
            primary = primary_by_isin[str(bse_by_security_id[token]["isin"])]
            match_type = "bse_security_id"
        elif token in bse_by_code:
            primary = primary_by_isin[str(bse_by_code[token]["isin"])]
            match_type = "bse_code"

        if primary is None or match_type is None:
            counts["unmatched"] += 1
            continue

        counts[match_type] += 1
        venues = sorted(
            venues_by_isin.get(primary["isin"], [primary]),
            key=lambda venue: (venue["exchange"] != "XNSE", venue["symbol"]),
        )
        for venue in venues:
            output_row = {
                "Symbol": venue["symbol"],
                "Exchange": venue["exchange"],
                "Industry (Sector)": str(source_row.get("Industry (Sector)") or "").strip(),
                "Subgroup (Theme)": str(source_row.get("Subgroup (Theme)") or "").strip(),
                "Sub-industry": str(source_row.get("Sub-industry") or "").strip(),
                "Source Symbol": token,
                "Match Type": match_type,
                "ISIN": primary["isin"],
                "Company Name": str(source_row.get("Company Name") or "").strip(),
                "Primary Company Name": primary["name"],
            }
            existing = selected_by_symbol.get(venue["symbol"])
            if existing is not None:
                counts["duplicate_primary"] += 1
                if _MATCH_PRIORITY[match_type] >= _MATCH_PRIORITY[existing["Match Type"]]:
                    continue
            selected_by_symbol[venue["symbol"]] = output_row

    return (
        [selected_by_symbol[symbol] for symbol in sorted(selected_by_symbol)],
        counts,
    )


def _write_output(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nexus-csv",
        default=str(_default_nexus_csv()),
        help="Path to the raw Nexus classification CSV.",
    )
    parser.add_argument(
        "--nse-csv",
        default=None,
        help="Optional path to a downloaded NSE EQUITY_L.csv snapshot.",
    )
    parser.add_argument(
        "--bse-json",
        default=None,
        help="Optional path to a downloaded BSE ListofScripData JSON snapshot.",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_csv()),
        help="Path to write the normalized India taxonomy CSV.",
    )
    args = parser.parse_args()

    service = OfficialMarketUniverseSourceService()
    nexus_rows = _read_nexus_rows(Path(args.nexus_csv))
    nse_rows = _load_nse_rows(
        service=service,
        csv_path=Path(args.nse_csv) if args.nse_csv else None,
    )
    bse_rows = _load_bse_rows(
        service=service,
        json_path=Path(args.bse_json) if args.bse_json else None,
    )
    output_rows, counts = build_india_taxonomy_rows(
        nexus_rows=nexus_rows,
        nse_rows=nse_rows,
        bse_rows=bse_rows,
    )
    output_path = Path(args.output)
    _write_output(output_path, output_rows)

    print("India taxonomy build complete:")
    print(f"  - nexus_csv: {Path(args.nexus_csv)}")
    print(f"  - output: {output_path}")
    print(f"  - rows: {len(output_rows)}")
    for key in sorted(counts):
        print(f"  - {key}: {counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

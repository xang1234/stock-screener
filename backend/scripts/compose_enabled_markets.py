#!/usr/bin/env python3
"""Derive Docker Compose worker profiles and Celery queues from ENABLED_MARKETS."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.tasks.market_queues import SUPPORTED_MARKETS  # noqa: E402


def normalize_markets(raw: str | None) -> list[str]:
    values = [part.strip().upper() for part in (raw or "").split(",") if part.strip()]
    if not values:
        values = ["US"]

    normalized: list[str] = []
    unsupported: list[str] = []
    for market in values:
        if market not in SUPPORTED_MARKETS:
            unsupported.append(market)
            continue
        if market not in normalized:
            normalized.append(market)

    if unsupported:
        supported = ", ".join(SUPPORTED_MARKETS)
        invalid = ", ".join(unsupported)
        raise ValueError(f"Unsupported market(s): {invalid}. Supported markets: {supported}")

    return normalized


def compose_profiles_for_markets(markets: Sequence[str]) -> list[str]:
    return [f"market-{market.lower()}" for market in markets]


def datafetch_queues_for_markets(markets: Sequence[str]) -> list[str]:
    queues = ["data_fetch_shared"]
    queues.extend(f"data_fetch_{market.lower()}" for market in markets)
    return queues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("profiles", "queues", "env"),
        help="Output Compose profiles, data-fetch queues, or shell env assignments.",
    )
    parser.add_argument(
        "--markets",
        default=os.environ.get("ENABLED_MARKETS", "US"),
        help="Comma-separated market codes. Defaults to ENABLED_MARKETS or US.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        markets = normalize_markets(args.markets)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    profiles = ",".join(compose_profiles_for_markets(markets))
    queues = ",".join(datafetch_queues_for_markets(markets))

    if args.command == "profiles":
        print(profiles)
    elif args.command == "queues":
        print(queues)
    else:
        print(f"COMPOSE_PROFILES={profiles}")
        print(f"DATAFETCH_QUEUES={queues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

from app.deployment.enabled_markets import (  # noqa: E402
    DEFAULT_DEPLOYMENT_ENABLED_MARKETS,
    normalize_deployment_enabled_markets,
)
from app.domain.markets import market_registry  # noqa: E402


SUPPORTED_MARKETS: tuple[str, ...] = market_registry.supported_market_codes()


def normalize_markets(raw: str | None) -> list[str]:
    return normalize_deployment_enabled_markets(raw)


def compose_profiles_for_markets(markets: Sequence[str]) -> list[str]:
    return [f"market-{market.lower()}" for market in markets]


def datafetch_queues_for_markets(markets: Sequence[str]) -> list[str]:
    queues = ["data_fetch_shared"]
    queues.extend(f"data_fetch_{market.lower()}" for market in markets)
    return queues


def user_scans_queues_for_markets(markets: Sequence[str]) -> list[str]:
    queues = ["user_scans_shared"]
    queues.extend(f"user_scans_{market.lower()}" for market in markets)
    return queues


def market_jobs_queues_for_markets(markets: Sequence[str]) -> list[str]:
    # market_jobs has no shared fallback queue — every task requires an explicit market.
    return [f"market_jobs_{market.lower()}" for market in markets]


QUEUE_BUILDERS = {
    "datafetch": datafetch_queues_for_markets,
    "userscans": user_scans_queues_for_markets,
    "marketjobs": market_jobs_queues_for_markets,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("profiles", "queues", "env"),
        help="Output Compose profiles, queues for a queue set, or shell env assignments.",
    )
    parser.add_argument(
        "--markets",
        default=os.environ.get(
            "ENABLED_MARKETS",
            ",".join(DEFAULT_DEPLOYMENT_ENABLED_MARKETS),
        ),
        help="Comma-separated market codes. Defaults to ENABLED_MARKETS or US.",
    )
    parser.add_argument(
        "--queue-set",
        choices=tuple(QUEUE_BUILDERS),
        default="datafetch",
        help="Which queue family to compute for the 'queues' command. Defaults to datafetch.",
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
    queues = ",".join(QUEUE_BUILDERS[args.queue_set](markets))

    if args.command == "profiles":
        print(profiles)
    elif args.command == "queues":
        print(queues)
    else:
        datafetch_queues = ",".join(datafetch_queues_for_markets(markets))
        userscans_queues = ",".join(user_scans_queues_for_markets(markets))
        marketjobs_queues = ",".join(market_jobs_queues_for_markets(markets))
        print(f"COMPOSE_PROFILES={profiles}")
        print(f"DATAFETCH_QUEUES={datafetch_queues}")
        print(f"USERSCANS_QUEUES={userscans_queues}")
        print(f"MARKETJOBS_QUEUES={marketjobs_queues}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

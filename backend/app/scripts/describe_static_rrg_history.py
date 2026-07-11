"""Describe canonical rolling-RRG capability and paths for automation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.static_rrg_history_contract import build_static_rrg_history_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()

    plan = build_static_rrg_history_plan(
        market=args.market,
        directory=Path(args.directory),
    )
    print(json.dumps(plan.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

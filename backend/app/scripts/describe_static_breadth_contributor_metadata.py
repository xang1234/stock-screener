"""Describe canonical rolling breadth-contributor metadata paths."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.services.static_breadth_contributor_metadata_contract import (
    build_static_breadth_contributor_metadata_plan,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args(argv)

    plan = build_static_breadth_contributor_metadata_plan(
        market=args.market,
        directory=Path(args.directory),
    )
    print(json.dumps(plan.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

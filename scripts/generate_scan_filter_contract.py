#!/usr/bin/env python3
"""Package the canonical scan-field contract and dictionary for each runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS_BY_SOURCE = {
    ROOT / "contracts" / "scan_filter_fields.json": (
        ROOT / "backend" / "app" / "domain" / "scanning" / "scan_filter_fields.json",
        ROOT / "frontend" / "src" / "features" / "scan" / "scanFilterFields.json",
    ),
    ROOT / "contracts" / "scan_field_dictionary.json": (
        ROOT / "stockscreen-mcp" / "src" / "stockscreen_mcp" / "fields.json",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a packaged artifact differs from the canonical contract",
    )
    args = parser.parse_args()

    copies = [
        (target, source.read_bytes())
        for source, targets in TARGETS_BY_SOURCE.items()
        for target in targets
    ]
    stale = [target for target, content in copies if not target.exists() or target.read_bytes() != content]
    if args.check:
        for target in stale:
            print(f"stale generated scan-field contract: {target.relative_to(ROOT)}")
        return 1 if stale else 0

    for target, content in copies:
        target.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

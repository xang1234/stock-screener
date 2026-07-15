#!/usr/bin/env python3
"""Package the canonical scan-field contract for each runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "contracts" / "scan_filter_fields.json"
TARGETS = (
    ROOT / "backend" / "app" / "domain" / "scanning" / "scan_filter_fields.json",
    ROOT / "frontend" / "src" / "features" / "scan" / "scanFilterFields.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a packaged artifact differs from the canonical contract",
    )
    args = parser.parse_args()
    content = SOURCE.read_bytes()

    stale = [target for target in TARGETS if not target.exists() or target.read_bytes() != content]
    if args.check:
        for target in stale:
            print(f"stale generated scan-field contract: {target.relative_to(ROOT)}")
        return 1 if stale else 0

    for target in TARGETS:
        target.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

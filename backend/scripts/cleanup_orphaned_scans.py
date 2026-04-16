"""Synchronously clean up cancelled and stale unfinished scans."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.tasks.cache_tasks import run_orphaned_scan_cleanup


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    result = run_orphaned_scan_cleanup()
    if "error" in result:
        logging.error("Orphaned scan cleanup failed: %s", result["error"])
        return 1
    logging.info(
        "Manual orphaned scan cleanup finished: deleted %s scans and %s results",
        result["deleted_scans"],
        result["deleted_results"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

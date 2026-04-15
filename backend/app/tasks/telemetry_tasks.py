"""Celery tasks for telemetry governance (bead asia.10.4).

Weekly audit task produces a signed governance report from the per-market
telemetry event log and alert table. Artifacts land at:

    data/governance/telemetry_audit/YYYY-MM-DD.{json,md,sha256}

Two complementary integrity checks:
  - content_hash (inside JSON): SHA-256 over canonical compact JSON with
    content_hash nulled — verified programmatically by re-canonicalizing.
  - file_hash (.sha256 sidecar): SHA-256 of the raw .json file bytes —
    verified with `sha256sum -c <stamp>.sha256` from the report directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..celery_app import celery_app
from ..database import SessionLocal
from ..services.telemetry.weekly_audit import (
    GovernanceReport,
    render_json,
    render_markdown,
    run_weekly_audit,
)

logger = logging.getLogger(__name__)

# Default output directory. Overridable for tests and for deployments that
# prefer a non-default mount. Resolved lazily so importing this module does
# not touch the filesystem.
_DEFAULT_REPORT_SUBPATH = Path("data/governance/telemetry_audit")


def _resolve_output_dir(override: Optional[str]) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("TELEMETRY_AUDIT_REPORT_DIR")
    if env:
        return Path(env)
    # Project-root-relative: backend/app/tasks/telemetry_tasks.py -> project root is 3 levels up.
    project_root = Path(__file__).resolve().parents[3]
    return project_root / _DEFAULT_REPORT_SUBPATH


def _write_report_artifacts(report: GovernanceReport, out_dir: Path) -> Dict[str, str]:
    """Write JSON, Markdown, and separate SHA-256 file. Return paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at[:10]  # YYYY-MM-DD prefix of ISO datetime

    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"
    hash_path = out_dir / f"{stamp}.sha256"

    json_blob = render_json(report)
    md_blob = render_markdown(report)

    json_path.write_text(json_blob, encoding="utf-8")
    md_path.write_text(md_blob, encoding="utf-8")
    # The .sha256 file is the *file-level* integrity check — sha256sum hashes
    # raw file bytes, so we must hash json_blob (the bytes actually written to
    # disk) not report.content_hash (which is SHA-256 of compact JSON with a
    # null content_hash field — a different document).
    #
    # Two complementary verification paths:
    #   sha256sum -c <stamp>.sha256          → file not truncated/corrupted
    #   programmatic Python re-canonicalize  → data not semantically altered
    #                                          (see governance_report.md)
    file_hash = hashlib.sha256(json_blob.encode("utf-8")).hexdigest()
    hash_path.write_text(
        f"{file_hash}  {json_path.name}\n", encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "sha256": str(hash_path),
    }


@celery_app.task(name="app.tasks.telemetry_tasks.weekly_telemetry_audit")
def weekly_telemetry_audit(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Generate and persist the weekly telemetry governance report.

    Returns a dict with artifact paths + content hash so the task result
    (stored in Celery's result backend) is a verifiable pointer to the
    artifact on disk.
    """
    out_dir = _resolve_output_dir(output_dir)
    db = SessionLocal()
    try:
        report = run_weekly_audit(db, now=datetime.now(timezone.utc))
    finally:
        db.close()

    paths = _write_report_artifacts(report, out_dir)
    logger.info(
        "telemetry governance report generated: content_hash=%s json=%s",
        report.content_hash, paths["json"],
    )
    return {
        "content_hash": report.content_hash,
        "generated_at": report.generated_at,
        "window_start": report.window_start,
        "window_end": report.window_end,
        "artifacts": paths,
    }

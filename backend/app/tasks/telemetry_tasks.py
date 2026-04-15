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

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ..celery_app import celery_app
from ..config.settings import get_project_root
from ..database import SessionLocal
from ..services.governance.signed_artifact import write_signed_artifact_trio
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
    return get_project_root() / _DEFAULT_REPORT_SUBPATH


def _write_report_artifacts(report: GovernanceReport, out_dir: Path) -> Dict[str, str]:
    """Write JSON, Markdown, and separate SHA-256 file. Return paths written.

    The dual-hash contract (content_hash inside JSON for semantic integrity,
    file_hash in .sha256 for ``sha256sum -c``) is enforced inside the shared
    helper — see signed_artifact.write_signed_artifact_trio.
    """
    return write_signed_artifact_trio(
        out_dir=out_dir,
        stamp=report.generated_at[:10],
        json_blob=render_json(report),
        md_blob=render_markdown(report),
    )


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

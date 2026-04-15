"""Launch-gate artifact writer (bead asia.11.1).

Mirrors the dual-hash pattern from ``backend/app/tasks/telemetry_tasks.py``:
- ``content_hash`` (inside JSON): SHA-256 over canonical compact JSON
  with content_hash nulled — programmatic integrity check.
- ``.sha256`` sidecar: SHA-256 of raw .json file bytes — verified with
  ``sha256sum -c <stamp>.sha256``.

Default output dir: ``data/governance/launch_gates/`` (sibling to the
weekly telemetry audit dir).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from ...config.settings import get_project_root
from .launch_gates import LaunchGateReport, render_json, render_markdown
from .signed_artifact import write_signed_artifact_trio


_DEFAULT_REPORT_SUBPATH = Path("data/governance/launch_gates")


def resolve_output_dir(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("LAUNCH_GATE_REPORT_DIR")
    if env:
        return Path(env)
    return get_project_root() / _DEFAULT_REPORT_SUBPATH


def write_artifacts(report: LaunchGateReport, out_dir: Path) -> Dict[str, str]:
    """Write JSON, Markdown, and SHA-256 file. Return absolute paths.

    Filename stem = the date portion of ``generated_at`` (YYYY-MM-DD) plus
    ``-{verdict}`` so a directory listing immediately surfaces the outcome.
    """
    return write_signed_artifact_trio(
        out_dir=out_dir,
        stamp=f"{report.generated_at[:10]}-{report.verdict}",
        json_blob=render_json(report),
        md_blob=render_markdown(report),
    )

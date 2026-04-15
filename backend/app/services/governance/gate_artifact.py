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

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

from .launch_gates import LaunchGateReport, render_json, render_markdown


_DEFAULT_REPORT_SUBPATH = Path("data/governance/launch_gates")


def resolve_output_dir(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("LAUNCH_GATE_REPORT_DIR")
    if env:
        return Path(env)
    # backend/app/services/governance/gate_artifact.py → project root is 4 levels up.
    project_root = Path(__file__).resolve().parents[4]
    return project_root / _DEFAULT_REPORT_SUBPATH


def write_artifacts(report: LaunchGateReport, out_dir: Path) -> Dict[str, str]:
    """Write JSON, Markdown, and SHA-256 file. Return absolute paths.

    Filename stem = the date portion of ``generated_at`` (YYYY-MM-DD) plus
    ``-{verdict}`` so a directory listing immediately surfaces the outcome.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{report.generated_at[:10]}-{report.verdict}"

    json_path = out_dir / f"{stamp}.json"
    md_path = out_dir / f"{stamp}.md"
    hash_path = out_dir / f"{stamp}.sha256"

    json_blob = render_json(report)
    md_blob = render_markdown(report)

    json_path.write_text(json_blob, encoding="utf-8")
    md_path.write_text(md_blob, encoding="utf-8")
    file_hash = hashlib.sha256(json_blob.encode("utf-8")).hexdigest()
    hash_path.write_text(f"{file_hash}  {json_path.name}\n", encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "sha256": str(hash_path),
    }

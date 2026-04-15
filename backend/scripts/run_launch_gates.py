#!/usr/bin/env python3
"""CLI entry point for the ASIA v2 launch-gate runner (bead asia.11.1).

Usage:
    python backend/scripts/run_launch_gates.py [options]

Options:
    --evidence GATE_ID=PATH ...   Attach external evidence files (G5/G6/G7).
                                  Pass multiple times: --evidence G5=qa.json --evidence G7=load.json
    --output-dir PATH             Override default output directory.
    --no-db                       Skip DB-backed gates (G2, G4) — they will report MISSING_EVIDENCE.

Exit codes:
    0 — verdict PASS (all hard gates pass)
    1 — verdict NO_GO (at least one hard gate MISSING_EVIDENCE)
    2 — verdict FAIL (at least one hard gate FAIL)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly: python backend/scripts/run_launch_gates.py
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.governance.launch_gates import (  # noqa: E402
    GateVerdict,
    run_all_gates,
)
from app.services.governance.gate_artifact import (  # noqa: E402
    resolve_output_dir,
    write_artifacts,
)


def _parse_evidence(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            print(f"--evidence requires GATE_ID=PATH (got {item!r})", file=sys.stderr)
            sys.exit(64)  # EX_USAGE
        gate_id, path = item.split("=", 1)
        out[gate_id.strip()] = path.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASIA v2 launch gates and emit signed artifact.")
    parser.add_argument("--evidence", action="append",
                        help="GATE_ID=PATH for external evidence (repeatable).")
    parser.add_argument("--output-dir", default=None,
                        help="Override report output directory.")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip DB-backed gates (G2, G4).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    external = _parse_evidence(args.evidence)

    db = None
    if not args.no_db:
        try:
            from app.database import SessionLocal
            db = SessionLocal()
        except Exception as exc:
            # Surface why DB-backed gates will report MISSING_EVIDENCE.
            print(f"[warn] DB session unavailable ({exc}); G2/G4 will report MISSING_EVIDENCE.",
                  file=sys.stderr)

    try:
        report = run_all_gates(
            project_root=project_root, db=db, external_evidence=external,
        )
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    out_dir = resolve_output_dir(args.output_dir)
    paths = write_artifacts(report, out_dir)

    print(f"Verdict: {report.verdict.upper()}")
    print(f"Hard gates: {report.hard_gate_count} total · "
          f"{report.hard_passed} pass · {report.hard_failed} fail · "
          f"{report.hard_missing_evidence} missing evidence")
    print(f"Content hash: {report.content_hash}")
    print(f"Artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    if report.verdict == GateVerdict.PASS:
        return 0
    if report.verdict == GateVerdict.NO_GO:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

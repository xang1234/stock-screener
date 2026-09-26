"""Run a company-exposure slice gate on real PostgreSQL.

Collects and runs the slice's test paths with the case-tag plugin, then
requires, for the slice's reviewed manifest:

* the live connection is PostgreSQL (SQLite never substitutes);
* every required ``(case, layer)`` pair has at least one collected test and
  every collected test carrying that pair passed — a passing sibling cannot
  mask a skipped, xfailed, xpassed or failed one;
* every named invariant node was collected and passed;
* the pytest session itself succeeded.

The run artifact records exact node IDs, outcomes, database identity and the
source revision. The existing Economic Taxonomy gate is unchanged; only its
PostgreSQL identity check is reused.

    cd backend
    DATABASE_URL=postgresql://ci:ci@localhost:5432/ci STOCKSCANNER_TEST_ALLOW_POSTGRES=1 \\
        ./venv/bin/python scripts/run_required_company_exposure_postgres.py --slice S1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.domain.company_exposure.manifest import (
    CaseInventory,
    CaseLayerRequirement,
    CaseRunReport,
    CollectedCase,
    validate_case_report,
)

DEFAULT_MANIFEST = BACKEND_ROOT / "tests/required_company_exposure_cases.json"
POSTGRES_MODE = "postgresql"


@dataclass(frozen=True, slots=True)
class SliceManifest:
    slice_id: str
    paths: tuple[str, ...]
    required: frozenset[CaseLayerRequirement]
    invariants: dict[str, str]


@dataclass(frozen=True, slots=True)
class GateDecision:
    passed: bool
    missing_requirements: tuple[str, ...]
    failed_nodeids: tuple[str, ...]
    missing_invariants: tuple[str, ...]
    failed_invariants: tuple[str, ...]
    errors: tuple[str, ...]


def load_slice(path: Path, slice_id: str) -> SliceManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["slices"][slice_id]
    required = frozenset(
        CaseLayerRequirement(row["case"], row["layer"]) for row in entry["required"]
    )
    if not required:
        raise ValueError("required_manifest_empty")
    if len(required) != len(entry["required"]):
        raise ValueError("duplicate_required_pair")
    invariants = {row["name"]: row["nodeid"] for row in entry.get("invariants", [])}
    if len(invariants) != len(entry.get("invariants", [])):
        raise ValueError("duplicate_invariant_name")
    return SliceManifest(slice_id, tuple(entry["paths"]), required, invariants)


def report_from_payload(payload: dict, execution_mode: str) -> CaseRunReport:
    items = tuple(
        CollectedCase(
            nodeid=row["nodeid"],
            case_ids=tuple(row["case_ids"]),
            layer=row["layer"],
            slices=tuple(row.get("slices", [])),
        )
        for row in payload.get("items", [])
    )
    return CaseRunReport(
        inventory=CaseInventory(items=items, errors=tuple(payload.get("errors", []))),
        outcomes=dict(payload.get("outcomes", {})),
        execution_mode=execution_mode,
    )


def evaluate(
    manifest: SliceManifest, payload: dict, *, execution_mode: str
) -> GateDecision:
    """Pure decision over a case-plugin report payload."""

    report = report_from_payload(payload, execution_mode)
    decision = validate_case_report(
        report, manifest.required, required_execution_mode=POSTGRES_MODE
    )
    outcomes = report.outcomes
    missing_invariants = tuple(
        name for name, node in manifest.invariants.items() if node not in outcomes
    )
    failed_invariants = tuple(
        name
        for name, node in manifest.invariants.items()
        if node in outcomes and outcomes[node] != "passed"
    )
    errors = list(decision.errors)
    if not report.inventory.items:
        errors.append("zero_collected")
    if int(payload.get("exit_status", 1)) != 0:
        errors.append(f"pytest_exit_status:{payload.get('exit_status')}")
    return GateDecision(
        passed=decision.passed
        and not missing_invariants
        and not failed_invariants
        and not errors,
        missing_requirements=tuple(
            f"{r.case_id}:{r.layer}" for r in decision.missing_requirements
        ),
        failed_nodeids=decision.failed_nodeids,
        missing_invariants=missing_invariants,
        failed_invariants=failed_invariants,
        errors=tuple(errors),
    )


def _source_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(manifest: SliceManifest, artifact: Path) -> GateDecision:
    from scripts.run_required_economic_taxonomy_postgres import (
        verify_postgresql_identity,
    )

    if os.environ.get("STOCKSCANNER_TEST_ALLOW_POSTGRES") != "1":
        raise SystemExit("STOCKSCANNER_TEST_ALLOW_POSTGRES=1 is required")
    database = verify_postgresql_identity()
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "cases.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--disable-warnings",
                f"--exposure-case-report={report_path}",
                *manifest.paths,
            ],
            cwd=BACKEND_ROOT,
            check=False,
        )
        payload = (
            json.loads(report_path.read_text("utf-8"))
            if report_path.exists()
            else {"items": [], "outcomes": {}, "exit_status": 1}
        )
    decision = evaluate(manifest, payload, execution_mode=database["dialect"])
    required_nodes = {
        f"{r.case_id}:{r.layer}": [
            item["nodeid"]
            for item in payload.get("items", [])
            if r.case_id in item["case_ids"] and item["layer"] == r.layer
        ]
        for r in sorted(manifest.required, key=lambda r: (r.case_id, r.layer))
    }
    artifact.write_text(
        json.dumps(
            {
                "slice": manifest.slice_id,
                "passed": decision.passed,
                "source_revision": _source_revision(),
                "database": database,
                "execution_mode": database["dialect"],
                "required_nodes": required_nodes,
                "invariants": manifest.invariants,
                "outcomes": payload.get("outcomes", {}),
                "missing_requirements": list(decision.missing_requirements),
                "failed_nodeids": list(decision.failed_nodeids),
                "missing_invariants": list(decision.missing_invariants),
                "failed_invariants": list(decision.failed_invariants),
                "errors": list(decision.errors),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slice", default="S1")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(tempfile.gettempdir()) / "company-exposure-gate.json",
    )
    args = parser.parse_args(argv)
    decision = run(load_slice(args.manifest, args.slice), args.artifact)
    print(
        f"company exposure {args.slice} gate: {'PASSED' if decision.passed else 'FAILED'}"
    )
    for label, values in (
        ("missing requirements", decision.missing_requirements),
        ("failed nodes", decision.failed_nodeids),
        ("missing invariants", decision.missing_invariants),
        ("failed invariants", decision.failed_invariants),
        ("errors", decision.errors),
    ):
        for value in values:
            print(f"  {label}: {value}")
    print(f"artifact: {args.artifact}")
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

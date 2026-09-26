from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.domain.company_exposure.manifest import CASE_IDS, LAYERS
from scripts.run_required_company_exposure_postgres import (
    DEFAULT_MANIFEST,
    SliceManifest,
    evaluate,
    load_slice,
)

BACKEND_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def case_gate():
    from app.domain.company_exposure.manifest import CaseLayerRequirement

    return SliceManifest(
        slice_id="T",
        paths=("tests",),
        required=frozenset({CaseLayerRequirement("R07", "postgres")}),
        invariants={"lease": "tests/x.py::test_lease"},
    )


def _payload(outcome: str, *, sibling_passes: bool = False) -> dict:
    items = []
    outcomes = {"tests/x.py::test_lease": "passed"}
    if outcome != "zero_collected":
        if outcome != "missing":
            items.append(
                {
                    "nodeid": "tests/a.py::test_r07",
                    "case_ids": ["R07"],
                    "layer": "postgres",
                }
            )
            outcomes["tests/a.py::test_r07"] = outcome
        if sibling_passes:
            items.append(
                {
                    "nodeid": "tests/b.py::test_r07",
                    "case_ids": ["R07"],
                    "layer": "postgres",
                }
            )
            outcomes["tests/b.py::test_r07"] = "passed"
        items.append(
            {"nodeid": "tests/c.py::test_other", "case_ids": ["E01"], "layer": "unit"}
        )
    return {"items": items, "outcomes": outcomes, "errors": [], "exit_status": 0}


@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(
    "outcome", ["missing", "skipped", "xfailed", "xpassed", "zero_collected"]
)
def test_tag_gate_rejects_incomplete_required_layer(case_gate, outcome):
    decision = evaluate(case_gate, _payload(outcome), execution_mode="postgresql")
    assert decision.passed is False


@pytest.mark.parametrize("outcome", ["skipped", "failed", "xfailed"])
def test_passing_sibling_cannot_mask_a_bad_outcome(case_gate, outcome):
    decision = evaluate(
        case_gate, _payload(outcome, sibling_passes=True), execution_mode="postgresql"
    )
    assert decision.passed is False
    assert decision.failed_nodeids == ("tests/a.py::test_r07",)


def test_complete_postgres_run_passes(case_gate):
    assert evaluate(case_gate, _payload("passed"), execution_mode="postgresql").passed


def test_sqlite_is_never_a_substitute(case_gate):
    decision = evaluate(case_gate, _payload("passed"), execution_mode="sqlite")
    assert decision.passed is False
    assert "wrong_execution_mode:sqlite" in decision.errors


def test_missing_or_failed_invariant_fails(case_gate):
    payload = _payload("passed")
    payload["outcomes"].pop("tests/x.py::test_lease")
    assert evaluate(
        case_gate, payload, execution_mode="postgresql"
    ).missing_invariants == ("lease",)
    payload["outcomes"]["tests/x.py::test_lease"] = "skipped"
    assert evaluate(
        case_gate, payload, execution_mode="postgresql"
    ).failed_invariants == ("lease",)


def test_failed_session_fails_even_when_tags_pass(case_gate):
    payload = _payload("passed")
    payload["exit_status"] = 1
    assert evaluate(case_gate, payload, execution_mode="postgresql").passed is False


def test_s1_manifest_is_well_formed():
    manifest = load_slice(DEFAULT_MANIFEST, "S1")
    assert all(r.case_id in CASE_IDS and r.layer in LAYERS for r in manifest.required)
    assert ("R13", "api") in {(r.case_id, r.layer) for r in manifest.required}
    raw = json.loads(DEFAULT_MANIFEST.read_text("utf-8"))["slices"]["S1"]
    required_cases = {row["case"] for row in raw["required"]}
    assert not required_cases & set(raw["excluded_future_scope"])


def test_current_collection_satisfies_the_s1_manifest(tmp_path):
    """Collection cannot silently shrink the reviewed manifest."""

    manifest = load_slice(DEFAULT_MANIFEST, "S1")
    report = tmp_path / "collect.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--exposure-case-report={report}",
            *manifest.paths,
        ],
        cwd=BACKEND_ROOT,
        check=True,
        capture_output=True,
    )
    payload = json.loads(report.read_text("utf-8"))
    collected = {
        (case, item["layer"]) for item in payload["items"] for case in item["case_ids"]
    }
    missing = {(r.case_id, r.layer) for r in manifest.required} - collected
    assert not missing
    nodes = {item["nodeid"] for item in payload["items"]}
    all_nodes = set()
    listing = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *manifest.paths,
        ],
        cwd=BACKEND_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    all_nodes = {line.strip() for line in listing if "::" in line}
    assert set(manifest.invariants.values()) <= all_nodes | nodes

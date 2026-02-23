"""Machine-check tests for E1 pipeline-scoped processing-state ADR artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADR_PATH = ROOT / "docs" / "theme_identity" / "adr_e1_pipeline_scoped_processing_state_v1.md"
INV_PATH = ROOT / "docs" / "theme_identity" / "adr_e1_pipeline_scoped_processing_state_v1.invariants.json"
EXPECTED_IDS = {
    "E1-INV-001",
    "E1-INV-002",
    "E1-INV-003",
    "E1-INV-004",
    "E1-INV-005",
    "E1-INV-006",
    "E1-INV-007",
}
EXPECTED_TASK_REFS = {
    "StockScreenClaude-bv9.1.2",
    "StockScreenClaude-bv9.1.3",
    "StockScreenClaude-bv9.1.4",
    "StockScreenClaude-bv9.1.5",
    "StockScreenClaude-bv9.1.6",
}


def test_pipeline_processing_adr_artifacts_exist():
    assert ADR_PATH.exists(), f"Missing ADR file: {ADR_PATH}"
    assert INV_PATH.exists(), f"Missing invariant spec: {INV_PATH}"


def test_invariant_spec_shape_and_ids():
    payload = json.loads(INV_PATH.read_text(encoding="utf-8"))
    assert payload["version"] == "v1"
    assert isinstance(payload["invariants"], list) and payload["invariants"]

    seen = set()
    allowed_severities = {"critical", "high", "medium", "low"}
    allowed_check_kinds = {"repo"}
    for invariant in payload["invariants"]:
        iid = invariant["id"]
        assert re.fullmatch(r"E1-INV-\d{3}", iid), f"Bad invariant id: {iid}"
        assert iid not in seen, f"Duplicate invariant id: {iid}"
        seen.add(iid)
        assert invariant["check_kind"] in allowed_check_kinds
        assert invariant["severity"] in allowed_severities
        assert invariant["description"].strip()

        repo_check = invariant["repo_check"]
        assert isinstance(repo_check, dict)
        paths = repo_check.get("paths")
        literals = repo_check.get("required_literals")
        assert isinstance(paths, list) and paths
        assert isinstance(literals, list) and literals

        assert invariant["compliance_rule"].strip()

    assert seen == EXPECTED_IDS


def test_adr_lists_all_invariant_ids():
    text = ADR_PATH.read_text(encoding="utf-8")
    for iid in EXPECTED_IDS:
        assert iid in text, f"ADR is missing invariant id reference: {iid}"


def test_adr_references_all_e1_implementation_tasks():
    text = ADR_PATH.read_text(encoding="utf-8")
    for task_ref in EXPECTED_TASK_REFS:
        assert task_ref in text, f"ADR is missing downstream E1 task reference: {task_ref}"


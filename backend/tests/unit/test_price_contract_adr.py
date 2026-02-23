"""Machine-check tests for LL2-E1 canonical price contract ADR artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADR_PATH = ROOT / "docs" / "learning_loop" / "adr_ll2_e1_canonical_price_contract_v1.md"
INV_PATH = ROOT / "docs" / "learning_loop" / "adr_ll2_e1_canonical_price_contract_v1.invariants.json"
EXPECTED_IDS = {
    "PRICE-INV-001",
    "PRICE-INV-002",
    "PRICE-INV-003",
    "PRICE-INV-004",
    "PRICE-INV-005",
    "PRICE-INV-006",
    "PRICE-INV-007",
    "PRICE-INV-008",
}


def test_price_contract_adr_artifacts_exist():
    assert ADR_PATH.exists(), f"Missing ADR file: {ADR_PATH}"
    assert INV_PATH.exists(), f"Missing invariant spec: {INV_PATH}"


def test_invariant_spec_shape_and_ids():
    payload = json.loads(INV_PATH.read_text(encoding="utf-8"))
    assert payload["version"] == "v1"
    assert isinstance(payload["invariants"], list) and payload["invariants"]

    seen = set()
    allowed_severities = {"critical", "high", "medium", "low"}
    allowed_check_kinds = {"sql", "repo"}
    for invariant in payload["invariants"]:
        iid = invariant["id"]
        assert re.fullmatch(r"PRICE-INV-\d{3}", iid), f"Bad invariant id: {iid}"
        assert iid not in seen, f"Duplicate invariant id: {iid}"
        seen.add(iid)
        assert invariant["check_kind"] in allowed_check_kinds
        assert invariant["severity"] in allowed_severities
        assert invariant["description"].strip()
        if invariant["check_kind"] == "sql":
            sql = invariant["sql_check"].strip()
            assert sql.upper().startswith("SELECT")
            assert "SELECT 0 AS VIOLATIONS;" not in sql.upper()
        else:
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

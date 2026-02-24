"""Machine-check tests for E2 canonical key normalization policy artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ADR_PATH = ROOT / "docs" / "theme_identity" / "adr_e2_canonical_key_normalization_v1.md"
CORPUS_PATH = ROOT / "docs" / "theme_identity" / "adr_e2_canonical_key_normalization_v1.corpus.json"
CASE_ID_RE = re.compile(r"E2-CASE-\d{3}")
EXPECTED_REQUIRED_CASES = {
    "E2-CASE-001",
    "E2-CASE-004",
    "E2-CASE-010",
    "E2-CASE-011",
    "E2-CASE-013",
    "E2-CASE-017",
    "E2-CASE-021",
    "E2-CASE-022",
}
EXPECTED_CATEGORIES = {
    "acronym",
    "numeric",
    "punctuation",
    "stopword",
    "unicode",
    "symbol",
    "plural",
    "protected",
    "near_collision",
    "fallback",
}


def test_e2_policy_artifacts_exist():
    assert ADR_PATH.exists(), f"Missing ADR file: {ADR_PATH}"
    assert CORPUS_PATH.exists(), f"Missing corpus file: {CORPUS_PATH}"


def test_e2_policy_references_issue_and_corpus():
    text = ADR_PATH.read_text(encoding="utf-8")
    assert "StockScreenClaude-bv9.2.1" in text
    assert "adr_e2_canonical_key_normalization_v1.corpus.json" in text
    assert "Normalization pipeline (ordered, deterministic)" in text
    assert "Stopword policy" in text
    assert "Protected-token policy" in text
    assert "Overflow and empties" in text


def test_e2_corpus_shape_and_key_constraints():
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert payload["version"] == "v1"
    assert payload["key_regex"] == "^[a-z0-9]+(?:_[a-z0-9]+)*$"
    assert payload["max_length"] == 96
    assert payload["default_fallback_key"] == "unknown_theme"

    cases = payload["cases"]
    assert isinstance(cases, list) and len(cases) >= 20

    case_ids = set()
    categories = set()
    key_re = re.compile(payload["key_regex"])
    for case in cases:
        case_id = case["id"]
        assert CASE_ID_RE.fullmatch(case_id), f"Bad case id: {case_id}"
        assert case_id not in case_ids, f"Duplicate case id: {case_id}"
        case_ids.add(case_id)

        category = case["category"]
        categories.add(category)
        assert category in EXPECTED_CATEGORIES, f"Unexpected category: {category}"

        raw = case["raw"]
        expected_key = case["expected_key"]
        assert isinstance(raw, str) and raw.strip(), f"Missing raw text for {case_id}"
        assert isinstance(expected_key, str) and expected_key.strip(), f"Missing key for {case_id}"
        assert len(expected_key) <= payload["max_length"], f"Key too long for {case_id}"
        assert key_re.fullmatch(expected_key), f"Key violates regex for {case_id}: {expected_key}"

    assert EXPECTED_REQUIRED_CASES.issubset(case_ids)
    assert EXPECTED_CATEGORIES.issubset(categories)


def test_unicode_normalization_examples_match():
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    mapping = {case["id"]: case["expected_key"] for case in payload["cases"]}
    assert mapping["E2-CASE-010"] == mapping["E2-CASE-011"] == "cafe_robotic"


def test_policy_regression_examples_for_reported_gaps():
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    mapping = {case["id"]: case["expected_key"] for case in payload["cases"]}
    assert mapping["E2-CASE-004"] == "glp1_weight_loss"
    assert mapping["E2-CASE-015"] == "ev_and_battery"
    assert mapping["E2-CASE-022"] == "as_software"

"""Unit tests for the IBD classification health report + gate (pure, no DB)."""
import json

from app.scripts.check_ibd_classification_gate import main as gate_main
from app.services.ibd_classification_health import (
    HEALTH_REPORT_SCHEMA_VERSION,
    HISTOGRAM_BINS,
    build_health_report,
    confidence_histogram,
    diff_classifications,
    evaluate_gate,
    health_asset_name,
    read_health_report,
    write_health_report,
    GateResult,
)


def test_confidence_histogram_bins_and_null():
    rows = [
        {"confidence": 0.91},   # -> [0.9,1.0]
        {"confidence": 1.0},    # -> [0.9,1.0] (clamped)
        {"confidence": 0.8},    # -> [0.8,0.9)
        {"confidence": 0.05},   # -> [0.0,0.1)
        {"confidence": None},   # -> null (LLM rows carry no confidence)
    ]
    hist = confidence_histogram(rows)

    assert hist["[0.9,1.0]"] == 2
    assert hist["[0.8,0.9)"] == 1
    assert hist["[0.0,0.1)"] == 1
    assert hist["null"] == 1
    # Every bin is always present (zero-initialised) plus the null bucket.
    assert set(hist) == set(HISTOGRAM_BINS) | {"null"}
    # Counts sum to the row count.
    assert sum(hist.values()) == len(rows)


def test_diff_classifications_counts_and_churn():
    prev = [
        {"symbol": "A", "industry_group": "G1"},
        {"symbol": "B", "industry_group": "G2"},
        {"symbol": "C", "industry_group": "G3"},  # removed next week
    ]
    new = [
        {"symbol": "A", "industry_group": "G1"},  # unchanged
        {"symbol": "B", "industry_group": "G9"},  # changed group
        {"symbol": "D", "industry_group": "G4"},  # added
    ]

    diff = diff_classifications(prev, new)

    assert diff["compared"] == 2          # A, B present both weeks
    assert diff["changed_group"] == 1     # B
    assert diff["added"] == 1             # D
    assert diff["removed"] == 1           # C
    assert diff["churn_pct"] == 50.0      # 1 changed / 2 compared
    assert {"symbol": "B", "prev": "G2", "new": "G9"} in diff["changed_examples"]


def test_diff_classifications_empty_prev_is_zero_churn():
    diff = diff_classifications([], [{"symbol": "A", "industry_group": "G1"}])
    assert diff["compared"] == 0
    assert diff["added"] == 1
    assert diff["churn_pct"] == 0.0


def _payload(rows, *, as_of, summary):
    return {
        "market": "HK",
        "as_of_date": as_of,
        "generated_at": f"{as_of}T00:00:00Z",
        "source_revision": f"ibd:{as_of}",
        "model_id": "deepseek-chat",
        "summary": summary,
        "classifications": rows,
    }


def test_build_health_report_with_prev():
    new = _payload(
        [{"symbol": "A", "industry_group": "G1", "confidence": 0.9},
         {"symbol": "B", "industry_group": "G9", "confidence": None}],
        as_of="2026-06-02",
        summary={"coverage_pct": 96.0, "by_source": {"embedding": 1, "llm": 1}},
    )
    prev = _payload(
        [{"symbol": "A", "industry_group": "G1", "confidence": 0.9},
         {"symbol": "B", "industry_group": "G2", "confidence": 0.7}],
        as_of="2026-05-26",
        summary={"coverage_pct": 95.0},
    )

    report = build_health_report(
        payload=new, prev_payload=prev, embedding_model="all-MiniLM-L6-v2"
    )

    assert report["schema_version"] == HEALTH_REPORT_SCHEMA_VERSION
    assert report["market"] == "HK"
    assert report["embedding_model"] == "all-MiniLM-L6-v2"
    assert report["summary"]["coverage_pct"] == 96.0
    assert report["confidence_histogram"]["null"] == 1
    assert report["diff"]["changed_group"] == 1          # B changed
    assert report["diff"]["churn_pct"] == 50.0
    assert report["diff"]["prev_as_of_date"] == "2026-05-26"
    assert report["diff"]["prev_source_revision"] == "ibd:2026-05-26"


def test_build_health_report_without_prev_has_null_diff():
    new = _payload(
        [{"symbol": "A", "industry_group": "G1", "confidence": 0.9}],
        as_of="2026-06-02",
        summary={"coverage_pct": 96.0},
    )
    report = build_health_report(payload=new, prev_payload=None, embedding_model="x")
    assert report["diff"] is None


def test_health_asset_name_and_roundtrip(tmp_path):
    assert health_asset_name("SG") == "ibd-classification-health-sg.json"

    report = {"schema_version": 1, "market": "SG", "summary": {"coverage_pct": 90.0}}
    path = tmp_path / health_asset_name("SG")
    write_health_report(path, report)
    assert path.read_text().endswith("\n")
    assert read_health_report(path) == report


_OK = {"summary": {"coverage_pct": 96.0}, "diff": {"churn_pct": 3.0}}
_HIGH_CHURN = {"summary": {"coverage_pct": 96.0}, "diff": {"churn_pct": 40.0}}
_LOW_COVERAGE = {"summary": {"coverage_pct": 40.0}, "diff": None}


def test_gate_passes_within_thresholds():
    res = evaluate_gate(_OK, max_churn_pct=25, min_coverage_pct=50, mode="enforce")
    assert isinstance(res, GateResult)
    assert res.passed
    assert res.breaches == []


def test_gate_enforce_fails_on_high_churn():
    res = evaluate_gate(_HIGH_CHURN, max_churn_pct=25, min_coverage_pct=50, mode="enforce")
    assert not res.passed
    assert any("churn" in b for b in res.breaches)


def test_gate_enforce_fails_on_low_coverage():
    res = evaluate_gate(_LOW_COVERAGE, max_churn_pct=25, min_coverage_pct=50, mode="enforce")
    assert not res.passed
    assert any("coverage" in b for b in res.breaches)


def test_gate_warn_mode_reports_but_passes():
    res = evaluate_gate(_HIGH_CHURN, max_churn_pct=25, min_coverage_pct=50, mode="warn")
    assert res.passed                 # warn never blocks
    assert res.breaches               # but breaches are still surfaced


def test_gate_off_mode_always_passes_with_no_breaches():
    res = evaluate_gate(_HIGH_CHURN, max_churn_pct=25, min_coverage_pct=50, mode="off")
    assert res.passed
    assert res.breaches == []


def test_gate_null_diff_skips_churn_check():
    res = evaluate_gate(
        {"summary": {"coverage_pct": 96.0}, "diff": None},
        max_churn_pct=25, min_coverage_pct=50, mode="enforce",
    )
    assert res.passed


def _write_report(tmp_path, report):
    path = tmp_path / "ibd-classification-health-hk.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def test_gate_cli_enforce_returns_1_on_breach(tmp_path, capsys):
    report = {"market": "HK", "summary": {"coverage_pct": 40.0}, "diff": None}
    rc = gate_main([
        "--health", _write_report(tmp_path, report),
        "--max-churn-pct", "25", "--min-coverage-pct", "50", "--mode", "enforce",
    ])
    assert rc == 1
    assert "::error::" in capsys.readouterr().out


def test_gate_cli_warn_returns_0_but_warns(tmp_path, capsys):
    report = {"market": "HK", "summary": {"coverage_pct": 40.0}, "diff": None}
    rc = gate_main([
        "--health", _write_report(tmp_path, report),
        "--max-churn-pct", "25", "--min-coverage-pct", "50", "--mode", "warn",
    ])
    assert rc == 0
    assert "::warning::" in capsys.readouterr().out


def test_gate_cli_ok_returns_0(tmp_path, capsys):
    report = {"market": "HK", "summary": {"coverage_pct": 96.0}, "diff": {"churn_pct": 2.0}}
    rc = gate_main([
        "--health", _write_report(tmp_path, report),
        "--max-churn-pct", "25", "--min-coverage-pct", "50", "--mode", "enforce",
    ])
    assert rc == 0
    assert "gate: OK" in capsys.readouterr().out


def test_confidence_histogram_exact_tenth_boundaries():
    # Each exact tenth must land in the bin whose lower edge it equals, robust to
    # float representation error from arbitrary upstream confidence sources.
    rows = [{"confidence": c} for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    hist = confidence_histogram(rows)
    for n in range(1, 10):
        assert hist[HISTOGRAM_BINS[n]] == 1, (n, HISTOGRAM_BINS[n], hist)
    assert hist["[0.0,0.1)"] == 0
    assert hist["null"] == 0


def test_gate_none_coverage_is_breach_not_crash():
    # A malformed report with coverage_pct=None must fail safe (coverage breach),
    # not raise TypeError inside the gate.
    res = evaluate_gate(
        {"summary": {"coverage_pct": None}, "diff": None},
        max_churn_pct=25, min_coverage_pct=50, mode="enforce",
    )
    assert not res.passed
    assert any("coverage" in b for b in res.breaches)


def test_gate_enforce_fails_when_configured_llm_has_zero_successes():
    report = {
        "model_id": "deepseek-v4-flash",
        "summary": {
            "coverage_pct": 100.0,
            "llm_attempts": 1,
            "llm_successes": 0,
            "llm_failures": 1,
            "llm_parse_failures": 0,
        },
        "diff": None,
    }

    result = evaluate_gate(
        report, max_churn_pct=25, min_coverage_pct=50, mode="enforce",
    )

    assert not result.passed
    assert any("zero successful LLM classifications" in breach for breach in result.breaches)


def test_gate_does_not_breach_when_configured_llm_was_not_needed():
    report = {
        "model_id": "deepseek-v4-flash",
        "summary": {
            "coverage_pct": 100.0,
            "llm_attempts": 0,
            "llm_successes": 0,
            "llm_failures": 0,
            "llm_parse_failures": 0,
        },
        "diff": None,
    }

    result = evaluate_gate(
        report, max_churn_pct=25, min_coverage_pct=50, mode="enforce",
    )

    assert result.passed

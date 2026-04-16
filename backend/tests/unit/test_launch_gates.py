"""Unit tests for the ASIA v2 launch-gate runner (bead asia.11.1).

Covers:
- Self-check gates (G1, G3, G8, G9) pass against controlled fixture trees —
  NOT the live repo docs/asia tree, which changes as JP/TW canary evidence
  lands and would silently flip assertions without any code regression.
- DB-backed gates (G2, G4) report MISSING_EVIDENCE when no session is passed
- External-evidence gates (G5, G6, G7) report MISSING_EVIDENCE without
  evidence, FAIL on threshold breach, PASS when all thresholds met
- Verdict aggregation: PASS only if every hard gate passes; NO_GO when any
  MISSING_EVIDENCE; FAIL when any FAIL (FAIL dominates NO_GO)
- Artifact writer: content_hash stays inside the JSON, file_hash goes in
  the .sha256 sidecar — same dual-hash pattern as bead 10.4
- Hash determinism: same input + pinned `now` produces the same hash
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.governance.launch_gates import (
    GateContext,
    GateStatus,
    GateVerdict,
    LaunchGateReport,
    _check_g2_universe,
    _check_g4_fundamentals,
    render_json,
    render_markdown,
    run_all_gates,
)
from app.services.governance.gate_artifact import write_artifacts
from app.services.telemetry.schema import completeness_distribution_payload


# G3 and G9 read the live benchmark registry and flag matrix doc — both are
# stable module-level constants, safe to test against the real repo. G1 and
# G8 are isolated via fixture project roots so that new drill/rehearsal
# artifacts for JP/TW don't silently change which file the gate selects.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NOW = datetime(2026, 4, 15, 18, 0, 0, tzinfo=timezone.utc)


def _make_docs_root(tmp_path: Path) -> Path:
    """Return a minimal fixture project root with just docs/asia/ populated."""
    docs = tmp_path / "docs" / "asia"
    docs.mkdir(parents=True)
    return tmp_path


def _write_rehearsal_report(docs: Path, date: str = "2026-04-15") -> Path:
    path = docs / f"asia_v2_e11_st2_migration_rehearsal_report_{date}.md"
    path.write_text(
        "# ASIA v2 E11 ST2 Migration Rehearsal Report\n\n"
        "- Outcome: **PASS**\n\nThis rehearsal demonstrates **no data-loss**.\n",
        encoding="utf-8",
    )
    return path


def _write_runbook(docs: Path) -> Path:
    path = docs / "asia_v2_operator_runbooks.md"
    path.write_text("# ASIA v2 Operator Runbooks\n\n## RB-01\n", encoding="utf-8")
    return path


def _write_drill(docs: Path, date: str = "2026-04-15") -> Path:
    path = docs / f"asia_v2_runbook_drill_{date}.md"
    path.write_text(
        f"# ASIA v2 Runbook Drill Record — {date}\n\nSigned at {date}.\n",
        encoding="utf-8",
    )
    return path


def _write_flag_matrix(docs: Path) -> Path:
    path = docs / "asia_v2_flag_matrix_and_rollback_runbook.md"
    path.write_text(
        "asia_master_enabled\nasia_market_hk_enabled\nasia_market_jp_enabled\n"
        "asia_market_tw_enabled\nasia_universe_apply_destructive_enabled\n"
        "asia_reconciliation_quarantine_enforced\n",
        encoding="utf-8",
    )
    return path


class TestSelfCheckGates:
    def test_g1_schema_passes_with_rehearsal_report(self, tmp_path):
        root = _make_docs_root(tmp_path)
        _write_rehearsal_report(root / "docs" / "asia")
        report = run_all_gates(project_root=root, now=_NOW)
        g1 = next(g for g in report.gates if g.gate_id == "G1")
        assert g1.status == GateStatus.PASS, g1.detail

    def test_g1_missing_without_report(self, tmp_path):
        root = _make_docs_root(tmp_path)  # no rehearsal report written
        report = run_all_gates(project_root=root, now=_NOW)
        g1 = next(g for g in report.gates if g.gate_id == "G1")
        assert g1.status == GateStatus.MISSING_EVIDENCE

    def test_g3_benchmark_passes_live_registry(self):
        # G3 reads app.services.benchmark_registry_service — stable module,
        # safe to test against the live import.
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        g3 = next(g for g in report.gates if g.gate_id == "G3")
        assert g3.status == GateStatus.PASS, g3.detail
        assert g3.metrics["checked_markets"] == ["US", "HK", "JP", "TW"]

    def test_g8_observability_passes_with_recent_drill(self, tmp_path):
        root = _make_docs_root(tmp_path)
        docs = root / "docs" / "asia"
        _write_runbook(docs)
        _write_drill(docs, date="2026-04-15")
        report = run_all_gates(project_root=root, now=_NOW)
        g8 = next(g for g in report.gates if g.gate_id == "G8")
        assert g8.status == GateStatus.PASS, g8.detail
        assert g8.metrics["drill_age_days"] == 0

    def test_g8_stale_drill_fails(self, tmp_path):
        root = _make_docs_root(tmp_path)
        docs = root / "docs" / "asia"
        _write_runbook(docs)
        _write_drill(docs, date="2026-04-15")
        report = run_all_gates(project_root=root, now=_NOW + timedelta(days=30))
        g8 = next(g for g in report.gates if g.gate_id == "G8")
        assert g8.status == GateStatus.FAIL
        assert g8.metrics["drill_age_days"] == 30

    def test_g9_flag_matrix_passes(self, tmp_path):
        root = _make_docs_root(tmp_path)
        _write_flag_matrix(root / "docs" / "asia")
        report = run_all_gates(project_root=root, now=_NOW)
        g9 = next(g for g in report.gates if g.gate_id == "G9")
        assert g9.status == GateStatus.PASS, g9.detail


class TestDbBackedGatesWithoutDb:
    def test_g2_missing_evidence_without_db(self):
        report = run_all_gates(project_root=_PROJECT_ROOT, db=None, now=_NOW)
        g2 = next(g for g in report.gates if g.gate_id == "G2")
        assert g2.status == GateStatus.MISSING_EVIDENCE

    def test_g4_missing_evidence_without_db(self):
        report = run_all_gates(project_root=_PROJECT_ROOT, db=None, now=_NOW)
        g4 = next(g for g in report.gates if g.gate_id == "G4")
        assert g4.status == GateStatus.MISSING_EVIDENCE


class _FakeComparableColumn:
    def __eq__(self, other):
        return self

    def __ge__(self, other):
        return self


class _FakeMarketTelemetryEvent:
    metric_key = _FakeComparableColumn()
    recorded_at = _FakeComparableColumn()


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


class _FakeDb:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return _FakeQuery(self._rows)


class TestDbBackedGateResilience:
    def test_g2_malformed_payload_returns_missing_evidence(self, monkeypatch):
        fake_models = ModuleType("app.models.market_telemetry")
        fake_models.MarketTelemetryEvent = _FakeMarketTelemetryEvent
        monkeypatch.setitem(sys.modules, "app.models.market_telemetry", fake_models)

        ctx = GateContext(project_root=_PROJECT_ROOT, now=_NOW)
        rows = [SimpleNamespace(payload={"prior_size": "bad", "delta": "1"})]

        g2 = _check_g2_universe(ctx, db=_FakeDb(rows))

        assert g2.status == GateStatus.MISSING_EVIDENCE
        assert g2.detail.startswith("DB query failed:")

    def test_g2_scopes_to_enabled_markets(self, monkeypatch):
        fake_models = ModuleType("app.models.market_telemetry")
        fake_models.MarketTelemetryEvent = _FakeMarketTelemetryEvent
        monkeypatch.setitem(sys.modules, "app.models.market_telemetry", fake_models)

        ctx = GateContext(project_root=_PROJECT_ROOT, now=_NOW, enabled_markets=("US", "HK"))
        rows = [
            SimpleNamespace(market="HK", recorded_at=_NOW, payload={"prior_size": 1000, "delta": 8}),
            SimpleNamespace(market="TW", recorded_at=_NOW, payload={"prior_size": 1000, "delta": 120}),
        ]

        g2 = _check_g2_universe(ctx, db=_FakeDb(rows))

        assert g2.status == GateStatus.PASS
        assert g2.metrics["worst_market"] == "HK"

    def test_g4_scopes_to_enabled_markets_and_checks_transparency(self, monkeypatch):
        fake_models = ModuleType("app.models.market_telemetry")
        fake_models.MarketTelemetryEvent = _FakeMarketTelemetryEvent
        monkeypatch.setitem(sys.modules, "app.models.market_telemetry", fake_models)

        ctx = GateContext(
            project_root=_PROJECT_ROOT,
            now=_NOW,
            enabled_markets=("US", "HK"),
            target_market="HK",
        )
        rows = [
            SimpleNamespace(
                market="HK",
                recorded_at=_NOW,
                payload=completeness_distribution_payload(
                    bucket_counts={"0-25": 20, "25-50": 120, "50-75": 220, "75-90": 260, "90-100": 380},
                    symbols_total=1000,
                ),
            ),
            SimpleNamespace(
                market="TW",
                recorded_at=_NOW,
                payload=completeness_distribution_payload(
                    bucket_counts={"0-25": 490, "25-50": 120, "50-75": 120, "75-90": 120, "90-100": 150},
                    symbols_total=1000,
                ),
            ),
        ]

        g4 = _check_g4_fundamentals(ctx, db=_FakeDb(rows))

        assert g4.status == GateStatus.PASS, g4.detail
        assert g4.metrics["worst_market"] == "HK"
        assert g4.metrics["transparency_sample_market"] == "HK"


class TestExternalEvidenceGates:
    def test_g5_missing_evidence_without_path(self):
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        g5 = next(g for g in report.gates if g.gate_id == "G5")
        assert g5.status == GateStatus.MISSING_EVIDENCE

    def test_g5_pass_with_all_thresholds_met(self, tmp_path):
        ev = tmp_path / "qa.json"
        ev.write_text(json.dumps({
            "precision": 0.90,
            "recall": 0.80,
            "false_positive_rate": 0.05,
        }))
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            external_evidence={"G5": str(ev)},
            now=_NOW,
        )
        g5 = next(g for g in report.gates if g.gate_id == "G5")
        assert g5.status == GateStatus.PASS

    def test_g5_fail_on_low_precision(self, tmp_path):
        ev = tmp_path / "qa.json"
        ev.write_text(json.dumps({
            "precision": 0.80,  # below 0.85 threshold
            "recall": 0.80,
            "false_positive_rate": 0.05,
        }))
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            external_evidence={"G5": str(ev)},
            now=_NOW,
        )
        g5 = next(g for g in report.gates if g.gate_id == "G5")
        assert g5.status == GateStatus.FAIL

    def test_g7_pass_with_good_performance(self, tmp_path):
        ev = tmp_path / "perf.json"
        ev.write_text(json.dumps({
            "scan_create_p95_ms": 1200,
            "scan_failure_rate": 0.005,
            "market_isolation_pass": True,
        }))
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            external_evidence={"G7": str(ev)},
            now=_NOW,
        )
        g7 = next(g for g in report.gates if g.gate_id == "G7")
        assert g7.status == GateStatus.PASS

    def test_g7_fail_on_high_p95(self, tmp_path):
        ev = tmp_path / "perf.json"
        ev.write_text(json.dumps({
            "scan_create_p95_ms": 2000,  # >1500ms
            "scan_failure_rate": 0.005,
            "market_isolation_pass": True,
        }))
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            external_evidence={"G7": str(ev)},
            now=_NOW,
        )
        g7 = next(g for g in report.gates if g.gate_id == "G7")
        assert g7.status == GateStatus.FAIL

    def test_g6_pass_with_both_keys_true(self, tmp_path):
        ev = tmp_path / "parity.json"
        ev.write_text(json.dumps({
            "us_parity_pass": True,
            "non_us_correctness_pass": True,
        }))
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            external_evidence={"G6": str(ev)},
            now=_NOW,
        )
        g6 = next(g for g in report.gates if g.gate_id == "G6")
        assert g6.status == GateStatus.PASS


class TestVerdictAggregation:
    def test_no_go_when_evidence_missing(self):
        # No external evidence, no DB — G2/G4/G5/G6/G7 all MISSING_EVIDENCE.
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        assert report.verdict == GateVerdict.NO_GO
        assert report.hard_missing_evidence >= 5

    def test_fail_dominates_no_go(self, tmp_path):
        # G5 FAIL + many MISSING_EVIDENCE — verdict must be FAIL, not NO_GO.
        ev = tmp_path / "qa.json"
        ev.write_text(json.dumps({
            "precision": 0.50, "recall": 0.50, "false_positive_rate": 0.50,
        }))
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            external_evidence={"G5": str(ev)},
            now=_NOW,
        )
        assert report.verdict == GateVerdict.FAIL


class TestArtifactWriter:
    def test_file_hash_matches_file_bytes_not_content_hash(self, tmp_path):
        """Sidecar stores SHA-256 of raw .json bytes (for sha256sum -c),
        NOT the content_hash inside the JSON. Same contract as bead 10.4.
        """
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        paths = write_artifacts(report, tmp_path)

        json_bytes = Path(paths["json"]).read_bytes()
        sidecar = Path(paths["sha256"]).read_text().strip()
        file_hash = sidecar.split("  ")[0]

        assert file_hash == hashlib.sha256(json_bytes).hexdigest()
        assert file_hash != report.content_hash  # distinct artifacts

    def test_filename_includes_verdict(self, tmp_path):
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        paths = write_artifacts(report, tmp_path)
        # NO_GO verdict given we passed no evidence.
        assert "no_go" in Path(paths["json"]).name

    def test_markdown_surfaces_verdict_and_hash(self, tmp_path):
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        md = render_markdown(report)
        assert report.verdict.upper() in md
        # Hash appears at top + bottom.
        assert md.count(report.content_hash) == 2

    def test_markdown_surfaces_optional_provenance(self):
        report = run_all_gates(
            project_root=_PROJECT_ROOT,
            now=_NOW,
            enabled_markets=["US", "HK"],
            target_market="HK",
            execution_mode="synthetic_seeded_harness",
            provenance_note="Synthetic rehearsal only; does not unblock downstream canaries.",
        )
        md = render_markdown(report)
        assert "Enabled markets: ['US', 'HK']" in md
        assert "Target market: HK" in md
        assert "Execution mode: synthetic_seeded_harness" in md
        assert "Synthetic rehearsal only; does not unblock downstream canaries." in md

    def test_target_market_must_be_in_enabled_markets(self):
        with pytest.raises(ValueError):
            run_all_gates(
                project_root=_PROJECT_ROOT,
                now=_NOW,
                enabled_markets=["US", "HK"],
                target_market="JP",
            )


class TestHashDeterminism:
    def test_same_inputs_same_content_hash(self):
        r1 = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        r2 = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        assert r1.content_hash == r2.content_hash
        assert len(r1.content_hash) == 64

    def test_content_hash_over_canonical_json(self):
        report = run_all_gates(project_root=_PROJECT_ROOT, now=_NOW)
        blob = json.loads(render_json(report))
        assert blob["content_hash"] == report.content_hash
        blob["content_hash"] = None
        recomputed = hashlib.sha256(
            json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        assert recomputed == report.content_hash

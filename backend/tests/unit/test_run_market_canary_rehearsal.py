from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.governance.launch_gates import GateVerdict
from scripts import run_market_canary_rehearsal as cli


def test_profile_for_jp_matches_expected_canary_envelope():
    profile = cli._profile_for_market("JP")

    assert profile.benchmark_symbol == "^N225"
    assert profile.symbols_refreshed == 1860
    assert profile.drift_ratios["JP"] == pytest.approx(0.012)
    assert profile.low_bucket_ratios["JP"] == pytest.approx(0.03)


def test_alembic_runs_via_current_python(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    code, output = cli._alembic("postgresql://user:pass@localhost/rehearsal", ["upgrade", "head"])

    assert code == 0
    assert output == "ok"
    assert captured["cmd"][:3] == [cli.sys.executable, "-m", "alembic"]


def test_cli_passes_default_provenance_and_evidence(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}
    evidence_dir = tmp_path / "jp-evidence"
    evidence_dir.mkdir()

    def fake_seed(database_url: str, market: str, now):
        assert database_url == "postgresql://user:pass@localhost/rehearsal"
        assert market == "JP"
        return {
            "universe_drift_rows": 4,
            "completeness_rows": 4,
            "freshness_rows": 1,
            "benchmark_rows": 1,
            "worst_drift_ratio": 0.012,
            "worst_low_bucket_ratio": 0.03,
            "worst_low_bucket_market": "TW",
            "benchmark_symbol": "^N225",
        }

    def fake_run_all_gates(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            verdict=GateVerdict.PASS,
            hard_gate_count=9,
            hard_passed=9,
            hard_failed=0,
            hard_missing_evidence=0,
            execution_mode=kwargs.get("execution_mode"),
            provenance_note=kwargs.get("provenance_note"),
            content_hash="abc123",
        )

    fake_db = SimpleNamespace(close=lambda: None, _canary_engine=SimpleNamespace(dispose=lambda: None))

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/rehearsal")
    monkeypatch.setattr(cli, "_ensure_migrated", lambda database_url: None)
    monkeypatch.setattr(
        cli,
        "_resolve_evidence_bundle",
        lambda raw_dir, root: {"G5": "qa.json", "G6": "parity.json", "G7": "load.json"},
    )
    monkeypatch.setattr(cli, "_seed_rehearsal_telemetry", fake_seed)
    monkeypatch.setattr(cli, "_open_session", lambda database_url: fake_db)
    monkeypatch.setattr(cli, "run_all_gates", fake_run_all_gates)
    monkeypatch.setattr(cli, "resolve_output_dir", lambda override: tmp_path)
    monkeypatch.setattr(
        cli,
        "write_artifacts",
        lambda report, out_dir: {
            "json": str(out_dir / "report.json"),
            "markdown": str(out_dir / "report.md"),
            "sha256": str(out_dir / "report.sha256"),
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_market_canary_rehearsal.py",
            "--market",
            "JP",
            "--evidence-dir",
            str(evidence_dir),
            "--now",
            "2026-04-16T12:00:00+00:00",
        ],
    )

    exit_code = cli.main()

    assert exit_code == 0
    assert captured["external_evidence"] == {"G5": "qa.json", "G6": "parity.json", "G7": "load.json"}
    assert captured["execution_mode"] == "ephemeral_postgres_dress_rehearsal"
    assert "Seeded PostgreSQL rehearsal telemetry for JP" in str(captured["provenance_note"])
    assert "jp-evidence" in str(captured["provenance_note"])

    out = capsys.readouterr().out
    assert "Verdict: PASS" in out
    assert "Execution mode: ephemeral_postgres_dress_rehearsal" in out
